#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Tensor = torch::Tensor;

constexpr int64_t kSmallMatmulSizes[] = {2, 4, 8, 16, 32};
constexpr int64_t kLargeMatmulSizes[] = {128, 256, 512};
constexpr int64_t kSmallLinalgSizes[] = {4, 8, 16, 32};
constexpr int64_t kMediumLinalgSizes[] = {64, 128};
constexpr int64_t kBatches[] = {16, 64, 256};
constexpr int64_t kBatchedSmallSizes[] = {2, 4, 8, 16};

struct BenchResult {
  std::string suite;
  std::string name;
  std::string dtype;
  std::string shape;
  int threads = 1;
  int iterations = 0;
  double mean_us = 0.0;
  double total_us = 0.0;
};

std::string shape_text(std::initializer_list<int64_t> dims) {
  std::ostringstream out;
  bool first = true;
  for (auto dim : dims) {
    if (!first) {
      out << "x";
    }
    out << dim;
    first = false;
  }
  return out.str();
}

Tensor f64_tensor(std::vector<int64_t> shape, int64_t seed) {
  int64_t len = 1;
  for (auto dim : shape) {
    len *= dim;
  }
  std::vector<double> data(static_cast<size_t>(len));
  for (int64_t idx = 0; idx < len; ++idx) {
    data[static_cast<size_t>(idx)] =
        static_cast<double>((idx * 17 + seed * 31 + 7) % 997) / 997.0 - 0.5;
  }
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  return torch::from_blob(data.data(), shape, options).clone();
}

Tensor c64_tensor(std::vector<int64_t> shape, int64_t seed) {
  int64_t len = 1;
  for (auto dim : shape) {
    len *= dim;
  }
  std::vector<c10::complex<double>> data(static_cast<size_t>(len));
  for (int64_t idx = 0; idx < len; ++idx) {
    double re = static_cast<double>((idx * 17 + seed * 31 + 7) % 997) / 997.0 - 0.5;
    double im = static_cast<double>((idx * 23 + seed * 19 + 11) % 991) / 991.0 - 0.5;
    data[static_cast<size_t>(idx)] = c10::complex<double>(re, im);
  }
  auto options =
      torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
  return torch::from_blob(data.data(), shape, options).clone();
}

Tensor f64_spd_tensor(int64_t n, int64_t seed) {
  auto a = torch::empty({n, n}, torch::TensorOptions().dtype(torch::kFloat64));
  auto acc = a.accessor<double, 2>();
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      if (row == col) {
        acc[row][col] = static_cast<double>(n) + 2.0 + static_cast<double>(row + seed) * 0.001;
      } else {
        acc[row][col] = static_cast<double>((row + col + seed) % 7) * 0.01;
      }
    }
  }
  return a;
}

Tensor c64_hermitian_tensor(int64_t n, int64_t seed) {
  auto a = torch::empty({n, n}, torch::TensorOptions().dtype(torch::kComplexDouble));
  auto acc = a.accessor<c10::complex<double>, 2>();
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      acc[row][col] = c10::complex<double>(0.0, 0.0);
    }
  }
  for (int64_t col = 0; col < n; ++col) {
    for (int64_t row = 0; row <= col; ++row) {
      c10::complex<double> value;
      if (row == col) {
        value = c10::complex<double>(
            static_cast<double>(n) + 2.0 + static_cast<double>(row + seed) * 0.001,
            0.0);
      } else {
        double re = static_cast<double>((row + col + seed) % 7) * 0.01;
        double im = static_cast<double>((row * 3 + col + seed) % 5) * 0.01;
        value = c10::complex<double>(re, im);
      }
      acc[row][col] = value;
      acc[col][row] = c10::complex<double>(value.real(), -value.imag());
    }
  }
  return a;
}

void consume(const Tensor& tensor) {
  volatile double value = tensor.abs().sum().item<double>();
  (void)value;
}

int iterations_for(const std::string& suite, int64_t scale) {
  if (suite == "ad") {
    return scale <= 16 ? 50 : 20;
  }
  if (scale <= 8) {
    return 1000;
  }
  if (scale <= 32) {
    return 300;
  }
  if (scale <= 128) {
    return 80;
  }
  return 20;
}

BenchResult run_bench(
    const std::string& suite,
    const std::string& name,
    const std::string& dtype,
    const std::string& shape,
    int threads,
    int iterations,
    const std::function<void()>& body) {
  for (int i = 0; i < 5; ++i) {
    body();
  }

  auto start = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    body();
  }
  auto end = Clock::now();

  double total_us =
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) /
      1000.0;
  return BenchResult{
      suite,
      name,
      dtype,
      shape,
      threads,
      iterations,
      total_us / static_cast<double>(iterations),
      total_us};
}

BenchResult run_ad_bench(
    const std::string& name,
    const std::string& shape,
    int threads,
    int iterations,
    const std::function<std::pair<Tensor, Tensor>()>& setup,
    const std::function<void(Tensor&, Tensor&)>& body) {
  for (int i = 0; i < 5; ++i) {
    auto inputs = setup();
    body(inputs.first, inputs.second);
  }

  double total_us = 0.0;
  for (int i = 0; i < iterations; ++i) {
    auto inputs = setup();
    auto start = Clock::now();
    body(inputs.first, inputs.second);
    auto end = Clock::now();
    total_us += static_cast<double>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) /
                1000.0;
  }
  return BenchResult{
      "ad",
      name,
      "f64",
      shape,
      threads,
      iterations,
      total_us / static_cast<double>(iterations),
      total_us};
}

void print_header() {
  std::cout << "suite,name,dtype,threads,shape,iterations,mean_us,total_us\n";
}

void print_result(const BenchResult& result) {
  std::cout << result.suite << "," << result.name << "," << result.dtype << ","
            << result.threads << "," << result.shape << "," << result.iterations << ","
            << std::fixed << std::setprecision(3) << result.mean_us << ","
            << result.total_us << "\n";
}

void bench_matmul(int threads) {
  for (auto n : kSmallMatmulSizes) {
    auto a = f64_tensor({n, n}, 1);
    auto b = f64_tensor({n, n}, 2);
    print_result(run_bench(
        "matmul",
        "f64_square",
        "f64",
        shape_text({n, n}),
        threads,
        iterations_for("matmul", n),
        [&]() { consume(torch::matmul(a, b)); }));

    auto ac = c64_tensor({n, n}, 3);
    auto bc = c64_tensor({n, n}, 4);
    print_result(run_bench(
        "matmul",
        "c64_square",
        "c64",
        shape_text({n, n}),
        threads,
        iterations_for("matmul", n),
        [&]() { consume(torch::matmul(ac, bc)); }));
  }

  for (auto n : kLargeMatmulSizes) {
    auto a = f64_tensor({n, n}, 1);
    auto b = f64_tensor({n, n}, 2);
    print_result(run_bench(
        "matmul",
        "f64_square",
        "f64",
        shape_text({n, n}),
        threads,
        iterations_for("matmul", n),
        [&]() { consume(torch::matmul(a, b)); }));
  }
}

void bench_linalg(int threads) {
  for (auto n : kSmallLinalgSizes) {
    auto a = f64_spd_tensor(n, 1);
    auto b_col = f64_tensor({n, 1}, 2);
    auto b_mat = f64_tensor({n, 4}, 3);

    print_result(run_bench(
        "linalg",
        "f64_svd",
        "f64",
        shape_text({n, n}),
        threads,
        iterations_for("linalg", n),
        [&]() {
          auto out = torch::linalg_svd(a, false);
          consume(std::get<1>(out));
        }));
    print_result(run_bench(
        "linalg",
        "f64_qr",
        "f64",
        shape_text({n, n}),
        threads,
        iterations_for("linalg", n),
        [&]() {
          auto out = torch::linalg_qr(a);
          consume(std::get<1>(out));
        }));
    print_result(run_bench(
        "linalg",
        "f64_eigh",
        "f64",
        shape_text({n, n}),
        threads,
        iterations_for("linalg", n),
        [&]() {
          auto out = torch::linalg_eigh(a);
          consume(std::get<0>(out));
        }));
    print_result(run_bench(
        "linalg",
        "f64_solve_column_rhs1",
        "f64",
        shape_text({n, 1}),
        threads,
        iterations_for("linalg", n),
        [&]() { consume(torch::linalg_solve(a, b_col)); }));
    print_result(run_bench(
        "linalg",
        "f64_solve_matrix_rhs4",
        "f64",
        shape_text({n, 4}),
        threads,
        iterations_for("linalg", n),
        [&]() { consume(torch::linalg_solve(a, b_mat)); }));

    auto c = c64_hermitian_tensor(n, 5);
    print_result(run_bench(
        "linalg",
        "c64_eigh",
        "c64",
        shape_text({n, n}),
        threads,
        iterations_for("linalg", n),
        [&]() {
          auto out = torch::linalg_eigh(c);
          consume(std::get<0>(out));
        }));
  }

  for (auto n : kMediumLinalgSizes) {
    auto a = f64_spd_tensor(n, 1);
    auto b_mat = f64_tensor({n, 4}, 3);
    print_result(run_bench(
        "linalg",
        "f64_svd",
        "f64",
        shape_text({n, n}),
        threads,
        iterations_for("linalg", n),
        [&]() {
          auto out = torch::linalg_svd(a, false);
          consume(std::get<1>(out));
        }));
    print_result(run_bench(
        "linalg",
        "f64_qr",
        "f64",
        shape_text({n, n}),
        threads,
        iterations_for("linalg", n),
        [&]() {
          auto out = torch::linalg_qr(a);
          consume(std::get<1>(out));
        }));
    print_result(run_bench(
        "linalg",
        "f64_eigh",
        "f64",
        shape_text({n, n}),
        threads,
        iterations_for("linalg", n),
        [&]() {
          auto out = torch::linalg_eigh(a);
          consume(std::get<0>(out));
        }));
    print_result(run_bench(
        "linalg",
        "f64_solve_matrix_rhs4",
        "f64",
        shape_text({n, 4}),
        threads,
        iterations_for("linalg", n),
        [&]() { consume(torch::linalg_solve(a, b_mat)); }));
  }
}

void bench_batched_einsum(int threads) {
  for (auto batch : kBatches) {
    for (auto n : kBatchedSmallSizes) {
      auto a = f64_tensor({n, n, batch}, 1);
      auto b = f64_tensor({n, n, batch}, 2);
      print_result(run_bench(
          "batched_einsum_rightmost_batch",
          "f64_ikb_knb_to_inb",
          "f64",
          shape_text({n, n, batch}),
          threads,
          iterations_for("batched_einsum", n),
          [&]() { consume(torch::einsum("ikb,knb->inb", {a, b})); }));
    }
  }
}

void bench_einsum_patterns(int threads) {
  auto a = f64_tensor({64, 64}, 1);
  auto b = f64_tensor({64, 64}, 2);
  auto c = f64_tensor({64, 64}, 3);
  print_result(run_bench(
      "einsum_patterns",
      "f64_binary_ij_jk_to_ik",
      "f64",
      shape_text({64, 64}),
      threads,
      iterations_for("einsum_patterns", 64),
      [&]() { consume(torch::einsum("ij,jk->ik", {a, b})); }));
  print_result(run_bench(
      "einsum_patterns",
      "f64_chain_ij_jk_kl_to_il",
      "f64",
      shape_text({64, 64}),
      threads,
      iterations_for("einsum_patterns", 64),
      [&]() { consume(torch::einsum("ij,jk,kl->il", {a, b, c})); }));

  auto x = f64_tensor({8, 16, 8}, 4);
  auto y = f64_tensor({16, 8, 8}, 5);
  print_result(run_bench(
      "einsum_patterns",
      "f64_multiedge_ijk_jkl_to_il",
      "f64",
      "8x16x8__16x8x8",
      threads,
      iterations_for("einsum_patterns", 16),
      [&]() { consume(torch::einsum("ijk,jkl->il", {x, y})); }));

  auto ac = c64_tensor({32, 32}, 6);
  auto bc = c64_tensor({32, 32}, 7);
  print_result(run_bench(
      "einsum_patterns",
      "c64_binary_ij_jk_to_ik",
      "c64",
      shape_text({32, 32}),
      threads,
      iterations_for("einsum_patterns", 32),
      [&]() { consume(torch::einsum("ij,jk->ik", {ac, bc})); }));
}

void bench_ad(int threads) {
  auto bench_svd_values = [&](int64_t n) {
    auto base_svd = f64_spd_tensor(n, 3);
    print_result(run_ad_bench(
        "f64_grad_sum_svd_values",
        shape_text({n, n}),
        threads,
        iterations_for("ad", n),
        [&]() {
          auto a = base_svd.detach().clone();
          a.set_requires_grad(true);
          return std::make_pair(a, Tensor());
        },
        [](Tensor& a, Tensor&) {
          auto out = torch::linalg_svd(a, false);
          auto loss = std::get<1>(out).sum();
          loss.backward();
          consume(a.grad());
        }));
  };

  for (auto n : kSmallLinalgSizes) {
    bench_svd_values(n);
  }
  for (auto n : kMediumLinalgSizes) {
    bench_svd_values(n);
  }

  for (auto n : {4, 16, 64}) {
    auto base_a = f64_tensor({n, n}, 1);
    auto base_b = f64_tensor({n, n}, 2);
    print_result(run_ad_bench(
        "f64_grad_sum_matmul",
        shape_text({n, n}),
        threads,
        iterations_for("ad", n),
        [&]() {
          auto a = base_a.detach().clone();
          auto b = base_b.detach().clone();
          a.set_requires_grad(true);
          b.set_requires_grad(true);
          return std::make_pair(a, b);
        },
        [](Tensor& a, Tensor& b) {
          auto loss = torch::matmul(a, b).sum();
          loss.backward();
          consume(a.grad());
          consume(b.grad());
        }));

    if (n <= 16) {
      auto base_solve = f64_spd_tensor(n, 4);
      auto base_rhs = f64_tensor({n, 1}, 5);
      print_result(run_ad_bench(
          "f64_grad_sum_solve",
          shape_text({n, 1}),
          threads,
          iterations_for("ad", n),
          [&]() {
            auto a = base_solve.detach().clone();
            auto b = base_rhs.detach().clone();
            a.set_requires_grad(true);
            b.set_requires_grad(true);
            return std::make_pair(a, b);
          },
          [](Tensor& a, Tensor& b) {
            auto loss = torch::linalg_solve(a, b).sum();
            loss.backward();
            consume(a.grad());
            consume(b.grad());
          }));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  int threads = 1;
  if (argc >= 2) {
    threads = std::max(1, std::atoi(argv[1]));
  }

  at::set_num_threads(threads);
  at::set_num_interop_threads(1);
  torch::manual_seed(0);

  print_header();
  bench_matmul(threads);
  bench_linalg(threads);
  bench_batched_einsum(threads);
  bench_einsum_patterns(threads);
  bench_ad(threads);
  return 0;
}
