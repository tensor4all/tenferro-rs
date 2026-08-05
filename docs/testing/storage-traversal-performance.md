# Storage traversal performance

```json
{
  "schema": "tenferro.storage-traversal-performance.v1",
  "candidate_commit": "652b5c45f753f04425d71541b387acedc39cfa04",
  "benchmark_path": "crates/tenferro-tensor/benches/element_access.rs",
  "baseline_obligation": "p1-element-access-baseline",
  "baseline_report": "docs/testing/storage-element-access-baseline.json",
  "baseline_measured_commit": "da7b36e699f9f4731dec08de6a4e1ca93f20cd6f",
  "command": "cargo bench --locked -p tenferro-tensor --bench element_access -- --warm-up-time 2 --measurement-time 5 --sample-size 100 --noplot",
  "environment": {
    "rustc_version": "rustc 1.97.1 (8bab26f4f 2026-07-14)",
    "cargo_version": "cargo 1.97.1 (c980f4866 2026-06-30)",
    "target": "x86_64-unknown-linux-gnu",
    "architecture": "x86_64",
    "os": "Linux 6.8.0-101-generic",
    "cpu_model": "AMD EPYC 7713P 64-Core Processor",
    "cpu_affinity": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60,
      61,
      62,
      63
    ],
    "thread_environment": {
      "RAYON_NUM_THREADS": "1",
      "OMP_NUM_THREADS": "1",
      "OPENBLAS_NUM_THREADS": "1",
      "MKL_NUM_THREADS": "1",
      "VECLIB_MAXIMUM_THREADS": "1"
    }
  },
  "sample_size": 100,
  "warm_up_seconds": 2.0,
  "measurement_seconds": 5.0,
  "medians_ns": {
    "contiguous_read": {
      "id": "linear_iteration/col_major/as_slice_iter",
      "estimate_ns": 56919.23942533886,
      "lower_bound_ns": 56432.545757908214,
      "upper_bound_ns": 57433.44832946766,
      "standard_error_ns": 255.43273591283173
    },
    "contiguous_write": {
      "id": "linear_iteration/col_major/tensor_iter_mut",
      "estimate_ns": 57755.59341827952,
      "lower_bound_ns": 56938.046228089865,
      "upper_bound_ns": 58598.40882780319,
      "standard_error_ns": 423.18996731119444
    },
    "dynamic_contiguous": {
      "id": "linear_iteration/col_major/dynamic_tensor_iter",
      "estimate_ns": 57378.25017464604,
      "lower_bound_ns": 56889.591543011404,
      "upper_bound_ns": 57883.08150665991,
      "standard_error_ns": 253.5593226841131
    },
    "fixed_rank": {
      "id": "rank_fixed/2d/col_major/get2/4096",
      "estimate_ns": 32438.07478352985,
      "lower_bound_ns": 32129.188886256834,
      "upper_bound_ns": 32755.945214234085,
      "standard_error_ns": 159.9440673008191
    },
    "strided": {
      "id": "strided_traversal/rectangular_transpose/logical_order_get/3840",
      "estimate_ns": 15323.395404269564,
      "lower_bound_ns": 15183.663358505179,
      "upper_bound_ns": 15476.343688684972,
      "standard_error_ns": 74.82083764096679
    },
    "empty": {
      "id": "linear_iteration/col_major/empty",
      "estimate_ns": 0.573697356626121,
      "lower_bound_ns": 0.5687443915408527,
      "upper_bound_ns": 0.5788174393166272,
      "standard_error_ns": 0.002573242713002129
    }
  },
  "comparisons": {
    "contiguous_read": {
      "baseline_ns": 54986.53687097039,
      "current_ns": 56919.23942533886,
      "ratio": 1.0351486502760427,
      "limit": 1.1
    },
    "dynamic_contiguous": {
      "baseline_ns": 54426.40404333462,
      "current_ns": 57378.25017464604,
      "ratio": 1.0542355531877714,
      "limit": 1.15
    },
    "fixed_rank": {
      "baseline_ns": 28555.716496630084,
      "current_ns": 32438.07478352985,
      "ratio": 1.135957306039158,
      "limit": 1.15
    },
    "strided": {
      "baseline_ns": 14849.827750481152,
      "current_ns": 15323.395404269564,
      "ratio": 1.0318904476028732,
      "limit": 1.15
    }
  },
  "setup_measurements": {
    "prepare_map_bind_dispatch": "not part of element_access benchmark; covered by storage preparation contracts"
  },
  "result": "pass",
  "reason": "comparable traversal cases satisfy the documented limits; new write/empty cases are recorded without a historical baseline"
}
```
