# CPU Benchmarks

The CPU benchmark suite is a normal reporting benchmark, not a publication
gate. Use it to collect comparable timings before deciding which cases should
become release or publication thresholds.

Recent result snapshots:

- [2026-05-23 CPU benchmark results](cpu-benchmark-results-2026-05-23.md)

## Tenferro Benchmarks

Run the Rust Criterion benchmarks through the wrapper script:

```bash
bash scripts/bench-cpu.sh --kind tenferro
```

The script runs thread counts `1,2,4` by default. Override them with:

```bash
bash scripts/bench-cpu.sh --kind tenferro --threads 1,4
```

The wrapper pins common CPU thread controls for each run:

- `TENFERRO_BENCH_THREADS`
- `RAYON_NUM_THREADS`
- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`

Criterion setup creates input tensors outside the measured closure. AD cases
use `iter_batched` so tensor data preparation happens in setup while the
measured closure covers forward graph construction and backward execution.

## LibTorch C++ Baseline

The canonical Torch baseline is the LibTorch C++ benchmark:

```bash
bash scripts/bench-cpu.sh --kind torch-cpp
```

This avoids Python dispatch overhead in small-matrix latency cases. The script
uses the official CPU-only LibTorch ZIP distribution and builds only the small
benchmark binary locally. It does not build PyTorch from source.

By default, the script stores downloaded LibTorch files under the main
worktree, not under linked git worktrees:

```text
third_party/libtorch/
```

This avoids one LibTorch download per temporary worktree. Set
`TENFERRO_BENCH_DEPS_DIR` to choose a different cache directory, or set
`LIBTORCH_DIR` to point at an existing LibTorch installation.

The default LibTorch URL can be overridden:

```bash
LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip \
  bash scripts/bench-cpu.sh --kind torch-cpp
```

## Scope

The initial CPU suite covers:

- matmul and matmul-like einsum for small and medium/large sizes
- `svd`, `qr`, `eigh`, and `solve`
- batched small einsum with the batch index on the right
- representative N-ary einsum patterns
- AD for `sum(matmul)`, `sum(svd(A).S)`, and `sum(solve(A, b))`
- `f64` as the primary dtype, with selected `c64` cases

The `sum(svd(A).S)` AD benchmark uses the same square sizes as the primal SVD
benchmark: `4x4`, `8x8`, `16x16`, `32x32`, `64x64`, and `128x128`.

GPU benchmarks and hard threshold comparisons are intentionally deferred.
