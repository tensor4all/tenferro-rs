# tenferro-linalg Benchmarks

This directory contains the crate-local benchmark suite for `tenferro-linalg`.

## What is covered

Forward kernels:

- `svd`
- `qr`
- `solve`
- `matrix_exp`

AD kernels:

- `svd_rrule`
- `solve_rrule`

Shape families included in the benchmark set:

- small square (`16x16`)
- medium square (`64x64`)
- tall matrices (for example `128x32`)
- wide matrices (for example `32x128`)
- batched small matrices (for example `8x8x32`)

## Run

Run all `tenferro-linalg` benchmarks:

```bash
cargo bench -p tenferro-linalg --bench linalg_benchmarks
```

Compile-check only:

```bash
cargo bench -p tenferro-linalg --bench linalg_benchmarks --no-run
```

Save baseline output:

```bash
cargo bench -p tenferro-linalg --bench linalg_benchmarks | tee /tmp/linalg-bench-baseline.txt
```

## Notes

- Benchmarks use `CpuContext::new(1)` to reduce thread-count noise.
- Inputs are deterministic and generated in-memory (no external dataset).
