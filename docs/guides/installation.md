# Installation

Add `tenferro` to your `Cargo.toml`:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro" }
```

The default feature set enables the `cpu-faer` backend. This is the easiest
starting point for CPU-only use. Once tenferro is published, you can switch the
path dependency to a crates.io version.

## CPU Backends

Exactly one CPU backend must be enabled:

| Feature | Use when |
| --- | --- |
| `cpu-faer` | You want the default pure-Rust CPU backend. |
| `cpu-blas` | You want BLAS/LAPACK-backed CPU linear algebra. |
| `src-openblas` | You want `cpu-blas` with OpenBLAS built from source. |

To choose BLAS instead of the default backend, disable default features:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro", default-features = false, features = ["cpu-blas"] }
```

For source-built OpenBLAS:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro", default-features = false, features = ["src-openblas"] }
```

Do not enable both `cpu-faer` and `cpu-blas` in the same build.

## CUDA Feature

CUDA support is enabled with the `cuda` feature alias:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro", features = ["cuda"] }
```

CUDA support is partial and experimental. It currently targets NVIDIA CUDA.
See [Devices and GPU](devices-and-gpu.md) for the transfer model and current
coverage.

Common CUDA runtime environment variables:

```bash
CUBECL_DEBUG_LOG=0
CUDA_PATH=/usr/local/cuda-12.0
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
```

If CUDA libraries are installed outside the default search paths, set
`TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH`, or `TENFERRO_CUBLAS_PATH`
to the concrete library path.
