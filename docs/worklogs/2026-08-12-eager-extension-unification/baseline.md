# Issue #1665 — baseline benchmark (pre-change)

Benchmark: `crates/tenferro-linalg/benches/eager_extension_dispatch.rs`
Command: `cargo bench -p tenferro-linalg --bench eager_extension_dispatch --features autodiff -- --save-baseline pre-1665`

Machine: Linux x86-64, 1 thread (`CpuBackend::with_threads(1)`), default faer backend.

| op (2x2, f64) | no_ad | eager_ad_forward |
|---|---:|---:|
| matmul | 26.3 µs | 28.9 µs |
| solve | 92.4 µs | 61.3 µs |
| svd | 43.8 µs | 27.0 µs |
| eigh | 26.3 µs | 21.6 µs |

Target: single-digit µs (PyTorch single-op dispatch is ~1–5 µs). The
`solve`/`svd`/`eigh` rows are extension ops routed through
`apply_eager_with_extension_session`, which pays a per-call module install
(`install_extension_module` → `runtime.reconfigure` → full candidate clone +
install mutex). #1664 measured the same regression at 4 threads (solve 169 µs).
