# tenferro-rs

A general-purpose tensor computation library in Rust with CPU/GPU support.

## Overview

`tenferro-rs` is a Rust workspace providing:

- Dense tensor types with CPU/GPU support
- cuTENSOR/hipTensor-compatible operation protocol (`TensorPrims<A>` trait)
- High-level einsum with N-ary contraction tree optimization
- Automatic differentiation (VJP/JVP)
- C FFI for Julia/Python integration

Built on top of [strided-rs](https://github.com/tensor4all/strided-rs) for cache-optimized strided array operations.

## Design

See [`docs/design/`](docs/design/) for architecture and design documents, including:

- [Architecture](docs/design/architecture.md) — workspace layers, crate dependency graph, device layer
- [Design Documents](docs/design/README.md) — per-crate API designs (tensor, prims, einsum, algebra, autodiff, etc.)

## Documentation

Generate a unified local docs site (design docs + Rust API docs):

```bash
./scripts/build_docs_site.sh
```

Output:

- `target/docs-site/index.html` (top page)
- `target/docs-site/design/` (formal design docs)
- `target/docs-site/api/` (`cargo doc --workspace` output)

## Coverage

Per-file line coverage is checked against thresholds in `coverage-thresholds.json`.
Files listed in `exclude` are skipped from threshold checking.

```bash
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
