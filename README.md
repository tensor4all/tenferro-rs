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

- [Unified Tensor Backend Design](docs/design/tenferro_unified_tensor_backend.md) — high-level architecture, crate structure, roadmap
- [tenferro Design](docs/design/tenferro_design.md) — detailed per-crate API designs
- [libtorch Reference](docs/design/libtorch_reference.md) — PyTorch feature survey for design reference

## Documentation

Generate a unified local docs site (design docs + Rust API docs):

```bash
./scripts/build_docs_site.sh
```

Output:

- `target/docs-site/index.html` (top page)
- `target/docs-site/design/` (formal design docs)
- `target/docs-site/api/` (`cargo doc --workspace` output)

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
