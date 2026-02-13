# tenferro-rs

Unified tensor backend library for the [tensor4all](https://github.com/tensor4all) ecosystem.

## Overview

`tenferro-rs` is a Rust workspace providing:

- Dense tensor types with CPU/GPU support
- cuTENSOR/hipTensor-compatible operation protocol (`TensorOps` trait)
- High-level einsum with N-ary contraction tree optimization
- Automatic differentiation (VJP/JVP)
- C FFI for Julia/Python integration

Built on top of [strided-rs](https://github.com/tensor4all/strided-rs) for cache-optimized strided array operations.

## Design

See [tensor4all/tensor4all-meta](https://github.com/tensor4all/tensor4all-meta) for architecture and design documents.

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
