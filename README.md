# strided-rs

`strided-rs` is a Rust workspace for strided tensor views, kernels, and einsum.
It is inspired by Julia's [Strided.jl](https://github.com/Jutho/Strided.jl),
[StridedViews.jl](https://github.com/Jutho/StridedViews.jl), and
[OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl).

## Workspace Layout

- [`strided-view`](strided-view/README.md): core dynamic-rank strided view/array types and metadata ops
- [`strided-kernel`](strided-kernel/README.md): cache-optimized elementwise/reduction kernels over strided views
- [`strided-einsum2`](strided-einsum2/README.md): binary einsum (`einsum2_into`) on strided tensors
- [`strided-opteinsum`](strided-opteinsum/README.md): N-ary einsum frontend with nested notation and contraction-order optimization
- [`mdarray-opteinsum`](mdarray-opteinsum/): einsum wrapper for `mdarray` arrays (row-major ↔ column-major transparent conversion)
- [`ndarray-opteinsum`](ndarray-opteinsum/): einsum wrapper for `ndarray` arrays (direct strides passthrough)

## Features

- **Dynamic-rank strided views** (`StridedView` / `StridedViewMut`) over contiguous memory
- **Owned strided arrays** (`StridedArray`) with row-major and column-major constructors
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: permuting, transposing, broadcasting
- **Cache-optimized iteration** with automatic blocking and loop reordering
- **Optional multi-threading** via Rayon (`parallel` feature) with recursive dimension splitting

## Installation

These crates are currently **not published to crates.io** (`publish = false`).
Use workspace path dependencies:

```toml
[dependencies]
strided-view = { path = "../strided-rs/strided-view" }
strided-kernel = { path = "../strided-rs/strided-kernel" }
strided-einsum2 = { path = "../strided-rs/strided-einsum2" }
strided-opteinsum = { path = "../strided-rs/strided-opteinsum" }
```

## Documentation

Generate API docs locally:

```bash
cargo doc --workspace --no-deps
```

Open docs locally:

```bash
open target/doc/index.html
```

CI also builds rustdoc on PRs and deploys workspace docs to GitHub Pages on `main`.

## Quick Start

```rust
use strided_kernel::{StridedArray, map_into};

// Create a row-major 2D array
let src = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| {
    (idx[0] * 10 + idx[1]) as f64
});
let mut dest = StridedArray::<f64>::row_major(&[2, 3]);

// Element-wise map: dest[i] = src[i] * 2
map_into(&mut dest.view_mut(), &src.view(), |x| x * 2.0).unwrap();
assert_eq!(dest.get(&[1, 2]), 24.0); // (1*10 + 2) * 2
```

See each sub-crate README for detailed API examples and benchmarks:
- [`strided-view`](strided-view/README.md) — types, view operations
- [`strided-kernel`](strided-kernel/README.md) — map/reduce/broadcast kernels, [benchmarks](strided-kernel/README.md#benchmarks)
- [`strided-einsum2`](strided-einsum2/README.md) — binary einsum with GEMM backend
- [`strided-opteinsum`](strided-opteinsum/README.md) — N-ary einsum, [benchmarks](strided-opteinsum/README.md#benchmarks)
- [`mdarray-opteinsum`](mdarray-opteinsum/README.md) — einsum wrapper for `mdarray` arrays
- [`ndarray-opteinsum`](ndarray-opteinsum/README.md) — einsum wrapper for `ndarray` arrays

## Acknowledgments

This crate is inspired by and ports functionality from:
- [Strided.jl](https://github.com/Jutho/Strided.jl) by Jutho
- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) by Jutho
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) for
  `strided-opteinsum` design ideas and reference test-case patterns

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

See `NOTICE` for upstream attribution (Strided.jl / StridedViews.jl are MIT-licensed).
