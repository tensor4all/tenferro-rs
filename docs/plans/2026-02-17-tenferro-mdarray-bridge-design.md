# tenferro-mdarray Bridge Design

## Goal

Provide bidirectional conversion between [mdarray](https://crates.io/crates/mdarray)
`Array<T, DynRank>` and tenferro `Tensor<T>`, enabling users to work with
mdarray as a familiar multi-dimensional array interface while using tenferro's
einsum and linalg operations for computation.

## Architecture

A thin extension crate `tenferro-mdarray` at `extension/tenferro-mdarray/`
implementing `From`/`Into` traits between the two tensor types.

```
extension/tenferro-mdarray/
├── Cargo.toml
└── src/
    └── lib.rs    — From/Into impls, crate-level docs
```

### Dependencies

- `tenferro-tensor` — `Tensor<T>`, `DataBuffer<T>`, `MemoryOrder`
- `tenferro-algebra` — `Scalar` trait
- `tenferro-device` — `LogicalMemorySpace`
- `mdarray` — `Array<T, DynRank>`

## Public API

### Standalone conversion functions

Due to Rust's orphan rule, `From`/`Into` trait impls cannot be provided
between two external types (`Array` and `Tensor`). Instead, standalone
functions are used (same pattern as `tenferro-burn::convert`).

```rust
/// mdarray → tenferro: copies data, preserves shape.
pub fn mdarray_to_tensor<T: Scalar>(array: Array<T, DynRank>) -> Tensor<T>

/// tenferro → mdarray: copies data (contiguifies to row-major if needed).
pub fn tensor_to_mdarray<T: Scalar>(tensor: Tensor<T>) -> Array<T, DynRank>
```

### Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Shape type | `DynRank` only | Matches tenferro's dynamic-rank `Vec<usize>` shape |
| Memory order | Row-major (C order) | Matches mdarray's native storage order |
| Copy vs zero-copy | Copy-based | Simple, correct. Zero-copy is a non-goal — the purpose of this bridge is convenient data exchange, not high-performance in-place operation. |
| Crate placement | `extension/` | Optional bridge, not part of core |

### Conversion internals

**mdarray → tenferro:**
1. `array.shape()` → extract dims as `Vec<usize>`
2. `array.into_vec()` → extract owned `Vec<T>`
3. Compute row-major strides from dims
4. `Tensor::from_vec(vec, &dims, &strides, 0)` → construct tensor

**tenferro → mdarray:**
1. `tensor.into_contiguous(MemoryOrder::RowMajor)` → ensure row-major layout
2. Extract `Vec<T>` from `DataBuffer`
3. Extract shape
4. `Array::from_raw_parts(ptr, shape, capacity)` → construct mdarray

### Non-goal: zero-copy

Zero-copy conversion is explicitly a non-goal. The purpose of this bridge is
convenient data exchange between mdarray and tenferro, not in-place
high-performance operation. Tensor network computations (einsum, SVD) are
O(n^3) or higher, so copy overhead during conversion is negligible in practice.

## Scope

POC skeleton phase: all function bodies use `todo!()`. Only type signatures
and documentation are defined.
