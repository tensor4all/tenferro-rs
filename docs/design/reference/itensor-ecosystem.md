# ITensor Ecosystem Analysis and tenferro-rs Layering

Date: 2026-02-16

## Purpose

Analyze how the ITensor Julia ecosystem maps to tenferro-rs, assess whether
tenferro-rs can serve as a backend for porting ITensor-like libraries to Rust,
and identify architectural gaps.

## ITensor Ecosystem: NDTensors.jl Refactoring

NDTensors.jl has been **deprecated** and split into modular packages:

```
ITensors.jl (v0.9)              User-facing API (named indices, MPS/MPO)
    |
ITensorBase.jl (v0.5)          Lightweight ITensor wrapper
    |
NamedDimsArrays.jl (v0.14)     Arrays with named dimensions
    |
TensorAlgebra.jl (v0.7)        Tensor algebra: contract, SVD, QR, eigen
    |                           (operates on plain AbstractArray)
MatrixAlgebraKit                Matrix decompositions (LAPACK wrapper)

BlockSparseArrays.jl (v0.10)   Block sparse arrays
SparseArraysBase.jl (v0.9)     Sparse array interface
DiagonalArrays.jl (v0.3)       Diagonal arrays
GradedArrays.jl (v0.6)         Quantum number graded arrays

Mooncake.jl                     AD (extension-based integration)
```

Source: cloned to `projects/tensor4all/extern/`.

### TensorAlgebra.jl Key Design

TensorAlgebra.jl is the core tensor algebra layer. It provides:

- **contract**: Binary tensor contraction
- **SVD, QR, LQ, polar, eigen**: Tensor decompositions
- **Matrix functions**: exp, log, sqrt, etc.

All operations follow the **matricize -> matrix op -> unmatricize** pattern:
reshape N-D tensor to 2D matrix, apply LAPACK, reshape back.

#### Dimension specification: three equivalent interfaces

All dispatch to the same internal implementation via numeric permutations:

1. **Label-based** (arbitrary Symbol names, NOT dimension indices):
   ```julia
   contract(a1, (:a, :b), a2, (:b, :c))   # :b in both -> contracted
   svd(A, (:a,:b,:c,:d), (:b,:a), (:d,:c)) # codomain/domain by label
   ```

2. **Numeric permutation** (dimension numbers, 1-indexed):
   ```julia
   svd(A, (2, 1), (4, 3))   # dims 2,1 -> codomain, dims 4,3 -> domain
   ```

3. **Val shorthand**:
   ```julia
   svd(A, Val(2))   # first 2 dims are codomain
   ```

Labels are NOT stored on tensors. They are external arguments at call sites.
Named dimensions live one layer up (NamedDimsArrays.jl / ITensorBase.jl).

## Layer Mapping: Julia -> Rust

```
Julia (ITensor ecosystem)          Rust (tenferro + hypothetical ports)
-------------------------------    -------------------------------------
ITensors.jl                        itensor-rs (hypothetical)
ITensorBase.jl                     itensor-base-rs (hypothetical)
NamedDimsArrays.jl                 named-dims-rs (hypothetical)
    ^ Named Index layer (above tenferro)
---------------------------------------------------------------------------
TensorAlgebra.jl (contract)        tenferro-einsum              [exists]
TensorAlgebra.jl (SVD/QR)          tenferro-linalg              [planned]
MatrixAlgebraKit                   faer / cuSOLVER              [external]
    ^ Dense tensor operation layer
---------------------------------------------------------------------------
BlockSparseArrays.jl               block-sparse crate           [separate]
DiagonalArrays.jl                  diagonal crate               [separate]
SparseArraysBase.jl                sparse-base crate            [separate]
GradedArrays.jl                    graded crate                 [separate]
    ^ Storage diversity layer (parallel to tenferro, not inside it)
---------------------------------------------------------------------------
Mooncake (AD)                      chainrules / chainrules-core [exists]
```

## What Works

### AD framework (chainrules)

`chainrules-core` is generic over `V: Differentiable`. Any tensor type
(Dense, BlockSparse, Diagonal, Graded) can implement `Differentiable`
and use the same `Tape<V>` / `TrackedValue<V>` AD engine. Dense and
Block sparse share the AD engine without modification.

### Device abstraction (tenferro-device)

Type-independent. Works for any tensor storage.

### Algebra traits (tenferro-algebra)

`HasAlgebra` / `Semiring` are additive — they don't conflict with
external libraries. Unique to tenferro, but not an obstacle.

### Prims layer (tenferro-prims)

The semiring/scalar/analytic execution families in `tenferro-prims` operate on
`StridedView<T>`. Individual dense blocks within a block sparse tensor can
delegate to `batched_gemm` and related family operations.

### Named Index layer

Would sit above tenferro — no changes needed in tenferro itself.
A separate crate wrapping `Tensor<T>` with named dimensions.

## What Needs Separate Implementation

### Block sparse einsum

Block sparse contraction requires different algorithms from dense:
iterate over non-zero block pairs, contract each pair (delegating
individual block GEMMs to `tenferro-prims`), assemble output blocks.

Architecture:

```
block-sparse-einsum (separate crate)
    |
    +-- uses: chainrules-core (Differentiable, ReverseRule for AD)
    +-- uses: tidu (Tape, TrackedValue for AD engine)
    +-- uses: tenferro-prims (batched_gemm for individual dense blocks)
    +-- uses: tenferro-device (device abstraction)
    |
    +-- owns: block iteration logic, non-zero block selection,
              output block structure determination
```

### Diagonal einsum

Diagonal tensor contractions can be element-wise multiplication in many
cases — a fundamentally different algorithm from matricize -> GEMM.

### Graded (QN-conserving) einsum

Requires quantum number fusion rules to determine which blocks exist.
Builds on block sparse but adds symmetry sector logic.

## Design Implications for tenferro-rs

### Current state: sufficient for dense backend

tenferro-einsum is currently hardcoded to `Tensor<T>`:

```rust
pub fn einsum<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>
```

This is **correct for the dense layer**. Block sparse does not reuse
this function — it needs its own einsum with different algorithms.

### No tensor trait needed (for now)

Unlike Julia's `AbstractArray` which unifies all array types under one
interface, Rust can use separate crate implementations for each storage
type. Each storage type implements `Differentiable` for AD, but einsum
dispatches by concrete type, not a trait.

A unifying tensor trait may be useful in the future for generic
algorithms (e.g., DMRG that works with both dense and block sparse),
but that belongs in the ITensor-level layer, not in tenferro.

### tenferro-linalg role

`tenferro-linalg` provides tensor-level SVD/QR/eigen for dense tensors:
users specify left/right dimensions by numeric indices
(e.g., `svd(tensor, &[0, 1], &[2, 3])`), the crate reshapes to a
matrix, calls the external decomposition (faer/cuSOLVER), and reshapes
back. AD rules operate at the tensor level.

This mirrors TensorAlgebra.jl's matricize -> decompose -> unmatricize
pattern, but with numeric indices only (no label-based interface).

## Conclusion

**The layering is realistic.** tenferro-rs can serve as the dense
backend for an ITensor-like Rust ecosystem:

1. **Dense layer**: tenferro-einsum + tenferro-linalg (exists/planned)
2. **Block sparse / Diagonal / Graded**: separate crate groups that
   reuse chainrules (AD), tenferro-prims (block-level GEMM), and
   tenferro-device, but implement their own einsum/linalg algorithms
3. **Named Index layer**: sits above tenferro, wrapping `Tensor<T>`
   with named dimensions — no tenferro changes needed
4. **AD**: already generic (`V: Differentiable`) — shared across all
   tensor types

The main architectural boundary is clear: **tenferro owns the dense
tensor computation stack; block sparse and beyond are parallel crates
that share the AD engine and device infrastructure but implement their
own contraction algorithms.**
