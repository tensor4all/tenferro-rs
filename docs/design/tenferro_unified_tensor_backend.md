# tenferro: Unified Tensor Backend Design Plan

> **POC implementation**: The proof-of-concept (12 POC crates) is at
> <https://github.com/tensor4all/tenferro-rs/>.

> **Detailed design documents** (in this repo):
> - [tenferro Design](./tenferro_design.md) — detailed per-crate API designs (Phase 1 + Future Considerations)
> - [Einsum Internal Design](./tenferro_einsum_internal_design.md) — tenferro-prims and tenferro-einsum internals
> - [Einsum Algorithm Comparison](./einsum_algorithm_comparison.md) — strided-rs vs omeinsum-rs optimization comparison
> - [chainrules-core Design](./chainrules_core_design.md) — AD trait design (Differentiable, ReverseRule, ForwardRule)
> - [Async/Ownership Integration Design](../plans/2026-02-14-tensor-async-ownership-integration-design.md) — CompletionEvent + TensorView decisions

## Context

Four independent Rust projects exist in tensor4all:
- **strided-rs**: Cache-optimized strided array kernels (view, map/reduce, einsum)
- **omeinsum-rs**: Einsum with tropical algebra, gradient support, GPU dispatch
- **ndtensors-rs**: Tensor types with storage hierarchy, linear algebra, autograd
- **tensor4all-rs**: Tensor network algorithms (TCI, Quantics, MPS) with ad-hoc tensor backend

These have significant overlap (3 einsum implementations, 3 scalar trait definitions, 3 dense storage types) yet critical gaps. The goal is to unify into a coherent, reusable tensor backend library **tenferro-\*** that:

1. Integrates selected strided-rs add-on crates (`strided-einsum2`, `strided-opteinsum`) and omeinsum-rs components into `tenferro-*` (while keeping `strided-traits/view/kernel` as external foundational dependencies)
2. Provides unified CPU/GPU dispatch via a **cuTENSOR/hipTensor-compatible protocol** (`tenferro-prims`)
3. Supports both NVIDIA and AMD GPUs via **runtime library loading** (no compile-time vendor lock-in)
4. Supports complex numbers natively
5. Supports custom scalar types (tropical semiring, etc.) with pluggable backends
6. Exposes VJP/JVP through C API for Julia ChainRules.jl (POC exists: `tenferro-capi` + `tenferro-tropical-capi`)
7. Can optionally bridge to Burn for NN workloads

**Key design principles**:
- **strided-rs as CPU backend foundation**: The general-purpose strided array crates (`strided-traits`, `strided-view`, `strided-kernel`) remain in an independent `strided-rs` workspace. They have no BLAS dependency and can be used standalone. Only `tenferro-prims` depends on strided-rs directly — higher-level crates (`tenferro-tensor`, `tenferro-einsum`, `tenferro-linalg`) use tenferro-owned traits (`Scalar`, `Conjugate` in `tenferro-algebra`) instead of strided-traits.
- **cuTENSOR/hipTensor-compatible protocol**: `tenferro-prims` defines a unified `TensorPrims<A>` trait parameterized by algebra `A`, with a cuTENSOR-compatible describe → plan → execute pattern for all operations. CPU, NVIDIA, and AMD backends implement the same trait.
- **Algebra-parameterized dispatch**: `TensorPrims<A>` is parameterized by algebra (e.g., `Standard`, `MaxPlus`). The `HasAlgebra` trait on scalar types enables automatic algebra inference: `Tensor<f64>` → `Standard`, `Tensor<MaxPlus<f64>>` → `MaxPlus`. Users can extend the system by defining new algebras in their own crates (orphan rule compatible).
- **Runtime GPU discovery**: GPU vendor libraries (cuTENSOR, hipTensor) are loaded at runtime via `dlopen`. The caller (Julia, Python) provides the `.so` path. No Cargo feature flags for GPU vendor selection.
- **Plan-based execution**: All operations follow the cuTENSOR pattern of `PrimDescriptor` → plan → execute. Plans cache expensive analysis (GPU kernel selection, CPU fusability checks) for reuse. Extended operations (contract, elementwise_mul) are dynamically queried via `has_extension_for::<T>()`.
- **Enums for all variant types**: All types with variants (`LogicalMemorySpace`, `ComputeDevice`, `OpDescriptor`, etc.) use Rust enums. This preserves exhaustive `match` checking — when a new variant is added, the compiler identifies every call site that needs updating. New variants constitute a semver breaking change (major version bump), managed through Cargo's version resolution. This is preferred over opaque structs because the cost of hidden unhandled variants (runtime bugs) outweighs the cost of explicit version migration (compile-time errors).
- **No implicit cross-space transfer**: Operations across different logical memory spaces fail by default. Data movement is explicit via `to_memory_space_async`.

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: Application                                         │
│   tensor4all-rs (TCI, Quantics, MPS algorithms)              │
│   Julia / Python (C API via tenferro-capi)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 5: C-API                                       [POC]     │
│   tenferro-capi: Opaque TfeTensorF64 handle, tfe_status_t    │
│     Tensor lifecycle (8), einsum (3), SVD (3), DLPack (2)    │
│     rlib crate-type for type sharing with extension capis    │
│   tenferro-tropical-capi: Tropical einsum via C-FFI          │
│     9 functions (3 algebras × einsum/rrule/frule)            │
│     Reuses TfeTensorF64 (#[repr(transparent)] MaxPlus<f64>)  │
│   Stateless rrule/frule only (no tape exposure), f64 only    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 4: Einsum Engine (tenferro-einsum)          [POC]      │
│   High-level API on Tensor<T>: einsum(), string notation     │
│   Subscripts (integer labels + string parse)                 │
│   ContractionTree (optimize, from_pairs)                     │
│   Three API levels: einsum, einsum_with_subscripts,          │
│                     einsum_with_plan                          │
│   Einsum AD rules: tracked_einsum, dual_einsum,              │
│     einsum_rrule, einsum_frule, einsum_hvp                   │
│                                                              │
│          Linear Algebra (tenferro-linalg)         [POC]      │
│   SVD, QR, LU, eigen with left/right dim indices             │
│   Matricize → decompose → unmatricize pattern                │
│   Full AD rules: tracked_*, dual_*, *_rrule, *_frule         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 3: Tensor Type (tenferro-tensor)             [POC]     │
│   Tensor<T> = DataBuffer + dims + strides + offset + device  │
│   Zero-copy view ops: permute, broadcast, diagonal, reshape  │
│   DataBuffer<T>: opaque struct (Owned Vec<T> or External     │
│     via DLPack with release callback)                        │
│   MemoryOrder (ColumnMajor, RowMajor) for allocation only    │
│   No strided-rs dependency (uses tenferro-algebra Scalar)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 2: Tensor Operation Protocol (tenferro-prims) [POC]│
│   TensorPrims<A> trait — parameterized by algebra A            │
│   cuTENSOR pattern: PrimDescriptor → plan → execute            │
│   Core ops: batched_gemm, reduce, trace, permute,            │
│     anti_trace, anti_diag                                    │
│   Extended ops (dynamic query): contract, elementwise_mul    │
│   HasAlgebra trait: T → A automatic inference                │
│   CpuBackend: impl TensorPrims<Standard>                      │
│   Uses StridedView<T> / StridedViewMut<T> directly           │
│                                                              │
│          Tropical Algebra (tenferro-tropical)        [POC]   │
│   MaxPlus<T>, MinPlus<T>, MaxMul<T> scalar wrappers          │
│   MaxPlusAlgebra, MinPlusAlgebra, MaxMulAlgebra markers      │
│   impl TensorPrims<MaxPlusAlgebra> for CpuBackend            │
│   ArgmaxTracker for tropical backward pass                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Shared Infrastructure                                        │
│                                                              │
│   Device Layer (tenferro-device)                    [POC]    │
│   LogicalMemorySpace (MainMemory, PinnedMemory, GpuMemory,   │
│     ManagedMemory) + ComputeDevice enums                     │
│   preferred_compute_devices(space, op_kind)                   │
│   Error types (thiserror): ShapeMismatch, RankMismatch,      │
│     DeviceError, InvalidArgument, StrideError                │
│   Result<T> type alias                                       │
│                                                              │
│   AD Core (chainrules-core)                         [POC]    │
│   Differentiable trait (tangent space definition)             │
│   ReverseRule<V>, ForwardRule<V> (per-operation AD rules)     │
│   AutodiffError, NodeId, SavePolicy                          │
│                                                              │
│   AD Engine (chainrules)                            [POC]    │
│   Tape<V>, TrackedTensor<V>, DualTensor<V>                    │
│   pullback(), hvp() (forward-over-reverse HVP)               │
│   Gradients<V>, PullbackPlan<V>, HvpResult<V>                │
│                                                              │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 1: Backend Implementations                    [future] │
│                                                              │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ CPU              │  │ NVIDIA       │  │ AMD          │    │
│  │                  │  │              │  │              │    │
│  │ Contraction:     │  │ cuTENSOR     │  │ hipTensor    │    │
│  │  strided-einsum2 │  │ (dlopen)     │  │ (dlopen)     │    │
│  │  fusability      │  │              │  │              │    │
│  │  trace reduction │  │ Common vtable│  │              │    │
│  │  EW bypass       │  │ (1:1 API)    │  │              │    │
│  │                  │  │              │  │              │    │
│  │ GEMM (feature):  │  │              │  │              │    │
│  │  faer (default)  │  │              │  │              │    │
│  │  cblas (opt-in)  │  │              │  │              │    │
│  │                  │  │              │  │              │    │
│  │ Elementwise:     │  │              │  │              │    │
│  │  strided-kernel  │  │              │  │              │    │
│  │  cache-opt       │  │              │  │              │    │
│  └─────────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Crate Structure

### POC Crates (implemented in tenferro-rs)

```
strided-rs/ (independent workspace) ── Foundation crates stay as-is ───────
│  General-purpose strided array library. No BLAS dependency.
│  Can be used standalone by projects other than tenferro.
│
├── strided-traits       # Scalar, ElementOp traits
├── strided-view         # StridedArray, StridedView, StridedViewMut (zero-copy strided views)
└── strided-kernel       # Cache-optimized map/reduce/broadcast kernels

tenferro-rs/ (workspace) ── 11 POC crates ────────────────────
│  Depends on strided-rs.
│
│  ── core (root level) ── tenferro-* essential stack ──────────
│
├── tenferro-device      # LogicalMemorySpace + ComputeDevice enums
│                        #   LogicalMemorySpace: MainMemory, PinnedMemory,
│                        #     GpuMemory { device_id }, ManagedMemory
│                        #   preferred_compute_devices(space, op_kind)
│                        #   Error (thiserror): ShapeMismatch, RankMismatch,
│                        #     DeviceError, InvalidArgument, StrideError
│                        #   Result<T> type alias
│                        #   Depends on: thiserror
│
├── tenferro-algebra     # HasAlgebra trait, Semiring trait, Standard type
│                        #   Scalar trait (blanket impl, replaces Scalar)
│                        #   Conjugate trait (complex conjugation)
│                        #   HasAlgebra: maps T → A (f64 → Standard, etc.)
│                        #   Minimal algebra foundation for TensorPrims<A>
│                        #   Depends on: num-complex, num-traits
│
├── tenferro-prims       # TensorPrims<A> trait — parameterized by algebra A
│                        #   PrimDescriptor enum (describe → plan → execute)
│                        #   Core ops: batched_gemm, reduce, trace, permute,
│                        #     anti_trace, anti_diag
│                        #   Extended ops (dynamic query): contract, elementwise_mul
│                        #   Associated type Plan<T> (no type erasure)
│                        #   CpuBackend: impl TensorPrims<Standard>
│                        #   Operates on StridedView<T> / StridedViewMut<T> directly
│                        #   Depends on: tenferro-device, tenferro-algebra,
│                        #     strided-view, strided-traits
│
├── tenferro-tensor      # Tensor<T> = DataBuffer + dims + strides + offset + device
│                        #   DataBuffer<T>: opaque struct (Owned Vec<T> or External
│                        #     with release callback for DLPack zero-copy import)
│                        #   MemoryOrder: ColumnMajor, RowMajor (allocation-time only)
│                        #   Constructors: zeros, ones, from_slice, from_vec
│                        #   View ops: permute, broadcast, diagonal, reshape
│                        #     (zero-copy metadata ops)
│                        #   Data ops: contiguous, into_contiguous, is_contiguous, conj
│                        #   impl Differentiable for Tensor<T>
│                        #   No strided-rs dependency (uses Scalar from tenferro-algebra)
│                        #   Depends on: tenferro-device, tenferro-algebra, chainrules-core
│
├── tenferro-einsum      # High-level einsum on Tensor<T>
│                        #   String notation: einsum("ij,jk->ik", &[&a, &b])
│                        #   Parenthesized order: einsum("ij,(jk,kl)->il", &[...])
│                        #   Subscripts struct (new for integers, parse for strings)
│                        #   ContractionTree (optimize, from_pairs)
│                        #   Allocating: einsum, einsum_with_subscripts, einsum_with_plan
│                        #   Accumulating: einsum_into, einsum_with_subscripts_into,
│                        #     einsum_with_plan_into (alpha/beta scaling)
│                        #   Einsum AD: tracked_einsum, dual_einsum, einsum_rrule,
│                        #     einsum_frule, einsum_hvp
│                        #   Depends on: tenferro-device, tenferro-algebra,
│                        #     tenferro-prims, tenferro-tensor, chainrules
│
├── tenferro-linalg      # Tensor-level linear algebra decompositions
│                        #   SVD, QR, LU, eigen with left/right dim indices
│                        #   Matricize → decompose → unmatricize pattern
│                        #   SvdOptions (max_rank, cutoff) for truncated SVD
│                        #   Tensor-level AD: handles dim permutation + 2D matrix AD
│                        #   Full AD: tracked_*, dual_*, *_rrule, *_frule
│                        #   GPU: cuSOLVER/rocSOLVER via tenferro-device
│                        #   Depends on: tenferro-device, tenferro-algebra,
│                        #     tenferro-tensor, chainrules
│
├── tenferro-capi        # C-API (FFI) for Julia/Python
│                        #   Opaque TfeTensorF64 handle, tfe_status_t error codes
│                        #   tfe_ prefix, _f64 suffix naming convention
│                        #   Tensor lifecycle: from_data, zeros, clone, release,
│                        #     ndim, shape, len, data (8 functions)
│                        #   DLPack v1.0 interop: tfe_tensor_f64_to_dlpack,
│                        #     tfe_tensor_f64_from_dlpack (2 functions, zero-copy)
│                        #   Einsum: tfe_einsum_f64, tfe_einsum_rrule_f64,
│                        #     tfe_einsum_frule_f64 (3 functions)
│                        #   SVD: tfe_svd_f64, tfe_svd_rrule_f64,
│                        #     tfe_svd_frule_f64 (3 functions)
│                        #   Stateless rrule/frule only (no tape exposure)
│                        #   f64 only, DLPack for zero-copy interop
│                        #   crate-type: cdylib + staticlib + rlib (rlib for
│                        #     type sharing with tenferro-tropical-capi)
│                        #   Depends on: tenferro-device, tenferro-tensor,
│                        #     tenferro-einsum, tenferro-linalg
│
│  ── extension/ ── optional extensions (depend on core + extern) ──
│
├── extension/
│   ├── tenferro-tropical    # Tropical semiring tensor operations
│   │                        #   MaxPlus<T> (⊕=max, ⊗=+), MinPlus<T> (⊕=min, ⊗=+),
│   │                        #     MaxMul<T> (⊕=max, ⊗=×) scalar wrappers
│   │                        #   #[repr(transparent)] newtypes satisfying Scalar
│   │                        #   MaxPlusAlgebra, MinPlusAlgebra, MaxMulAlgebra markers
│   │                        #   HasAlgebra impls: MaxPlus<f32/f64> → MaxPlusAlgebra, etc.
│   │                        #   Semiring impls (f64 only for POC)
│   │                        #   impl TensorPrims<MaxPlusAlgebra> for CpuBackend
│   │                        #     (and MinPlus, MaxMul) — orphan rule compatible
│   │                        #   TropicalPlan<T> (analogous to CpuPlan<T>)
│   │                        #   ArgmaxTracker for tropical backward pass (AD)
│   │                        #   Depends on: tenferro-device, tenferro-algebra,
│   │                        #     tenferro-prims, strided-view, strided-traits, num-traits
│   │
│   └── tenferro-tropical-capi # C-API (FFI) for tropical einsum
│                            #   Extends tenferro-capi with tropical einsum functions
│                            #   Reuses TfeTensorF64 handles (MaxPlus<f64> is
│                            #     #[repr(transparent)], same layout as f64)
│                            #   9 functions (3 algebras × 3 functions each):
│                            #     tfe_tropical_einsum_{maxplus,minplus,maxmul}_f64
│                            #     tfe_tropical_einsum_rrule_{...}_f64 (VJP)
│                            #     tfe_tropical_einsum_frule_{...}_f64 (JVP)
│                            #   Algebra selected by function name, not handle type
│                            #   Separate .so from tenferro-capi; C consumers load both
│                            #   Depends on: tenferro-device, tenferro-capi,
│                            #     tenferro-tropical
│
│  ── extern/ ── general-purpose crates (no tenferro dependency) ──
│
└── extern/
    ├── chainrules-core      # Core AD traits (like Julia's ChainRulesCore.jl)
    │                        #   Differentiable trait (tangent space definition)
    │                        #   ReverseRule<V> (pullback), ForwardRule<V> (pushforward)
    │                        #   AutodiffError, AdResult, NodeId, SavePolicy
    │                        #   impl Differentiable for f64, f32
    │                        #   Depends on: thiserror
    │                        #   See: docs/design/chainrules_core_design.md
    │
    ├── chainrules           # AD engine (like Zygote.jl in Julia)
    │                        #   Tape<V>, TrackedTensor<V>, DualTensor<V>
    │                        #   pullback() (reverse-mode), hvp() (forward-over-reverse)
    │                        #   Gradients<V>, PullbackPlan<V>, HvpResult<V>
    │                        #   Re-exports all of chainrules-core
    │                        #   Depends on: chainrules-core
    │
    └── (no more extern linalg crates — matrix-level AD rules
          are now in tenferro-linalg following PyTorch convention)
```

### Future Crates (not in POC)

```
tenferro-rs/ (future additions) ──────────────────────────────
│
├── tenferro-hdf5        # Tensor<T> HDF5 I/O (via hdf5-rt, dlopen)
├── tenferro-mdarray     # Tensor<T> ←→ mdarray conversion (memory copy)
├── tenferro-ndarray     # Tensor<T> ←→ ndarray conversion (memory copy)
└── burn-tenferro        # Burn Backend bridge [OPTIONAL, for NN only]
│
│  Note: AD is now chainrules-core + chainrules (POC exists).
│  tenferro-linalg (POC exists), tenferro-capi (POC exists).
│  tenferro-tropical (POC exists) — proves algebra extensibility.

tenferro-structured-rs/ (future separate workspace) ── Structured tensor types ──
│
├── tenferro-blocksparse # BlockSparseTensor (single DataBuffer + block offsets)
├── tenferro-diag        # DiagTensor (1D Tensor of diagonal elements)
└── tenferro-graded      # GradedTensor (future: quantum number sectors)

tensor4all-rs/ (workspace) ── Tensor network algorithms ────
│
├── TCI, Quantics, MPS, ...
└── depends on tenferro-rs + tenferro-structured-rs
```

### Dependency Graph (POC)

```
strided-rs (independent workspace):
strided-traits → strided-view → strided-kernel

tenferro-rs workspace:

┌─ extern/ (general-purpose, no tenferro dependency) ──────────┐
│                                                              │
│  chainrules-core              (← thiserror)                  │
│      │                                                       │
│      ↓                                                       │
│  chainrules                                                  │
│   (← chainrules-core)                                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    │
    │  depends on extern/ only ↓
    │
┌─ core (root level) ── tenferro-* essential stack ────────────┐
│                                                              │
│  tenferro-device              tenferro-algebra               │
│    (← thiserror)               (← num-complex, ← num-traits)│
│      │                            │                          │
│      ├────────────┐       ┌───────┤                          │
│      │            ↓       ↓       │                          │
│      │       tenferro-prims       │                          │
│      │         (← strided-view,   │                          │
│      │          ← strided-traits) │                          │
│      │            │               │                          │
│      ↓            │               │                          │
│  tenferro-tensor  │               │                          │
│    (← tenferro-algebra,           │                          │
│     ← chainrules-core)            │                          │
│      │            │               │                          │
│      └──────┬─────┘       ┌───────┘                          │
│             ↓             ↓                                  │
│      tenferro-einsum                                         │
│        (← tenferro-algebra, ← chainrules)                    │
│      tenferro-linalg                                         │
│        (← tenferro-algebra, ← chainrules-core,               │
│         ← tenferro-tensor, ← tenferro-device)                │
│             │                                                │
│             ↓                                                │
│      tenferro-capi                                           │
│        (← tenferro-tensor, ← tenferro-einsum,                │
│         ← tenferro-linalg, ← tenferro-device)                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    │
    │  depends on core + extern/ ↓
    │
┌─ extension/ ── optional extensions ──────────────────────────┐
│                                                              │
│  tenferro-tropical                                           │
│    (← tenferro-device, ← tenferro-algebra,                   │
│     ← tenferro-prims, ← strided-view,                       │
│     ← strided-traits, ← num-traits)                          │
│                                                              │
│  tenferro-tropical-capi                                      │
│    (← tenferro-device, ← tenferro-capi [rlib],               │
│     ← tenferro-tropical)                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Future Dependency Graph (full vision)

```
┌─ extern/ (general-purpose) ──────────────────────────────────┐
│  chainrules-core → chainrules                                │
└──────────────────────────────────────────────────────────────┘
    │
┌─ core ───────────────────────────────────────────────────────┐
│  tenferro-device, tenferro-algebra                           │
│      │                │                                      │
│      ├────────────────┤                                      │
│      ↓                ↓                                      │
│  tenferro-prims  tenferro-tensor (← chainrules-core)         │
│      │                │                                      │
│      └──────┬─────────┘                                      │
│             ↓                                                │
│  tenferro-einsum (← chainrules)                              │
│  tenferro-linalg (← chainrules-core)                         │
│             │                                                │
│             ↓                                                │
│  tenferro-capi                                               │
│    (← tenferro-einsum, ← tenferro-linalg,                    │
│     ← tenferro-tensor, ← tenferro-device)                    │
│                                                              │
│  tenferro-hdf5 ← tenferro-tensor, hdf5-rt (dlopen) [future] │
└──────────────────────────────────────────────────────────────┘
    │
┌─ extension/ ─────────────────────────────────────────────────┐
│  tenferro-tropical  (POC exists)                             │
│    ← tenferro-device, tenferro-algebra, tenferro-prims       │
│    impl TensorPrims<MaxPlusAlgebra> for CpuBackend (orphan OK)│
│    impl TensorPrims<MinPlusAlgebra> for CpuBackend (orphan OK)│
│    impl TensorPrims<MaxMulAlgebra> for CpuBackend (orphan OK)│
│                                                              │
│  tenferro-tropical-capi  (POC exists)                        │
│    ← tenferro-device, tenferro-capi [rlib], tenferro-tropical│
│    Separate .so; reuses TfeTensorF64 handles                 │
│    9 FFI functions: 3 algebras × (einsum + rrule + frule)    │
└──────────────────────────────────────────────────────────────┘

[separate workspace: tenferro-structured-rs]
tenferro-blocksparse ← tenferro-tensor
tenferro-diag        ← tenferro-tensor
tenferro-graded      ← tenferro-blocksparse (future)

[optional]
burn-tenferro ← tenferro-tensor, burn-backend
```

### Origin of Each Crate

| tenferro crate | Origin | What changes |
|----------------|--------|--------------|
| tenferro-device | **New** (POC) | LogicalMemorySpace/ComputeDevice enums, preferred device selection, Error/Result types |
| tenferro-prims | **New** (POC), will absorb strided-einsum2 | TensorPrims\<A\> trait (algebra-parameterized), PrimDescriptor, CpuBackend |
| tenferro-tensor | **New** (POC) | Tensor\<T\>, DataBuffer\<T\>, MemoryOrder, zero-copy view ops |
| tenferro-einsum | **New** (POC), will absorb strided-opteinsum + omeinsum-rs | Subscripts, ContractionTree, einsum/einsum_with_subscripts/einsum_with_plan |
| tenferro-algebra | omeinsum-rs (Algebra traits) | Standalone crate for Semiring/tropical types [future] |
| chainrules-core | **New** (POC) | Core AD traits: Differentiable, ReverseRule, ForwardRule (like Julia ChainRulesCore.jl) |
| chainrules | **New** (POC) | AD engine: Tape, TrackedTensor, DualTensor, pullback, hvp (like Julia Zygote.jl) |
| tenferro-linalg | ndtensors-rs (linalg + linalg AD) | **POC** API skeleton: batched matrix SVD/QR/LU/eigen following torch.linalg convention adapted for col-major. `(m, n, *)` shape (first 2 dims = matrix), col-major contiguous, matrix-level AD rules (Mathieu 2019). GPU: cuSOLVER/rocSOLVER |
| tenferro-capi | ndtensors-rs (capi) + tensor4all-rs (capi) | **POC** API skeleton: einsum + SVD, f64 only, stateless rrule/frule (14 functions) |
| tenferro-tropical | omeinsum-rs (algebra) | **POC** API skeleton: MaxPlus, MinPlus, MaxMul scalars + algebra markers + TensorPrims impls + ArgmaxTracker |
| tenferro-tropical-capi | **New** (POC) | C-API for tropical einsum: 9 FFI functions (3 algebras × einsum/rrule/frule), reuses TfeTensorF64 from tenferro-capi |
| tenferro-hdf5 | New, uses hdf5-rt | Tensor\<T\> HDF5 I/O via runtime library loading [future] |
| burn-tenferro | New | Burn Backend bridge [future] |
| **tenferro-structured-rs (separate workspace):** | | |
| tenferro-blocksparse | ndtensors-rs (blocksparse) | Port with single-buffer layout [future] |
| tenferro-diag | ndtensors-rs (diag) | Port DiagTensor [future] |
| tenferro-graded | New (future) | Quantum number graded tensors [future] |

---

## Compile-Time vs Runtime Decision Summary

| Choice | Mechanism | Rationale |
|--------|-----------|-----------|
| GPU vendor (cuTENSOR/hipTensor) | **Runtime** (dlopen) [future] | Single binary for all platforms; Julia/Python inject .so path |
| CPU GEMM (faer/cblas) | **Compile-time** (Cargo feature) [future] | Fundamentally different linking (pure Rust vs C ABI) |
| Elementwise ops | **Enum-based** in TensorPrims; closures via strided-kernel | cuTENSOR-compatible operator enums for GPU; custom closures via strided-kernel directly (CPU only) |
| libloading dependency | **Always ON** (in tenferro-device) [future] | Lightweight, no overhead when GPU absent, no feature gate needed |
| .so path for GPU libs | **Caller-injected** [future] | Rust does not search; Julia/Python provide exact path |

---

## Phase 1: Dense Array Foundation (POC)

> **Detailed API designs**: See [tenferro Design](./tenferro_design.md) in tenferro-rs for full per-crate API designs including code examples.

The POC implements eleven crates:

- **tenferro-device** — `LogicalMemorySpace` (MainMemory, PinnedMemory, GpuMemory, ManagedMemory) + `ComputeDevice` enums, `OpKind`, `preferred_compute_devices()`, shared `Error`/`Result` types. DLPack-aligned device model.
- **tenferro-algebra** — `HasAlgebra` trait (maps scalar T → algebra A), `Semiring` trait, `Standard` type for standard arithmetic. `Scalar` trait (blanket impl, replaces strided-traits' `Scalar`). `Conjugate` trait for complex conjugation.
- **chainrules-core** — Core AD traits (like Julia's ChainRulesCore.jl): `Differentiable` (tangent space), `ReverseRule<V>` (pullback), `ForwardRule<V>` (pushforward), `AutodiffError`, `NodeId`, `SavePolicy`.
- **chainrules** — AD engine (like Julia's Zygote.jl): `Tape<V>`, `TrackedTensor<V>`, `DualTensor<V>`, `pullback()`, `hvp()` (forward-over-reverse HVP), `Gradients<V>`, `PullbackPlan<V>`. Re-exports all of `chainrules-core`.
- **tenferro-prims** — `TensorPrims<A>` trait with cuTENSOR-compatible plan-based execution. Core ops (batched_gemm, reduce, trace, permute, anti_trace, anti_diag) + dynamically-queried extended ops (contract, elementwise_mul). `CpuBackend` implements `TensorPrims<Standard>`.
- **tenferro-tensor** — `Tensor<T>` with `DataBuffer<T>` (opaque struct: Owned `Vec<T>` or External with DLPack release callback), shape/strides, zero-copy view ops (permute, broadcast, diagonal, reshape), `CompletionEvent` for async execution, `TensorView<'a, T>` for borrowed views, consuming variants (`into_contiguous`, `into_conj`). Implements `Differentiable` for `Tensor<T>`. No strided-rs dependency.
- **tenferro-einsum** — High-level einsum on `Tensor<T>` with string notation, parenthesized contraction order, `Subscripts`, `ContractionTree`. Nine API functions: allocating, accumulating (`_into` with alpha/beta), and consuming (`_owned` for buffer reuse). Einsum AD rules: `tracked_einsum`, `dual_einsum`, `einsum_rrule`, `einsum_frule`, `einsum_hvp`.
- **tenferro-linalg** — Batched matrix SVD, QR, LU, eigendecomposition adapted from `torch.linalg` for col-major: `(m, n, *)` shape (first 2 dims are the matrix, rest are batch), column-major contiguous input. Stateless AD rules (`svd_rrule`/`svd_frule`, etc.) using batched matrix operations (Mathieu 2019) that broadcast over batch dims. No `tracked_*`/`dual_*` — chainrules tape engine composes via chain rule. GPU: cuSOLVER/rocSOLVER.
- **tenferro-capi** — C-API (FFI) for Julia/Python: opaque `TfeTensorF64` handle, `tfe_status_t` error codes. 16 functions: tensor lifecycle (8) + DLPack interop (2: `tfe_tensor_f64_to_dlpack`, `tfe_tensor_f64_from_dlpack`) + einsum (3) + SVD (3). DLPack v1.0 zero-copy tensor exchange (CPU/CUDA/ROCm/managed memory). Stateless `rrule`/`frule` only (no tape exposure). f64 only in POC phase. Produces rlib in addition to cdylib/staticlib, enabling type sharing with extension capi crates.
- **tenferro-tropical-capi** — C-API (FFI) for tropical einsum: extends `tenferro-capi` with tropical-specific functions. 9 functions: 3 algebras (MaxPlus, MinPlus, MaxMul) × 3 functions (einsum, rrule, frule). Reuses `TfeTensorF64` handles since `MaxPlus<f64>` is `#[repr(transparent)]` (same memory layout as f64). Algebra is selected by function name (`tfe_tropical_einsum_maxplus_f64`, etc.), not by handle type. Produces a separate `.so` from `tenferro-capi`; C consumers load both.
- **tenferro-tropical** — Tropical semiring tensor operations: `MaxPlus<T>` (⊕=max, ⊗=+), `MinPlus<T>` (⊕=min, ⊗=+), `MaxMul<T>` (⊕=max, ⊗=×) scalar wrappers with `#[repr(transparent)]`. Algebra markers (`MaxPlusAlgebra`, `MinPlusAlgebra`, `MaxMulAlgebra`) with `HasAlgebra` and `Semiring` impls (f64 only for POC). `TensorPrims` impls for `CpuBackend` (all three algebras, orphan rule compatible). `TropicalPlan<T>` for plan-based execution. `ArgmaxTracker` for tropical backward pass (AD).

---

## tenferro-tropical (POC exists)

> **POC API skeleton exists** in the tenferro-rs workspace with three
> tropical semiring scalar wrappers, algebra markers, `TensorPrims` impls,
> and `ArgmaxTracker`. All function bodies use `todo!()`.

Tropical algebra types and `TensorPrims` implementations that prove
the extensibility of the algebra-parameterized design:

```rust
// tenferro-tropical crate (in tenferro-rs workspace)
pub struct MaxPlus<T>(pub T);    // ⊕ = max, ⊗ = +
pub struct MinPlus<T>(pub T);    // ⊕ = min, ⊗ = +
pub struct MaxMul<T>(pub T);     // ⊕ = max, ⊗ = ×

// Algebra markers (zero-sized)
pub struct MaxPlusAlgebra;
pub struct MinPlusAlgebra;
pub struct MaxMulAlgebra;

impl HasAlgebra for MaxPlus<f64> { type Algebra = MaxPlusAlgebra; }

impl TensorPrims<MaxPlusAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;
    // has_extension_for returns false (no extended ops for tropical)
    ...
}
```

Also provides:
- `ArgmaxTracker` for tropical backward pass (AD winner-index tracking)
- `Semiring` impls for each algebra (f64 only for POC)
- Future: SIMD-optimized tropical-gemm via `TypeId`-based runtime dispatch (from omeinsum-rs)

Being a workspace crate with locally-defined algebra markers proves that
`TensorPrims<MyAlgebra> for CpuBackend` is orphan rule compatible.

---

## tenferro-linalg (POC exists)

> **POC API skeleton exists** with batched matrix SVD, QR, LU, eigen
> + stateless AD rules, following the `torch.linalg` convention.

Batched matrix decompositions adapted from PyTorch's `torch.linalg` for col-major:
input shape `(m, n, *)` where the first 2 dims are the matrix and all following
dims are independent batch dimensions. Column-major contiguous input required
(LAPACK/cuSOLVER native). This design is **context-agnostic** — the module
does not know about tensor networks, MPS, or any specific application.

If you need to decompose a tensor along arbitrary legs:
`permute` → `reshape` → `contiguous(ColumnMajor)` → call linalg function.

**Primary functions**: `svd`, `qr`, `lu`, `eigen`.
**Result types**: `SvdResult`, `QrResult`, `LuResult`, `EigenResult`.
**SVD truncation**: `SvdOptions` (`max_rank`, `cutoff`).

**AD rules** (stateless, matrix-level, batched via broadcasting):
- Reverse-mode: `svd_rrule`, `qr_rrule`, `lu_rrule`, `eigen_rrule`
- Forward-mode: `svd_frule`, `qr_frule`, `lu_frule`, `eigen_frule`
- No `tracked_*`/`dual_*` — the chainrules tape engine composes
  `permute_backward` + `reshape_backward` + `svd_rrule` via chain rule.

**GPU path**: cuSOLVER/rocSOLVER via runtime-loaded vendor library (same dlopen pattern).

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/backend/faer_interop.rs` -- faer bridge pattern
- `ndtensors-rs/crates/ndtensors/src/linalg/` -- SVD, QR implementations + AD rules

### Feature subset plan (based on `torch.linalg` analysis)

Following the [libtorch reference](libtorch_reference.md), the implementation
plan is prioritized as follows (see also
[Issue #12](https://github.com/tensor4all/tensor4all-meta/issues/12)):

**P0 — POC (current skeleton)**:

| Function | Description | AD (rrule/frule) |
|---|---|---|
| `svd` | SVD decomposition | Yes / Yes |
| `qr` | QR decomposition | Yes / Yes |
| `lu` | LU with partial pivoting | Yes / Yes |
| `eigen` | Eigendecomposition (general) | Yes / Yes |

**P1 — Post-POC, high value**:

| Function | Description | Notes |
|---|---|---|
| `svdvals` | Singular values only | Numerically stable gradients; faster |
| `eigvals` | Eigenvalues only | Faster than full decomposition |
| `eigh` | Eigendecomp (Hermitian) | More stable for symmetric matrices |
| `cholesky` | Cholesky (Hermitian PD) | For positive-definite systems |
| `solve` | Solve AX = B | Common; can use LU internally |
| `solve_triangular` | Triangular solve | Needed internally for some AD rules |
| `inv` | Matrix inverse | Common (but `solve` often preferred) |
| `det` / `slogdet` | Determinant / sign+log-abs-det | `slogdet` important for AD stability |
| `norm` / `vector_norm` | Vector/matrix norms | Useful; can be built from reduce ops |

**P2 — Lower priority**:

| Function | Description | Notes |
|---|---|---|
| `lstsq` | Least squares | Less common in tensor networks |
| `pinv` | Moore-Penrose pseudoinverse | Can be built from SVD |
| `matrix_rank` | Numerical rank | Diagnostic tool |
| `cond` | Condition number | Diagnostic tool |
| `matrix_norm` | Matrix norms (Frobenius, nuclear, etc.) | Specialized |

**Not planned**: `lu_factor`, `lu_solve`, `ldl_*`, `tensorsolve`, `tensorinv`,
`matrix_power`, `multi_dot`, `cross`, `householder_product`, `_ex` variants
(Rust uses `Result<>` natively).

---

## chainrules-core + chainrules (POC exists)

> **POC API skeletons exist**. This is the **default** AD system for
> tenferro users. Burn's autodiff is only for NN workloads.

The AD system follows the Julia ChainRulesCore.jl / Zygote.jl pattern:
core traits are separated from the AD engine, and operation-specific AD
rules live with their operations (not in the AD crate).

See [chainrules-core Design](./chainrules_core_design.md) for detailed design rationale.

### chainrules-core (Core AD Traits)

```rust
/// Tangent space definition (like Julia ChainRulesCore.AbstractTangent)
pub trait Differentiable {
    type Tangent: Clone;
    fn zero_tangent(&self) -> Self::Tangent;
    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent;
}

/// Per-operation reverse-mode rule (rrule / pullback)
pub trait ReverseRule<V: Differentiable> {
    fn pullback(&self, cotangent: &V::Tangent) -> AdResult<Vec<(NodeId, V::Tangent)>>;
    fn inputs(&self) -> Vec<NodeId>;
    fn pullback_with_tangents(...) -> AdResult<...>;  // for HVP
}

/// Per-operation forward-mode rule (frule / pushforward)
pub trait ForwardRule<V: Differentiable> {
    fn pushforward(&self, tangents: &[Option<&V::Tangent>]) -> AdResult<V::Tangent>;
}
```

Also provides: `AutodiffError`, `AdResult<T>`, `NodeId`, `SavePolicy`.
`Differentiable` is implemented for `f64`, `f32` in chainrules-core,
and for `Tensor<T>` in tenferro-tensor.

### chainrules (AD Engine)

```rust
/// Reverse-mode tape (like Zygote.jl / TensorFlow GradientTape)
pub struct Tape<V: Differentiable> { ... }

impl<V: Differentiable> Tape<V> {
    pub fn new() -> Self;
    pub fn leaf(&self, value: V) -> TrackedTensor<V>;
    pub fn leaf_with_tangent(&self, value: V, tangent: V::Tangent) -> AdResult<TrackedTensor<V>>;
    pub fn pullback(&self, loss: &TrackedTensor<V>) -> AdResult<Gradients<V>>;
    pub fn hvp(&self, loss: &TrackedTensor<V>) -> AdResult<HvpResult<V>>;
}

/// Reverse-mode wrapper
pub struct TrackedTensor<V: Differentiable> { value, node_id, tape, requires_grad, tangent }

/// Forward-mode wrapper (dual numbers)
pub struct DualTensor<V: Differentiable> { primal, tangent: Option<V::Tangent> }

/// Gradient container
pub struct Gradients<V: Differentiable> { ... }

/// HVP result (forward-over-reverse)
pub struct HvpResult<V: Differentiable> { gradients: Gradients<V>, hvp: Gradients<V> }
```

### Operation-specific AD rules

AD rules live in their operation crates, not in chainrules:

- **tenferro-einsum**: `tracked_einsum`, `dual_einsum`, `einsum_rrule`, `einsum_frule`, `einsum_hvp`
- **tenferro-linalg**: `svd_rrule`/`svd_frule` (and same for QR, LU, eigen) — batched matrix-level AD (Mathieu 2019), broadcasts over batch dims `*`. No `tracked_*`/`dual_*` — tape engine composes via chain rule.
- **tenferro-capi**: Exposes stateless `rrule`/`frule` only via FFI

### Contraction VJP/JVP

Both VJP and JVP delegate to `TensorPrims`, so they work on CPU and GPU uniformly.
The two-tier TensorPrims design provides clean **adjoint pairs** for each
primitive operation:

| Forward | Backward (adjoint) |
|---------|-------------------|
| `trace(A)` | `anti_trace(∂y)` — scatter-add to diagonal |
| `diag(A)` (on Tensor) | `anti_diag(∂y)` — write to diagonal positions |
| `reduce(A, dim)` | `repeat(∂y, dim)` — broadcast gradient |
| `permute(A, p)` | `permute(∂y, p⁻¹)` — inverse permutation |
| `batched_gemm(A, B)` | `∂A = batched_gemm(∂C, B^T)`, `∂B = batched_gemm(A^T, ∂C)` |

**Existing code to reuse**:
- `ndtensors-rs/crates/ndtensors/src/autodiff/` -- backward pass, graph, TrackedTensor
- `ndtensors-rs/crates/ndtensors/src/contract/naive.rs:222` -- contract_vjp
- `ndtensors-rs/crates/ndtensors/src/autodiff/ops/dual_contract.rs` -- JVP

**Verification**:
- Numerical gradient checks (finite difference vs AD)
- Forward-mode vs reverse-mode consistency
- Complex-valued contraction gradients (Wirtinger calculus)

---

## tenferro-capi (POC exists)

> **POC API skeleton exists**: 16 functions covering tensor lifecycle,
> DLPack interop, einsum, and SVD (including AD rules). f64 only,
> stateless rrule/frule. Produces rlib for type sharing with
> `tenferro-tropical-capi` (shared types: `TfeTensorF64`, `tfe_status_t`).

### Design Principles

- **Opaque handles**: `TfeTensorF64` wraps `Tensor<f64>`. Host languages never see Rust internals.
- **Naming convention**: `tfe_` prefix (tenferro), `_f64` suffix for scalar type.
- **Status codes**: `tfe_status_t` (i32) — `TFE_SUCCESS` (0), `TFE_INVALID_ARGUMENT` (-1), `TFE_SHAPE_MISMATCH` (-2), `TFE_INTERNAL_ERROR` (-3).
- **Stateless AD rules**: Only `rrule` (VJP) and `frule` (JVP) are exposed. `Tape` / `TrackedTensor` / `DualTensor` are **not** exposed. Host languages manage their own AD tapes (ChainRules.jl, PyTorch autograd, JAX custom_vjp).
- **DLPack v1.0** for zero-copy tensor exchange (CPU, CUDA, ROCm, managed memory). Memory layout communicated via strides (no explicit memory order parameter).
- **Copy semantics** for convenience functions: `tfe_tensor_f64_from_data` copies the caller's data. For zero-copy, use DLPack.
- **Panic safety**: All functions use `catch_unwind` and convert panics to `TFE_INTERNAL_ERROR`.

### Implemented API (16 functions)

```c
// Opaque type
typedef struct tfe_tensor_f64 tfe_tensor_f64;
typedef int32_t tfe_status_t;

// Tensor lifecycle (8 functions)
tfe_tensor_f64* tfe_tensor_f64_from_data(const double* data, size_t len,
    const size_t* shape, size_t ndim, tfe_status_t* status);
tfe_tensor_f64* tfe_tensor_f64_zeros(const size_t* shape, size_t ndim,
    tfe_status_t* status);
tfe_tensor_f64* tfe_tensor_f64_clone(const tfe_tensor_f64* tensor,
    tfe_status_t* status);
void tfe_tensor_f64_release(tfe_tensor_f64* tensor);
size_t tfe_tensor_f64_ndim(const tfe_tensor_f64* tensor);
void tfe_tensor_f64_shape(const tfe_tensor_f64* tensor, size_t* out_shape);
size_t tfe_tensor_f64_len(const tfe_tensor_f64* tensor);
const double* tfe_tensor_f64_data(const tfe_tensor_f64* tensor);

// DLPack interop (2 functions) — zero-copy tensor exchange
DLManagedTensorVersioned* tfe_tensor_f64_to_dlpack(
    tfe_tensor_f64* tensor, tfe_status_t* status);  // consuming
tfe_tensor_f64* tfe_tensor_f64_from_dlpack(
    DLManagedTensorVersioned* managed, tfe_status_t* status);  // takes ownership

// Einsum (3 functions) — uses string notation
tfe_tensor_f64* tfe_einsum_f64(const char* subscripts,
    const tfe_tensor_f64** operands, size_t num_operands,
    tfe_status_t* status);
void tfe_einsum_rrule_f64(const char* subscripts,
    const tfe_tensor_f64** operands, size_t num_operands,
    const tfe_tensor_f64* cotangent,
    tfe_tensor_f64** grads_out, tfe_status_t* status);
tfe_tensor_f64* tfe_einsum_frule_f64(const char* subscripts,
    const tfe_tensor_f64** primals, size_t num_operands,
    const tfe_tensor_f64** tangents, tfe_status_t* status);

// SVD (3 functions) — with left/right dim indices
void tfe_svd_f64(const tfe_tensor_f64* tensor,
    const size_t* left, size_t left_len,
    const size_t* right, size_t right_len,
    size_t max_rank, double cutoff,
    tfe_tensor_f64** u_out, tfe_tensor_f64** s_out,
    tfe_tensor_f64** vt_out, tfe_status_t* status);
tfe_tensor_f64* tfe_svd_rrule_f64(const tfe_tensor_f64* tensor,
    const size_t* left, size_t left_len,
    const size_t* right, size_t right_len,
    size_t max_rank, double cutoff,
    const tfe_tensor_f64* cotangent_u,   // nullable
    const tfe_tensor_f64* cotangent_s,   // nullable
    const tfe_tensor_f64* cotangent_vt,  // nullable
    tfe_status_t* status);
void tfe_svd_frule_f64(const tfe_tensor_f64* tensor,
    const size_t* left, size_t left_len,
    const size_t* right, size_t right_len,
    size_t max_rank, double cutoff,
    const tfe_tensor_f64* tangent,  // nullable
    tfe_tensor_f64** u_out, tfe_tensor_f64** s_out,
    tfe_tensor_f64** vt_out, tfe_status_t* status);
```

### ChainRules.jl Integration

Julia's ChainRules.jl defines:
- `rrule(f, args...)` -> `(result, pullback)` -- reverse-mode rule
- `frule((Dself, Dargs...), f, args...)` -> `(result, Dresult)` -- forward-mode rule

tenferro-capi exposes stateless `rrule`/`frule` functions that Julia wraps
as ChainRules rules. The AD tape is managed entirely by Julia (Zygote.jl).

Julia example:
```julia
# einsum via C API
result = ccall((:tfe_einsum_f64, libtenferro), Ptr{Cvoid},
    (Cstring, Ptr{Ptr{Cvoid}}, Csize_t, Ptr{Cint}),
    "ij,jk->ik", ops, 2, status)
```

### Backend Loading API (Future)

Backend loading via `tfe_backend_load_cutensor` / `tfe_backend_load_hiptensor`
is planned but not yet in the POC.

```julia
using cuTENSOR_jll
# Load GPU backend via jll-managed path (future)
ccall((:tfe_backend_load_cutensor, libtenferro), Cint, (Cstring,),
      cuTENSOR_jll.libcutensor_path)
```

---

## Future Phase: tenferro-hdf5

HDF5 I/O for `Tensor<T>`, using [`hdf5-rt`](https://github.com/tensor4all/hdf5-rt)
(runtime library loading via dlopen — same pattern as GPU backends).

The primary goal is to define a **common HDF5 file format** that Rust,
Julia, and Python implementations all follow, enabling cross-language
interoperability.

### File Format Specification

A tensor is stored as an HDF5 **N-dimensional dataset** with metadata
attributes. The dataset shape directly represents the tensor's logical
shape — no flattening to 1D.

```
/group/tensor_name           (HDF5 Dataset: N-D array, shape = tensor shape)
├── @format_version          (Attribute: string, e.g. "1.0")
├── @dtype                   (Attribute: string, e.g. "float64", "complex128")
└── @memory_order            (Attribute: string, "column_major" or "row_major")
```

**N-D dataset** (not 1D flat array): The HDF5 dataset's shape matches
the tensor's logical shape (e.g., a `[3, 4, 5]` tensor is stored as a
3×4×5 dataset). This enables direct reading by h5py (`f['tensor'][:]`
→ NumPy array) and HDF5.jl (`read(f, "tensor")` → Julia array) without
custom deserialization logic.

Note: ITensors.jl stores data as a 1D flat array with separate Index
metadata. We deliberately choose N-D datasets for broader interoperability
at the cost of ITensors.jl format compatibility.

**Data type encoding**:

| dtype string | Rust type | Julia type | NumPy dtype |
|-------------|-----------|------------|-------------|
| `float32` | `f32` | `Float32` | `np.float32` |
| `float64` | `f64` | `Float64` | `np.float64` |
| `complex64` | `Complex<f32>` | `ComplexF32` | `np.complex64` |
| `complex128` | `Complex<f64>` | `ComplexF64` | `np.complex128` |
| `int32` | `i32` | `Int32` | `np.int32` |
| `int64` | `i64` | `Int64` | `np.int64` |

**Complex number storage**: HDF5 has no native complex type (until
HDF5 2.0.0, which is adding one). Complex numbers are stored as an
HDF5 compound type. The field naming varies across ecosystems:

| Library | Field names |
|---------|------------|
| h5py (Python) | `"r"`, `"i"` |
| HDF5.jl (Julia) | reads both; writes `"r"`, `"i"` |
| Octave | `"real"`, `"imag"` |

**tenferro convention**: Use `"r"` and `"i"` (h5py/HDF5.jl compatible).
Readers should also accept `"real"`/`"imag"` for interoperability.
When HDF5 2.0.0 native complex types are available, prefer those.

**Memory order**: HDF5 internally stores N-D data in row-major (C) order.
The `memory_order` attribute records the writer's logical convention
(`"column_major"` or `"row_major"`). Details are an open issue (see below).

### Rust API

```rust
use tenferro_hdf5::{read_tensor, write_tensor};
use tenferro_device::LogicalMemorySpace;

// Write tensor to HDF5 file
let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
write_tensor(&file, "group/tensor_name", &a)?;

// Read tensor from HDF5 file
let b: Tensor<f64> = read_tensor(&file, "group/tensor_name")?;
```

### Design Notes

- **Format first**: The file format spec is the primary deliverable.
  Rust, Julia, and Python I/O libraries are implementations of that spec.
- Separate crate to avoid mandatory C library dependency
- Uses `hdf5-rt` (dlopen): no compile-time HDF5 linking required
- Julia integration: Julia provides `libhdf5` path at runtime (same
  as cuTENSOR/hipTensor injection pattern)

### Open Issues

1. **Memory order and dimension convention**: HDF5 is internally
   row-major. Column-major writers (Julia, Fortran, tenferro) must
   choose between:
   - **Option A**: Follow HDF5.jl — reverse dimensions in HDF5
     (`[3,4]` Julia → `[4,3]` HDF5). Julia↔Julia round-trip is
     transparent, but h5py sees reversed shape.
   - **Option B**: Write dimensions as-is (`[3,4]` → `[3,4]` HDF5).
     h5py sees correct shape, but raw byte order is column-major
     inside a row-major container. Requires `memory_order` attribute
     for correct interpretation.
   - **Option C**: Always materialize as row-major before writing.
     All readers see consistent data, but Julia/Fortran writers pay
     a transpose cost.

2. **Complex number field names**: `"r"`/`"i"` is proposed, but
   HDF5 2.0.0 will introduce native complex types. Migration strategy
   (when to switch, backward compat) is TBD.

3. **Structured tensor formats**: How to store `BlockSparseTensor`,
   `DiagTensor` in HDF5. ITensors.jl uses a `"type"` attribute
   (`"Dense{Float64}"`, `"BlockSparse{Float64}"`) with type-specific
   sub-structure. Whether to adopt a similar convention or define a
   new one is TBD.

4. **ITensors.jl format compatibility**: ITensors.jl stores data as
   1D flat arrays with separate Index metadata. Our N-D dataset format
   is intentionally different for broader interoperability. Whether to
   provide an ITensors.jl compatibility reader/writer is TBD.

---

## Future Phase: tenferro-mdarray / tenferro-ndarray (Bridge Crates)

Separate bridge crates for converting between `Tensor<T>` and other
Rust array libraries. Conversions involve memory copies (not zero-copy),
which is acceptable since these are used at ecosystem boundaries, not
in hot loops.

```rust
// tenferro-mdarray crate
use tenferro_tensor::Tensor;
use mdarray::Array;

impl<T: Scalar> From<&Tensor<T>> for Array<T, Dyn> {
    fn from(tensor: &Tensor<T>) -> Self {
        let t = tensor.contiguous(MemoryOrder::RowMajor);  // mdarray is row-major
        Array::from_slice(t.as_slice(), t.dims())
    }
}

impl<T: Scalar> From<&Array<T, Dyn>> for Tensor<T> {
    fn from(array: &Array<T, Dyn>) -> Self {
        Tensor::from_slice(array.as_slice(), array.shape(), MemoryOrder::RowMajor).unwrap()
    }
}
```

```rust
// tenferro-ndarray crate
use tenferro_tensor::Tensor;
use ndarray::ArrayD;

impl<T: Scalar> From<&Tensor<T>> for ArrayD<T> {
    fn from(tensor: &Tensor<T>) -> Self {
        let t = tensor.contiguous(MemoryOrder::RowMajor);  // ndarray is row-major
        ArrayD::from_shape_vec(t.dims().to_vec(), t.as_slice().to_vec()).unwrap()
    }
}

impl<T: Scalar> From<&ArrayD<T>> for Tensor<T> {
    fn from(array: &ArrayD<T>) -> Self {
        // ndarray may not be contiguous; make contiguous first
        let standard = array.as_standard_layout();
        Tensor::from_slice(standard.as_slice().unwrap(), array.shape(), MemoryOrder::RowMajor).unwrap()
    }
}
```

**Design notes**:
- Each bridge crate depends only on `tenferro-tensor` + the target library
- Memory order is handled automatically: both mdarray and ndarray use
  row-major; tenferro defaults to column-major, so conversions transpose
- `contiguous()` call ensures data is laid out correctly before copy
- No feature flags on `tenferro-tensor` — dependencies are fully isolated

---

## Future Phase: tenferro-structured-rs (separate workspace)

Structured tensor types built on top of `Tensor<T>`. These live in a
**separate workspace** so they can be used by projects other than
tensor4all-rs without pulling in application-level dependencies.

### tenferro-diag

```rust
pub struct DiagTensor<T: Scalar> {
    diag: Tensor<T>,
    full_sizes: Vec<usize>,
}
```

### tenferro-blocksparse

Follows the **ITensors.jl/NDTensors pattern**: all blocks in a **single contiguous `DataBuffer`** with block offset mapping.

```rust
pub struct BlockSparseTensor<T: Scalar> {
    buffer: DataBuffer<T>,
    block_offsets: HashMap<BlockIndex, usize>,
    block_sizes: HashMap<BlockIndex, Vec<usize>>,
    full_sizes: Vec<usize>,
}
```

### tenferro-graded (future)

Quantum number graded tensors. BlockSparseTensor with sector-labeled
block indices and fusion-rule-constrained block structure.

---

## Future Phase: burn-tenferro (Optional Burn Backend for NN)

**Only needed when**: Users want Burn's NN modules with tenferro tensors.

| Burn Method | tenferro Implementation |
|---|---|
| `float_add` | `TensorPrims::reduce` or strided-kernel `axpy` |
| `float_matmul` | `TensorPrims contract (extended op)` (or decomposed to `batched_gemm`) |
| `float_mul` | `TensorPrims elementwise_mul (extended op)` |
| `float_exp/sin/...` | strided-kernel map (CPU only) |
| `float_reshape/permute/slice` | Metadata-only (zero-copy) |
| `float_sum/sum_dim` | `TensorPrims::reduce` (ReduceOp::Sum) |

---

## GPU Strategy

### Runtime Library Loading (Not Compile-Time Features)

GPU support is entirely runtime-based. A single compiled binary supports
CPU, NVIDIA, and AMD:

```
Build once -> Ship one binary
                |
            Runtime:
            |-- libcutensor.so found? -> NVIDIA GPU enabled
            |-- libhiptensor.so found? -> AMD GPU enabled
            +-- Neither? -> CPU only (no error, graceful fallback)
```

No `#[cfg(feature = "cuda")]` or `#[cfg(feature = "hip")]` in the
codebase (except for GPU-specific buffer allocation internals).

### cuTENSOR / hipTensor API Compatibility

AMD intentionally mirrors NVIDIA's API. The mapping is 1:1:

| cuTENSOR | hipTensor |
|----------|-----------|
| `cutensorCreate` | `hiptensorCreate` |
| `cutensorCreateTensorDescriptor` | `hiptensorCreateTensorDescriptor` |
| `cutensorCreateContraction` | `hiptensorCreateContraction` |
| `cutensorCreatePermutation` | `hiptensorCreatePermutation` |
| `cutensorCreateElementwiseBinary` | `hiptensorCreateElementwiseBinary` |
| `cutensorCreateReduction` | `hiptensorCreateReduction` |
| `cutensorCreatePlan` | `hiptensorCreatePlan` |
| `cutensorContract` | `hiptensorContract` |

This enables a single `TensorLibVtable` struct populated from either library.

### GPU Backend Design (Future)

When GPU support is added, the `tenferro-device` crate will be extended with:

```rust
/// BackendRegistry manages all devices (CPU + GPU).
pub struct BackendRegistry {
    cpu: CpuBackend,
    gpu: Option<GpuBackend>,
}

impl BackendRegistry {
    pub fn new() -> Self;  // CPU only
    pub fn load_cutensor(&mut self, path: &str) -> Result<()>;
    pub fn load_hiptensor(&mut self, path: &str) -> Result<()>;
    pub fn available_compute_devices(&self) -> Vec<ComputeDevice>;
}

/// GPU vtable abstracting over cuTENSOR / hipTensor.
struct TensorLibVtable {
    create_handle: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32>,
    contract: Symbol<unsafe extern "C" fn(/* ... */) -> i32>,
    // ...
}

pub struct GpuBackend {
    vtable: TensorLibVtable,
    handle: *mut c_void,
    _lib: Library,  // prevent unloading
}
```

`libloading` will be an unconditional dependency (lightweight, no overhead when GPU absent).

### GPU Plan Caching

Plans are expensive to create on GPU. A `PlanCache` avoids re-creation
for repeated contraction patterns:

```rust
pub struct PlanCache {
    cache: HashMap<PlanCacheKey, GpuPlan>,
    capacity: usize,
}
```

Cache key: `(shapes, strides, modes, dtype)`.

### No Metal (Apple GPU) Support

M-series CPUs are fast enough for tensor network workloads (TCI, Quantics,
MPS algorithms). Metal is not supported because:

1. **No cuTENSOR equivalent** -- Apple has no dedicated N-dimensional tensor
   contraction library. Contraction must be decomposed into reshape + matmul,
   which is slow for high-rank tensors (rank-8+ common in tensor networks).
2. **Incompatible API paradigm** -- MPSGraph uses graph-based execution
   (Objective-C), not the operation-level C API used by cuTENSOR/hipTensor.
   Cannot share the dlopen vtable abstraction.
3. **Implementation cost vs benefit** -- The primary use case (tensor networks)
   runs well on CPU. GPU acceleration benefits come from NVIDIA/AMD HPC
   clusters, not Apple laptops.

---

## Custom Scalar Type Support

Users extend the system at the appropriate level:

| Level | What to implement | Result |
|-------|-------------------|--------|
| `Scalar` only | Basic trait bounds | Einsum via naive loop, map/reduce work |
| `Scalar` + custom GEMM | Custom `batched_gemm` in tenferro-prims | Einsum decomposes to diag → trace → permute → custom batched_gemm |
| Full `TensorPrims` | Custom backend implementation | Complete control over all core operations |
| Full `TensorPrims<A>` with extensions | Custom backend with extended ops (contract, elementwise_mul) | Maximum performance (has_extension_for returns true) |

**Algebra-parameterized dispatch** (via `TensorPrims<A>`):
- `impl TensorPrims<Standard> for CpuBackend` → faer/cblas GEMM for f64/f32/Complex, naive for i32/i64
- `impl TensorPrims<MaxPlusAlgebra> for CpuBackend` (tenferro-tropical) → tropical-gemm SIMD [future]
- `impl TensorPrims<Standard> for GpuBackend` [future] → cuTENSOR/hipTensor
- `impl TensorPrims<MyAlgebra> for CpuBackend` (user crate) → user-provided kernels

---

## Migration Strategy (tensor4all-rs)

1. Replace `tensor4all-tensorbackend::Storage` with tenferro types
2. Replace `DenseStorage<T>` (mdarray-based) with `Tensor<T>`
3. Adapt `TensorLike` trait to use tenferro Tensor
4. Remove mdarray dependency from tensor4all-rs core

This happens **after tenferro core is stable**.

---

## Verification Plan

### After POC (Phase 1 core):
```bash
cd tenferro-rs && cargo test --workspace
```
- Unit tests for all Tensor operations
- Zero-copy verification: assert same buffer pointer after view ops
- **TensorPrims conformance tests**: verify CpuBackend passes the same
  test suite that GPU backends will use
- Integer type einsum test (i32, i64 via naive backend)
- Benchmark: tenferro einsum vs current tensor4all-rs mdarray-einsum

**chainrules-core + chainrules**:
- Numerical gradient checks (finite difference vs reverse-mode AD)
- Forward-mode vs reverse-mode consistency
- HVP correctness (forward-over-reverse)
- Complex-valued gradient test (Wirtinger calculus)

**tenferro-linalg**:
- Batched matrix SVD/QR/LU/eigen correctness (2D and batched `(m, n, *)`)
- Matrix-level AD rule correctness: finite-difference vs svd_rrule/svd_frule
- Complex matrix SVD test (Wirtinger calculus)
- Batched AD: verify broadcasting over batch dims `*`
- Cross-validate results against ndtensors-rs

**tenferro-capi + tenferro-tropical-capi**:
- Round-trip test: Julia -> C API -> Rust -> C API -> Julia
- Einsum rrule/frule roundtrip test
- SVD rrule/frule roundtrip test
- Tropical einsum roundtrip test (MaxPlus, MinPlus, MaxMul)
- ChainRules.jl integration test with Zygote.jl

### After future phases:

**tenferro-algebra**:
- Tropical semiring contraction test (tenferro-algebra + naive backend)
- Custom type extensibility tests: `ModInt<P>` test type through
  all three dispatch tiers

**tenferro-capi (backend loading)**:
- Backend loading test: `tfe_backend_load_cutensor` / `tfe_backend_load_hiptensor`

---

## Key Files Reference

| Component | Source | Destination |
|---|---|---|
| Scalar trait | `tenferro-rs/tenferro-algebra/src/lib.rs` | tenferro-owned (replaces strided-traits' ScalarBase). strided-rs' ScalarBase still used internally by tenferro-prims |
| Conjugate trait | `tenferro-rs/tenferro-algebra/src/lib.rs` | tenferro-owned (replaces strided-traits' ElementOpApply) |
| StridedArray/View | `strided-rs/strided-view/src/` | Stays in strided-rs; used directly in tenferro-prims only (not tensor/device/einsum/linalg) |
| map/reduce kernels | `strided-rs/strided-kernel/src/` | Stays in strided-rs; tenferro-prims will depend on it |
| einsum2_into | `strided-rs/strided-einsum2/src/lib.rs` | **Absorbed** into tenferro-prims `PrimDescriptor::Contract` (CPU contraction) [future] |
| BgemmBackend | `strided-rs/strided-einsum2/src/backend.rs` | **Absorbed** into tenferro-prims `TensorPrims::batched_gemm` [future] |
| reduce_trace_axes | `strided-rs/strided-einsum2/src/trace.rs` | **Absorbed** into tenferro-prims `TensorPrims::trace` [future] |
| diagonal_view | `strided-rs/strided-view/src/view.rs` | Stays in strided-rs; used by `Tensor::diagonal()` [future] |
| opteinsum | `strided-rs/strided-opteinsum/src/lib.rs` | **Absorbed** into tenferro-einsum [future] |
| Algebra traits | `omeinsum-rs/src/algebra/` | **Absorbed** into tenferro-algebra [future] |
| Backend trait | `omeinsum-rs/src/backend/traits.rs` | **Absorbed** into tenferro-prims (evolved into TensorPrims) |
| cuTENSOR wrapper | `omeinsum-rs/src/backend/cuda/cutensor/` | **Absorbed** into tenferro-device (GPU vtable) [future] |
| PlanCache | `omeinsum-rs/src/backend/cuda/cutensor/contract.rs` | **Absorbed** into tenferro-device [future] |
| Matrix AD rules | `ndtensors-rs/.../linalg/` | tenferro-linalg: batched matrix-level svd_rrule/svd_frule etc. (POC API exists) |
| faer bridge | `ndtensors-rs/.../faer_interop.rs` | tenferro-linalg (may use faer internally for CPU path) |
| contract_vjp | `ndtensors-rs/.../contract/naive.rs` | tenferro-einsum einsum_rrule (POC API exists) |
| TrackedTensor | `ndtensors-rs/.../autodiff/tensor.rs` | chainrules TrackedTensor (POC API exists) |
| C API patterns | `tensor4all-rs/crates/tensor4all-capi/src/` | tenferro-capi (POC: 16 functions) + tenferro-tropical-capi (POC: 9 functions) |
| Tropical scalars | `tenferro-rs/tenferro-tropical/src/scalar.rs` | MaxPlus, MinPlus, MaxMul wrappers (POC API exists) |
| Tropical algebra | `tenferro-rs/tenferro-tropical/src/algebra.rs` | Algebra markers + HasAlgebra + Semiring impls (POC API exists) |
| Tropical prims | `tenferro-rs/tenferro-tropical/src/prims.rs` | TensorPrims impls for CpuBackend (POC API exists) |
| Tropical argmax | `tenferro-rs/tenferro-tropical/src/argmax.rs` | ArgmaxTracker for AD backward pass (POC API exists) |
| Tropical C-API | `tenferro-rs/tenferro-tropical-capi/src/lib.rs` | 9 FFI functions for tropical einsum (POC API exists) |

---

## Future Considerations

> **Detailed design**: See [tenferro Design](./tenferro_design.md#future-considerations) for full details on these topics.

The following tenferro-specific design topics are documented in the tenferro-rs repository:

- **GPU/CPU overlap and async execution** — `CompletionEvent` embedded in `Tensor<T>`, transparent accelerator-to-accelerator chaining, multi-threaded CPU parallelism
- **Tensor / TensorView ownership split** — `Tensor<T>` (owned) + `TensorView<'a, T>` (borrowed), compile-time buffer uniqueness guarantees, two-tier API (public waits, internal propagates events)
- **einsum variants** — allocating, accumulating (`_into`), and consuming (`_owned`) API families
- **Complex-valued differentiation** — Wirtinger calculus for complex SVD/QR/eigen backward rules (chainrules-core + tenferro-linalg)
- **JAX/PyTorch integration via C-FFI** — tenferro-capi (POC API exists for einsum + SVD)
- **Multi-GPU distributed contraction** — batch-level parallelism, tensor splitting, contraction tree parallelism
- **ITensor Julia ecosystem insights** — sparse storage, axis fusion patterns
- **mdarray / mdarray-linalg relationship** — numpy-equivalent vs PyTorch-equivalent positioning
