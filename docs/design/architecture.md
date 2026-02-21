# Architecture

## Scope

This is the top-level architecture document for tenferro-rs. It covers:

- Workspace layer structure and crate dependency graph
- Device layer (`tenferro-device`)
- Compile-time vs runtime decision summary
- Relationship with mdarray and ITensor ecosystems

Per-crate API details are in companion documents:
[tensor-prims](./tensor-prims.md),
[einsum](./einsum.md),
[tensor](./tensor.md),
[algebra](./algebra.md),
[autodiff](./autodiff.md),
[linalg](./linalg.md),
[contraction-pipeline](./contraction-pipeline.md).

## Layered Architecture

```
Layer 5: tenferro-capi         — C-API (FFI) for Julia/Python: einsum + SVD, f64, stateless rrule/frule
Layer 4: tenferro-einsum       — High-level einsum on Tensor<T>, N-ary tree, algebra dispatch, AD rules
         tenferro-linalg       — Tensor-level SVD/QR/LU/eigen, linalg AD rules
Layer 3: tenferro-prims        — "Tensor BLAS": TensorPrims<A> trait, plan-based execution
                                 (depends on tenferro-tensor for resolve_conj)
Layer 2: tenferro-tensor       — Tensor<T> = DataBuffer + shape + strides, zero-copy view ops,
                                 impl Differentiable for Tensor<T>
Shared:  chainrules-core       — Core AD traits: Differentiable, ReverseRule<V>, ForwardRule<V>
         chainrules            — AD engine: Tape<V>, TrackedTensor<V>, DualTensor<V>
         tenferro-algebra      — HasAlgebra trait, Semiring trait, Standard type
         tenferro-device       — Device enum, Error/Result types
Layer 1: CPU backends          — strided-kernel + GEMM (faer/cblas)
         GPU backends          — cuTENSOR / hipTensor via tenferro-device vtable [future]

Foundation: strided-rs         — Independent workspace (strided-traits → strided-view → strided-kernel)
```

### Design Rationale

strided-rs serves as a "tensor-level BLAS" — the CPU counterpart to
cuTENSOR — with standardized interfaces applicable to CPU, GPU, and tropical
tensors. `tenferro-prims` defines a **universal set** of primitive operations
(`batched_gemm`, `trace`, `diag`, `permute`, `repeat`, `anti_diag`,
`anti_trace`) that any backend must implement, plus an **extended set** of
optimized composites (`contract`, `elementwise_mul`) that backends may
optionally provide.

The core operations form **adjoint pairs** for clean AD support:
`trace ↔ anti_trace`, `diag ↔ anti_diag`, `reduce ↔ repeat`,
`permute ↔ inverse permute`, `batched_gemm` uses the Leibniz rule.

**POC status**: The [tenferro-rs POC](https://github.com/tensor4all/tenferro-rs/)
implements the full workspace structure with API skeletons (`todo!()` bodies).
All crates exist including `tenferro-linalg`, `tenferro-capi`,
`chainrules-core`, and `chainrules`. GPU backend types (`CudaBackend`,
`RocmBackend`) are defined as stubs in `tenferro-prims`; actual GPU
implementation, `BackendRegistry`, and `TensorLibVtable` are future work.

---

## tenferro-device

The device crate provides shared infrastructure used across all tenferro crates.

```rust
/// Logical memory space where tensor buffers reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalMemorySpace {
    MainMemory,
    PinnedMemory,
    GpuMemory { device_id: usize },
    ManagedMemory,
}

/// Compute device where kernels execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeDevice {
    Cpu { device_id: usize },
    Cuda { device_id: usize },
    Rocm { device_id: usize },
}

/// Operation kind used for capability filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind { Contract, BatchedGemm, Reduce, Trace, Permute, ElementwiseMul }

/// Returns executable compute devices in descending preference order.
pub fn preferred_compute_devices(
    space: LogicalMemorySpace,
    op_kind: OpKind,
) -> Result<Vec<ComputeDevice>>;
```

**Error types** using `thiserror`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("rank mismatch: expected {expected}, got {got}")]
    RankMismatch { expected: usize, got: usize },

    #[error("device error: {0}")]
    DeviceError(String),

    #[error("no compatible compute device for {op:?} in memory space {space:?}")]
    NoCompatibleComputeDevice { space: LogicalMemorySpace, op: OpKind },

    #[error("operation across distinct logical memory spaces is not allowed by default")]
    CrossMemorySpaceOperation,

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Dependencies**: `strided-view` (for `StridedError`), `thiserror`.

**Note**: `BackendRegistry`, `CudaBackend`, `RocmBackend`, and
`TensorLibVtable` are **not** in the POC. They are planned for future
GPU support.

---

## Crate Dependency Graph

```
strided-rs (independent workspace):
strided-traits -> strided-view -> strided-kernel

tenferro-rs (workspace, depends on strided-rs):

tenferro-device              tenferro-algebra
  (LogicalMemorySpace +        (HasAlgebra trait,
   ComputeDevice, Error,       Semiring trait,
   Result)                     Standard type)
  (depends on: strided-view
   for StridedError,
   thiserror)
                              (depends on: strided-traits,
                               num-complex)
        │                              │
        ├──────────────┐       ┌───────┤
        │              ↓       ↓       │
        │         tenferro-tensor      │
        │           (Tensor<T> =       │
        │            DataBuffer        │
        │            + dims/strides/   │
        │            offset, view ops) │
        │           (depends on:       │
        │            tenferro-device,  │
        │            tenferro-algebra, │
        │            strided-view,     │
        │            strided-traits,   │
        │            num-traits,       │
        │            chainrules-core)  │
        │              │               │
        ↓              ↓               │
   tenferro-prims                      │
     (TensorPrims<A>,                  │
      PrimDescriptor,                  │
      CpuBackend,                      │
      Extension,                       │
      ReduceOp)                        │
     (depends on:                      │
      tenferro-device,                 │
      tenferro-algebra,                │
      tenferro-tensor,                 │
      strided-view,                    │
      strided-traits)                  │
        │                      ┌───────┘
               ↓               ↓
          tenferro-einsum
            (einsum, einsum_with_subscripts,
             einsum_with_plan,
             Subscripts, ContractionTree)
            (depends on: tenferro-device,
             tenferro-algebra,
             tenferro-prims,
             tenferro-tensor,
             strided-traits)
```

POC skeleton crates (API defined, `todo!()` bodies):
- `tenferro-linalg` — Tensor-level linalg wrapper (SVD, QR, eigen) with AD rules
- `chainrules-core` / `chainrules` — AD traits and engine
- `tenferro-capi` — C FFI for Julia/Python integration

Extension crates (separate workspace, POC stubs):
- `tenferro-tropical` — Tropical algebra types, `TensorPrims<MaxPlus>` for CpuBackend

---

## Compile-Time vs Runtime Decision Summary

| Choice | Mechanism | Rationale |
|--------|-----------|-----------|
| GPU vendor (cuTENSOR/hipTensor) | **Runtime** dlopen (future) | Single binary for all platforms; Julia/Python inject .so path |
| CPU GEMM (faer/cblas) | **Compile-time** feature (future) | Fundamentally different linking (pure Rust vs C ABI) |
| Elementwise ops | **Enum-based** in TensorPrims; closures via strided-kernel | cuTENSOR-compatible for GPU; custom closures via strided-kernel (CPU only) |
| libloading | **Always ON** (future, in tenferro-device) | Lightweight, no overhead when GPU absent |
| .so path | **Caller-injected** (future, via tenferro-device) | No auto-search; Julia/Python manage library versions |

---

## Relationship with mdarray / mdarray-linalg

| | mdarray / mdarray-linalg | tenferro-* |
|---|---|---|
| Role | **numpy equivalent** — general-purpose multidimensional array | **PyTorch equivalent** — high-performance tensor library |
| Memory | Owned `Array<T, D>` | `DataBuffer<T>` (CPU/GPU) |
| GPU | No | cuTENSOR, hipTensor (no Metal) |
| Autodiff | No | chainrules-core (VJP/JVP; API skeleton in POC) |
| Dispatch | Direct function calls | `TensorPrims` trait (backend selection) |

Both are needed. mdarray is a foundational array library; tenferro builds a
richer tensor ecosystem with GPU support and automatic differentiation.

tenferro-linalg and mdarray-linalg are **parallel** (both call faer directly),
not serial. `tenferro-linalg` is a thin wrapper over external matrix
decomposition libraries providing tensor-level APIs with numeric dimension
indices (e.g., `svd(tensor, &[0, 1], &[2, 3])`).

```
faer (SVD, QR, eigen)       ← external matrix algorithms
    ^                ^
tenferro-linalg  mdarray-linalg-faer
(Tensor<T>       (Array<T, D>
 -> MatRef)       -> MatRef)
```

## No Metal (Apple GPU) Support

M-series CPUs are fast enough for our workloads (tensor network algorithms).
Metal lacks a cuTENSOR-equivalent tensor contraction library, requiring
reshape+matmul decomposition that would be slow for high-rank tensors. Not
worth the implementation cost.

## ITensor Ecosystem Mapping

| Aspect | ITensor Julia | tenferro | Notes |
|---|---|---|---|
| Sparse storage | DOK-of-Arrays | Single DataBuffer + offset map | tenferro is GPU-friendly |
| Axis fusion | FusionStyle dispatch | Not yet designed | Critical for quantum number tensors |

See [reference/itensor-ecosystem.md](./reference/itensor-ecosystem.md) for
the full ecosystem analysis.

## Gap Analysis: DFT Application (OpenMX) as a Target Use Case

We analysed [OpenMX 3.9.9 GPU](https://www.openmx-square.org/) — a
production DFT code using MPI + OpenACC + cuBLAS/cuSOLVER — to evaluate
whether tenferro-rs could serve as a pure-Rust computation foundation for
similar electronic-structure applications.

### What tenferro-rs already covers

| OpenMX need | tenferro equivalent | Status |
|-------------|---------------------|--------|
| Dense GEMM (`dgemm`/`zgemm`, `cublasDgemm`) | `TensorPrims::BatchedGemm`, einsum | API defined |
| Standard eigenvalue (`dsyev`/`zheev`) | `tenferro-linalg::eigen()` | API defined |
| SVD, QR, LU | `tenferro-linalg` | API defined |
| Complex128 support | `num-complex::Complex64` | Supported |
| GPU memory model | `LogicalMemorySpace::GpuMemory` | Designed |
| Async GPU execution | `CompletionEvent` | Designed |
| C FFI + DLPack | `tenferro-capi` | API defined |

### Gaps identified and decisions

**Add to `tenferro-linalg` (POC phase):**

| Operation | Rationale | Backend mapping |
|-----------|-----------|-----------------|
| `cholesky()` | Overlap matrix S factorisation (`dpotrf`) | faer (CPU), cuSOLVER (GPU) |
| `solve()` | General linear solve A·x = b (Green's functions, NEGF) | faer (CPU), cuSOLVER (GPU) |
| `inv()` | Explicit matrix inversion (LU-based) | Composed from `lu()` + `solve()` |

**Deferred — introduce when application development requires them:**

| Feature | Why deferred | Where it would live |
|---------|-------------|---------------------|
| Generalised eigenvalue (`geig`: A·x = λ·B·x) | Core of SCF loop but requires application-level validation first | `tenferro-linalg` |
| FFT (3D forward/inverse) | Poisson solver; orthogonal to tensor contraction — better served by external `rustfft` + `cuFFT` via `Tensor::view()` | Application layer or thin wrapper crate |
| Sparse / block-sparse matrices | Hamiltonian construction; application manages sparsity, passes dense blocks to tenferro | Application layer + external crates (`sprs`) |
| MPI / distributed parallel | Node-level distribution; tenferro stays single-node (CPU/GPU) | Application layer |
| ScaLAPACK-equivalent distributed solvers | Distributed eigenvalue (`pdsyev`/`pzheev`) | External crate (ELPA Rust binding) |
| Extended `UnaryOp` (`Exp`, `Log`, `Pow`) | XC functionals on grids; handled by `strided-kernel::zip_map` | `TensorPrims` if needed |

### Architectural assessment

The current layered structure (device → algebra → prims → tensor →
einsum/linalg → capi) is **natural and sufficient** for this use case.
No structural changes are needed — adding `cholesky`/`solve`/`inv` to
`tenferro-linalg` is a straightforward extension within the existing
design. Application-specific concerns (sparsity, MPI, FFT) belong in the
application layer, keeping tenferro-rs slim as a general-purpose tensor
library.
