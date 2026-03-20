# Tenferro Prims CUDA Phase-1 Design

This design defines the CUDA phase-1 completion target for `tenferro-prims`.
The goal is to make the public primitive surface truthful, documented, and
GPU-resident without introducing CPU fallbacks for operations that execute on
CUDA.

## Goals

- Freeze the `tenferro-prims` public operation vocabulary in a crate-local
  `README.md`.
- Complete CUDA phase-1 for the current public surface of:
  - `TensorSemiringCore`
  - `TensorSemiringFastPath`
  - `TensorScalarPrims`
  - `TensorAnalyticPrims`
- Keep all execution on GPU memory once tensors are resident on CUDA.
- Avoid introducing new public CUDA-specific traits or descriptors.
- Add persistent caching for custom CUDA kernels compiled at runtime.
- Keep CUDA and ROCm as optional backend-specific Cargo features across crates.

## Implementation Invariants

These invariants are mandatory for phase-1 implementation and must be re-audited
at the end of the work before the final completion commit.

- keep CUDA execution GPU-resident once tensors are on device
- do not add CPU fallback inside CUDA execution paths
- keep `has_*_support()` and planning behavior truthful to current
  implementation
- do not add new public CUDA-specific traits or descriptors
- use direct `cuTENSOR` only where the mapping is clean and truthful
- use custom kernels through the minimal in-tree NVRTC runtime and persistent
  cache
- keep workspace and scratch allocation explicit and bounded
- keep backend features optional and backend-specific (`cuda`, `rocm`)
- keep README, rustdoc, support tables, and tests aligned with the actual
  implementation

## Non-Goals

- Do not add CPU fallback execution for unsupported CUDA operations.
- Do not depend on `cubecl` directly.
- Do not create a backend-agnostic `gpu` Cargo feature.
- Do not claim ROCm execution support beyond truthful stubs in this phase.
- Do not redesign the public family traits.

## Public Surface Freeze

The first implementation task is to create `tenferro-prims/README.md` and make
it the crate-local source of truth for:

- family boundaries
- operation vocabulary
- backend status matrix
- CUDA phase-1 completion target

The README must stay aligned with:

- `tenferro-prims/src/lib.rs`
- `docs/design/tensor-prims.md`
- `docs/design/supported-ops.md`

## Cargo Feature Model

GPU support remains optional and backend-specific.

- `tenferro-tensor` keeps `cuda` and adds `rocm`
- `tenferro-prims` keeps `cuda` and adds `rocm`
- higher-level crates propagate the same feature names downward
- `tenferro` acts as a convenience entrypoint, not the only entrypoint

This preserves standalone usage such as:

- `tenferro-tensor --features cuda`
- `tenferro-prims --features cuda`

and mirrors the same pattern for `rocm`.

## CI Policy

Default CI remains CPU-only because current CI environments do not provide GPU
runtime access.

- standard CI runs with GPU features disabled
- GPU runtime tests run only when CUDA device + required libraries are present
- if possible, a later compile-only CI lane should check `--features cuda` and
  `--features rocm` without requiring runtime execution

Docs must explicitly state that GPU features are opt-in and not exercised by
default CI.

## CUDA Architecture

CUDA phase-1 uses a hybrid backend:

- direct `cuTENSOR` for operations that map cleanly
- custom CUDA kernels for operations that do not map cleanly
- composed GPU-only pipelines for multi-step operations such as `Mean`

No CPU transfer is permitted inside CUDA execution paths.

### cuTENSOR capability audit

The direct/custom split is based on the official cuTENSOR unary operator table,
not on the currently minimal wrapper in `tenferro-prims`.

- real dtypes support direct unary operators such as `SQRT`, `RCP`, `EXP`,
  `LOG`, `ABS`, `NEG`, `SIN`, `COS`, `TAN`, `SINH`, `COSH`, `ASIN`, `ACOS`,
  `ATAN`, `ASINH`, `ACOSH`, and `ATANH`
- complex direct unary support is limited to `IDENTITY` and `CONJ`
- binary direct support remains limited to the library's binary element-wise
  operators such as `ADD`, `MUL`, `MAX`, and `MIN`

This means the existing classification stays mostly intact, but the wrapper
layer must be extended to expose the required cuTENSOR operator constants and
support predicates must remain truthful per dtype.

### Why not depend on `cubecl`

`cubecl` is a full runtime/compiler stack rather than a small helper library.
It would introduce a second GPU execution model alongside the existing
`cuTENSOR` path, plus larger dependency and toolchain surface. The project
should instead borrow two ideas from `cubecl`:

- NVRTC compilation of generated CUDA code
- persistent cache of compilation artifacts keyed by a stable hash

## Primitive Classification

The implementation strategy per primitive is fixed as follows.

### Semiring Core

- `BatchedGemm`: direct `cuTENSOR`
- `ReduceAdd`: direct `cuTENSOR`
- `MakeContiguous`: direct `cuTENSOR`
- `Trace`: custom CUDA kernel
- `AntiTrace`: custom CUDA kernel
- `AntiDiag`: custom CUDA kernel

### Scalar Prims

- `Neg`: direct for ordered real types, custom for complex
- `Conj`: direct for complex, identity fast-path for real
- `Abs`: direct for ordered real types, custom for complex
- `Reciprocal`: direct for ordered real types, custom for complex
- `Real`: identity fast-path for real, custom for complex
- `Imag`: custom
- `Square`: composed GPU op
- `Add`: direct `cuTENSOR`
- `Sub`: composed GPU op
- `Mul`: direct `cuTENSOR`
- `Div`: composed for real, custom for complex
- `Maximum`: direct `cuTENSOR`
- `Minimum`: direct `cuTENSOR`
- `ClampMin`: direct `cuTENSOR`
- `ClampMax`: direct `cuTENSOR`
- `Sum`: direct `cuTENSOR`
- `Prod`: direct `cuTENSOR`
- `Mean`: composed GPU op
- `Max`: direct `cuTENSOR`
- `Min`: direct `cuTENSOR`

### Analytic Prims

- `Sqrt`: direct for ordered real types, custom for complex
- `Rsqrt`: custom
- `Exp`: direct for ordered real types, custom for complex
- `Expm1`: custom
- `Log`: direct for ordered real types, custom for complex
- `Log1p`: custom
- `Sin`: direct for ordered real types, custom for complex
- `Cos`: direct for ordered real types, custom for complex
- `Tan`: direct for ordered real types, custom for complex
- `Tanh`: direct for ordered real types, custom for complex
- `Asin`: direct for ordered real types, custom for complex
- `Acos`: direct for ordered real types, custom for complex
- `Atan`: direct for ordered real types, custom for complex
- `Sinh`: direct for ordered real types, custom for complex
- `Cosh`: direct for ordered real types, custom for complex
- `Asinh`: direct for ordered real types, custom for complex
- `Acosh`: direct for ordered real types, custom for complex
- `Atanh`: direct for ordered real types, custom for complex
- `Pow`: custom
- `Atan2`: custom
- `Hypot`: custom
- `Xlogy`: custom
- `Var`: custom reduction
- `Std`: custom reduction

## Custom Kernel Split

Custom CUDA kernels are split into four source families:

- `pointwise_unary.cu`
- `pointwise_binary.cu`
- `reduction.cu`
- `diagonal_family.cu`

This keeps compile/cache granularity reasonable while separating reduction and
diagonal semantics from ordinary pointwise code.

## Runtime Compilation and Persistent Cache

The custom kernel system lives under `tenferro-prims/src/cuda/custom/`.

- CUDA C++ kernel sources live under `tenferro-prims/src/cuda/kernel_src/`
- Rust loads source with `include_str!`
- kernels are compiled with NVRTC through `cudarc`
- compilation artifacts are cached persistently on disk

### Cache Keys

The stable cache key includes:

- primitive family and op
- dtype
- rank/layout variant
- custom-runtime ABI version
- device SM architecture
- compile options
- kernel source hash
- CUDA driver version
- cuTENSOR library version

### Cache Invalidation

The cache is invalidated by construction through the stable key. The key must
change when any of these inputs change:

- embedded CUDA C++ source bytes
- entrypoint name
- launch/compile options
- SM architecture
- CUDA driver version
- cuTENSOR version
- custom-runtime ABI version constant

The ABI version constant is a manually bumped integer for any change to launch
metadata encoding, scalar packing, or artifact serialization.

### Cache Layout

- root: `TENFERRO_CACHE_DIR` if set
- otherwise: `~/.cache/tenferro/cuda/`
- one key maps to one artifact record
- preferred artifact format: `cubin`
- fallback artifact format: `ptx`
- sidecar metadata stores entrypoint name and dynamic shared memory size

### Cache Usage

Execution first checks:

1. in-process loaded function cache
2. persistent disk artifact cache
3. NVRTC compile path

Tensor payloads never leave device memory during this process.

## CUDA Plan Model

The public family traits remain unchanged. CUDA internal planning is extended to
represent three execution forms:

- `NativeCutensor`
- `CustomKernel`
- `Pipeline`

`Pipeline` is an ordered list of `NativeCutensor` and `CustomKernel` steps with
explicit scratch slots for intermediates.

This model is required for:

- `Sub`
- `Square`
- `Mean`
- multi-stage moment operations

### cuTENSOR Workspace

The current CUDA backend estimates workspace size when creating plans but does
not use that value during execution. Phase-1 must fix this.

- plan metadata must retain the estimated workspace size
- execute-time code must allocate GPU workspace when the requested size is
  non-zero
- null workspace is valid only when the estimated workspace size is zero

Workspace allocation may share the same bounded GPU allocation machinery as
scratch buffers, but it must remain explicit in the internal plan/execution
model.

## Scratch Allocation

Scratch buffers are GPU-resident and managed by `CudaContext`.

- keyed by dtype, size, alignment, and device
- acquired at execute time
- reused across invocations where possible
- returned to a small context-local cache after execution
- bounded to avoid unbounded GPU memory growth

Expected needs:

- `Mean`: one scratch slot at most
- `Var`: at least two scratch slots
- `Std`: same as `Var`, followed by final square-root step
- diagonal-family ops: zero or one scratch slot depending on kernel path

Retention policy:

- at most four cached scratch buffers per `(device, dtype)` bucket
- buffers larger than `64 MiB` are not retained after execution
- excess retained buffers are evicted in least-recently-used order

## Error Model

- unsupported op/dtype/backend phase: `InvalidArgument`
- missing CUDA runtime / missing cuTENSOR / arch mismatch: `DeviceError`
- NVRTC compilation failure: `DeviceError`
- non-GPU inputs passed to CUDA execute: `DeviceError`
- ROCm feature enabled but not implemented: truthful `has_*_support = false`
  and planning failure

`has_*_support()` must reflect what genuinely executes in the current backend,
not future intent.

## Testing Policy

### CPU-only CI tests

- feature-surface tests compile and run without GPU features
- `has_*_support()` truthfulness checks
- unsupported planning checks
- cache key/hash stability tests

### GPU-available tests

- family smoke tests
- `alpha/beta` execution contract
- non-contiguous input coverage
- complex execution paths
- persistent custom-kernel cache hit after first compile

## Documentation Updates

Update:

- `tenferro-prims/README.md`
- `docs/design/tensor-prims.md`
- `docs/design/supported-ops.md`
- public rustdoc where backend behavior changes

README and docs must explicitly say:

- CUDA uses GPU-resident execution only
- CPU fallback inside CUDA execution is forbidden
- ROCm feature surface may exist before full runtime implementation

## References

- NVIDIA cuTENSOR user guide and API
- PyTorch CUDA kernel implementations
- `TensorBFS/tropical-gemm` NVRTC runtime-compilation pattern
- `cubecl` persistent compilation cache pattern
