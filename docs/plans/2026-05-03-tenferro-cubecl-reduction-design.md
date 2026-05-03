# tenferro-cubecl Reduction Split Design

Issue: <https://github.com/tensor4all/tenferro-rs/issues/832>

## Summary

Split tenferro's CubeCL kernel layer into a dedicated `tenferro-cubecl`
workspace crate, starting with the reduction kernel family. The new crate
should follow the useful parts of `cubek`'s blueprint/routine/launch
architecture while keeping tenferro's current CubeCL fork dependency.

The first implementation should not replace the full GPU backend. It should
move reduction kernels and their launch policy out of `tenferro-tensor`, while
leaving tensor ownership, placement checks, upload/download, cuTENSOR,
cuSOLVER, cuBLAS, and `TensorBackend` dispatch in `tenferro-tensor`.

## Architecture

`tenferro-cubecl` is a CubeCL kernel crate. It must not depend on
`tenferro-tensor` and must not know about `Tensor`, `TypedTensor`, `Placement`,
or `ComputeDevice`.

The first crate layout should stay feature-local:

```text
tenferro-cubecl/
  Cargo.toml
  src/
    lib.rs
    error.rs
    reduce/
      mod.rs
      definition.rs
      launch.rs
      routines.rs
      kernels.rs
      cpu_reference.rs
```

This keeps the first PR small while still establishing the structure needed for
later scatter/indexing, fusion, and LU postprocessing work. If multiple kernel
families start sharing substantial code, shared `definition`, `launch`,
`routines`, or `components` modules can be lifted to the crate root later.

The crate boundary is:

```text
tenferro-tensor
  - TensorBackend implementation
  - TypedTensor / CubeclBuffer ownership
  - device residency checks
  - output allocation
  - tenferro Error mapping
  - axis-removal shape semantics

        calls

tenferro-cubecl
  - reduction problem definitions
  - launch validation
  - cubek-style routine selection
  - CubeCL #[cube] kernels
  - optional CPU reference helpers
```

## Reduction Scope

The first kernel family covers:

| Operation | Dtypes |
| --- | --- |
| `reduce_sum` | `f32`, `f64`, `i64`, `Complex32`, `Complex64` |
| `reduce_prod` | `f32`, `f64`, `i64`, `Complex32`, `Complex64` |
| `reduce_max` | `f32`, `f64` |
| `reduce_min` | `f32`, `f64` |

Complex `sum` and `prod` stay in scope because they only need basic complex
addition, multiplication, and identity values, which the current CubeCL fork
supports. Complex `max` and `min` remain unsupported because complex numbers
have no canonical ordering.

`i64 reduce_sum` and `i64 reduce_prod` are in scope to restore backend
capability parity with the CPU backend. `i64 reduce_max` and `i64 reduce_min`
are explicit non-goals for this first split because the CPU backend does not
currently support them either; adding them would be a separate CPU/GPU
capability expansion.

## Data Flow

`tenferro-tensor` adapts tenferro tensors into CubeCL tensor bindings:

```text
TypedTensor<T>
  -> ensure_resident_on_runtime(...)
  -> CubeclBuffer<T>
  -> TensorBinding<CudaRuntime> { handle, shape, strides }
  -> tenferro_cubecl::reduce::launch(...)
  -> output CubeclBuffer<T>
  -> TypedTensor<T> with tenferro output shape
```

`tenferro-cubecl` public launch functions should accept `ComputeClient<R>` and
`TensorBinding<R>` rather than tenferro runtime types. `tenferro-tensor` remains
responsible for allocating output buffers and converting kernel errors into
`crate::Error`.

The kernel-internal output shape may use keepdims semantics when that matches
the reduction routine. `tenferro-tensor` must adapt the final `TypedTensor`
shape back to tenferro's axis-removal semantics without copying when element
counts match.

## Layout Contract

Column-major layout must be explicit at the crate boundary. The reduction crate
must not rely on ad hoc flat-index helpers that silently assume tenferro's
layout.

Rules:

- `tenferro-tensor` passes explicit shape and strides in each `TensorBinding`.
- Dense column-major tenferro tensors use strides `[1, d0, d0*d1, ...]`.
- `tenferro-cubecl` computes offsets from runtime strides or CubeCL tensor
  metadata.
- Layout-specific fast paths may exist, but they must be represented as
  routine or strategy choices, not as hidden global assumptions.
- Shape, strides, lengths, and handles are runtime metadata. `Blueprint`
  values should only contain choices that change generated kernel structure.

## Cubek Alignment

The design should follow these `cubek` principles:

- Keep `Blueprint` minimal and limited to JIT-structural choices.
- Keep tensor shape, strides, lengths, and handles as runtime arguments.
- Make routines choose algorithm and launch settings from problem metadata and
  device capability.
- Keep launch validation separate from kernel device code.
- Keep CPU reference helpers feature-gated.

Direct dependency on `cubek` is out of scope while tenferro depends on the
`shinaoka/cubecl` fork for complex support. Upstream `cubek` uses a different
CubeCL revision, so direct dependency would create incompatible `cubecl` crate
identities. Instead, adapt the required reduction algorithms source-side into
`tenferro-cubecl`.

Any cubek-derived implementation must preserve copyright and license notices,
record the source commit and source paths, and mark tenferro-specific changes.
The copied or adapted scope should be narrow; do not vendor unrelated cubek
crates or kernel families.

## Error Handling

`tenferro-cubecl` owns a small typed error surface, for example:

```rust
pub enum CubeclKernelError {
    InvalidAxis { axis: usize, rank: usize },
    MismatchOutputShape { expected: Vec<usize>, actual: Vec<usize> },
    UnsupportedDType { op: ReduceOp, dtype: ReduceDType },
    InvalidStrategy { reason: String },
}
```

Validation failures should return `Result`, not panic. Kernel-internal true
invariants may be compile-time validation or debug assertions. `tenferro-tensor`
maps `CubeclKernelError` to the existing tenferro error contract, normally
`Error::BackendFailure { op, message }` for GPU backend failures.

## Linalg Boundary

`tenferro-cubecl` is not a replacement for vendor linalg libraries. GPU linalg
factorizations remain in `tenferro-tensor` and continue to use cuSOLVER/cuBLAS:

| Operation | Backend |
| --- | --- |
| SVD | cuSOLVER `gesvd` |
| QR | cuSOLVER `geqrf` plus `orgqr` / `ungqr` |
| Cholesky | cuSOLVER `potrf` |
| LU | cuSOLVER `getrf` |
| Eigh | cuSOLVER `syevd` / `heevd` |
| triangular solve | cuBLAS `trsm` |
| general `eig` | unsupported on GPU because cuSOLVER has no LAPACK `geev` equivalent |

Future linalg-related CubeCL work should be postprocessing only, such as
building LU `P`, `L`, `U`, and `parity` on device from compact LU plus pivots
after `getrf`.

## Testing

Testing has two layers.

`tenferro-cubecl` should cover:

- launch validation,
- output shape validation,
- dtype/op support tables,
- CPU reference helpers where practical,
- ignored CUDA tests for reduction kernels.

`tenferro-tensor` should cover:

- backend-level GPU reduction behavior,
- `i64 reduce_sum` and `i64 reduce_prod`,
- nontrivial column-major shapes such as `[2, 3, 4]`,
- reductions over axis `0`, `1`, `2`, and multiple axes,
- final tenferro axis-removal output shape even if the kernel used keepdims
  internally.

Targeted verification for the first implementation PR:

```sh
cargo fmt --all --check
cargo test -p tenferro-cubecl --features cubecl
cargo test -p tenferro-tensor
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.0 \
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-tensor --features cubecl reduction -- --ignored
```

The full repository pre-push checklist still applies before opening a PR. CUDA
ignored tests require an NVIDIA CUDA 12 environment and must be reported
explicitly if unavailable.

## Rollout

Recommended PR sequence:

1. Create `tenferro-cubecl`, adapt the reduction family, and route
   `CubeclBackend` reductions through the new crate.
2. Refine reduction routines and performance if the first implementation keeps
   conservative algorithms.
3. Add further kernel families under the established crate boundary, starting
   with scatter/indexing or LU postprocessing.

## Non-Goals

- Do not replace cuTENSOR, cuSOLVER, or cuBLAS with CubeCL kernels.
- Do not add a direct `cubek` dependency while CubeCL revisions are
  incompatible.
- Do not implement scatter, indexing, fusion, or LU postprocessing in the first
  reduction split.
- Do not introduce implicit CPU/GPU transfers.
- Do not silently change tenferro's column-major semantics.
- Do not add `i64 reduce_max` or `i64 reduce_min` in this first split.
