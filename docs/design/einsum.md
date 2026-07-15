# Einsum Design

Einsum is a standard extension crate, not part of a root facade.
The public user-facing paths live under `tenferro_einsum` as crate-root
extension traits: `TensorEinsumExt` and `TypedTensorEinsumExt` for owned
concrete inputs, `TensorReadEinsumExt` and `TypedTensorReadEinsumExt` for
borrowed inputs, and their `*IntoExt` counterparts for preallocated output
execution,
`GraphCompilerEinsumExt` for traced graph construction, `EagerEinsumExt` for
autodiff eager execution, and tensor extension traits for `tensordot`
contraction sugar. `ConcreteEinsumPlan` owns repeated concrete executions with
fixed input dtype and shape metadata. `tensordot` is not a `tenferro-linalg`
API.

The workspace intentionally has no root `tenferro` crate and no einsum facade
paths. Programs that use traced einsum must explicitly register the extension
runtime with their executor.

The implementation is split between:

- `crates/tenferro-einsum/src/traced.rs` for the user-facing traced API,
  contraction strategy selection, symbolic-shape handling, and graph cache
  integration,
- `crates/tenferro-einsum/src/concrete.rs` for the user-facing concrete tensor,
  read, typed, and prepared-plan APIs,
- `crates/tenferro-einsum/src/extension.rs` for runtime extension execution,
- `crates/tenferro-einsum/src/syntax/` for subscript and nested-order parsing,
- `crates/tenferro-einsum/src/planning/` for contraction tree planning and per-step
  lowering plans,
- `crates/tenferro-einsum/src/builder.rs` for graph-fragment lowering,
- `crates/tenferro-einsum/src/eager.rs` for the shared concrete executor used by
  concrete wrappers and extension runtime execution,
- `crates/tenferro-einsum/src/eager_ad.rs` for eager tensor execution.

Historical design notes that refer to direct `CudaBackend`/`RocmBackend`,
`tenferro-prims`, or the old nine-function einsum API are not current.

---

## Public Traced API

The extension crate exposes lazy traced einsum:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let mut compiler = GraphCompiler::new();
let c = compiler.einsum(&[&a, &b], "ij,jk->ik").unwrap();
let program = compiler.compile(&c).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_einsum::register_runtime).unwrap();
let result = executor.run(&program).unwrap();
assert_eq!(result.shape(), &[2, 2]);
```

## Concrete Tensor API

Concrete non-AD execution is exposed through crate-root extension traits on
input slices and arrays, not through public module free functions:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
let mut backend = CpuBackend::new();
let c = [&a, &b].einsum("ij,jk->ik", &mut backend).unwrap();

assert_eq!(c.shape(), &[2, 2]);
```

`TensorEinsumExt` is the unsuffixed API for compact `Tensor` references.
`TypedTensorEinsumExt` preserves a statically known scalar type for owned typed
tensors. Borrowed typed strided views use
`TypedTensorReadEinsumExt::einsum_read`; dtype-erased borrowed inputs use
`TensorReadEinsumExt::einsum_read`. Both follow the repository-wide `_read`
suffix convention. The matching `*IntoExt` traits write into caller-provided
outputs. Typed output methods accept `TypedTensorWrite`, which represents an
owned typed tensor or a mutable typed view. All paths validate output dtype and
shape before any writes and never resize the destination.

`ConcreteEinsumPlan` precomputes the contraction tree for fixed input metadata
and validates later executions against the prepared input count, dtypes, and
shapes. Plan execution exposes both owned-output methods and `execute_into`,
`execute_typed_into`, and `execute_read_into`.

`einsum_with` accepts an explicit `EinsumOptimize` strategy:

| Strategy | Meaning |
| --- | --- |
| `Auto(ContractionOptimizerOptions)` | TreeSA/omeco path optimization with configured score |
| `False` | left-to-right contraction |
| `Nested(NestedEinsum)` | explicit parenthesized contraction tree |
| `Path(Vec<(usize, usize)>)` | JAX-compatible shrinking-list path; shape-independent and valid for symbolic traced inputs |
| `Tree(ContractionTree)` | concrete/precomputed tree; accepted only when concrete shapes are available |

`EinsumOptimize::default()` is time-optimized automatic planning.
The traced API stores the resolved planning policy as a shape-independent plan
specification in the extension payload. `Path` pairs remain positional over the
current shrinking operand list; `Tree` values are converted to fixed
contraction pairs when accepted for concrete inputs.

## Eager Tensor API

`EagerEinsumExt` exposes immediate execution over `EagerTensor` input
slices/arrays:

```rust
use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_einsum::EagerEinsumExt;

let ctx = EagerRuntime::new();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let a = ctx.constant_from(a).unwrap();
let b = ctx.constant_from(b).unwrap();
let c = [&a, &b].einsum("ij,jk->ik").unwrap();

assert_eq!(c.shape(), &[2, 2]);
```

Autodiff eager execution remains separate from concrete `Tensor` execution:
`EagerEinsumExt` is available only when the `autodiff` feature is enabled.

## Subscripts And Repeated Labels

`Subscripts::parse` accepts flat NumPy/PyTorch-style labels and rejects
parenthesized contraction-order notation. Use `NestedEinsum::parse` when
contraction order must be preserved.

Repeated-label semantics follow the usual einsum rules:

| Pattern | Meaning |
| --- | --- |
| `ii->` | extract the diagonal, then reduce it to a scalar trace |
| `ii->i` | extract the diagonal |
| `iij->ij` | extract the diagonal across the first two axes and preserve `j` |
| `i->ii` | embed the vector on a diagonal matrix |

The implementation applies these rules before ordinary contraction:

1. `diagonalize_repeated` repeatedly applies `extract_diagonal` to duplicate
   labels within one operand.
2. Labels absent from the output or from later live operands are reduced with
   `reduce_sum`.
3. `embed_repeated` applies `embed_diagonal` when the output repeats a label
   more often than the current value.
4. `transpose_to_labels` restores requested output order.

Strict binary/GEMM lowering intentionally rejects repeated labels and returns
`None`. Those cases stay on the general eager/builder path, which handles
diagonalization explicitly.

## Static And Symbolic Shapes

The traced extension API chooses the lowering mode from input shape availability:

| Inputs | Build-time behavior | Runtime behavior |
| --- | --- | --- |
| All concrete shapes | optimize the contraction tree at graph build time and lower into ordinary graph ops where possible | execute the lowered graph |
| Any symbolic shape | emit one einsum extension op | optimize from actual input shapes at runtime |

`tenferro_einsum` caches concrete-shape contraction trees in the extension
cache. Runtime contraction trees are keyed by subscripts, input shapes, and the
resolved planning policy or explicit path so repeated symbolic-shape runs with
the same concrete shapes and policy amortize planning cost without conflating
different optimizer settings. The same plan specification participates in
traced extension payload identity, so otherwise identical ops that use
different planner options or paths remain distinct extension ops. Runtime
extension execution also caches the compiled inner execution program keyed by
subscripts, concrete input shapes, input dtypes, and the resolved planning
policy, so repeated eager or traced extension runs do not rebuild and compile
the lowered inner graph.

## Planning

`ContractionTree` records the pairwise contraction sequence, live operand
labels, size dictionary, and compiled step plans. Automatic planning first asks
omeco/TreeSA for a path. If omeco does not return one, the local self-greedy
fallback chooses the pair with the smallest intermediate output size.

Planner invariants are checked with normal `Result` propagation:

- input rank and shape labels are validated by `build_size_dict`,
- explicit paths must reference distinct live operands,
- the final explicit path must leave exactly one live value,
- contraction-cost labels must have known sizes.

## Lowering And Execution

Each pairwise step classifies labels into:

- left-only free labels,
- right-only free labels,
- shared batch labels that survive,
- shared contraction labels that are reduced.

When a strict binary plan applies, the step caches the canonical matrix/GEMM
layout metadata. When it does not apply, the builder and eager executor use the
general path of diagonalization, reductions, broadcast/outer product, and
`DotGeneral`.

Column-major ordering matters. For GEMM-like steps, compute dimensions stay on
the left and batch dimensions stay on the right so each batch slice remains a
contiguous block for the underlying tensor backend.

For a whole-expression binary GEMM-compatible contraction such as
`ij,jk->ik`, `einsum_into` and `ConcreteEinsumPlan::execute_into` dispatch to
the backend `dot_general_read_into` hook before the owned-output fallback. The
CPU faer path writes directly into caller-provided output storage through
`strided_einsum2` and does not allocate the final output tensor. General
multi-step einsums may still allocate intermediates; their `*_into` contract is
that the final result is copied into the preallocated destination after output
validation.

## GPU Interaction

Einsum itself remains backend-agnostic at the graph level. GPU execution happens
when a compiled program is evaluated with `CudaBackend` from
`crates/tenferro-gpu/src/cubecl/`.

Current GPU status:

- CUDA uses CubeCL/CubeCL-CUDA under the public `cuda` feature.
- cuTENSOR/cuBLAS paths cover selected contractions and GEMM-like operations.
- ROCm is a stub and not a supported execution path.
- Complex CubeCL expansion is blocked on upstream CubeCL support and is not part
  of this batch.
- GPU benchmarking is outside this batch.

## AD

Graph-level AD rules for einsum live in `tenferro-einsum` and are registered as
extension AD rules. Primitive operations emitted by lowering still use the core
AD rules from `crates/tenferro-internal-ops/src/ad/`.

VJP construction preserves the primal planning policy. For explicit
`EinsumOptimize::Path` payloads, the AD rule remaps the positional path to the
VJP operand list so the gradient contraction inherits the caller's selected
order where that order is still meaningful.

## Tests

Primary local checks for this surface are:

```bash
cargo test -p tenferro-einsum
cargo test -p tenferro-einsum --doc
```

GPU-specific execution tests require CUDA and are ignored by default; see
[gpu-backend-design.md](./gpu-backend-design.md) for the command.
