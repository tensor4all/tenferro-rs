# Einsum Design

Einsum is a standard extension crate, not part of the `tenferro` facade.
The public user-facing paths live under `tenferro_einsum`, for example
`tenferro_einsum::traced_tensor::einsum` for traced graph construction and
`tenferro_einsum::eager_tensor::einsum` for immediate eager execution.

`tenferro` must not expose einsum facade paths such as `tenferro::einsum`,
`tenferro::traced_tensor::einsum`, `tenferro::eager_tensor::einsum`,
`tenferro::tensor::einsum`, or `tenferro::typed_tensor::einsum`. Programs that
use traced einsum must explicitly register the extension runtime with their
executor.

The implementation is split between:

- `tenferro-einsum/src/traced.rs` for the user-facing traced API,
  contraction strategy selection, symbolic-shape handling, and graph cache
  integration,
- `tenferro-einsum/src/extension.rs` for runtime extension execution,
- `tenferro-einsum/src/syntax/` for subscript and nested-order parsing,
- `tenferro-einsum/src/planning/` for contraction tree planning and per-step
  lowering plans,
- `tenferro-einsum/src/builder.rs` for graph-fragment lowering,
- `tenferro-einsum/src/eager_tensor.rs` for eager tensor execution.

Historical design notes that refer to direct `CudaBackend`/`RocmBackend`,
`tenferro-prims`, or the old nine-function einsum API are not current.

---

## Public Traced API

The extension crate exposes lazy traced einsum:

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_einsum::traced_tensor::einsum;

let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let mut compiler = GraphCompiler::new();
let c = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
let program = compiler.compile(&c).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_einsum::register_runtime).unwrap();
let result = executor.run(&program).unwrap();
assert_eq!(result.shape(), &[2, 2]);
```

`einsum_with` accepts an explicit `EinsumOptimize` strategy:

| Strategy | Meaning |
| --- | --- |
| `Auto(ContractionOptimizerOptions)` | TreeSA/omeco path optimization with configured score |
| `False` | left-to-right contraction |
| `Nested(NestedEinsum)` | explicit parenthesized contraction tree |
| `Path(Vec<(usize, usize)>)` | JAX-compatible shrinking-list path |
| `Tree(ContractionTree)` | precomputed tree |

`EinsumOptimize::default()` is time-optimized automatic planning.

## Eager Tensor API

`tenferro_einsum::eager_tensor` exposes immediate execution over `EagerTensor`
values:

```rust
use tenferro::{EagerRuntime, Tensor};
use tenferro_einsum::eager_tensor::einsum;

let ctx = EagerRuntime::new();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let a = ctx.constant_from(a);
let b = ctx.constant_from(b);
let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();

assert_eq!(c.shape(), &[2, 2]);
```

Runtime-owned concrete execution is internal to the extension runtime. It is
not exposed through `tenferro`.

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
cache. Runtime contraction trees are keyed by `(subscripts, input shapes)` so
repeated symbolic-shape runs with the same concrete shapes amortize planning
cost.

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

## GPU Interaction

Einsum itself remains backend-agnostic at the graph level. GPU execution happens
when a compiled program is evaluated with `CubeclBackend` from
`tenferro-internal-tensor/src/cubecl/`.

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
AD rules from `tenferro-internal-ops/src/ad/`.

## Tests

Primary local checks for this surface are:

```bash
cargo test -p tenferro-einsum
cargo test -p tenferro-einsum --doc
```

GPU-specific execution tests require CUDA and are ignored by default; see
[gpu-backend-design.md](./gpu-backend-design.md) for the command.
