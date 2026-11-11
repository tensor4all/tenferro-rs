# Einsum Design

Einsum is a standard extension crate, not part of a root facade.
The public user-facing paths live under `tenferro_einsum` as the
registry-generated traits of the 0.12 call grammar (#2552; the operation
table is `docs/spec/generated/api-ops.md`): `EinsumOps` on input slices and
arrays of any `AsView` (`Tensor`, `TensorRead`, `TypedTensor<T>`,
`TypedTensorView<T>`) with `einsum`, `einsum_with`, and `einsum_into`;
`TensordotOps` for concrete `tensordot`; `TracedEinsumOps` on `TraceContext`
for traced graph construction; `TracedTensordotOps` on `TracedTensor` for
traced `tensordot`; and `EagerEinsumOps` on a borrowed `EagerSession` for
autodiff eager `einsum` and `tensordot`. `ConcreteEinsumPlan` owns repeated
concrete executions with fixed input dtype and shape metadata. `tensordot` is not a `tenferro-linalg`
API.

The workspace intentionally has no root `tenferro` crate and no einsum facade
paths. Programs that use traced einsum must explicitly register the extension
module with their `Runtime`.

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

The extension crate exposes lazy traced einsum on the `TraceContext` that
owns the inputs:

```rust
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumOptimize, EinsumOptions, TracedEinsumOps};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};

let trace = TraceContext::new();
let a = trace.from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let b = trace.from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
let c = trace.einsum(&[&a, &b], "ij,jk->ik")?;
let d = trace.einsum_with(
    &[&a, &b, &b],
    "ij,jk,kl->il",
    &EinsumOptions::default().optimize(EinsumOptimize::False),
)?;

let program = GraphCompiler::new().compile_many(&[&c, &d])?;
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = {
    let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
    runtime.run(&mut engines, &prepared, &[])?
};
for output in outputs {
    let t = output.into_tensor(&mut engine)?;
    assert_eq!(t.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
}
```

## Concrete Tensor API

Concrete non-AD execution is exposed through `EinsumOps` on input slices
and arrays, not through public module free functions:

```rust
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{ConjMask, EinsumOps, EinsumOptions};
use tenferro_tensor::{Conj, Dest, Tensor};

let mut engine = CpuEngine::serial();
let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
let c = [&a, &b].einsum("ij,jk->ik", &mut engine)?;

// Options: conjugate the first operand (a no-op for real data).
let opts = EinsumOptions::default().conj(ConjMask::new([Conj::Yes]));
let d = [&a, &b].einsum_with("ij,jk->ik", &opts, &mut engine)?;
assert_eq!(d.as_slice::<f64>()?, c.as_slice::<f64>()?);

// Accumulate into an existing result: out = a b + out.
let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
[&a, &b].einsum_into(
    "ij,jk->ik",
    &EinsumOptions::default(),
    &mut engine,
    Dest::accumulate(&mut out),
)?;
assert_eq!(out.as_slice::<f64>()?, &[2.0, 3.0, 4.0, 5.0]);
```

The input representation, not a method suffix, selects the route: owned
`Tensor` and borrowed `TensorRead` inputs give a dtype-erased `Tensor`,
`TypedTensor<T>` and `TypedTensorView<T>` inputs keep the statically known
scalar type. The spec is a `&str`, `EinsumNotation`, `EinsumSubscripts`, or
`Spec`. `einsum_into` writes into a caller-provided `Dest` (`Overwrite`,
`Accumulate`, or `Scaled`; typed destinations accept an owned typed tensor or a
mutable typed view). All paths validate output dtype and shape before any
writes and never resize the destination.

`ConcreteEinsumPlan` precomputes the contraction tree for fixed input metadata
and validates later executions against the prepared input count, dtypes, and
shapes. It is built with `ConcreteEinsumPlan::prepare(inputs, spec, &opts)`
(the plan records `opts.optimize` and `opts.conj`) and run with `execute`
(new result in the inputs' representation) or `execute_into(.., Dest)`.

`EinsumOptions::optimize` holds an explicit `EinsumOptimize` strategy:

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

`EagerEinsumOps` exposes immediate execution over `EagerTensor` inputs
with the borrowed `EagerSession` as receiver, so eager einsum composes with
other eager operations inside one `with_eager_session` callback and never opens
a session of its own:

```rust
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EagerEinsumOps, EinsumOptimize, EinsumOptions};
use tenferro_tensor::Tensor;

let runtime = EagerRuntime::new()?;
let (c, d) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?)?;
    let b = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?)?;
    let c = s.einsum(&[&a, &b], "ij,jk->ik")?;
    let opts = EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(0, 1)]));
    let d = s.einsum_with(&[&a, &b], "ij,jk->ik", &opts)?;
    Ok::<_, tenferro_ad::Error>((c, d))
})?;
assert_eq!(c.value()?.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
assert_eq!(d.value()?.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
```

The trait also provides `tensordot` / `tensordot_with`. The former
tensor-side eager traits, which each opened their own eager session, were
removed by #1946 F5.

Autodiff eager execution remains separate from concrete `Tensor` execution:
`EagerEinsumOps` is available only when the `autodiff` feature is enabled.

## Subscripts And Repeated Labels

`Subscripts::parse` accepts flat NumPy/PyTorch-style explicit labels and rejects
parenthesized contraction-order notation. Use `NestedEinsum::parse` when
contraction order must be preserved. Rank-polymorphic flat notation is parsed
as [`EinsumNotation`] and resolved only after input ranks are known; one
right-aligned ellipsis per term supports equal-or-one batch broadcasting. The
resolved result is converted to `Subscripts` before planning and execution.
Parenthesized ellipsis and implicit-output equations remain unsupported.

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

The non-AD concrete `_into` APIs use a compact binary-dot configuration path
for supported two-input contractions. Concrete owned and borrowed-view string
calls attempt this path before parsing owned notation; borrowed-view notation
and parsed-label calls inspect
borrowed labels before resolution or owned-subscript conversion. The planner
supports arbitrary ranks and axis positions when unique labels form a supported
exact-output dot; ellipsis, Unicode string labels, repeated labels, broadcasting,
and other unsupported forms keep the existing parser/planner fallback. It uses
borrowed ASCII byte labels for string calls and inline integer-label scratch for
explicit notation. `DotGeneralConfig` owns four `TinyVec<[usize; 4]>` axis
lists: each stores up to four axes inline and spills for larger lists. This is
not a tensor-rank restriction. Ordinary low-rank direct calls therefore need no
axis-list heap allocation. Larger expressions and general fallbacks may still
allocate. The separate `EagerTensor` API has its own AD-aware dispatch and is not
covered by this concrete path.

This is a source-breaking field-type change: construct literal axes with
`[1].as_slice().into()`, empty axes with `Default::default()`, and generated axes
with `.collect()` into the inferred field type. Existing `Vec<usize>` values can
be consumed with `.into()`; avoid constructing an intermediate `Vec` when the
intent is allocation-free inline storage. Axis order, equality, hashing and
canonical graph identity remain value-based. Retained-byte accounting excludes
inline storage already counted in the enclosing operation and counts spilled
storage under each cache's existing logical-length or capacity policy.

CPU rank-two, single-contracting-axis, no-batch GEMM analysis validates the two
axis bounds and paired extent as it derives dimensions. Other or invalid
configurations go through the general validator/analyzer. Placement, layout,
dtype, output, accumulation, and provider checks remain in the checked runtime
path; there is no unchecked public execution entry point.

## Static And Symbolic Shapes

The traced extension API chooses the lowering mode from input shape availability:

| Inputs | Build-time behavior | Runtime behavior |
| --- | --- | --- |
| All concrete shapes | optimize the contraction tree at graph build time and lower into ordinary graph ops where possible | execute the lowered graph |
| Any symbolic shape | emit one einsum extension op; unresolved ellipsis returns a typed planning error | optimize from actual input shapes at runtime |

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
labels, size dictionary, and compiled step plans. Automatic planning without an
annealing schedule (`niters == 0` or empty `betas`, the default) runs the local
deterministic greedy planner: it contracts the connected pair with the smallest
intermediate output, breaks ties by the smallest operand index pair, and falls
back to an outer product of the two smallest live operands only when no pair
shares a label. This is omeco's greedy rule (`alpha = 0`, `temperature = 0`)
with the ordering omeco adopted after 0.2.6; omeco 0.2.6 itself broke ties in
per-process `HashMap` order, so the same spec produced different paths in
different processes (#1963).

Guarantee: for fixed subscripts, shapes and non-annealing options the planned
path is identical in every process. `tests/integration/deterministic_path.rs`
re-executes its test binary and compares the children's paths. With an
annealing schedule the path comes from omeco/TreeSA, which carries no such
guarantee until omeco's own greedy initializer is deterministic in a released
version; if omeco returns no path, the local greedy planner is used.

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
the backend `dot_general_read_into` hook (a provider contract) before the owned-output fallback. The
CPU faer provider writes directly into caller-provided output storage through
its validated `FaerGemm` path and does not allocate the final output tensor. General
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
- ROCm has no implementation and is not a supported execution path.
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
