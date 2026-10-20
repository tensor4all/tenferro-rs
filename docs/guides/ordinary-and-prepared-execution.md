# Ordinary calls and prepared execution

Start with the ordinary API. Prepare explicitly when you repeat a compatible
contraction, not merely because the equation is a string. Build the engine and
the runtime once and reuse them. This page uses CPU tensors with **column-major**
data; see [crate setup and full einsum examples](einsum.md) for dependencies.

## Choose by need

| Need | Supported recipe |
|---|---|
| Execute concrete tensors now, without AD | `TensorEinsumExt::einsum` on `[&lhs, &rhs]` with the caller's `&mut engine` |
| Execute eager tensors with AD | Import `EagerSessionEinsumExt` (`autodiff` feature), then `ctx.with_eager_session(&mut engine, \|s\| s.einsum(&[&lhs, &rhs], "ij,jk->ik"))?`; core ops are `EagerSession` methods such as `s.dot_general(&lhs, &rhs, config)` |
| Repeat one compatible concrete contraction | Keep a `ConcreteEinsumPlan`; prepare once and execute with new inputs |
| Repeat a traced computation | Compile once and reuse the program and configured runtime; see [traced execution](einsum.md#traced-matrix-multiply) |
| Generate an equation in code | Use `EinsumSubscripts` integer labels; use `EinsumNotation` for unresolved ellipsis |

Concrete calls take the engine the caller holds. Borrow it for the whole
sequence of compatible operations instead of rebuilding it per call; nothing is
entered or locked. Eager tensors instead retain their `EagerRuntime`, and each
eager session borrows the same engine; keep inputs in the same runtime and
preserve AD/capture semantics. Do not replace eager work with concrete
operations when gradients are needed. See the executable
[eager example](einsum.md#eagertensor).

## Ordinary, prepared, and programmatic recipes

The following is extracted from the tested consumer-skill binary. To run it in
this checkout: `cargo run --manifest-path docs/tutorial-code/Cargo.toml --bin tenferro_compute_skill`.
For a standalone program, place the fragment inside
`fn main() -> Result<(), Box<dyn std::error::Error>>` and finish with `Ok(())`.

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#ordinary-and-prepared-einsum -->
```rust
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::Tensor;

let lhs = Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let rhs = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 1.0, 2.0])?;
let mut engine = CpuEngine::serial();
// Ordinary execution: no explicit preparation needed.
let ordinary = [&lhs, &rhs].einsum("ij,jk->ik", &mut engine)?;
assert_eq!(ordinary.as_slice::<f64>()?, &[2.0, 4.0, 7.0, 10.0]);

// Strings are fine for one-time preparation. The plan does not retain inputs.
let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
// Borrow the same engine for the whole batch. The engine holds the plan cache
// and buffer pool, so reusing it is what makes the per-call cost the numerical
// operation alone.
for (data, expected) in [
    (vec![1.0_f64, 2.0, 3.0, 4.0], [2.0, 4.0, 7.0, 10.0]),
    (vec![2.0_f64, 4.0, 6.0, 8.0], [4.0, 8.0, 14.0, 20.0]),
] {
    let next_lhs = Tensor::from_vec_col_major([2, 2], data)?;
    let result = plan.execute([&next_lhs, &rhs], &mut engine)?;
    assert_eq!(result.as_slice::<f64>()?, &expected);
}

// Integer labels describe the equation; this ordinary call still plans.
let equation = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let structured = [&lhs, &rhs].einsum_subscripts(&equation, &mut engine)?;
assert_eq!(structured.as_slice::<f64>()?, &[2.0, 4.0, 7.0, 10.0]);
```
<!-- end-snippet-source -->

### Reuse contract

Keep the equation, operand ordering, input count, dtypes and shapes compatible
with the prepared plan. Values may change, as the second execution demonstrates.
The plan owns its prepared contraction tree and metadata, not the input tensors
or the engine. Execution still validates inputs and performs backend
work: preparation does not remove validation, allocation, dispatch or required
materialization. Prepare a new plan when the equation or input metadata changes.
Device placement and backend capabilities must still match; a plan does not
transfer data or make an unsupported dtype/device executable. For repeated traced
work, keep the compiled program, the prepared program and the configured runtime,
pass new concrete inputs to `Runtime::run`, and recompile if the program's input
metadata contract changes; register its extension modules before preparation.

Typed owned inputs use `TypedTensorEinsumExt` and `prepare_typed` /
`execute_typed`. Borrowed views use the `*ReadEinsumExt` traits and
`prepare_read` / `execute_read` (or typed-read variants). Preallocated output
uses the matching `*IntoExt` or plan `execute_into` / `execute_read_into` with a
validated write destination; these overwrite rather than accumulate. Choosing
`_read` or `_into` describes input/output ownership, not whether planning is
cached. See [tested view and output recipes](einsum.md#tensorread-and-prepared-plans).

### Representation is not reuse

Integer labels avoid textual parsing; ordinary `einsum_subscripts` still plans.
For repeated generated equations, prepare with `ConcreteEinsumPlan::prepare_subscripts`.
For ellipsis, use `EinsumNotation` and `prepare_notation`. Strings are appropriate
for one-time preparation; there is no general reason to ban them.

**Do not flatten parenthesized contraction order into integer label arrays.**
`EinsumSubscripts` expresses labels, not grouping. The flat ordinary/prepare
recipes above do not accept parentheses. Preserve an intended order through the
supported explicit path/tree controls on the traced route, described in
[einsum optimization controls](einsum.md#optimization-controls), rather than
silently discarding it. Parenthesized ellipsis is unsupported.

## One engine across ordinary, eager and prepared work

Concrete operations, eager sessions and prepared runs all borrow the same
engine, so a stage that mixes them keeps one value instead of one per layer:

```text
let mut engine = CpuEngine::serial();
let eager = EagerRuntime::new()?;
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
    tenferro_cpu::runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;

let product = [&lhs, &rhs].einsum("ij,jk->ik", &mut engine)?;
let loss = eager.with_eager_session(&mut engine, |s| s.einsum(&[&product, &rhs], "ij,jk->ik"))?;
let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
let outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
```

Nothing is entered or locked, so there is no scope to nest and no reentry error
to avoid: the `&mut` borrow is the whole exclusion contract. Returned outputs
remain usable after the borrow ends. The ownership rules behind this shape are
in [execution engine ownership](../design/execution-engine-ownership.md).

For steady-state measurement, create inputs, build the engine, compile and
prepare graphs and warm up **before** starting the clock. Stop the clock before
cleanup. Allocation-returning operations still allocate outputs; a shared engine
is not an output-reuse API (the `_into` forms are). Thread count and CPU affinity
remain pool properties. Reusing one engine removes repeated construction, not
every high-level per-call cost, and is not a promise that four threads outperform
one.

## Interpret overhead without inventing a budget

`1 ms = 1000 us = 1,000,000 ns`. Repeated cost is `N × per-call cost`.
For illustration only, 5 us of setup contributes `5 / (1000 + 5) ≈ 0.5%`
beside 1 ms of useful work, but `5 / (1 + 5) ≈ 83%` beside 1 us.
These are arithmetic examples, not measured tenferro constants.

Two fixed historical Linux records inform this guidance, not a maintained
“latest benchmark” table:

- [Binary einsum record](https://github.com/tensor4all/tenferro-rs/blob/7dfc01127f4a8752a8bb504641feb396683576c3/docs/worklogs/issue-1761-binary-einsum.md):
  the accepted ordinary-string small-matrix aggregate candidate/base ratio was
  0.6851, 95% CI [0.6661, 0.7046]. This is dimensionless, **not a per-call latency**.
  The record retains rejected/inconclusive earlier experiments and the accepted
  explicit-one-worker result, revisions, raw evidence and timing boundaries.
- [Eager/session attribution, 2026-09-06](https://github.com/tensor4all/tenferro-rs/blob/552b4793/docs/worklogs/issue-1762-eager-session.md):
  library `7dfc0112`, Linux `primerose`, AMD EPYC 7713P, Rust 1.97.1,
  release/faer, explicit one worker and separate affinity to CPU 32. Ten dependent
  2x2 F64 matmuls took median 25.182 us in one shared concrete session versus
  191.056 us through ordinary no-AD eager, **per ten-op chain**, not per call.
  Seven process medians had CV 2.5% and 2.4%; paired eager/shared ratio was
  7.572 [7.414, 7.734] (95% CI). Initial input/backend/leaf construction was
  excluded, final owned result observation included. The gap includes different
  execution/ownership work, not just AD or session entry; production changes
  were deferred.

Follow the linked records for exact source and normalization. For broader or
newer measurements consult [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark).
Neither these numbers nor historical Apple measurements are latency promises,
execution thresholds, default-thread results or GPU budgets. GPU transfer and
synchronization require their own measurements. Performance validity is separate
from numerical correctness: INCONCLUSIVE demonstrates neither speedup nor
non-regression. Revalidate examples and measurement provenance when upgrading.
For overhead measurements explicitly configure and verify worker count; CPU
pinning alone does not set it. See [parallelism and caching](parallelism-and-caching.md).
