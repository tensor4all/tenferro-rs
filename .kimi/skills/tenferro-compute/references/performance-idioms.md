# Performance idioms

## Reuse owners

Construct one engine for a related sequence of operations. A `CpuEngine` owns
the plan cache, the buffer pool and the extension caches, and it computes on the
pool you built; repeatedly constructing it throws away those resources. For
traced work, likewise reuse the compiler, runtime and prepared program across
executions when their shape and registration contracts match.

## Ordinary versus prepared einsum

Use ordinary `TensorEinsumExt::einsum` by default. Reuse a `ConcreteEinsumPlan`
when equation, operand order, input count, dtypes and shapes match; values may
change. The plan retains metadata/tree, not inputs or the engine. Execution
still validates and allocates; engine capabilities still apply.
`EinsumSubscripts` is a programmatic representation, not a cache. Strings are
fine for one-time preparation. Preserve explicit contraction order rather than
flattening parentheses into labels.

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

See [Ordinary calls and prepared execution](../../../../docs/guides/ordinary-and-prepared-execution.md)
for typed/read/output variants and historical measurements with exact provenance.
Do not interpret ten-op chain times as one-call costs or CPU measurements as GPU
budgets. `1 ms = 1000 us`; measure the complete caller-visible workflow, including
final result observation. For overhead diagnosis explicitly configure and verify
one worker separately from CPU affinity; do not change production defaults based
on tiny-work measurements.

## Amortize session entry

Reuse a backend **and** group related operations inside one explicit session.
Pool reuse alone does not remove each external worker handoff. An M5 Max
microbenchmark measured about 26 us for an external install into 18 Rayon
workers, while the same-pool path was negligible; these are machine-specific
measurements, not a per-operation constant. For unavoidable tiny single calls,
measure the one-thread backend, which executes inline. Do not reopen a backend
inside a session or substitute an ambient global pool.

Read [CPU Session Entry and Rayon Dispatch Cost](https://tensor4all.org/tenferro-rs/guides/session-entry-cost.html)
for the conditions, historical session comparison, reproducible probes, batching,
and direct-provider limitations. Direct faer use does not remove Rayon's
handoff if it still enters a pool.

## Compile once, run many

A traced graph is a reusable program. Build its input metadata, compile once,
register the backend and extensions once, and reuse the compiled program,
backend, and runtime while passing new concrete inputs:

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#compile-once-run-many -->
```rust
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, Tensor, TraceContext};

let mut trace = TraceContext::new();
let input = trace.input(ProgramInputSpec::new(tenferro_runtime::DType::F64, [2.into()]))?;
let output = trace.einsum(&[input], "i->i")?;
let graph = trace.finish(&[output])?;
// Compile the shape-specialized program once.
let program = GraphCompiler::new().compile_traced_graph(&graph)?;

// Reuse one engine, extension registration, runtime and prepared program for
// repeated inputs: `prepare` runs once, `run` replays it.
let mut engine = CpuEngine::serial();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&engine)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
let target = engine.target();
let first = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
let prepared = runtime.prepare(
    &program,
    &[tenferro_runtime::TensorRead::from_tensor(&first)],
    &[target],
)?;
for (input, expected) in [
    (vec![1.0_f64, 2.0], vec![1.0, 2.0]),
    (vec![3.0_f64, 4.0], vec![3.0, 4.0]),
] {
    let value = Tensor::from_vec_col_major(vec![2], input)?;
    let read = tenferro_runtime::TensorRead::from_tensor(&value);
    let outputs = runtime.run(&mut [&mut engine], &prepared, &[read])?;
    let result = outputs.into_iter().next().expect("one output");
    assert_eq!(result.into_tensor(&mut engine)?.as_slice::<f64>()?, &expected);
}
```
<!-- end-snippet-source -->

The compiled program is shape-specialized. Recompile when the input shape or
other compile-time metadata changes.

## Threading

Build the pool yourself with `tenferro_cpu::Pool::builder().threads(n)`
and hand it to `CpuEngine::new`, or use `CpuEngine::serial()` for no pool at all.
tenferro never builds threads and never enters an ambient pool. `native`
follows the engine's pool; BLAS/LAPACK provider threads are configured with
variables such as `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, or
`VECLIB_MAXIMUM_THREADS`. Avoid outer parallel loops that oversubscribe the
provider.

For long-running processes, use the documented engine/runtime cache stats and
clear operations. Do not defeat cache ownership by constructing a fresh engine
for every iteration.
