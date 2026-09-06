# Performance idioms

## Reuse owners

Construct one backend for a related sequence of operations. `CpuBackend` owns
threading, placement, pools, and runtime caches; repeatedly constructing it
throws away those resources. For traced work, likewise reuse the compiler and
runtime across executions when their shape and registration contracts match.

## Ordinary versus prepared einsum

Use ordinary `TensorEinsumExt::einsum` by default. Reuse a `ConcreteEinsumPlan`
when equation, operand order, input count, dtypes and shapes match; values may
change. The plan retains metadata/tree, not inputs or a session. Execution still
validates and allocates; placement and backend capabilities still apply.
`EinsumSubscripts` is a programmatic representation, not a cache. Strings are
fine for one-time preparation. Preserve explicit contraction order rather than
flattening parentheses into labels.

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#ordinary-and-prepared-einsum -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::{BackendSessionHost, Tensor};

let lhs = Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let rhs = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 1.0, 2.0])?;
let mut backend = CpuBackend::new();
// Ordinary execution: no explicit preparation needed.
let ordinary = backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum("ij,jk->ik", session)
})?;
assert_eq!(ordinary.as_slice::<f64>()?, &[2.0, 4.0, 7.0, 10.0]);

// Strings are fine for one-time preparation. The plan does not retain inputs.
let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
for (data, expected) in [
    (vec![1.0_f64, 2.0, 3.0, 4.0], [2.0, 4.0, 7.0, 10.0]),
    (vec![2.0_f64, 4.0, 6.0, 8.0], [4.0, 8.0, 14.0, 20.0]),
] {
    let next_lhs = Tensor::from_vec_col_major([2, 2], data)?;
    let result = backend.with_backend_session(|session| {
        plan.execute([&next_lhs, &rhs], session)
    })?;
    assert_eq!(result.as_slice::<f64>()?, &expected);
}

// Integer labels describe the equation; this ordinary call still plans.
let equation = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let structured = backend.with_backend_session(|session| {
    [&lhs, &rhs].einsum_subscripts(&equation, session)
})?;
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

## Compile once, run many

A traced graph is a reusable program. Build its input metadata, compile once,
register the backend and extensions once, and reuse the compiled program,
backend, and runtime while passing new concrete inputs:

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#compile-once-run-many -->
```rust
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, Runtime, Tensor, TraceContext};

let mut trace = TraceContext::new();
let input = trace.input(ProgramInputSpec::new(tenferro_runtime::DType::F64, [2.into()]))?;
let output = trace.einsum(&[input], "i->i")?;
let graph = trace.finish(&[output])?;
// Compile the shape-specialized program once.
let program = GraphCompiler::new().compile_traced_graph(&graph)?;

// Reuse one backend, extension registration, and runtime for repeated inputs.
let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
    runtime_engine_id()?,
)?)?;
let runtime = builder.build()?;
for (input, expected) in [
    (vec![1.0_f64, 2.0], vec![1.0, 2.0]),
    (vec![3.0_f64, 4.0], vec![3.0, 4.0]),
] {
    let value = Tensor::from_vec_col_major(vec![2], input)?;
    let mut outputs = runtime.run_compiled(&program, &[&value])?;
    assert_eq!(outputs.remove(0).as_slice::<f64>()?, &expected);
}
```
<!-- end-snippet-source -->

The compiled program is shape-specialized. Recompile when the input shape or
other compile-time metadata changes.

## Threading

Use `CpuBackend::with_threads(n)` or the documented backend configuration
rather than creating an independent Rayon pool inside an operation.
`cpu-faer` follows tenferro's CPU context; BLAS/LAPACK provider threads are
configured with variables such as `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, or
`VECLIB_MAXIMUM_THREADS`. Avoid outer parallel loops that oversubscribe the
provider.

For long-running processes, use the documented runtime/backend cache stats and
clear operations. Do not defeat cache ownership by constructing a fresh backend
for every iteration.
