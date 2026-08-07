# Performance idioms

## Reuse owners

Construct one backend for a related sequence of operations. `CpuBackend` owns
threading, placement, pools, and runtime caches; repeatedly constructing it
throws away those resources. For traced work, likewise reuse the compiler and
runtime across executions when their shape and registration contracts match.

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
