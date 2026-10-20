//! Shared execution helper for the application integration tests.
//!
//! A runtime prepares a program for a caller-named engine target and the caller
//! passes the engine to execution. Tests that do not observe caches use this
//! helper instead of spelling the prepare/run/output conversion each time.

#![allow(dead_code)]

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{CompiledGraph, ExecutionEngine, Runtime};
use tenferro_tensor::{Tensor, TensorRead};

/// Run `program` on `engine`, which must belong to `runtime`.
///
/// # Errors
///
/// Propagates preparation, execution, and output-copying errors.
pub fn run_compiled_on(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let target = engine.target();
    let prepared = runtime.prepare(program, &reads, std::slice::from_ref(&target))?;
    let outputs = runtime.run(&mut [engine], &prepared, &reads)?;
    outputs
        .into_iter()
        .map(|output| {
            output
                .into_tensor(engine)
                .map_err(tenferro_runtime::Error::from)
        })
        .collect()
}
