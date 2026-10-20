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
/// Returns [`tenferro_runtime::Error::RuntimeState`] when the target is not
/// registered or the prepared run belongs to another runtime,
/// [`tenferro_runtime::Error::Unsupported`] when no installed module or
/// provider serves the program, [`tenferro_runtime::Error::TensorRuntime`]
/// when the engine fails or an output cannot be copied, and
/// [`tenferro_runtime::Error::Extension`] for an extension kernel's typed
/// failure.
pub fn run_on_engine(
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
