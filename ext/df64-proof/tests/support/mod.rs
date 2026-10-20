//! Shared execution helpers for the Df64 integration tests.
//!
//! The runtime no longer runs a compiled graph itself: preparation names the
//! caller's engine target and execution takes the engine. These helpers keep the
//! test bodies focused on the contribution by owning that boilerplate.

#![allow(dead_code)]

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{CompiledGraph, ExecutionEngine, Prepared, Runtime};
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

/// Prepare `program` for `engine` without running it.
///
/// # Errors
///
/// Propagates the runtime's preparation errors.
pub fn prepare_on(
    runtime: &Runtime,
    engine: &CpuEngine,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Prepared> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let target = engine.target();
    runtime.prepare(program, &reads, std::slice::from_ref(&target))
}

/// Run an already prepared program on `engine`.
///
/// # Errors
///
/// Propagates execution and output-copying errors.
pub fn run_prepared_on(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    prepared: &Prepared,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let outputs = runtime.run(&mut [engine], prepared, &reads)?;
    outputs
        .into_iter()
        .map(|output| {
            output
                .into_tensor(engine)
                .map_err(tenferro_runtime::Error::from)
        })
        .collect()
}

/// Run `program` on a fresh serial CPU engine.
///
/// Use [`run_compiled_on`] instead when a test observes engine- or runtime-owned
/// caches across calls.
///
/// # Errors
///
/// Propagates preparation, execution, and output-copying errors.
pub fn run_compiled(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let mut engine = CpuEngine::serial();
    run_compiled_on(runtime, &mut engine, program, inputs)
}
