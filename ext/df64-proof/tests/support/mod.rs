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

/// Prepare `program` for `engine` without running it.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeState`] when the target is not
/// registered, [`tenferro_runtime::Error::Unsupported`] when no eligible
/// engine or ingress serves an input, and the typed preparation failure with
/// its [`tenferro_runtime::PrepareError`] source.
pub fn prepare_for_engine(
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
/// Returns [`tenferro_runtime::Error::RuntimeState`] when the prepared run
/// belongs to another runtime or its target does not match the engine,
/// [`tenferro_runtime::Error::TensorRuntime`] when the engine fails or an
/// output cannot be copied, and [`tenferro_runtime::Error::Extension`] for an
/// extension kernel's typed failure.
pub fn run_prepared(
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
/// Use [`run_on_engine`] instead when a test observes engine- or runtime-owned
/// caches across calls.
///
/// # Errors
///
/// Returns the same [`tenferro_runtime::Error`] variants as [`run_on_engine`]:
/// [`tenferro_runtime::Error::RuntimeState`] for an unregistered target,
/// [`tenferro_runtime::Error::Unsupported`] for an unserved operation,
/// [`tenferro_runtime::Error::TensorRuntime`] for an engine or output-copy
/// failure, and [`tenferro_runtime::Error::Extension`] for an extension
/// kernel's typed failure.
pub fn run_on_fresh_engine(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let mut engine = CpuEngine::serial();
    run_on_engine(runtime, &mut engine, program, inputs)
}
