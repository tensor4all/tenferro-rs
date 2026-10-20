#![allow(dead_code)]

use tenferro_runtime::error::Result;
use tenferro_runtime::{
    CompiledGraph, DType, Error, ErrorPhase, ExecutionEngine, GraphCompiler, Runtime, Tensor,
    TensorRead, TracedTensor,
};

pub fn runtime_from_cpu_backend(backend: &tenferro_cpu::CpuEngine) -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(backend).unwrap())
        .unwrap();
    builder.build().unwrap()
}

pub fn cpu_runtime() -> Runtime {
    runtime_from_cpu_backend(&tenferro_cpu::CpuEngine::serial())
}

/// Prepare and run a compiled graph on a fresh serial CPU engine.
///
/// # Errors
///
/// Propagates preparation and execution errors from the runtime.
pub fn run_on_cpu(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    run_on(
        &mut tenferro_cpu::CpuEngine::serial(),
        runtime,
        program,
        inputs,
    )
}

/// Prepare and run a compiled graph on `engine`, returning owned outputs.
///
/// # Errors
///
/// Propagates preparation and execution errors from the runtime.
pub fn run_on(
    engine: &mut dyn ExecutionEngine,
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let target = engine.target();
    let prepared = runtime.prepare(program, &reads, std::slice::from_ref(&target))?;
    let outputs = runtime.run(&mut [&mut *engine], &prepared, &reads)?;
    outputs
        .into_iter()
        .map(|output| output.into_tensor(&mut *engine).map_err(Error::from))
        .collect()
}

/// Execute a compiled graph and return its single output.
///
/// # Errors
///
/// Propagates [`Error::Validation`] for input binding or shape mismatches,
/// [`Error::RuntimeState`] when the graph returns any output count other than
/// one, and typed backend/runtime execution errors from
/// [`Runtime::run`].
pub fn run_compiled_one(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Tensor> {
    let mut outputs = run_on_cpu(runtime, program, inputs)?;
    let actual = outputs.len();
    if actual != 1 {
        return Err(Error::runtime_state(
            "support::run_compiled_one",
            ErrorPhase::Execution,
            format!("expected one runtime output, got {actual}"),
        ));
    }
    outputs.pop().ok_or_else(|| {
        Error::runtime_state(
            "support::run_compiled_one",
            ErrorPhase::Execution,
            "runtime returned no output after successful output-count validation",
        )
    })
}

pub trait RunCompiledTestExt {
    /// Execute a compiled graph and return its single output.
    ///
    /// # Errors
    ///
    /// Propagates [`Error::Validation`] for input binding or shape mismatches,
    /// [`Error::RuntimeState`] when the graph returns any output count other
    /// than one, and typed backend/runtime execution errors from
    /// [`Runtime::run`].
    fn run_compiled_one_output(
        &self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<Tensor>;
}

impl RunCompiledTestExt for Runtime {
    fn run_compiled_one_output(
        &self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<Tensor> {
        run_compiled_one(self, program, inputs)
    }
}

pub trait RunTraced {
    /// Compile and execute one traced output.
    ///
    /// # Errors
    ///
    /// Propagates [`tenferro_runtime::Error::Validation`] for invalid graph
    /// metadata, [`Error::RuntimeState`] for missing executor state, or a
    /// typed backend error from execution.
    fn run_with(&self, runtime: &Runtime) -> Result<Tensor>;

    /// Compile and execute one traced output with automatic input specs.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] for dtype/rank/shape
    /// binding mismatches, [`Error::RuntimeState`] for missing bindings or
    /// executor state, or a typed backend error from execution.
    fn run_with_inputs_auto(
        &self,
        runtime: &Runtime,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<Tensor>;
}

impl RunTraced for TracedTensor {
    fn run_with(&self, runtime: &Runtime) -> Result<Tensor> {
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(self)?;
        run_compiled_one(runtime, &program, &[])
    }

    fn run_with_inputs_auto(
        &self,
        runtime: &Runtime,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<Tensor> {
        let spec_storage: Vec<(&TracedTensor, DType, Vec<usize>)> = bindings
            .iter()
            .map(|(placeholder, tensor)| (*placeholder, tensor.dtype(), tensor.shape().to_vec()))
            .collect();
        let specs: Vec<(&TracedTensor, DType, &[usize])> = spec_storage
            .iter()
            .map(|(placeholder, dtype, shape)| (*placeholder, *dtype, shape.as_slice()))
            .collect();

        let mut compiler = GraphCompiler::new();
        let program = compiler.compile_with_input_specs(self, &specs)?;
        let input_refs: Vec<&Tensor> = bindings.iter().map(|(_, tensor)| *tensor).collect();
        run_compiled_one(runtime, &program, &input_refs)
    }
}

/// Compile and execute several traced outputs with one executor.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::Validation`] for inconsistent graph
/// metadata or duplicate bindings, [`Error::RuntimeState`] for missing
/// executor state, or a typed backend error from execution.
pub fn run_many_traced_with(runtime: &Runtime, outputs: &[&TracedTensor]) -> Result<Vec<Tensor>> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(outputs)?;
    run_on_cpu(runtime, &program, &[])
}
