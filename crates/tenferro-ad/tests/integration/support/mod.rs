#![allow(dead_code)]

use tenferro_runtime::error::Result;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TensorBackend, TracedTensor};

pub trait RunTraced {
    /// Compile and execute one traced output.
    ///
    /// # Errors
    ///
    /// Propagates [`tenferro_runtime::Error::Validation`] for invalid graph
    /// metadata, [`Error::RuntimeState`] for missing executor state, or a
    /// typed backend error from execution.
    fn run_with<B: TensorBackend + 'static>(
        &self,
        executor: &mut GraphExecutor<B>,
    ) -> Result<Tensor>;

    /// Compile and execute one traced output with automatic input specs.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] for dtype/rank/shape
    /// binding mismatches, [`Error::RuntimeState`] for missing bindings or
    /// executor state, or a typed backend error from execution.
    fn run_with_inputs_auto<B: TensorBackend + 'static>(
        &self,
        executor: &mut GraphExecutor<B>,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<Tensor>;
}

impl RunTraced for TracedTensor {
    fn run_with<B: TensorBackend + 'static>(
        &self,
        executor: &mut GraphExecutor<B>,
    ) -> Result<Tensor> {
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(self)?;
        executor.run(&program)
    }

    fn run_with_inputs_auto<B: TensorBackend + 'static>(
        &self,
        executor: &mut GraphExecutor<B>,
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
        executor.run_with_bindings(&program, bindings)
    }
}

/// Compile and execute several traced outputs with one executor.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::Validation`] for inconsistent graph
/// metadata or duplicate bindings, [`Error::RuntimeState`] for missing
/// executor state, or a typed backend error from execution.
pub fn run_many_traced_with<B: TensorBackend + 'static>(
    executor: &mut GraphExecutor<B>,
    outputs: &[&TracedTensor],
) -> Result<Vec<Tensor>> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(outputs)?;
    executor.run_many(&program)
}
