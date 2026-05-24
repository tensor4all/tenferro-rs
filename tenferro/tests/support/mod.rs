#![allow(dead_code)]

use tenferro::error::Result;
use tenferro::{DType, GraphCompiler, GraphExecutor, Tensor, TensorBackend, TracedTensor};

pub trait RunTraced {
    fn run_with<B: TensorBackend + 'static>(
        &self,
        executor: &mut GraphExecutor<B>,
    ) -> Result<Tensor>;

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
        executor.run_with_inputs(&program, bindings)
    }
}

pub fn run_many_traced_with<B: TensorBackend + 'static>(
    executor: &mut GraphExecutor<B>,
    outputs: &[&TracedTensor],
) -> Result<Vec<Tensor>> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(outputs)?;
    executor.run_many(&program)
}
