#![allow(dead_code)]

use tenferro::error::Result;
use tenferro::traced_tensor::EinsumOptimize;
use tenferro::{
    DType, EinsumSubscripts, GraphCompiler, GraphExecutor, Tensor, TensorBackend, TracedTensor,
};

pub trait TestEinsumContext {
    fn with_compiler<R>(&mut self, f: impl FnOnce(&mut GraphCompiler) -> Result<R>) -> Result<R>;
}

impl TestEinsumContext for GraphCompiler {
    fn with_compiler<R>(&mut self, f: impl FnOnce(&mut GraphCompiler) -> Result<R>) -> Result<R> {
        f(self)
    }
}

impl<B: TensorBackend> TestEinsumContext for GraphExecutor<B> {
    fn with_compiler<R>(&mut self, f: impl FnOnce(&mut GraphCompiler) -> Result<R>) -> Result<R> {
        let mut compiler = GraphCompiler::new();
        f(&mut compiler)
    }
}

pub trait RunTraced {
    fn run_with<B: TensorBackend>(&self, executor: &mut GraphExecutor<B>) -> Result<Tensor>;

    fn run_with_inputs_auto<B: TensorBackend>(
        &self,
        executor: &mut GraphExecutor<B>,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<Tensor>;
}

impl RunTraced for TracedTensor {
    fn run_with<B: TensorBackend>(&self, executor: &mut GraphExecutor<B>) -> Result<Tensor> {
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(self)?;
        executor.run(&program)
    }

    fn run_with_inputs_auto<B: TensorBackend>(
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

pub fn run_many_traced_with<B: TensorBackend>(
    executor: &mut GraphExecutor<B>,
    outputs: &[&TracedTensor],
) -> Result<Vec<Tensor>> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(outputs)?;
    executor.run_many(&program)
}

pub fn einsum<C: TestEinsumContext>(
    ctx: &mut C,
    inputs: &[&TracedTensor],
    subscripts: &str,
) -> Result<TracedTensor> {
    ctx.with_compiler(|compiler| tenferro::traced_tensor::einsum(compiler, inputs, subscripts))
}

pub fn einsum_with<C: TestEinsumContext>(
    ctx: &mut C,
    inputs: &[&TracedTensor],
    subscripts: &str,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    ctx.with_compiler(|compiler| {
        tenferro::traced_tensor::einsum_with(compiler, inputs, subscripts, optimize)
    })
}

pub fn einsum_subscripts<C: TestEinsumContext>(
    ctx: &mut C,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
) -> Result<TracedTensor> {
    ctx.with_compiler(|compiler| {
        tenferro::traced_tensor::einsum_subscripts(compiler, inputs, subscripts)
    })
}

pub fn einsum_subscripts_with<C: TestEinsumContext>(
    ctx: &mut C,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    ctx.with_compiler(|compiler| {
        tenferro::traced_tensor::einsum_subscripts_with(compiler, inputs, subscripts, optimize)
    })
}
