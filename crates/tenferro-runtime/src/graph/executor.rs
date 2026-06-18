use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::{
    DType, RuntimeCacheControl, Tensor, TensorBackend, TensorRead, TensorValue, TypedTensor,
};

use super::cache::GraphExecutorCacheStats;
use super::program::{GraphProgram, GraphProgramInput};
use crate::error::{Error, Result};
use crate::exec::{ExecProgram, ExecSlot};
use crate::extension_runtime::{ExtensionExecutor, ExtensionRuntimeRegistryError};
use crate::traced::TracedTensor;

/// Executes compiled graph programs on a concrete tensor backend.
///
/// A graph executor owns backend execution state only: backend runtime caches,
/// extension runtime state, and reusable execution workspace. Compilation
/// state lives in [`GraphCompiler`](super::GraphCompiler).
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let y = (&x + &x).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
///
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
/// ```
pub struct GraphExecutor<B: TensorBackend + 'static> {
    backend: B,
    backend_cache: B::RuntimeCache,
    extension_executor: ExtensionExecutor<B>,
    slot_workspace: Vec<Option<ExecSlot<'static>>>,
    borrowed_slot_workspace_capacity: usize,
}

impl<B: TensorBackend + 'static> fmt::Debug for GraphExecutor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphExecutor")
            .field("backend_type", &std::any::type_name::<B>())
            .field("cache_stats", &self.cache_stats())
            .field("slot_workspace_len", &self.slot_workspace.len())
            .field(
                "borrowed_slot_workspace_capacity",
                &self.borrowed_slot_workspace_capacity,
            )
            .finish_non_exhaustive()
    }
}

impl<B: TensorBackend + 'static> GraphExecutor<B> {
    /// Create an executor with the given backend and bounded default caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// assert_eq!(executor.cache_stats().extensions.entries, 0);
    /// ```
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            backend_cache: B::RuntimeCache::default(),
            extension_executor: ExtensionExecutor::new(),
            slot_workspace: Vec::new(),
            borrowed_slot_workspace_capacity: 0,
        }
    }

    /// Borrow the backend used by this executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// let _backend = executor.backend();
    /// ```
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Return output tensors to the executor backend's reusable buffer pool.
    ///
    /// This is useful for tight benchmark or serving loops that consume an
    /// output before the next run and want backend-level output allocation
    /// behavior to match caching allocators.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x.neg()).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let out = executor.run(&program).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[-2.0]);
    /// executor.reclaim_outputs(vec![out]);
    /// ```
    pub fn reclaim_outputs(&mut self, outputs: Vec<Tensor>) {
        for tensor in outputs {
            self.backend.reclaim_buffer(tensor);
        }
    }

    /// Return compact value outputs to the backend pool when ownership is unique.
    ///
    /// Lazy owned views are intentionally ignored because their base storage may
    /// be aliased by view metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor, Tensor, TensorValue};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.reclaim_value_outputs(vec![TensorValue::from_tensor(tensor)]);
    /// ```
    pub fn reclaim_value_outputs(&mut self, outputs: Vec<TensorValue>) {
        for value in outputs {
            if let TensorValue::Tensor(tensor) = value {
                if let Ok(tensor) = Arc::try_unwrap(tensor) {
                    self.backend.reclaim_buffer(tensor);
                }
            }
        }
    }

    /// Borrow the extension runtime executor owned by this graph executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// assert_eq!(executor.extension_executor().cache_stats().entries, 0);
    /// ```
    pub fn extension_executor(&self) -> &ExtensionExecutor<B> {
        &self.extension_executor
    }

    /// Mutably borrow the extension runtime executor owned by this graph executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.extension_executor_mut().clear_caches();
    /// ```
    pub fn extension_executor_mut(&mut self) -> &mut ExtensionExecutor<B> {
        &mut self.extension_executor
    }

    /// Register one extension runtime on this executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::GraphExecutor;
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.register_extension(|_| Ok(())).unwrap();
    /// ```
    pub fn register_extension(
        &mut self,
        register: impl FnOnce(
            &mut ExtensionExecutor<B>,
        ) -> std::result::Result<(), ExtensionRuntimeRegistryError>,
    ) -> std::result::Result<(), ExtensionRuntimeRegistryError> {
        register(&mut self.extension_executor)
    }

    /// Run a one-output program using the program's default input tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x.neg()).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let out = executor.run(&program).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[-3.0]);
    /// ```
    pub fn run(&mut self, program: &GraphProgram) -> Result<Tensor> {
        let mut outputs = self.run_many(program)?;
        expect_single_output(&mut outputs)
    }

    /// Run a one-output program and preserve lazy owned output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TensorValue, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// )
    /// .unwrap();
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let value = executor.run_value(&program).unwrap();
    /// assert!(matches!(&value, TensorValue::View(_)));
    /// assert_eq!(value.shape(), &[3, 2]);
    /// assert_eq!(
    ///     value.to_tensor().unwrap().as_slice::<f64>().unwrap(),
    ///     &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    /// );
    /// ```
    pub fn run_value(&mut self, program: &GraphProgram) -> Result<TensorValue> {
        let mut outputs = self.run_many_values(program)?;
        expect_single_value(&mut outputs)
    }

    /// Run a program using the program's default input tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    /// let y = x.neg();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile_many(&[&x, &y]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let outputs = executor.run_many(&program).unwrap();
    /// assert_eq!(outputs.len(), 2);
    /// ```
    pub fn run_many(&mut self, program: &GraphProgram) -> Result<Vec<Tensor>> {
        self.run_many_with_inputs(program, &[])
    }

    /// Run a program and preserve lazy owned output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TensorValue, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile_many(&[&y]).unwrap();
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let outputs = executor.run_many_values(&program).unwrap();
    /// assert_eq!(outputs.len(), 1);
    /// assert!(matches!(&outputs[0], TensorValue::View(_)));
    /// assert_eq!(outputs[0].shape(), &[2, 2]);
    /// ```
    pub fn run_many_values(&mut self, program: &GraphProgram) -> Result<Vec<TensorValue>> {
        self.run_many_values_with_inputs(program, &[])
    }

    /// Run a one-output program with explicit runtime placeholder bindings.
    ///
    /// Explicit bindings override program defaults and are validated against
    /// the ordered input specs captured in the compiled program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    /// let y = (&x + &x).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let out = executor.run_with_inputs(&program, &[(&x, &bound)]).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    pub fn run_with_inputs(
        &mut self,
        program: &GraphProgram,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<Tensor> {
        let mut outputs = self.run_many_with_inputs(program, bindings)?;
        expect_single_output(&mut outputs)
    }

    /// Run a one-output program with explicit bindings and preserve lazy output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TensorValue, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2);
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let value = executor.run_value_with_inputs(&program, &[(&x, &bound)]).unwrap();
    /// assert!(matches!(&value, TensorValue::View(_)));
    /// assert_eq!(value.to_tensor().unwrap().as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn run_value_with_inputs(
        &mut self,
        program: &GraphProgram,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<TensorValue> {
        let mut outputs = self.run_many_values_with_inputs(program, bindings)?;
        expect_single_value(&mut outputs)
    }

    /// Run a one-output program with explicit borrowed runtime placeholder bindings.
    ///
    /// Unlike [`run_with_inputs`](Self::run_with_inputs), caller-owned input
    /// tensors are read through [`TensorRead`] and are not cloned into executor
    /// slots.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{
    ///     DType, GraphCompiler, GraphExecutor, TensorRead, TensorView, TracedTensor, TypedTensorView,
    /// };
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    /// let y = (&x + &x).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// let data = [1.0_f64, 99.0, 2.0];
    /// let view = TypedTensorView::from_slice([2], [2], 0, &data).unwrap();
    /// let read = TensorRead::from_view(TensorView::F64(view));
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let out = executor.run_with_input_reads(&program, &[(&x, read)]).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    pub fn run_with_input_reads<'a>(
        &mut self,
        program: &'a GraphProgram,
        bindings: &[(&TracedTensor, TensorRead<'a>)],
    ) -> Result<Tensor> {
        let mut outputs = self.run_many_with_input_reads(program, bindings)?;
        expect_single_output(&mut outputs)
    }

    /// Run a one-output program with borrowed bindings and preserve lazy output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{
    ///     DType, GraphCompiler, GraphExecutor, TensorRead, TensorValue, TensorView, TracedTensor,
    ///     TypedTensorView,
    /// };
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2);
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
    ///     .unwrap();
    /// let data = [1.0_f64, 2.0, 3.0, 4.0];
    /// let view = TypedTensorView::from_slice([2, 2], [1, 2], 0, &data).unwrap();
    /// let read = TensorRead::from_view(TensorView::F64(view));
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let value = executor
    ///     .run_value_with_input_reads(&program, &[(&x, read)])
    ///     .unwrap();
    /// assert!(matches!(&value, TensorValue::View(_)));
    /// assert_eq!(value.shape(), &[2, 2]);
    /// ```
    pub fn run_value_with_input_reads<'a>(
        &mut self,
        program: &'a GraphProgram,
        bindings: &[(&TracedTensor, TensorRead<'a>)],
    ) -> Result<TensorValue> {
        let mut outputs = self.run_many_values_with_input_reads(program, bindings)?;
        expect_single_value(&mut outputs)
    }

    /// Run a program with explicit runtime placeholder bindings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    /// let sum = (&x + &x).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&sum, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let outputs = executor.run_many_with_inputs(&program, &[(&x, &bound)]).unwrap();
    /// assert_eq!(outputs.len(), 1);
    /// assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    pub fn run_many_with_inputs(
        &mut self,
        program: &GraphProgram,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<Vec<Tensor>> {
        let input_tensors = resolve_inputs(program, bindings, &mut self.backend)?;
        self.eval_exec_ir(&program.exec, input_tensors)
    }

    /// Run a program with explicit bindings and preserve lazy output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TensorValue, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2);
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let outputs = executor
    ///     .run_many_values_with_inputs(&program, &[(&x, &bound)])
    ///     .unwrap();
    /// assert_eq!(outputs.len(), 1);
    /// assert!(matches!(&outputs[0], TensorValue::View(_)));
    /// ```
    pub fn run_many_values_with_inputs(
        &mut self,
        program: &GraphProgram,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<Vec<TensorValue>> {
        let input_tensors = resolve_inputs(program, bindings, &mut self.backend)?;
        self.eval_exec_ir_values(&program.exec, input_tensors)
    }

    /// Run a program with explicit borrowed runtime placeholder bindings.
    ///
    /// Bindings override program defaults and are validated against the input
    /// specs captured in the compiled program. Bound tensors are borrowed by
    /// the executor for this call instead of cloned into input slots.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{
    ///     DType, GraphCompiler, GraphExecutor, TensorRead, TensorView, TracedTensor, TypedTensorView,
    /// };
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    /// let y = (&x + &x).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// let data = [1.0_f64, 99.0, 2.0];
    /// let view = TypedTensorView::from_slice([2], [2], 0, &data).unwrap();
    /// let read = TensorRead::from_view(TensorView::F64(view));
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let outputs = executor.run_many_with_input_reads(&program, &[(&x, read)]).unwrap();
    /// assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    pub fn run_many_with_input_reads<'a>(
        &mut self,
        program: &'a GraphProgram,
        bindings: &[(&TracedTensor, TensorRead<'a>)],
    ) -> Result<Vec<Tensor>> {
        let inputs = resolve_input_reads(program, bindings, &mut self.backend)?;
        self.eval_exec_ir_slots(&program.exec, inputs)
    }

    /// Run a program with borrowed bindings and preserve lazy output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{
    ///     DType, GraphCompiler, GraphExecutor, TensorRead, TensorValue, TensorView, TracedTensor,
    ///     TypedTensorView,
    /// };
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2);
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
    ///     .unwrap();
    /// let data = [1.0_f64, 2.0, 3.0, 4.0];
    /// let view = TypedTensorView::from_slice([2, 2], [1, 2], 0, &data).unwrap();
    /// let read = TensorRead::from_view(TensorView::F64(view));
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let outputs = executor
    ///     .run_many_values_with_input_reads(&program, &[(&x, read)])
    ///     .unwrap();
    /// assert_eq!(outputs.len(), 1);
    /// assert!(matches!(&outputs[0], TensorValue::View(_)));
    /// ```
    pub fn run_many_values_with_input_reads<'a>(
        &mut self,
        program: &'a GraphProgram,
        bindings: &[(&TracedTensor, TensorRead<'a>)],
    ) -> Result<Vec<TensorValue>> {
        let inputs = resolve_input_reads(program, bindings, &mut self.backend)?;
        self.eval_exec_ir_slot_values(&program.exec, inputs)
    }

    /// Evaluate an execution program through this executor's backend state.
    ///
    /// This lower-level entry point is intended for code that already owns an
    /// execution program and concrete ordered input tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x.neg()).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let out = executor.run(&program).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[-2.0]);
    /// ```
    pub fn eval_exec_ir(
        &mut self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>> {
        validate_exec_input_count(program, inputs.len())?;
        crate::segment::eval_exec_segmented_with_cache_and_workspace(
            &mut self.backend,
            program,
            inputs,
            &mut self.slot_workspace,
            &mut self.backend_cache,
            Some(&mut self.extension_executor),
        )
    }

    /// Evaluate an execution program and preserve lazy owned output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TensorValue, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let value = executor.run_value(&program).unwrap();
    /// assert!(matches!(&value, TensorValue::View(_)));
    /// assert_eq!(value.shape(), &[2, 2]);
    /// ```
    pub fn eval_exec_ir_values(
        &mut self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>> {
        validate_exec_input_count(program, inputs.len())?;
        let inputs = inputs.into_iter().map(ExecSlot::Owned).collect();
        crate::segment::eval_exec_segmented_slot_values_with_cache_and_workspace(
            &mut self.backend,
            program,
            inputs,
            &mut self.slot_workspace,
            &mut self.backend_cache,
            Some(&mut self.extension_executor),
        )
    }

    /// Evaluate an execution program without consuming caller-owned inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x.neg()).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let out = executor.run(&program).unwrap();
    /// assert_eq!(out.shape(), &[1]);
    /// ```
    pub fn eval_exec_ir_non_consuming(
        &mut self,
        program: &ExecProgram,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let inputs = inputs
            .iter()
            .map(|tensor| ExecSlot::Read(TensorRead::from_tensor(tensor)))
            .collect();
        self.eval_exec_ir_slots(program, inputs)
    }

    /// Evaluate an execution program without consuming inputs and preserve lazy outputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TensorValue, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let value = executor.run_value(&program).unwrap();
    /// assert!(matches!(&value, TensorValue::View(_)));
    /// assert_eq!(value.to_tensor().unwrap().shape(), &[2, 2]);
    /// ```
    pub fn eval_exec_ir_non_consuming_values(
        &mut self,
        program: &ExecProgram,
        inputs: &[Tensor],
    ) -> Result<Vec<TensorValue>> {
        let inputs = inputs
            .iter()
            .map(|tensor| ExecSlot::Read(TensorRead::from_tensor(tensor)))
            .collect();
        self.eval_exec_ir_slot_values(program, inputs)
    }

    fn eval_exec_ir_slots<'a>(
        &mut self,
        program: &ExecProgram,
        inputs: Vec<ExecSlot<'a>>,
    ) -> Result<Vec<Tensor>> {
        validate_exec_input_count(program, inputs.len())?;
        let mut slot_workspace = Vec::with_capacity(self.borrowed_slot_workspace_capacity);
        let result = crate::segment::eval_exec_segmented_slots_with_cache_and_workspace(
            &mut self.backend,
            program,
            inputs,
            &mut slot_workspace,
            &mut self.backend_cache,
            Some(&mut self.extension_executor),
        );
        self.borrowed_slot_workspace_capacity = slot_workspace.capacity();
        result
    }

    fn eval_exec_ir_slot_values<'a>(
        &mut self,
        program: &ExecProgram,
        inputs: Vec<ExecSlot<'a>>,
    ) -> Result<Vec<TensorValue>> {
        validate_exec_input_count(program, inputs.len())?;
        let mut slot_workspace = Vec::with_capacity(self.borrowed_slot_workspace_capacity);
        let result = crate::segment::eval_exec_segmented_slot_values_with_cache_and_workspace(
            &mut self.backend,
            program,
            inputs,
            &mut slot_workspace,
            &mut self.backend_cache,
            Some(&mut self.extension_executor),
        );
        self.borrowed_slot_workspace_capacity = slot_workspace.capacity();
        result
    }

    /// Clear backend-specific runtime analysis cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.clear_backend_cache();
    /// assert_eq!(executor.cache_stats().backend.entries, 0);
    /// ```
    pub fn clear_backend_cache(&mut self) {
        self.backend_cache.clear();
    }

    /// Clear generic extension runtime cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.clear_extension_caches();
    /// assert_eq!(executor.cache_stats().extensions.entries, 0);
    /// ```
    pub fn clear_extension_caches(&mut self) {
        self.extension_executor.clear_caches();
    }

    /// Clear every executor-owned runtime cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.clear_caches();
    /// assert_eq!(executor.cache_stats().backend.entries, 0);
    /// ```
    pub fn clear_caches(&mut self) {
        self.clear_extension_caches();
        self.clear_backend_cache();
    }

    /// Return executor runtime cache-entry and retained-byte stats.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// let stats = executor.cache_stats();
    /// assert_eq!(stats.extensions.entries, 0);
    /// ```
    pub fn cache_stats(&self) -> GraphExecutorCacheStats {
        GraphExecutorCacheStats {
            extensions: self.extension_executor.cache_stats(),
            backend: self.backend_cache.stats(),
        }
    }
}

impl<B: TensorBackend + 'static> Default for GraphExecutor<B>
where
    B: Default,
{
    fn default() -> Self {
        Self::new(B::default())
    }
}

fn validate_exec_input_count(program: &ExecProgram, actual: usize) -> Result<()> {
    let expected = program.input_slots.len();
    if actual != expected {
        return Err(Error::Internal(format!(
            "expected {expected} inputs for execution program, got {actual}"
        )));
    }
    Ok(())
}

fn expect_single_output(outputs: &mut Vec<Tensor>) -> Result<Tensor> {
    if outputs.len() != 1 {
        return Err(Error::Internal(format!(
            "expected 1 output, got {}",
            outputs.len()
        )));
    }
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("missing graph output".to_string()))
}

fn expect_single_value(outputs: &mut Vec<TensorValue>) -> Result<TensorValue> {
    if outputs.len() != 1 {
        return Err(Error::Internal(format!(
            "expected 1 output, got {}",
            outputs.len()
        )));
    }
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("missing graph output".to_string()))
}

fn resolve_inputs(
    program: &GraphProgram,
    bindings: &[(&TracedTensor, &Tensor)],
    backend: &mut impl TensorBackend,
) -> Result<Vec<Tensor>> {
    let program_keys: HashSet<_> = program
        .inputs
        .iter()
        .map(|input| input.key.clone())
        .collect();
    let tangent_root_specs = tangent_root_specs(&program.inputs);
    let default_map: HashMap<_, _> = program
        .inputs
        .iter()
        .filter_map(|input| {
            input
                .default_tensor
                .as_ref()
                .map(|tensor| (input.key.clone(), tensor.as_ref()))
        })
        .collect();
    let mut binding_map = HashMap::new();
    for (index, (placeholder, tensor)) in bindings.iter().enumerate() {
        if placeholder.data.is_some() {
            return Err(Error::UnexpectedBinding {
                binding_index: index,
            });
        }
        let key = placeholder.input_key().ok_or(Error::UnexpectedBinding {
            binding_index: index,
        })?;
        validate_binding_placeholder(index, placeholder, tensor)?;
        let is_program_input = program_keys.contains(&key);
        if !is_program_input && !tangent_root_specs.contains_key(&key) {
            return Err(Error::UnexpectedBinding {
                binding_index: index,
            });
        }
        if binding_map.insert(key.clone(), *tensor).is_some() {
            return Err(Error::DuplicateBinding {
                input_key: format!("{:?}", key),
            });
        }
    }

    program
        .inputs
        .iter()
        .map(|input| resolve_input(input, &binding_map, &default_map, backend))
        .collect()
}

fn resolve_input_reads<'a>(
    program: &'a GraphProgram,
    bindings: &[(&TracedTensor, TensorRead<'a>)],
    backend: &mut impl TensorBackend,
) -> Result<Vec<ExecSlot<'a>>> {
    let program_keys: HashSet<_> = program
        .inputs
        .iter()
        .map(|input| input.key.clone())
        .collect();
    let tangent_root_specs = tangent_root_specs(&program.inputs);
    let default_map: HashMap<_, _> = program
        .inputs
        .iter()
        .filter_map(|input| {
            input
                .default_tensor
                .as_ref()
                .map(|tensor| (input.key.clone(), tensor.as_ref()))
        })
        .collect();
    let mut binding_map = HashMap::new();
    for (index, (placeholder, read)) in bindings.iter().enumerate() {
        if placeholder.data.is_some() {
            return Err(Error::UnexpectedBinding {
                binding_index: index,
            });
        }
        let key = placeholder.input_key().ok_or(Error::UnexpectedBinding {
            binding_index: index,
        })?;
        validate_binding_placeholder_read(index, placeholder, read)?;
        let is_program_input = program_keys.contains(&key);
        if !is_program_input && !tangent_root_specs.contains_key(&key) {
            return Err(Error::UnexpectedBinding {
                binding_index: index,
            });
        }
        if binding_map.insert(key.clone(), read.clone()).is_some() {
            return Err(Error::DuplicateBinding {
                input_key: format!("{:?}", key),
            });
        }
    }

    program
        .inputs
        .iter()
        .map(|input| resolve_input_read(input, &binding_map, &default_map, backend))
        .collect()
}

fn tangent_root_specs(inputs: &[GraphProgramInput]) -> HashMap<TensorInputKey, &GraphProgramInput> {
    let mut specs = HashMap::new();
    for input in inputs {
        if !matches!(input.key, TensorInputKey::User { .. }) {
            specs
                .entry(tangent_primal_root(&input.key).clone())
                .or_insert(input);
        }
    }
    specs
}

fn resolve_input(
    input: &GraphProgramInput,
    bindings: &HashMap<TensorInputKey, &Tensor>,
    defaults: &HashMap<TensorInputKey, &Tensor>,
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    let tensor = if let Some(bound) = bindings.get(&input.key) {
        (*bound).clone()
    } else if let Some(default) = &input.default_tensor {
        resolve_default_tensor(default.as_ref(), backend)?
    } else if let Some(zero) = deferred_zero_for_tangent_key(&input.key, bindings, defaults) {
        zero
    } else {
        return Err(Error::UnboundPlaceholder {
            input_key: format!("{:?}", input.key),
        });
    };
    validate_input_tensor(input, &tensor)?;
    Ok(tensor)
}

fn resolve_input_read<'a>(
    input: &GraphProgramInput,
    bindings: &HashMap<TensorInputKey, TensorRead<'a>>,
    defaults: &HashMap<TensorInputKey, &'a Tensor>,
    backend: &mut impl TensorBackend,
) -> Result<ExecSlot<'a>> {
    let slot = if let Some(bound) = bindings.get(&input.key) {
        ExecSlot::Read(bound.clone())
    } else if let Some(default) = defaults.get(&input.key) {
        if should_upload_default_tensor(default) {
            ExecSlot::Owned(backend.upload_host_tensor(default)?)
        } else {
            ExecSlot::Read(TensorRead::from_tensor(default))
        }
    } else if let Some(zero) = deferred_zero_for_tangent_key_read(&input.key, bindings, defaults) {
        ExecSlot::Owned(zero)
    } else {
        return Err(Error::UnboundPlaceholder {
            input_key: format!("{:?}", input.key),
        });
    };
    validate_input_slot(input, &slot)?;
    Ok(slot)
}

fn resolve_default_tensor(default: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
    if should_upload_default_tensor(default) {
        Ok(backend.upload_host_tensor(default)?)
    } else {
        Ok(default.clone())
    }
}

fn should_upload_default_tensor(default: &Tensor) -> bool {
    default.shape().is_empty() && tensor_has_host_buffer(default)
}

fn tensor_has_host_buffer(tensor: &Tensor) -> bool {
    !tensor.is_backend_buffer()
}

fn validate_binding_placeholder(
    index: usize,
    placeholder: &TracedTensor,
    tensor: &Tensor,
) -> Result<()> {
    if placeholder.data.is_some() {
        return Err(Error::UnexpectedBinding {
            binding_index: index,
        });
    }
    if placeholder.dtype != tensor.dtype() {
        return Err(Error::PlaceholderDtypeMismatch {
            expected: placeholder.dtype,
            actual: tensor.dtype(),
        });
    }
    match placeholder.try_concrete_shape() {
        Some(expected_shape) => {
            if expected_shape.as_slice() != tensor.shape() {
                return Err(Error::PlaceholderShapeMismatch {
                    expected: expected_shape,
                    actual: tensor.shape().to_vec(),
                });
            }
        }
        None => {
            if placeholder.rank != tensor.shape().len() {
                return Err(Error::PlaceholderRankMismatch {
                    expected: placeholder.rank,
                    actual: tensor.shape().len(),
                });
            }
        }
    }
    Ok(())
}

fn validate_binding_placeholder_read(
    index: usize,
    placeholder: &TracedTensor,
    read: &TensorRead<'_>,
) -> Result<()> {
    if placeholder.data.is_some() {
        return Err(Error::UnexpectedBinding {
            binding_index: index,
        });
    }
    if placeholder.dtype != read.dtype() {
        return Err(Error::PlaceholderDtypeMismatch {
            expected: placeholder.dtype,
            actual: read.dtype(),
        });
    }
    match placeholder.try_concrete_shape() {
        Some(expected_shape) => {
            if expected_shape.as_slice() != read.shape() {
                return Err(Error::PlaceholderShapeMismatch {
                    expected: expected_shape,
                    actual: read.shape().to_vec(),
                });
            }
        }
        None => {
            if placeholder.rank != read.shape().len() {
                return Err(Error::PlaceholderRankMismatch {
                    expected: placeholder.rank,
                    actual: read.shape().len(),
                });
            }
        }
    }
    Ok(())
}

fn validate_input_tensor(input: &GraphProgramInput, tensor: &Tensor) -> Result<()> {
    if input.dtype != tensor.dtype() {
        return Err(Error::PlaceholderDtypeMismatch {
            expected: input.dtype,
            actual: tensor.dtype(),
        });
    }
    if input.shape.as_slice() != tensor.shape() {
        return Err(Error::PlaceholderShapeMismatch {
            expected: input.shape.clone(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

fn validate_input_slot(input: &GraphProgramInput, slot: &ExecSlot<'_>) -> Result<()> {
    if input.dtype != slot.dtype() {
        return Err(Error::PlaceholderDtypeMismatch {
            expected: input.dtype,
            actual: slot.dtype(),
        });
    }
    if input.shape.as_slice() != slot.shape() {
        return Err(Error::PlaceholderShapeMismatch {
            expected: input.shape.clone(),
            actual: slot.shape().to_vec(),
        });
    }
    Ok(())
}

fn deferred_zero_for_tangent_key(
    key: &TensorInputKey,
    bindings: &HashMap<TensorInputKey, &Tensor>,
    defaults: &HashMap<TensorInputKey, &Tensor>,
) -> Option<Tensor> {
    if !key.is_tangent() {
        return None;
    }
    let root = tangent_primal_root(key);
    let primal = bindings.get(root).or_else(|| defaults.get(root))?;
    Some(zeros_tensor(primal.dtype(), primal.shape().to_vec()))
}

fn deferred_zero_for_tangent_key_read<'a>(
    key: &TensorInputKey,
    bindings: &HashMap<TensorInputKey, TensorRead<'a>>,
    defaults: &HashMap<TensorInputKey, &'a Tensor>,
) -> Option<Tensor> {
    if !key.is_tangent() {
        return None;
    }
    let root = tangent_primal_root(key);
    if let Some(primal) = bindings.get(root) {
        return Some(zeros_tensor(primal.dtype(), primal.shape().to_vec()));
    }
    let primal = defaults.get(root)?;
    Some(zeros_tensor(primal.dtype(), primal.shape().to_vec()))
}

fn tangent_primal_root(key: &TensorInputKey) -> &TensorInputKey {
    key.primal_root()
}

fn zeros_tensor(dtype: DType, shape: Vec<usize>) -> Tensor {
    // Invariant: executor deferred zeros are derived from already-validated tensor shapes.
    match dtype {
        DType::F32 => {
            Tensor::F32(TypedTensor::zeros(shape).expect("validated default tensor shape"))
        }
        DType::F64 => {
            Tensor::F64(TypedTensor::zeros(shape).expect("validated default tensor shape"))
        }
        DType::I32 => {
            Tensor::I32(TypedTensor::zeros(shape).expect("validated default tensor shape"))
        }
        DType::I64 => {
            Tensor::I64(TypedTensor::zeros(shape).expect("validated default tensor shape"))
        }
        DType::Bool => {
            let len = shape.iter().product();
            Tensor::Bool(
                TypedTensor::from_vec_col_major(shape, vec![false; len])
                    .expect("validated bool default tensor shape"),
            )
        }
        DType::C32 => {
            Tensor::C32(TypedTensor::zeros(shape).expect("validated default tensor shape"))
        }
        DType::C64 => {
            Tensor::C64(TypedTensor::zeros(shape).expect("validated default tensor shape"))
        }
    }
}

#[cfg(test)]
mod tests;
