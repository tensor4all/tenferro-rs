use std::fmt;
use std::sync::Arc;

use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::{DType, RuntimeCacheControl, Tensor, TensorBackend, TensorRead, TensorValue};

use super::cache::GraphExecutorCacheStats;
use super::program::CompiledGraph;
use crate::error::{Error, Result};
use crate::exec::{ExecProgram, ExecSlot};
use crate::extension_runtime::{ExtensionExecutor, ExtensionRuntimeRegistryError};

/// Executes compiled graph programs on a concrete tensor backend.
///
/// A graph executor owns backend execution state only: backend runtime caches,
/// extension runtime state, and reusable execution workspace. Compilation
/// state lives in [`GraphCompiler`](super::GraphCompiler). Retained symbolic
/// shape guards are checked against ordered input shapes before any segmented,
/// backend, host, or extension dispatch. Failures are returned as typed
/// [`Error::ShapeConstraintViolation`] or [`Error::ShapeConstraintEvaluation`]
/// values.
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

    /// Materialize an executor value as a compact tensor on its current placement.
    ///
    /// Compact owned values are cloned without backend dispatch. Lazy views are
    /// canonicalized by this executor's active backend session, preserving its
    /// allocation, threading, and device-placement policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::GraphExecutor;
    /// use tenferro_tensor::{Tensor, TensorValue};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let value = TensorValue::from_tensor(tensor).transpose_view([1, 0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let compact = executor.materialize_value(&value).unwrap();
    /// assert_eq!(compact.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] containing a typed
    /// `BackendSource` when backend canonicalization fails, or
    /// `RuntimeStateSource` when the value's placement/storage state cannot be
    /// materialized.
    pub fn materialize_value(&mut self, value: &TensorValue) -> Result<Tensor> {
        if let Some(tensor) = value.as_tensor_arc() {
            return Ok(tensor.as_ref().clone());
        }
        self.backend
            .with_backend_session(|exec| exec.to_contiguous_read(value.tensor_read()))
            .map_err(Error::from)
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
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
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
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionRuntimeRegistryError::MalformedFamilyId`] when a
    /// registered runtime advertises an invalid family identifier.
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
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let out = executor.run(&program).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[-3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnboundPlaceholder`] for a missing default input,
    /// [`Error::TensorRuntime`] for typed backend/validation failures, or
    /// [`Error::Internal`] when the program produces zero or multiple outputs.
    pub fn run(&mut self, program: &CompiledGraph) -> Result<Tensor> {
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
    ///     executor.materialize_value(&value).unwrap().as_slice::<f64>().unwrap(),
    ///     &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnboundPlaceholder`] for a missing default input,
    /// [`Error::TensorRuntime`] for typed backend/validation failures, or
    /// [`Error::Internal`] when the program produces zero or multiple outputs.
    pub fn run_value(&mut self, program: &CompiledGraph) -> Result<TensorValue> {
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
    /// let y = x.neg().unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile_many(&[&x, &y]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let outputs = executor.run_many(&program).unwrap();
    /// assert_eq!(outputs.len(), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnboundPlaceholder`] when a program input has no
    /// default tensor, [`Error::ShapeConstraintViolation`] or
    /// [`Error::ShapeConstraintEvaluation`] for failed symbolic guards, and
    /// [`Error::TensorRuntime`] for typed backend execution failures.
    pub fn run_many(&mut self, program: &CompiledGraph) -> Result<Vec<Tensor>> {
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
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnboundPlaceholder`] when a program input has no
    /// default tensor, [`Error::ShapeConstraintViolation`] or
    /// [`Error::ShapeConstraintEvaluation`] for failed symbolic guards, and
    /// [`Error::TensorRuntime`] for typed backend execution failures.
    pub fn run_many_values(&mut self, program: &CompiledGraph) -> Result<Vec<TensorValue>> {
        self.run_many_values_with_inputs(program, &[])
    }

    /// Run a one-output program with ordered runtime inputs.
    ///
    /// A non-empty slice supplies every semantic input in order. An empty
    /// slice uses the graph's frozen default bindings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    /// let y = (&x + &x).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let out = executor.run_with_inputs(&program, &[&bound]).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphInputCountMismatch`] for a non-empty slice with
    /// the wrong number of inputs,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::UnboundPlaceholder`] for a missing input, and
    /// [`Error::ShapeConstraintViolation`],
    /// [`Error::ShapeConstraintEvaluation`], or [`Error::TensorRuntime`] when
    /// execution discovers a symbolic-guard or typed backend failure.
    pub fn run_with_inputs(
        &mut self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<Tensor> {
        let mut outputs = self.run_many_with_inputs(program, inputs)?;
        expect_single_output(&mut outputs)
    }

    /// Run a one-output program with ordered inputs and preserve lazy output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TensorValue, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let value = executor.run_value_with_inputs(&program, &[&bound]).unwrap();
    /// assert!(matches!(&value, TensorValue::View(_)));
    /// assert_eq!(executor.materialize_value(&value).unwrap().as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphInputCountMismatch`] for a non-empty slice with
    /// the wrong number of inputs,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::UnboundPlaceholder`] for a missing input, and
    /// [`Error::ShapeConstraintViolation`],
    /// [`Error::ShapeConstraintEvaluation`], or [`Error::TensorRuntime`] when
    /// execution discovers a symbolic-guard or typed backend failure.
    pub fn run_value_with_inputs(
        &mut self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<TensorValue> {
        let mut outputs = self.run_many_values_with_inputs(program, inputs)?;
        expect_single_value(&mut outputs)
    }

    /// Run a one-output program with ordered borrowed runtime inputs.
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
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
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
    /// let out = executor.run_with_input_reads(&program, &[read]).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphInputCountMismatch`] for a non-empty slice with
    /// the wrong number of inputs,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::UnboundPlaceholder`] for a missing input, and
    /// [`Error::ShapeConstraintViolation`],
    /// [`Error::ShapeConstraintEvaluation`], or [`Error::TensorRuntime`] when
    /// execution discovers a symbolic-guard or typed backend failure.
    pub fn run_with_input_reads<'a>(
        &mut self,
        program: &'a CompiledGraph,
        inputs: &[TensorRead<'a>],
    ) -> Result<Tensor> {
        let mut outputs = self.run_many_with_input_reads(program, inputs)?;
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
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
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
    ///     .run_value_with_input_reads(&program, &[read])
    ///     .unwrap();
    /// assert!(matches!(&value, TensorValue::View(_)));
    /// assert_eq!(value.shape(), &[2, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphInputCountMismatch`] for a non-empty slice with
    /// the wrong number of inputs,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::UnboundPlaceholder`] for a missing input, and
    /// [`Error::ShapeConstraintViolation`],
    /// [`Error::ShapeConstraintEvaluation`], or [`Error::TensorRuntime`] when
    /// execution discovers a symbolic-guard or typed backend failure.
    pub fn run_value_with_input_reads<'a>(
        &mut self,
        program: &'a CompiledGraph,
        inputs: &[TensorRead<'a>],
    ) -> Result<TensorValue> {
        let mut outputs = self.run_many_values_with_input_reads(program, inputs)?;
        expect_single_value(&mut outputs)
    }

    /// Run a program with ordered runtime inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    /// let sum = (&x + &x).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&sum, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let outputs = executor.run_many_with_inputs(&program, &[&bound]).unwrap();
    /// assert_eq!(outputs.len(), 1);
    /// assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphInputCountMismatch`] for a non-empty slice with
    /// the wrong number of inputs,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::UnboundPlaceholder`] for a missing input, and
    /// [`Error::ShapeConstraintViolation`],
    /// [`Error::ShapeConstraintEvaluation`], or [`Error::TensorRuntime`] when
    /// execution discovers a symbolic-guard or typed backend failure.
    pub fn run_many_with_inputs(
        &mut self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        let input_tensors = resolve_ordered_inputs(program, inputs, &mut self.backend)?;
        self.eval_exec_ir(&program.staging, input_tensors)
    }

    /// Run a program with ordered inputs and preserve lazy output views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TensorValue, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    /// let y = x.transpose(&[1, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    ///
    /// let outputs = executor
    ///     .run_many_values_with_inputs(&program, &[&bound])
    ///     .unwrap();
    /// assert_eq!(outputs.len(), 1);
    /// assert!(matches!(&outputs[0], TensorValue::View(_)));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphInputCountMismatch`] for a non-empty slice with
    /// the wrong number of inputs,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::UnboundPlaceholder`] for a missing input, and
    /// [`Error::ShapeConstraintViolation`],
    /// [`Error::ShapeConstraintEvaluation`], or [`Error::TensorRuntime`] when
    /// execution discovers a symbolic-guard or typed backend failure.
    pub fn run_many_values_with_inputs(
        &mut self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<Vec<TensorValue>> {
        let input_tensors = resolve_ordered_inputs(program, inputs, &mut self.backend)?;
        self.eval_exec_ir_values(&program.staging, input_tensors)
    }

    /// Run a program with ordered borrowed runtime inputs.
    ///
    /// A non-empty slice supplies every semantic input in order. Input tensors
    /// are borrowed by the executor for this call instead of cloned into slots.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{
    ///     DType, GraphCompiler, GraphExecutor, TensorRead, TensorView, TracedTensor, TypedTensorView,
    /// };
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
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
    /// let outputs = executor.run_many_with_input_reads(&program, &[read]).unwrap();
    /// assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphInputCountMismatch`] for a non-empty slice with
    /// the wrong number of inputs,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::UnboundPlaceholder`] for a missing input, and
    /// [`Error::ShapeConstraintViolation`],
    /// [`Error::ShapeConstraintEvaluation`], or [`Error::TensorRuntime`] when
    /// execution discovers a symbolic-guard or typed backend failure.
    pub fn run_many_with_input_reads<'a>(
        &mut self,
        program: &'a CompiledGraph,
        inputs: &[TensorRead<'a>],
    ) -> Result<Vec<Tensor>> {
        let inputs = resolve_ordered_input_reads(program, inputs, &mut self.backend)?;
        self.eval_exec_ir_slots(&program.staging, inputs)
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
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
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
    ///     .run_many_values_with_input_reads(&program, &[read])
    ///     .unwrap();
    /// assert_eq!(outputs.len(), 1);
    /// assert!(matches!(&outputs[0], TensorValue::View(_)));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphInputCountMismatch`] for a non-empty slice with
    /// the wrong number of inputs,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::UnboundPlaceholder`] for a missing input, and
    /// [`Error::ShapeConstraintViolation`],
    /// [`Error::ShapeConstraintEvaluation`], or [`Error::TensorRuntime`] when
    /// execution discovers a symbolic-guard or typed backend failure.
    pub fn run_many_values_with_input_reads<'a>(
        &mut self,
        program: &'a CompiledGraph,
        inputs: &[TensorRead<'a>],
    ) -> Result<Vec<TensorValue>> {
        let inputs = resolve_ordered_input_reads(program, inputs, &mut self.backend)?;
        self.eval_exec_ir_slot_values(&program.staging, inputs)
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
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let out = executor.run(&program).unwrap();
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[-2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when the input count does not match the
    /// execution program, [`Error::ShapeConstraintViolation`] or
    /// [`Error::ShapeConstraintEvaluation`] when symbolic guards fail, and
    /// [`Error::TensorRuntime`] or [`Error::Extension`] when typed backend or
    /// extension execution fails.
    pub(crate) fn eval_exec_ir(
        &mut self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>> {
        validate_exec_input_count(program, inputs.len())?;
        let input_shapes: Vec<&[usize]> = inputs.iter().map(Tensor::shape).collect();
        crate::exec::validate_shape_guards(program, &input_shapes)?;
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
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when the input count does not match the
    /// execution program, [`Error::ShapeConstraintViolation`] or
    /// [`Error::ShapeConstraintEvaluation`] when symbolic guards fail, and
    /// [`Error::TensorRuntime`] or [`Error::Extension`] when typed backend or
    /// extension execution fails.
    pub(crate) fn eval_exec_ir_values(
        &mut self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>> {
        validate_exec_input_count(program, inputs.len())?;
        let input_shapes: Vec<&[usize]> = inputs.iter().map(Tensor::shape).collect();
        crate::exec::validate_shape_guards(program, &input_shapes)?;
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
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let out = executor.run(&program).unwrap();
    /// assert_eq!(out.shape(), &[1]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when the input count does not match the
    /// execution program, [`Error::ShapeConstraintViolation`] or
    /// [`Error::ShapeConstraintEvaluation`] when symbolic guards fail, and
    /// [`Error::TensorRuntime`] or [`Error::Extension`] when typed backend or
    /// extension execution fails.
    #[cfg(test)]
    pub(crate) fn eval_exec_ir_non_consuming(
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
    /// assert_eq!(executor.materialize_value(&value).unwrap().shape(), &[2, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when the input count does not match the
    /// execution program, [`Error::ShapeConstraintViolation`] or
    /// [`Error::ShapeConstraintEvaluation`] when symbolic guards fail, and
    /// [`Error::TensorRuntime`] or [`Error::Extension`] when typed backend or
    /// extension execution fails.
    #[cfg(test)]
    pub(crate) fn eval_exec_ir_non_consuming_values(
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
        let input_shapes: Vec<&[usize]> = inputs.iter().map(ExecSlot::shape).collect();
        crate::exec::validate_shape_guards(program, &input_shapes)?;
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
        let input_shapes: Vec<&[usize]> = inputs.iter().map(ExecSlot::shape).collect();
        crate::exec::validate_shape_guards(program, &input_shapes)?;
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

fn resolve_ordered_inputs(
    program: &CompiledGraph,
    inputs: &[&Tensor],
    backend: &mut impl TensorBackend,
) -> Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return resolve_default_inputs(program, backend);
    }
    validate_ordered_input_metadata(program, inputs)?;
    let input_shapes: Vec<_> = inputs.iter().map(|tensor| tensor.shape()).collect();
    crate::exec::validate_shape_guards(&program.staging, &input_shapes)?;
    Ok(inputs.iter().map(|tensor| (*tensor).clone()).collect())
}

fn resolve_ordered_input_reads<'a>(
    program: &'a CompiledGraph,
    inputs: &[TensorRead<'a>],
    backend: &mut impl TensorBackend,
) -> Result<Vec<ExecSlot<'a>>> {
    if inputs.is_empty() {
        return resolve_default_input_reads(program, backend);
    }
    validate_ordered_input_metadata(program, inputs)?;
    let input_shapes: Vec<_> = inputs.iter().map(TensorRead::shape).collect();
    crate::exec::validate_shape_guards(&program.staging, &input_shapes)?;
    Ok(inputs
        .iter()
        .cloned()
        .map(ExecSlot::Read)
        .collect::<Vec<_>>())
}

fn semantic_default_inputs(program: &CompiledGraph) -> Result<Vec<&Tensor>> {
    program
        .program()
        .inputs()
        .iter()
        .enumerate()
        .map(|(input_index, value)| {
            program
                .bindings()
                .tensor_ref_for_input(*value)
                .map(AsRef::as_ref)
                .ok_or_else(|| Error::UnboundPlaceholder {
                    input_key: format!("semantic input {input_index}"),
                })
        })
        .collect()
}

fn resolve_default_inputs(
    program: &CompiledGraph,
    backend: &mut impl TensorBackend,
) -> Result<Vec<Tensor>> {
    let defaults = semantic_default_inputs(program)?;
    validate_ordered_input_metadata(program, &defaults)?;
    let input_shapes: Vec<_> = defaults.iter().map(|tensor| tensor.shape()).collect();
    crate::exec::validate_shape_guards(&program.staging, &input_shapes)?;
    defaults
        .into_iter()
        .map(|default| resolve_default_tensor(default, backend))
        .collect()
}

fn resolve_default_input_reads<'a>(
    program: &'a CompiledGraph,
    backend: &mut impl TensorBackend,
) -> Result<Vec<ExecSlot<'a>>> {
    let defaults = semantic_default_inputs(program)?;
    validate_ordered_input_metadata(program, &defaults)?;
    let input_shapes: Vec<_> = defaults.iter().map(|tensor| tensor.shape()).collect();
    crate::exec::validate_shape_guards(&program.staging, &input_shapes)?;
    defaults
        .into_iter()
        .map(|default| {
            if should_upload_default_tensor(default) {
                Ok(ExecSlot::Owned(backend.upload_host_tensor(default)?))
            } else {
                Ok(ExecSlot::Read(TensorRead::from_tensor(default)))
            }
        })
        .collect()
}

fn validate_ordered_input_metadata<T: InputMetadata>(
    program: &CompiledGraph,
    inputs: &[T],
) -> Result<()> {
    let expected = program.input_count();
    if inputs.len() != expected {
        return Err(Error::GraphInputCountMismatch {
            expected,
            actual: inputs.len(),
        });
    }
    let input_shapes: Vec<_> = inputs.iter().map(InputMetadata::shape).collect();
    for (input_value, actual) in program.program().inputs().iter().zip(inputs) {
        let metadata = program
            .program()
            .value_metadata(*input_value)
            .map_err(|source| Error::RuntimeState {
                op: "GraphExecutor::validate_ordered_input_metadata",
                phase: crate::error::ErrorPhase::Execution,
                message: source.to_string(),
            })?;
        if metadata.dtype() != actual.dtype() {
            return Err(Error::PlaceholderDtypeMismatch {
                expected: metadata.dtype(),
                actual: actual.dtype(),
            });
        }
        if metadata.shape().len() != actual.shape().len() {
            return Err(Error::PlaceholderRankMismatch {
                expected: metadata.shape().len(),
                actual: actual.shape().len(),
            });
        }
        let mut expected_shape = actual.shape().to_vec();
        let mut exact_mismatch = false;
        for (axis, (extent, actual_size)) in metadata.shape().iter().zip(actual.shape()).enumerate()
        {
            match extent {
                ShapeExtent::Exact(expression) => {
                    let expected = expression.eval(&input_shapes).map_err(|source| {
                        Error::runtime_state_source(
                            "GraphExecutor::validate_ordered_input_metadata",
                            crate::error::ErrorPhase::Execution,
                            source,
                        )
                    })?;
                    expected_shape[axis] = expected;
                    exact_mismatch |= expected != *actual_size;
                }
                ShapeExtent::UpperBound(expression) => {
                    let bound = expression.eval(&input_shapes).map_err(|source| {
                        Error::runtime_state_source(
                            "GraphExecutor::validate_ordered_input_metadata",
                            crate::error::ErrorPhase::Execution,
                            source,
                        )
                    })?;
                    if *actual_size > bound {
                        return Err(Error::PlaceholderShapeBoundExceeded {
                            axis,
                            bound,
                            actual: *actual_size,
                        });
                    }
                }
                ShapeExtent::Unknown => {}
            }
        }
        if exact_mismatch {
            return Err(Error::PlaceholderShapeMismatch {
                expected: expected_shape,
                actual: actual.shape().to_vec(),
            });
        }
    }
    Ok(())
}

trait InputMetadata {
    fn dtype(&self) -> DType;
    fn shape(&self) -> &[usize];
}

impl InputMetadata for &Tensor {
    fn dtype(&self) -> DType {
        Tensor::dtype(self)
    }

    fn shape(&self) -> &[usize] {
        Tensor::shape(self)
    }
}

impl InputMetadata for TensorRead<'_> {
    fn dtype(&self) -> DType {
        TensorRead::dtype(self)
    }

    fn shape(&self) -> &[usize] {
        TensorRead::shape(self)
    }
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

#[cfg(test)]
mod tests;
