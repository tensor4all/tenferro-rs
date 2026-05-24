use std::collections::{HashMap, HashSet};

use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::{
    buffer_pool::BufferPoolStats, cpu::CpuBackend, DType, RuntimeCacheControl, Tensor,
    TensorBackend, TypedTensor,
};

use super::cache::{CpuGraphExecutorCacheStats, GraphExecutorCacheStats};
use super::program::{GraphProgram, GraphProgramInput};
use crate::error::{Error, Result};
use crate::exec::ExecProgram;
use crate::traced::TracedTensor;
use tenferro_runtime::extension_runtime::{ExtensionExecutor, ExtensionRuntimeRegistryError};

/// Executes compiled graph programs on a concrete tensor backend.
///
/// A graph executor owns backend execution state only: backend runtime caches,
/// extension runtime state, and reusable execution workspace. Compilation
/// state lives in [`GraphCompiler`](super::GraphCompiler).
///
/// # Examples
///
/// ```
/// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let y = &x + &x;
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
    slot_workspace: Vec<Option<Tensor>>,
}

impl<B: TensorBackend + 'static> GraphExecutor<B> {
    /// Create an executor with the given backend and bounded default caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
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
        }
    }

    /// Borrow the backend used by this executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// let _backend = executor.backend();
    /// ```
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Borrow the extension runtime executor owned by this graph executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
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
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.extension_executor_mut().clear_caches();
    /// ```
    pub fn extension_executor_mut(&mut self) -> &mut ExtensionExecutor<B> {
        &mut self.extension_executor
    }

    /// Register one extension runtime on this executor.
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
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64]);
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

    /// Run a program using the program's default input tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64]);
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

    /// Run a one-output program with explicit runtime placeholder bindings.
    ///
    /// Explicit bindings override program defaults and are validated against
    /// the ordered input specs captured in the compiled program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    /// let y = &x + &x;
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
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

    /// Run a program with explicit runtime placeholder bindings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    /// let sum = &x + &x;
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&sum, &[(&x, DType::F64, &[2])])
    ///     .unwrap();
    /// let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
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
        let input_tensors = resolve_inputs(program, bindings)?;
        self.eval_exec_ir(&program.exec, input_tensors)
    }

    /// Evaluate an execution program through this executor's backend state.
    ///
    /// This lower-level entry point is intended for code that already owns an
    /// execution program and concrete ordered input tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]);
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

    /// Evaluate an execution program without consuming caller-owned inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]);
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
        self.eval_exec_ir(program, inputs.to_vec())
    }

    /// Clear backend-specific runtime analysis cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
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
    /// use tenferro::{CpuBackend, GraphExecutor};
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
    /// use tenferro::{CpuBackend, GraphExecutor};
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
    /// use tenferro::{CpuBackend, GraphExecutor};
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

impl GraphExecutor<CpuBackend> {
    /// Number of reusable typed host buffers retained by the CPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// assert_eq!(executor.buffer_pool_len(), 0);
    /// ```
    pub fn buffer_pool_len(&self) -> usize {
        self.backend.buffer_pool_len()
    }

    /// Snapshot reusable typed host buffers retained by the CPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// let stats = executor.buffer_pool_stats();
    /// assert_eq!(stats.buffers, 0);
    /// ```
    pub fn buffer_pool_stats(&self) -> BufferPoolStats {
        self.backend.buffer_pool_stats()
    }

    /// Reset all reusable typed host buffers retained by the CPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.reset_buffer_pool();
    /// assert_eq!(executor.buffer_pool_len(), 0);
    /// ```
    pub fn reset_buffer_pool(&mut self) {
        self.backend.reset_buffer_pool();
    }

    /// Return stats for executor caches and the CPU buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// let stats = executor.cpu_cache_stats();
    /// assert_eq!(stats.executor.extensions.entries, 0);
    /// assert_eq!(stats.buffer_pool.entries, 0);
    /// ```
    pub fn cpu_cache_stats(&self) -> CpuGraphExecutorCacheStats {
        CpuGraphExecutorCacheStats {
            executor: self.cache_stats(),
            buffer_pool: self.backend.buffer_pool_cache_stats(),
        }
    }

    /// Clear executor-owned caches and the CPU backend buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.clear_all_caches();
    /// assert_eq!(executor.cpu_cache_stats().buffer_pool.entries, 0);
    /// ```
    pub fn clear_all_caches(&mut self) {
        self.clear_caches();
        self.reset_buffer_pool();
    }

    /// Return the CPU GEMM analysis-cache slot capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// assert!(executor.gemm_analysis_cache_capacity() > 0);
    /// ```
    pub fn gemm_analysis_cache_capacity(&self) -> usize {
        self.backend_cache.capacity()
    }

    /// Resize the CPU GEMM analysis-cache slot capacity.
    ///
    /// A capacity of zero disables retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.set_gemm_analysis_cache_capacity(0);
    /// assert_eq!(executor.gemm_analysis_cache_capacity(), 0);
    /// ```
    pub fn set_gemm_analysis_cache_capacity(&mut self, capacity: usize) {
        self.backend_cache.set_capacity(capacity);
    }

    /// Return the CPU buffer-pool retention limit in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let executor = GraphExecutor::new(CpuBackend::new());
    /// assert!(executor.buffer_pool_limit_bytes() > 0);
    /// ```
    pub fn buffer_pool_limit_bytes(&self) -> usize {
        self.backend.buffer_pool_limit_bytes()
    }

    /// Update the CPU buffer-pool retention limit in bytes.
    ///
    /// A limit of zero disables buffer retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphExecutor};
    ///
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// executor.set_buffer_pool_limit_bytes(0);
    /// assert_eq!(executor.buffer_pool_limit_bytes(), 0);
    /// ```
    pub fn set_buffer_pool_limit_bytes(&mut self, max_retained_capacity_bytes: usize) {
        self.backend
            .set_buffer_pool_limit_bytes(max_retained_capacity_bytes);
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

fn resolve_inputs(
    program: &GraphProgram,
    bindings: &[(&TracedTensor, &Tensor)],
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
        .map(|input| resolve_input(input, &binding_map, &default_map))
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
) -> Result<Tensor> {
    let tensor = if let Some(bound) = bindings.get(&input.key) {
        (*bound).clone()
    } else if let Some(default) = &input.default_tensor {
        default.as_ref().clone()
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

fn deferred_zero_for_tangent_key(
    key: &TensorInputKey,
    bindings: &HashMap<TensorInputKey, &Tensor>,
    defaults: &HashMap<TensorInputKey, &Tensor>,
) -> Option<Tensor> {
    if matches!(key, TensorInputKey::User { .. }) {
        return None;
    }
    let root = tangent_primal_root(key);
    let primal = bindings.get(root).or_else(|| defaults.get(root))?;
    Some(zeros_tensor(primal.dtype(), primal.shape().to_vec()))
}

fn tangent_primal_root(key: &TensorInputKey) -> &TensorInputKey {
    match key {
        TensorInputKey::User { .. } => key,
        #[cfg(feature = "autodiff")]
        TensorInputKey::Tangent { of, .. } => tangent_primal_root(of),
    }
}

fn zeros_tensor(dtype: DType, shape: Vec<usize>) -> Tensor {
    match dtype {
        DType::F32 => Tensor::F32(TypedTensor::zeros(shape)),
        DType::F64 => Tensor::F64(TypedTensor::zeros(shape)),
        DType::I32 => Tensor::I32(TypedTensor::zeros(shape)),
        DType::I64 => Tensor::I64(TypedTensor::zeros(shape)),
        DType::Bool => {
            let len = shape.iter().product();
            Tensor::Bool(TypedTensor::from_vec_col_major(shape, vec![false; len]))
        }
        DType::C32 => Tensor::C32(TypedTensor::zeros(shape)),
        DType::C64 => Tensor::C64(TypedTensor::zeros(shape)),
    }
}
