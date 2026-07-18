use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use computegraph::compile::{compile, CompiledProgram};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::ValueKey;
use lru::LruCache;
use num_complex::{Complex32, Complex64};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, Tensor, TensorScalar};

use super::cache::{
    compile_cache_stats, compute_cache_key, CacheKey, GraphCompilerCacheStats,
    DEFAULT_COMPILE_CACHE_CAPACITY,
};
use super::program::{GraphProgram, GraphProgramInput};
use crate::compiler::{compile_std_to_exec_with_options_and_constraints, CompilerOptions};
use crate::error::{Error, Result};
use crate::exec::ExecProgram;
use crate::extension_cache::{ExtensionCacheSelector, ExtensionCacheStore};
use crate::shape_constraint::SlotScopedShapeConstraint;
use crate::traced::{try_concrete_shape, TracedTensor};

#[derive(Clone)]
struct InputDescriptor {
    key: TensorInputKey,
    dtype: DType,
    shape: Vec<usize>,
    default_tensor: Option<Arc<Tensor>>,
}

/// Compiler for traced tensor graphs.
///
/// A graph compiler lowers one or more [`TracedTensor`] outputs to a reusable
/// [`GraphProgram`] without requiring a backend.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{GraphCompiler, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let y = (&x + &x).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// assert_eq!(program.output_count(), 1);
/// ```
pub struct GraphCompiler {
    compile_cache: LruCache<CacheKey, ExecProgram>,
    extension_cache: ExtensionCacheStore,
    compiler_options: CompilerOptions,
}

impl fmt::Debug for GraphCompiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphCompiler")
            .field("cache_stats", &self.cache_stats())
            .field("compile_cache_capacity", &self.compile_cache_capacity())
            .field("compiler_options", &self.compiler_options)
            .field("extension_cache", &self.extension_cache)
            .finish_non_exhaustive()
    }
}

impl GraphCompiler {
    /// Create a compiler with bounded default caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            compile_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPILE_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
            ),
            extension_cache: ExtensionCacheStore::new(),
            compiler_options: CompilerOptions::default(),
        }
    }

    /// Create a compiler with explicit lowering and optimizer options.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CompilerOptions, OptimizerConfig};
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::with_compiler_options(CompilerOptions {
    ///     optimizer: OptimizerConfig {
    ///         dot_decomposer: true,
    ///         ..OptimizerConfig::default()
    ///     },
    /// });
    /// assert!(compiler.compiler_options().optimizer.dot_decomposer);
    /// ```
    pub fn with_compiler_options(compiler_options: CompilerOptions) -> Self {
        Self {
            compile_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPILE_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
            ),
            extension_cache: ExtensionCacheStore::new(),
            compiler_options,
        }
    }

    /// Compile one traced output into a graph program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// assert_eq!(program.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch`, `RankMismatch`,
    /// `DTypeMismatch`, or `InvalidArgument` for invalid graph metadata or
    /// shape constraints, [`Error::RuntimeState`] for missing/inconsistent
    /// metadata or cache state, and [`Error::Internal`] when the graph
    /// violates a compiler invariant. Extension lowering failures retain
    /// their typed [`Error::Extension`] source.
    pub fn compile(&mut self, output: &TracedTensor) -> Result<GraphProgram> {
        self.compile_many(&[output])
    }

    /// Compile multiple traced outputs into one graph program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let y = x.neg().unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile_many(&[&x, &y]).unwrap();
    /// assert_eq!(program.output_count(), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch`, `RankMismatch`,
    /// `DTypeMismatch`, or `InvalidArgument` for invalid graph metadata or
    /// shape constraints, [`Error::RuntimeState`] for missing/inconsistent
    /// metadata or cache state, and [`Error::Internal`] when the graph
    /// violates a compiler invariant. Extension lowering failures retain
    /// their typed [`Error::Extension`] source.
    pub fn compile_many(&mut self, outputs: &[&TracedTensor]) -> Result<GraphProgram> {
        let mut all_inputs = HashMap::new();
        for output in outputs {
            for (key, tensor) in output.inputs_map.iter() {
                if let Some(existing) = all_inputs.get(key) {
                    if !default_tensors_equivalent(existing, tensor) {
                        return Err(Error::DuplicateBinding {
                            input_key: format!("{:?}", key),
                        });
                    }
                    continue;
                }
                all_inputs.insert(key.clone(), tensor.clone());
            }
        }
        self.compile_many_with_descriptors(outputs, &HashMap::new(), &all_inputs)
    }

    /// Compile one traced output with concrete placeholder specs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler
    ///     .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
    ///     .unwrap();
    /// assert_eq!(program.input_specs()[0].shape(), &[3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnexpectedBinding`] for a data-carrying tensor,
    /// [`Error::DuplicateBinding`] for repeated placeholders,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::PlaceholderRankMismatch`] for incompatible specs, and
    /// [`Error::Validation`] with `ShapeMismatch`, `RankMismatch`,
    /// `DTypeMismatch`, or `InvalidArgument` / [`Error::RuntimeState`] when
    /// compilation or metadata lowering fails.
    pub fn compile_with_input_specs(
        &mut self,
        output: &TracedTensor,
        bindings: &[(&TracedTensor, DType, &[usize])],
    ) -> Result<GraphProgram> {
        let mut binding_specs = HashMap::new();
        for (index, (placeholder, dtype, shape)) in bindings.iter().enumerate() {
            validate_placeholder_spec(index, placeholder, *dtype, shape)?;
            let key = placeholder.input_key().ok_or(Error::UnexpectedBinding {
                binding_index: index,
            })?;
            if binding_specs
                .insert(
                    key.clone(),
                    InputDescriptor {
                        key: key.clone(),
                        dtype: *dtype,
                        shape: (*shape).to_vec(),
                        default_tensor: None,
                    },
                )
                .is_some()
            {
                return Err(Error::DuplicateBinding {
                    input_key: format!("{:?}", key),
                });
            }
        }

        self.compile_many_with_descriptors(&[output], &binding_specs, output.inputs_map.as_ref())
    }

    /// Number of compiled programs currently retained.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn compile_cache_len(&self) -> usize {
        self.compile_cache.len()
    }

    /// Current compiled-program cache capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert!(compiler.compile_cache_capacity().get() > 0);
    /// ```
    pub fn compile_cache_capacity(&self) -> NonZeroUsize {
        self.compile_cache.cap()
    }

    /// Resize the compiled-program cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.set_compile_cache_capacity(NonZeroUsize::new(2).unwrap());
    /// assert_eq!(compiler.compile_cache_capacity().get(), 2);
    /// ```
    pub fn set_compile_cache_capacity(&mut self, capacity: NonZeroUsize) {
        self.compile_cache.resize(capacity);
    }

    /// Return the compiler options used for future graph lowerings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::CompilerOptions;
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compiler_options(), CompilerOptions::default());
    /// ```
    pub fn compiler_options(&self) -> CompilerOptions {
        self.compiler_options
    }

    /// Replace compiler options and clear compiled graph cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CompilerOptions, OptimizerConfig};
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// let options = CompilerOptions {
    ///     optimizer: OptimizerConfig {
    ///         dot_decomposer: true,
    ///         ..OptimizerConfig::default()
    ///     },
    /// };
    /// compiler.set_compiler_options(options);
    /// assert_eq!(compiler.compiler_options(), options);
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn set_compiler_options(&mut self, compiler_options: CompilerOptions) {
        if self.compiler_options == compiler_options {
            return;
        }
        self.compiler_options = compiler_options;
        self.clear_compile_cache();
    }

    /// Clear the compiled-program cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_compile_cache();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn clear_compile_cache(&mut self) {
        self.compile_cache.clear();
    }

    /// Clear generic extension compile-time cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_extension_caches();
    /// assert_eq!(compiler.cache_stats().extensions.entries, 0);
    /// ```
    pub fn clear_extension_caches(&mut self) {
        self.extension_cache.clear();
    }

    /// Clear every cache owned by the compiler.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_caches();
    /// assert_eq!(compiler.cache_stats().compile.entries, 0);
    /// ```
    pub fn clear_caches(&mut self) {
        self.clear_compile_cache();
        self.clear_extension_caches();
    }

    /// Return cache-entry and retained-byte stats.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// let stats = compiler.cache_stats();
    /// assert_eq!(stats.compile.entries, 0);
    /// ```
    pub fn cache_stats(&self) -> GraphCompilerCacheStats {
        GraphCompilerCacheStats {
            compile: compile_cache_stats(&self.compile_cache),
            extensions: self.extension_cache.stats(ExtensionCacheSelector::All),
        }
    }

    /// Borrow generic extension compile-time cache storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert!(compiler.extension_caches().is_empty());
    /// ```
    pub fn extension_caches(&self) -> &ExtensionCacheStore {
        &self.extension_cache
    }

    /// Mutably borrow generic extension compile-time cache storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.extension_caches_mut().clear();
    /// ```
    pub fn extension_caches_mut(&mut self) -> &mut ExtensionCacheStore {
        &mut self.extension_cache
    }

    fn compile_many_with_descriptors(
        &mut self,
        outputs: &[&TracedTensor],
        binding_specs: &HashMap<TensorInputKey, InputDescriptor>,
        default_inputs: &HashMap<TensorInputKey, Arc<Tensor>>,
    ) -> Result<GraphProgram> {
        let mut constraint_scopes = Vec::new();
        let mut seen_constraint_scopes = std::collections::HashSet::new();
        for output in outputs {
            for scope in output.constraint_scopes.as_slice() {
                if seen_constraint_scopes.insert(Arc::as_ptr(scope)) {
                    #[cfg(test)]
                    test_support::record_constraint_scope_clones(1);
                    constraint_scopes.push(Arc::clone(scope));
                }
            }
        }

        let mut roots = Vec::new();
        let mut output_keys = Vec::with_capacity(outputs.len());
        for output in outputs {
            roots.extend(output.resolve_roots());
            output_keys.push(output.graph.values()[output.val].key.clone());
        }

        let view = resolve(roots);
        let graph = materialize_merge(&view, &output_keys);
        let mut compiled = compile(&graph);
        prune_compiled_extension_outputs(&mut compiled)?;
        let slot_by_key: HashMap<_, _> = graph
            .values
            .iter()
            .enumerate()
            .map(|(slot, value)| (value.key.clone(), slot))
            .collect();
        let mut scoped_constraints = Vec::new();
        for scope in constraint_scopes {
            for scoped in scope.constraints() {
                let origin_slots: Vec<_> = scoped
                    .origins
                    .iter()
                    .filter_map(|key| slot_by_key.get(key).copied())
                    .collect();
                if origin_slots.is_empty() {
                    continue;
                }
                let origin_instruction = origin_slots.iter().find_map(|&slot| {
                    graph
                        .values
                        .get(slot)
                        .and_then(|value| value.producer.map(|p| p.0))
                });
                let mut local = scoped.local.clone();
                if let Some(instruction_index) = origin_instruction {
                    local.source = local.source.with_instruction(instruction_index);
                }
                let mut input_slots = Vec::with_capacity(scoped.inputs.len());
                for (input_idx, key) in scoped.inputs.iter().enumerate() {
                    let Some(slot) = slot_by_key.get(key).copied() else {
                        return Err(Error::ShapeConstraintEvaluation {
                            family: local.source.family_id,
                            instruction_index: local.source.instruction_index,
                            relation: local.relation,
                            expression: format!("{:?}", local.lhs),
                            cause: crate::ShapeConstraintEvalError::MissingInput {
                                input_idx,
                                input_count: scoped.inputs.len(),
                            },
                        });
                    };
                    input_slots.push(slot);
                }
                scoped_constraints.push(SlotScopedShapeConstraint {
                    origin_slots,
                    input_slots,
                    local,
                });
            }
        }

        let mut descriptors = Vec::with_capacity(graph.inputs.len());
        let mut input_dtypes = Vec::with_capacity(graph.inputs.len());
        let mut input_shapes = Vec::with_capacity(graph.inputs.len());
        for key in &graph.inputs {
            let ValueKey::Input(input_key) = key else {
                return Err(Error::Internal(
                    "expected Input key in graph inputs".to_string(),
                ));
            };
            let descriptor = descriptor_for_input(input_key, binding_specs, default_inputs)?;
            input_dtypes.push(descriptor.dtype);
            input_shapes.push(DimExpr::from_concrete(&descriptor.shape));
            descriptors.push(GraphProgramInput::new(
                descriptor.key,
                descriptor.dtype,
                descriptor.shape.clone(),
                DimExpr::from_concrete(&descriptor.shape),
                descriptor.default_tensor,
            ));
        }

        let exec = compile_std_to_exec_with_options_and_constraints(
            &compiled,
            &input_dtypes,
            &input_shapes,
            self.compiler_options,
            &scoped_constraints,
            &input_shapes,
        )?;
        let exec = self.get_or_compile(exec);
        Ok(GraphProgram::new(exec, descriptors))
    }

    fn get_or_compile(&mut self, exec: ExecProgram) -> ExecProgram {
        let key = compute_cache_key(&exec);
        if let Some(cached) = self.compile_cache.get(&key) {
            let mut current = cached.clone();
            current.shape_guards = exec.shape_guards;
            return current;
        }
        self.compile_cache.put(key, exec.clone());
        exec
    }
}

impl Default for GraphCompiler {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_placeholder_spec(
    index: usize,
    placeholder: &TracedTensor,
    dtype: DType,
    shape: &[usize],
) -> Result<()> {
    if placeholder.data.is_some() {
        return Err(Error::UnexpectedBinding {
            binding_index: index,
        });
    }
    placeholder.input_key().ok_or(Error::UnexpectedBinding {
        binding_index: index,
    })?;

    if placeholder.dtype != dtype {
        return Err(Error::PlaceholderDtypeMismatch {
            expected: placeholder.dtype,
            actual: dtype,
        });
    }
    validate_placeholder_shape(placeholder, shape)
}

fn validate_placeholder_shape(placeholder: &TracedTensor, shape: &[usize]) -> Result<()> {
    match try_concrete_shape(placeholder) {
        Some(expected_shape) => {
            if expected_shape.as_slice() != shape {
                return Err(Error::PlaceholderShapeMismatch {
                    expected: expected_shape,
                    actual: shape.to_vec(),
                });
            }
        }
        None => {
            if placeholder.rank != shape.len() {
                return Err(Error::PlaceholderRankMismatch {
                    expected: placeholder.rank,
                    actual: shape.len(),
                });
            }
        }
    }
    Ok(())
}

fn descriptor_for_input(
    key: &TensorInputKey,
    binding_specs: &HashMap<TensorInputKey, InputDescriptor>,
    default_inputs: &HashMap<TensorInputKey, Arc<Tensor>>,
) -> Result<InputDescriptor> {
    if let Some(tensor) = default_inputs.get(key) {
        return Ok(InputDescriptor {
            key: key.clone(),
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            default_tensor: Some(tensor.clone()),
        });
    }
    if let Some(spec) = binding_specs.get(key) {
        return Ok(spec.clone());
    }
    Err(Error::UnboundPlaceholder {
        input_key: format!("{:?}", key),
    })
}

fn prune_compiled_extension_outputs(prog: &mut CompiledProgram<StdTensorOp>) -> Result<()> {
    let mut live_slots = vec![false; prog.n_slots];
    for &slot in &prog.output_slots {
        let Some(live) = live_slots.get_mut(slot) else {
            return Err(invalid_compiled_graph(format!(
                "program output slot {slot} is outside slot table of length {}",
                prog.n_slots
            )));
        };
        *live = true;
    }

    for instr in prog.instructions.iter_mut().rev() {
        let live_outputs = instr
            .outputs
            .iter()
            .map(|&slot| {
                live_slots.get(slot).copied().ok_or_else(|| {
                    invalid_compiled_graph(format!(
                        "instruction output slot {slot} is outside slot table of length {}",
                        prog.n_slots
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if let StdTensorOp::Extension(ext) = &instr.operation {
            if let Some(pruned) = ext.prune_outputs(&live_outputs) {
                let kept_outputs = instr
                    .outputs
                    .iter()
                    .zip(live_outputs.iter())
                    .filter_map(|(&slot, &live)| live.then_some(slot))
                    .collect::<Vec<_>>();
                if pruned.output_count() != kept_outputs.len() {
                    return Err(invalid_compiled_graph(format!(
                        "extension family_id={:?} pruned to {} outputs for {} live slots",
                        ext.family_id(),
                        pruned.output_count(),
                        kept_outputs.len()
                    )));
                }
                instr.operation = StdTensorOp::Extension(pruned);
                instr.outputs = kept_outputs;
            }
        }

        if live_outputs.iter().any(|&live| live) {
            for &slot in &instr.inputs {
                let Some(live) = live_slots.get_mut(slot) else {
                    return Err(invalid_compiled_graph(format!(
                        "instruction input slot {slot} is outside slot table of length {}",
                        prog.n_slots
                    )));
                };
                *live = true;
            }
        }
    }

    Ok(())
}

fn invalid_compiled_graph(message: impl Into<String>) -> Error {
    Error::Internal(message.into())
}

fn default_tensors_equivalent(lhs: &Arc<Tensor>, rhs: &Arc<Tensor>) -> bool {
    if Arc::ptr_eq(lhs, rhs) {
        return true;
    }
    if lhs.dtype() != rhs.dtype() || lhs.shape() != rhs.shape() {
        return false;
    }
    match lhs.dtype() {
        DType::F32 => default_slices_equivalent::<f32>(lhs, rhs),
        DType::F64 => default_slices_equivalent::<f64>(lhs, rhs),
        DType::I32 => default_slices_equivalent::<i32>(lhs, rhs),
        DType::I64 => default_slices_equivalent::<i64>(lhs, rhs),
        DType::Bool => default_slices_equivalent::<bool>(lhs, rhs),
        DType::C32 => default_slices_equivalent::<Complex32>(lhs, rhs),
        DType::C64 => default_slices_equivalent::<Complex64>(lhs, rhs),
    }
}

fn default_slices_equivalent<T: TensorScalar + PartialEq>(lhs: &Tensor, rhs: &Tensor) -> bool {
    match (lhs.as_slice::<T>(), rhs.as_slice::<T>()) {
        (Ok(lhs), Ok(rhs)) => lhs == rhs,
        // Backend-resident defaults cannot be inspected here; only the same
        // Arc<Tensor> is considered equivalent by `default_tensors_equivalent`.
        _ => false,
    }
}

#[cfg(test)]
mod constraint_scope_tests;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape_constraint::ConstraintSource;
    use crate::ShapeGuard;
    use std::any::Any;
    use std::hash::Hasher;
    use std::sync::Arc;
    use tenferro_ops::{dim_expr::DimExpr, ext_op::ExtensionOp, ShapeRelation, SymDim};
    use tenferro_tensor::{
        Buffer, BufferHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
        TypedTensor,
    };

    #[test]
    fn compile_cache_hit_preserves_current_guard_provenance() {
        use tenferro_cpu::CpuBackend;

        let mut compiler = GraphCompiler::new();
        let mut first = ExecProgram {
            instructions: Vec::new(),
            input_slots: vec![0],
            output_slots: vec![0],
            n_slots: 1,
            shape_guards: vec![test_guard("example.first.v1", 3)],
        };
        let mut second = first.clone();
        second.shape_guards = vec![test_guard("example.second.v1", 9)];

        first = compiler.get_or_compile(first);
        let cached = compiler.get_or_compile(second);

        assert_eq!(compiler.compile_cache_len(), 1);
        assert_eq!(first.shape_guards[0].source.family_id, "example.first.v1");
        assert_eq!(
            cached.shape_guards[0].source,
            ConstraintSource {
                family_id: "example.second.v1",
                instruction_index: Some(9),
            }
        );

        let mut executor = crate::GraphExecutor::new(CpuBackend::new());
        let error = executor
            .eval_exec_ir(
                &cached,
                vec![Tensor::from_vec_col_major(vec![3], vec![0.0_f64; 3]).unwrap()],
            )
            .unwrap_err();
        assert!(matches!(
            error,
            Error::ShapeConstraintViolation {
                family: "example.second.v1",
                instruction_index: Some(9),
                lhs_value: 3,
                rhs_value: 2,
                ..
            }
        ));
    }

    fn test_guard(family_id: &'static str, instruction_index: usize) -> ShapeGuard {
        ShapeGuard {
            source: ConstraintSource {
                family_id,
                instruction_index: Some(instruction_index),
            },
            relation: ShapeRelation::Equal,
            lhs: DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
            rhs: DimExpr::Const(2),
        }
    }

    #[test]
    fn compile_many_rejects_conflicting_default_inputs_for_same_key() {
        let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
        let y1 = x.neg().unwrap();
        let mut y2 = x.neg().unwrap();
        let key = x.input_key().expect("concrete traced tensor has input key");
        let replacement = Arc::new(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap());
        let mut inputs = (*y2.inputs_map).clone();
        inputs.insert(key.clone(), replacement);
        y2.inputs_map = Arc::new(inputs);

        let err = GraphCompiler::new().compile_many(&[&y1, &y2]).unwrap_err();

        assert!(matches!(
            err,
            Error::DuplicateBinding { ref input_key } if input_key.contains(&format!("{key:?}"))
        ));
    }

    #[test]
    fn default_tensors_equivalent_rejects_distinct_backend_buffers() {
        let placement = Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
        };
        let lhs = Arc::new(Tensor::F64(
            TypedTensor::from_buffer_col_major(
                vec![2],
                Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(1, 2))),
                placement.clone(),
            )
            .unwrap(),
        ));
        let rhs = Arc::new(Tensor::F64(
            TypedTensor::from_buffer_col_major(
                vec![2],
                Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(2, 2))),
                placement,
            )
            .unwrap(),
        ));

        assert!(
            !default_tensors_equivalent(&lhs, &rhs),
            "distinct backend-resident default tensors must not compare equal just because both are unreadable on host"
        );
        assert!(default_tensors_equivalent(&lhs, &lhs));
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct PrunableTestOp {
        pruned: bool,
    }

    impl ExtensionOp for PrunableTestOp {
        fn family_id(&self) -> &'static str {
            "tenferro-runtime.test-prunable.v1"
        }

        fn payload_hash(&self, hasher: &mut dyn Hasher) {
            hasher.write_u8(u8::from(self.pruned));
        }

        fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
            other
                .as_any()
                .downcast_ref::<Self>()
                .is_some_and(|that| self == that)
        }

        fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
            Arc::new(self.clone())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn input_count(&self) -> usize {
            1
        }

        fn output_count(&self) -> usize {
            if self.pruned {
                1
            } else {
                3
            }
        }

        fn infer_output_meta(
            &self,
            ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
        ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
            let dtype = ctx.input_dtype(0)?;
            let shape = ctx.input_shape(0)?.to_vec();
            Ok((0..self.output_count())
                .map(|_| (dtype, shape.clone()))
                .collect())
        }

        fn prune_outputs(&self, live_outputs: &[bool]) -> Option<Arc<dyn ExtensionOp>> {
            (!self.pruned && live_outputs == [false, true, false])
                .then(|| Arc::new(Self { pruned: true }) as Arc<dyn ExtensionOp>)
        }
    }

    #[test]
    fn compile_prunes_extension_outputs_with_replacement_op() {
        let input = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let outputs =
            crate::extension::apply(Arc::new(PrunableTestOp { pruned: false }), &[&input]).unwrap();

        let program = GraphCompiler::new().compile(&outputs[1]).unwrap();
        let pruned_instruction = program
            .lowering_view()
            .instructions()
            .find_map(|inst| match inst.op() {
                crate::GraphOpView::Extension { op }
                    if op.family_id() == "tenferro-runtime.test-prunable.v1" =>
                {
                    Some((
                        inst.output_slots().to_vec(),
                        format!("{op:?}"),
                        op.output_count(),
                    ))
                }
                _ => None,
            })
            .expect("compiled program should contain the test extension");

        assert_eq!(pruned_instruction.0.len(), 1);
        assert!(pruned_instruction.1.contains("pruned: true"));
        assert_eq!(pruned_instruction.2, 1);
    }
}
