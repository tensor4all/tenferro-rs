use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use computegraph::compile::{compile, CompiledProgram};
use computegraph::materialize::{
    materialize_merge, MaterializedGraph, MaterializedOperation, MaterializedValue,
};
use computegraph::resolve::{resolve, ResolvedView, ValueDef};
use computegraph::types::{OperationKey, ValueKey};
use computegraph::GraphOperation;
use num_complex::{Complex32, Complex64};
use tenferro_ops::dim_expr::{DimExpr, DimExprEvalError};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeExtent, ShapeRelation, SymDim};
use tenferro_tensor::{CacheStats, DType, SliceConfig, Tensor, TensorScalar};

use super::program::CompiledGraph;
#[cfg(test)]
use crate::compiler::semantic_staging::stage_semantic_program;
use crate::compiler::{lower_scoped_dim_expr, CompilerOptions};
use crate::error::{Error, Result};
use crate::extension_cache::{ExtensionCacheSelector, ExtensionCacheStore};
use crate::metadata::registered_meta;
use crate::program::{
    CoreSemanticOp, FrozenProgram, ProgramInputSpec, ProgramShapeRelation, ProgramValueMetadata,
    SemanticOpRef, SemanticProgramBuilder, ShapeGuard as ProgramShapeGuard,
};
use crate::shape_constraint::{discharge, LocalShapeConstraint, SlotScopedShapeConstraint};
use crate::shape_infer::{infer_extension_output_meta, infer_output_shapes};
use crate::trace::TracedGraph;
use crate::traced::{try_concrete_shape, TracedTensor};

#[derive(Clone)]
struct InputDescriptor {
    dtype: DType,
    shape: Vec<usize>,
    extent_identity: InputExtentIdentity,
    default_tensor: Option<Arc<Tensor>>,
}

#[derive(Clone, Copy)]
enum InputExtentIdentity {
    Concrete,
    Symbolic,
}

impl InputDescriptor {
    fn semantic_shape(&self, input_idx: usize) -> Vec<DimExpr> {
        match self.extent_identity {
            InputExtentIdentity::Concrete => DimExpr::from_concrete(&self.shape),
            InputExtentIdentity::Symbolic => (0..self.shape.len())
                .map(|axis| DimExpr::InputDim { input_idx, axis })
                .collect(),
        }
    }

    fn constraint_guard_shape(&self, input_idx: usize) -> Vec<DimExpr> {
        if self.default_tensor.is_some() {
            return input_dim_shape(input_idx, self.shape.len());
        }
        self.semantic_shape(input_idx)
    }
}

fn input_dim_shape(input_idx: usize, rank: usize) -> Vec<DimExpr> {
    (0..rank)
        .map(|axis| DimExpr::InputDim { input_idx, axis })
        .collect()
}

/// Compiler for traced tensor graphs.
///
/// A graph compiler lowers one or more [`TracedTensor`] outputs to a reusable
/// [`CompiledGraph`] without requiring a backend.
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
    extension_cache: ExtensionCacheStore,
    compiler_options: CompilerOptions,
}

impl fmt::Debug for GraphCompiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphCompiler")
            .field("extension_cache_stats", &self.cache_stats())
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
    /// assert!(compiler.extension_caches().is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
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
    pub fn compile(&mut self, output: &TracedTensor) -> Result<CompiledGraph> {
        self.compile_many(&[output])
    }

    /// Compile an immutable semantic trace without consulting a backend.
    ///
    /// This is the forward-only trace boundary. The compiler preserves the
    /// frozen semantic program and bindings without preparing backend/runtime
    /// staging. Runtime preparation owns backend-private staging and plan caches.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for invalid metadata or shape constraints,
    /// [`Error::Extension`] when extension lowering fails,
    /// [`Error::RuntimeState`] for inconsistent staging state, or
    /// [`Error::Internal`] when compilation encounters an invariant violation.
    pub fn compile_traced_graph(&mut self, graph: &TracedGraph) -> Result<CompiledGraph> {
        self.compile_frozen(graph.frozen())
    }

    /// Compile an immutable semantic program for ordered execution.
    ///
    /// This entry is used by validation-preserving semantic transforms such as
    /// whole-program AD. Tensor bindings remain outside semantic identity and
    /// are preserved in the returned [`CompiledGraph`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::dim_expr::DimExpr;
    /// use tenferro_runtime::program::{
    ///     CoreSemanticOp, ProgramInputSpec, SemanticProgramBuilder,
    /// };
    /// use tenferro_runtime::{DType, GraphCompiler};
    ///
    /// let mut builder = SemanticProgramBuilder::new();
    /// let input = builder
    ///     .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
    ///     .unwrap();
    /// let output = builder.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    /// let frozen = builder.finish(&[output]).unwrap();
    /// let compiled = GraphCompiler::new()
    ///     .compile_frozen_program(&frozen)
    ///     .unwrap();
    /// assert_eq!(compiled.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for invalid metadata or shape constraints,
    /// [`Error::Extension`] when extension lowering fails,
    /// [`Error::RuntimeState`] for inconsistent staging state, or
    /// [`Error::Internal`] when compilation encounters an invariant violation.
    pub fn compile_frozen_program(&mut self, frozen: &FrozenProgram) -> Result<CompiledGraph> {
        self.compile_frozen(frozen)
    }

    fn compile_frozen(&mut self, frozen: &FrozenProgram) -> Result<CompiledGraph> {
        validate_bound_shape_guards(frozen)?;
        Ok(CompiledGraph::new(
            frozen.clone(),
            self.compiler_options,
            [],
        ))
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
    pub fn compile_many(&mut self, outputs: &[&TracedTensor]) -> Result<CompiledGraph> {
        let all_inputs = collect_default_inputs(outputs)?;
        self.compile_many_with_descriptors(
            outputs,
            &HashMap::new(),
            &all_inputs,
            None,
            false,
            false,
        )
    }

    pub(crate) fn compile_ad_source(&mut self, output: &TracedTensor) -> Result<CompiledGraph> {
        let all_inputs = collect_default_inputs(&[output])?;
        self.compile_many_with_descriptors(
            &[output],
            &HashMap::new(),
            &all_inputs,
            None,
            true,
            true,
        )
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
    /// assert_eq!(program.input_count(), 1);
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
    ) -> Result<CompiledGraph> {
        let mut binding_specs = HashMap::new();
        let mut input_order = Vec::with_capacity(bindings.len());
        for (index, (placeholder, dtype, shape)) in bindings.iter().enumerate() {
            validate_placeholder_spec(index, placeholder, *dtype, shape)?;
            let key = placeholder.input_key().ok_or(Error::UnexpectedBinding {
                binding_index: index,
            })?;
            if binding_specs
                .insert(
                    key.clone(),
                    InputDescriptor {
                        dtype: *dtype,
                        shape: (*shape).to_vec(),
                        extent_identity: InputExtentIdentity::Concrete,
                        default_tensor: None,
                    },
                )
                .is_some()
            {
                return Err(Error::DuplicateBinding {
                    input_key: format!("{:?}", key),
                });
            }
            input_order.push(key);
        }

        self.compile_many_with_descriptors(
            &[output],
            &binding_specs,
            output.inputs_map.as_ref(),
            Some(&input_order),
            false,
            false,
        )
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

    /// Replace compiler options and clear compiler-owned extension cache entries.
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
    /// assert_eq!(compiler.cache_stats().entries, 0);
    /// ```
    pub fn set_compiler_options(&mut self, compiler_options: CompilerOptions) {
        if self.compiler_options == compiler_options {
            return;
        }
        self.compiler_options = compiler_options;
        self.clear_extension_caches();
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
    /// assert_eq!(compiler.cache_stats().entries, 0);
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
    /// assert_eq!(compiler.cache_stats().entries, 0);
    /// ```
    pub fn clear_caches(&mut self) {
        self.clear_extension_caches();
    }

    /// Return compiler-owned extension cache-entry and retained-byte stats.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// let stats = compiler.cache_stats();
    /// assert_eq!(stats.entries, 0);
    /// ```
    pub fn cache_stats(&self) -> CacheStats {
        self.extension_cache.stats(ExtensionCacheSelector::All)
    }

    /// Borrow generic compiler-owned extension cache storage.
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

    /// Mutably borrow generic compiler-owned extension cache storage.
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
        explicit_input_order: Option<&[TensorInputKey]>,
        include_checkpoint_aliases: bool,
        allow_unbound_placeholders: bool,
    ) -> Result<CompiledGraph> {
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
        let mut checkpoint_aliases = HashMap::new();
        let mut output_keys = Vec::with_capacity(outputs.len());
        for output in outputs {
            roots.extend(output.resolve_roots());
            if include_checkpoint_aliases {
                if let Some(chain) = &output.checkpoint_chain {
                    roots.extend(chain.collect_graphs());
                    for (alias_key, target_key) in chain.collect_aliases() {
                        insert_checkpoint_alias(&mut checkpoint_aliases, alias_key, target_key)?;
                    }
                }
            }
            output_keys.push(output.graph.values()[output.val].key.clone());
        }

        let view = resolve(roots);
        let graph = if checkpoint_aliases.is_empty() {
            materialize_merge(&view, &output_keys)
        } else {
            let checkpoint_alias_shapes = checkpoint_aliases
                .keys()
                .filter_map(|key| {
                    default_inputs
                        .get(key)
                        .map(|tensor| (key.clone(), tensor.shape().to_vec()))
                })
                .collect::<HashMap<_, _>>();
            materialize_merge_with_input_aliases(
                &view,
                &output_keys,
                &checkpoint_aliases,
                &checkpoint_alias_shapes,
            )
        };
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
        let mut input_keys = Vec::with_capacity(graph.inputs.len());
        for key in &graph.inputs {
            let ValueKey::Input(input_key) = key else {
                return Err(Error::Internal(
                    "expected Input key in graph inputs".to_string(),
                ));
            };
            let descriptor = descriptor_for_input(
                input_key,
                binding_specs,
                default_inputs,
                allow_unbound_placeholders,
            )?;
            descriptors.push(descriptor);
            input_keys.push(input_key.clone());
        }
        if let Some(explicit_input_order) = explicit_input_order {
            let input_position_by_key: HashMap<_, _> = graph
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(position, key)| match key {
                    ValueKey::Input(key) => Some((key.clone(), position)),
                    _ => None,
                })
                .collect();
            let mut ordered_positions = Vec::with_capacity(graph.inputs.len());
            let mut selected_positions = vec![false; graph.inputs.len()];
            for key in explicit_input_order {
                if let Some(&position) = input_position_by_key.get(key) {
                    ordered_positions.push(position);
                    selected_positions[position] = true;
                }
            }
            for (position, selected) in selected_positions.iter().enumerate() {
                if !selected {
                    ordered_positions.push(position);
                }
            }
            compiled.input_slots = ordered_positions
                .iter()
                .map(|&position| compiled.input_slots[position])
                .collect();
            descriptors = ordered_positions
                .iter()
                .map(|&position| descriptors[position].clone())
                .collect();
            input_keys = ordered_positions
                .into_iter()
                .map(|position| input_keys[position].clone())
                .collect();
        }

        let semantic =
            compile_materialized_semantic_program(&compiled, &descriptors, &scoped_constraints)?;
        validate_bound_shape_guards(&semantic)?;
        Ok(CompiledGraph::new(
            semantic,
            self.compiler_options,
            input_keys,
        ))
    }
}

fn validate_bound_shape_guards(frozen: &FrozenProgram) -> Result<()> {
    let input_shapes = compile_time_input_shapes(frozen)?;
    for operation in frozen.program.operations() {
        let fallback_family = match operation.op() {
            SemanticOpRef::Core(_) => "tenferro-runtime.core.v1",
            SemanticOpRef::Extension(extension) => extension.family_id(),
        };
        for guard in operation.shape_guards() {
            if guard.source_family().is_some() {
                continue;
            }
            validate_bound_shape_guard(guard, fallback_family, &input_shapes)?;
        }
    }
    Ok(())
}

fn compile_time_input_shapes(frozen: &FrozenProgram) -> Result<Vec<Option<Vec<usize>>>> {
    let metadata = frozen.input_metadata_with_bound_shapes();
    frozen
        .program
        .inputs()
        .iter()
        .enumerate()
        .map(|(input_idx, &input)| {
            if let Some(tensor) = frozen.bindings.tensor_ref_for_input(input) {
                return Ok(Some(tensor.shape().to_vec()));
            }
            let Some(metadata) = metadata.get(input_idx) else {
                return Err(invalid_compiled_graph(format!(
                    "semantic input metadata index {input_idx} is outside metadata table"
                )));
            };
            concrete_shape_from_input_metadata(metadata)
        })
        .collect()
}

fn concrete_shape_from_input_metadata(
    metadata: &ProgramValueMetadata,
) -> Result<Option<Vec<usize>>> {
    let mut shape = Vec::with_capacity(metadata.shape().len());
    for extent in metadata.shape() {
        let ShapeExtent::Exact(expression) = extent else {
            return Ok(None);
        };
        let Some(value) = evaluate_static_dim_expr(expression)? else {
            return Ok(None);
        };
        shape.push(value);
    }
    Ok(Some(shape))
}

fn validate_bound_shape_guard(
    guard: &ProgramShapeGuard,
    fallback_family: &'static str,
    input_shapes: &[Option<Vec<usize>>],
) -> Result<()> {
    let ProgramShapeRelation::Equal = guard.relation() else {
        return Ok(());
    };
    let family = guard.source_family().unwrap_or(fallback_family);
    let relation = ShapeRelation::Equal;
    let lhs = evaluate_bound_shape_guard_expression(family, relation, guard.lhs(), input_shapes)?;
    let rhs = evaluate_bound_shape_guard_expression(family, relation, guard.rhs(), input_shapes)?;
    let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
        return Ok(());
    };
    if lhs == rhs {
        return Ok(());
    }
    Err(Error::ShapeConstraintViolation {
        family,
        instruction_index: None,
        relation,
        lhs_expr: format!("{:?}", guard.lhs()),
        rhs_expr: format!("{:?}", guard.rhs()),
        lhs_value: lhs,
        rhs_value: rhs,
    })
}

fn evaluate_bound_shape_guard_expression(
    family: &'static str,
    relation: ShapeRelation,
    expression: &DimExpr,
    input_shapes: &[Option<Vec<usize>>],
) -> Result<Option<usize>> {
    evaluate_static_dim_expr_with_inputs(expression, input_shapes).map_err(|cause| {
        Error::ShapeConstraintEvaluation {
            family,
            instruction_index: None,
            relation,
            expression: format!("{expression:?}"),
            cause: cause.into(),
        }
    })
}

fn evaluate_static_dim_expr(expression: &DimExpr) -> Result<Option<usize>> {
    evaluate_static_dim_expr_without_inputs(expression).map_err(|cause| {
        Error::ShapeConstraintEvaluation {
            family: "tenferro-runtime.input.v1",
            instruction_index: None,
            relation: ShapeRelation::Equal,
            expression: format!("{expression:?}"),
            cause: cause.into(),
        }
    })
}

fn evaluate_static_dim_expr_without_inputs(
    expression: &DimExpr,
) -> std::result::Result<Option<usize>, DimExprEvalError> {
    match expression {
        DimExpr::Const(value) => Ok(Some(*value)),
        DimExpr::InputDim { .. } => Ok(None),
        DimExpr::Add(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_without_inputs(a)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_without_inputs(b)? else {
                return Ok(None);
            };
            lhs.checked_add(rhs)
                .map(Some)
                .ok_or(DimExprEvalError::AddOverflow { lhs, rhs })
        }
        DimExpr::Sub(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_without_inputs(a)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_without_inputs(b)? else {
                return Ok(None);
            };
            lhs.checked_sub(rhs)
                .map(Some)
                .ok_or(DimExprEvalError::SubUnderflow { lhs, rhs })
        }
        DimExpr::Mul(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_without_inputs(a)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_without_inputs(b)? else {
                return Ok(None);
            };
            lhs.checked_mul(rhs)
                .map(Some)
                .ok_or(DimExprEvalError::MulOverflow { lhs, rhs })
        }
        DimExpr::FloorDiv(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_without_inputs(a)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_without_inputs(b)? else {
                return Ok(None);
            };
            if rhs == 0 {
                return Err(DimExprEvalError::FloorDivByZero { lhs, rhs });
            }
            Ok(Some(lhs / rhs))
        }
        DimExpr::Min(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_without_inputs(a)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_without_inputs(b)? else {
                return Ok(None);
            };
            Ok(Some(lhs.min(rhs)))
        }
        DimExpr::Max(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_without_inputs(a)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_without_inputs(b)? else {
                return Ok(None);
            };
            Ok(Some(lhs.max(rhs)))
        }
    }
}

fn evaluate_static_dim_expr_with_inputs(
    expression: &DimExpr,
    input_shapes: &[Option<Vec<usize>>],
) -> std::result::Result<Option<usize>, DimExprEvalError> {
    match expression {
        DimExpr::Const(value) => Ok(Some(*value)),
        DimExpr::InputDim { input_idx, axis } => match input_shapes.get(*input_idx) {
            Some(Some(shape)) => {
                shape
                    .get(*axis)
                    .copied()
                    .map(Some)
                    .ok_or(DimExprEvalError::AxisOutOfBounds {
                        input_idx: *input_idx,
                        axis: *axis,
                        rank: shape.len(),
                    })
            }
            Some(None) => Ok(None),
            None => Err(DimExprEvalError::InputOutOfBounds {
                input_idx: *input_idx,
                input_count: input_shapes.len(),
            }),
        },
        DimExpr::Add(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_with_inputs(a, input_shapes)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_with_inputs(b, input_shapes)? else {
                return Ok(None);
            };
            lhs.checked_add(rhs)
                .map(Some)
                .ok_or(DimExprEvalError::AddOverflow { lhs, rhs })
        }
        DimExpr::Sub(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_with_inputs(a, input_shapes)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_with_inputs(b, input_shapes)? else {
                return Ok(None);
            };
            lhs.checked_sub(rhs)
                .map(Some)
                .ok_or(DimExprEvalError::SubUnderflow { lhs, rhs })
        }
        DimExpr::Mul(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_with_inputs(a, input_shapes)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_with_inputs(b, input_shapes)? else {
                return Ok(None);
            };
            lhs.checked_mul(rhs)
                .map(Some)
                .ok_or(DimExprEvalError::MulOverflow { lhs, rhs })
        }
        DimExpr::FloorDiv(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_with_inputs(a, input_shapes)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_with_inputs(b, input_shapes)? else {
                return Ok(None);
            };
            if rhs == 0 {
                return Err(DimExprEvalError::FloorDivByZero { lhs, rhs });
            }
            Ok(Some(lhs / rhs))
        }
        DimExpr::Min(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_with_inputs(a, input_shapes)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_with_inputs(b, input_shapes)? else {
                return Ok(None);
            };
            Ok(Some(lhs.min(rhs)))
        }
        DimExpr::Max(a, b) => {
            let Some(lhs) = evaluate_static_dim_expr_with_inputs(a, input_shapes)? else {
                return Ok(None);
            };
            let Some(rhs) = evaluate_static_dim_expr_with_inputs(b, input_shapes)? else {
                return Ok(None);
            };
            Ok(Some(lhs.max(rhs)))
        }
    }
}

fn compile_materialized_semantic_program(
    compiled: &CompiledProgram<StdTensorOp>,
    descriptors: &[InputDescriptor],
    scoped_constraints: &[SlotScopedShapeConstraint],
) -> Result<FrozenProgram> {
    if compiled.input_slots.len() != descriptors.len() {
        return Err(Error::runtime_state(
            "graph_compile_semantic",
            crate::ErrorPhase::Compile,
            "materialized input count does not match semantic descriptors",
        ));
    }

    let mut builder = SemanticProgramBuilder::new();
    let mut values = vec![None; compiled.n_slots];
    let mut slot_shapes = vec![None; compiled.n_slots];
    let mut guard_slot_shapes = vec![None; compiled.n_slots];
    let mut slot_dtypes = vec![None; compiled.n_slots];
    for (input_idx, (&slot, descriptor)) in compiled.input_slots.iter().zip(descriptors).enumerate()
    {
        let Some(value_slot) = values.get_mut(slot) else {
            return Err(invalid_compiled_graph(format!(
                "semantic input slot {slot} is outside slot table of length {}",
                compiled.n_slots
            )));
        };
        let semantic_shape = descriptor.semantic_shape(input_idx);
        let value = builder
            .input(ProgramInputSpec::new(
                descriptor.dtype,
                semantic_shape.clone(),
            ))
            .map_err(semantic_build_error)?;
        if let Some(tensor) = &descriptor.default_tensor {
            builder
                .bind_input(value, Arc::clone(tensor))
                .map_err(semantic_build_error)?;
        }
        *value_slot = Some(value);
        slot_shapes[slot] = Some(semantic_shape);
        guard_slot_shapes[slot] = Some(descriptor.constraint_guard_shape(input_idx));
        slot_dtypes[slot] = Some(descriptor.dtype);
    }

    for instruction in &compiled.instructions {
        let inputs = instruction
            .inputs
            .iter()
            .map(|&slot| {
                values.get(slot).and_then(|value| *value).ok_or_else(|| {
                    invalid_compiled_graph(format!(
                        "semantic operation input slot {slot} is unavailable"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let guard_input_shapes = instruction
            .inputs
            .iter()
            .map(|&slot| {
                guard_slot_shapes
                    .get(slot)
                    .and_then(|shape| shape.as_deref())
                    .ok_or_else(|| {
                        invalid_compiled_graph(format!(
                            "semantic operation guard-shape input slot {slot} is unavailable"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let guard_output_shapes = match &instruction.operation {
            StdTensorOp::Extension(extension) => {
                let input_dtypes = instruction
                    .inputs
                    .iter()
                    .map(|&slot| {
                        slot_dtypes
                            .get(slot)
                            .and_then(|dtype| *dtype)
                            .ok_or_else(|| {
                                invalid_compiled_graph(format!(
                                    "semantic operation dtype input slot {slot} is unavailable"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                infer_extension_output_meta(extension.as_ref(), &input_dtypes, &guard_input_shapes)?
                    .into_iter()
                    .map(|(_dtype, shape)| shape)
                    .collect::<Vec<_>>()
            }
            operation => infer_output_shapes(operation, &guard_input_shapes)?,
        };
        let outputs = match &instruction.operation {
            StdTensorOp::Extension(extension) => builder
                .add_extension(Arc::clone(extension), &inputs)
                .map_err(semantic_build_error)?,
            operation => builder
                .add_op(
                    CoreSemanticOp::try_from(operation).map_err(|source| {
                        Error::runtime_state_source(
                            "graph_compile_semantic",
                            crate::ErrorPhase::Compile,
                            source,
                        )
                    })?,
                    &inputs,
                )
                .map_err(semantic_build_error)?,
        };
        if outputs.len() != instruction.outputs.len() {
            return Err(invalid_compiled_graph(format!(
                "semantic operation produced {} outputs for {} materialized slots",
                outputs.len(),
                instruction.outputs.len()
            )));
        }
        if guard_output_shapes.len() != instruction.outputs.len() {
            return Err(invalid_compiled_graph(format!(
                "semantic operation inferred {} guard shapes for {} materialized slots",
                guard_output_shapes.len(),
                instruction.outputs.len()
            )));
        }
        for ((&slot, &value), guard_shape) in instruction
            .outputs
            .iter()
            .zip(outputs.iter())
            .zip(guard_output_shapes.iter())
        {
            let Some(value_slot) = values.get_mut(slot) else {
                return Err(invalid_compiled_graph(format!(
                    "semantic output slot {slot} is outside slot table of length {}",
                    compiled.n_slots
                )));
            };
            if value_slot.replace(value).is_some() {
                return Err(invalid_compiled_graph(format!(
                    "semantic output slot {slot} has multiple producers"
                )));
            }
            let metadata = builder
                .value_metadata(value)
                .map_err(semantic_build_error)?;
            slot_shapes[slot] = Some(
                metadata
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(axis, extent)| match extent {
                        tenferro_ops::ShapeExtent::Exact(expression)
                        | tenferro_ops::ShapeExtent::UpperBound(expression) => expression.clone(),
                        tenferro_ops::ShapeExtent::Unknown => DimExpr::InputDim {
                            input_idx: slot,
                            axis,
                        },
                    })
                    .collect(),
            );
            guard_slot_shapes[slot] = Some(guard_shape.clone());
            slot_dtypes[slot] = Some(metadata.dtype());
        }
    }

    for scoped in scoped_constraints {
        let target = scoped
            .origin_slots
            .iter()
            .find_map(|&slot| values.get(slot).and_then(|value| *value))
            .ok_or_else(|| {
                invalid_compiled_graph(
                    "semantic shape constraint has no available origin".to_string(),
                )
            })?;
        let lhs = lower_scoped_dim_expr(
            &scoped.local.lhs,
            &scoped.input_slots,
            &guard_slot_shapes,
            &scoped.local,
        )?;
        let rhs = lower_scoped_dim_expr(
            &scoped.local.rhs,
            &scoped.input_slots,
            &guard_slot_shapes,
            &scoped.local,
        )?;
        let relation = match scoped.local.relation {
            tenferro_ops::ShapeRelation::Equal => ProgramShapeRelation::Equal,
        };
        let lowered = LocalShapeConstraint {
            source: scoped.local.source.clone(),
            relation: scoped.local.relation,
            lhs,
            rhs,
        };
        let retained_guards = discharge(vec![lowered])?;
        if retained_guards.is_empty() {
            continue;
        }
        let guards = retained_guards.into_iter().map(|guard| {
            ProgramShapeGuard::new(relation, guard.lhs, guard.rhs)
                .with_source_family(guard.source.family_id)
        });
        builder
            .add_shape_guards_to_output(target, guards)
            .map_err(semantic_build_error)?;
    }

    let outputs = compiled
        .output_slots
        .iter()
        .map(|&slot| {
            values.get(slot).and_then(|value| *value).ok_or_else(|| {
                invalid_compiled_graph(format!(
                    "semantic program output slot {slot} is unavailable"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    builder.finish(&outputs).map_err(|source| {
        Error::runtime_state_source("graph_compile_semantic", crate::ErrorPhase::Compile, source)
    })
}

fn semantic_build_error(source: crate::program::ProgramBuildError) -> Error {
    Error::runtime_state_source("graph_compile_semantic", crate::ErrorPhase::Compile, source)
}

impl Default for GraphCompiler {
    fn default() -> Self {
        Self::new()
    }
}

fn collect_default_inputs(
    outputs: &[&TracedTensor],
) -> Result<HashMap<TensorInputKey, Arc<Tensor>>> {
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
    Ok(all_inputs)
}

fn insert_checkpoint_alias(
    aliases: &mut HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
    alias_key: TensorInputKey,
    target_key: ValueKey<StdTensorOp>,
) -> Result<()> {
    if let Some(existing) = aliases.get(&alias_key) {
        if existing != &target_key {
            return Err(Error::Internal(format!(
                "checkpoint alias {alias_key:?} targets both {existing:?} and {target_key:?}"
            )));
        }
        return Ok(());
    }
    aliases.insert(alias_key, target_key);
    Ok(())
}

struct AliasAwareMaterializer<'a> {
    view: &'a ResolvedView<StdTensorOp>,
    aliases: &'a HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
    alias_shapes: &'a HashMap<TensorInputKey, Vec<usize>>,
    val_map: HashMap<ValueKey<StdTensorOp>, usize>,
    op_map: HashMap<Arc<OperationKey<StdTensorOp>>, usize>,
    values: Vec<MaterializedValue<StdTensorOp>>,
    operations: Vec<MaterializedOperation<StdTensorOp>>,
    input_keys: Vec<ValueKey<StdTensorOp>>,
}

impl<'a> AliasAwareMaterializer<'a> {
    fn new(
        view: &'a ResolvedView<StdTensorOp>,
        aliases: &'a HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
        alias_shapes: &'a HashMap<TensorInputKey, Vec<usize>>,
    ) -> Self {
        Self {
            view,
            aliases,
            alias_shapes,
            val_map: HashMap::new(),
            op_map: HashMap::new(),
            values: Vec::new(),
            operations: Vec::new(),
            input_keys: Vec::new(),
        }
    }

    fn visit(&mut self, key: &ValueKey<StdTensorOp>) -> usize {
        if let Some(&index) = self.val_map.get(key) {
            return index;
        }
        if let Some(target) = self.alias_target(key).cloned() {
            let index = self.visit(&target);
            let index = self.refine_alias_to_checkpoint_shape(key, index);
            self.val_map.insert(key.clone(), index);
            return index;
        }

        let resolved = self.view.resolve_value(key);
        assert!(
            resolved.is_some(),
            "key not found in resolved view: {:?}",
            key
        );
        match resolved {
            Some(ValueDef::Input { .. }) => self.materialize_input(key),
            Some(ValueDef::Produced {
                operation,
                input_keys,
                role,
                output_slot,
            }) => self.materialize_produced(operation, input_keys, role, output_slot),
            None => unreachable!("asserted above"),
        }
    }

    fn alias_target(&self, key: &ValueKey<StdTensorOp>) -> Option<&ValueKey<StdTensorOp>> {
        let ValueKey::Input(input_key) = key else {
            return None;
        };
        self.aliases.get(input_key)
    }

    fn refine_alias_to_checkpoint_shape(
        &mut self,
        key: &ValueKey<StdTensorOp>,
        target_index: usize,
    ) -> usize {
        let ValueKey::Input(input_key) = key else {
            return target_index;
        };
        let Some(shape) = self.alias_shapes.get(input_key) else {
            return target_index;
        };
        let target_key = self.values[target_index].key.clone();
        let rank = shape.len();
        self.materialize_produced(
            StdTensorOp::Slice(SliceConfig {
                starts: vec![0; rank],
                limits: shape.clone(),
                strides: vec![1; rank],
            }),
            vec![target_key],
            computegraph::types::OperationRole::Primary,
            0,
        )
    }

    fn materialize_input(&mut self, key: &ValueKey<StdTensorOp>) -> usize {
        let index = self.values.len();
        self.values.push(MaterializedValue {
            key: key.clone(),
            producer: None,
        });
        self.val_map.insert(key.clone(), index);
        self.input_keys.push(key.clone());
        index
    }

    fn materialize_produced(
        &mut self,
        operation: StdTensorOp,
        input_keys: Vec<ValueKey<StdTensorOp>>,
        role: computegraph::types::OperationRole,
        output_slot: usize,
    ) -> usize {
        let op_key = Arc::new(OperationKey::new(
            operation.clone(),
            input_keys.clone(),
            role.clone(),
        ));

        if self.op_map.contains_key(&op_key) {
            let output_key = ValueKey::Derived {
                operation: op_key,
                output_slot: output_slot as u8,
            };
            let val_index = self.val_map.get(&output_key).copied();
            assert!(
                val_index.is_some(),
                "materialized op {:?} is missing output slot {}",
                operation,
                output_slot
            );
            return match val_index {
                Some(index) => index,
                None => unreachable!("asserted above"),
            };
        }

        let materialized_inputs = input_keys.iter().map(|input| self.visit(input)).collect();
        let op_index = self.operations.len();
        self.op_map.insert(Arc::clone(&op_key), op_index);
        self.operations.push(MaterializedOperation {
            operation: operation.clone(),
            inputs: materialized_inputs,
            outputs: Vec::with_capacity(operation.output_count()),
            role,
        });

        for slot in 0..operation.output_count() {
            let output_key = ValueKey::Derived {
                operation: Arc::clone(&op_key),
                output_slot: slot as u8,
            };
            let val_index = self.values.len();
            self.values.push(MaterializedValue {
                key: output_key.clone(),
                producer: Some((op_index, slot)),
            });
            self.val_map.insert(output_key, val_index);
            self.operations[op_index].outputs.push(val_index);
        }

        self.operations[op_index].outputs[output_slot]
    }
}

fn materialize_merge_with_input_aliases(
    view: &ResolvedView<StdTensorOp>,
    outputs: &[ValueKey<StdTensorOp>],
    aliases: &HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
    alias_shapes: &HashMap<TensorInputKey, Vec<usize>>,
) -> MaterializedGraph<StdTensorOp> {
    let mut materializer = AliasAwareMaterializer::new(view, aliases, alias_shapes);
    let mut materialized_outputs = Vec::with_capacity(outputs.len());

    for output in outputs {
        let output_slot = materializer.visit(output);
        materialized_outputs.push(materializer.values[output_slot].key.clone());
    }

    MaterializedGraph {
        values: materializer.values,
        operations: materializer.operations,
        inputs: materializer.input_keys,
        outputs: materialized_outputs,
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
    allow_unbound_placeholders: bool,
) -> Result<InputDescriptor> {
    if let Some(tensor) = default_inputs.get(key) {
        return Ok(InputDescriptor {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            extent_identity: default_input_extent_identity(key, tensor)?,
            default_tensor: Some(tensor.clone()),
        });
    }
    if let Some(spec) = binding_specs.get(key) {
        return Ok(spec.clone());
    }
    if allow_unbound_placeholders {
        return descriptor_for_unbound_input(key);
    }
    Err(Error::UnboundPlaceholder {
        input_key: format!("{:?}", key),
    })
}

fn descriptor_for_unbound_input(key: &TensorInputKey) -> Result<InputDescriptor> {
    let metadata = registered_meta(&ValueKey::Input(key.clone()))?;
    if let Some(shape) = metadata
        .exact_shape()
        .as_deref()
        .and_then(concrete_shape_from_sym_dims)
    {
        return Ok(InputDescriptor {
            dtype: metadata.dtype,
            shape,
            extent_identity: InputExtentIdentity::Concrete,
            default_tensor: None,
        });
    }
    Ok(InputDescriptor {
        dtype: metadata.dtype,
        shape: vec![0; metadata.rank()],
        extent_identity: InputExtentIdentity::Symbolic,
        default_tensor: None,
    })
}

fn concrete_shape_from_sym_dims(shape: &[SymDim]) -> Option<Vec<usize>> {
    shape.iter().map(SymDim::constant_value).collect()
}

fn default_input_extent_identity(
    key: &TensorInputKey,
    tensor: &Tensor,
) -> Result<InputExtentIdentity> {
    let metadata = registered_meta(&ValueKey::Input(key.clone()))?;
    let exact_shape = metadata.exact_shape();
    if metadata.dtype == tensor.dtype()
        && exact_shape_matches_tensor_shape(exact_shape.as_deref(), tensor.shape())
    {
        Ok(InputExtentIdentity::Concrete)
    } else {
        Ok(InputExtentIdentity::Symbolic)
    }
}

fn exact_shape_matches_tensor_shape(
    shape: Option<&[tenferro_ops::SymDim]>,
    tensor_shape: &[usize],
) -> bool {
    let Some(shape) = shape else {
        return false;
    };
    shape.len() == tensor_shape.len()
        && shape
            .iter()
            .zip(tensor_shape)
            .all(|(dim, &extent)| dim.constant_value() == Some(extent))
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
    use std::any::Any;
    use std::hash::Hasher;
    use std::sync::Arc;
    use tenferro_ops::{
        ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp},
        SymDim,
    };
    use tenferro_tensor::{
        BackendStorageHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
        StorageBuffer, TypedTensor,
    };

    #[test]
    fn compile_publishes_semantic_program_and_separate_default_bindings() {
        let input = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let output = input.neg().unwrap();

        let program = GraphCompiler::new().compile(&output).unwrap();

        assert_eq!(program.program().inputs().len(), 1);
        assert_eq!(program.program().outputs().len(), 1);
        assert_eq!(program.program().operations().count(), 1);
        assert_eq!(program.bindings().len(), 1);
        assert_eq!(
            program
                .program()
                .value_metadata(program.program().inputs()[0])
                .unwrap()
                .shape(),
            &[ShapeExtent::Exact(DimExpr::Const(2))]
        );
        assert_eq!(
            program
                .program()
                .value_metadata(program.program().outputs()[0])
                .unwrap()
                .dtype(),
            DType::F64
        );
    }

    #[test]
    fn compile_preserves_symbolic_default_input_extent_identity() {
        let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let input = TracedTensor::from_tensor_symbolic_shape(tensor).unwrap();
        let output = input.neg().unwrap();

        let program = GraphCompiler::new().compile(&output).unwrap();

        assert!(matches!(
            program
                .program()
                .value_metadata(program.program().inputs()[0])
                .unwrap()
                .shape(),
            [ShapeExtent::Exact(DimExpr::InputDim {
                input_idx: 0,
                axis: 0
            })]
        ));
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
            cpu_affinity: None,
        };
        let lhs = Arc::new(Tensor::F64(
            TypedTensor::from_buffer_col_major(
                vec![2],
                StorageBuffer::Backend(Box::new(BackendStorageHandle::<f64>::new_with_len(1, 2))),
                placement.clone(),
            )
            .unwrap(),
        ));
        let rhs = Arc::new(Tensor::F64(
            TypedTensor::from_buffer_col_major(
                vec![2],
                StorageBuffer::Backend(Box::new(BackendStorageHandle::<f64>::new_with_len(2, 2))),
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

    #[test]
    fn compile_frozen_program_rejects_static_unbound_shape_guard_mismatch() {
        let mut builder = SemanticProgramBuilder::new();
        let lhs = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let rhs = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
            .unwrap();
        let output = builder.add_op(CoreSemanticOp::Neg, &[lhs]).unwrap()[0];
        builder
            .add_shape_guards_to_output(
                output,
                [ProgramShapeGuard::new(
                    ProgramShapeRelation::Equal,
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    },
                )],
            )
            .unwrap();
        let frozen = builder.finish(&[output]).unwrap();

        let err = GraphCompiler::new()
            .compile_frozen_program(&frozen)
            .unwrap_err();

        assert!(matches!(
            err,
            Error::ShapeConstraintViolation {
                lhs_value: 2,
                rhs_value: 3,
                ..
            }
        ));
        let _ = rhs;
    }

    #[test]
    fn compile_frozen_program_defers_dynamic_unbound_shape_guard() {
        let mut builder = SemanticProgramBuilder::new();
        let lhs = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let rhs = builder
            .input(ProgramInputSpec::from_metadata(
                ProgramValueMetadata::from_extents(DType::F64, [ShapeExtent::Unknown]),
            ))
            .unwrap();
        let output = builder.add_op(CoreSemanticOp::Neg, &[lhs]).unwrap()[0];
        builder
            .add_shape_guards_to_output(
                output,
                [ProgramShapeGuard::new(
                    ProgramShapeRelation::Equal,
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    },
                )],
            )
            .unwrap();
        let frozen = builder.finish(&[output]).unwrap();

        GraphCompiler::new()
            .compile_frozen_program(&frozen)
            .unwrap();
        let _ = rhs;
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

        fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
            ExtensionEffectDeclaration::Declared(&[])
        }

        fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
            ExtensionAliasDeclaration::AllFresh
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
        let staging =
            stage_semantic_program(program.program(), program.compiler_options()).unwrap();
        let pruned_instruction = staging
            .instructions
            .iter()
            .find_map(|inst| match &inst.op {
                crate::exec::ExecOp::Extension(op)
                    if op.family_id() == "tenferro-runtime.test-prunable.v1" =>
                {
                    Some((
                        inst.output_slots.clone(),
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

    #[test]
    fn compiled_graph_input_keys_preserve_order_for_binary_graph() {
        let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
        let a_key = a.input_key().expect("concrete traced tensor has input key");
        let b_key = b.input_key().expect("concrete traced tensor has input key");
        let c = a.mul(&b).unwrap();

        let program = GraphCompiler::new().compile_many(&[&c]).unwrap();

        assert_eq!(program.input_count(), 2);
        assert_eq!(program.input_keys().len(), 2);
        let keys: Vec<_> = program.input_keys().to_vec();
        assert!(keys.contains(&a_key), "input_keys must contain a's key");
        assert!(keys.contains(&b_key), "input_keys must contain b's key");
        // input_key_index maps each key to its position
        assert_eq!(
            program.input_key_index(&a_key),
            keys.iter().position(|k| k == &a_key)
        );
        assert_eq!(
            program.input_key_index(&b_key),
            keys.iter().position(|k| k == &b_key)
        );
    }

    #[test]
    fn compile_with_input_specs_reorders_inputs_by_explicit_order() {
        let a = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
        let b = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
        let a_key = a.input_key().expect("symbolic traced tensor has input key");
        let b_key = b.input_key().expect("symbolic traced tensor has input key");
        let c = a.mul(&b).unwrap();

        // Request b first, then a
        let program = GraphCompiler::new()
            .compile_with_input_specs(&c, &[(&b, DType::F64, &[2]), (&a, DType::F64, &[2])])
            .unwrap();

        assert_eq!(program.input_count(), 2);
        assert_eq!(program.input_keys().len(), 2);
        // b must be at position 0 per the explicit order
        assert_eq!(
            program.input_key_index(&b_key),
            Some(0),
            "b must be first in explicit order"
        );
        assert_eq!(
            program.input_key_index(&a_key),
            Some(1),
            "a must be second in explicit order"
        );
    }
}
