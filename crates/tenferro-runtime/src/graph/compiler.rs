use std::fmt;

use tenferro_ops::dim_expr::{DimExpr, DimExprEvalError};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::{ShapeExtent, ShapeRelation};
#[cfg(test)]
use tenferro_tensor::Tensor;
use tenferro_tensor::{CacheStats, DType};

use super::program::CompiledGraph;
#[cfg(test)]
use crate::compiler::semantic_staging::stage_semantic_program;
use crate::compiler::CompilerOptions;
use crate::error::{Error, ErrorPhase, Result};
use crate::extension_cache::{ExtensionCacheSelector, ExtensionCacheStore};
use crate::program::{
    FrozenProgram, ProgramShapeRelation, ProgramValueMetadata, SemanticOpRef,
    ShapeGuard as ProgramShapeGuard,
};
use crate::trace::TracedGraph;
use crate::traced::{try_concrete_shape, TracedTensor};

/// Compiler for traced tensor graphs.
///
/// A graph compiler lowers one or more [`TracedTensor`] outputs to a reusable
/// [`CompiledGraph`] without requiring a backend.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{TraceContext, GraphCompiler, TracedTensor};
///
/// let trace = TraceContext::new();
/// let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
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

impl Default for GraphCompiler {
    fn default() -> Self {
        Self::new()
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
    /// use tenferro_runtime::{CompilerOptions, GraphCompiler, OptimizerConfig};
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

    /// Compile one traced output by snapshotting its context.
    ///
    /// The output's reachable closure is frozen into one semantic program; the
    /// compile leaves the context usable for further tracing.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TraceContext};
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.input_concrete_shape(tenferro_tensor::DType::F64, &[1])?;
    /// let y = x.neg()?;
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y)?;
    /// assert_eq!(program.output_count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid snapshot or bound shape
    /// guard, or [`Error::RuntimeState`] for inconsistent compiler state.
    pub fn compile(&mut self, output: &TracedTensor) -> Result<CompiledGraph> {
        self.compile_many(&[output])
    }

    /// Compile an immutable semantic trace without consulting a backend.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for invalid metadata or shape constraints,
    /// or [`Error::RuntimeState`] for inconsistent staging state.
    pub fn compile_traced_graph(&mut self, graph: &TracedGraph) -> Result<CompiledGraph> {
        self.compile_frozen(graph.frozen(), Vec::new())
    }

    /// Compile an immutable semantic program for ordered execution.
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
    /// or [`Error::RuntimeState`] for inconsistent staging state.
    pub fn compile_frozen_program(&mut self, frozen: &FrozenProgram) -> Result<CompiledGraph> {
        self.compile_frozen(frozen, Vec::new())
    }

    fn compile_frozen(
        &mut self,
        frozen: &FrozenProgram,
        input_keys: Vec<TensorInputKey>,
    ) -> Result<CompiledGraph> {
        validate_bound_shape_guards(frozen)?;
        Ok(CompiledGraph::new(
            frozen.clone(),
            self.compiler_options,
            input_keys,
        ))
    }

    /// Compile multiple traced outputs into one program.
    ///
    /// Every output must belong to the same trace context.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `ValidationError::InvalidArgument` when
    /// the output list is empty (`argument: "outputs"`) or when an output belongs
    /// to a different trace context, and the typed snapshot error from
    /// [`compile`](Self::compile) when the program cannot be frozen.
    pub fn compile_many(&mut self, outputs: &[&TracedTensor]) -> Result<CompiledGraph> {
        let Some(first) = outputs.first() else {
            return Err(Error::invalid_argument(
                "GraphCompiler::compile_many",
                ErrorPhase::Compile,
                "outputs",
                "at least one output is required",
            ));
        };
        for output in outputs.iter().skip(1) {
            crate::traced::ensure_same_context(first, output)?;
        }
        let values: Vec<_> = outputs.iter().map(|output| output.value).collect();
        let (frozen, kept) = first
            .ctx
            .plain_snapshot_with_inputs(&values)
            .map_err(graph_error_with_context("GraphCompiler::compile_many"))?;
        let input_keys = kept.into_iter().map(|(key, _)| key).collect();
        self.compile_frozen(&frozen, input_keys)
    }

    /// Compile one traced output with concrete placeholder specs.
    ///
    /// Every declared placeholder must be a data-less input of the output's
    /// context, with the declared dtype and shape. The compiled program keeps the
    /// context's input order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnexpectedBinding`] for a data-carrying tensor,
    /// [`Error::DuplicateBinding`] for repeated placeholders,
    /// [`Error::PlaceholderDtypeMismatch`],
    /// [`Error::PlaceholderShapeMismatch`], or
    /// [`Error::PlaceholderRankMismatch`] for incompatible specs,
    /// [`Error::Validation`] with `InvalidArgument` for a declared placeholder the
    /// output does not depend on, and the snapshot failures of
    /// [`compile`](Self::compile).
    pub fn compile_with_input_specs(
        &mut self,
        output: &TracedTensor,
        bindings: &[(&TracedTensor, DType, &[usize])],
    ) -> Result<CompiledGraph> {
        let mut seen = std::collections::HashSet::new();
        let mut specs = Vec::with_capacity(bindings.len());
        for (index, (placeholder, dtype, shape)) in bindings.iter().enumerate() {
            validate_placeholder_spec(index, placeholder, *dtype, shape)?;
            let key = placeholder.input_key().ok_or(Error::UnexpectedBinding {
                binding_index: index,
            })?;
            if !seen.insert(key.clone()) {
                return Err(Error::DuplicateBinding {
                    input_key: format!("{key:?}"),
                });
            }
            specs.push((
                key,
                crate::program::ProgramInputSpec::new(
                    *dtype,
                    shape
                        .iter()
                        .map(|&size| tenferro_ops::dim_expr::DimExpr::Const(size)),
                ),
            ));
        }
        let (frozen, kept) = output
            .ctx
            .snapshot_with_input_specs(&[output.value], &specs)
            .map_err(graph_error_with_context(
                "GraphCompiler::compile_with_input_specs",
            ))?;
        let input_keys = kept.into_iter().map(|(key, _)| key).collect();
        let program = self.compile_frozen(&frozen, input_keys)?;
        if let Some(binding_index) = bindings.iter().position(|(placeholder, _, _)| {
            placeholder
                .input_key()
                .is_none_or(|key| program.input_key_index(&key).is_none())
        }) {
            return Err(Error::invalid_argument(
                "GraphCompiler::compile_with_input_specs",
                ErrorPhase::Compile,
                "bindings",
                format!(
                    "binding {binding_index} declares a placeholder the output does not \
                     depend on; remove it from the bindings"
                ),
            ));
        }
        Ok(program)
    }

    /// Return the compiler options used for future lowerings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CompilerOptions, GraphCompiler};
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compiler_options(), CompilerOptions::default());
    /// ```
    pub fn compiler_options(&self) -> CompilerOptions {
        self.compiler_options
    }

    /// Replace compiler options and clear compiler-owned extension cache entries.
    pub fn set_compiler_options(&mut self, compiler_options: CompilerOptions) {
        self.compiler_options = compiler_options;
        self.clear_caches();
    }

    /// Drop every extension cache entry owned by this compiler.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_extension_caches();
    /// assert!(compiler.extension_caches().is_empty());
    /// ```
    pub fn clear_extension_caches(&mut self) {
        self.extension_cache.clear();
    }

    /// Drop every cache owned by this compiler.
    pub fn clear_caches(&mut self) {
        self.clear_extension_caches();
    }

    /// Return aggregate cache statistics for this compiler.
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

    /// Borrow the extension cache store owned by this compiler.
    pub fn extension_caches(&self) -> &ExtensionCacheStore {
        &self.extension_cache
    }

    /// Borrow the extension cache store mutably.
    pub fn extension_caches_mut(&mut self) -> &mut ExtensionCacheStore {
        &mut self.extension_cache
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

fn invalid_compiled_graph(message: impl Into<String>) -> Error {
    Error::Internal(message.into())
}

fn graph_error_with_context(
    op: &'static str,
) -> impl FnOnce(crate::program::ProgramFinishError) -> Error {
    move |source| Error::runtime_state_source(op, ErrorPhase::Compile, source)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{CoreSemanticOp, ProgramInputSpec, SemanticProgramBuilder};
    use std::any::Any;
    use std::hash::Hasher;
    use std::sync::Arc;
    use tenferro_ops::{
        ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp},
        SymDim,
    };

    #[test]
    fn compile_publishes_semantic_program_and_separate_default_bindings() {
        let trace = crate::TraceContext::new();
        let input = trace
            .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
            .unwrap();
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
    fn compiler_options_constructor_keeps_the_given_options() {
        let options = CompilerOptions {
            optimizer: crate::OptimizerConfig {
                dot_decomposer: true,
                ..crate::OptimizerConfig::default()
            },
        };
        let compiler = GraphCompiler::with_compiler_options(options);
        assert!(compiler.compiler_options().optimizer.dot_decomposer);
        assert!(compiler.extension_caches().is_empty());
    }

    #[test]
    fn compile_many_refuses_outputs_from_two_contexts() {
        let first = crate::TraceContext::new();
        let second = crate::TraceContext::new();
        let left = first
            .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
            .unwrap()
            .neg()
            .unwrap();
        let right = second
            .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
            .unwrap()
            .neg()
            .unwrap();

        let error = GraphCompiler::new()
            .compile_many(&[&left, &right])
            .unwrap_err();
        assert!(error.to_string().contains("different contexts"), "{error}");
    }

    #[test]
    fn compile_preserves_symbolic_default_input_extent_identity() {
        let trace = crate::TraceContext::new();
        let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let input = trace.from_tensor_symbolic_shape(tensor).unwrap();
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
        let trace = crate::TraceContext::new();
        let input = trace
            .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
            .unwrap();
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
        let trace = crate::TraceContext::new();
        let a = trace
            .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
            .unwrap();
        let b = trace
            .from_vec_col_major(vec![2], vec![3.0_f64, 4.0])
            .unwrap();
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
}
