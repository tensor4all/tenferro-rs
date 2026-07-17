//! Public surface for out-of-tree extension primitives.
//!
//! This module exposes the Stage 6 `ExtensionOp` mechanism through the
//! runtime crate. External crates implement
//! [`tenferro_ops::ext_op::ExtensionOp`] and build traced graphs containing
//! the extension via [`apply`].
//!
//! See `docs/spec/extension-op.md` for the normative contract.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_runtime::extension::{apply, ExtensionOp};
//!
//! // Construct an `Arc<dyn ExtensionOp>` and call `apply(op, &[input])`
//! // to lower it into a `TracedTensor`.
//! ```

use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueRef};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::SymDim;
use tenferro_tensor::{Tensor, TensorBackend};

use crate::checkpoint::CheckpointNode;
use crate::error::{Error, ErrorPhase, Result};
use crate::metadata::{
    register_scoped_graph_analysis, registered_meta, MetadataScopeChain, RegisteredGraphAnalysis,
};
use crate::shape_constraint::{ConstraintScopeChain, ScopedShapeConstraint, ShapeConstraintScope};
use crate::shape_infer::{infer_extension_output_meta_with_constraints, InferredExtensionMeta};
use crate::traced::{merge_traced_inputs_map, next_traced_id, TracedTensor};

type ExpandedOutputMetas = Vec<(tenferro_tensor::DType, Vec<SymDim>)>;

pub use crate::compiler::CompilerOptions;
#[doc(hidden)]
pub use crate::compiler::{compile_std_to_exec, compile_std_to_exec_with_options};
#[doc(hidden)]
pub use crate::exec::{ExecInstruction, ExecOp, ExecOutputExtents, ExecOutputShapes, ExecProgram};
#[doc(hidden)]
pub use crate::shape_infer::{
    infer_output_dtype, infer_output_extents, infer_output_shapes, promote_dtype,
    promote_dtype_div_like, promote_dtype_for_binary_op, promote_dtypes,
};
pub use tenferro_ops::ext_op::{ExtensionOp, HostReference};
pub use tenferro_ops::ExtensionFamilyId;

pub use crate::extension_cache::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
pub use crate::extension_runtime::{
    ExtensionExecutionContext, ExtensionExecutor, ExtensionRegistry, ExtensionRuntime,
    ExtensionRuntimeRegistryError, HostReferenceRuntime,
};

/// Execute a lowered core program with caller-owned backend runtime cache state.
///
/// This owner-scoped hook is for operation-family runtimes that expand an
/// extension into core tensor operations and need to run that lowered program
/// while preserving the runtime cache owned by the outer graph executor.
#[doc(hidden)]
pub fn execute_lowered_program_with_backend_cache<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    backend_cache: &mut B::RuntimeCache,
) -> Result<Vec<Tensor>> {
    crate::exec::ensure_core_exec_program(
        program,
        "extension::execute_lowered_program_with_backend_cache",
    )?;
    crate::exec::eval_exec_ir_with_backend_cache(backend, program, inputs, backend_cache)
}

/// Apply an extension op in the traced graph.
///
/// The `op` value is cloned into a `StdTensorOp::Extension(Arc<dyn ExtensionOp>)`
/// carrier. The returned vector contains one [`TracedTensor`] per declared
/// output slot of the extension. Output shapes are inferred via
/// [`ExtensionOp::infer_output_meta`] using the input shape hints.
///
/// `inputs.len()` must equal `op.input_count()`, and each input's
/// `shape_hint` must be present (i.e. the extension must be used on
/// tensors whose rank is known at graph-build time). For symbolic-shape
/// composition, bind the placeholder tensors via
/// [`crate::GraphExecutor::run_with_inputs`] at evaluation time.
///
/// # Examples
///
/// ```rust
/// # use std::any::Any;
/// use std::sync::Arc;
/// use tenferro_runtime::extension::{apply, ExtensionOp};
/// use tenferro_runtime::{DType, SymDim, TracedTensor};
/// use tenferro_ops::ExtensionShapeContext;
///
/// # #[derive(Clone, Debug)]
/// # struct IdentityExt;
/// # impl ExtensionOp for IdentityExt {
/// #     fn family_id(&self) -> &'static str { "example.identity.v1" }
/// #     fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
/// #     fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
/// #         other.as_any().downcast_ref::<IdentityExt>().is_some()
/// #     }
/// #     fn clone_arc(&self) -> Arc<dyn ExtensionOp> { Arc::new(self.clone()) }
/// #     fn as_any(&self) -> &dyn Any { self }
/// #     fn input_count(&self) -> usize { 1 }
/// #     fn output_count(&self) -> usize { 1 }
/// #     fn infer_output_meta(
/// #         &self,
/// #         ctx: &mut ExtensionShapeContext<'_>,
/// #     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
/// #         Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
/// #     }
/// # }
/// let op: Arc<dyn ExtensionOp> = Arc::new(IdentityExt);
/// let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let outputs = apply(op, &[&a])?;
/// assert_eq!(outputs.len(), 1);
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns a graph-build validation error when the extension receives the
/// wrong number of traced inputs. Canonical metadata inference failures,
/// including a returned metadata count that differs from
/// [`ExtensionOp::output_count`], are returned as [`Error::TensorRuntime`]
/// containing the typed tensor validation source.
pub fn apply(op: Arc<dyn ExtensionOp>, inputs: &[&TracedTensor]) -> Result<Vec<TracedTensor>> {
    if inputs.len() != op.input_count() {
        return Err(Error::invalid_argument(
            "extension::apply",
            ErrorPhase::GraphBuild,
            "inputs",
            format!(
                "op family {:?} expects {} inputs, got {}",
                op.family_id(),
                op.input_count(),
                inputs.len()
            ),
        ));
    }

    // Build the carrier graph first; the graph-analysis pass below resolves
    // registered input metadata and invokes canonical extension inference once.
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    for input in inputs {
        builder.add_parent(input.graph.clone());
    }
    let op_inputs: Vec<ValueRef<StdTensorOp>> = inputs
        .iter()
        .map(|t| ValueRef::External(t.graph.values()[t.val].key.clone()))
        .collect();
    let carrier = StdTensorOp::Extension(op.clone());
    let outputs = builder.add_operation(carrier, op_inputs, OperationRole::Primary);
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    let analysis = register_scoped_graph_analysis(graph.as_ref(), std::iter::empty())?;
    let output_metas = outputs
        .iter()
        .map(|&output| {
            let meta = registered_meta(&graph.values()[output].key)?;
            let shape = meta.bound_shape().ok_or_else(|| {
                Error::invalid_argument(
                    "extension::apply",
                    ErrorPhase::Compile,
                    "output_metadata",
                    format!(
                        "extension family {:?} produced unknown output shape metadata",
                        op.family_id()
                    ),
                )
            })?;
            Ok((meta.dtype, shape))
        })
        .collect::<Result<Vec<_>>>()?;
    traced_outputs_from_analysis(inputs, graph, &outputs, output_metas, analysis)
}

/// Attach an extension's inferred shape contract to an equivalent expanded output.
///
/// Standard extension crates use this when a traced fast path lowers an extension
/// directly to core operations. The extension remains the single source of truth
/// for metadata equalities while the executable graph keeps the core-operation
/// fast path.
///
/// # Examples
///
/// ```rust
/// # use std::{any::Any, sync::Arc};
/// use tenferro_ops::ExtensionShapeContext;
/// use tenferro_runtime::extension::{attach_expanded_shape_contract, ExtensionOp};
/// use tenferro_runtime::{DType, SymDim, TracedTensor};
///
/// # #[derive(Clone, Debug)]
/// # struct SameShapeAdd;
/// # impl ExtensionOp for SameShapeAdd {
/// #     fn family_id(&self) -> &'static str { "example.same-shape-add.v1" }
/// #     fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
/// #     fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
/// #         other.as_any().downcast_ref::<Self>().is_some()
/// #     }
/// #     fn clone_arc(&self) -> Arc<dyn ExtensionOp> { Arc::new(self.clone()) }
/// #     fn as_any(&self) -> &dyn Any { self }
/// #     fn input_count(&self) -> usize { 2 }
/// #     fn output_count(&self) -> usize { 1 }
/// #     fn infer_output_meta(
/// #         &self,
/// #         ctx: &mut ExtensionShapeContext<'_>,
/// #     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
/// #         ctx.require_same_shape(0, 1)?;
/// #         Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
/// #     }
/// # }
/// let lhs = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
/// let rhs = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
/// let expanded = (&lhs + &rhs)?;
/// let output = attach_expanded_shape_contract(&SameShapeAdd, &[&lhs, &rhs], expanded)?;
/// assert_eq!(output.rank, 1);
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
#[doc(hidden)]
pub fn attach_expanded_shape_contract(
    op: &dyn ExtensionOp,
    inputs: &[&TracedTensor],
    output: TracedTensor,
) -> Result<TracedTensor> {
    if op.output_count() != 1 {
        return Err(Error::invalid_argument(
            "extension::attach_expanded_shape_contract",
            ErrorPhase::GraphBuild,
            "outputs",
            format!(
                "extension family {:?} contract expects {} outputs, got one expanded output",
                op.family_id(),
                op.output_count(),
            ),
        ));
    }
    let (_, inferred) = infer_expanded_shape_contract(op, inputs)?;
    attach_inferred_expanded_shape_contract(inputs, vec![output], inferred)?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("expanded shape contract returned no output".into()))
}

fn infer_expanded_shape_contract(
    op: &dyn ExtensionOp,
    inputs: &[&TracedTensor],
) -> Result<(ExpandedOutputMetas, InferredExtensionMeta)> {
    if inputs.len() != op.input_count() {
        return Err(Error::invalid_argument(
            "extension::infer_expanded_shape_contract",
            ErrorPhase::GraphBuild,
            "inputs",
            format!(
                "extension family {:?} contract expects {} inputs, got {}",
                op.family_id(),
                op.input_count(),
                inputs.len()
            ),
        ));
    }
    let input_dtypes: Vec<_> = inputs.iter().map(|input| input.dtype).collect();
    let input_shapes: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(input_idx, input)| DimExpr::input_shape(input_idx, input.rank))
        .collect();
    let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
    let inferred =
        infer_extension_output_meta_with_constraints(op, &input_dtypes, &input_shape_refs)?;
    let input_sym_shapes = inputs
        .iter()
        .map(|input| {
            (0..input.rank)
                .map(|axis| input.axis_sym_dim(axis))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let input_sym_shape_refs = input_sym_shapes
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let output_metas = inferred
        .output_metas
        .iter()
        .map(|(dtype, shape)| {
            (
                *dtype,
                shape
                    .iter()
                    .map(|dim| SymDim::from_dim_expr(dim, &input_sym_shape_refs))
                    .collect(),
            )
        })
        .collect();
    Ok((output_metas, inferred))
}

fn attach_inferred_expanded_shape_contract(
    inputs: &[&TracedTensor],
    mut outputs: Vec<TracedTensor>,
    inferred: InferredExtensionMeta,
) -> Result<Vec<TracedTensor>> {
    if inferred.output_metas.len() != outputs.len() {
        return Err(Error::invalid_argument(
            "extension::attach_expanded_shape_contract",
            ErrorPhase::GraphBuild,
            "outputs",
            format!(
                "extension contract inferred {} outputs, but expanded graph produced {}",
                inferred.output_metas.len(),
                outputs.len()
            ),
        ));
    }
    for (output, (dtype, local_shape)) in outputs.iter().zip(inferred.output_metas.iter()) {
        if output.dtype != *dtype || output.rank != local_shape.len() {
            return Err(Error::invalid_argument(
                "extension::attach_expanded_shape_contract",
                ErrorPhase::GraphBuild,
                "outputs",
                format!(
                    "extension contract inferred output {:?} rank {}, but expanded output is {:?} rank {}",
                    dtype,
                    local_shape.len(),
                    output.dtype,
                    output.rank
                ),
            ));
        }
    }
    if inferred.constraints.is_empty() {
        return Ok(outputs);
    }

    let origins = outputs
        .iter()
        .map(|output| output.graph.values()[output.val].key.clone())
        .collect::<Vec<_>>();
    let input_keys = inputs
        .iter()
        .map(|input| input.graph.values()[input.val].key.clone())
        .collect::<Vec<_>>();
    let constraints = inferred
        .constraints
        .into_iter()
        .map(|local| ScopedShapeConstraint {
            origins: origins.clone(),
            inputs: input_keys.clone(),
            local,
        })
        .collect();
    let scope = Arc::new(ShapeConstraintScope::new(constraints));
    for output in &mut outputs {
        output.constraint_scopes =
            ConstraintScopeChain::with_scope(Arc::clone(&scope), [&output.constraint_scopes]);
    }
    Ok(outputs)
}

/// Apply an expanded core graph while retaining one extension metadata contract.
///
/// Metadata inference runs exactly once. Its output metadata builds the traced
/// outputs and its equality constraints are attached to those same outputs.
///
/// # Examples
///
/// ```rust
/// # use std::{any::Any, sync::Arc};
/// use computegraph::types::OperationRole;
/// use tenferro_ops::{std_tensor_op::StdTensorOp, ExtensionShapeContext};
/// use tenferro_runtime::extension::{apply_expanded_graph_with_shape_contract, ExtensionOp};
/// use tenferro_runtime::{DType, SymDim, TracedTensor};
///
/// # #[derive(Clone, Debug)]
/// # struct SameShapeAdd;
/// # impl ExtensionOp for SameShapeAdd {
/// #     fn family_id(&self) -> &'static str { "example.expanded-add.v1" }
/// #     fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
/// #     fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
/// #         other.as_any().downcast_ref::<Self>().is_some()
/// #     }
/// #     fn clone_arc(&self) -> Arc<dyn ExtensionOp> { Arc::new(self.clone()) }
/// #     fn as_any(&self) -> &dyn Any { self }
/// #     fn input_count(&self) -> usize { 2 }
/// #     fn output_count(&self) -> usize { 1 }
/// #     fn infer_output_meta(
/// #         &self,
/// #         ctx: &mut ExtensionShapeContext<'_>,
/// #     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
/// #         ctx.require_same_shape(0, 1)?;
/// #         Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
/// #     }
/// # }
/// let lhs = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
/// let rhs = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
/// let outputs = apply_expanded_graph_with_shape_contract(
///     &SameShapeAdd,
///     &[&lhs, &rhs],
///     |builder, inputs| {
///         Ok(builder.add_operation(StdTensorOp::Add, inputs.to_vec(), OperationRole::Primary))
///     },
/// )?;
/// assert_eq!(outputs[0].rank, 1);
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
#[doc(hidden)]
pub fn apply_expanded_graph_with_shape_contract(
    op: &dyn ExtensionOp,
    inputs: &[&TracedTensor],
    build: impl FnOnce(&mut GraphBuilder<StdTensorOp>, &[ValueRef<StdTensorOp>]) -> Result<Vec<usize>>,
) -> Result<Vec<TracedTensor>> {
    let (output_metas, inferred) = infer_expanded_shape_contract(op, inputs)?;
    let outputs = apply_expanded_graph(inputs, output_metas, build)?;
    attach_inferred_expanded_shape_contract(inputs, outputs, inferred)
}

/// Apply an extension-provided lowering as ordinary traced graph operations.
///
/// This is for extension crates whose operation can be expanded at graph-build
/// time. It preserves the same parent graph and metadata merging behavior as
/// [`apply`], but does not insert a `StdTensorOp::Extension` carrier.
pub fn apply_expanded_graph(
    inputs: &[&TracedTensor],
    output_metas: Vec<(tenferro_tensor::DType, Vec<SymDim>)>,
    build: impl FnOnce(&mut GraphBuilder<StdTensorOp>, &[ValueRef<StdTensorOp>]) -> Result<Vec<usize>>,
) -> Result<Vec<TracedTensor>> {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    for input in inputs {
        builder.add_parent(input.graph.clone());
    }
    let op_inputs: Vec<ValueRef<StdTensorOp>> = inputs
        .iter()
        .map(|t| ValueRef::External(t.graph.values()[t.val].key.clone()))
        .collect();
    let outputs = build(&mut builder, &op_inputs)?;
    if outputs.len() != output_metas.len() {
        return Err(Error::invalid_argument(
            "extension::apply_expanded_graph",
            ErrorPhase::GraphBuild,
            "outputs",
            format!(
                "extension expanded graph returned {} outputs for {} output metadata entries",
                outputs.len(),
                output_metas.len()
            ),
        ));
    }
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    let analysis = register_scoped_graph_analysis(graph.as_ref(), std::iter::empty())?;
    traced_outputs_from_analysis(inputs, graph, &outputs, output_metas, analysis)
}

fn traced_outputs_from_analysis(
    inputs: &[&TracedTensor],
    graph: Arc<computegraph::graph::Graph<StdTensorOp>>,
    outputs: &[usize],
    output_metas: Vec<(tenferro_tensor::DType, Vec<SymDim>)>,
    analysis: RegisteredGraphAnalysis,
) -> Result<Vec<TracedTensor>> {
    let metadata_scope = Arc::new(analysis.metadata);
    let constraint_scope = Arc::new(analysis.constraints);

    let merged_map = merge_traced_inputs_map(inputs.iter().copied());
    let mut extra_roots = Vec::new();
    let mut checkpoint_chain = None;
    let metadata_scopes = MetadataScopeChain::with_scope(
        Arc::clone(&metadata_scope),
        inputs.iter().map(|input| &input.metadata_scopes),
    );
    let constraint_scopes = if constraint_scope.is_empty() {
        ConstraintScopeChain::merge(inputs.iter().map(|input| &input.constraint_scopes))
    } else {
        ConstraintScopeChain::with_scope(
            constraint_scope,
            inputs.iter().map(|input| &input.constraint_scopes),
        )
    };
    for input in inputs {
        extra_roots.extend(input.extra_roots.iter().cloned());
        checkpoint_chain =
            CheckpointNode::merge_chains(checkpoint_chain, input.checkpoint_chain.clone());
    }
    let all_inputs_concrete = inputs.iter().all(|t| t.shape_hint.is_some());
    Ok(outputs
        .iter()
        .zip(output_metas)
        .map(|(&val, (dtype, shape))| {
            let shape_hint = if all_inputs_concrete {
                Some(shape.clone())
            } else {
                None
            };
            TracedTensor {
                id: next_traced_id(),
                rank: shape.len(),
                dtype,
                graph: graph.clone(),
                val,
                data: None,
                shape_hint,
                inputs_map: merged_map.clone(),
                extra_roots: extra_roots.clone(),
                checkpoint_chain: checkpoint_chain.clone(),
                metadata_scopes: metadata_scopes.clone(),
                constraint_scopes: constraint_scopes.clone(),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests;
