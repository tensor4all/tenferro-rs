//! Public surface for out-of-tree extension primitives.
//!
//! This module exposes the Stage 6 `ExtensionOp` mechanism through the
//! runtime crate. External crates implement
//! [`ExtensionOp`] and build traced graphs containing the extension via
//! [`apply`].
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

use computegraph::GraphOperation;
use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::error::{Error, ErrorPhase, Result};
use crate::trace_ops::{append_core, append_extension, finish_derived};
use crate::traced::{ensure_same_context, TracedTensor};

pub use crate::compiler::CompilerOptions;
#[doc(hidden)]
pub use crate::shape_infer::{
    infer_output_dtype, infer_output_extents, infer_output_shapes, promote_dtype,
    promote_dtype_for_binary_op, promote_dtypes,
};
pub use tenferro_extension_macros::define_extension_runtime;
pub use tenferro_ops::ext_op::{
    ExtensionAlias, ExtensionAliasDeclaration, ExtensionEffect, ExtensionEffectAccess,
    ExtensionEffectDeclaration, ExtensionOp,
};
pub use tenferro_ops::{ExtensionFamilyId, ExtensionShapeContext, SymDim};

pub use crate::extension_cache::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
pub use crate::extension_execution_context::ExtensionExecutionContext;
pub use crate::runtime::{ExtensionModule, ExtensionModuleId, ExtensionModuleRegistrar};

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
/// composition, pass concrete tensors to [`crate::Runtime::run`] at
/// evaluation time.
///
/// # Examples
///
/// ```rust
/// # use std::any::Any;
/// use std::sync::Arc;
/// use tenferro_runtime::extension::{apply, ExtensionOp, ExtensionShapeContext};
/// use tenferro_runtime::{TraceContext, DType, SymDim, TracedTensor};
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
/// #     // The program builder validates these two declarations when the op is
/// #     // appended, so an extension must state them explicitly.
/// #     fn semantic_effects(&self) -> tenferro_runtime::extension::ExtensionEffectDeclaration<'_> {
/// #         tenferro_runtime::extension::ExtensionEffectDeclaration::Declared(&[])
/// #     }
/// #     fn semantic_aliases(&self) -> tenferro_runtime::extension::ExtensionAliasDeclaration<'_> {
/// #         tenferro_runtime::extension::ExtensionAliasDeclaration::AllFresh
/// #     }
/// # }
/// let op: Arc<dyn ExtensionOp> = Arc::new(IdentityExt);
/// let trace = TraceContext::new();
/// let a = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let outputs = apply(op, &[&a])?;
/// assert_eq!(outputs.len(), 1);
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Validation`] with `ValidationError::InvalidArgument` when
/// the extension receives the wrong number of traced inputs or produces an
/// unknown output shape. Canonical metadata inference failures, including a
/// returned metadata count that differs from [`ExtensionOp::output_count`],
/// are returned as [`Error::TensorRuntime`] containing the typed tensor
/// validation source, while poisoned metadata state is retained as
/// [`Error::RuntimeStateSource`].
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
    let Some(first) = inputs.first() else {
        return Err(Error::invalid_argument(
            "extension::apply",
            ErrorPhase::GraphBuild,
            "inputs",
            "an extension op needs at least one traced input",
        ));
    };
    for input in inputs.iter().skip(1) {
        ensure_same_context(first, input)?;
    }
    let values: Vec<_> = inputs.iter().map(|input| input.value).collect();
    let outputs = append_extension(&first.ctx, op, &values)?;
    outputs
        .iter()
        .map(|&value| finish_derived(&first.ctx, value))
        .collect()
}

/// Apply one standard primitive op to traced inputs in their shared context.
///
/// # Errors
///
/// Returns [`Error::Validation`] with `ValidationError::InvalidArgument` when the
/// input count or a traced input does not fit the op, and the same typed build
/// failures as [`apply`] when the derived output cannot be finished.
pub fn apply_standard_op(op: StdTensorOp, inputs: &[&TracedTensor]) -> Result<Vec<TracedTensor>> {
    let expected = op.input_count();
    if inputs.len() != expected {
        return Err(Error::invalid_argument(
            "extension::apply_standard_op",
            ErrorPhase::GraphBuild,
            "inputs",
            format!("op expects {expected} inputs, got {}", inputs.len()),
        ));
    }
    let Some(first) = inputs.first() else {
        return Err(Error::invalid_argument(
            "extension::apply_standard_op",
            ErrorPhase::GraphBuild,
            "inputs",
            "a standard op needs at least one traced input",
        ));
    };
    for input in inputs.iter().skip(1) {
        ensure_same_context(first, input)?;
    }
    let values: Vec<_> = inputs.iter().map(|input| input.value).collect();
    let outputs = append_core(&first.ctx, &op, &values)?;
    outputs
        .iter()
        .map(|&value| finish_derived(&first.ctx, value))
        .collect()
}

#[cfg(test)]
mod tests;
