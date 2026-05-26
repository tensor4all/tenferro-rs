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
//! use tenferro_runtime::extension::{apply, ExtensionOpTrait};
//!
//! // Construct an `Arc<dyn ExtensionOpTrait>` and call `apply(op, &[input])`
//! // to lower it into a `TracedTensor`.
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{OpMode, ValRef};
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::SymDim;

use crate::checkpoint::CheckpointNode;
use crate::metadata::{push_metadata_scope, register_scoped_fragment_metadata};
use crate::traced::{next_traced_id, TracedTensor};

pub use tenferro_ops::ext_op::ExtensionOp as ExtensionOpTrait;
pub use tenferro_ops::ExtensionFamilyId;

pub use crate::extension_cache::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
pub use crate::extension_runtime::{
    ExtensionExecutionContext, ExtensionExecutor, ExtensionRegistry, ExtensionRuntime,
    ExtensionRuntimeRegistryError,
};

/// Apply an extension op in the traced graph.
///
/// The `op` value is cloned into a `StdTensorOp::Extension(Arc<dyn ExtensionOp>)`
/// carrier. The returned vector contains one [`TracedTensor`] per declared
/// output slot of the extension. Output shapes are inferred via
/// [`ExtensionOp::infer_output_meta`] using the input shape hints.
///
/// `inputs.len()` must equal `op.n_inputs()`, and each input's
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
/// use tenferro_runtime::extension::{apply, ExtensionOpTrait};
/// use tenferro_runtime::{DType, SymDim, Tensor, TracedTensor};
///
/// # #[derive(Clone, Debug)]
/// # struct IdentityExt;
/// # impl ExtensionOpTrait for IdentityExt {
/// #     fn family_id(&self) -> &'static str { "example.identity.v1" }
/// #     fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
/// #     fn payload_eq(&self, other: &dyn ExtensionOpTrait) -> bool {
/// #         other.as_any().downcast_ref::<IdentityExt>().is_some()
/// #     }
/// #     fn clone_arc(&self) -> Arc<dyn ExtensionOpTrait> { Arc::new(self.clone()) }
/// #     fn as_any(&self) -> &dyn Any { self }
/// #     fn n_inputs(&self) -> usize { 1 }
/// #     fn n_outputs(&self) -> usize { 1 }
/// #     fn infer_output_meta(
/// #         &self,
/// #         dtypes: &[DType],
/// #         shapes: &[&[SymDim]],
/// #     ) -> Vec<(DType, Vec<SymDim>)> {
/// #         vec![(dtypes[0], shapes[0].to_vec())]
/// #     }
/// #     fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
/// #         Ok(vec![inputs[0].clone()])
/// #     }
/// # }
/// let op: Arc<dyn ExtensionOpTrait> = Arc::new(IdentityExt);
/// let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let outputs = apply(op, &[&a]);
/// assert_eq!(outputs.len(), 1);
/// ```
pub fn apply(op: Arc<dyn ExtensionOp>, inputs: &[&TracedTensor]) -> Vec<TracedTensor> {
    assert_eq!(
        inputs.len(),
        op.n_inputs(),
        "extension::apply: op family {:?} expects {} inputs, got {}",
        op.family_id(),
        op.n_inputs(),
        inputs.len()
    );

    // Build the per-input dtype / shape slices the extension's
    // `infer_output_meta` wants. Symbolic-shape inputs (shape_hint =
    // None) use per-axis TensorAxis symbolic dims keyed by the input
    // TracedTensor's id so downstream composition still resolves
    // correctly via tenferro-internal-ops's SymDim API.
    let input_dtypes: Vec<_> = inputs.iter().map(|t| t.dtype).collect();
    let input_shape_storage: Vec<Vec<SymDim>> = inputs
        .iter()
        .map(|t| {
            if let Some(hint) = t.shape_hint.clone() {
                hint
            } else {
                (0..t.rank)
                    .map(|axis| SymDim::tensor_axis(t.id, axis))
                    .collect()
            }
        })
        .collect();
    let input_shape_refs: Vec<&[SymDim]> = input_shape_storage.iter().map(Vec::as_slice).collect();

    let output_metas = op.infer_output_meta(&input_dtypes, &input_shape_refs);
    assert_eq!(
        output_metas.len(),
        op.n_outputs(),
        "extension::apply: op family {:?} declared {} outputs but \
         infer_output_meta returned {}",
        op.family_id(),
        op.n_outputs(),
        output_metas.len()
    );

    // Build the fragment that carries the Extension op.
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    for input in inputs {
        builder.add_parent(input.fragment.clone());
    }
    let op_inputs: Vec<ValRef<StdTensorOp>> = inputs
        .iter()
        .map(|t| ValRef::External(t.fragment.vals()[t.val].key.clone()))
        .collect();
    let carrier = StdTensorOp::Extension(op.clone());
    let outputs = builder.add_op(carrier, op_inputs, OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    let metadata_scope = Arc::new(register_scoped_fragment_metadata(
        fragment.as_ref(),
        std::iter::empty(),
    ));

    // Merge shared-state across all parent TracedTensors.
    let mut merged_map = HashMap::new();
    let mut extra_roots = Vec::new();
    let mut checkpoint_chain = None;
    let mut metadata_scopes = vec![Arc::clone(&metadata_scope)];
    for input in inputs {
        merged_map.extend(input.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));
        extra_roots.extend(input.extra_roots.iter().cloned());
        checkpoint_chain =
            CheckpointNode::merge_chains(checkpoint_chain, input.checkpoint_chain.clone());
        for scope in &input.metadata_scopes {
            push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
        }
    }
    let merged_map = Arc::new(merged_map);

    // The primal dtype for the emitted `TracedTensor`s is the per-output
    // dtype returned by `infer_output_meta`. For consistency with the rest
    // of the tenferro public API the extension's output dtype is reported
    // on the `TracedTensor::dtype` field of each output.
    //
    // `shape_hint` is preserved only if the output shape is fully concrete
    // (no TensorAxis references). Otherwise we drop it to keep the hint
    // invariant `Some(...) => fully known at build time` that the rest of
    // the traced frontend relies on.
    let all_inputs_concrete = inputs.iter().all(|t| t.shape_hint.is_some());
    outputs
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
                fragment: fragment.clone(),
                val,
                data: None,
                shape_hint,
                inputs_map: merged_map.clone(),
                extra_roots: extra_roots.clone(),
                checkpoint_chain: checkpoint_chain.clone(),
                metadata_scopes: metadata_scopes.clone(),
            }
        })
        .collect()
}
