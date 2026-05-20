//! Public surface for out-of-tree extension primitives.
//!
//! This module exposes the Stage 6 `ExtensionOp` mechanism through the
//! `tenferro` facade. External crates implement
//! [`tenferro_ops::ext_op::ExtensionOp`], register an
//! [`ExtensionFactory`] through [`register_extension`], register AD rules
//! through [`register_extension_rule`], and build traced or eager graphs
//! containing the extension via [`apply`] / [`apply_eager`].
//!
//! See `docs/spec/extension-op.md` for the normative contract.
//!
//! # Examples
//!
//! ```ignore
//! use std::sync::Arc;
//! use tenferro::extension::{apply, register_extension, ExtensionFactory, ExtensionOp};
//!
//! # struct MyFactory;
//! # impl ExtensionFactory for MyFactory {
//! #     fn family_id(&self) -> &'static str { "my-crate.my_op.v1" }
//! #     fn version(&self) -> u32 { 1 }
//! # }
//! register_extension(Arc::new(MyFactory)).expect("register");
//! // Once registered, construct an `Arc<dyn ExtensionOp>` and call
//! // `apply(op, &[input])` to lower it into a `TracedTensor`.
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{OpMode, ValRef};
use computegraph::GraphOp;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::SymDim;
use tenferro_tensor::Tensor;

use crate::checkpoint::CheckpointNode;
use crate::eager::{record_eager_outputs, EagerTensor};
use crate::eager_exec::exec_op_on_tensors;
use crate::error::{Error, Result};
use crate::metadata::{push_metadata_scope, register_scoped_fragment_metadata};
use crate::traced::{next_traced_id, TracedTensor};

pub use tenferro_ops::ext_op::{
    is_extension_registered, is_extension_rule_registered, lookup_extension_factory,
    lookup_extension_rule, register_extension, register_extension_rule,
    ExtensionAdRule as _ExtensionAdRuleReexport, ExtensionFactory,
    ExtensionOp as _ExtensionOpReexport, ExtensionRegistryError,
};

// Re-export under a canonical name (the `_ExtensionOpReexport` alias above
// exists only so the macro-generated doc-test type bounds can find the
// trait; downstream callers should use this name).
pub use tenferro_ops::ext_op::ExtensionAdRule as ExtensionAdRuleTrait;
pub use tenferro_ops::ext_op::ExtensionOp as ExtensionOpTrait;
pub use tenferro_ops::ExtensionFamilyId;

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
/// [`TracedTensor::eval_with_inputs`] at evaluation time.
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use tenferro::extension::apply;
///
/// # let op: Arc<dyn tenferro::extension::ExtensionOpTrait> = unimplemented!();
/// # let a: tenferro::TracedTensor = unimplemented!();
/// # let b: tenferro::TracedTensor = unimplemented!();
/// let outputs = apply(op, &[&a, &b]);
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
    // correctly via tenferro-ops's SymDim API.
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

/// Apply an extension op to eager tensors.
///
/// The forward pass delegates to [`ExtensionOp::eager_execute`] through the
/// usual eager execution path, then records the same `StdTensorOp::Extension`
/// node used by traced graphs. Eager backward therefore uses registered
/// [`ExtensionAdRuleTrait`] rules through the same AD source of truth.
pub fn apply_eager(op: Arc<dyn ExtensionOp>, inputs: &[&EagerTensor]) -> Result<Vec<EagerTensor>> {
    let Some(first) = inputs.first() else {
        return Err(Error::Internal(
            "extension::apply_eager requires at least one input tensor".to_string(),
        ));
    };
    if inputs.len() != op.n_inputs() {
        return Err(Error::Internal(format!(
            "extension::apply_eager: op family {:?} expects {} inputs, got {}",
            op.family_id(),
            op.n_inputs(),
            inputs.len()
        )));
    }

    let ctx = Arc::clone(&first.ctx);
    for tensor in inputs.iter().skip(1) {
        if !first.same_context(tensor) {
            return Err(Error::ContextMismatch {
                lhs: first.ctx_id(),
                rhs: tensor.ctx_id(),
            });
        }
    }

    let op = StdTensorOp::Extension(op);
    let concrete_inputs: Vec<&Tensor> = inputs.iter().map(|tensor| tensor.data.as_ref()).collect();
    let outputs = {
        let mut backend = ctx.backend.lock().unwrap();
        exec_op_on_tensors(&op, &concrete_inputs, &mut *backend)?
    };
    if outputs.len() != op.n_outputs() {
        return Err(Error::Internal(format!(
            "expected {} eager outputs for {:?}, got {}",
            op.n_outputs(),
            op,
            outputs.len()
        )));
    }

    if !inputs.iter().any(|input| input.requires_grad) {
        return Ok(outputs
            .into_iter()
            .map(|output| EagerTensor::new_untracked_result(Arc::clone(&ctx), output))
            .collect());
    }

    let outputs: Vec<Arc<Tensor>> = outputs.into_iter().map(Arc::new).collect();
    let recorded = record_eager_outputs(&op, &outputs, inputs);
    if recorded.traces.len() != outputs.len() {
        return Err(Error::Internal(format!(
            "expected {} eager traces for {:?}, got {}",
            outputs.len(),
            op,
            recorded.traces.len()
        )));
    }
    let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
    for input in inputs {
        for scope in &input.metadata_scopes {
            push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
        }
    }

    Ok(recorded
        .traces
        .into_iter()
        .zip(outputs)
        .map(|(trace, output)| {
            EagerTensor::new_result(
                Arc::clone(&ctx),
                trace.key,
                output.as_ref().clone(),
                trace.requires_grad,
                trace.node,
                metadata_scopes.clone(),
            )
        })
        .collect())
}
