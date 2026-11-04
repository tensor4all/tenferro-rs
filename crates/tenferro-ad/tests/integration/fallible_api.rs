use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_ad::{EagerRuntime, EagerTensor, TracedTensorAdExt};
use tenferro_runtime::{DType, ErrorPhase, TracedTensor};
use tenferro_tensor::{Error as TensorError, Tensor};

#[test]
fn eager_public_tensor_accessors_are_fallible_source_contract() {
    let eager_source = include_str!("../../src/eager.rs");

    for forbidden in [
        "pub fn data(&self) -> &Tensor",
        "pub fn from_tensor_in(tensor: Tensor, ctx: Arc<EagerRuntime>) -> Self",
        "pub fn requires_grad_in(tensor: Tensor, ctx: Arc<EagerRuntime>) -> Self",
        "pub fn constant_from(self: &Arc<Self>, tensor: Tensor) -> EagerTensor",
        "pub fn variable_from(self: &Arc<Self>, tensor: Tensor) -> EagerTensor",
        "pub fn detach_into(&self, ctx: &Arc<EagerRuntime>) -> Self",
        "pub fn debug_trace_saved_value_count(&self) -> Option<usize>",
        "pub fn backend_broadcast_multiply_untracked(",
        "pub fn apply_standard_graph(",
        ".expect(\"fresh eager leaf metadata registration failed\")",
        ".expect(\"validated eager tensor value\")",
    ] {
        assert!(
            !eager_source.contains(forbidden),
            "eager public API must not expose infallible tensor access/import path: {forbidden}"
        );
    }
}

#[test]
fn eager_axis_ops_validate_before_recording_source_contract() {
    let source = include_str!("../../src/eager_ops.rs");

    let sum = source
        .split_once("fn reduce_sum_in_session(")
        .and_then(|(_, rest)| {
            rest.split_once("StdTensorOp::ReduceSum")
                .map(|(body, _)| body)
        })
        .expect("missing borrowed reduction sum source section");
    assert!(sum.contains("validate_eager_axes("));

    let session_source = include_str!("../../src/eager.rs");
    for (method, op_variant) in [
        (
            "pub fn reduce_max(\n        &mut self,",
            "StdTensorOp::ReduceMax",
        ),
        (
            "pub fn reduce_min(\n        &mut self,",
            "StdTensorOp::ReduceMin",
        ),
    ] {
        let body = session_source
            .split_once(method)
            .and_then(|(_, rest)| rest.split_once(op_variant).map(|(before_op, _)| before_op))
            .unwrap_or_else(|| panic!("missing borrowed source contract section for {method}"));
        assert!(
            body.contains("validate_eager_axes("),
            "{method} must validate axes before recording"
        );
    }
    let product = session_source
        .split_once("pub fn reduce_prod(\n        &mut self,")
        .and_then(|(_, rest)| {
            rest.split_once("StdTensorOp::ReduceProd")
                .map(|(body, _)| body)
        })
        .expect("missing EagerSession::reduce_prod source section");
    assert!(
        product.contains("validate_eager_axes("),
        "borrowed product reduction must validate axes before recording"
    );
    let reverse = session_source
        .split_once("pub fn reverse(&mut self, input: &EagerTensor, axes: &[usize])")
        .and_then(|(_, rest)| {
            rest.split_once("StdTensorOp::Reverse")
                .map(|(before_op, _)| before_op)
        })
        .expect("missing EagerSession::reverse source section");
    assert!(
        reverse.contains("validate_eager_axes("),
        "borrowed reverse must validate axes before recording"
    );
}

#[test]
fn eager_runtime_lock_scopes_are_bounded_source_contract() {
    let source = include_str!("../../src/eager.rs");

    let clear_grads = source
        .split_once("pub fn clear_grads(&self) -> Result<()>")
        .and_then(|(_, rest)| rest.split_once("fn store_grads").map(|(body, _)| body))
        .expect("missing EagerRuntime::clear_grads source section");
    assert!(
        clear_grads.contains("let live_slots ="),
        "clear_grads should collect live slots under the grad-slot map lock"
    );
    let map_lock_section = clear_grads
        .split_once("for slot in live_slots")
        .map(|(before_loop, _)| before_loop)
        .expect("clear_grads should process slots after collecting them");
    assert!(
        !map_lock_section.contains("slot.lock()"),
        "clear_grads must not hold the grad-slot map lock while locking each slot"
    );

    // Only extension ops reach the owner-context fallback (#1946 F9), so the
    // runtime's extension locks are never taken for a standard op.
    let extension_fallback = source
        .split_once("pub(crate) fn exec_extension_outputs_read(")
        .and_then(|(_, rest)| rest.split_once("#[cfg(test)]").map(|(body, _)| body))
        .expect("missing EagerRuntime::exec_extension_outputs_read source section");
    assert!(
        extension_fallback.contains("op: &Arc<dyn tenferro_ops::ext_op::ExtensionOp>"),
        "the owner-context fallback must accept extension ops only"
    );
    assert!(
        !source.contains("fn exec_outputs_with_runtime"),
        "the mixed standard/extension owner-context helper is removed"
    );
}

#[test]
fn eager_dot_general_surfaces_validate_config_before_dispatch_source_contract() {
    let session_source = include_str!("../../src/eager.rs");
    let dot_general = session_source
        .split_once("pub fn dot_general(\n        &mut self,")
        .and_then(|(_, rest)| {
            rest.split_once("EagerTensor::nary_op_in_session")
                .map(|(before_dispatch, _)| before_dispatch)
        })
        .expect("missing EagerSession::dot_general source section");
    assert!(
        dot_general.contains("validate_dims_with_ranks("),
        "EagerSession::dot_general must validate DotGeneralConfig before dispatch"
    );

    let dot_general_with_conj = session_source
        .split_once("pub fn dot_general_with_conj(")
        .and_then(|(_, rest)| rest.split_once("if !lhs.requires_grad"))
        .map(|(before_dispatch, _)| before_dispatch)
        .expect("missing EagerSession::dot_general_with_conj source section");
    assert!(
        dot_general_with_conj.contains("config: DotGeneralConfig"),
        "EagerSession dot-general surfaces should consistently take owned configs"
    );
    assert!(
        dot_general_with_conj.contains("validate_dims_with_ranks("),
        "EagerSession::dot_general_with_conj must validate DotGeneralConfig before fast-path dispatch"
    );
}

#[test]
fn traced_jvp_vjp_grad_return_typed_errors_for_inactive_inputs() {
    let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
    let tangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let cotangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let loss = (&y * &y).unwrap();

    // The documented inactive-`wrt` state is user-reachable input validation, not
    // an internal invariant: the strict wrappers must return a typed error.
    for err in [
        loss.jvp(&x, &tangent).unwrap_err(),
        loss.vjp(&x, &cotangent).unwrap_err(),
        loss.grad(&x).unwrap_err(),
    ] {
        assert!(matches!(
            err,
            tenferro_runtime::Error::Validation {
                phase: ErrorPhase::GraphBuild,
                source: tenferro_tensor::ValidationError::InvalidArgument {
                    argument: "wrt",
                    ..
                },
                ..
            }
        ));
    }

    assert!(loss.jvp_optional(&x, &tangent).unwrap().is_none());
    assert!(loss.vjp_optional(&x, &cotangent).unwrap().is_none());
    assert!(loss.grad_optional(&x).unwrap().is_none());
}

#[test]
fn traced_jvp_vjp_return_errors_for_symbolic_seed_tensors() {
    let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let loss = (&x * &x).unwrap();
    let tangent = TracedTensor::input_symbolic_shape(DType::F64, 0).unwrap();
    let cotangent = TracedTensor::input_symbolic_shape(DType::F64, 0).unwrap();

    let jvp = catch_unwind(AssertUnwindSafe(|| loss.jvp(&x, &tangent)));
    assert!(jvp.is_ok(), "jvp should return Err, not panic");
    let err = jvp.unwrap().unwrap_err().to_string();
    assert!(err.contains("jvp tangent"), "{err}");

    let vjp = catch_unwind(AssertUnwindSafe(|| loss.vjp(&x, &cotangent)));
    assert!(vjp.is_ok(), "vjp should return Err, not panic");
    let err = vjp.unwrap().unwrap_err().to_string();
    assert!(err.contains("vjp cotangent"), "{err}");
}

#[test]
fn eager_binary_methods_return_shape_errors() {
    let ctx = EagerRuntime::new().unwrap();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let err = x
        .runtime()
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| s.add(&x, &y))
        .unwrap_err();

    assert!(matches!(
        err,
        tenferro_ad::Error::TensorRuntime(TensorError::Validation {
            op: "add",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
}
