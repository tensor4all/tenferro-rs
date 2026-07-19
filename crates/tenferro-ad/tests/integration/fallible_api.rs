use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_ad::{EagerRuntime, EagerTensor, TracedTensorAdExt};
use tenferro_runtime::{DType, TracedTensor};
use tenferro_tensor::{Error as TensorError, Tensor};

#[test]
fn eager_public_tensor_accessors_are_fallible_source_contract() {
    let eager_source = include_str!("../../src/eager.rs");
    let eager_builder_source = include_str!("../../src/eager_builder.rs");

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
    for forbidden in [
        "pub struct EagerPrimitiveBuilder",
        "pub fn tensor(&self, id: LocalValueId) -> Arc<Tensor>",
    ] {
        assert!(
            !eager_builder_source.contains(forbidden),
            "eager primitive builder must stay internal and fallible: {forbidden}"
        );
    }
}

#[test]
fn eager_axis_ops_validate_before_recording_source_contract() {
    let source = include_str!("../../src/eager_ops.rs");

    for (method, op_variant) in [
        (
            "pub fn reduce_sum(&self, axes: Option<&[usize]>)",
            "StdTensorOp::ReduceSum",
        ),
        (
            "pub fn reduce_prod(&self, axes: Option<&[usize]>)",
            "StdTensorOp::ReduceProd",
        ),
        (
            "pub fn reduce_max(&self, axes: Option<&[usize]>)",
            "StdTensorOp::ReduceMax",
        ),
        (
            "pub fn reduce_min(&self, axes: Option<&[usize]>)",
            "StdTensorOp::ReduceMin",
        ),
        (
            "pub fn reverse(&self, axes: &[usize])",
            "StdTensorOp::Reverse",
        ),
    ] {
        let body = source
            .split_once(method)
            .and_then(|(_, rest)| rest.split_once(op_variant).map(|(before_op, _)| before_op))
            .unwrap_or_else(|| panic!("missing source contract section for {method}"));
        assert!(
            body.contains("validate_eager_axes("),
            "{method} must validate axes before recording {op_variant}"
        );
    }
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

    let exec_outputs = source
        .split_once("pub(crate) fn exec_outputs(")
        .and_then(|(_, rest)| {
            rest.split_once("pub(crate) fn exec_outputs_read")
                .map(|(body, _)| body)
        })
        .expect("missing EagerRuntime::exec_outputs source section");
    assert!(
        exec_outputs.contains("exec_outputs_with_optional_extension_lock("),
        "exec_outputs should centralize backend/extension lock ordering and avoid extension locks for standard ops"
    );

    let lock_helper = source
        .split_once("fn exec_outputs_with_optional_extension_lock")
        .and_then(|(_, rest)| rest.split_once("#[cfg(test)]").map(|(body, _)| body))
        .expect("missing eager runtime lock helper source section");
    assert!(
        lock_helper.contains("StdTensorOp::Extension"),
        "extension executor lock should be acquired only for extension ops"
    );
    assert!(
        lock_helper.contains("Lock ordering:"),
        "backend/extension lock ordering must be documented at the helper that co-holds the locks"
    );
}

#[test]
fn eager_dot_general_surfaces_validate_config_before_dispatch_source_contract() {
    let source = include_str!("../../src/eager_ops.rs");

    let dot_general = source
        .split_once("pub fn dot_general(&self, other: &Self, config: DotGeneralConfig)")
        .and_then(|(_, rest)| {
            rest.split_once("self.binary_op")
                .map(|(before_dispatch, _)| before_dispatch)
        })
        .expect("missing EagerTensor::dot_general source section");
    assert!(
        dot_general.contains("validate_eager_dot_general_config("),
        "EagerTensor::dot_general must validate DotGeneralConfig before dispatch"
    );

    let dot_general_with_conj = source
        .split_once("pub fn dot_general_with_conj(")
        .and_then(|(_, rest)| {
            rest.split_once("exec_dot_general_with_conj_on_tensor_reads")
                .map(|(before_dispatch, _)| before_dispatch)
        })
        .expect("missing EagerTensor::dot_general_with_conj source section");
    assert!(
        dot_general_with_conj.contains("config: DotGeneralConfig"),
        "EagerTensor dot-general surfaces should consistently take owned configs"
    );
    assert!(
        dot_general_with_conj.contains("validate_eager_dot_general_config("),
        "EagerTensor::dot_general_with_conj must validate DotGeneralConfig before fast-path dispatch"
    );
}

#[test]
fn traced_jvp_vjp_return_errors_for_inactive_inputs() {
    let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
    let tangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let cotangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let loss = (&y * &y).unwrap();

    let _ = loss.jvp(&x, &tangent).unwrap_err();
    let _ = loss.vjp(&x, &cotangent).unwrap_err();
    assert!(loss.jvp_optional(&x, &tangent).unwrap().is_none());
    assert!(loss.vjp_optional(&x, &cotangent).unwrap().is_none());
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
    let ctx = EagerRuntime::new();
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

    let err = x.add(&y).unwrap_err();

    assert!(matches!(
        err,
        tenferro_ad::Error::TensorRuntime(TensorError::Validation {
            op: "add",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
}
