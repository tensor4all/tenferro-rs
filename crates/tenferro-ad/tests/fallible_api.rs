use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_ad::{EagerRuntime, EagerTensor, TracedTensorAdExt};
use tenferro_runtime::{DType, TracedTensor};
use tenferro_tensor::{Error as TensorError, Tensor};

#[test]
fn eager_public_tensor_accessors_are_fallible_source_contract() {
    let eager_source = include_str!("../src/eager.rs");
    let eager_builder_source = include_str!("../src/eager_builder.rs");

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
        tenferro_ad::Error::TensorRuntime(TensorError::ShapeMismatch { op: "add", .. })
    ));
}
