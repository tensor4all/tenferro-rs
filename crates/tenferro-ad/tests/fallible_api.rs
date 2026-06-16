use tenferro_ad::{EagerRuntime, EagerTensor, TracedTensorAdExt};
use tenferro_runtime::TracedTensor;
use tenferro_tensor::{Error as TensorError, Tensor};

#[test]
fn traced_jvp_vjp_return_errors_for_inactive_inputs() {
    let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]);
    let tangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]);
    let cotangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]);
    let loss = &y * &y;

    let _ = loss.jvp(&x, &tangent).unwrap_err();
    let _ = loss.vjp(&x, &cotangent).unwrap_err();
    assert!(loss.jvp_optional(&x, &tangent).unwrap().is_none());
    assert!(loss.vjp_optional(&x, &cotangent).unwrap().is_none());
}

#[test]
fn eager_binary_methods_return_shape_errors() {
    let ctx = EagerRuntime::new();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx.clone(),
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx,
    );

    let err = x.add(&y).unwrap_err();

    assert!(matches!(
        err,
        tenferro_ad::Error::TensorRuntime(TensorError::ShapeMismatch { op: "add", .. })
    ));
}
