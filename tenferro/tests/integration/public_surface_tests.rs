use tenferro::{ScalarType, Tensor};

#[test]
fn tensor_from_slice_reports_dtype_shape_and_layout_flags() {
    let value = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    assert_eq!(value.scalar_type(), ScalarType::F64);
    assert_eq!(value.dims(), &[2, 2]);
    assert!(value.is_dense());
    assert!(!value.is_diag());
    assert_eq!(value.try_to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_detach_drops_reverse_tracking() {
    let value = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);

    let detached = value.detach();

    assert!(value.requires_grad());
    assert!(!detached.requires_grad());
    assert_eq!(detached.try_to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
}
