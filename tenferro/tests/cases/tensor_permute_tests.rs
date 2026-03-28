use crate::support::{diag_f64, with_axis_classes_f64};
use tenferro::{grad, GradOptions, Tensor};
use tenferro_device::Error as DeviceError;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn dense2(values: &[f64], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap(),
    )
}

#[test]
fn permute_transposes_dense_logical_axes() {
    let x = dense2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = x.permute(&[1, 0]).unwrap();
    let dense = y
        .as_f64()
        .unwrap()
        .primal()
        .contiguous(MemoryOrder::ColumnMajor);

    assert_eq!(y.dims(), &[3, 2]);
    assert_eq!(y.axis_classes(), &[0, 1]);
    assert!(y.is_dense());
    assert_eq!(
        dense.buffer().as_slice().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn permute_preserves_diagonal_layout() {
    let x = diag_f64(&[2.0, 3.0, 4.0]);
    let y = x.permute(&[1, 0]).unwrap();

    assert_eq!(y.dims(), &[3, 3]);
    assert_eq!(y.axis_classes(), &[0, 0]);
    assert!(y.is_diag());
    assert_eq!(
        y.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[2.0, 3.0, 4.0]
    );
}

#[test]
fn permute_reorders_multi_class_structured_layouts() {
    let x = with_axis_classes_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[0, 0, 1, 1]);
    let y = x.permute(&[2, 3, 0, 1]).unwrap();
    let payload = y
        .as_f64()
        .unwrap()
        .primal()
        .contiguous(MemoryOrder::ColumnMajor);

    assert_eq!(y.dims(), &[2, 2, 2, 2]);
    assert_eq!(y.axis_classes(), &[0, 0, 1, 1]);
    assert_eq!(y.as_f64().unwrap().primal().dims(), &[2, 2]);
    assert_eq!(payload.buffer().as_slice().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn permute_rejects_invalid_permutations() {
    let x = dense2(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = match x.permute(&[0, 0]) {
        Ok(_) => panic!("permute should reject duplicate logical axes"),
        Err(err) => err,
    };
    let message = match err {
        tenferro::Error::Backend(DeviceError::InvalidArgument(message)) => message,
        other => panic!("expected backend invalid argument, got {other:?}"),
    };
    assert!(message.contains("perm"));
}

#[test]
fn permute_preserves_reverse_mode_metadata() {
    let mut x = dense2(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    x.set_requires_grad(true).unwrap();
    let y = x.permute(&[1, 0]).unwrap();
    let cotangent = dense2(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let grads = grad(&[&y], &[&x], Some(&[cotangent]), GradOptions::default()).unwrap();
    let gx = grads[0].as_ref().unwrap();
    assert_eq!(gx.dims(), &[2, 2]);
}
