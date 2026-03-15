use tenferro_dyadtensor::{forward_ad, StructuredTensor, Tensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn matrix2(values: &[f64; 4]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn vector(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn as_slice(tensor: &DenseTensor<f64>) -> &[f64] {
    tensor
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

#[test]
fn dyn_ad_tensor_reshape_preserves_forward_mode() {
    let (reshaped, tangent) = forward_ad::dual_level(|fw| {
        let x = fw.make_dual(
            &Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0])),
            &Tensor::from_tensor(matrix2(&[0.5, -0.25, 0.75, 1.0])),
        )?;
        let reshaped = x.reshape(&[4])?;
        fw.unpack_dual(&reshaped)
    })
    .unwrap();

    let reshaped = reshaped.as_f64().unwrap();
    assert_eq!(reshaped.dims(), &[4]);
    assert_eq!(as_slice(reshaped.primal()), &[1.0, 2.0, 3.0, 4.0]);
    let tangent = tangent.unwrap();
    assert_eq!(
        as_slice(tangent.as_f64().unwrap().primal()),
        &[0.5, -0.25, 0.75, 1.0]
    );
}

#[test]
fn dyn_ad_tensor_reshape_pullback_restores_original_shape() {
    let mut x = Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0]));
    x.set_requires_grad(true).unwrap();

    let reshaped = x.reshape(&[4]).unwrap();
    let cotangent = Tensor::from_tensor(vector(&[1.0, -2.0, 0.5, 3.0]));
    let grad_outputs = [cotangent];
    let grads =
        tenferro_dyadtensor::grad(&[&reshaped], &[&x], Some(&grad_outputs), Default::default())
            .unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2, 2]);
    assert_eq!(
        as_slice(grad.as_f64().unwrap().primal()),
        &[1.0, -2.0, 0.5, 3.0]
    );
}

#[test]
fn dyn_ad_tensor_take_prefix_preserves_forward_mode() {
    let (sliced, tangent) = forward_ad::dual_level(|fw| {
        let x = fw.make_dual(
            &Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0])),
            &Tensor::from_tensor(matrix2(&[0.5, -0.25, 0.75, 1.0])),
        )?;
        let sliced = x.take_prefix(1, 1)?;
        fw.unpack_dual(&sliced)
    })
    .unwrap();

    let sliced = sliced.as_f64().unwrap();
    assert_eq!(sliced.dims(), &[2, 1]);
    assert_eq!(as_slice(sliced.primal()), &[1.0, 2.0]);
    assert_eq!(
        as_slice(tangent.unwrap().as_f64().unwrap().primal()),
        &[0.5, -0.25]
    );
}

#[test]
fn dyn_ad_tensor_take_prefix_pullback_zero_fills_dropped_entries() {
    let mut x = Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0]));
    x.set_requires_grad(true).unwrap();

    let sliced = x.take_prefix(1, 1).unwrap();
    let cotangent = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.5, -0.5], &[2, 1], MemoryOrder::ColumnMajor).unwrap(),
    );

    let grad_outputs = [cotangent];
    let grads =
        tenferro_dyadtensor::grad(&[&sliced], &[&x], Some(&grad_outputs), Default::default())
            .unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2, 2]);
    assert_eq!(
        as_slice(grad.as_f64().unwrap().primal()),
        &[1.5, -0.5, 0.0, 0.0]
    );
}

#[test]
fn dyn_ad_tensor_diag_embed_preserves_reverse_pullback() {
    let mut x = Tensor::from_tensor(vector(&[2.0, 3.0]));
    x.set_requires_grad(true).unwrap();

    let diag = x.diag_embed(2).unwrap();
    assert!(diag.is_diag());
    let cotangent = Tensor::from_structured(
        StructuredTensor::from_diagonal_vector(vector(&[0.5, -1.0]), 2).unwrap(),
    );

    let grad_outputs = [cotangent];
    let grads = tenferro_dyadtensor::grad(&[&diag], &[&x], Some(&grad_outputs), Default::default())
        .unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2]);
    assert_eq!(as_slice(grad.as_f64().unwrap().primal()), &[0.5, -1.0]);
}
