use tenferro_dyadtensor::{AdMode, DynAdTensor, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor};

fn matrix2(values: &[f64; 4]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn as_slice(tensor: &Tensor<f64>) -> &[f64] {
    tensor
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

#[test]
fn dyn_ad_tensor_reshape_preserves_forward_mode() {
    let x = DynAdTensor::new_forward(
        matrix2(&[1.0, 2.0, 3.0, 4.0]),
        matrix2(&[0.5, -0.25, 0.75, 1.0]),
    )
    .unwrap();

    let reshaped = x.reshape(&[4]).unwrap();

    assert_eq!(reshaped.mode(), AdMode::Forward);
    let reshaped = reshaped.as_f64().unwrap();
    assert_eq!(reshaped.dims(), &[4]);
    assert_eq!(as_slice(reshaped.primal()), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        as_slice(reshaped.tangent().unwrap()),
        &[0.5, -0.25, 0.75, 1.0]
    );
}

#[test]
fn dyn_ad_tensor_reshape_pullback_restores_original_shape() {
    let x = DynAdTensor::new_reverse_leaf(matrix2(&[1.0, 2.0, 3.0, 4.0])).unwrap();

    let reshaped = x.reshape(&[4]).unwrap();
    let cotangent = DynAdTensor::new_primal(vector(&[1.0, -2.0, 0.5, 3.0]));
    let grads = reshaped.pullback_wrt(&cotangent, &[&x]).unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2, 2]);
    assert_eq!(
        as_slice(grad.as_f64().unwrap().primal()),
        &[1.0, -2.0, 0.5, 3.0]
    );
}

#[test]
fn dyn_ad_tensor_take_prefix_preserves_forward_mode() {
    let x = DynAdTensor::new_forward(
        matrix2(&[1.0, 2.0, 3.0, 4.0]),
        matrix2(&[0.5, -0.25, 0.75, 1.0]),
    )
    .unwrap();

    let sliced = x.take_prefix(1, 1).unwrap();

    assert_eq!(sliced.mode(), AdMode::Forward);
    let sliced = sliced.as_f64().unwrap();
    assert_eq!(sliced.dims(), &[2, 1]);
    assert_eq!(as_slice(sliced.primal()), &[1.0, 2.0]);
    assert_eq!(as_slice(sliced.tangent().unwrap()), &[0.5, -0.25]);
}

#[test]
fn dyn_ad_tensor_take_prefix_pullback_zero_fills_dropped_entries() {
    let x = DynAdTensor::new_reverse_leaf(matrix2(&[1.0, 2.0, 3.0, 4.0])).unwrap();

    let sliced = x.take_prefix(1, 1).unwrap();
    let cotangent = DynAdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.5, -0.5], &[2, 1], MemoryOrder::ColumnMajor).unwrap(),
    );

    let grads = sliced.pullback_wrt(&cotangent, &[&x]).unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2, 2]);
    assert_eq!(
        as_slice(grad.as_f64().unwrap().primal()),
        &[1.5, -0.5, 0.0, 0.0]
    );
}

#[test]
fn dyn_ad_tensor_diag_embed_preserves_reverse_pullback() {
    let x = DynAdTensor::new_reverse_leaf(vector(&[2.0, 3.0])).unwrap();

    let diag = x.diag_embed(2).unwrap();
    assert!(diag.is_diag());
    let cotangent = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[0.5, -1.0]), 2).unwrap(),
    );

    let grads = diag.pullback_wrt(&cotangent, &[&x]).unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2]);
    assert_eq!(as_slice(grad.as_f64().unwrap().primal()), &[0.5, -1.0]);
}
