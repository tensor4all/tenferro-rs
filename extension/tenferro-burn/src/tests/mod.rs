use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorPrimitive};
use tenferro_tensor::{MemoryOrder, Tensor as TfTensor};

use crate::{burn_to_tenferro, einsum, tenferro_to_burn};

#[test]
fn burn_to_tenferro_preserves_shape_and_row_major_values() {
    let device = Default::default();
    let burn = Tensor::<NdArray<f64>, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    let tenferro = burn_to_tenferro::<NdArray<f64>>(burn.into_primitive().tensor());

    assert_eq!(tenferro.dims(), &[2, 3]);
    assert_eq!(tenferro.strides(), &[3, 1]);
    assert_eq!(
        tenferro
            .into_contiguous(MemoryOrder::RowMajor)
            .try_into_data_vec()
            .unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn tenferro_to_burn_roundtrip_preserves_values() {
    let device = Default::default();
    let tenferro =
        TfTensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &[2, 1], 0).unwrap();

    let burn =
        Tensor::<NdArray<f64>, 2>::from_primitive(TensorPrimitive::Float(tenferro_to_burn::<
            NdArray<f64>,
        >(
            tenferro, &device
        )));

    assert_eq!(
        burn.into_data().to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn forward_einsum_matches_matrix_multiply() {
    let device = Default::default();
    let a = Tensor::<NdArray<f64>, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let b = Tensor::<NdArray<f64>, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);

    let c = einsum("ij,jk->ik", vec![a, b]);

    assert_eq!(
        c.into_data().to_vec::<f64>().unwrap(),
        vec![19.0, 22.0, 43.0, 50.0]
    );
}

#[test]
fn autodiff_unary_einsum_propagates_identity_gradient() {
    type Backend = Autodiff<NdArray<f64>>;

    let device = Default::default();
    let a = Tensor::<Backend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device).require_grad();

    let loss = einsum("ij->ij", vec![a.clone()]).sum();
    let grads = loss.backward();
    let grad_a = a.grad(&grads).unwrap();

    assert_eq!(
        grad_a.into_data().to_vec::<f64>().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn autodiff_binary_einsum_propagates_matmul_gradients() {
    type Backend = Autodiff<NdArray<f64>>;

    let device = Default::default();
    let a = Tensor::<Backend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device).require_grad();
    let b = Tensor::<Backend, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device).require_grad();

    let loss = einsum("ij,jk->ik", vec![a.clone(), b.clone()]).sum();
    let grads = loss.backward();
    let grad_a = a.grad(&grads).unwrap();
    let grad_b = b.grad(&grads).unwrap();

    assert_eq!(
        grad_a.into_data().to_vec::<f64>().unwrap(),
        vec![11.0, 15.0, 11.0, 15.0]
    );
    assert_eq!(
        grad_b.into_data().to_vec::<f64>().unwrap(),
        vec![4.0, 4.0, 6.0, 6.0]
    );
}
