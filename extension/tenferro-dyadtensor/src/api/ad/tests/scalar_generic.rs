use crate::{NodeId, TapeId};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{add_ad, exp_ad, mean_ad, set_default_runtime, AdTensor, RuntimeContext};

fn tensor_from_slice(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn tensor_to_vec(tensor: &Tensor<f64>) -> Vec<f64> {
    let tensor = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = tensor.offset() as usize;
    let len: usize = tensor.dims().iter().product();
    tensor.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn max_abs_diff(actual: &Tensor<f64>, expected: &Tensor<f64>) -> f64 {
    assert_eq!(actual.dims(), expected.dims());
    tensor_to_vec(actual)
        .iter()
        .zip(tensor_to_vec(expected).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn ad_unary_binary_reduction_generic_surface_exists() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let y = Tensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_x = AdTensor::new_primal(x);
    let ad_y = AdTensor::new_primal(y);

    let out_unary = exp_ad(&ad_x).run().unwrap();
    assert_eq!(out_unary.dims(), &[2]);

    let out_binary = add_ad(&ad_x, &ad_y).run().unwrap();
    assert_eq!(out_binary.dims(), &[2]);

    let out_reduced = mean_ad(&out_binary).run().unwrap();
    assert_eq!(out_reduced.dims(), &[]);

    let eager_unary = crate::ad::exp(&ad_x).unwrap();
    let eager_binary = crate::ad::add(&ad_x, &ad_y).unwrap();
    let eager_mean = crate::ad::mean(&eager_binary).unwrap();

    assert_eq!(eager_unary.dims(), &[2]);
    assert_eq!(eager_binary.dims(), &[2]);
    assert_eq!(eager_mean.dims(), &[]);
}

#[test]
fn exp_ad_forward_matches_elementwise_derivative() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = tensor_from_slice(&[0.0, 1.0], &[2]);
    let dx = tensor_from_slice(&[1.5, -2.0], &[2]);
    let ad_x = AdTensor::new_forward(x, dx).unwrap();

    let out = exp_ad(&ad_x).run().unwrap();
    let tangent = out.tangent().unwrap().clone();
    let expected = tensor_from_slice(&[1.5, -2.0 * std::f64::consts::E], &[2]);

    assert!(max_abs_diff(&tangent, &expected) < 1e-12);
}

#[test]
fn mean_add_reverse_pullback_matches_expected_dense_gradients() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = AdTensor::new_reverse(
        tensor_from_slice(&[1.0, 2.0], &[2]),
        NodeId(10),
        TapeId(20),
        None,
    )
    .unwrap();
    let y = AdTensor::new_reverse(
        tensor_from_slice(&[3.0, 4.0], &[2]),
        NodeId(11),
        TapeId(20),
        None,
    )
    .unwrap();

    let added = add_ad(&x, &y).run().unwrap();
    let out = mean_ad(&added).run().unwrap();
    let cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0], &[]));

    let grads = crate::ad::pullback_wrt(&out, &cotangent, &[&x, &y]).unwrap();
    let expected = tensor_from_slice(&[0.5, 0.5], &[2]);

    assert!(max_abs_diff(grads[0].as_ref().unwrap().payload(), &expected) < 1e-12);
    assert!(max_abs_diff(grads[1].as_ref().unwrap().payload(), &expected) < 1e-12);
}
