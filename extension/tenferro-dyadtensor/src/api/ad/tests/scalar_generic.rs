use crate::{NodeId, TapeId};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    add_ad, atan2_ad, cos_ad, exp_ad, log_ad, mean_ad, set_default_runtime, sin_ad, std_ad,
    tanh_ad, var_ad, AdTensor, RuntimeContext,
};

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
    let out_log = log_ad(&ad_x).run().unwrap();
    assert_eq!(out_log.dims(), &[2]);
    let out_sin = sin_ad(&ad_x).run().unwrap();
    let out_cos = cos_ad(&ad_x).run().unwrap();
    let out_tanh = tanh_ad(&ad_x).run().unwrap();
    assert_eq!(out_sin.dims(), &[2]);
    assert_eq!(out_cos.dims(), &[2]);
    assert_eq!(out_tanh.dims(), &[2]);

    let out_binary = add_ad(&ad_x, &ad_y).run().unwrap();
    assert_eq!(out_binary.dims(), &[2]);
    let out_atan2 = atan2_ad(&ad_y, &ad_x).run().unwrap();
    assert_eq!(out_atan2.dims(), &[2]);

    let out_reduced = mean_ad(&out_binary).run().unwrap();
    assert_eq!(out_reduced.dims(), &[]);
    let out_var = var_ad(&out_binary).run().unwrap();
    let out_std = std_ad(&out_binary).run().unwrap();
    assert_eq!(out_var.dims(), &[]);
    assert_eq!(out_std.dims(), &[]);

    let eager_unary = crate::ad::exp(&ad_x).unwrap();
    let eager_sin = crate::ad::sin(&ad_x).unwrap();
    let eager_cos = crate::ad::cos(&ad_x).unwrap();
    let eager_tanh = crate::ad::tanh(&ad_x).unwrap();
    let eager_binary = crate::ad::add(&ad_x, &ad_y).unwrap();
    let eager_mean = crate::ad::mean(&eager_binary).unwrap();

    assert_eq!(eager_unary.dims(), &[2]);
    assert_eq!(eager_sin.dims(), &[2]);
    assert_eq!(eager_cos.dims(), &[2]);
    assert_eq!(eager_tanh.dims(), &[2]);
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

#[test]
fn log_ad_forward_matches_inverse_scaling_rule() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = tensor_from_slice(&[2.0, 4.0], &[2]);
    let dx = tensor_from_slice(&[3.0, -8.0], &[2]);
    let ad_x = AdTensor::new_forward(x, dx).unwrap();

    let out = log_ad(&ad_x).run().unwrap();
    let tangent = out.tangent().unwrap().clone();
    let expected = tensor_from_slice(&[1.5, -2.0], &[2]);

    assert!(max_abs_diff(&tangent, &expected) < 1e-12);
}

#[test]
fn sin_ad_forward_matches_cos_rule() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = tensor_from_slice(&[0.0, std::f64::consts::FRAC_PI_3], &[2]);
    let dx = tensor_from_slice(&[2.0, -3.0], &[2]);
    let ad_x = AdTensor::new_forward(x, dx).unwrap();

    let out = sin_ad(&ad_x).run().unwrap();
    let tangent = out.tangent().unwrap().clone();
    let expected = tensor_from_slice(&[2.0, -1.5], &[2]);

    assert!(max_abs_diff(&tangent, &expected) < 1e-12);
}

#[test]
fn atan2_and_moment_reductions_reverse_pullback_match_expected_dense_gradients() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let y = AdTensor::new_reverse(
        tensor_from_slice(&[3.0, 4.0], &[2]),
        NodeId(30),
        TapeId(40),
        None,
    )
    .unwrap();
    let x = AdTensor::new_reverse(
        tensor_from_slice(&[4.0, 3.0], &[2]),
        NodeId(31),
        TapeId(40),
        None,
    )
    .unwrap();

    let atan2_out = atan2_ad(&y, &x).run().unwrap();
    let cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, 1.0], &[2]));
    let atan2_grads = crate::ad::pullback_wrt(&atan2_out, &cotangent, &[&y, &x]).unwrap();
    let expected_dy = tensor_from_slice(&[0.16, 0.12], &[2]);
    let expected_dx = tensor_from_slice(&[-0.12, -0.16], &[2]);
    assert!(max_abs_diff(atan2_grads[0].as_ref().unwrap().payload(), &expected_dy) < 1e-12);
    assert!(max_abs_diff(atan2_grads[1].as_ref().unwrap().payload(), &expected_dx) < 1e-12);

    let z = AdTensor::new_reverse(
        tensor_from_slice(&[1.0, 3.0, 5.0, 7.0], &[2, 2]),
        NodeId(32),
        TapeId(41),
        None,
    )
    .unwrap();
    let var_out = var_ad(&z).run().unwrap();
    let std_out = std_ad(&z).run().unwrap();
    let scalar_cot = AdTensor::new_primal(tensor_from_slice(&[1.0], &[]));
    let var_grads = crate::ad::pullback_wrt(&var_out, &scalar_cot, &[&z]).unwrap();
    let std_grads = crate::ad::pullback_wrt(&std_out, &scalar_cot, &[&z]).unwrap();
    let expected_var = tensor_from_slice(&[-1.5, -0.5, 0.5, 1.5], &[2, 2]);
    let inv_two_std = 1.0 / (2.0 * 5.0_f64.sqrt());
    let expected_std = tensor_from_slice(
        &[
            -1.5 * inv_two_std,
            -0.5 * inv_two_std,
            0.5 * inv_two_std,
            1.5 * inv_two_std,
        ],
        &[2, 2],
    );
    assert!(max_abs_diff(var_grads[0].as_ref().unwrap().payload(), &expected_var) < 1e-12);
    assert!(max_abs_diff(std_grads[0].as_ref().unwrap().payload(), &expected_std) < 1e-12);
}

#[test]
fn cos_and_tanh_reverse_pullback_match_expected_dense_gradients() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = AdTensor::new_reverse(
        tensor_from_slice(&[0.0, std::f64::consts::FRAC_PI_6], &[2]),
        NodeId(50),
        TapeId(60),
        None,
    )
    .unwrap();

    let cos_out = cos_ad(&x).run().unwrap();
    let cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, 1.0], &[2]));
    let cos_grads = crate::ad::pullback_wrt(&cos_out, &cotangent, &[&x]).unwrap();
    let expected_cos = tensor_from_slice(&[-0.0, -0.5], &[2]);
    assert!(max_abs_diff(cos_grads[0].as_ref().unwrap().payload(), &expected_cos) < 1e-12);

    let y = AdTensor::new_reverse(
        tensor_from_slice(&[-1.0, 0.5], &[2]),
        NodeId(51),
        TapeId(61),
        None,
    )
    .unwrap();
    let tanh_out = tanh_ad(&y).run().unwrap();
    let scalar_cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, 1.0], &[2]));
    let tanh_grads = crate::ad::pullback_wrt(&tanh_out, &scalar_cotangent, &[&y]).unwrap();
    let expected_tanh = tensor_from_slice(
        &[
            1.0 - (-1.0f64).tanh().powi(2),
            1.0 - (0.5f64).tanh().powi(2),
        ],
        &[2],
    );
    assert!(max_abs_diff(tanh_grads[0].as_ref().unwrap().payload(), &expected_tanh) < 1e-12);
}
