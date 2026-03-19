use super::{max_abs_diff, reverse_leaf_f64, tensor_from_vec_f64 as tensor_from_slice};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::Tape;

use crate::ops::{
    acos_ad, acosh_ad, add_ad, asin_ad, asinh_ad, atan2_ad, atan_ad, atanh_ad, cos_ad, cosh_ad,
    exp_ad, expm1_ad, hypot_ad, log1p_ad, log_ad, mean_ad, pow_ad, sin_ad, sinh_ad, sqrt_ad,
    std_ad, tanh_ad, var_ad,
};
use crate::{set_default_runtime, AdTensor, RuntimeContext};

#[test]
fn ad_unary_binary_reduction_generic_surface_exists() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let y = DenseTensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_x = AdTensor::new_primal(x);
    let ad_y = AdTensor::new_primal(y);

    let out_unary = exp_ad(&ad_x).run().unwrap();
    assert_eq!(out_unary.dims(), &[2]);
    let out_log = log_ad(&ad_x).run().unwrap();
    let out_sqrt = sqrt_ad(&ad_x).run().unwrap();
    let out_expm1 = expm1_ad(&ad_x).run().unwrap();
    let out_log1p = log1p_ad(&ad_x).run().unwrap();
    assert_eq!(out_log.dims(), &[2]);
    assert_eq!(out_sqrt.dims(), &[2]);
    assert_eq!(out_expm1.dims(), &[2]);
    assert_eq!(out_log1p.dims(), &[2]);
    let out_sin = sin_ad(&ad_x).run().unwrap();
    let out_cos = cos_ad(&ad_x).run().unwrap();
    let out_tanh = tanh_ad(&ad_x).run().unwrap();
    let out_asin = asin_ad(&ad_x).run().unwrap();
    let out_acos = acos_ad(&ad_x).run().unwrap();
    let out_atan = atan_ad(&ad_x).run().unwrap();
    let out_sinh = sinh_ad(&ad_x).run().unwrap();
    let out_cosh = cosh_ad(&ad_x).run().unwrap();
    let out_asinh = asinh_ad(&ad_x).run().unwrap();
    let out_acosh = acosh_ad(&AdTensor::new_primal(tensor_from_slice(&[2.0, 3.0], &[2])))
        .run()
        .unwrap();
    let out_atanh = atanh_ad(&AdTensor::new_primal(tensor_from_slice(
        &[0.25, -0.5],
        &[2],
    )))
    .run()
    .unwrap();
    assert_eq!(out_sin.dims(), &[2]);
    assert_eq!(out_cos.dims(), &[2]);
    assert_eq!(out_tanh.dims(), &[2]);
    assert_eq!(out_asin.dims(), &[2]);
    assert_eq!(out_acos.dims(), &[2]);
    assert_eq!(out_atan.dims(), &[2]);
    assert_eq!(out_sinh.dims(), &[2]);
    assert_eq!(out_cosh.dims(), &[2]);
    assert_eq!(out_asinh.dims(), &[2]);
    assert_eq!(out_acosh.dims(), &[2]);
    assert_eq!(out_atanh.dims(), &[2]);

    let out_binary = add_ad(&ad_x, &ad_y).run().unwrap();
    assert_eq!(out_binary.dims(), &[2]);
    let out_atan2 = atan2_ad(&ad_y, &ad_x).run().unwrap();
    let out_pow = pow_ad(&ad_y, &ad_x).run().unwrap();
    let out_hypot = hypot_ad(&ad_y, &ad_x).run().unwrap();
    assert_eq!(out_atan2.dims(), &[2]);
    assert_eq!(out_pow.dims(), &[2]);
    assert_eq!(out_hypot.dims(), &[2]);

    let out_reduced = mean_ad(&out_binary).run().unwrap();
    assert_eq!(out_reduced.dims(), &[]);
    let out_var = var_ad(&out_binary).run().unwrap();
    let out_std = std_ad(&out_binary).run().unwrap();
    assert_eq!(out_var.dims(), &[]);
    assert_eq!(out_std.dims(), &[]);

    let eager_unary = crate::ops::ad::exp(&ad_x).unwrap();
    let eager_sqrt = crate::ops::ad::sqrt(&ad_x).unwrap();
    let eager_expm1 = crate::ops::ad::expm1(&ad_x).unwrap();
    let eager_log1p = crate::ops::ad::log1p(&ad_x).unwrap();
    let eager_sin = crate::ops::ad::sin(&ad_x).unwrap();
    let eager_cos = crate::ops::ad::cos(&ad_x).unwrap();
    let eager_tanh = crate::ops::ad::tanh(&ad_x).unwrap();
    let eager_asin = crate::ops::ad::asin(&ad_x).unwrap();
    let eager_acos = crate::ops::ad::acos(&ad_x).unwrap();
    let eager_atan = crate::ops::ad::atan(&ad_x).unwrap();
    let eager_sinh = crate::ops::ad::sinh(&ad_x).unwrap();
    let eager_cosh = crate::ops::ad::cosh(&ad_x).unwrap();
    let eager_asinh = crate::ops::ad::asinh(&ad_x).unwrap();
    let eager_pow = crate::ops::ad::pow(&ad_y, &ad_x).unwrap();
    let eager_hypot = crate::ops::ad::hypot(&ad_y, &ad_x).unwrap();
    let eager_binary = crate::ops::ad::add(&ad_x, &ad_y).unwrap();
    let eager_mean = crate::ops::ad::mean(&eager_binary).unwrap();

    assert_eq!(eager_unary.dims(), &[2]);
    assert_eq!(eager_sqrt.dims(), &[2]);
    assert_eq!(eager_expm1.dims(), &[2]);
    assert_eq!(eager_log1p.dims(), &[2]);
    assert_eq!(eager_sin.dims(), &[2]);
    assert_eq!(eager_cos.dims(), &[2]);
    assert_eq!(eager_tanh.dims(), &[2]);
    assert_eq!(eager_asin.dims(), &[2]);
    assert_eq!(eager_acos.dims(), &[2]);
    assert_eq!(eager_atan.dims(), &[2]);
    assert_eq!(eager_sinh.dims(), &[2]);
    assert_eq!(eager_cosh.dims(), &[2]);
    assert_eq!(eager_asinh.dims(), &[2]);
    assert_eq!(eager_pow.dims(), &[2]);
    assert_eq!(eager_hypot.dims(), &[2]);
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

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(tensor_from_slice(&[1.0, 2.0], &[2]), &tape);
    let y = reverse_leaf_f64(tensor_from_slice(&[3.0, 4.0], &[2]), &tape);

    let added = add_ad(&x, &y).run().unwrap();
    let out = mean_ad(&added).run().unwrap();
    let cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0], &[]));

    let grads = crate::ops::ad::pullback_wrt(&out, &cotangent, &[&x, &y]).unwrap();
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
fn sqrt_and_log1p_forward_match_expected_rules() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = tensor_from_slice(&[4.0, 3.0], &[2]);
    let dx = tensor_from_slice(&[6.0, -8.0], &[2]);
    let ad_x = AdTensor::new_forward(x.clone(), dx).unwrap();

    let sqrt_out = sqrt_ad(&ad_x).run().unwrap();
    let sqrt_tangent = sqrt_out.tangent().unwrap().clone();
    let sqrt_expected = tensor_from_slice(&[1.5, -4.0 / 3.0_f64.sqrt()], &[2]);
    assert!(max_abs_diff(&sqrt_tangent, &sqrt_expected) < 1e-12);

    let log1p_out = log1p_ad(&ad_x).run().unwrap();
    let log1p_tangent = log1p_out.tangent().unwrap().clone();
    let log1p_expected = tensor_from_slice(&[6.0 / 5.0, -2.0], &[2]);
    assert!(max_abs_diff(&log1p_tangent, &log1p_expected) < 1e-12);
}

#[test]
fn expm1_reverse_pullback_matches_exp_rule() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(tensor_from_slice(&[0.0, 1.0], &[2]), &tape);

    let out = expm1_ad(&x).run().unwrap();
    let cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, -2.0], &[2]));
    let grads = crate::ops::ad::pullback_wrt(&out, &cotangent, &[&x]).unwrap();
    let expected = tensor_from_slice(&[1.0, -2.0 * std::f64::consts::E], &[2]);

    assert!(max_abs_diff(grads[0].as_ref().unwrap().payload(), &expected) < 1e-12);
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

    let tape_xy = Tape::<crate::DynTensor>::new();
    let y = reverse_leaf_f64(tensor_from_slice(&[3.0, 4.0], &[2]), &tape_xy);
    let x = reverse_leaf_f64(tensor_from_slice(&[4.0, 3.0], &[2]), &tape_xy);

    let atan2_out = atan2_ad(&y, &x).run().unwrap();
    let cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, 1.0], &[2]));
    let atan2_grads = crate::ops::ad::pullback_wrt(&atan2_out, &cotangent, &[&y, &x]).unwrap();
    let expected_dy = tensor_from_slice(&[0.16, 0.12], &[2]);
    let expected_dx = tensor_from_slice(&[-0.12, -0.16], &[2]);
    assert!(max_abs_diff(atan2_grads[0].as_ref().unwrap().payload(), &expected_dy) < 1e-12);
    assert!(max_abs_diff(atan2_grads[1].as_ref().unwrap().payload(), &expected_dx) < 1e-12);

    let tape_z = Tape::<crate::DynTensor>::new();
    let z = reverse_leaf_f64(tensor_from_slice(&[1.0, 3.0, 5.0, 7.0], &[2, 2]), &tape_z);
    let var_out = var_ad(&z).run().unwrap();
    let std_out = std_ad(&z).run().unwrap();
    let scalar_cot = AdTensor::new_primal(tensor_from_slice(&[1.0], &[]));
    let var_grads = crate::ops::ad::pullback_wrt(&var_out, &scalar_cot, &[&z]).unwrap();
    let std_grads = crate::ops::ad::pullback_wrt(&std_out, &scalar_cot, &[&z]).unwrap();
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
fn remaining_analytic_unary_forward_rules_match_reference_derivatives() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let trig_input = tensor_from_slice(&[0.25, -0.5], &[2]);
    let trig_tangent = tensor_from_slice(&[2.0, -3.0], &[2]);
    let trig_ad = AdTensor::new_forward(trig_input.clone(), trig_tangent.clone()).unwrap();

    let asin_out = asin_ad(&trig_ad).run().unwrap();
    let asin_expected = tensor_from_slice(
        &[
            2.0 / (1.0_f64 - 0.25_f64.powi(2)).sqrt(),
            -3.0 / (1.0_f64 - 0.5_f64.powi(2)).sqrt(),
        ],
        &[2],
    );
    assert!(max_abs_diff(asin_out.tangent().unwrap(), &asin_expected) < 1e-12);

    let atan_out = atan_ad(&trig_ad).run().unwrap();
    let atan_expected = tensor_from_slice(
        &[
            2.0 / (1.0 + 0.25_f64.powi(2)),
            -3.0 / (1.0 + 0.5_f64.powi(2)),
        ],
        &[2],
    );
    assert!(max_abs_diff(atan_out.tangent().unwrap(), &atan_expected) < 1e-12);

    let hyper_input = tensor_from_slice(&[0.25, 1.25], &[2]);
    let hyper_tangent = tensor_from_slice(&[1.5, -0.75], &[2]);
    let hyper_ad = AdTensor::new_forward(hyper_input.clone(), hyper_tangent.clone()).unwrap();

    let sinh_out = sinh_ad(&hyper_ad).run().unwrap();
    let sinh_expected = tensor_from_slice(&[1.5 * 0.25_f64.cosh(), -0.75 * 1.25_f64.cosh()], &[2]);
    assert!(max_abs_diff(sinh_out.tangent().unwrap(), &sinh_expected) < 1e-12);

    let asinh_out = asinh_ad(&hyper_ad).run().unwrap();
    let asinh_expected = tensor_from_slice(
        &[
            1.5 / (1.0_f64 + 0.25_f64.powi(2)).sqrt(),
            -0.75 / (1.0_f64 + 1.25_f64.powi(2)).sqrt(),
        ],
        &[2],
    );
    assert!(max_abs_diff(asinh_out.tangent().unwrap(), &asinh_expected) < 1e-12);
}

#[test]
fn pow_and_hypot_reverse_pullbacks_match_reference_gradients() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape_pow = Tape::<crate::DynTensor>::new();
    let base = reverse_leaf_f64(tensor_from_slice(&[2.0, 3.0], &[2]), &tape_pow);
    let exponent = reverse_leaf_f64(tensor_from_slice(&[3.0, 2.0], &[2]), &tape_pow);
    let pow_out = pow_ad(&base, &exponent).run().unwrap();
    let pow_cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, -0.5], &[2]));
    let pow_grads =
        crate::ops::ad::pullback_wrt(&pow_out, &pow_cotangent, &[&base, &exponent]).unwrap();
    let expected_base = tensor_from_slice(&[12.0, -3.0], &[2]);
    let expected_exponent =
        tensor_from_slice(&[8.0 * 2.0_f64.ln(), -0.5 * 9.0 * 3.0_f64.ln()], &[2]);
    assert!(max_abs_diff(pow_grads[0].as_ref().unwrap().payload(), &expected_base) < 1e-12);
    assert!(max_abs_diff(pow_grads[1].as_ref().unwrap().payload(), &expected_exponent) < 1e-12);

    let tape_hypot = Tape::<crate::DynTensor>::new();
    let lhs = reverse_leaf_f64(tensor_from_slice(&[3.0, 5.0], &[2]), &tape_hypot);
    let rhs = reverse_leaf_f64(tensor_from_slice(&[4.0, 12.0], &[2]), &tape_hypot);
    let hypot_out = hypot_ad(&lhs, &rhs).run().unwrap();
    let hypot_cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, -2.0], &[2]));
    let hypot_grads =
        crate::ops::ad::pullback_wrt(&hypot_out, &hypot_cotangent, &[&lhs, &rhs]).unwrap();
    let expected_lhs = tensor_from_slice(&[3.0 / 5.0, -10.0 / 13.0], &[2]);
    let expected_rhs = tensor_from_slice(&[4.0 / 5.0, -24.0 / 13.0], &[2]);
    assert!(max_abs_diff(hypot_grads[0].as_ref().unwrap().payload(), &expected_lhs) < 1e-12);
    assert!(max_abs_diff(hypot_grads[1].as_ref().unwrap().payload(), &expected_rhs) < 1e-12);
}

#[test]
fn cos_and_tanh_reverse_pullback_match_expected_dense_gradients() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape_cos = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(
        tensor_from_slice(&[0.0, std::f64::consts::FRAC_PI_6], &[2]),
        &tape_cos,
    );

    let cos_out = cos_ad(&x).run().unwrap();
    let cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, 1.0], &[2]));
    let cos_grads = crate::ops::ad::pullback_wrt(&cos_out, &cotangent, &[&x]).unwrap();
    let expected_cos = tensor_from_slice(&[-0.0, -0.5], &[2]);
    assert!(max_abs_diff(cos_grads[0].as_ref().unwrap().payload(), &expected_cos) < 1e-12);

    let tape_tanh = Tape::<crate::DynTensor>::new();
    let y = reverse_leaf_f64(tensor_from_slice(&[-1.0, 0.5], &[2]), &tape_tanh);
    let tanh_out = tanh_ad(&y).run().unwrap();
    let scalar_cotangent = AdTensor::new_primal(tensor_from_slice(&[1.0, 1.0], &[2]));
    let tanh_grads = crate::ops::ad::pullback_wrt(&tanh_out, &scalar_cotangent, &[&y]).unwrap();
    let expected_tanh = tensor_from_slice(
        &[
            1.0 - (-1.0f64).tanh().powi(2),
            1.0 - (0.5f64).tanh().powi(2),
        ],
        &[2],
    );
    assert!(max_abs_diff(tanh_grads[0].as_ref().unwrap().payload(), &expected_tanh) < 1e-12);
}

#[test]
fn remaining_analytic_unary_reverse_pullbacks_match_reference_derivatives() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    fn assert_reverse_unary(
        input: DenseTensor<f64>,
        cotangent: DenseTensor<f64>,
        expected: DenseTensor<f64>,
        run: impl Fn(&AdTensor<f64>) -> crate::Result<AdTensor<f64>>,
    ) {
        let tape = Tape::<crate::DynTensor>::new();
        let x = reverse_leaf_f64(input, &tape);
        let out = run(&x).unwrap();
        let cotangent = AdTensor::new_primal(cotangent);
        let grads = crate::ops::ad::pullback_wrt(&out, &cotangent, &[&x]).unwrap();
        assert!(max_abs_diff(grads[0].as_ref().unwrap().payload(), &expected) < 1e-12);
    }

    assert_reverse_unary(
        tensor_from_slice(&[2.0, 4.0], &[2]),
        tensor_from_slice(&[1.0, -2.0], &[2]),
        tensor_from_slice(&[0.5, -0.5], &[2]),
        |x| log_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.5, 3.0], &[2]),
        tensor_from_slice(&[2.0, -4.0], &[2]),
        tensor_from_slice(&[4.0 / 3.0, -1.0], &[2]),
        |x| log1p_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.0, std::f64::consts::FRAC_PI_3], &[2]),
        tensor_from_slice(&[1.0, -2.0], &[2]),
        tensor_from_slice(&[1.0, -1.0], &[2]),
        |x| sin_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.25, -0.5], &[2]),
        tensor_from_slice(&[2.0, -3.0], &[2]),
        tensor_from_slice(
            &[
                2.0 / (1.0_f64 - 0.25_f64.powi(2)).sqrt(),
                -3.0 / (1.0_f64 - 0.5_f64.powi(2)).sqrt(),
            ],
            &[2],
        ),
        |x| asin_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.25, -0.5], &[2]),
        tensor_from_slice(&[2.0, -3.0], &[2]),
        tensor_from_slice(
            &[
                -2.0 / (1.0_f64 - 0.25_f64.powi(2)).sqrt(),
                3.0 / (1.0_f64 - 0.5_f64.powi(2)).sqrt(),
            ],
            &[2],
        ),
        |x| acos_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.25, -0.5], &[2]),
        tensor_from_slice(&[2.0, -3.0], &[2]),
        tensor_from_slice(
            &[
                2.0 / (1.0_f64 + 0.25_f64.powi(2)),
                -3.0 / (1.0_f64 + 0.5_f64.powi(2)),
            ],
            &[2],
        ),
        |x| atan_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.25, 1.25], &[2]),
        tensor_from_slice(&[1.5, -0.75], &[2]),
        tensor_from_slice(&[1.5 * 0.25_f64.cosh(), -0.75 * 1.25_f64.cosh()], &[2]),
        |x| sinh_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.25, 1.25], &[2]),
        tensor_from_slice(&[1.5, -0.75], &[2]),
        tensor_from_slice(&[1.5 * 0.25_f64.sinh(), -0.75 * 1.25_f64.sinh()], &[2]),
        |x| cosh_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.25, 1.25], &[2]),
        tensor_from_slice(&[1.5, -0.75], &[2]),
        tensor_from_slice(
            &[
                1.5 / (1.0_f64 + 0.25_f64.powi(2)).sqrt(),
                -0.75 / (1.0_f64 + 1.25_f64.powi(2)).sqrt(),
            ],
            &[2],
        ),
        |x| asinh_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[2.0, 3.0], &[2]),
        tensor_from_slice(&[1.25, -0.5], &[2]),
        tensor_from_slice(
            &[
                1.25 / (2.0_f64 - 1.0).sqrt() / (2.0_f64 + 1.0).sqrt(),
                -0.5 / (3.0_f64 - 1.0).sqrt() / (3.0_f64 + 1.0).sqrt(),
            ],
            &[2],
        ),
        |x| acosh_ad(x).run(),
    );
    assert_reverse_unary(
        tensor_from_slice(&[0.25, -0.5], &[2]),
        tensor_from_slice(&[2.0, -3.0], &[2]),
        tensor_from_slice(
            &[
                2.0 / (1.0_f64 - 0.25_f64.powi(2)),
                -3.0 / (1.0_f64 - 0.5_f64.powi(2)),
            ],
            &[2],
        ),
        |x| atanh_ad(x).run(),
    );
}
