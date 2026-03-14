use num_complex::Complex64;
use tenferro_dyadtensor::{set_default_runtime, AdMode, DynAdTensor, RuntimeContext};
use tenferro_prims::CpuContext;

mod support;

use support::{
    dyn_rank0_value_f64, dyn_values_f64, primal_rank0_f64, reverse_rank0_f64,
    reverse_rank0_f64_like, reverse_vector_f64, reverse_vector_f64_like, vector_c64, vector_f64,
};

#[test]
fn scale_registers_reverse_gradients_for_tensor_and_scalar_inputs() {
    let x = reverse_vector_f64(&[1.0, 2.0]);
    let a = reverse_rank0_f64_like(3.0_f64, &x);

    let out = x.scale(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let cotangent = DynAdTensor::new_primal(vector_f64(&[0.5, 1.25]));
    let grads = out.pullback_wrt(&cotangent, &[&x, &a]).unwrap();

    assert_eq!(dyn_values_f64(grads[0].as_ref().unwrap()), &[1.5, 3.75]);
    assert_eq!(dyn_rank0_value_f64(grads[1].as_ref().unwrap()), 3.0);
}

#[test]
fn axpby_registers_reverse_gradients_for_all_inputs() {
    let x = reverse_vector_f64(&[1.0, 4.0]);
    let y = reverse_vector_f64_like(&[3.0, -1.0], &x);
    let a = reverse_rank0_f64_like(2.0_f64, &x);
    let b = reverse_rank0_f64_like(-1.0_f64, &x);

    let out = x.axpby(&a, &y, &b).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let cotangent = DynAdTensor::new_primal(vector_f64(&[0.5, -0.25]));
    let grads = out.pullback_wrt(&cotangent, &[&x, &y, &a, &b]).unwrap();

    assert_eq!(dyn_values_f64(grads[0].as_ref().unwrap()), &[1.0, -0.5]);
    assert_eq!(dyn_values_f64(grads[1].as_ref().unwrap()), &[-0.5, 0.25]);
    assert_eq!(dyn_rank0_value_f64(grads[2].as_ref().unwrap()), -0.5);
    assert_eq!(dyn_rank0_value_f64(grads[3].as_ref().unwrap()), 1.75);
}

#[test]
fn div_scalar_registers_reverse_gradients_for_tensor_and_scalar_inputs() {
    let x = reverse_vector_f64(&[2.0, 4.0]);
    let a = reverse_rank0_f64_like(2.0_f64, &x);

    let out = x.div_scalar(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let cotangent = DynAdTensor::new_primal(vector_f64(&[0.5, -1.0]));
    let grads = out.pullback_wrt(&cotangent, &[&x, &a]).unwrap();

    assert_eq!(dyn_values_f64(grads[0].as_ref().unwrap()), &[0.25, -0.5]);
    assert_eq!(dyn_rank0_value_f64(grads[1].as_ref().unwrap()), 0.75);
}

#[test]
fn scale_propagates_reverse_gradients_through_unary_scalar_coefficients() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = reverse_vector_f64(&[1.0, 2.0]);
    let a = reverse_rank0_f64_like(4.0_f64, &x);
    let coeff = a.sqrt().unwrap();

    let out = x.scale(&coeff).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let cotangent = DynAdTensor::new_primal(vector_f64(&[0.5, 1.25]));
    let grads = out.pullback_wrt(&cotangent, &[&x, &a]).unwrap();

    assert_eq!(dyn_values_f64(grads[0].as_ref().unwrap()), &[1.0, 2.5]);
    assert_eq!(dyn_rank0_value_f64(grads[1].as_ref().unwrap()), 0.75);
}

#[test]
fn scale_propagates_reverse_gradients_through_negated_scalar_coefficients() {
    let x = DynAdTensor::new_primal(vector_f64(&[2.0, -1.0]));
    let a = reverse_rank0_f64(3.0_f64);
    let coeff = a.scale(&primal_rank0_f64(-1.0)).unwrap();

    let out = x.scale(&coeff).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let cotangent = DynAdTensor::new_primal(vector_f64(&[0.5, -1.25]));
    let grads = out.pullback_wrt(&cotangent, &[&a]).unwrap();

    assert_eq!(dyn_rank0_value_f64(grads[0].as_ref().unwrap()), -2.25);
}

#[test]
fn scale_propagates_reverse_gradients_through_negative_real_promotion() {
    let x = DynAdTensor::new_primal(c64_vec(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
    ]));
    let coeff = reverse_rank0_f64(-4.0_f64);

    let out = x.scale(&coeff).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);
    let cotangent = DynAdTensor::new_primal(c64_vec(&[
        Complex64::new(0.5, 0.0),
        Complex64::new(1.25, 0.0),
    ]));
    let grads = out.pullback_wrt(&cotangent, &[&coeff]).unwrap();
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[3.0]
    );
}

fn c64_vec(values: &[Complex64]) -> tenferro_tensor::Tensor<Complex64> {
    vector_c64(values)
}
