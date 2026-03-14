use num_complex::Complex64;
use tenferro_dyadtensor::{AdMode, DynAdTensor, DynTape};
use tenferro_tensor::Tensor;

mod support;

use support::{forward_rank0_f64, reverse_rank0_f64, reverse_vector_c64, vector_c64};

fn c64_vec(values: &[Complex64]) -> Tensor<Complex64> {
    vector_c64(values)
}

#[test]
fn c64_tensor_scale_accepts_f64_scalar_in_forward_mode() {
    let x = DynAdTensor::new_forward(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .unwrap();
    let a = forward_rank0_f64(2.0_f64, 0.25_f64);

    let out = x.scale(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Forward);

    let out_t = out.as_c64().unwrap();
    assert_eq!(
        out_t.primal().buffer().as_slice().unwrap(),
        &[Complex64::new(2.0, 0.0), Complex64::new(-6.0, 0.0)]
    );
    assert_eq!(
        out_t.tangent().unwrap().buffer().as_slice().unwrap(),
        &[Complex64::new(1.25, 0.0), Complex64::new(2.25, 0.0)]
    );
}

#[test]
fn c64_tensor_div_scalar_accepts_f64_scalar_in_forward_mode() {
    let x = DynAdTensor::new_forward(
        c64_vec(&[Complex64::new(4.0, 0.0), Complex64::new(-6.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .unwrap();
    let a = forward_rank0_f64(2.0_f64, 0.5_f64);

    let out = x.div_scalar(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Forward);

    let out_t = out.as_c64().unwrap();
    assert_eq!(
        out_t.primal().buffer().as_slice().unwrap(),
        &[Complex64::new(2.0, 0.0), Complex64::new(-3.0, 0.0)]
    );
    assert_eq!(
        out_t.tangent().unwrap().buffer().as_slice().unwrap(),
        &[Complex64::new(-0.25, 0.0), Complex64::new(1.5, 0.0)]
    );
}

#[test]
fn c64_tensor_axpby_accepts_real_coefficients() {
    let x = DynAdTensor::new_forward(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .unwrap();
    let y = DynAdTensor::new_forward(
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0)]),
        c64_vec(&[Complex64::new(-0.5, 0.0), Complex64::new(0.25, 0.0)]),
    )
    .unwrap();
    let a = forward_rank0_f64(2.0_f64, 0.25_f64);
    let b = forward_rank0_f64(-0.5_f64, 1.0_f64);

    let out = x.axpby(&a, &y, &b).unwrap();
    assert_eq!(out.mode(), AdMode::Forward);

    let out_t = out.as_c64().unwrap();
    assert_eq!(
        out_t.primal().buffer().as_slice().unwrap(),
        &[Complex64::new(1.75, 0.0), Complex64::new(-7.0, 0.0)]
    );
    assert_eq!(
        out_t.tangent().unwrap().buffer().as_slice().unwrap(),
        &[Complex64::new(2.0, 0.0), Complex64::new(4.125, 0.0)]
    );
}

#[test]
fn c64_tensor_scale_reverse_casts_back_scalar_gradient_to_real_dtype() {
    let tape = DynTape::new();
    let x = reverse_vector_c64(
        &[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)],
        &tape,
    );
    let a = reverse_rank0_f64(2.0_f64, &tape);

    let out = x.scale(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let cotangent = DynAdTensor::new_primal(c64_vec(&[
        Complex64::new(0.5, 0.0),
        Complex64::new(-1.0, 0.0),
    ]));
    let grads = out.pullback_wrt(&cotangent, &[&x, &a]).unwrap();
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .as_c64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)]
    );
    assert_eq!(
        grads[1]
            .as_ref()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[3.5]
    );
}

#[test]
fn c64_tensor_div_scalar_reverse_casts_back_scalar_gradient_to_real_dtype() {
    let tape = DynTape::new();
    let x = reverse_vector_c64(
        &[Complex64::new(4.0, 0.0), Complex64::new(-6.0, 0.0)],
        &tape,
    );
    let a = reverse_rank0_f64(2.0_f64, &tape);

    let out = x.div_scalar(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let cotangent = DynAdTensor::new_primal(c64_vec(&[
        Complex64::new(0.5, 0.0),
        Complex64::new(-1.0, 0.0),
    ]));
    let grads = out.pullback_wrt(&cotangent, &[&x, &a]).unwrap();
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .as_c64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(0.25, 0.0), Complex64::new(-0.5, 0.0)]
    );
    assert_eq!(
        grads[1]
            .as_ref()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[-2.0]
    );
}
