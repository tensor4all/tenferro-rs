use chainrules::Tape;
use num_complex::Complex64;
use tenferro_dyadtensor::{AdMode, AdTensor, DynAdTensor, Error};
use tenferro_tensor::Tensor;

mod support;

use support::{forward_rank0_f64, reverse_rank0_f64, reverse_vector_c64, vector_c64};

fn c64_vec(values: &[Complex64]) -> Tensor<Complex64> {
    vector_c64(values)
}

#[test]
fn c64_tensor_scale_accepts_f64_scalar_in_forward_mode() {
    let x: DynAdTensor = AdTensor::new_forward(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .unwrap()
    .into();
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
    let x: DynAdTensor = AdTensor::new_forward(
        c64_vec(&[Complex64::new(4.0, 0.0), Complex64::new(-6.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .unwrap()
    .into();
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
    let x: DynAdTensor = AdTensor::new_forward(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .unwrap()
    .into();
    let y: DynAdTensor = AdTensor::new_forward(
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0)]),
        c64_vec(&[Complex64::new(-0.5, 0.0), Complex64::new(0.25, 0.0)]),
    )
    .unwrap()
    .into();
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
fn c64_tensor_scale_reverse_rejects_real_scalar_input() {
    let x = reverse_vector_c64(
        &[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)],
        &Tape::<tenferro_dyadtensor::DynTensor>::new(),
    );
    let tape = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let a = reverse_rank0_f64(2.0_f64, &tape);

    let err = match x.scale(&a) {
        Ok(_) => panic!("mixed-dtype reverse scale should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "mixed_dtype_tensor_reverse"));
}

#[test]
fn c64_tensor_div_scalar_reverse_rejects_real_scalar_input() {
    let x = reverse_vector_c64(
        &[Complex64::new(4.0, 0.0), Complex64::new(-6.0, 0.0)],
        &Tape::<tenferro_dyadtensor::DynTensor>::new(),
    );
    let tape = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let a = reverse_rank0_f64(2.0_f64, &tape);

    let err = match x.div_scalar(&a) {
        Ok(_) => panic!("mixed-dtype reverse div_scalar should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "mixed_dtype_tensor_reverse"));
}
