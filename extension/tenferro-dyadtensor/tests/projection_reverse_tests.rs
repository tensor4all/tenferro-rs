use chainrules::Tape;
use num_complex::Complex64;
use tenferro_dyadtensor::{AdTensor, DynAdTensor, Error};

mod support;

use support::{reverse_rank0_c64, reverse_rank0_f64, reverse_vector_c64, vector_f64};

#[test]
fn scalar_complex_real_part_reverse_is_unsupported_on_homogeneous_tape() {
    let tape = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let z = reverse_rank0_c64(Complex64::new(3.0, -4.0), &tape);

    let err = match z.real_part() {
        Ok(_) => panic!("real_part reverse should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "real_part_reverse"));
}

#[test]
fn scalar_complex_imag_part_reverse_is_unsupported_on_homogeneous_tape() {
    let tape = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let z = reverse_rank0_c64(Complex64::new(3.0, -4.0), &tape);

    let err = match z.imag_part() {
        Ok(_) => panic!("imag_part reverse should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "imag_part_reverse"));
}

#[test]
fn scalar_compose_complex_reverse_is_unsupported_on_homogeneous_tape() {
    let tape_a = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let tape_b = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let re = reverse_rank0_f64(2.0, &tape_a);
    let im = reverse_rank0_f64(-3.0, &tape_b);

    let err = match DynAdTensor::compose_complex(re, im) {
        Ok(_) => panic!("compose_complex reverse should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "mixed_dtype_tensor_reverse"));
}

#[test]
fn tensor_complex_real_part_reverse_is_unsupported_on_homogeneous_tape() {
    let tape = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let x = reverse_vector_c64(
        &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
        &tape,
    );

    let err = match x.real_part() {
        Ok(_) => panic!("real_part reverse should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "real_part_reverse"));
}

#[test]
fn tensor_compose_complex_primal_still_works_for_real_inputs() {
    let re: DynAdTensor = AdTensor::new_primal(vector_f64(&[1.0, -3.0])).into();
    let im: DynAdTensor = AdTensor::new_primal(vector_f64(&[2.0, 4.0])).into();

    let out = DynAdTensor::compose_complex(re, im).unwrap();
    let values = out.as_c64().unwrap().primal().buffer().as_slice().unwrap();
    assert_eq!(values[0], Complex64::new(1.0, 2.0));
    assert_eq!(values[1], Complex64::new(-3.0, 4.0));
}
