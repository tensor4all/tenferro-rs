use num_complex::Complex64;
use tenferro_dyadtensor::{Error, Tensor};

mod support;

use support::{
    grad_wrt, reverse_rank0_c64, reverse_rank0_f64, reverse_rank0_f64_like, reverse_vector_c64,
    scalar_c64, vector_f64,
};

#[test]
fn scalar_complex_real_part_reverse_is_unsupported_on_homogeneous_tape() {
    let z = reverse_rank0_c64(Complex64::new(3.0, -4.0));

    let err = match z.real_part() {
        Ok(_) => panic!("real_part reverse should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "real_part_reverse"));
}

#[test]
fn scalar_complex_imag_part_reverse_is_unsupported_on_homogeneous_tape() {
    let z = reverse_rank0_c64(Complex64::new(3.0, -4.0));

    let err = match z.imag_part() {
        Ok(_) => panic!("imag_part reverse should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "imag_part_reverse"));
}

#[test]
fn scalar_compose_complex_reverse_splits_cotangent_back_into_real_components() {
    let re = reverse_rank0_f64(2.0);
    let im = reverse_rank0_f64_like(-3.0, &re);

    let z = Tensor::compose_complex(re.clone(), im.clone()).unwrap();
    let cotangent = Tensor::from_tensor(scalar_c64(Complex64::new(0.5, -1.25)));
    let grads = grad_wrt(&z, &cotangent, &[&re, &im]);
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
        &[0.5]
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
        &[-1.25]
    );
}

#[test]
fn tensor_complex_real_part_reverse_is_unsupported_on_homogeneous_tape() {
    let x = reverse_vector_c64(&[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]);

    let err = match x.real_part() {
        Ok(_) => panic!("real_part reverse should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "real_part_reverse"));
}

#[test]
fn tensor_compose_complex_primal_still_works_for_real_inputs() {
    let re = Tensor::from_tensor(vector_f64(&[1.0, -3.0]));
    let im = Tensor::from_tensor(vector_f64(&[2.0, 4.0]));

    let out = Tensor::compose_complex(re, im).unwrap();
    let values = out.as_c64().unwrap().primal().buffer().as_slice().unwrap();
    assert_eq!(values[0], Complex64::new(1.0, 2.0));
    assert_eq!(values[1], Complex64::new(-3.0, 4.0));
}
