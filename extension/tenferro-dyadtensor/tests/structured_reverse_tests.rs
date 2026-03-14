use chainrules::Tape;
use num_complex::Complex64;
use tenferro_dyadtensor::{ad, AdTensor, DynAdTensor, Error, StructuredTensor};
use tenferro_dyadtensor::{set_default_runtime, RuntimeContext};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

mod support;

use support::{reverse_rank0_f64, vector_c64, vector_f64};

fn scalar(value: f64) -> Tensor<f64> {
    Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn diag_scale_reverse_keeps_diag_cotangent_space() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x: DynAdTensor = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector_f64(&[2.0, 3.0]), 2).unwrap(),
        &tape,
    )
    .unwrap()
    .into();
    let a = reverse_rank0_f64(2.0_f64, &tape);
    let y = x.scale(&a).unwrap();
    let cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector_f64(&[1.0, 1.0]), 2).unwrap(),
    );

    let grads = ad::pullback_wrt(y.as_f64().unwrap(), &cotangent, &[x.as_f64().unwrap()]).unwrap();
    let grad = grads[0].as_ref().unwrap();
    assert!(grad.is_diag());
    assert_eq!(grad.payload().dims(), &[2]);
}

#[test]
fn diag_axpby_reverse_keeps_diag_cotangent_space() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x: DynAdTensor = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector_f64(&[2.0, 3.0]), 2).unwrap(),
        &tape,
    )
    .unwrap()
    .into();
    let y: DynAdTensor = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector_f64(&[5.0, 7.0]), 2).unwrap(),
        &tape,
    )
    .unwrap()
    .into();
    let a = reverse_rank0_f64(2.0_f64, &tape);
    let b = reverse_rank0_f64(-1.0_f64, &tape);
    let out = x.axpby(&a, &y, &b).unwrap();
    let cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector_f64(&[1.0, -0.5]), 2).unwrap(),
    );

    let grads = ad::pullback_wrt(
        out.as_f64().unwrap(),
        &cotangent,
        &[x.as_f64().unwrap(), y.as_f64().unwrap()],
    )
    .unwrap();

    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
    assert_eq!(grads[0].as_ref().unwrap().payload().dims(), &[2]);
    assert_eq!(grads[1].as_ref().unwrap().payload().dims(), &[2]);
}

#[test]
fn diag_complex_real_part_reverse_keeps_diag_cotangent_space() {
    let tape = Tape::<StructuredTensor<Complex64>>::new();
    let x: DynAdTensor = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(
            vector_c64(&[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]),
            2,
        )
        .unwrap(),
        &tape,
    )
    .unwrap()
    .into();

    let err = match x.real_part() {
        Ok(_) => panic!("real_part reverse should be unsupported for homogeneous mixed-dtype tape"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "real_part_reverse"));
}

#[test]
fn diag_complex_compose_complex_reverse_is_unsupported_on_homogeneous_tape() {
    let tape_a = Tape::<StructuredTensor<f64>>::new();
    let tape_b = Tape::<StructuredTensor<f64>>::new();
    let re: DynAdTensor = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector_f64(&[1.0, -3.0]), 2).unwrap(),
        &tape_a,
    )
    .unwrap()
    .into();
    let im: DynAdTensor = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector_f64(&[2.0, 4.0]), 2).unwrap(),
        &tape_b,
    )
    .unwrap()
    .into();

    let err = match DynAdTensor::compose_complex(re.clone(), im.clone()) {
        Ok(_) => panic!("compose_complex reverse should be unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "mixed_dtype_tensor_reverse"));
}

#[test]
fn root_einsum_keeps_diag_output_in_structured_carrier() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector_f64(&[1.0, 2.0]), 2).unwrap(),
    );
    let b = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector_f64(&[3.0, 4.0]), 2).unwrap(),
    );

    let out = ad::einsum("ij,jk->ik", &[&a, &b]).unwrap();

    assert!(out.is_diag());
    assert_eq!(out.primal().dims(), &[2]);
    assert_eq!(out.dims(), &[2, 2]);
}

#[test]
fn root_einsum_reverse_keeps_diag_cotangent_space() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = Tape::<StructuredTensor<f64>>::new();
    let a = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector_f64(&[1.0, 2.0]), 2).unwrap(),
        &tape,
    )
    .unwrap();
    let b = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector_f64(&[3.0, 4.0]), 2).unwrap(),
        &tape,
    )
    .unwrap();

    let out = ad::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector_f64(&[0.5, -1.0]), 2).unwrap(),
    );
    let grads = ad::pullback_wrt(&out, &cotangent, &[&a, &b]).unwrap();

    assert!(out.is_diag());
    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
    assert_eq!(grads[0].as_ref().unwrap().payload().dims(), &[2]);
    assert_eq!(grads[1].as_ref().unwrap().payload().dims(), &[2]);
}

#[test]
fn root_sum_reverse_keeps_diag_cotangent_space() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector_f64(&[2.0, 3.0]), 2).unwrap(),
        &tape,
    )
    .unwrap();

    let out = ad::sum(&x).unwrap();
    let cotangent = AdTensor::new_primal(scalar(1.5));
    let grads = ad::pullback_wrt(&out, &cotangent, &[&x]).unwrap();

    assert_eq!(out.dims(), &[]);
    assert!(grads[0].as_ref().unwrap().is_diag());
    assert_eq!(grads[0].as_ref().unwrap().payload().dims(), &[2]);
}
