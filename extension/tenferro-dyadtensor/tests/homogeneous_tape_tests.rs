use chainrules::Tape;
use num_complex::Complex64;
use tenferro_dyadtensor::{
    ad, set_default_runtime, AdTensor, DynAdTensor, Error, RuntimeContext, StructuredTensor,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar(value: f64) -> Tensor<f64> {
    Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn c64_vector(values: &[Complex64]) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn reverse_scale_accepts_rank0_tensor_scalar_on_same_tape() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let x: DynAdTensor = AdTensor::new_reverse_leaf(vector(&[2.0, 3.0]), &tape)
        .unwrap()
        .into();
    let alpha: DynAdTensor = AdTensor::new_reverse_leaf(scalar(2.0), &tape)
        .unwrap()
        .into();

    let out = x.scale(&alpha).unwrap();
    let cotangent = AdTensor::new_primal(vector(&[0.5, -1.0]));
    let grads = ad::pullback_wrt(
        out.as_f64().unwrap(),
        &cotangent,
        &[x.as_f64().unwrap(), alpha.as_f64().unwrap()],
    )
    .unwrap();

    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].as_ref().unwrap().logical_dims(), &[2]);
    assert_eq!(grads[1].as_ref().unwrap().logical_dims(), &[]);
}

#[test]
fn structured_qr_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let x = AdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
        &tape,
    )
    .unwrap();

    let err = match ad::qr(&x) {
        Ok(_) => panic!("structured reverse qr should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "qr_ad_structured"));
}

#[test]
fn real_part_reverse_rejects_mixed_dtype_graphs() {
    let tape = Tape::<tenferro_dyadtensor::DynTensor>::new();
    let x: DynAdTensor = AdTensor::new_reverse_leaf(
        c64_vector(&[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]),
        &tape,
    )
    .unwrap()
    .into();

    let err = match x.real_part() {
        Ok(_) => panic!("real_part reverse should reject mixed-dtype graphs"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "real_part_reverse"));
}
