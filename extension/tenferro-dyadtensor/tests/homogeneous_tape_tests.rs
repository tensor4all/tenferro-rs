use num_complex::Complex64;
use tenferro_dyadtensor::{set_default_runtime, Error, RuntimeContext, StructuredTensor, Tensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn vector(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar(value: f64) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn c64_vector(values: &[Complex64]) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn reverse_scale_accepts_rank0_tensor_scalar_on_same_tape() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut x = Tensor::from_tensor(vector(&[2.0, 3.0]));
    x.set_requires_grad(true).unwrap();
    let mut alpha = Tensor::from_tensor(scalar(2.0));
    alpha.set_requires_grad(true).unwrap();

    let out = x.scale(&alpha).unwrap();
    let cotangent = Tensor::from_tensor(vector(&[0.5, -1.0]));
    let grad_outputs = [cotangent];
    let grads = tenferro_dyadtensor::grad(
        &[&out],
        &[&x, &alpha],
        Some(&grad_outputs),
        Default::default(),
    )
    .unwrap();

    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].as_ref().unwrap().dims(), &[2]);
    assert_eq!(grads[1].as_ref().unwrap().dims(), &[]);
}

#[test]
fn structured_qr_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut x = Tensor::from_structured(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    );
    x.set_requires_grad(true).unwrap();

    let err = match x.qr() {
        Ok(_) => panic!("structured reverse qr should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "qr"));
}

#[test]
fn real_part_reverse_rejects_mixed_dtype_graphs() {
    let mut x = Tensor::from_tensor(c64_vector(&[
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 4.0),
    ]));
    x.set_requires_grad(true).unwrap();

    let err = match x.real_part() {
        Ok(_) => panic!("real_part reverse should reject mixed-dtype graphs"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "real_part_reverse"));
}
