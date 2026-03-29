use num_complex::Complex64;
use tenferro::{set_default_runtime, Error, RuntimeContext, Tensor};
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

fn diag(values: &[f64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector(values))).unwrap()
}

fn dense(values: &[f64], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap(),
    )
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
    let grads = tenferro::grad(
        &[&out],
        &[&x, &alpha],
        Some(&grad_outputs),
        Default::default(),
    )
    .unwrap();

    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].as_ref().unwrap().dims(), &[2]);
    assert!(grads[1].as_ref().unwrap().dims().is_empty());
}

#[test]
fn structured_qr_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut x = diag(&[1.0, 2.0]);
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

#[test]
fn ensure_common_reverse_tape_allows_sequential_qr_branch_absorption() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let mut left = dense(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mut center = dense(&[1.0, 0.0, 0.5, 2.0, 3.0, 1.0, 0.0, 4.0], &[2, 2, 2]);
    let mut right = dense(&[2.0, 0.5, 1.0, 3.0], &[2, 2]);
    left.set_requires_grad(true).unwrap();
    center.set_requires_grad(true).unwrap();
    right.set_requires_grad(true).unwrap();

    Tensor::ensure_common_reverse_tape(&[&left, &center, &right]).unwrap();

    let left_qr = left.qr().unwrap();
    let center_after_left = Tensor::einsum("abc,da->dbc", &[&center, &left_qr.r]).unwrap();

    let right_qr = right.qr().unwrap();
    let dense = Tensor::einsum("abc,dc->abd", &[&center_after_left, &right_qr.r]).unwrap();
    let loss = dense.sum().unwrap();
    loss.backward(None, &[&left, &center, &right], Default::default())
        .unwrap();

    assert!(left.grad().is_some());
    assert!(center.grad().is_some());
    assert!(right.grad().is_some());
}

#[test]
fn ensure_common_reverse_tape_allows_sequential_svd_branch_absorption() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let mut left = dense(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mut center = dense(&[1.0, 0.0, 0.5, 2.0, 3.0, 1.0, 0.0, 4.0], &[2, 2, 2]);
    let mut right = dense(&[2.0, 0.5, 1.0, 3.0], &[2, 2]);
    left.set_requires_grad(true).unwrap();
    center.set_requires_grad(true).unwrap();
    right.set_requires_grad(true).unwrap();

    Tensor::ensure_common_reverse_tape(&[&left, &center, &right]).unwrap();

    let left_svd = left.svd().unwrap();
    let left_sigma = Tensor::diag(&left_svd.s).unwrap();
    let left_rhs = Tensor::einsum("ab,bc->ac", &[&left_sigma, &left_svd.vt]).unwrap();
    let center_after_left = Tensor::einsum("abc,da->dbc", &[&center, &left_rhs]).unwrap();

    let right_svd = right.svd().unwrap();
    let right_sigma = Tensor::diag(&right_svd.s).unwrap();
    let right_rhs = Tensor::einsum("ab,bc->ac", &[&right_sigma, &right_svd.vt]).unwrap();
    let dense = Tensor::einsum("abc,dc->abd", &[&center_after_left, &right_rhs]).unwrap();
    let loss = dense.sum().unwrap();
    loss.backward(None, &[&left, &center, &right], Default::default())
        .unwrap();

    assert!(left.grad().is_some());
    assert!(center.grad().is_some());
    assert!(right.grad().is_some());
}
