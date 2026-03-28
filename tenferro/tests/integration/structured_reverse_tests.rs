use super::support::{grad_wrt, reverse_rank0_f64_like, vector_c64, vector_f64};
use num_complex::Complex64;
use tenferro::{set_default_runtime, RuntimeContext};
use tenferro::{Error, Tensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn scalar(value: f64) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn diag_f64(values: &[f64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector_f64(values))).unwrap()
}

fn diag_c64(values: &[Complex64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector_c64(values))).unwrap()
}

#[test]
fn diag_scale_reverse_keeps_diag_cotangent_space() {
    let mut x = diag_f64(&[2.0, 3.0]);
    x.set_requires_grad(true).unwrap();
    let a = reverse_rank0_f64_like(2.0_f64, &x);
    let y = x.scale(&a).unwrap();
    let cotangent = diag_f64(&[1.0, 1.0]);

    let grads = grad_wrt(&y, &cotangent, &[&x]);
    let grad = grads[0].as_ref().unwrap();
    assert!(grad.is_diag());
    assert_eq!(grad.as_f64().unwrap().primal().dims(), &[2]);
}

#[test]
fn diag_axpby_reverse_keeps_diag_cotangent_space() {
    let mut x = diag_f64(&[2.0, 3.0]);
    x.set_requires_grad(true).unwrap();
    let mut y = diag_f64(&[5.0, 7.0]);
    y.set_requires_grad(true).unwrap();
    let a = reverse_rank0_f64_like(2.0_f64, &x);
    let b = reverse_rank0_f64_like(-1.0_f64, &x);
    let out = x.axpby(&a, &y, &b).unwrap();
    let cotangent = diag_f64(&[1.0, -0.5]);

    let grads = grad_wrt(&out, &cotangent, &[&x, &y]);

    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
    assert_eq!(
        grads[0].as_ref().unwrap().as_f64().unwrap().primal().dims(),
        &[2]
    );
    assert_eq!(
        grads[1].as_ref().unwrap().as_f64().unwrap().primal().dims(),
        &[2]
    );
}

#[test]
fn diag_complex_real_part_reverse_keeps_diag_cotangent_space() {
    let mut x = diag_c64(&[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]);
    x.set_requires_grad(true).unwrap();

    let err = match x.real_part() {
        Ok(_) => panic!("real_part reverse should be unsupported for homogeneous mixed-dtype tape"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "real_part_reverse"));
}

#[test]
fn diag_complex_compose_complex_reverse_splits_diag_cotangent_back_into_real_components() {
    let mut re = diag_f64(&[1.0, -3.0]);
    re.set_requires_grad(true).unwrap();
    let mut im = diag_f64(&[2.0, 4.0]);
    im.set_requires_grad(true).unwrap();

    let z = Tensor::compose_complex(re.clone(), im.clone()).unwrap();
    let cotangent = diag_c64(&[Complex64::new(0.5, -1.25), Complex64::new(1.0, 2.0)]);
    let grads = grad_wrt(&z, &cotangent, &[&re, &im]);
    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
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
        &[0.5, 1.0]
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
        &[-1.25, 2.0]
    );
}

#[test]
fn root_einsum_keeps_diag_output_in_structured_carrier() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = diag_f64(&[1.0, 2.0]);
    let b = diag_f64(&[3.0, 4.0]);

    let out = Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap();

    assert!(out.is_diag());
    assert_eq!(out.dims(), &[2, 2]);
    assert_eq!(out.as_f64().unwrap().primal().dims(), &[2]);
}

#[test]
fn root_einsum_owned_keeps_diag_output_in_structured_carrier() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let out = Tensor::einsum_owned(
        "ij,jk->ik",
        vec![diag_f64(&[1.0, 2.0]), diag_f64(&[3.0, 4.0])],
    )
    .unwrap();

    assert!(out.is_diag());
    assert_eq!(out.dims(), &[2, 2]);
    assert_eq!(out.as_f64().unwrap().primal().dims(), &[2]);
}

#[test]
fn root_einsum_reverse_keeps_diag_cotangent_space() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut a = diag_f64(&[1.0, 2.0]);
    a.set_requires_grad(true).unwrap();
    let mut b = diag_f64(&[3.0, 4.0]);
    b.set_requires_grad(true).unwrap();

    let out = Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let cotangent = diag_f64(&[0.5, -1.0]);
    let grads = grad_wrt(&out, &cotangent, &[&a, &b]);

    assert!(out.is_diag());
    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
    assert_eq!(
        grads[0].as_ref().unwrap().as_f64().unwrap().primal().dims(),
        &[2]
    );
    assert_eq!(
        grads[1].as_ref().unwrap().as_f64().unwrap().primal().dims(),
        &[2]
    );
}

#[test]
fn root_einsum_owned_reverse_keeps_diag_cotangent_space() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut a = diag_f64(&[1.0, 2.0]);
    a.set_requires_grad(true).unwrap();
    let mut b = diag_f64(&[3.0, 4.0]);
    b.set_requires_grad(true).unwrap();

    let out = Tensor::einsum_owned("ij,jk->ik", vec![a.clone(), b.clone()]).unwrap();
    let cotangent = diag_f64(&[0.5, -1.0]);
    let grads = grad_wrt(&out, &cotangent, &[&a, &b]);

    assert!(out.is_diag());
    assert!(grads[0].as_ref().unwrap().is_diag());
    assert!(grads[1].as_ref().unwrap().is_diag());
    assert_eq!(
        grads[0].as_ref().unwrap().as_f64().unwrap().primal().dims(),
        &[2]
    );
    assert_eq!(
        grads[1].as_ref().unwrap().as_f64().unwrap().primal().dims(),
        &[2]
    );
}

#[test]
fn root_sum_reverse_keeps_diag_cotangent_space() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut x = diag_f64(&[2.0, 3.0]);
    x.set_requires_grad(true).unwrap();

    let out = x.sum().unwrap();
    let cotangent = Tensor::from_tensor(scalar(1.5));
    let grads = grad_wrt(&out, &cotangent, &[&x]);

    assert!(out.dims().is_empty());
    assert!(grads[0].as_ref().unwrap().is_diag());
    assert_eq!(
        grads[0].as_ref().unwrap().as_f64().unwrap().primal().dims(),
        &[2]
    );
}
