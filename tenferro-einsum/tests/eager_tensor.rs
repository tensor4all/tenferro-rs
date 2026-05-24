#![cfg(feature = "autodiff")]

use std::sync::Arc;

use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
use tenferro_einsum::eager_tensor::{einsum, einsum_subscripts};
use tenferro_einsum::EinsumSubscripts;

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn test_ctx() -> Arc<EagerRuntime> {
    unsafe {
        std::env::set_var("TENFERRO_PROFILE_EAGER_OP_AGG", "1");
        std::env::set_var("TENFERRO_PROFILE_EAGER_OP_PRINT_EVERY", "1");
    }
    EagerRuntime::with_cpu_backend(CpuBackend::new())
}

#[test]
fn eager_tensor_einsum_matmul_primal_matches_expected_values() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(c.data().shape(), &[2, 2]);
    assert_eq!(f64_data(c.data()), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn eager_tensor_einsum_integer_subscripts_match_string_path() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);

    let c = einsum_subscripts(&[&a, &b], &subscripts).unwrap();

    assert_eq!(c.data().shape(), &[2, 2]);
    assert_eq!(f64_data(c.data()), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn eager_tensor_einsum_backward_populates_input_grads() {
    let ctx = test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[3, 2]);
    assert_eq!(f64_data(grad_a.as_ref()), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    assert_eq!(f64_data(grad_b.as_ref()), &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);
}

#[test]
fn eager_tensor_einsum_repeated_backward_accumulates_across_calls() {
    let ctx = test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();
    assert_eq!(
        f64_data(a.grad().unwrap().as_ref()),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(
        f64_data(b.grad().unwrap().as_ref()),
        &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]
    );

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();
    assert_eq!(
        f64_data(a.grad().unwrap().as_ref()),
        &[10.0, 10.0, 14.0, 14.0, 18.0, 18.0]
    );
    assert_eq!(
        f64_data(b.grad().unwrap().as_ref()),
        &[6.0, 14.0, 22.0, 6.0, 14.0, 22.0]
    );
}

#[test]
fn eager_tensor_einsum_context_clear_grads_resets_all_live_leaves() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx.clone(),
    );

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();

    ctx.clear_grads();

    assert!(a.grad().is_none());
    assert!(b.grad().is_none());

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();

    assert_eq!(
        f64_data(a.grad().unwrap().as_ref()),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(
        f64_data(b.grad().unwrap().as_ref()),
        &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]
    );
}
