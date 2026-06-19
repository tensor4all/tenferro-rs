#![cfg(feature = "autodiff")]

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EagerEinsumExt, EagerTensorEinsumExt};
use tenferro_einsum::{EinsumSubscripts, TensorDotAxes};
use tenferro_runtime::Tensor;

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
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = [&a, &b].einsum("ij,jk->ik").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(
        f64_data(c.materialized().unwrap().as_ref()),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_tensordot_count_contracts_last_lhs_with_first_rhs_axes() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3, 4], (1..=24).map(f64::from).collect::<Vec<_>>())
            .unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 4, 2],
            (1..=24).map(|value| f64::from(value) * 0.5).collect(),
        )
        .unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let out = lhs.tensordot(&rhs, TensorDotAxes::Count(2)).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(
        f64_data(out.materialized().unwrap().as_ref()),
        &[611.0, 650.0, 1475.0, 1586.0]
    );
}

#[test]
fn eager_tensor_tensordot_explicit_axes_accept_negative_indices() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let out = lhs
        .tensordot(
            &rhs,
            TensorDotAxes::Axes {
                lhs: &[-1],
                rhs: &[0],
            },
        )
        .unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(
        f64_data(out.materialized().unwrap().as_ref()),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_tensordot_rejects_shape_mismatch() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let err = match lhs.tensordot(&rhs, TensorDotAxes::Count(1)) {
        Ok(_) => panic!("expected tensordot shape mismatch"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("contracted dimensions differ"));
}

#[test]
fn eager_tensor_tensordot_rejects_explicit_out_of_bounds_axis() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let err = match lhs.tensordot(
        &rhs,
        TensorDotAxes::Axes {
            lhs: &[2],
            rhs: &[0],
        },
    ) {
        Ok(_) => panic!("expected explicit tensordot axis bounds error"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("lhs axis 2 out of bounds"));
}

#[test]
fn eager_tensor_einsum_integer_subscripts_match_string_path() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);

    let c = [&a, &b].einsum_subscripts(&subscripts).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(
        f64_data(c.materialized().unwrap().as_ref()),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_einsum_backward_populates_input_grads() {
    let ctx = test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = [&a, &b].einsum("ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad_a = a.grad().unwrap().unwrap();
    let grad_b = b.grad().unwrap().unwrap();

    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[3, 2]);
    assert_eq!(f64_data(grad_a.as_ref()), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    assert_eq!(f64_data(grad_b.as_ref()), &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);
}

#[test]
fn eager_tensor_einsum_repeated_backward_accumulates_across_calls() {
    let ctx = test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = [&a, &b].einsum("ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();
    assert_eq!(
        f64_data(a.grad().unwrap().unwrap().as_ref()),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(
        f64_data(b.grad().unwrap().unwrap().as_ref()),
        &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]
    );

    let c = [&a, &b].einsum("ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();
    assert_eq!(
        f64_data(a.grad().unwrap().unwrap().as_ref()),
        &[10.0, 10.0, 14.0, 14.0, 18.0, 18.0]
    );
    assert_eq!(
        f64_data(b.grad().unwrap().unwrap().as_ref()),
        &[6.0, 14.0, 22.0, 6.0, 14.0, 22.0]
    );
}

#[test]
fn eager_tensor_einsum_context_clear_grads_resets_all_live_leaves() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = [&a, &b].einsum("ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();

    ctx.clear_grads().unwrap();

    assert!(a.grad().unwrap().is_none());
    assert!(b.grad().unwrap().is_none());

    let c = [&a, &b].einsum("ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();

    assert_eq!(
        f64_data(a.grad().unwrap().unwrap().as_ref()),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(
        f64_data(b.grad().unwrap().unwrap().as_ref()),
        &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]
    );
}
