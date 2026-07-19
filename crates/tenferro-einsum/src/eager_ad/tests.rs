use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{Tensor, TensorRead, TensorView};

use super::{backend_broadcast_multiply_untracked, einsum, einsum_whole_program_untracked};
use crate::{ContractionTree, Subscripts};

#[test]
fn binary_einsum_col_major_matmul_uses_direct_dot_general_fast_path() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
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

    let out = einsum(&[&lhs, &rhs], "ji,kj->ki").unwrap();

    assert_eq!(out.shape(), &[4, 3]);
    assert_eq!(
        out.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[2.0_f64; 12]
    );
    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 0);
}

#[test]
fn whole_program_untracked_matches_per_op_nary_result() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let a_data: Vec<f64> = (0..6).map(|i| i as f64 + 1.0).collect();
    let b_data: Vec<f64> = (0..12).map(|i| i as f64 * 0.5 - 2.0).collect();
    let c_data: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], a_data).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 4], b_data).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let c = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4, 5], c_data).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    // Reference: default per-op N-ary path.
    let reference = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();

    // Whole-program path on an explicit contraction tree (same logical result).
    let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4], &[4, 5]], &[(0, 1), (2, 3)])
        .unwrap();
    let whole = einsum_whole_program_untracked(&[&a, &b, &c], &tree).unwrap();

    assert_eq!(whole.shape(), reference.shape());
    let got_tensor = whole.materialized().unwrap();
    let want_tensor = reference.materialized().unwrap();
    let got = got_tensor.as_slice::<f64>().unwrap();
    let want = want_tensor.as_slice::<f64>().unwrap();
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(
            (g - w).abs() < 1e-10,
            "whole-program result {g} != per-op {w}"
        );
    }
}

#[test]
fn whole_program_untracked_rejects_tracked_inputs() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 2], &[2, 2]], &[(0, 1)]).unwrap();
    assert!(einsum_whole_program_untracked(&[&a, &b], &tree).is_err());
}

#[test]
fn nary_eager_einsum_expands_to_standard_ops_with_runtime_cache() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let c = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let out = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(out.shape(), &[2, 5]);
    assert_eq!(
        out.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[12.0_f64; 10]
    );
    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 1);
}

#[test]
fn nary_eager_einsum_expanded_standard_ops_reuse_runtime_cache() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let c = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let first = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(first.shape(), &[2, 5]);
    let after_first = ctx.cache_stats().unwrap().extensions;
    assert_eq!(after_first.entries, 1);
    assert!(after_first.retained_bytes > 0);

    let second = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert_eq!(second.shape(), &[2, 5]);
    let after_second = ctx.cache_stats().unwrap().extensions;
    assert_eq!(after_second.entries, 1);
    assert_eq!(after_second.retained_bytes, after_first.retained_bytes);

    ctx.clear_extension_caches().unwrap();
    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 0);
}

#[test]
fn expanded_eager_einsum_cache_hit_preserves_lazy_view_output() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let k = 2;
    let j = 3;
    let o = 4;
    let t = 5;
    let lhs_data: Vec<f64> = (0..k * j * t).map(|idx| idx as f64 + 1.0).collect();
    let rhs_data: Vec<f64> = (0..o * t).map(|idx| idx as f64 + 101.0).collect();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![k, j, t], lhs_data).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![o, t], rhs_data).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let first = einsum(&[&lhs, &rhs], "kjt,ot->jkot").unwrap();
    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 1);
    assert!(matches!(
        first.tensor_read(),
        TensorRead::View(TensorView::F64(_))
    ));

    let second = einsum(&[&lhs, &rhs], "kjt,ot->jkot").unwrap();

    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 1);
    match second.tensor_read() {
        TensorRead::View(TensorView::F64(view)) => {
            assert_eq!(view.shape(), &[j, k, o, t]);
            assert_eq!(
                view.strides(),
                &[k as isize, 1, (k * j) as isize, (k * j * o) as isize]
            );
        }
        other => panic!("expected lazy f64 view from cached expansion, got {other:?}"),
    }
}

#[test]
fn nary_eager_einsum_expanded_standard_ops_preserve_backward() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let c = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]).unwrap(),
        ctx,
    )
    .unwrap();

    let loss = einsum(&[&a, &b, &c], "ij,jk,kl->il")
        .unwrap()
        .reduce_sum(Some(&[0, 1]))
        .unwrap();
    let _ = loss.backward().unwrap();

    assert_eq!(
        a.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[20.0; 6]
    );
}

#[test]
fn tracked_nary_einsum_gradients_match_expected_values() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let c = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]).unwrap(),
        ctx,
    )
    .unwrap();

    let out = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();
    assert_eq!(out.shape(), &[2, 5]);

    let loss = out.reduce_sum(Some(&[0, 1])).unwrap();
    let _ = loss.backward().unwrap();

    assert_eq!(
        a.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[20.0; 6]
    );
    assert_eq!(
        b.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[10.0; 12]
    );
    assert_eq!(
        c.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[6.0; 20]
    );
}

#[test]
fn eager_outer_product_can_use_untracked_backend_broadcast_multiply() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![5.0_f64, 7.0, 11.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let out = backend_broadcast_multiply_untracked(&lhs, &[2, 3], &[0], &rhs, &[2, 3], &[1])
        .unwrap()
        .expect("untracked CPU eager tensors should use backend broadcast multiply");

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(
        out.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[10.0, 15.0, 14.0, 21.0, 22.0, 33.0]
    );
    assert!(!out.tracks_grad());
}

#[test]
fn eager_outer_product_can_return_lazy_noncompact_output() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let k = 2;
    let j = 3;
    let o = 4;
    let t = 5;
    let lhs_data: Vec<f64> = (0..k * j * t).map(|idx| idx as f64 + 1.0).collect();
    let rhs_data: Vec<f64> = (0..o * t).map(|idx| idx as f64 + 101.0).collect();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![k, j, t], lhs_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![o, t], rhs_data.clone()).unwrap(),
        ctx,
    )
    .unwrap();

    let out = einsum(&[&lhs, &rhs], "kjt,ot->jkot").unwrap();

    assert_eq!(out.shape(), &[j, k, o, t]);
    match out.tensor_read() {
        TensorRead::View(TensorView::F64(view)) => {
            assert_eq!(view.shape(), &[j, k, o, t]);
            assert_eq!(
                view.strides(),
                &[k as isize, 1, (k * j) as isize, (k * j * o) as isize]
            );
        }
        other => panic!("expected lazy f64 view, got {other:?}"),
    }

    let expected: Vec<f64> = (0..t)
        .flat_map(|tt| {
            let lhs_data = &lhs_data;
            let rhs_data = &rhs_data;
            (0..o).flat_map(move |oo| {
                (0..k).flat_map(move |kk| {
                    (0..j).map(move |jj| lhs_data[kk + k * jj + k * j * tt] * rhs_data[oo + o * tt])
                })
            })
        })
        .collect();
    assert_eq!(
        out.materialized().unwrap().as_slice::<f64>().unwrap(),
        expected.as_slice()
    );
}
