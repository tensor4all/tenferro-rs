#![cfg(feature = "autodiff")]

use num_complex::Complex64;
use std::sync::{Arc, OnceLock};
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::cubecl::{download_tensor, gpu_available, upload_tensor, CubeclBackend};

fn test_ctx() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::with_cpu_backend(CpuBackend::new()))
        .clone()
}

fn ad_test_ctx() -> Arc<EagerRuntime> {
    let ad = AdContext::builder()
        .with_core_rules()
        .with_extension_rules(tenferro_linalg::ad_rules().unwrap())
        .build()
        .unwrap();
    EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad)
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn c64_data(tensor: &Tensor) -> &[Complex64] {
    tensor.as_slice::<Complex64>().unwrap()
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "idx {idx}: expected {expected}, got {actual}, diff {diff}"
        );
    }
}

#[test]
fn svd_returns_correct_shapes() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]),
        test_ctx(),
    );
    let (u, s, vt) = tenferro_linalg::eager_tensor::svd(&a).unwrap();

    assert_eq!(u.data().shape(), &[2, 2]);
    assert_eq!(s.data().shape(), &[2]);
    assert_eq!(vt.data().shape(), &[2, 2]);
}

#[test]
fn qr_returns_correct_shapes() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]),
        test_ctx(),
    );
    let (q, r) = tenferro_linalg::eager_tensor::qr(&a).unwrap();

    assert_eq!(q.data().shape(), &[2, 2]);
    assert_eq!(r.data().shape(), &[2, 2]);
}

#[test]
fn cholesky_of_identity() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]),
        test_ctx(),
    );
    let l = tenferro_linalg::eager_tensor::cholesky(&a).unwrap();

    assert_eq!(l.data().shape(), &[2, 2]);
    assert_eq!(f64_data(l.data()), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn lu_returns_expected_factors_for_swap_matrix() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]),
        test_ctx(),
    );
    let (p, l, u, parity) = tenferro_linalg::eager_tensor::lu(&a).unwrap();

    assert_eq!(p.data().shape(), &[2, 2]);
    assert_eq!(l.data().shape(), &[2, 2]);
    assert_eq!(u.data().shape(), &[2, 2]);
    assert_eq!(parity.data().shape(), &[] as &[usize]);

    assert_eq!(f64_data(p.data()), &[0.0, 1.0, 1.0, 0.0]);
    assert_eq!(f64_data(l.data()), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(f64_data(u.data()), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(f64_data(parity.data()), &[-1.0]);
}

#[test]
fn full_piv_lu_solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]),
        test_ctx(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0]),
        test_ctx(),
    );
    let x = tenferro_linalg::eager_tensor::full_piv_lu_solve(&a, &b).unwrap();

    assert_eq!(x.data().shape(), &[2, 1]);
    assert_eq!(f64_data(x.data()), &[4.0, -1.0]);
}

#[test]
fn solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]),
        test_ctx(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0]),
        test_ctx(),
    );
    let x = tenferro_linalg::eager_tensor::solve(&a, &b).unwrap();

    assert_eq!(x.data().shape(), &[2, 1]);
    assert_eq!(f64_data(x.data()), &[2.0, 2.0]);
}

#[test]
fn batched_solve_sum_backward_wrt_matrix_uses_native_batch_layout() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(
            vec![2, 2, 2],
            vec![2.0_f64, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 5.0],
        ),
        ctx.clone(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1, 2], vec![4.0_f64, 8.0, 6.0, 10.0]),
        ctx,
    );

    let x = tenferro_linalg::eager_tensor::solve(&a, &b).unwrap();
    let loss = x.reduce_sum(&[0, 1, 2]).unwrap();
    let _ = loss.backward().unwrap();
    let grad = a.grad().unwrap();

    assert_eq!(grad.shape(), &[2, 2, 2]);
    assert_close_slice(
        f64_data(grad.as_ref()),
        &[-1.0, -0.5, -1.0, -0.5, -2.0 / 3.0, -0.4, -2.0 / 3.0, -0.4],
        1.0e-12,
    );
}

#[test]
fn eig_returns_expected_complex_values_for_diagonal_matrix() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]),
        test_ctx(),
    );
    let (values, vectors) = tenferro_linalg::eager_tensor::eig(&a).unwrap();

    assert_eq!(values.data().shape(), &[2]);
    assert_eq!(vectors.data().shape(), &[2, 2]);

    let mut sorted = c64_data(values.data()).to_vec();
    sorted.sort_by(|lhs, rhs| {
        lhs.re
            .partial_cmp(&rhs.re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assert_eq!(
        sorted,
        vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)]
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_eager_solve_uses_registered_linalg_runtime() {
    if !gpu_available() {
        eprintln!("skipping cuda_eager_solve_uses_registered_linalg_runtime: no CUDA device");
        return;
    }

    let a_host = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 1.0, 1.0, 2.0]);
    let b_host = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 1.0]);
    let upload_backend = CubeclBackend::new(0).unwrap();
    let a_gpu = upload_tensor(upload_backend.runtime(), &a_host).unwrap();
    let b_gpu = upload_tensor(upload_backend.runtime(), &b_host).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend);
    let a = EagerTensor::from_tensor_in(a_gpu, ctx.clone());
    let b = EagerTensor::from_tensor_in(b_gpu, ctx);

    let x = tenferro_linalg::eager_tensor::solve(&a, &b).unwrap();

    let download_backend = CubeclBackend::new(0).unwrap();
    let x_host = download_tensor(download_backend.runtime(), x.data()).unwrap();
    assert_eq!(x_host.shape(), &[2, 1]);
    assert_close_slice(f64_data(&x_host), &[1.8, -0.4], 1.0e-9);
}
