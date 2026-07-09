#![cfg(feature = "autodiff")]

use num_complex::Complex64;
use std::sync::{Arc, OnceLock};
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::{download_tensor, gpu_available, upload_tensor, CudaBackend};
use tenferro_linalg::{
    EagerTensorLinalgExt, EighGauge, EighOptions, QrGauge, QrOptions, SvdGauge, SvdOptions,
};

fn test_ctx() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::with_cpu_backend(CpuBackend::new()))
        .clone()
}

fn ad_test_ctx() -> Arc<EagerRuntime> {
    let ad = AdContext::builder()
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

fn assert_finite_f64_tensor(tensor: &Tensor) {
    let values = f64_data(tensor);
    assert!(
        values.iter().all(|value| value.is_finite()),
        "expected finite f64 tensor, got {values:?}"
    );
}

fn weighted_square_sum(input: &EagerTensor, weights: Vec<f64>) -> EagerTensor {
    let weights = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(input.shape().to_vec(), weights).unwrap(),
        input.runtime().clone(),
    )
    .unwrap();
    let squared = input.mul(input).unwrap();
    let weighted = squared.mul(&weights).unwrap();
    let axes: Vec<usize> = (0..input.shape().len()).collect();
    weighted.reduce_sum(&axes).unwrap()
}

fn matmul2(lhs: &[f64], rhs: &[f64]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for col in 0..2 {
        for row in 0..2 {
            out[row + 2 * col] = lhs[row] * rhs[2 * col] + lhs[row + 2] * rhs[1 + 2 * col];
        }
    }
    out
}

fn transpose2(matrix: &[f64]) -> [f64; 4] {
    [matrix[0], matrix[2], matrix[1], matrix[3]]
}

fn well_conditioned_4x4() -> Vec<f64> {
    let n = 4;
    let mut data: Vec<f64> = (0..n * n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(10_u64.wrapping_mul(1442695040888963407));
            ((x % 1024) as f64 - 512.0) / 512.0
        })
        .collect();
    for j in 0..n {
        for i in 0..n {
            data[i + n * j] *= 0.05;
        }
        data[j + n * j] += 1.0 + j as f64 / n as f64;
    }
    data
}

#[test]
fn svd_returns_correct_shapes() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let (u, s, vt) = a.svd().unwrap();

    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);
}

#[test]
fn eager_decomposition_options_execute_and_return_expected_shapes() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let (u, s, vt) = a
        .svd_with_options(
            SvdOptions::default()
                .gauge(SvdGauge::CanonicalPivot)
                .derivative_eps(1.0e-10),
        )
        .unwrap();
    let (eigh_values, eigh_vectors) = a
        .eigh_with_options(
            EighOptions::default()
                .gauge(EighGauge::CanonicalPivot)
                .derivative_eps(1.0e-10),
        )
        .unwrap();
    let (q, r) = a
        .qr_with_options(QrOptions::default().gauge(QrGauge::PositiveDiagonal))
        .unwrap();

    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);
    assert_eq!(eigh_values.shape(), &[2]);
    assert_eq!(eigh_vectors.shape(), &[2, 2]);
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);
}

#[test]
fn svd_singular_value_sum_backward_does_not_panic() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4, 4], well_conditioned_4x4()).unwrap(),
        ctx,
    )
    .unwrap();
    let (_, s, _) = a.svd().unwrap();
    let loss = s.reduce_sum(&[0]).unwrap();

    loss.backward().unwrap();

    assert!(a.grad().unwrap().is_some());
}

#[test]
fn svd_vector_observable_backward_grad_is_finite() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.2, -0.3, 0.7, 0.4, 1.5, -0.8]).unwrap(),
        ctx,
    )
    .unwrap();
    let (u, _s, vt) = a.svd().unwrap();
    let u_loss = weighted_square_sum(&u, vec![0.5, -0.2, 0.7, 1.1, -0.4, 0.3]);
    let vt_loss = weighted_square_sum(&vt, vec![1.3, -0.6, 0.8, 0.2]);
    let loss = u_loss.add(&vt_loss).unwrap();

    loss.backward().unwrap();

    let grad = a.grad().unwrap().unwrap();
    assert_finite_f64_tensor(grad.as_ref());
}

#[test]
fn qr_returns_correct_shapes() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let (q, r) = a.qr().unwrap();

    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);
}

#[test]
fn cholesky_of_identity() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let l = a.cholesky().unwrap();

    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(
        f64_data(l.materialized().unwrap().as_ref()),
        &[1.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn lu_returns_expected_factors_for_swap_matrix() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let (p, l, u, parity) = a.lu().unwrap();

    assert_eq!(p.shape(), &[2, 2]);
    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(parity.shape(), &[] as &[usize]);

    assert_eq!(
        f64_data(p.materialized().unwrap().as_ref()),
        &[0.0, 1.0, 1.0, 0.0]
    );
    assert_eq!(
        f64_data(l.materialized().unwrap().as_ref()),
        &[1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        f64_data(u.materialized().unwrap().as_ref()),
        &[1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(f64_data(parity.materialized().unwrap().as_ref()), &[-1.0]);
}

#[test]
fn full_piv_lu_solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let x = a.full_piv_lu_solve(&b).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    assert_eq!(f64_data(x.materialized().unwrap().as_ref()), &[4.0, -1.0]);
}

#[test]
fn full_piv_lu_reconstructs_input() {
    let data = vec![0.0_f64, 2.0, 1.0, 3.0];
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], data.clone()).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let (p, l, u, q, parity) = a.full_piv_lu().unwrap();

    assert_eq!(p.shape(), &[2, 2]);
    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(parity.shape(), &[] as &[usize]);

    let p = p.materialized().unwrap();
    let l = l.materialized().unwrap();
    let u = u.materialized().unwrap();
    let q = q.materialized().unwrap();
    let lu = matmul2(f64_data(l.as_ref()), f64_data(u.as_ref()));
    let luq = matmul2(&lu, f64_data(q.as_ref()));
    let reconstructed = matmul2(&transpose2(f64_data(p.as_ref())), &luq);
    assert_close_slice(&reconstructed, &data, 1.0e-12);
}

#[test]
fn solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let x = a.solve(&b).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    assert_eq!(f64_data(x.materialized().unwrap().as_ref()), &[2.0, 2.0]);
}

#[test]
fn batched_solve_sum_backward_wrt_matrix_uses_native_batch_layout() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(
            vec![2, 2, 2],
            vec![2.0_f64, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 5.0],
        )
        .unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1, 2], vec![4.0_f64, 8.0, 6.0, 10.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let x = a.solve(&b).unwrap();
    let loss = x.reduce_sum(&[0, 1, 2]).unwrap();
    let _ = loss.backward().unwrap();
    let grad = a.grad().unwrap().unwrap();

    assert_eq!(grad.shape(), &[2, 2, 2]);
    assert_close_slice(
        f64_data(grad.as_ref()),
        &[-1.0, -0.5, -1.0, -0.5, -2.0 / 3.0, -0.4, -2.0 / 3.0, -0.4],
        1.0e-12,
    );
}

#[test]
fn eigh_returns_expected_values_for_diagonal_matrix() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let (values, vectors) = a.eigh().unwrap();

    assert_eq!(values.shape(), &[2]);
    assert_eq!(vectors.shape(), &[2, 2]);
    assert_close_slice(
        f64_data(values.materialized().unwrap().as_ref()),
        &[1.0, 3.0],
        1.0e-12,
    );
}

#[test]
fn eigh_vector_observable_backward_grad_is_finite() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.2, 0.2, 4.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let (_values, vectors) = a.eigh().unwrap();
    let loss = weighted_square_sum(&vectors, vec![0.6, -0.7, 1.3, 0.4]);

    loss.backward().unwrap();

    let grad = a.grad().unwrap().unwrap();
    assert_finite_f64_tensor(grad.as_ref());
}

#[test]
fn eig_returns_expected_complex_values_for_diagonal_matrix() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let (values, vectors) = a.eig().unwrap();

    assert_eq!(values.shape(), &[2]);
    assert_eq!(vectors.shape(), &[2, 2]);

    let mut sorted = c64_data(values.materialized().unwrap().as_ref()).to_vec();
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

#[test]
fn triangular_solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 7.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let x = a.triangular_solve(&b, true, true, false, false).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    assert_close_slice(
        f64_data(x.materialized().unwrap().as_ref()),
        &[1.0, 2.0],
        1.0e-12,
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

    let a_host = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 1.0, 1.0, 2.0]).unwrap();
    let b_host = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 1.0]).unwrap();
    let upload_backend = CudaBackend::new(0).unwrap();
    let a_gpu = upload_tensor(upload_backend.runtime(), &a_host).unwrap();
    let b_gpu = upload_tensor(upload_backend.runtime(), &b_host).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend);
    let a = EagerTensor::from_tensor_in(a_gpu, ctx.clone()).unwrap();
    let b = EagerTensor::from_tensor_in(b_gpu, ctx).unwrap();

    let x = a.solve(&b).unwrap();

    let download_backend = CudaBackend::new(0).unwrap();
    let x_host = download_tensor(
        download_backend.runtime(),
        x.materialized().unwrap().as_ref(),
    )
    .unwrap();
    assert_eq!(x_host.shape(), &[2, 1]);
    assert_close_slice(f64_data(&x_host), &[1.8, -0.4], 1.0e-9);
}
