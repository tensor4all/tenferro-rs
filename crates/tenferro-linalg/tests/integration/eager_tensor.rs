#![cfg(feature = "autodiff")]

use num_complex::Complex64;
use std::sync::{Arc, OnceLock};
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::{
    cuda::download_tensor, cuda::gpu_available, cuda::upload_tensor, cuda::CudaBackend,
};
use tenferro_linalg::{
    EagerTensorLinalgExt, EighGauge, EighOptions, LinalgBackend, QrGauge, QrOptions,
    RankRevealingQrOptions, SvdGauge, SvdOptions,
};

fn test_ctx() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap())
        .clone()
}

fn ad_test_ctx() -> Arc<EagerRuntime> {
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad).unwrap()
}

#[test]
fn eager_rank_revealing_qr_keeps_metadata_in_runtime() {
    let runtime = test_ctx();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap(),
        Arc::clone(&runtime),
    )
    .unwrap();
    let result = input
        .rank_revealing_qr(RankRevealingQrOptions::default().rtol(1.0e-12))
        .unwrap();
    assert!(Arc::ptr_eq(result.q.runtime(), &runtime));
    assert!(Arc::ptr_eq(result.r.runtime(), &runtime));
    assert!(Arc::ptr_eq(result.column_permutation.runtime(), &runtime));
    assert!(Arc::ptr_eq(result.rank.runtime(), &runtime));
    assert_eq!(
        result.rank.to_tensor().unwrap().as_slice::<i64>().unwrap(),
        &[1]
    );
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn f32_data(tensor: &Tensor) -> &[f32] {
    tensor.as_slice::<f32>().unwrap()
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

fn assert_close_c64_slice(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).norm();
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
    weighted.reduce_sum(Some(&axes)).unwrap()
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

// ---------------------------------------------------------------------------
// solve AD verification helpers: analytic 2x2 solves and finite differences
// ---------------------------------------------------------------------------

// Nonsymmetric, well-conditioned A = [[2, 1], [3, 4]] in column-major order
// (cond ~ 5.8). A diagonal matrix cannot detect an omitted transpose in the
// A^-H adjoint, so the VJP tests use this matrix instead.
const SOLVE_A_CM: [f64; 4] = [2.0, 3.0, 1.0, 4.0];
const SOLVE_B: [f64; 2] = [1.0, 2.0];

// Nonsymmetric complex A with distinct real and imaginary parts (so
// A^T != A^H): A = [[1+2i, 0.5-1i], [3, 4+2i]] in column-major order.
const SOLVE_C64_A: [Complex64; 4] = [
    Complex64::new(1.0, 2.0),
    Complex64::new(3.0, 0.0),
    Complex64::new(0.5, -1.0),
    Complex64::new(4.0, 2.0),
];
const SOLVE_C64_B: [Complex64; 2] = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];

/// Solve a column-major 2x2 real system via the explicit inverse (analytic
/// helper; finite-difference checks below evaluate the real eager `solve`).
fn solve2_f64(a: [f64; 4], b: [f64; 2]) -> [f64; 2] {
    let det = a[0] * a[3] - a[1] * a[2];
    let inv = [a[3] / det, -a[1] / det, -a[2] / det, a[0] / det];
    [inv[0] * b[0] + inv[2] * b[1], inv[1] * b[0] + inv[3] * b[1]]
}

/// Column-major 2x2 real matrix-vector product.
fn matvec2_f64(a: [f64; 4], x: [f64; 2]) -> [f64; 2] {
    [a[0] * x[0] + a[2] * x[1], a[1] * x[0] + a[3] * x[1]]
}

/// Solve a column-major 2x2 complex system via the explicit inverse.
fn solve2_c64(a: [Complex64; 4], b: [Complex64; 2]) -> [Complex64; 2] {
    let det = a[0] * a[3] - a[1] * a[2];
    let inv = [a[3] / det, -a[1] / det, -a[2] / det, a[0] / det];
    [inv[0] * b[0] + inv[2] * b[1], inv[1] * b[0] + inv[3] * b[1]]
}

/// Column-major 2x2 complex matrix-vector product.
fn matvec2_c64(a: [Complex64; 4], x: [Complex64; 2]) -> [Complex64; 2] {
    [a[0] * x[0] + a[2] * x[1], a[1] * x[0] + a[3] * x[1]]
}

fn transpose2_c64(a: [Complex64; 4]) -> [Complex64; 4] {
    [a[0], a[2], a[1], a[3]]
}

fn conj2_c64(a: [Complex64; 4]) -> [Complex64; 4] {
    [a[0].conj(), a[1].conj(), a[2].conj(), a[3].conj()]
}

/// Central-difference directional derivative of a 2-vector solve function at
/// `(a, b)` along the direction `(da, db)`.
fn fd_directional_vec2(
    f: impl Fn(&[f64], &[f64]) -> [f64; 2],
    a: &[f64],
    b: &[f64],
    da: &[f64],
    db: &[f64],
    step: f64,
) -> [f64; 2] {
    let mut a_plus = a.to_vec();
    let mut a_minus = a.to_vec();
    let mut b_plus = b.to_vec();
    let mut b_minus = b.to_vec();
    for (idx, &d) in da.iter().enumerate() {
        a_plus[idx] += step * d;
        a_minus[idx] -= step * d;
    }
    for (idx, &d) in db.iter().enumerate() {
        b_plus[idx] += step * d;
        b_minus[idx] -= step * d;
    }
    let plus = f(&a_plus, &b_plus);
    let minus = f(&a_minus, &b_minus);
    [
        (plus[0] - minus[0]) / (2.0 * step),
        (plus[1] - minus[1]) / (2.0 * step),
    ]
}

/// Central-difference gradient of a scalar loss over the entries of `a` then
/// `b`.
fn fd_loss_grad_f64(
    loss: impl Fn(&[f64], &[f64]) -> f64,
    a: &[f64],
    b: &[f64],
    step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut g_a = vec![0.0; a.len()];
    let mut g_b = vec![0.0; b.len()];
    for idx in 0..a.len() {
        let mut plus = a.to_vec();
        let mut minus = a.to_vec();
        plus[idx] += step;
        minus[idx] -= step;
        g_a[idx] = (loss(&plus, b) - loss(&minus, b)) / (2.0 * step);
    }
    for idx in 0..b.len() {
        let mut plus = b.to_vec();
        let mut minus = b.to_vec();
        plus[idx] += step;
        minus[idx] -= step;
        g_b[idx] = (loss(a, &plus) - loss(a, &minus)) / (2.0 * step);
    }
    (g_a, g_b)
}

/// Central-difference directional derivative of a complex 2-vector solve
/// function. The direction is complex, so `solve`'s complex-linearity is
/// exercised with a single complex perturbation.
fn fd_directional_c64(
    f: impl Fn(&[Complex64], &[Complex64]) -> [Complex64; 2],
    a: &[Complex64],
    b: &[Complex64],
    da: &[Complex64],
    db: &[Complex64],
    step: f64,
) -> [Complex64; 2] {
    let mut a_plus = a.to_vec();
    let mut a_minus = a.to_vec();
    let mut b_plus = b.to_vec();
    let mut b_minus = b.to_vec();
    for (idx, &d) in da.iter().enumerate() {
        a_plus[idx] += Complex64::new(step * d.re, step * d.im);
        a_minus[idx] -= Complex64::new(step * d.re, step * d.im);
    }
    for (idx, &d) in db.iter().enumerate() {
        b_plus[idx] += Complex64::new(step * d.re, step * d.im);
        b_minus[idx] -= Complex64::new(step * d.re, step * d.im);
    }
    let plus = f(&a_plus, &b_plus);
    let minus = f(&a_minus, &b_minus);
    [
        (plus[0] - minus[0]) / Complex64::new(2.0 * step, 0.0),
        (plus[1] - minus[1]) / Complex64::new(2.0 * step, 0.0),
    ]
}

/// Central-difference gradient of a real loss over complex entries: entry `i`
/// is `dL/dRe(x_i) + i * dL/dIm(x_i)`.
fn fd_loss_grad_c64(
    loss: impl Fn(&[Complex64], &[Complex64]) -> f64,
    a: &[Complex64],
    b: &[Complex64],
    step: f64,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut g_a = vec![Complex64::new(0.0, 0.0); a.len()];
    let mut g_b = vec![Complex64::new(0.0, 0.0); b.len()];
    for idx in 0..a.len() {
        let mut plus = a.to_vec();
        let mut minus = a.to_vec();
        plus[idx] += Complex64::new(step, 0.0);
        minus[idx] -= Complex64::new(step, 0.0);
        g_a[idx].re = (loss(&plus, b) - loss(&minus, b)) / (2.0 * step);
        plus[idx] = a[idx] + Complex64::new(0.0, step);
        minus[idx] = a[idx] - Complex64::new(0.0, step);
        g_a[idx].im = (loss(&plus, b) - loss(&minus, b)) / (2.0 * step);
    }
    for idx in 0..b.len() {
        let mut plus = b.to_vec();
        let mut minus = b.to_vec();
        plus[idx] += Complex64::new(step, 0.0);
        minus[idx] -= Complex64::new(step, 0.0);
        g_b[idx].re = (loss(a, &plus) - loss(a, &minus)) / (2.0 * step);
        plus[idx] = b[idx] + Complex64::new(0.0, step);
        minus[idx] = b[idx] - Complex64::new(0.0, step);
        g_b[idx].im = (loss(a, &plus) - loss(a, &minus)) / (2.0 * step);
    }
    (g_a, g_b)
}

/// Eager `solve` on an f64 2x2 system.
fn eager_solve_f64(ctx: &Arc<EagerRuntime>, a: &[f64], b: &[f64]) -> [f64; 2] {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], a.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], b.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let solved = a.solve(&b).unwrap().to_tensor().unwrap();
    let values = f64_data(&solved);
    [values[0], values[1]]
}

/// Eager `solve` on an f32 2x2 system, returned as f64 for differencing.
fn eager_solve_f32(ctx: &Arc<EagerRuntime>, a: &[f64], b: &[f64]) -> [f64; 2] {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], a.iter().map(|v| *v as f32).collect()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], b.iter().map(|v| *v as f32).collect()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let solved = a.solve(&b).unwrap().to_tensor().unwrap();
    let values = f32_data(&solved);
    [values[0] as f64, values[1] as f64]
}

/// Eager `solve` on a Complex64 2x2 system.
fn eager_solve_c64(ctx: &Arc<EagerRuntime>, a: &[Complex64], b: &[Complex64]) -> [Complex64; 2] {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], a.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], b.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let solved = a.solve(&b).unwrap().to_tensor().unwrap();
    let values = c64_data(&solved);
    [values[0], values[1]]
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
    let loss = s.reduce_sum(Some(&[0])).unwrap();

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
    assert_finite_f64_tensor(&grad.to_tensor().unwrap());
}

#[test]
fn incremental_householder_qr_backward_runs_eagerly() {
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0]).unwrap(),
        ad_test_ctx(),
    )
    .unwrap();
    let state = a.householder_qr().unwrap();
    let q = state
        .q_columns(0..2, QrOptions::default().gauge(QrGauge::PositiveDiagonal))
        .unwrap();
    let r = state
        .r(QrOptions::default().gauge(QrGauge::PositiveDiagonal))
        .unwrap();
    let loss = q
        .reduce_sum(Some(&[0, 1]))
        .unwrap()
        .add(&r.reduce_sum(Some(&[0, 1])).unwrap())
        .unwrap();

    loss.backward().unwrap();

    let grad = a.grad().unwrap().unwrap().to_tensor().unwrap();
    assert_finite_f64_tensor(&grad);
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
    assert_eq!(f64_data(&l.to_tensor().unwrap()), &[1.0, 0.0, 0.0, 1.0]);
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

    assert_eq!(f64_data(&p.to_tensor().unwrap()), &[0.0, 1.0, 1.0, 0.0]);
    assert_eq!(f64_data(&l.to_tensor().unwrap()), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(f64_data(&u.to_tensor().unwrap()), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(f64_data(&parity.to_tensor().unwrap()), &[-1.0]);
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
    assert_eq!(f64_data(&x.to_tensor().unwrap()), &[4.0, -1.0]);
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

    let p = p.to_tensor().unwrap();
    let l = l.to_tensor().unwrap();
    let u = u.to_tensor().unwrap();
    let q = q.to_tensor().unwrap();
    let lu = matmul2(f64_data(&l), f64_data(&u));
    let luq = matmul2(&lu, f64_data(&q));
    let reconstructed = matmul2(&transpose2(f64_data(&p)), &luq);
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
    assert_eq!(f64_data(&x.to_tensor().unwrap()), &[2.0, 2.0]);
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
    let loss = x.reduce_sum(Some(&[0, 1, 2])).unwrap();
    let _ = loss.backward().unwrap();
    let grad = a.grad().unwrap().unwrap();

    assert_eq!(grad.shape(), &[2, 2, 2]);
    assert_close_slice(
        f64_data(&grad.to_tensor().unwrap()),
        &[-1.0, -0.5, -1.0, -0.5, -2.0 / 3.0, -0.4, -2.0 / 3.0, -0.4],
        1.0e-12,
    );
}

#[test]
fn solve_matches_concrete_partial_pivot_backend() {
    // The eager single-op solve must run the same partial-pivot kernel as the
    // concrete `LinalgBackend::solve` on well- and ill-conditioned square
    // systems, for f64 and f32.
    let mut backend = CpuBackend::new();
    for (case, matrix, rhs) in [
        ("f64 well-conditioned", [2.0, 0.0, 0.0, 4.0], [4.0, 8.0]),
        (
            "f64 ill-conditioned",
            [1.0, 1.0, 1.0, 1.0 + 1.0e-8],
            [1.0, 2.0],
        ),
    ] {
        let a = Tensor::from_vec_col_major(vec![2, 2], matrix.to_vec()).unwrap();
        let b = Tensor::from_vec_col_major(vec![2, 1], rhs.to_vec()).unwrap();
        let concrete =
            crate::support::with_cpu_linalg(&mut backend, |session| session.solve(&a, &b)).unwrap();
        let eager = EagerTensor::from_tensor_in(a, test_ctx())
            .unwrap()
            .solve(&EagerTensor::from_tensor_in(b, test_ctx()).unwrap())
            .unwrap()
            .to_tensor()
            .unwrap();
        let actual = f64_data(&eager);
        let expected = f64_data(&concrete);
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1.0e-13,
                "{case} idx {idx}: expected {expected}, got {actual}, diff {diff}"
            );
        }
        let x = expected;
        let residual = [
            matrix[0] * x[0] + matrix[2] * x[1] - rhs[0],
            matrix[1] * x[0] + matrix[3] * x[1] - rhs[1],
        ];
        assert!(
            residual.iter().all(|value| value.abs() <= 1.0e-10),
            "{case}: partial-pivot solve left residual {residual:?}"
        );
    }
    for (case, matrix, rhs) in [
        (
            "f32 well-conditioned",
            [2.0_f32, 0.0, 0.0, 4.0],
            [4.0_f32, 8.0],
        ),
        (
            "f32 ill-conditioned",
            [1.0_f32, 1.0, 1.0, 1.001],
            [1.0_f32, 2.0],
        ),
    ] {
        let a = Tensor::from_vec_col_major(vec![2, 2], matrix.to_vec()).unwrap();
        let b = Tensor::from_vec_col_major(vec![2, 1], rhs.to_vec()).unwrap();
        let concrete =
            crate::support::with_cpu_linalg(&mut backend, |session| session.solve(&a, &b)).unwrap();
        let eager = EagerTensor::from_tensor_in(a, test_ctx())
            .unwrap()
            .solve(&EagerTensor::from_tensor_in(b, test_ctx()).unwrap())
            .unwrap()
            .to_tensor()
            .unwrap();
        let actual = eager.as_slice::<f32>().unwrap();
        let expected = concrete.as_slice::<f32>().unwrap();
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1.0e-6,
                "{case} idx {idx}: expected {expected}, got {actual}, diff {diff}"
            );
        }
        let x = expected;
        let residual = [
            matrix[0] * x[0] + matrix[2] * x[1] - rhs[0],
            matrix[1] * x[0] + matrix[3] * x[1] - rhs[1],
        ];
        assert!(
            residual.iter().all(|value| value.abs() <= 1.0e-4),
            "{case}: partial-pivot solve left residual {residual:?}"
        );
    }
}

#[test]
fn solve_forward_jvp_wrt_matrix_and_rhs_matches_analytic_and_finite_difference() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], SOLVE_A_CM.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 1], SOLVE_B.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let x = a.solve(&b).unwrap();
    let da = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let db = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 1.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    // JVP wrt A: dx = solve(A, -dA x). x = [0.4, 0.2], dA = I ->
    // dx = -A^-1 x = [-0.28, 0.16].
    let dx_a = ctx.jvp(&x, &a, &da).unwrap();
    let x_expected = solve2_f64(SOLVE_A_CM, SOLVE_B);
    let dax = matvec2_f64([1.0, 0.0, 0.0, 1.0], x_expected);
    let dx_a_expected = solve2_f64(SOLVE_A_CM, [-dax[0], -dax[1]]);
    assert_close_slice(
        f64_data(&dx_a.to_tensor().unwrap()),
        &dx_a_expected,
        1.0e-12,
    );
    let dx_a_fd = fd_directional_vec2(
        |a, b| eager_solve_f64(&ctx, a, b),
        &SOLVE_A_CM,
        &SOLVE_B,
        &[1.0, 0.0, 0.0, 1.0],
        &[0.0, 0.0],
        1.0e-5,
    );
    assert_close_slice(f64_data(&dx_a.to_tensor().unwrap()), &dx_a_fd, 1.0e-6);

    // JVP wrt b: dx = solve(A, db). db = [1, 1] -> dx = A^-1 [1, 1] =
    // [0.6, -0.2].
    let dx_b = ctx.jvp(&x, &b, &db).unwrap();
    let dx_b_expected = solve2_f64(SOLVE_A_CM, [1.0, 1.0]);
    assert_close_slice(
        f64_data(&dx_b.to_tensor().unwrap()),
        &dx_b_expected,
        1.0e-12,
    );
    let dx_b_fd = fd_directional_vec2(
        |a, b| eager_solve_f64(&ctx, a, b),
        &SOLVE_A_CM,
        &SOLVE_B,
        &[0.0, 0.0, 0.0, 0.0],
        &[1.0, 1.0],
        1.0e-5,
    );
    assert_close_slice(f64_data(&dx_b.to_tensor().unwrap()), &dx_b_fd, 1.0e-6);
}

#[test]
fn solve_backward_wrt_matrix_and_rhs_matches_analytic_and_finite_difference() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], SOLVE_A_CM.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 1], SOLVE_B.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let x = a.solve(&b).unwrap();
    let loss = x.reduce_sum(Some(&[0, 1])).unwrap();
    let _ = loss.backward().unwrap();

    // For L = sum(x), x = A^-1 b: g_b = A^-T 1, g_A = -g_b x^T. With the
    // nonsymmetric A above, g_b = A^-T [1, 1] = [0.2, 0.2] and
    // g_A = -[[0.08, 0.04], [0.08, 0.04]] in column-major order. An omitted
    // transpose in the adjoint solve would give g_b = A^-1 [1, 1] =
    // [0.6, -0.2] instead and fail this check.
    let x_expected = solve2_f64(SOLVE_A_CM, SOLVE_B);
    let g_b_expected = solve2_f64(transpose2(&SOLVE_A_CM), [1.0, 1.0]);
    let mut g_a_expected = [0.0; 4];
    for i in 0..2 {
        for j in 0..2 {
            g_a_expected[i + 2 * j] = -g_b_expected[i] * x_expected[j];
        }
    }
    assert_close_slice(
        f64_data(&b.grad().unwrap().unwrap().to_tensor().unwrap()),
        &g_b_expected,
        1.0e-12,
    );
    assert_close_slice(
        f64_data(&a.grad().unwrap().unwrap().to_tensor().unwrap()),
        &g_a_expected,
        1.0e-12,
    );

    // Independent finite-difference gradient over both inputs.
    let (fd_g_a, fd_g_b) = fd_loss_grad_f64(
        |a, b| {
            let x = eager_solve_f64(&ctx, a, b);
            x[0] + x[1]
        },
        &SOLVE_A_CM,
        &SOLVE_B,
        1.0e-5,
    );
    assert_close_slice(
        f64_data(&a.grad().unwrap().unwrap().to_tensor().unwrap()),
        &fd_g_a,
        1.0e-6,
    );
    assert_close_slice(
        f64_data(&b.grad().unwrap().unwrap().to_tensor().unwrap()),
        &fd_g_b,
        1.0e-6,
    );
}

#[test]
fn solve_forward_and_backward_match_finite_difference_in_f32() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], SOLVE_A_CM.iter().map(|v| *v as f32).collect())
            .unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 1], SOLVE_B.iter().map(|v| *v as f32).collect())
            .unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let x = a.solve(&b).unwrap();
    let da = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let db = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f32, 1.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    // f32 JVP wrt A and b against finite differences (step 1e-2, the f32
    // analogue of the 1e-5 f64 step scaled by sqrt(f32 epsilon)).
    let dx_a = ctx.jvp(&x, &a, &da).unwrap();
    let dx_a_values: Vec<f64> = f32_data(&dx_a.to_tensor().unwrap())
        .iter()
        .map(|v| *v as f64)
        .collect();
    let dx_a_fd = fd_directional_vec2(
        |a, b| eager_solve_f32(&ctx, a, b),
        &SOLVE_A_CM,
        &SOLVE_B,
        &[1.0, 0.0, 0.0, 1.0],
        &[0.0, 0.0],
        1.0e-2,
    );
    assert_close_slice(&dx_a_values, &dx_a_fd, 1.0e-3);
    let dx_b = ctx.jvp(&x, &b, &db).unwrap();
    let dx_b_values: Vec<f64> = f32_data(&dx_b.to_tensor().unwrap())
        .iter()
        .map(|v| *v as f64)
        .collect();
    let dx_b_fd = fd_directional_vec2(
        |a, b| eager_solve_f32(&ctx, a, b),
        &SOLVE_A_CM,
        &SOLVE_B,
        &[0.0, 0.0, 0.0, 0.0],
        &[1.0, 1.0],
        1.0e-2,
    );
    assert_close_slice(&dx_b_values, &dx_b_fd, 1.0e-3);

    // f32 VJP wrt A and b against finite differences.
    let loss = x.reduce_sum(Some(&[0, 1])).unwrap();
    let _ = loss.backward().unwrap();
    let (fd_g_a, fd_g_b) = fd_loss_grad_f64(
        |a, b| {
            let x = eager_solve_f32(&ctx, a, b);
            x[0] + x[1]
        },
        &SOLVE_A_CM,
        &SOLVE_B,
        1.0e-2,
    );
    let g_a_values: Vec<f64> = f32_data(&a.grad().unwrap().unwrap().to_tensor().unwrap())
        .iter()
        .map(|v| *v as f64)
        .collect();
    let g_b_values: Vec<f64> = f32_data(&b.grad().unwrap().unwrap().to_tensor().unwrap())
        .iter()
        .map(|v| *v as f64)
        .collect();
    assert_close_slice(&g_a_values, &fd_g_a, 1.0e-3);
    assert_close_slice(&g_b_values, &fd_g_b, 1.0e-3);
}

#[test]
fn solve_forward_jvp_complex_matches_analytic_and_finite_difference() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], SOLVE_C64_A.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 1], SOLVE_C64_B.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let x = a.solve(&b).unwrap();
    let da_data = [
        Complex64::new(1.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, -1.0),
    ];
    let db_data = [Complex64::new(1.0, 0.5), Complex64::new(-0.5, 1.0)];
    let da = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], da_data.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let db = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], db_data.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    // JVP wrt A: dx = solve(A, -dA x); JVP wrt b: dx = solve(A, db). The JVP
    // is transpose-free; the VJP test below pins down the A^H adjoint.
    let x_expected = solve2_c64(SOLVE_C64_A, SOLVE_C64_B);
    let dx_a_expected = {
        let dax = matvec2_c64(da_data, x_expected);
        solve2_c64(SOLVE_C64_A, [-dax[0], -dax[1]])
    };
    let dx_b_expected = solve2_c64(SOLVE_C64_A, db_data);
    let dx_a = ctx.jvp(&x, &a, &da).unwrap();
    let dx_b = ctx.jvp(&x, &b, &db).unwrap();
    assert_close_c64_slice(
        c64_data(&dx_a.to_tensor().unwrap()),
        &dx_a_expected,
        1.0e-12,
    );
    assert_close_c64_slice(
        c64_data(&dx_b.to_tensor().unwrap()),
        &dx_b_expected,
        1.0e-12,
    );

    // Finite differences along the complex directions.
    let dx_a_fd = fd_directional_c64(
        |a, b| eager_solve_c64(&ctx, a, b),
        &SOLVE_C64_A,
        &SOLVE_C64_B,
        &da_data,
        &[Complex64::new(0.0, 0.0); 2],
        1.0e-5,
    );
    assert_close_c64_slice(c64_data(&dx_a.to_tensor().unwrap()), &dx_a_fd, 1.0e-6);
    let dx_b_fd = fd_directional_c64(
        |a, b| eager_solve_c64(&ctx, a, b),
        &SOLVE_C64_A,
        &SOLVE_C64_B,
        &[Complex64::new(0.0, 0.0); 4],
        &db_data,
        1.0e-5,
    );
    assert_close_c64_slice(c64_data(&dx_b.to_tensor().unwrap()), &dx_b_fd, 1.0e-6);
}

#[test]
fn solve_backward_complex_uses_adjoint_and_matches_finite_difference() {
    let ctx = ad_test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], SOLVE_C64_A.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 1], SOLVE_C64_B.to_vec()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let x = a.solve(&b).unwrap();
    // Real loss L = sum(|x|^2); tenferro's Hermitian convention gives the
    // cotangent into x as 2x (grad(sum(|z|^2)) = 2z).
    let x_sq = x.mul(&x.conj().unwrap()).unwrap();
    let loss = x_sq.reduce_sum(Some(&[0, 1])).unwrap();
    let _ = loss.backward().unwrap();

    // Analytic: g_b = A^-H ct, g_A = -g_b X^H with ct = 2x. A^T != A^H for
    // this matrix, so using the plain transpose would give a different g_b.
    let x_expected = solve2_c64(SOLVE_C64_A, SOLVE_C64_B);
    let ct = [2.0 * x_expected[0], 2.0 * x_expected[1]];
    let a_h = conj2_c64(transpose2_c64(SOLVE_C64_A));
    let g_b_expected = solve2_c64(a_h, ct);
    let mut g_a_expected = [Complex64::new(0.0, 0.0); 4];
    for i in 0..2 {
        for j in 0..2 {
            g_a_expected[i + 2 * j] = -g_b_expected[i] * x_expected[j].conj();
        }
    }
    assert_close_c64_slice(
        c64_data(&b.grad().unwrap().unwrap().to_tensor().unwrap()),
        &g_b_expected,
        1.0e-12,
    );
    assert_close_c64_slice(
        c64_data(&a.grad().unwrap().unwrap().to_tensor().unwrap()),
        &g_a_expected,
        1.0e-12,
    );

    // The chosen matrix genuinely distinguishes A^H from A^T: an adjoint
    // built with the plain transpose would give a measurably different g_b.
    let g_b_if_transpose = solve2_c64(transpose2_c64(SOLVE_C64_A), ct);
    assert!(
        (g_b_expected[0] - g_b_if_transpose[0]).norm() > 1.0e-6
            || (g_b_expected[1] - g_b_if_transpose[1]).norm() > 1.0e-6,
        "test matrix must distinguish A^H from A^T"
    );

    // Independent finite differences of the real loss over the Re and Im
    // parts of both inputs.
    let (fd_g_a, fd_g_b) = fd_loss_grad_c64(
        |a, b| {
            let x = eager_solve_c64(&ctx, a, b);
            x[0].norm_sqr() + x[1].norm_sqr()
        },
        &SOLVE_C64_A,
        &SOLVE_C64_B,
        1.0e-5,
    );
    assert_close_c64_slice(
        c64_data(&a.grad().unwrap().unwrap().to_tensor().unwrap()),
        &fd_g_a,
        1.0e-6,
    );
    assert_close_c64_slice(
        c64_data(&b.grad().unwrap().unwrap().to_tensor().unwrap()),
        &fd_g_b,
        1.0e-6,
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
    assert_close_slice(f64_data(&values.to_tensor().unwrap()), &[1.0, 3.0], 1.0e-12);
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
    assert_finite_f64_tensor(&grad.to_tensor().unwrap());
}

#[test]
fn mixed_real_constant_and_tracked_complex_eigh_vector_backward() {
    let ctx = ad_test_ctx();
    let real = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let complex = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, -0.25),
                Complex64::new(0.0, 0.25),
                Complex64::new(0.5, 0.0),
            ],
        )
        .unwrap(),
        ctx,
    )
    .unwrap();
    let mixed = real.add(&complex).unwrap();
    let (_values, vectors) = mixed.eigh().unwrap();

    vectors
        .reduce_sum(Some(&[0, 1]))
        .unwrap()
        .backward()
        .unwrap();
    assert!(complex.grad().unwrap().is_some());
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

    let mut sorted = c64_data(&values.to_tensor().unwrap()).to_vec();
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
    assert_close_slice(f64_data(&x.to_tensor().unwrap()), &[1.0, 2.0], 1.0e-12);
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
    let upload_backend =
        CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0)).unwrap();
    let a_gpu = upload_tensor(upload_backend.runtime(), &a_host).unwrap();
    let b_gpu = upload_tensor(upload_backend.runtime(), &b_host).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend.clone()).unwrap();
    let a = EagerTensor::from_tensor_in(a_gpu, ctx.clone()).unwrap();
    let b = EagerTensor::from_tensor_in(b_gpu, ctx).unwrap();

    let x = a.solve(&b).unwrap();

    let x_host = download_tensor(upload_backend.runtime(), &x.to_tensor().unwrap()).unwrap();
    assert_eq!(x_host.shape(), &[2, 1]);
    assert_close_slice(f64_data(&x_host), &[1.8, -0.4], 1.0e-9);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_eager_qr_and_svd_f64_stay_resident_and_reconstruct() {
    if !gpu_available() {
        eprintln!("skipping CUDA eager f64 decomposition test: no CUDA device");
        return;
    }

    let expected = [1.0_f64, 3.0, 2.0, 4.0];
    let host = Tensor::from_vec_col_major(vec![2, 2], expected.to_vec()).unwrap();
    let backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0)).unwrap();
    let device = upload_tensor(backend.runtime(), &host).unwrap();
    let runtime = EagerRuntime::with_cuda_backend(backend.clone()).unwrap();
    let input = EagerTensor::from_tensor_in(device, runtime.clone()).unwrap();

    let (q, r) = input.qr().unwrap();
    let rrqr = input
        .rank_revealing_qr(RankRevealingQrOptions::default().rtol(1.0e-12))
        .unwrap();
    let (u, s, vt) = input.svd().unwrap();
    for output in [&q, &r, &rrqr.q, &rrqr.r, &u, &s, &vt] {
        assert_eq!(output.runtime().id(), runtime.id());
        assert!(output.to_tensor().unwrap().as_slice::<f64>().is_err());
    }
    for output in [&rrqr.column_permutation, &rrqr.rank] {
        assert_eq!(output.runtime().id(), runtime.id());
        assert!(output.to_tensor().unwrap().as_slice::<i64>().is_err());
    }

    let download = |output: &EagerTensor| {
        download_tensor(backend.runtime(), &output.to_tensor().unwrap()).unwrap()
    };
    let q = download(&q);
    let r = download(&r);
    assert_eq!(download(&rrqr.rank).as_slice::<i64>().unwrap(), &[2]);
    let u = download(&u);
    let s = download(&s);
    let vt = download(&vt);
    let qr = matmul_2x2_f64(f64_data(&q), f64_data(&r));
    let mut us = f64_data(&u).to_vec();
    for col in 0..2 {
        for row in 0..2 {
            us[row + 2 * col] *= f64_data(&s)[col];
        }
    }
    let usvt = matmul_2x2_f64(&us, f64_data(&vt));
    assert_close_slice(&qr, &expected, 1.0e-9);
    assert_close_slice(&usvt, &expected, 1.0e-9);
}

#[cfg(feature = "cuda")]
fn matmul_2x2_f64(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    (0..4)
        .map(|linear| {
            let row = linear % 2;
            let col = linear / 2;
            (0..2).map(|k| lhs[row + 2 * k] * rhs[k + 2 * col]).sum()
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_eager_qr_and_svd_c64_stay_resident_and_reconstruct() {
    if !gpu_available() {
        eprintln!("skipping CUDA eager c64 decomposition test: no CUDA device");
        return;
    }

    let expected = [
        Complex64::new(1.0, 0.5),
        Complex64::new(3.0, -0.5),
        Complex64::new(2.0, -1.0),
        Complex64::new(4.0, 0.25),
    ];
    let host = Tensor::from_vec_col_major(vec![2, 2], expected.to_vec()).unwrap();
    let backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0)).unwrap();
    let device = upload_tensor(backend.runtime(), &host).unwrap();
    let runtime = EagerRuntime::with_cuda_backend(backend.clone()).unwrap();
    let input = EagerTensor::from_tensor_in(device, runtime.clone()).unwrap();

    let (q, r) = input.qr().unwrap();
    let (u, s, vt) = input.svd().unwrap();
    for output in [&q, &r, &u, &s, &vt] {
        assert_eq!(output.runtime().id(), runtime.id());
        let resident = output.to_tensor().unwrap();
        assert!(match output.dtype() {
            tenferro_tensor::DType::C64 => resident.as_slice::<Complex64>().is_err(),
            tenferro_tensor::DType::F64 => resident.as_slice::<f64>().is_err(),
            dtype => panic!("unexpected decomposition dtype {dtype:?}"),
        });
    }

    let download = |output: &EagerTensor| {
        download_tensor(backend.runtime(), &output.to_tensor().unwrap()).unwrap()
    };
    let q = download(&q);
    let r = download(&r);
    let u = download(&u);
    let s = download(&s);
    let vt = download(&vt);
    let qr = matmul_2x2_c64(c64_data(&q), c64_data(&r));
    let mut us = c64_data(&u).to_vec();
    for col in 0..2 {
        for row in 0..2 {
            us[row + 2 * col] *= f64_data(&s)[col];
        }
    }
    let usvt = matmul_2x2_c64(&us, c64_data(&vt));
    assert_close_c64_slice(&qr, &expected, 1.0e-8);
    assert_close_c64_slice(&usvt, &expected, 1.0e-8);
}

#[cfg(feature = "cuda")]
fn matmul_2x2_c64(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
    (0..4)
        .map(|linear| {
            let row = linear % 2;
            let col = linear / 2;
            (0..2).map(|k| lhs[row + 2 * k] * rhs[k + 2 * col]).sum()
        })
        .collect()
}
