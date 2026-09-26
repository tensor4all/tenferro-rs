#![allow(dead_code)]

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::{ffi::OsString, sync::MutexGuard};

use num_complex::{Complex32, Complex64};

#[cfg(feature = "cpu-blas")]
use crate::CpuBackendKind;
#[cfg(feature = "cpu-faer")]
use crate::FaerParallelismExt;
use crate::{
    abs, add, broadcast_in_dim, clamp, compare, conj, div, dynamic_slice, dynamic_update_slice,
    embed_diagonal, extract_diagonal, gather, maximum, minimum, mul, neg, pad, pow, reduce_max,
    reduce_min, reduce_prod, reduce_sum, reduce_sum_squares, rem, reshape, scatter, select, sign,
    transpose, tril, triu, with_cpu_exec_session, CpuBackend, CpuContext, CpuExecSession, Error,
};
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
#[cfg(feature = "cpu-blas")]
use tenferro_tensor::StridedSliceSpec;
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, ContractionScalar,
    DotGeneralAccumulation, SessionCachedDot, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::{
    DType, Tensor, TensorRead, TensorView, TensorViewMut, TensorWrite, TypedTensor,
    TypedTensorView, TypedTensorViewMut,
};

#[test]
fn with_cpu_exec_session_checks_exact_marker_and_scopes_borrow() {
    let mut backend = CpuBackend::new();
    let value = backend
        .with_backend_session(|session| {
            // The session the owner builds is the CPU execution session, which
            // is what the capability bridge recognizes.
            assert_eq!(
                session.session_type_id(),
                std::any::TypeId::of::<crate::exec_session::CpuExecSessionMarker>()
            );
            with_cpu_exec_session(session, |session: &mut CpuExecSession<'_>| {
                let _: &mut CpuExecSession<'_> = session;
                17usize
            })
        })
        .expect("CpuBackend must expose its scoped CpuExecSession");
    assert_eq!(value, 17);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_parallelism_capability_runs_inside_a_cpu_session() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    backend
        .with_backend_session(|session| {
            session.with_faer_parallelism(|parallel| {
                let _ = parallel;
                Ok(())
            })
        })
        .unwrap();

    // A session that is not a CPU execution session is rejected with
    // `Unsupported`; this crate can only build CPU sessions, so that half is
    // asserted where a foreign session exists (the GPU and extension crates).
}

fn get_f64(t: &Tensor, idx: &[usize]) -> f64 {
    *t.as_typed::<f64>()
        .expect("expected F64 tensor")
        .get(idx)
        .unwrap()
}

fn get_c64(t: &Tensor, idx: &[usize]) -> Complex64 {
    *t.as_typed::<Complex64>()
        .expect("expected C64 tensor")
        .get(idx)
        .unwrap()
}

fn get_f32(t: &Tensor, idx: &[usize]) -> f32 {
    *t.as_typed::<f32>()
        .expect("expected F32 tensor")
        .get(idx)
        .unwrap()
}

fn get_c32(t: &Tensor, idx: &[usize]) -> Complex32 {
    *t.as_typed::<Complex32>()
        .expect("expected C32 tensor")
        .get(idx)
        .unwrap()
}

fn get_i64(t: &Tensor, idx: &[usize]) -> i64 {
    *t.as_typed::<i64>()
        .expect("expected I64 tensor")
        .get(idx)
        .unwrap()
}

fn get_i32(t: &Tensor, idx: &[usize]) -> i32 {
    *t.as_typed::<i32>()
        .expect("expected I32 tensor")
        .get(idx)
        .unwrap()
}

fn get_bool(t: &Tensor, idx: &[usize]) -> bool {
    *t.as_typed::<bool>()
        .expect("expected Bool tensor")
        .get(idx)
        .unwrap()
}

fn assert_f64_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "expected {expected}, got {actual}"
    );
}

fn assert_f64_close_tol(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "expected {expected}, got {actual}, tol={tol}"
    );
}

fn assert_c64_close(actual: Complex64, expected: Complex64) {
    assert_f64_close(actual.re, expected.re);
    assert_f64_close(actual.im, expected.im);
}

fn assert_c64_close_tol(actual: Complex64, expected: Complex64, tol: f64) {
    assert_f64_close_tol(actual.re, expected.re, tol);
    assert_f64_close_tol(actual.im, expected.im, tol);
}

fn col_major_index(rows: usize, row: usize, col: usize) -> usize {
    row + col * rows
}

fn transpose_f64(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn conjugate_transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
        }
    }
    out
}

fn grouped_gemm_reference_f64(
    lhs: &[f64],
    rhs: &[f64],
    out: &mut [f64],
    job: GroupedGemmJob,
    alpha: f64,
    beta: f64,
) {
    for col in 0..job.cols() {
        for row in 0..job.rows() {
            let mut acc = 0.0;
            for kk in 0..job.contracted() {
                let a = lhs[job.lhs_offset() + row + kk * job.rows()];
                let b = rhs[job.rhs_offset() + kk + col * job.contracted()];
                acc += a * b;
            }
            let out_idx = job.out_offset() + row + col * job.rows();
            out[out_idx] = alpha * acc + beta * out[out_idx];
        }
    }
}

fn grouped_gemm_reference_c64(
    lhs: &[Complex64],
    rhs: &[Complex64],
    out: &mut [Complex64],
    job: GroupedGemmJob,
    alpha: Complex64,
    beta: Complex64,
) {
    for col in 0..job.cols() {
        for row in 0..job.rows() {
            let mut acc = Complex64::new(0.0, 0.0);
            for kk in 0..job.contracted() {
                let a = lhs[job.lhs_offset() + row + kk * job.rows()];
                let b = rhs[job.rhs_offset() + kk + col * job.contracted()];
                acc += a * b;
            }
            let out_idx = job.out_offset() + row + col * job.rows();
            out[out_idx] = alpha * acc + beta * out[out_idx];
        }
    }
}

fn grouped_gemm_reference_f32(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    job: GroupedGemmJob,
    alpha: f32,
    beta: f32,
) {
    for col in 0..job.cols() {
        for row in 0..job.rows() {
            let mut acc = 0.0;
            for kk in 0..job.contracted() {
                let a = lhs[job.lhs_offset() + row + kk * job.rows()];
                let b = rhs[job.rhs_offset() + kk + col * job.contracted()];
                acc += a * b;
            }
            let out_idx = job.out_offset() + row + col * job.rows();
            out[out_idx] = alpha * acc + beta * out[out_idx];
        }
    }
}

fn grouped_gemm_reference_c32(
    lhs: &[Complex32],
    rhs: &[Complex32],
    out: &mut [Complex32],
    job: GroupedGemmJob,
    alpha: Complex32,
    beta: Complex32,
) {
    for col in 0..job.cols() {
        for row in 0..job.rows() {
            let mut acc = Complex32::new(0.0, 0.0);
            for kk in 0..job.contracted() {
                let a = lhs[job.lhs_offset() + row + kk * job.rows()];
                let b = rhs[job.rhs_offset() + kk + col * job.contracted()];
                acc += a * b;
            }
            let out_idx = job.out_offset() + row + col * job.rows();
            out[out_idx] = alpha * acc + beta * out[out_idx];
        }
    }
}

#[test]
fn grouped_gemm_shared_buffers_f64_matches_sequential_reference() {
    let lhs_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let rhs_data = vec![1.0, -1.0, 2.0, 3.0, 0.5, 4.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let out_initial = vec![10.0; 7];
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 3, 2),
        GroupedGemmJob::new(4, 6, 6, 1, 2, 3),
    ];
    let mut expected = out_initial.clone();
    for job in jobs {
        grouped_gemm_reference_f64(&lhs_data, &rhs_data, &mut expected, job, 2.0, 3.0);
    }

    let lhs = Tensor::from_vec_col_major(vec![lhs_data.len()], lhs_data).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![rhs_data.len()], rhs_data).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![out_initial.len()], out_initial).unwrap();
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F64(2.0),
        beta: ContractionScalar::F64(3.0),
    };
    let config = GroupedGemmConfig::new(&jobs, accumulation);
    let mut backend = CpuBackend::new();
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();

    backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.grouped_gemm_cached(
                Some(0),
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), expected.as_slice());
}

#[test]
fn grouped_gemm_shared_buffers_c64_matches_sequential_reference() {
    let lhs_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(3.0, 0.5),
        Complex64::new(4.0, -0.5),
    ];
    let rhs_data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(-1.0, 2.0),
    ];
    let out_initial = vec![Complex64::new(1.0, -1.0); 4];
    let jobs = [GroupedGemmJob::new(0, 0, 0, 2, 2, 2)];
    let alpha = Complex64::new(0.5, 1.0);
    let beta = Complex64::new(-1.0, 0.25);
    let mut expected = out_initial.clone();
    grouped_gemm_reference_c64(&lhs_data, &rhs_data, &mut expected, jobs[0], alpha, beta);

    let lhs = Tensor::from_vec_col_major(vec![lhs_data.len()], lhs_data).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![rhs_data.len()], rhs_data).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![out_initial.len()], out_initial).unwrap();
    let config = GroupedGemmConfig::new(
        &jobs,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::C64(alpha),
            beta: ContractionScalar::C64(beta),
        },
    );
    let mut backend = CpuBackend::new();
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();

    backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.grouped_gemm_cached(
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap();

    for (actual, expected) in out.as_slice::<Complex64>().unwrap().iter().zip(expected) {
        assert_c64_close_tol(*actual, expected, 1.0e-10);
    }
}

#[test]
fn grouped_gemm_covers_f32_and_c32() {
    let f32_job = GroupedGemmJob::new(0, 0, 0, 2, 2, 2);
    let f32_lhs = vec![1.0_f32, 2.0, 3.0, 4.0];
    let f32_rhs = vec![5.0_f32, 6.0, 7.0, 8.0];
    let f32_initial = vec![1.0_f32; 4];
    let mut f32_expected = f32_initial.clone();
    grouped_gemm_reference_f32(&f32_lhs, &f32_rhs, &mut f32_expected, f32_job, 0.5, 2.0);
    let lhs = Tensor::from_vec_col_major(vec![4], f32_lhs).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![4], f32_rhs).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![4], f32_initial).unwrap();
    let f32_jobs = [f32_job];
    let config = GroupedGemmConfig::new(
        &f32_jobs,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F32(0.5),
            beta: ContractionScalar::F32(2.0),
        },
    );
    let mut backend = CpuBackend::new();
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
    backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.grouped_gemm_cached(
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap();
    for (actual, expected) in out.as_slice::<f32>().unwrap().iter().zip(f32_expected) {
        assert!((*actual - expected).abs() < 1.0e-5);
    }

    let c32_job = GroupedGemmJob::new(0, 0, 0, 1, 2, 2);
    let c32_lhs = vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)];
    let c32_rhs = vec![
        Complex32::new(0.0, 1.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 1.0),
        Complex32::new(-1.0, 0.5),
    ];
    let c32_initial = vec![Complex32::new(0.5, -0.5); 2];
    let alpha = Complex32::new(1.0, -0.25);
    let beta = Complex32::new(0.25, 0.5);
    let mut c32_expected = c32_initial.clone();
    grouped_gemm_reference_c32(&c32_lhs, &c32_rhs, &mut c32_expected, c32_job, alpha, beta);
    let lhs = Tensor::from_vec_col_major(vec![2], c32_lhs).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![4], c32_rhs).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![2], c32_initial).unwrap();
    let c32_jobs = [c32_job];
    let config = GroupedGemmConfig::new(
        &c32_jobs,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::C32(alpha),
            beta: ContractionScalar::C32(beta),
        },
    );
    backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.grouped_gemm_cached(
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap();
    for (actual, expected) in out
        .as_slice::<Complex32>()
        .unwrap()
        .iter()
        .zip(c32_expected)
    {
        assert!((actual.re - expected.re).abs() < 1.0e-5);
        assert!((actual.im - expected.im).abs() < 1.0e-5);
    }
}

#[test]
fn grouped_gemm_rejects_overlapping_output_ranges() {
    let lhs = Tensor::from_vec_col_major(vec![8], vec![1.0_f64; 8]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![8], vec![1.0_f64; 8]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![8], vec![0.0_f64; 8]).unwrap();
    let jobs = [
        GroupedGemmJob::new(0, 0, 0, 2, 2, 2),
        GroupedGemmJob::new(2, 4, 4, 2, 2, 2),
    ];
    let config = GroupedGemmConfig::new(
        &jobs,
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    let mut backend = CpuBackend::new();
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
    let err = backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.grouped_gemm_cached(
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap_err();
    assert!(format!("{err}").contains("overlaps"));
    assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0; 8]);
}

#[test]
fn grouped_gemm_zero_jobs_is_noop_and_empty_contract_scales_output() {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![4], vec![2.0_f64, 3.0, 4.0, 5.0]).unwrap();
    let mut backend = CpuBackend::new();
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
    let no_jobs: [GroupedGemmJob; 0] = [];
    let noop = GroupedGemmConfig::new(
        &no_jobs,
        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
    );
    backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.grouped_gemm_cached(
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &noop,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 3.0, 4.0, 5.0]);

    let empty_jobs = [GroupedGemmJob::new(0, 0, 0, 2, 0, 2)];
    let scale = GroupedGemmConfig::new(
        &empty_jobs,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(3.0),
        },
    );
    backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.grouped_gemm_cached(
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &scale,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[6.0, 9.0, 12.0, 15.0]);
}

fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn matmul_c64(
    lhs: &[Complex64],
    rhs: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn diag_f64(values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}

fn diag_c64(values: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}

fn batch_matrix_f64_from_tensor(
    t: &Tensor,
    rows: usize,
    cols: usize,
    batch_idx: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(rows, i, j)] = get_f64(t, &[i, j, batch_idx]);
        }
    }
    out
}

fn matrix_f64_from_tensor(t: &Tensor, rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(rows, i, j)] = get_f64(t, &[i, j]);
        }
    }
    out
}

fn batch_vector_f64_from_tensor(t: &Tensor, len: usize, batch_idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    for (i, value) in out.iter_mut().enumerate().take(len) {
        *value = get_f64(t, &[i, batch_idx]);
    }
    out
}

fn matrix_c64_from_tensor(t: &Tensor, rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(rows, i, j)] = get_c64(t, &[i, j]);
        }
    }
    out
}

fn batch_matrix_c64_from_tensor(
    t: &Tensor,
    rows: usize,
    cols: usize,
    batch_idx: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(rows, i, j)] = get_c64(t, &[i, j, batch_idx]);
        }
    }
    out
}

fn vector_c64_from_tensor(t: &Tensor, len: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    for (i, value) in out.iter_mut().enumerate().take(len) {
        *value = get_c64(t, &[i]);
    }
    out
}

fn batch_vector_c64_from_tensor(t: &Tensor, len: usize, batch_idx: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    for (i, value) in out.iter_mut().enumerate().take(len) {
        *value = get_c64(t, &[i, batch_idx]);
    }
    out
}

fn simple_gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    }
}

fn diagonal_scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0, 1],
        scatter_dims_to_operand_dims: vec![0, 1],
        index_vector_dim: 1,
    }
}

#[test]
fn cpu_elementwise_kernels_live_in_internal_crate() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let cpu_manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml"))
        .expect("tenferro-cpu manifest must be readable");
    assert!(
        cpu_manifest.contains("tenferro-internal-cpu-kernels"),
        "tenferro-cpu must depend on the internal CPU kernel crate"
    );
    assert!(
        !manifest_dir.join("src/elementwise.rs").exists(),
        "elementwise kernels must not move back into the public tenferro-cpu crate"
    );
    assert!(
        !manifest_dir.join("src/buffer_pool.rs").exists(),
        "buffer pool implementation must not move back into the public tenferro-cpu crate"
    );

    let internal_dir = manifest_dir
        .parent()
        .expect("crate has workspace crates directory")
        .join("tenferro-internal-cpu-kernels");
    assert!(
        internal_dir.join("src/elementwise.rs").exists(),
        "internal CPU kernel crate must own elementwise kernels"
    );
    assert!(
        internal_dir
            .join("../tenferro-cpu-basic/src/buffer_pool.rs")
            .exists(),
        "basic CPU crate must own the shared buffer pool"
    );
    assert!(
        internal_dir
            .join("../tenferro-cpu-fused/src/lib.rs")
            .exists(),
        "fused CPU crate must own fused elementwise kernels"
    );
}

#[path = "tests/cpu_tests/backend_misc.rs"]
mod backend_misc;
#[path = "tests/cpu_tests/basic_ops.rs"]
mod basic_ops;
#[path = "tests/cpu_tests/blas1.rs"]
mod blas1;
#[path = "tests/cpu_tests/capability.rs"]
mod capability;
mod cast_matrix_coverage_tests;
#[path = "tests/cpu_tests/context.rs"]
mod context;
#[path = "tests/cpu_tests/dot_structural_analytic.rs"]
mod dot_structural_analytic;
#[path = "tests/cpu_tests/elementwise_reduction_helpers.rs"]
mod elementwise_reduction_helpers;
#[path = "tests/cpu_tests/indexing.rs"]
mod indexing;
#[path = "tests/cpu_indexing_coverage_tests.rs"]
mod indexing_coverage;
#[cfg(feature = "cpu-faer")]
#[path = "tests/cpu_tests/uninit_dot_output.rs"]
mod uninit_dot_output;

/// The analytic read view carries one arm per preset scalar, and the read entry for `pow` takes it for both
/// operands. The owned-tensor tests reach the floating arms only, so this drives every dtype, including the
/// boolean arm, whose analytic view is a unit marker and is therefore refused.
#[test]
fn analytic_read_view_covers_every_preset_scalar() {
    let mut buffers = crate::buffer_pool::BufferPool::new();

    macro_rules! pow_output {
        ($scalar:ty, $values:expr) => {{
            let values: Vec<$scalar> = $values;
            let lhs: Tensor =
                Tensor::from_vec_col_major(vec![values.len()], values.clone()).unwrap();
            let rhs: Tensor = Tensor::from_vec_col_major(vec![values.len()], values).unwrap();
            crate::analytic::pow_read_with_pool(
                &mut buffers,
                tenferro_tensor::TensorRead::from_tensor(&lhs),
                tenferro_tensor::TensorRead::from_tensor(&rhs),
            )
            .expect("pow admits this preset scalar")
        }};
    }

    macro_rules! pow_case_exact {
        ($scalar:ty, $values:expr, $expected:expr) => {{
            let output = pow_output!($scalar, $values);
            assert_eq!(output.as_slice::<$scalar>().unwrap(), $expected.as_slice());
        }};
    }

    // The complex arm computes through `exp`/`log`, so its result is exact only to the
    // precision of that pair; compare within a tolerance instead of demanding bit equality.
    macro_rules! pow_case_close {
        ($scalar:ty, $values:expr, $expected:expr) => {{
            let output = pow_output!($scalar, $values);
            let actual = output.as_slice::<$scalar>().unwrap();
            let expected: Vec<$scalar> = $expected;
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.iter().zip(&expected) {
                assert!(
                    (a.re - e.re).abs() < 1e-5 && (a.im - e.im).abs() < 1e-5,
                    "pow({}) = {:?}, expected {:?}",
                    stringify!($scalar),
                    actual,
                    expected
                );
            }
        }};
    }

    pow_case_exact!(f32, vec![2.0_f32, 3.0], vec![4.0_f32, 27.0]);
    pow_case_exact!(f64, vec![2.0_f64, 3.0], vec![4.0_f64, 27.0]);
    pow_case_exact!(i32, vec![2_i32, 3], vec![4_i32, 27]);
    pow_case_exact!(i64, vec![2_i64, 3], vec![4_i64, 27]);
    pow_case_close!(
        Complex32,
        vec![Complex32::new(2.0, 0.0), Complex32::new(3.0, 0.0)],
        vec![Complex32::new(4.0, 0.0), Complex32::new(27.0, 0.0)]
    );
    pow_case_close!(
        Complex64,
        vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
        vec![Complex64::new(4.0, 0.0), Complex64::new(27.0, 0.0)]
    );

    // The boolean analytic view is a unit marker, so `pow` has no arm for it and reports a
    // dtype mismatch rather than silently returning a value.
    let bools: Tensor = Tensor::from_vec_col_major(vec![2], vec![false, true]).unwrap();
    let error = crate::analytic::pow_read_with_pool(
        &mut buffers,
        tenferro_tensor::TensorRead::from_tensor(&bools),
        tenferro_tensor::TensorRead::from_tensor(&bools),
    )
    .expect_err("pow refuses the boolean analytic view");
    assert!(matches!(
        error,
        tenferro_tensor::Error::Validation {
            op: "pow",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        }
    ));
}
