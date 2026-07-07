#![allow(dead_code)]

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::{ffi::OsString, sync::MutexGuard};

use num_complex::{Complex32, Complex64};

use crate::buffer_pool::BufferPool;
#[cfg(feature = "cpu-blas")]
use crate::CpuBackendKind;
use crate::{
    abs, add, broadcast_in_dim, clamp, compare, conj, div, dynamic_slice, dynamic_update_slice,
    embed_diagonal, extract_diagonal, gather, maximum, minimum, mul, neg, pad, pow, reduce_max,
    reduce_min, reduce_prod, reduce_sum, rem, reshape, scatter, select, sign, transpose, tril,
    triu, typed_array_uninit_from_pool, CpuBackend, CpuContext, Error,
};
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
#[cfg(feature = "cpu-blas")]
use tenferro_tensor::StridedSliceSpec;
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSessionHost, ContractionScalar,
    DotGeneralAccumulation, SessionCachedDot, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::{
    DType, Tensor, TensorRead, TensorView, TensorViewMut, TensorWrite, TypedTensor,
    TypedTensorViewMut,
};

fn get_f64(t: &Tensor, idx: &[usize]) -> f64 {
    match t {
        Tensor::F64(inner) => *inner.get(idx).unwrap(),
        _ => panic!("expected F64 tensor"),
    }
}

fn get_c64(t: &Tensor, idx: &[usize]) -> Complex64 {
    match t {
        Tensor::C64(inner) => *inner.get(idx).unwrap(),
        _ => panic!("expected C64 tensor"),
    }
}

fn get_f32(t: &Tensor, idx: &[usize]) -> f32 {
    match t {
        Tensor::F32(inner) => *inner.get(idx).unwrap(),
        _ => panic!("expected F32 tensor"),
    }
}

fn get_c32(t: &Tensor, idx: &[usize]) -> Complex32 {
    match t {
        Tensor::C32(inner) => *inner.get(idx).unwrap(),
        _ => panic!("expected C32 tensor"),
    }
}

fn get_i64(t: &Tensor, idx: &[usize]) -> i64 {
    match t {
        Tensor::I64(inner) => *inner.get(idx).unwrap(),
        _ => panic!("expected I64 tensor"),
    }
}

fn get_i32(t: &Tensor, idx: &[usize]) -> i32 {
    match t {
        Tensor::I32(inner) => *inner.get(idx).unwrap(),
        _ => panic!("expected I32 tensor"),
    }
}

fn get_bool(t: &Tensor, idx: &[usize]) -> bool {
    match t {
        Tensor::Bool(inner) => *inner.get(idx).unwrap(),
        _ => panic!("expected Bool tensor"),
    }
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

    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        Some(0),
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
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

    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
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
    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
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
    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
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
    let err = BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap_err();
    assert!(format!("{err}").contains("overlaps"));
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
    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &noop,
        TensorWrite::from_tensor(&mut out),
    )
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
    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &scale,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[6.0, 9.0, 12.0, 15.0]);
}

#[test]
fn typed_array_uninit_from_pool_rejects_shape_product_overflow_without_panicking() {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
        let mut buffers = BufferPool::new();
        typed_array_uninit_from_pool::<f64>(&mut buffers, &[usize::MAX, 2])
    }));

    assert!(
        result.is_ok(),
        "typed_array_uninit_from_pool must return a typed error, not panic"
    );
    assert!(matches!(
        result.unwrap(),
        Err(tenferro_tensor::Error::InvalidConfig {
            op: "typed_array_uninit_from_pool",
            ..
        })
    ));
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

#[path = "tests/cpu_tests/backend_misc.rs"]
mod backend_misc;
#[path = "tests/cpu_tests/basic_ops.rs"]
mod basic_ops;
#[path = "tests/cpu_tests/capability.rs"]
mod capability;
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
