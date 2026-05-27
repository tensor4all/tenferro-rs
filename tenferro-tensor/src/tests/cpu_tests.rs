use std::rc::Rc;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::{ffi::OsString, sync::MutexGuard};

use num_complex::{Complex32, Complex64};

use crate::backend::TensorBackend;
#[cfg(feature = "cpu-faer")]
use crate::buffer_pool::BufferPool;
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
#[cfg(feature = "cpu-faer")]
use crate::cpu::linalg::faer_linalg;
use crate::cpu::{
    abs, add, broadcast_in_dim, clamp, compare, conj, div, dynamic_slice, dynamic_update_slice,
    embed_diagonal, extract_diagonal, gather, maximum, minimum, mul, neg, pad, reduce_max,
    reduce_min, reduce_prod, reduce_sum, reshape, scatter, select, sign, transpose, tril, triu,
    CpuBackend, CpuContext,
};
use crate::types::{DType, Tensor, TensorRead, TensorView, TypedTensor};

fn get_f64(t: &Tensor, idx: &[usize]) -> f64 {
    match t {
        Tensor::F64(inner) => *inner.get(idx),
        _ => panic!("expected F64 tensor"),
    }
}

fn get_c64(t: &Tensor, idx: &[usize]) -> Complex64 {
    match t {
        Tensor::C64(inner) => *inner.get(idx),
        _ => panic!("expected C64 tensor"),
    }
}

fn get_f32(t: &Tensor, idx: &[usize]) -> f32 {
    match t {
        Tensor::F32(inner) => *inner.get(idx),
        _ => panic!("expected F32 tensor"),
    }
}

fn get_c32(t: &Tensor, idx: &[usize]) -> Complex32 {
    match t {
        Tensor::C32(inner) => *inner.get(idx),
        _ => panic!("expected C32 tensor"),
    }
}

fn get_i64(t: &Tensor, idx: &[usize]) -> i64 {
    match t {
        Tensor::I64(inner) => *inner.get(idx),
        _ => panic!("expected I64 tensor"),
    }
}

fn get_i32(t: &Tensor, idx: &[usize]) -> i32 {
    match t {
        Tensor::I32(inner) => *inner.get(idx),
        _ => panic!("expected I32 tensor"),
    }
}

fn get_bool(t: &Tensor, idx: &[usize]) -> bool {
    match t {
        Tensor::Bool(inner) => *inner.get(idx),
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
    for i in 0..len {
        out[i] = get_f64(t, &[i, batch_idx]);
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
    for i in 0..len {
        out[i] = get_c64(t, &[i]);
    }
    out
}

fn batch_vector_c64_from_tensor(t: &Tensor, len: usize, batch_idx: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    for i in 0..len {
        out[i] = get_c64(t, &[i, batch_idx]);
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
fn cpu_context_from_env_respects_rayon_num_threads() {
    with_rayon_num_threads(Some("3"), || {
        let ctx = CpuContext::from_env();
        assert_eq!(ctx.num_threads(), 3);
    });
}

#[test]
fn cpu_context_from_env_falls_back_to_affinity_when_rayon_num_threads_is_absent() {
    with_rayon_num_threads(None, || {
        let ctx = CpuContext::from_env();
        assert_eq!(ctx.num_threads(), crate::cpu::available_parallelism());
    });
}

#[test]
fn cpu_context_try_from_env_rejects_invalid_rayon_num_threads() {
    with_rayon_num_threads(Some("not-a-number"), || {
        assert!(CpuContext::try_from_env().is_err());
    });
}

#[test]
fn cpu_context_try_from_env_rejects_zero_rayon_num_threads() {
    with_rayon_num_threads(Some("0"), || {
        let err = match CpuContext::try_from_env() {
            Ok(_) => panic!("expected zero RAYON_NUM_THREADS to be rejected"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains("CpuContext::try_from_env"));
        assert!(format!("{err}").contains("thread count must be at least 1"));
    });
}

#[test]
#[should_panic(expected = "thread count must be at least 1")]
fn cpu_context_with_threads_zero_panics_for_compatibility() {
    let _ctx = CpuContext::with_threads(0);
}

#[test]
fn cpu_backend_new_matches_context_from_env() {
    with_rayon_num_threads(Some("2"), || {
        let backend = CpuBackend::new();
        assert_eq!(backend.num_threads(), 2);
    });
}

#[test]
fn cpu_backend_new_falls_back_to_affinity_when_rayon_num_threads_is_absent() {
    with_rayon_num_threads(None, || {
        let backend = CpuBackend::new();
        assert_eq!(backend.num_threads(), crate::cpu::available_parallelism());
    });
}

#[test]
fn cpu_backend_try_new_propagates_invalid_rayon_num_threads() {
    with_rayon_num_threads(Some("not-a-number"), || {
        assert!(CpuBackend::try_new().is_err());
    });
}

#[test]
fn test_with_backend_session_runs_compiled_ops() {
    let mut backend = CpuBackend::with_threads(2);
    let result = backend.with_backend_session(|session| {
        session
            .add(
                &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0])),
                &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0])),
            )
            .unwrap()
    });
    assert_eq!(get_f64(&result, &[0]), 4.0);
    assert_eq!(get_f64(&result, &[1]), 6.0);
}

#[test]
fn cpu_context_install_runs_on_caller_thread() {
    let ctx = CpuContext::with_threads(2);
    let caller_thread = std::thread::current().id();
    let seen_thread = ctx.install(|| std::thread::current().id());
    assert_eq!(seen_thread, caller_thread);
}

#[test]
fn cpu_install_accepts_non_send_state() {
    let ctx = CpuContext::with_threads(2);
    let state = Rc::new(41usize);
    let seen = ctx.install(|| *state + 1);
    assert_eq!(seen, 42);

    let backend = CpuBackend::with_threads(2);
    let state = Rc::new(20usize);
    let seen = backend.install(|| *state + 2);
    assert_eq!(seen, 22);
}

#[test]
fn cpu_backend_exec_session_runs_on_caller_thread() {
    let mut backend = CpuBackend::with_threads(2);
    let caller_thread = std::thread::current().id();
    let seen_thread = backend.with_backend_session(|_| std::thread::current().id());
    assert_eq!(seen_thread, caller_thread);
}

#[test]
fn cpu_backend_shared_context() {
    let ctx = Arc::new(CpuContext::with_threads(3));
    let b1 = CpuBackend::from_context(ctx.clone());
    let b2 = CpuBackend::from_context(ctx);
    assert!(Arc::ptr_eq(&b1.ctx, &b2.ctx));
}

#[test]
fn cpu_affinity_available_parallelism_reports_positive_count() {
    assert!(crate::cpu::available_parallelism() >= 1);
}

#[test]
fn cpu_backend_from_context_shares_runtime_owner() {
    let ctx = Arc::new(CpuContext::with_threads(3));
    let b1 = CpuBackend::from_context(ctx.clone());
    let b2 = CpuBackend::from_context(ctx);
    assert_eq!(b1.num_threads(), 3);
    assert_eq!(b2.num_threads(), 3);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn cpu_context_faer_policy_is_seq_for_one_thread() {
    let ctx = CpuContext::with_threads(1);
    assert!(matches!(ctx.faer_par(), faer::Par::Seq));
}

#[test]
fn cpu_context_with_threads_reports_requested_size() {
    let ctx = CpuContext::with_threads(2);
    assert_eq!(ctx.num_threads(), 2);
}

#[test]
fn cpu_context_install_executes_closure() {
    let ctx = CpuContext::with_threads(1);
    let seen = ctx.install(|| 1 + 1);
    assert_eq!(seen, 2);
}

fn env_lock() -> MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct RayonNumThreadsEnvGuard {
    _lock: MutexGuard<'static, ()>,
    prev: Option<OsString>,
}

impl RayonNumThreadsEnvGuard {
    fn new(value: Option<&str>) -> Self {
        let lock = env_lock();
        let prev = std::env::var_os("RAYON_NUM_THREADS");

        match value {
            Some(value) => std::env::set_var("RAYON_NUM_THREADS", value),
            None => std::env::remove_var("RAYON_NUM_THREADS"),
        }

        Self { _lock: lock, prev }
    }
}

impl Drop for RayonNumThreadsEnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(value) => std::env::set_var("RAYON_NUM_THREADS", value),
            None => std::env::remove_var("RAYON_NUM_THREADS"),
        }
    }
}

fn with_rayon_num_threads<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _guard = RayonNumThreadsEnvGuard::new(value);
    f()
}

#[test]
fn test_zeros_ones() {
    let z = TypedTensor::<f64>::zeros(vec![2, 3]);
    assert_eq!(z.shape, vec![2, 3]);
    assert_eq!(z.n_elements(), 6);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(*z.get(&[i, j]), 0.0);
        }
    }

    let o = TypedTensor::<f64>::ones(vec![2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(*o.get(&[i, j]), 1.0);
        }
    }
}

#[test]
fn test_from_vec_uses_column_major_indices() {
    let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(*t.get(&[0, 0]), 1.0);
    assert_eq!(*t.get(&[1, 0]), 2.0);
    assert_eq!(*t.get(&[0, 1]), 3.0);
    assert_eq!(*t.get(&[1, 1]), 4.0);
    assert_eq!(*t.get(&[0, 2]), 5.0);
    assert_eq!(*t.get(&[1, 2]), 6.0);
}

#[test]
fn test_tensor_metadata() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]));
    assert_eq!(t.shape(), &[2, 1]);
    assert_eq!(t.dtype(), DType::F64);
}

#[test]
fn test_reshape() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let r = reshape(&t, &[3, 2]).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(get_f64(&r, &[0, 0]), 1.0);
    assert_eq!(get_f64(&r, &[1, 0]), 2.0);
    assert_eq!(get_f64(&r, &[2, 0]), 3.0);
    assert_eq!(get_f64(&r, &[0, 1]), 4.0);
    assert_eq!(get_f64(&r, &[1, 1]), 5.0);
    assert_eq!(get_f64(&r, &[2, 1]), 6.0);
}

#[test]
fn test_add_mul() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![10.0, 20.0, 30.0, 40.0],
    ));
    let sum = add(&a, &b).unwrap();
    let prod = mul(&a, &b).unwrap();

    assert_eq!(get_f64(&sum, &[0, 0]), 11.0);
    assert_eq!(get_f64(&sum, &[1, 0]), 22.0);
    assert_eq!(get_f64(&sum, &[0, 1]), 33.0);
    assert_eq!(get_f64(&sum, &[1, 1]), 44.0);

    assert_eq!(get_f64(&prod, &[0, 0]), 10.0);
    assert_eq!(get_f64(&prod, &[1, 0]), 40.0);
    assert_eq!(get_f64(&prod, &[0, 1]), 90.0);
    assert_eq!(get_f64(&prod, &[1, 1]), 160.0);
}

#[test]
fn test_add_mul_i64() {
    let a = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1, 2, 3, 4],
    ));
    let b = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![10, 20, 30, 40],
    ));
    let sum = add(&a, &b).unwrap();
    let prod = mul(&a, &b).unwrap();

    assert_eq!(get_i64(&sum, &[0, 0]), 11);
    assert_eq!(get_i64(&sum, &[1, 0]), 22);
    assert_eq!(get_i64(&sum, &[0, 1]), 33);
    assert_eq!(get_i64(&sum, &[1, 1]), 44);

    assert_eq!(get_i64(&prod, &[0, 0]), 10);
    assert_eq!(get_i64(&prod, &[1, 0]), 40);
    assert_eq!(get_i64(&prod, &[0, 1]), 90);
    assert_eq!(get_i64(&prod, &[1, 1]), 160);
}

#[test]
fn test_add_mul_rank0_broadcast() {
    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0]));
    let tensor = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));

    let scalar_plus_tensor = add(&scalar, &tensor).unwrap();
    let tensor_plus_scalar = add(&tensor, &scalar).unwrap();
    let scalar_times_tensor = mul(&scalar, &tensor).unwrap();
    let tensor_times_scalar = mul(&tensor, &scalar).unwrap();

    for actual in [&scalar_plus_tensor, &tensor_plus_scalar] {
        assert_eq!(actual.shape(), &[2, 2]);
        assert_eq!(get_f64(actual, &[0, 0]), 3.0);
        assert_eq!(get_f64(actual, &[1, 0]), 4.0);
        assert_eq!(get_f64(actual, &[0, 1]), 5.0);
        assert_eq!(get_f64(actual, &[1, 1]), 6.0);
    }

    for actual in [&scalar_times_tensor, &tensor_times_scalar] {
        assert_eq!(actual.shape(), &[2, 2]);
        assert_eq!(get_f64(actual, &[0, 0]), 2.0);
        assert_eq!(get_f64(actual, &[1, 0]), 4.0);
        assert_eq!(get_f64(actual, &[0, 1]), 6.0);
        assert_eq!(get_f64(actual, &[1, 1]), 8.0);
    }
}

#[test]
fn test_mul_rank0_real_scalar_broadcasts_over_complex_tensor() {
    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0]));
    let tensor = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
    ));

    let scalar_times_tensor = mul(&scalar, &tensor).unwrap();
    let tensor_times_scalar = mul(&tensor, &scalar).unwrap();

    for actual in [&scalar_times_tensor, &tensor_times_scalar] {
        assert_eq!(actual.shape(), &[2]);
        assert_c64_close(get_c64(actual, &[0]), Complex64::new(2.0, 4.0));
        assert_c64_close(get_c64(actual, &[1]), Complex64::new(-6.0, 1.0));
    }
}

#[test]
fn test_mul_rank0_complex_scalar_broadcasts_over_complex_tensor() {
    let scalar = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![],
        vec![Complex64::new(2.0, -1.0)],
    ));
    let tensor = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
    ));
    let output = mul(&scalar, &tensor).unwrap();

    assert_c64_close(get_c64(&output, &[0]), Complex64::new(4.0, 3.0));
    assert_c64_close(get_c64(&output, &[1]), Complex64::new(-5.5, 4.0));
}

#[test]
fn test_rank0_typed_tensor_behaves_like_scalar() {
    let mut zeros = TypedTensor::<f64>::zeros(vec![]);
    assert_eq!(zeros.shape, Vec::<usize>::new());
    assert_eq!(zeros.n_elements(), 1);
    assert_eq!(zeros.linear_offset(&[]), 0);
    assert_eq!(zeros.get(&[]), &0.0);

    *zeros.get_mut(&[]) = 2.5;
    assert_eq!(zeros.host_data(), &[2.5]);

    let ones = TypedTensor::<f64>::ones(vec![]);
    assert_eq!(ones.shape, Vec::<usize>::new());
    assert_eq!(ones.n_elements(), 1);
    assert_eq!(ones.get(&[]), &1.0);

    let scalar = TypedTensor::<f64>::from_vec_col_major(vec![], vec![7.0]);
    assert_eq!(scalar.shape, Vec::<usize>::new());
    assert_eq!(scalar.n_elements(), 1);
    assert_eq!(scalar.linear_offset(&[]), 0);
    assert_eq!(scalar.get(&[]), &7.0);
}

#[test]
fn test_reduce_sum() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let r = reduce_sum(&t, &[0]).unwrap();
    assert_eq!(r.shape(), &[3]);
    assert_eq!(get_f64(&r, &[0]), 3.0);
    assert_eq!(get_f64(&r, &[1]), 7.0);
    assert_eq!(get_f64(&r, &[2]), 11.0);

    let all = reduce_sum(&t, &[0, 1]).unwrap();
    assert!(all.shape().is_empty());
    assert_eq!(get_f64(&all, &[]), 21.0);
}

#[test]
fn test_reduce_prod() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let r = reduce_prod(&t, &[0]).unwrap();
    assert_eq!(r.shape(), &[3]);
    assert_eq!(get_f64(&r, &[0]), 2.0);
    assert_eq!(get_f64(&r, &[1]), 12.0);
    assert_eq!(get_f64(&r, &[2]), 30.0);

    let all = reduce_prod(&t, &[0, 1]).unwrap();
    assert!(all.shape().is_empty());
    assert_eq!(get_f64(&all, &[]), 720.0);
}

#[test]
fn test_reduce_max_and_min() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let max_cols = reduce_max(&t, &[0]).unwrap();
    assert_eq!(max_cols.shape(), &[3]);
    assert_eq!(get_f64(&max_cols, &[0]), 2.0);
    assert_eq!(get_f64(&max_cols, &[1]), 4.0);
    assert_eq!(get_f64(&max_cols, &[2]), 6.0);

    let max_all = reduce_max(&t, &[0, 1]).unwrap();
    assert!(max_all.shape().is_empty());
    assert_eq!(get_f64(&max_all, &[]), 6.0);

    let min_rows = reduce_min(&t, &[1]).unwrap();
    assert_eq!(min_rows.shape(), &[2]);
    assert_eq!(get_f64(&min_rows, &[0]), 1.0);
    assert_eq!(get_f64(&min_rows, &[1]), 2.0);

    let min_all = reduce_min(&t, &[0, 1]).unwrap();
    assert!(min_all.shape().is_empty());
    assert_eq!(get_f64(&min_all, &[]), 1.0);
}

#[test]
fn test_backend_reduce_prod_max_and_min_delegate_to_cpu_reduction_impls() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let mut backend = CpuBackend::new();

    let prod = backend.reduce_prod(&t, &[0]).unwrap();
    assert_eq!(prod.shape(), &[3]);
    assert_eq!(get_f64(&prod, &[0]), 2.0);
    assert_eq!(get_f64(&prod, &[1]), 12.0);
    assert_eq!(get_f64(&prod, &[2]), 30.0);

    let max = backend.reduce_max(&t, &[1]).unwrap();
    assert_eq!(max.shape(), &[2]);
    assert_eq!(get_f64(&max, &[0]), 5.0);
    assert_eq!(get_f64(&max, &[1]), 6.0);

    let min = backend.reduce_min(&t, &[0, 1]).unwrap();
    assert!(min.shape().is_empty());
    assert_eq!(get_f64(&min, &[]), 1.0);
}

#[test]
fn test_slice() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4, 4],
        (1..=16).map(|value| value as f64).collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend
        .slice(
            &input,
            &SliceConfig {
                starts: vec![1, 1],
                limits: vec![3, 3],
                strides: vec![1, 1],
            },
        )
        .unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64(&out, &[0, 0]), 6.0);
    assert_eq!(get_f64(&out, &[1, 0]), 7.0);
    assert_eq!(get_f64(&out, &[0, 1]), 10.0);
    assert_eq!(get_f64(&out, &[1, 1]), 11.0);
}

#[test]
fn test_reverse_axis_zero() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let mut backend = CpuBackend::new();
    let out = backend.reverse(&input, &[0]).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 2.0);
    assert_eq!(get_f64(&out, &[1, 0]), 1.0);
    assert_eq!(get_f64(&out, &[0, 1]), 4.0);
    assert_eq!(get_f64(&out, &[1, 1]), 3.0);
    assert_eq!(get_f64(&out, &[0, 2]), 6.0);
    assert_eq!(get_f64(&out, &[1, 2]), 5.0);
}

#[test]
fn test_reverse_accepts_i64_data_tensor() {
    let input = Tensor::from_vec_col_major(vec![3], vec![1_i64, 2, 3]);
    let mut backend = CpuBackend::new();

    let out = backend.reverse(&input, &[0]).unwrap();

    assert_eq!(out.dtype(), DType::I64);
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.as_slice::<i64>(), Some([3, 2, 1].as_slice()));
}

#[test]
fn tensor_index_select_trailing_axis_returns_expected_values() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let out = input.index_select(-1, &[2, 0, 2], &mut backend).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_f64_close(get_f64(&out, &[0, 0]), 5.0);
    assert_f64_close(get_f64(&out, &[1, 0]), 6.0);
    assert_f64_close(get_f64(&out, &[0, 1]), 1.0);
    assert_f64_close(get_f64(&out, &[1, 1]), 2.0);
    assert_f64_close(get_f64(&out, &[0, 2]), 5.0);
    assert_f64_close(get_f64(&out, &[1, 2]), 6.0);
}

#[test]
fn tensor_index_select_rejects_invalid_axis_and_position() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);

    let axis_err = input.index_select(-2, &[0], &mut backend).unwrap_err();
    assert!(axis_err.to_string().contains("index_select"));
    assert!(axis_err.to_string().contains("axis"));

    let position_err = input.index_select(0, &[3], &mut backend).unwrap_err();
    assert!(position_err.to_string().contains("index_select"));
    assert!(position_err.to_string().contains("position"));
}

#[test]
fn tensor_stack_trailing_axis_packs_scalars_vectors_and_matrices() {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![], vec![1.0_f64]);
    let b = Tensor::from_vec_col_major(vec![], vec![2.0_f64]);
    let scalars = Tensor::stack(&[&a, &b], -1, &mut backend).unwrap();
    assert_eq!(scalars.shape(), &[2]);
    assert_eq!(scalars.as_slice::<f64>().unwrap(), &[1.0, 2.0]);

    let v0 = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let v1 = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
    let vectors = Tensor::stack(&[&v0, &v1], -1, &mut backend).unwrap();
    assert_eq!(vectors.shape(), &[2, 2]);
    assert_f64_close(get_f64(&vectors, &[0, 0]), 1.0);
    assert_f64_close(get_f64(&vectors, &[1, 0]), 2.0);
    assert_f64_close(get_f64(&vectors, &[0, 1]), 3.0);
    assert_f64_close(get_f64(&vectors, &[1, 1]), 4.0);

    let m0 = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]);
    let m1 = Tensor::from_vec_col_major(vec![2, 1], vec![3.0_f64, 4.0]);
    let matrices = Tensor::stack(&[&m0, &m1], -1, &mut backend).unwrap();
    assert_eq!(matrices.shape(), &[2, 1, 2]);
    assert_f64_close(get_f64(&matrices, &[0, 0, 0]), 1.0);
    assert_f64_close(get_f64(&matrices, &[1, 0, 0]), 2.0);
    assert_f64_close(get_f64(&matrices, &[0, 0, 1]), 3.0);
    assert_f64_close(get_f64(&matrices, &[1, 0, 1]), 4.0);
}

#[test]
fn tensor_index_select_reuses_reclaimed_cpu_buffer() {
    let mut backend = CpuBackend::new();
    let reusable = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6]);
    let expected_ptr = reusable.as_slice::<f64>().unwrap().as_ptr();
    backend.reclaim_buffer(reusable);

    let input = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = input.index_select(-1, &[2, 0, 1], &mut backend).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap().as_ptr(), expected_ptr);
}

#[test]
fn tensor_stack_reuses_reclaimed_cpu_buffer() {
    let mut backend = CpuBackend::new();
    let reusable = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]);
    let expected_ptr = reusable.as_slice::<f64>().unwrap().as_ptr();
    backend.reclaim_buffer(reusable);

    let x0 = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let x1 = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
    let out = Tensor::stack(&[&x0, &x1], -1, &mut backend).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap().as_ptr(), expected_ptr);
}

#[test]
fn test_reverse_axis_out_of_bounds_returns_error() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let mut backend = CpuBackend::new();

    let err = backend.reverse(&input, &[1]).unwrap_err();

    assert!(matches!(
        err,
        crate::Error::AxisOutOfBounds {
            op: "reverse",
            axis: 1,
            rank: 1,
        }
    ));
}

#[test]
fn test_gather_rejects_fractional_float_indices() {
    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1], vec![1.5]));
    let mut backend = CpuBackend::new();

    let err = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::InvalidConfig {
            op: "index_tensor",
            ..
        }
    ));
}

#[test]
fn test_gather_rejects_complex_indices() {
    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![1, 1],
        vec![Complex64::new(1.0, 0.0)],
    ));
    let mut backend = CpuBackend::new();

    let err = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::InvalidConfig {
            op: "index_tensor",
            ..
        }
    ));
}

#[test]
fn test_dynamic_slice_rejects_oversized_window() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    let starts = Tensor::from_vec_col_major(vec![1], vec![0_i64]);
    let mut backend = CpuBackend::new();

    let err = backend.dynamic_slice(&input, &starts, &[3]).unwrap_err();

    assert!(matches!(
        err,
        crate::Error::InvalidConfig {
            op: "dynamic_slice",
            ..
        }
    ));
}

#[test]
fn test_large_float_index_outside_exact_integer_range_returns_error() {
    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![1, 1],
        vec![9_007_199_254_740_995.0f64],
    ));
    let mut backend = CpuBackend::new();

    let err = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::InvalidConfig {
            op: "index_tensor",
            ..
        }
    ));
}

#[test]
fn test_invalid_slice_config_returns_error() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let mut backend = CpuBackend::new();

    let err = backend
        .slice(
            &input,
            &SliceConfig {
                starts: vec![0, 0, 0],
                limits: vec![2, 2],
                strides: vec![1, 1],
            },
        )
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::RankMismatch { op: "slice", .. }
    ));
}

#[test]
fn test_invalid_pad_config_returns_error() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    let mut backend = CpuBackend::new();

    let err = backend
        .pad(
            &input,
            &PadConfig {
                edge_padding_low: vec![0],
                edge_padding_high: vec![0, 0],
                interior_padding: vec![0],
            },
        )
        .unwrap_err();
    assert!(matches!(err, crate::Error::RankMismatch { op: "pad", .. }));
}

#[test]
fn test_gather_rejects_malformed_offset_dims() {
    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let start_indices = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 1, 2]);
    let mut backend = CpuBackend::new();
    let config = GatherConfig {
        offset_dims: vec![2],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 2],
    };

    let err = backend
        .gather(&operand, &start_indices, &config)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::AxisOutOfBounds { op: "gather", .. }
    ));
}

#[test]
fn test_scatter_rejects_update_window_dim_out_of_bounds() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3, 3]));
    let scatter_indices = Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 0, 1, 1, 2, 2]);
    let updates = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3, 3],
        vec![0.0; 27],
    ));
    let mut backend = CpuBackend::new();
    let config = ScatterConfig {
        update_window_dims: vec![0, 3],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![1, 2],
        index_vector_dim: 1,
    };

    let err = backend
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::AxisOutOfBounds {
            op: "scatter",
            axis: 3,
            ..
        }
    ));
}

#[test]
fn test_scatter_rejects_too_many_update_window_dims() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3, 3]));
    let scatter_indices = Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 0, 1, 1, 2, 2]);
    let updates = Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![0.0; 3]));
    let mut backend = CpuBackend::new();
    let config = ScatterConfig {
        update_window_dims: vec![0, 1],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![1, 2],
        index_vector_dim: 1,
    };

    let err = backend
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::InvalidConfig { op: "scatter", ref message }
        if message.contains("exceeds update rank")
    ));
}

#[test]
fn test_concatenate_axis_zero() {
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ));
    let mut backend = CpuBackend::new();
    let out = backend.concatenate(&[&lhs, &rhs], 0).unwrap();

    assert_eq!(out.shape(), &[4, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 1.0);
    assert_eq!(get_f64(&out, &[1, 0]), 2.0);
    assert_eq!(get_f64(&out, &[2, 0]), 7.0);
    assert_eq!(get_f64(&out, &[3, 0]), 8.0);
    assert_eq!(get_f64(&out, &[0, 1]), 3.0);
    assert_eq!(get_f64(&out, &[1, 1]), 4.0);
    assert_eq!(get_f64(&out, &[2, 1]), 9.0);
    assert_eq!(get_f64(&out, &[3, 1]), 10.0);
    assert_eq!(get_f64(&out, &[0, 2]), 5.0);
    assert_eq!(get_f64(&out, &[1, 2]), 6.0);
    assert_eq!(get_f64(&out, &[2, 2]), 11.0);
    assert_eq!(get_f64(&out, &[3, 2]), 12.0);
}

#[test]
fn test_dot_general_matmul() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 4],
        vec![
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
    ));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    assert_eq!(c.shape(), &[2, 4]);
    assert_eq!(get_f64(&c, &[0, 0]), 38.0);
    assert_eq!(get_f64(&c, &[1, 0]), 83.0);
    assert_eq!(get_f64(&c, &[0, 1]), 44.0);
    assert_eq!(get_f64(&c, &[1, 1]), 98.0);
    assert_eq!(get_f64(&c, &[0, 3]), 56.0);
    assert_eq!(get_f64(&c, &[1, 3]), 128.0);
}

#[test]
fn test_dot_general_with_conj_matches_materialized_complex_matmul() {
    let lhs_data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 0.5),
        Complex64::new(2.0, -1.0),
        Complex64::new(0.25, 4.0),
    ];
    let rhs_data = vec![
        Complex64::new(-2.0, 1.0),
        Complex64::new(1.5, -0.25),
        Complex64::new(0.5, 3.0),
        Complex64::new(-1.0, -2.0),
    ];
    let lhs = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        lhs_data.clone(),
    ));
    let rhs = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        rhs_data.clone(),
    ));
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut backend = CpuBackend::new();

    let out = backend
        .dot_general_with_conj(&lhs, &rhs, &config, true, true)
        .unwrap();

    let lhs_conj: Vec<Complex64> = lhs_data.iter().map(|value| value.conj()).collect();
    let rhs_conj: Vec<Complex64> = rhs_data.iter().map(|value| value.conj()).collect();
    let expected = matmul_c64(&lhs_conj, &rhs_conj, 2, 2, 2);
    for col in 0..2 {
        for row in 0..2 {
            assert_c64_close(
                get_c64(&out, &[row, col]),
                expected[col_major_index(2, row, col)],
            );
        }
    }
}

#[test]
fn test_dot_general_read_accepts_tensor_and_view_inputs() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs_shape = [3usize, 2];
    let rhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_view = TensorView::f64(&rhs_shape, &rhs_data).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut backend = CpuBackend::new();

    let direct = backend
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_view(rhs_view),
            &config,
        )
        .unwrap();
    assert_eq!(direct.shape(), &[2, 2]);
    assert_eq!(direct.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);

    let session = backend.with_backend_session(|exec| {
        exec.dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_view(rhs_view),
            &config,
        )
    });
    let session = session.unwrap();
    assert_eq!(
        session.as_slice::<f64>().unwrap(),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn test_dot_general_inner_product_returns_rank0_scalar() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![4.0, 5.0, 6.0],
    ));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![0],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    assert!(c.shape().is_empty());
    assert_eq!(get_f64(&c, &[]), 32.0);
}

#[test]
fn test_dot_general_zero_sized_matmul_returns_empty_matrix() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 0], Vec::new()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 0], Vec::new()));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();

    assert_eq!(c.shape(), &[0, 0]);
    match c {
        Tensor::F64(inner) => assert!(inner.host_data().is_empty()),
        _ => panic!("expected F64 tensor"),
    }
}

#[test]
fn test_dot_general_zero_contracting_dim_returns_zero_filled_output() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 0], Vec::new()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 3], Vec::new()));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    match c {
        Tensor::F64(inner) => assert_eq!(inner.host_data(), &[0.0; 6]),
        _ => panic!("expected F64 tensor"),
    }
}

#[test]
fn test_dot_general_falls_back_for_unfusable_lhs_batch_layout() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2, 2],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2, 2],
        vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ],
    ));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![3],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![0, 2],
                rhs_batch_dims: vec![2, 3],
            },
        )
        .unwrap();

    assert_eq!(c.shape(), &[2, 2, 2, 2]);
    assert_eq!(get_f64(&c, &[0, 0, 0, 0]), 1.0);
    assert_eq!(get_f64(&c, &[1, 0, 0, 0]), 3.0);
    assert_eq!(get_f64(&c, &[0, 1, 0, 0]), 9.0);
    assert_eq!(get_f64(&c, &[1, 1, 0, 0]), 11.0);
    assert_eq!(get_f64(&c, &[0, 0, 1, 0]), 2.0);
    assert_eq!(get_f64(&c, &[1, 0, 1, 0]), 4.0);
    assert_eq!(get_f64(&c, &[0, 1, 1, 0]), 10.0);
    assert_eq!(get_f64(&c, &[1, 1, 1, 0]), 12.0);
    assert_eq!(get_f64(&c, &[0, 0, 0, 1]), 5.0);
    assert_eq!(get_f64(&c, &[1, 0, 0, 1]), 7.0);
    assert_eq!(get_f64(&c, &[0, 1, 0, 1]), 13.0);
    assert_eq!(get_f64(&c, &[1, 1, 0, 1]), 15.0);
    assert_eq!(get_f64(&c, &[0, 0, 1, 1]), 6.0);
    assert_eq!(get_f64(&c, &[1, 0, 1, 1]), 8.0);
    assert_eq!(get_f64(&c, &[0, 1, 1, 1]), 14.0);
    assert_eq!(get_f64(&c, &[1, 1, 1, 1]), 16.0);
}

#[test]
fn test_transpose() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let tr = transpose(&t, &[1, 0]).unwrap();
    assert_eq!(tr.shape(), &[3, 2]);
    assert_eq!(get_f64(&tr, &[0, 0]), 1.0);
    assert_eq!(get_f64(&tr, &[0, 1]), 2.0);
    assert_eq!(get_f64(&tr, &[1, 0]), 3.0);
    assert_eq!(get_f64(&tr, &[1, 1]), 4.0);
    assert_eq!(get_f64(&tr, &[2, 0]), 5.0);
    assert_eq!(get_f64(&tr, &[2, 1]), 6.0);
}

#[test]
fn test_broadcast_in_dim() {
    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![5.0]));
    let broadcast = broadcast_in_dim(&scalar, &[3], &[]).unwrap();
    assert_eq!(broadcast.shape(), &[3]);
    assert_eq!(get_f64(&broadcast, &[0]), 5.0);
    assert_eq!(get_f64(&broadcast, &[1]), 5.0);
    assert_eq!(get_f64(&broadcast, &[2]), 5.0);

    let v = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let m = broadcast_in_dim(&v, &[3, 2], &[0]).unwrap();
    assert_eq!(m.shape(), &[3, 2]);
    for j in 0..2 {
        assert_eq!(get_f64(&m, &[0, j]), 1.0);
        assert_eq!(get_f64(&m, &[1, j]), 2.0);
        assert_eq!(get_f64(&m, &[2, j]), 3.0);
    }
}

#[test]
fn test_tril_3x3() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let lower = tril(&t, 0).unwrap();
    assert_eq!(lower.shape(), &[3, 3]);
    assert_eq!(
        match &lower {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
    );
}

#[test]
fn test_triu_3x3() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let upper = triu(&t, 0).unwrap();
    assert_eq!(upper.shape(), &[3, 3]);
    assert_eq!(
        match &upper {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_tril_triu_zero_sized_batch_return_empty_tensor() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2, 0], Vec::new()));

    let lower = tril(&t, 0).unwrap();
    assert_eq!(lower.shape(), &[2, 2, 0]);
    match lower {
        Tensor::F64(inner) => assert!(inner.host_data().is_empty()),
        _ => panic!("expected f64 tensor"),
    }

    let upper = triu(&t, 0).unwrap();
    assert_eq!(upper.shape(), &[2, 2, 0]);
    match upper {
        Tensor::F64(inner) => assert!(inner.host_data().is_empty()),
        _ => panic!("expected f64 tensor"),
    }
}

#[test]
fn test_tril_triu_extreme_offsets_do_not_overflow() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));

    let lower_min = tril(&t, i64::MIN).unwrap();
    assert_eq!(
        match &lower_min {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[0.0, 0.0, 0.0, 0.0]
    );

    let upper_min = triu(&t, i64::MIN).unwrap();
    assert_eq!(
        match &upper_min {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[1.0, 2.0, 3.0, 4.0]
    );

    let lower_max = tril(&t, i64::MAX).unwrap();
    assert_eq!(
        match &lower_max {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[1.0, 2.0, 3.0, 4.0]
    );

    let upper_max = triu(&t, i64::MAX).unwrap();
    assert_eq!(
        match &upper_max {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn test_neg_and_conj() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, -7.0]));
    let n = neg(&t).unwrap();
    assert_eq!(get_f64(&n, &[0]), -3.0);
    assert_eq!(get_f64(&n, &[1]), 7.0);

    let c = conj(&t).unwrap();
    assert_eq!(get_f64(&c, &[0]), 3.0);
    assert_eq!(get_f64(&c, &[1]), -7.0);
}

#[test]
fn test_cpu_backend_analytic_ops_real() {
    let mut backend = CpuBackend::new();

    let exp_input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![0.0, 1.0]));
    let exp_out = backend.exp(&exp_input).unwrap();
    assert_f64_close(get_f64(&exp_out, &[0]), 1.0);
    assert_f64_close(get_f64(&exp_out, &[1]), std::f64::consts::E);

    let log_input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 4.0]));
    let log_out = backend.log(&log_input).unwrap();
    assert_f64_close(get_f64(&log_out, &[0]), 0.0);
    assert_f64_close(get_f64(&log_out, &[1]), 4.0_f64.ln());

    let trig_input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![0.0, std::f64::consts::FRAC_PI_2],
    ));
    let sin_out = backend.sin(&trig_input).unwrap();
    let cos_out = backend.cos(&trig_input).unwrap();
    assert_f64_close(get_f64(&sin_out, &[0]), 0.0);
    assert_f64_close(get_f64(&sin_out, &[1]), 1.0);
    assert_f64_close(get_f64(&cos_out, &[0]), 1.0);
    assert_f64_close(get_f64(&cos_out, &[1]), 0.0);

    let tanh_input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![0.0, 1.0]));
    let tanh_out = backend.tanh(&tanh_input).unwrap();
    assert_f64_close(get_f64(&tanh_out, &[0]), 0.0);
    assert_f64_close(get_f64(&tanh_out, &[1]), 1.0_f64.tanh());

    let sqrt_input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 4.0]));
    let sqrt_out = backend.sqrt(&sqrt_input).unwrap();
    let rsqrt_out = backend.rsqrt(&sqrt_input).unwrap();
    assert_f64_close(get_f64(&sqrt_out, &[0]), 1.0);
    assert_f64_close(get_f64(&sqrt_out, &[1]), 2.0);
    assert_f64_close(get_f64(&rsqrt_out, &[0]), 1.0);
    assert_f64_close(get_f64(&rsqrt_out, &[1]), 0.5);

    let expm1_out = backend.expm1(&exp_input).unwrap();
    let log1p_out = backend.log1p(&log_input).unwrap();
    assert_f64_close(get_f64(&expm1_out, &[0]), 0.0);
    assert_f64_close(get_f64(&expm1_out, &[1]), 1.0_f64.exp_m1());
    assert_f64_close(get_f64(&log1p_out, &[0]), 2.0_f64.ln());
    assert_f64_close(get_f64(&log1p_out, &[1]), 5.0_f64.ln());

    let pow_base = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0, 9.0]));
    let pow_exp = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 0.5]));
    let pow_out = backend.pow(&pow_base, &pow_exp).unwrap();
    assert_f64_close(get_f64(&pow_out, &[0]), 8.0);
    assert_f64_close(get_f64(&pow_out, &[1]), 3.0);
}

#[test]
fn test_cpu_backend_analytic_ops_complex() {
    let mut backend = CpuBackend::new();

    let exp_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 1.0)],
    ));
    let exp_out = backend.exp(&exp_input).unwrap();
    assert_c64_close(get_c64(&exp_out, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&exp_out, &[1]), Complex64::new(1.0, 1.0).exp());

    let log_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, -0.5)],
    ));
    let log_out = backend.log(&log_input).unwrap();
    assert_c64_close(get_c64(&log_out, &[0]), Complex64::new(1.0, 0.0).ln());
    assert_c64_close(get_c64(&log_out, &[1]), Complex64::new(2.0, -0.5).ln());

    let trig_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, -0.25)],
    ));
    let sin_out = backend.sin(&trig_input).unwrap();
    let cos_out = backend.cos(&trig_input).unwrap();
    let tanh_out = backend.tanh(&trig_input).unwrap();
    assert_c64_close(get_c64(&sin_out, &[0]), Complex64::new(0.0, 0.0).sin());
    assert_c64_close(get_c64(&sin_out, &[1]), Complex64::new(0.5, -0.25).sin());
    assert_c64_close(get_c64(&cos_out, &[0]), Complex64::new(0.0, 0.0).cos());
    assert_c64_close(get_c64(&cos_out, &[1]), Complex64::new(0.5, -0.25).cos());
    assert_c64_close(get_c64(&tanh_out, &[0]), Complex64::new(0.0, 0.0).tanh());
    assert_c64_close(get_c64(&tanh_out, &[1]), Complex64::new(0.5, -0.25).tanh());

    let sqrt_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(4.0, 3.0)],
    ));
    let sqrt_out = backend.sqrt(&sqrt_input).unwrap();
    let rsqrt_out = backend.rsqrt(&sqrt_input).unwrap();
    assert_c64_close(get_c64(&sqrt_out, &[0]), Complex64::new(1.0, 0.0).sqrt());
    assert_c64_close(get_c64(&sqrt_out, &[1]), Complex64::new(4.0, 3.0).sqrt());
    assert_c64_close_tol(
        get_c64(&rsqrt_out, &[0]),
        Complex64::new(1.0, 0.0) / Complex64::new(1.0, 0.0).sqrt(),
        1.0e-12,
    );
    assert_c64_close_tol(
        get_c64(&rsqrt_out, &[1]),
        Complex64::new(1.0, 0.0) / Complex64::new(4.0, 3.0).sqrt(),
        1.0e-12,
    );

    let expm1_out = backend.expm1(&exp_input).unwrap();
    let log1p_out = backend.log1p(&log_input).unwrap();
    assert_c64_close(
        get_c64(&expm1_out, &[0]),
        Complex64::new(0.0, 0.0).exp() - Complex64::new(1.0, 0.0),
    );
    assert_c64_close(
        get_c64(&expm1_out, &[1]),
        Complex64::new(1.0, 1.0).exp() - Complex64::new(1.0, 0.0),
    );
    assert_c64_close(
        get_c64(&log1p_out, &[0]),
        (Complex64::new(1.0, 0.0) + Complex64::new(1.0, 0.0)).ln(),
    );
    assert_c64_close(
        get_c64(&log1p_out, &[1]),
        (Complex64::new(2.0, -0.5) + Complex64::new(1.0, 0.0)).ln(),
    );

    let pow_base = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    ));
    let pow_exp = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.5, 0.25)],
    ));
    let pow_out = backend.pow(&pow_base, &pow_exp).unwrap();
    assert_c64_close(
        get_c64(&pow_out, &[0]),
        Complex64::new(1.0, 1.0).powc(Complex64::new(2.0, 0.0)),
    );
    assert_c64_close(
        get_c64(&pow_out, &[1]),
        Complex64::new(2.0, -1.0).powc(Complex64::new(0.5, 0.25)),
    );
}

#[test]
fn test_extract_diagonal() {
    let square = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let d = extract_diagonal(&square, 0, 1).unwrap();
    assert_eq!(d.shape(), &[3]);
    assert_eq!(get_f64(&d, &[0]), 1.0);
    assert_eq!(get_f64(&d, &[1]), 5.0);
    assert_eq!(get_f64(&d, &[2]), 9.0);

    let cube = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3, 3],
        (1..=18).map(|x| x as f64).collect(),
    ));
    let diag = extract_diagonal(&cube, 1, 2).unwrap();
    assert_eq!(diag.shape(), &[2, 3]);
    assert_eq!(get_f64(&diag, &[0, 0]), 1.0);
    assert_eq!(get_f64(&diag, &[1, 1]), 10.0);
    assert_eq!(get_f64(&diag, &[1, 2]), 18.0);
}

#[test]
fn test_embed_diagonal() {
    let v = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let m = embed_diagonal(&v, 0, 1).unwrap();
    assert_eq!(m.shape(), &[3, 3]);
    assert_eq!(get_f64(&m, &[0, 0]), 1.0);
    assert_eq!(get_f64(&m, &[1, 1]), 2.0);
    assert_eq!(get_f64(&m, &[2, 2]), 3.0);
    assert_eq!(get_f64(&m, &[0, 1]), 0.0);
    assert_eq!(get_f64(&m, &[2, 0]), 0.0);
}

#[test]
fn test_cpu_backend_dispatches_tensor_backend_ops() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]));
    let mut backend = CpuBackend::new();
    let out = TensorBackend::add(&mut backend, &a, &b).unwrap();
    assert_eq!(get_f64(&out, &[0]), 4.0);
    assert_eq!(get_f64(&out, &[1]), 6.0);
}

#[test]
fn test_tier2_elementwise_ops_real() {
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![8.0, -2.0, 9.0],
    ));
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![2.0, 5.0, 3.0],
    ));
    let pred = Tensor::Bool(TypedTensor::from_vec_col_major(
        vec![3],
        vec![false, true, true],
    ));
    let on_true = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![10.0, 20.0, 30.0],
    ));
    let on_false = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let lower = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![-1.0, -1.0, 0.0],
    ));
    let upper = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 0.25, 4.0],
    ));
    let mut backend = CpuBackend::new();

    let div = backend.div(&lhs, &rhs).unwrap();
    assert_eq!(get_f64(&div, &[0]), 4.0);
    assert_eq!(get_f64(&div, &[1]), -0.4);
    assert_eq!(get_f64(&div, &[2]), 3.0);

    let abs = backend.abs(&lhs).unwrap();
    assert_eq!(get_f64(&abs, &[0]), 8.0);
    assert_eq!(get_f64(&abs, &[1]), 2.0);
    assert_eq!(get_f64(&abs, &[2]), 9.0);

    let sign = backend.sign(&lhs).unwrap();
    assert_eq!(get_f64(&sign, &[0]), 1.0);
    assert_eq!(get_f64(&sign, &[1]), -1.0);
    assert_eq!(get_f64(&sign, &[2]), 1.0);

    let maximum = backend.maximum(&lhs, &rhs).unwrap();
    assert_eq!(get_f64(&maximum, &[0]), 8.0);
    assert_eq!(get_f64(&maximum, &[1]), 5.0);
    assert_eq!(get_f64(&maximum, &[2]), 9.0);

    let minimum = backend.minimum(&lhs, &rhs).unwrap();
    assert_eq!(get_f64(&minimum, &[0]), 2.0);
    assert_eq!(get_f64(&minimum, &[1]), -2.0);
    assert_eq!(get_f64(&minimum, &[2]), 3.0);

    let eq = backend.compare(&lhs, &rhs, &CompareDir::Eq).unwrap();
    assert!(!get_bool(&eq, &[0]));
    assert!(!get_bool(&eq, &[1]));
    assert!(!get_bool(&eq, &[2]));

    let lt = backend.compare(&lhs, &rhs, &CompareDir::Lt).unwrap();
    assert!(!get_bool(&lt, &[0]));
    assert!(get_bool(&lt, &[1]));
    assert!(!get_bool(&lt, &[2]));

    let le = backend.compare(&lhs, &rhs, &CompareDir::Le).unwrap();
    assert!(!get_bool(&le, &[0]));
    assert!(get_bool(&le, &[1]));
    assert!(!get_bool(&le, &[2]));

    let gt = backend.compare(&lhs, &rhs, &CompareDir::Gt).unwrap();
    assert!(get_bool(&gt, &[0]));
    assert!(!get_bool(&gt, &[1]));
    assert!(get_bool(&gt, &[2]));

    let ge = backend.compare(&lhs, &rhs, &CompareDir::Ge).unwrap();
    assert!(get_bool(&ge, &[0]));
    assert!(!get_bool(&ge, &[1]));
    assert!(get_bool(&ge, &[2]));

    let select = backend.select(&pred, &on_true, &on_false).unwrap();
    assert_eq!(get_f64(&select, &[0]), 1.0);
    assert_eq!(get_f64(&select, &[1]), 20.0);
    assert_eq!(get_f64(&select, &[2]), 30.0);

    let clamp = backend.clamp(&lhs, &lower, &upper).unwrap();
    assert_eq!(get_f64(&clamp, &[0]), 1.0);
    assert_eq!(get_f64(&clamp, &[1]), -1.0);
    assert_eq!(get_f64(&clamp, &[2]), 4.0);
}

#[test]
fn test_tier2_elementwise_ops_complex() {
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
    ));
    let lhs = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
    ));
    let rhs = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 2.0)],
    ));
    let mut backend = CpuBackend::new();

    let abs = backend.abs(&input).unwrap();
    assert_c64_close(get_c64(&abs, &[0]), Complex64::new(5.0, 0.0));
    assert_c64_close(get_c64(&abs, &[1]), Complex64::new(0.0, 0.0));

    let sign = backend.sign(&input).unwrap();
    assert_c64_close(get_c64(&sign, &[0]), Complex64::new(0.6, 0.8));
    assert_c64_close(get_c64(&sign, &[1]), Complex64::new(0.0, 0.0));

    let maximum = backend.maximum(&lhs, &rhs).unwrap();
    assert_c64_close(get_c64(&maximum, &[0]), Complex64::new(3.0, 4.0));
    assert_c64_close(get_c64(&maximum, &[1]), Complex64::new(0.0, 2.0));

    let minimum = backend.minimum(&lhs, &rhs).unwrap();
    assert_c64_close(get_c64(&minimum, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&minimum, &[1]), Complex64::new(1.0, 0.0));
}

#[test]
fn test_direct_elementwise_helpers_cover_f32_c32_and_error_paths() {
    let lhs_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![8.0f32, -2.0]));
    let rhs_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![2.0f32, 5.0]));
    let pred_bool = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, true]));
    let lower_f32 = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![-1.0f32, -1.0],
    ));
    let upper_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 4.0]));

    let div_out = div(&lhs_f32, &rhs_f32).unwrap();
    assert_eq!(get_f32(&div_out, &[0]), 4.0);
    assert_eq!(get_f32(&div_out, &[1]), -0.4);

    let abs_out = abs(&lhs_f32).unwrap();
    assert_eq!(get_f32(&abs_out, &[0]), 8.0);
    assert_eq!(get_f32(&abs_out, &[1]), 2.0);

    let sign_out = sign(&lhs_f32).unwrap();
    assert_eq!(get_f32(&sign_out, &[0]), 1.0);
    assert_eq!(get_f32(&sign_out, &[1]), -1.0);

    let max_out = maximum(&lhs_f32, &rhs_f32).unwrap();
    assert_eq!(get_f32(&max_out, &[0]), 8.0);
    assert_eq!(get_f32(&max_out, &[1]), 5.0);

    let min_out = minimum(&lhs_f32, &rhs_f32).unwrap();
    assert_eq!(get_f32(&min_out, &[0]), 2.0);
    assert_eq!(get_f32(&min_out, &[1]), -2.0);

    let cmp_out = compare(&lhs_f32, &rhs_f32, &CompareDir::Gt).unwrap();
    assert!(get_bool(&cmp_out, &[0]));
    assert!(!get_bool(&cmp_out, &[1]));

    let select_out = select(&pred_bool, &lhs_f32, &rhs_f32).unwrap();
    assert_eq!(get_f32(&select_out, &[0]), 2.0);
    assert_eq!(get_f32(&select_out, &[1]), -2.0);

    let clamp_out = clamp(&lhs_f32, &lower_f32, &upper_f32).unwrap();
    assert_eq!(get_f32(&clamp_out, &[0]), 1.0);
    assert_eq!(get_f32(&clamp_out, &[1]), -1.0);

    let input_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(3.0, 4.0), Complex32::new(0.0, 0.0)],
    ));
    let lhs_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(3.0, 4.0), Complex32::new(1.0, 0.0)],
    ));
    let rhs_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 2.0)],
    ));
    let lower_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(0.5, 0.0), Complex32::new(0.5, 0.0)],
    ));
    let upper_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(4.0, 0.0), Complex32::new(2.0, 2.0)],
    ));

    let abs_c32 = abs(&input_c32).unwrap();
    assert_eq!(get_c32(&abs_c32, &[0]), Complex32::new(5.0, 0.0));
    assert_eq!(get_c32(&abs_c32, &[1]), Complex32::new(0.0, 0.0));

    let sign_c32 = sign(&input_c32).unwrap();
    assert_eq!(get_c32(&sign_c32, &[1]), Complex32::new(0.0, 0.0));

    let max_c32 = maximum(&lhs_c32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&max_c32, &[0]), Complex32::new(3.0, 4.0));

    let min_c32 = minimum(&lhs_c32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&min_c32, &[1]), Complex32::new(1.0, 0.0));

    let cmp_c32 = compare(&lhs_c32, &rhs_c32, &CompareDir::Eq).unwrap();
    assert!(!get_bool(&cmp_c32, &[0]));

    let select_c32 = select(&pred_bool, &lhs_c32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&select_c32, &[0]), Complex32::new(1.0, 0.0));
    assert_eq!(get_c32(&select_c32, &[1]), Complex32::new(1.0, 0.0));

    let clamp_c32 = clamp(&lhs_c32, &lower_c32, &upper_c32).unwrap();
    assert_eq!(get_c32(&clamp_c32, &[0]), Complex32::new(4.0, 0.0));

    let scalar_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![2.0f32]));
    let add_c32 = add(&scalar_f32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&add_c32, &[0]), Complex32::new(3.0, 0.0));

    let mul_c32 = mul(&scalar_f32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&mul_c32, &[1]), Complex32::new(0.0, 4.0));

    let scalar_div_c32 = div(&scalar_f32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&scalar_div_c32, &[0]), Complex32::new(2.0, 0.0));
    assert_eq!(get_c32(&scalar_div_c32, &[1]), Complex32::new(0.0, -1.0));

    let c32_div_scalar = div(&rhs_c32, &scalar_f32).unwrap();
    assert_eq!(get_c32(&c32_div_scalar, &[0]), Complex32::new(0.5, 0.0));
    assert_eq!(get_c32(&c32_div_scalar, &[1]), Complex32::new(0.0, 1.0));

    assert!(matches!(
        div(
            &lhs_f32,
            &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]))
        ),
        Err(crate::Error::DTypeMismatch { op: "div", .. })
    ));
    assert!(matches!(
        clamp(
            &lhs_f32,
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![1], vec![0.0f32])),
            &upper_f32
        ),
        Err(crate::Error::ShapeMismatch { op: "clamp", .. })
    ));
}

#[test]
fn test_direct_elementwise_helpers_cover_f64_c64_dispatch_and_mismatch_paths() {
    let lhs_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.5f64, -3.0]));
    let rhs_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0f64, 4.0]));
    let scalar_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0f64]));
    let pred_bool = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, true]));
    let lower_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![0.0f64, -2.0]));
    let upper_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0f64, 3.0]));
    let short_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0f64]));
    let lhs_i32 = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1i32, 3]));
    let rhs_i32 = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![2i32, 3]));
    let lhs_i64 = Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![5i64, -1]));
    let rhs_i64 = Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![2i64, -1]));
    let lhs_bool = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]));
    let rhs_bool = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, false]));

    let add_out = add(&lhs_f64, &rhs_f64).unwrap();
    assert_eq!(get_f64(&add_out, &[0]), 3.5);
    assert_eq!(get_f64(&add_out, &[1]), 1.0);

    let mul_out = mul(&lhs_f64, &rhs_f64).unwrap();
    assert_eq!(get_f64(&mul_out, &[0]), 3.0);
    assert_eq!(get_f64(&mul_out, &[1]), -12.0);

    let div_out = div(
        &rhs_f64,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0, 2.0])),
    )
    .unwrap();
    assert_eq!(get_f64(&div_out, &[0]), 1.0);
    assert_eq!(get_f64(&div_out, &[1]), 2.0);

    let neg_out = neg(&lhs_f64).unwrap();
    assert_eq!(get_f64(&neg_out, &[0]), -1.5);
    assert_eq!(get_f64(&neg_out, &[1]), 3.0);

    let conj_out = conj(&lhs_f64).unwrap();
    assert_eq!(get_f64(&conj_out, &[0]), 1.5);
    assert_eq!(get_f64(&conj_out, &[1]), -3.0);

    let compare_out = compare(&lhs_f64, &rhs_f64, &CompareDir::Lt).unwrap();
    assert!(get_bool(&compare_out, &[0]));
    assert!(get_bool(&compare_out, &[1]));

    let select_out = select(&pred_bool, &lhs_f64, &rhs_f64).unwrap();
    assert_eq!(get_f64(&select_out, &[0]), 2.0);
    assert_eq!(get_f64(&select_out, &[1]), -3.0);

    assert!(get_bool(
        &compare(&lhs_i32, &rhs_i32, &CompareDir::Lt).unwrap(),
        &[0]
    ));
    assert!(get_bool(
        &compare(&lhs_i32, &rhs_i32, &CompareDir::Le).unwrap(),
        &[1]
    ));
    assert!(get_bool(
        &compare(&lhs_i64, &rhs_i64, &CompareDir::Gt).unwrap(),
        &[0]
    ));
    assert!(get_bool(
        &compare(&lhs_i64, &rhs_i64, &CompareDir::Ge).unwrap(),
        &[1]
    ));
    assert!(get_bool(
        &compare(&lhs_bool, &rhs_bool, &CompareDir::Eq).unwrap(),
        &[1]
    ));

    let select_i32 = select(&pred_bool, &lhs_i32, &rhs_i32).unwrap();
    assert_eq!(get_i32(&select_i32, &[0]), 2);
    assert_eq!(get_i32(&select_i32, &[1]), 3);
    let select_i64 = select(&pred_bool, &lhs_i64, &rhs_i64).unwrap();
    assert_eq!(get_i64(&select_i64, &[0]), 2);
    assert_eq!(get_i64(&select_i64, &[1]), -1);
    let select_bool = select(&pred_bool, &lhs_bool, &rhs_bool).unwrap();
    assert!(!get_bool(&select_bool, &[0]));
    assert!(!get_bool(&select_bool, &[1]));

    let clamp_out = clamp(&lhs_f64, &lower_f64, &upper_f64).unwrap();
    assert_eq!(get_f64(&clamp_out, &[0]), 1.5);
    assert_eq!(get_f64(&clamp_out, &[1]), -2.0);

    let lhs_c64 = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
    ));
    let rhs_c64 = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 2.0)],
    ));
    let lower_c64 = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)],
    ));
    let upper_c64 = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(4.0, 0.0), Complex64::new(2.0, 2.0)],
    ));

    let add_left_scalar = add(&scalar_f64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&add_left_scalar, &[0]), Complex64::new(3.0, 0.0));
    let add_right_scalar = add(&lhs_c64, &scalar_f64).unwrap();
    assert_c64_close(get_c64(&add_right_scalar, &[1]), Complex64::new(3.0, 0.0));

    let mul_left_scalar = mul(&scalar_f64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&mul_left_scalar, &[1]), Complex64::new(0.0, 4.0));
    let mul_right_scalar = mul(&lhs_c64, &scalar_f64).unwrap();
    assert_c64_close(get_c64(&mul_right_scalar, &[0]), Complex64::new(6.0, 8.0));

    let div_left_scalar = div(&scalar_f64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&div_left_scalar, &[0]), Complex64::new(2.0, 0.0));
    assert_c64_close(get_c64(&div_left_scalar, &[1]), Complex64::new(0.0, -1.0));

    let div_right_scalar = div(&lhs_c64, &scalar_f64).unwrap();
    assert_c64_close(get_c64(&div_right_scalar, &[0]), Complex64::new(1.5, 2.0));
    assert_c64_close(get_c64(&div_right_scalar, &[1]), Complex64::new(0.5, 0.0));

    let div_c64 = div(
        &lhs_c64,
        &Tensor::C64(TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 1.0), Complex64::new(1.0, 0.0)],
        )),
    )
    .unwrap();
    assert_c64_close(get_c64(&div_c64, &[0]), Complex64::new(3.5, 0.5));
    assert_c64_close(get_c64(&div_c64, &[1]), Complex64::new(1.0, 0.0));

    let neg_c64 = neg(&lhs_c64).unwrap();
    assert_c64_close(get_c64(&neg_c64, &[0]), Complex64::new(-3.0, -4.0));
    let conj_c64 = conj(&lhs_c64).unwrap();
    assert_c64_close(get_c64(&conj_c64, &[0]), Complex64::new(3.0, -4.0));

    let compare_lt = compare(&lhs_c64, &rhs_c64, &CompareDir::Lt).unwrap();
    let compare_le = compare(&lhs_c64, &rhs_c64, &CompareDir::Le).unwrap();
    let compare_gt = compare(&lhs_c64, &rhs_c64, &CompareDir::Gt).unwrap();
    let compare_ge = compare(&lhs_c64, &rhs_c64, &CompareDir::Ge).unwrap();
    assert!(!get_bool(&compare_lt, &[0]));
    assert!(!get_bool(&compare_le, &[0]));
    assert!(get_bool(&compare_gt, &[0]));
    assert!(get_bool(&compare_ge, &[0]));

    let select_c64 = select(&pred_bool, &lhs_c64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&select_c64, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&select_c64, &[1]), Complex64::new(1.0, 0.0));

    let clamp_c64 = clamp(&lhs_c64, &lower_c64, &upper_c64).unwrap();
    assert_c64_close(get_c64(&clamp_c64, &[0]), Complex64::new(4.0, 0.0));
    assert_c64_close(get_c64(&clamp_c64, &[1]), Complex64::new(1.0, 0.0));

    let lhs_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]));

    assert!(matches!(
        add(&lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "add", .. })
    ));
    assert!(matches!(
        mul(&lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "mul", .. })
    ));
    assert!(matches!(
        maximum(&lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "maximum", .. })
    ));
    assert!(matches!(
        minimum(&lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "minimum", .. })
    ));
    assert!(matches!(
        compare(&lhs_f32, &rhs_f64, &CompareDir::Eq),
        Err(crate::Error::DTypeMismatch { op: "compare", .. })
    ));
    assert!(matches!(
        select(&lhs_f32, &lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "select", .. })
    ));
    assert!(matches!(
        clamp(&lhs_f32, &lhs_f32, &rhs_f64),
        Err(crate::Error::BackendFailure {
            op: "clamp",
            message,
        }) if message == "dtype mismatch"
    ));

    assert!(matches!(
        add(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "add", .. })
    ));
    assert!(matches!(
        mul(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "mul", .. })
    ));
    assert!(matches!(
        div(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "div", .. })
    ));
    assert!(matches!(
        maximum(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "maximum", .. })
    ));
    assert!(matches!(
        minimum(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "minimum", .. })
    ));
    assert!(matches!(
        compare(&lhs_f64, &short_f64, &CompareDir::Eq),
        Err(crate::Error::ShapeMismatch { op: "compare", .. })
    ));
    assert!(matches!(
        select(&pred_bool, &short_f64, &rhs_f64),
        Err(crate::Error::ShapeMismatch { op: "select", .. })
    ));
    assert!(matches!(
        select(&pred_bool, &rhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "select", .. })
    ));
    assert!(matches!(
        clamp(&lhs_f64, &lower_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "clamp", .. })
    ));
}

#[test]
fn test_reduction_helpers_cover_complex_and_error_paths() {
    let complex = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, -1.0),
            Complex32::new(4.0, 2.0),
        ],
    ));
    let sum = reduce_sum(&complex, &[0]).unwrap();
    assert_eq!(get_c32(&sum, &[0]), Complex32::new(3.0, 1.0));
    assert_eq!(get_c32(&sum, &[1]), Complex32::new(7.0, 1.0));

    let prod = reduce_prod(&complex, &[]).unwrap();
    assert_eq!(prod.shape(), &[2, 2]);

    assert!(matches!(
        reduce_sum(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0])),
            &[2]
        ),
        Err(crate::Error::AxisOutOfBounds {
            op: "reduce_sum",
            ..
        })
    ));
    assert!(matches!(
        reduce_prod(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0])),
            &[0, 0]
        ),
        Err(crate::Error::DuplicateAxis {
            op: "reduce_prod",
            ..
        })
    ));
    assert!(matches!(
        reduce_max(&complex, &[0]),
        Err(crate::Error::BackendFailure {
            op: "reduce_max",
            ..
        })
    ));
    assert!(matches!(
        reduce_min(&complex, &[0]),
        Err(crate::Error::BackendFailure {
            op: "reduce_min",
            ..
        })
    ));
}

#[test]
fn test_structural_helpers_cover_f32_success_and_error_paths() {
    let matrix = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0f32, 2.0, 3.0, 4.0],
    ));
    let transposed = transpose(&matrix, &[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(get_f32(&transposed, &[1, 0]), 3.0);

    let scalar = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![5.0f32]));
    let broadcast = broadcast_in_dim(&scalar, &[2], &[]).unwrap();
    assert_eq!(get_f32(&broadcast, &[1]), 5.0);

    let diag = extract_diagonal(&matrix, 0, 1).unwrap();
    assert_eq!(get_f32(&diag, &[0]), 1.0);
    assert_eq!(get_f32(&diag, &[1]), 4.0);

    let embedded = embed_diagonal(
        &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![7.0f32, 8.0])),
        0,
        1,
    )
    .unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(get_f32(&embedded, &[1, 1]), 8.0);

    let lower = tril(&matrix, 0).unwrap();
    assert_eq!(get_f32(&lower, &[0, 1]), 0.0);
    let upper = triu(&matrix, 0).unwrap();
    assert_eq!(get_f32(&upper, &[1, 0]), 0.0);

    assert!(matches!(
        transpose(&matrix, &[0]),
        Err(crate::Error::RankMismatch {
            op: "transpose",
            ..
        })
    ));
    assert!(matches!(
        transpose(&matrix, &[0, 0]),
        Err(crate::Error::DuplicateAxis {
            op: "transpose",
            ..
        })
    ));
    assert!(matches!(
        broadcast_in_dim(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0])),
            &[3, 2],
            &[0]
        ),
        Err(crate::Error::ShapeMismatch {
            op: "broadcast_in_dim",
            ..
        })
    ));
    assert!(matches!(
        extract_diagonal(&matrix, 1, 1),
        Err(crate::Error::DuplicateAxis {
            op: "extract_diagonal",
            ..
        })
    ));
    assert!(matches!(
        embed_diagonal(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0])),
            0,
            2
        ),
        Err(crate::Error::AxisOutOfBounds {
            op: "embed_diagonal",
            ..
        })
    ));
    let vector = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]));
    assert!(matches!(
        tril(&vector, 0),
        Err(crate::Error::RankMismatch { op: "tril", .. })
    ));
    assert!(matches!(
        triu(&vector, 0),
        Err(crate::Error::RankMismatch { op: "triu", .. })
    ));
}

#[test]
fn test_batched_cholesky() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.cholesky(&input).unwrap();

    assert_eq!(out.shape(), &[3, 3, 2]);
    for batch_idx in 0..2 {
        let l = batch_matrix_f64_from_tensor(&out, 3, 3, batch_idx);
        let recon = matmul_f64(&l, &transpose_f64(&l, 3, 3), 3, 3, 3);
        let expected = batch_matrix_f64_from_tensor(&input, 3, 3, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_batched_svd() {
    let a0 = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 1.5, 2.0, 0.0, 1.0, -0.5];
    let a1 = vec![
        2.0, -1.0, 0.5, 3.0, -0.25, 1.5, -2.0, 0.75, 1.0, 2.5, -1.0, 4.0,
    ];
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.svd(&input).unwrap();

    assert_eq!(out.len(), 3);
    assert_eq!(out[0].shape(), &[4, 3, 2]);
    assert_eq!(out[1].shape(), &[3, 2]);
    assert_eq!(out[2].shape(), &[3, 3, 2]);

    for batch_idx in 0..2 {
        let u = batch_matrix_f64_from_tensor(&out[0], 4, 3, batch_idx);
        let s = batch_vector_f64_from_tensor(&out[1], 3, batch_idx);
        let vt = batch_matrix_f64_from_tensor(&out[2], 3, 3, batch_idx);
        let recon = matmul_f64(&matmul_f64(&u, &diag_f64(&s), 4, 3, 3), &vt, 4, 3, 3);
        let expected = batch_matrix_f64_from_tensor(&input, 4, 3, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-9);
        }
    }
}

#[test]
fn test_batched_qr() {
    let a0 = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0];
    let a1 = vec![2.0, -1.0, 0.5, 3.0, -0.25, 1.5];
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 2, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.qr(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[3, 2, 2]);
    assert_eq!(out[1].shape(), &[2, 2, 2]);

    for batch_idx in 0..2 {
        let q = batch_matrix_f64_from_tensor(&out[0], 3, 2, batch_idx);
        let r = batch_matrix_f64_from_tensor(&out[1], 2, 2, batch_idx);
        let recon = matmul_f64(&q, &r, 3, 2, 2);
        let expected = batch_matrix_f64_from_tensor(&input, 3, 2, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-9);
        }
    }
}

#[test]
fn test_batched_solve() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 1, 2],
        vec![1.0, 2.0, 3.0, -1.0, 4.0, 0.5],
    ));

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();

    assert_eq!(x.shape(), &[3, 1, 2]);
    for batch_idx in 0..2 {
        let a_batch = batch_matrix_f64_from_tensor(&a, 3, 3, batch_idx);
        let x_batch = batch_matrix_f64_from_tensor(&x, 3, 1, batch_idx);
        let recon = matmul_f64(&a_batch, &x_batch, 3, 3, 1);
        let expected = batch_matrix_f64_from_tensor(&b, 3, 1, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_triangular_solve_lower() {
    let l_data = vec![2.0, 1.0, -0.5, 0.0, 3.0, 1.25, 0.0, 0.0, 1.5];
    let b_data = vec![1.0, -2.0, 0.5];
    let l = Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 3], l_data.clone()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 1], b_data.clone()));

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&l, &b, true, true, false, false)
        .unwrap();

    assert_eq!(x.shape(), &[3, 1]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&l_data, x_data, 3, 3, 1);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_right_side_unit_transpose() {
    let a_data = vec![1.0, 2.0, 0.0, 1.0];
    let b_data = vec![7.0, 5.0];
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 2], b_data.clone()));

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&a, &b, false, true, true, true)
        .unwrap();

    assert_eq!(x.shape(), &[1, 2]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&x_data, &transpose_f64(&a_data, 2, 2), 1, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_covers_all_real_branch_combinations() {
    let expected_x = vec![1.0, -2.0, 0.5, 3.0];

    for &left_side in &[true, false] {
        for &lower in &[true, false] {
            for &transpose_a in &[false, true] {
                for &unit_diagonal in &[false, true] {
                    let diagonal = if unit_diagonal {
                        (1.0, 1.0)
                    } else {
                        (2.0, 1.5)
                    };
                    let a_data = if lower {
                        vec![diagonal.0, -0.75, 0.0, diagonal.1]
                    } else {
                        vec![diagonal.0, 0.0, 0.5, diagonal.1]
                    };
                    let op_a = if transpose_a {
                        transpose_f64(&a_data, 2, 2)
                    } else {
                        a_data.clone()
                    };
                    let b_data = if left_side {
                        matmul_f64(&op_a, &expected_x, 2, 2, 2)
                    } else {
                        matmul_f64(&expected_x, &op_a, 2, 2, 2)
                    };

                    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data));
                    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], b_data));
                    let mut backend = CpuBackend::new();
                    let x = backend
                        .triangular_solve(&a, &b, left_side, lower, transpose_a, unit_diagonal)
                        .unwrap();

                    let x_data = match &x {
                        Tensor::F64(inner) => inner.host_data(),
                        _ => panic!("expected f64 tensor"),
                    };
                    for (actual, expected) in x_data.iter().zip(expected_x.iter()) {
                        assert_f64_close_tol(*actual, *expected, 1.0e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn test_batched_complex_solve() {
    let l0 = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let l1 = vec![
        Complex64::new(1.25, 0.0),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let a0 = matmul_c64(&l0, &conjugate_transpose_c64(&l0, 2, 2), 2, 2, 2);
    let a1 = matmul_c64(&l1, &conjugate_transpose_c64(&l1, 2, 2), 2, 2, 2);
    let a = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let b = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 1, 2],
        vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(0.5, 2.0),
            Complex64::new(-2.0, 0.25),
            Complex64::new(1.5, -0.75),
        ],
    ));

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();

    assert_eq!(x.shape(), &[2, 1, 2]);
    for batch_idx in 0..2 {
        let a_batch = batch_matrix_c64_from_tensor(&a, 2, 2, batch_idx);
        let x_batch = batch_matrix_c64_from_tensor(&x, 2, 1, batch_idx);
        let recon = matmul_c64(&a_batch, &x_batch, 2, 2, 1);
        let expected = batch_matrix_c64_from_tensor(&b, 2, 1, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_real_solve_non_batched() {
    let a_data = vec![3.0, 1.0, 1.0, 2.0];
    let b_data = vec![5.0, 1.0, -2.0, 4.0];
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], b_data.clone()));

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();

    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&a_data, x_data, 2, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_real_lu_returns_permutation_factors_and_parity() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![0.0, 1.0, 1.0, 0.0],
    ));
    let mut backend = CpuBackend::new();
    let outputs = backend.lu(&input).unwrap();

    assert_eq!(outputs.len(), 4);
    let p = matrix_f64_from_tensor(&outputs[0], 2, 2);
    let l = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let u = matrix_f64_from_tensor(&outputs[2], 2, 2);
    let parity = get_f64(&outputs[3], &[]);

    let pa = matmul_f64(&p, &matrix_f64_from_tensor(&input, 2, 2), 2, 2, 2);
    let lu = matmul_f64(&l, &u, 2, 2, 2);
    for (actual, expected) in pa.iter().zip(lu.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
    assert_f64_close(parity, -1.0);
}

#[test]
fn test_real_eig_returns_complex_outputs() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 0.0, 0.0, 3.0],
    ));
    let mut backend = CpuBackend::new();
    let outputs = backend.eig(&input).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let mut values = vector_c64_from_tensor(&outputs[0], 2);
    values.sort_by(|lhs, rhs| {
        lhs.re
            .partial_cmp(&rhs.re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assert_c64_close(values[0], Complex64::new(1.0, 0.0));
    assert_c64_close(values[1], Complex64::new(3.0, 0.0));
}

#[test]
fn test_batched_complex_eigh() {
    let l0 = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let l1 = vec![
        Complex64::new(1.25, 0.0),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let a0 = matmul_c64(&l0, &conjugate_transpose_c64(&l0, 2, 2), 2, 2, 2);
    let a1 = matmul_c64(&l1, &conjugate_transpose_c64(&l1, 2, 2), 2, 2, 2);
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));

    let mut backend = CpuBackend::new();
    let out = backend.eigh(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_eq!(out[1].shape(), &[2, 2, 2]);

    for batch_idx in 0..2 {
        let values = batch_vector_c64_from_tensor(&out[0], 2, batch_idx);
        let vectors = batch_matrix_c64_from_tensor(&out[1], 2, 2, batch_idx);
        let recon = matmul_c64(
            &matmul_c64(&vectors, &diag_c64(&values), 2, 2, 2),
            &conjugate_transpose_c64(&vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = batch_matrix_c64_from_tensor(&input, 2, 2, batch_idx);
        for value in &values {
            assert_f64_close_tol(value.im, 0.0, 1.0e-12);
        }
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_real_eigh() {
    let a_data = vec![4.0, 1.0, 1.0, 3.0];
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()));

    let mut backend = CpuBackend::new();
    let out = backend.eigh(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let values = match &out[0] {
        Tensor::F64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected f64 tensor"),
    };
    let vectors = matrix_f64_from_tensor(&out[1], 2, 2);
    let recon = matmul_f64(
        &matmul_f64(&vectors, &diag_f64(&values), 2, 2, 2),
        &transpose_f64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    for (actual, expected) in recon.iter().zip(a_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_real_cholesky_returns_error_for_non_positive_definite_input() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 2.0, 1.0],
    ));
    let mut backend = CpuBackend::new();
    let err = backend.cholesky(&input).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "cholesky", .. }
    ));
}

#[test]
fn test_real_solve_returns_error_for_singular_matrix() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 2.0, 4.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 1.0]));
    let mut backend = CpuBackend::new();
    let err = backend.solve(&a, &b).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "solve", .. }
    ));
}

#[test]
fn test_complex_cholesky() {
    let l = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let a = matmul_c64(&l, &conjugate_transpose_c64(&l, 2, 2), 2, 2, 2);
    let input = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a.clone()));

    let mut backend = CpuBackend::new();
    let out = backend.cholesky(&input).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    let l_out = matrix_c64_from_tensor(&out, 2, 2);
    let recon = matmul_c64(&l_out, &conjugate_transpose_c64(&l_out, 2, 2), 2, 2, 2);
    for (actual, expected) in recon.iter().zip(a.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_complex_cholesky_returns_error_for_non_positive_definite_input() {
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    ));
    let mut backend = CpuBackend::new();
    let err = backend.cholesky(&input).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "cholesky", .. }
    ));
}

#[test]
fn test_complex_qr() {
    let input_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -0.5),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(-0.25, 1.5),
        Complex64::new(3.0, 0.75),
    ];
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![3, 2],
        input_data.clone(),
    ));

    let mut backend = CpuBackend::new();
    let out = backend.qr(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let q = matrix_c64_from_tensor(&out[0], 3, 2);
    let r = matrix_c64_from_tensor(&out[1], 2, 2);
    let recon = matmul_c64(&q, &r, 3, 2, 2);
    for (actual, expected) in recon.iter().zip(input_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn test_complex_svd() {
    let input_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -0.5),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(-0.25, 1.5),
        Complex64::new(3.0, 0.75),
    ];
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![3, 2],
        input_data.clone(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.svd(&input).unwrap();

    assert_eq!(out.len(), 3);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2]);
    assert_eq!(out[2].shape(), &[2, 2]);

    let u = matrix_c64_from_tensor(&out[0], 3, 2);
    let s = vector_c64_from_tensor(&out[1], 2);
    let vt = matrix_c64_from_tensor(&out[2], 2, 2);
    let recon = matmul_c64(&matmul_c64(&u, &diag_c64(&s), 3, 2, 2), &vt, 3, 2, 2);
    for (actual, expected) in recon.iter().zip(input_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn test_complex_triangular_solve_right_side_unit_transpose() {
    let a_data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let b_data = vec![Complex64::new(2.0, 1.0), Complex64::new(-1.0, 0.5)];
    let a = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()));
    let b = Tensor::C64(TypedTensor::from_vec_col_major(vec![1, 2], b_data.clone()));

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&a, &b, false, true, true, true)
        .unwrap();

    assert_eq!(x.shape(), &[1, 2]);
    let x_data = match &x {
        Tensor::C64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected c64 tensor"),
    };
    let recon = matmul_c64(&x_data, &transpose_c64(&a_data, 2, 2), 1, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_covers_all_complex_branch_combinations() {
    let expected_x = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, -0.5),
        Complex64::new(3.0, -1.0),
    ];

    for &left_side in &[true, false] {
        for &lower in &[true, false] {
            for &transpose_a in &[false, true] {
                for &unit_diagonal in &[false, true] {
                    let diagonal = if unit_diagonal {
                        (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0))
                    } else {
                        (Complex64::new(2.0, 0.0), Complex64::new(1.5, 0.0))
                    };
                    let a_data = if lower {
                        vec![
                            diagonal.0,
                            Complex64::new(-0.75, 0.25),
                            Complex64::new(0.0, 0.0),
                            diagonal.1,
                        ]
                    } else {
                        vec![
                            diagonal.0,
                            Complex64::new(0.0, 0.0),
                            Complex64::new(0.5, -0.25),
                            diagonal.1,
                        ]
                    };
                    let op_a = if transpose_a {
                        transpose_c64(&a_data, 2, 2)
                    } else {
                        a_data.clone()
                    };
                    let b_data = if left_side {
                        matmul_c64(&op_a, &expected_x, 2, 2, 2)
                    } else {
                        matmul_c64(&expected_x, &op_a, 2, 2, 2)
                    };

                    let a = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a_data));
                    let b = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], b_data));
                    let mut backend = CpuBackend::new();
                    let x = backend
                        .triangular_solve(&a, &b, left_side, lower, transpose_a, unit_diagonal)
                        .unwrap();

                    let x_data = match &x {
                        Tensor::C64(inner) => inner.host_data(),
                        _ => panic!("expected c64 tensor"),
                    };
                    for (actual, expected) in x_data.iter().zip(expected_x.iter()) {
                        assert_c64_close_tol(*actual, *expected, 1.0e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn test_complex_solve_returns_error_for_singular_matrix() {
    let a = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    ));
    let b = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 1],
        vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    ));
    let mut backend = CpuBackend::new();
    let err = backend.solve(&a, &b).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "solve", .. }
    ));
}

#[test]
fn test_gather_1d_indices() {
    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 2, 4]);

    let out = gather(&operand, &start_indices, &simple_gather_config()).unwrap();

    assert_eq!(out.shape(), &[3]);
    assert_eq!(get_f64(&out, &[0]), 10.0);
    assert_eq!(get_f64(&out, &[1]), 30.0);
    assert_eq!(get_f64(&out, &[2]), 50.0);
}

#[test]
fn test_gather_accepts_i64_indices() {
    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 2, 4]);

    let out = gather(&operand, &start_indices, &simple_gather_config()).unwrap();

    assert_eq!(start_indices.dtype(), DType::I64);
    assert_eq!(start_indices.as_slice::<i64>(), Some([0, 2, 4].as_slice()));
    assert_eq!(out.shape(), &[3]);
    assert_eq!(get_f64(&out, &[0]), 10.0);
    assert_eq!(get_f64(&out, &[1]), 30.0);
    assert_eq!(get_f64(&out, &[2]), 50.0);
}

#[test]
fn test_gather_with_implicit_index_vector_dim() {
    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::from_vec_col_major(vec![3], vec![4_i64, 1, 0]);
    let config = GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };

    let out = gather(&operand, &start_indices, &config).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(get_f64(&out, &[0]), 50.0);
    assert_eq!(get_f64(&out, &[1]), 20.0);
    assert_eq!(get_f64(&out, &[2]), 10.0);
}

#[test]
fn test_scatter_accepts_i64_indices() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3]));
    let scatter_indices = Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 1, 2, 0, 1, 2]);
    let updates = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![5.0, 6.0, 7.0],
    ));

    let out = scatter(
        &operand,
        &scatter_indices,
        &updates,
        &diagonal_scatter_config(),
    )
    .unwrap();

    assert_eq!(scatter_indices.dtype(), DType::I64);
    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 5.0);
    assert_eq!(get_f64(&out, &[1, 1]), 6.0);
    assert_eq!(get_f64(&out, &[2, 2]), 7.0);
}

#[test]
fn test_scatter_to_diagonal() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3]));
    let scatter_indices = Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 1, 2, 0, 1, 2]);
    let updates = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![5.0, 6.0, 7.0],
    ));

    let out = scatter(
        &operand,
        &scatter_indices,
        &updates,
        &diagonal_scatter_config(),
    )
    .unwrap();

    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 5.0);
    assert_eq!(get_f64(&out, &[1, 1]), 6.0);
    assert_eq!(get_f64(&out, &[2, 2]), 7.0);
    assert_eq!(get_f64(&out, &[1, 0]), 0.0);
    assert_eq!(get_f64(&out, &[0, 2]), 0.0);
}

#[test]
fn test_scatter_skips_negative_and_out_of_bounds_windows() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![4]));
    let scatter_indices = Tensor::from_vec_col_major(vec![3, 1], vec![-1_i64, 2, 4]);
    let updates = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![5.0, 6.0, 7.0],
    ));
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let out = scatter(&operand, &scatter_indices, &updates, &config).unwrap();
    assert_eq!(out.shape(), &[4]);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 0.0);
    assert_eq!(get_f64(&out, &[2]), 6.0);
    assert_eq!(get_f64(&out, &[3]), 0.0);
}

#[test]
fn test_pad_adds_zero_edges() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let config = PadConfig {
        edge_padding_low: vec![1, 1],
        edge_padding_high: vec![1, 1],
        interior_padding: vec![0, 0],
    };

    let out = pad(&input, &config).unwrap();

    assert_eq!(out.shape(), &[4, 5]);
    assert_eq!(get_f64(&out, &[1, 1]), 1.0);
    assert_eq!(get_f64(&out, &[2, 1]), 2.0);
    assert_eq!(get_f64(&out, &[1, 2]), 3.0);
    assert_eq!(get_f64(&out, &[2, 3]), 6.0);
    assert_eq!(get_f64(&out, &[0, 0]), 0.0);
    assert_eq!(get_f64(&out, &[3, 4]), 0.0);
}

#[test]
fn test_pad_with_interior_spacing() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    let config = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![1],
    };

    let out = pad(&input, &config).unwrap();
    assert_eq!(out.shape(), &[5]);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 1.0);
    assert_eq!(get_f64(&out, &[2]), 0.0);
    assert_eq!(get_f64(&out, &[3]), 2.0);
    assert_eq!(get_f64(&out, &[4]), 0.0);
}

#[test]
fn test_dynamic_slice_clamps_starts() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    ));
    let starts = Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]);

    let out = dynamic_slice(&input, &starts, &[2, 2]).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64(&out, &[0, 0]), 11.0);
    assert_eq!(get_f64(&out, &[1, 0]), 12.0);
    assert_eq!(get_f64(&out, &[0, 1]), 15.0);
    assert_eq!(get_f64(&out, &[1, 1]), 16.0);
}

#[test]
fn test_dynamic_slice_accepts_i64_starts() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    ));
    let starts = Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]);

    let out = dynamic_slice(&input, &starts, &[2, 2]).unwrap();

    assert_eq!(starts.dtype(), DType::I64);
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64(&out, &[0, 0]), 11.0);
    assert_eq!(get_f64(&out, &[1, 0]), 12.0);
    assert_eq!(get_f64(&out, &[0, 1]), 15.0);
    assert_eq!(get_f64(&out, &[1, 1]), 16.0);
}

#[test]
fn test_dynamic_update_slice_clamps_starts() {
    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![5],
        vec![10.0, 11.0, 12.0, 13.0, 14.0],
    ));
    let update = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let starts = Tensor::from_vec_col_major(vec![1], vec![4_i64]);

    let out = dynamic_update_slice(&operand, &update, &starts).unwrap();

    assert_eq!(out.shape(), &[5]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[10.0, 11.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_slice_concatenate_and_reverse_edge_cases() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));
    let config = SliceConfig {
        starts: vec![0, 0],
        limits: vec![4, 3],
        strides: vec![2, 2],
    };
    let mut backend = CpuBackend::new();
    let sliced = backend.slice(&input, &config).unwrap();
    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(get_f64(&sliced, &[0, 0]), 1.0);
    assert_eq!(get_f64(&sliced, &[1, 0]), 3.0);
    assert_eq!(get_f64(&sliced, &[0, 1]), 9.0);
    assert_eq!(get_f64(&sliced, &[1, 1]), 11.0);

    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![3.0, 4.0]));
    let c = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![5.0, 6.0]));
    let concatenated = backend.concatenate(&[&a, &b, &c], 1).unwrap();
    assert_eq!(concatenated.shape(), &[2, 3]);
    assert_eq!(get_f64(&concatenated, &[0, 0]), 1.0);
    assert_eq!(get_f64(&concatenated, &[1, 1]), 4.0);
    assert_eq!(get_f64(&concatenated, &[0, 2]), 5.0);

    let reversed = backend.reverse(&input, &[0, 1]).unwrap();
    assert_eq!(reversed.shape(), &[4, 3]);
    assert_eq!(get_f64(&reversed, &[0, 0]), 12.0);
    assert_eq!(get_f64(&reversed, &[3, 2]), 1.0);
}

#[test]
fn test_structural_convert_helper_returns_result() {
    let input = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![1.25_f32, -2.5_f32],
    ));

    let output = crate::cpu::structural::convert(&input, DType::F64).unwrap();

    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.dtype(), DType::F64);
    assert_eq!(get_f64(&output, &[0]), 1.25);
    assert_eq!(get_f64(&output, &[1]), -2.5);
}

#[test]
fn test_backend_convert_supports_real_complex_and_precision_changes() {
    let mut backend = CpuBackend::new();
    let f32_input = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![1.25_f32, -2.5_f32],
    ));
    let f64_input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![1.25_f64, -2.5_f64],
    ));
    let i64_input = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![1_i64, -2_i64],
    ));
    let c32_input = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.25, -0.5), Complex32::new(-2.5, 4.0)],
    ));
    let c64_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.25, -0.5), Complex64::new(-2.5, 4.0)],
    ));

    let cases = [
        (&f32_input, DType::F32),
        (&f32_input, DType::F64),
        (&f32_input, DType::I64),
        (&f32_input, DType::C32),
        (&f32_input, DType::C64),
        (&f64_input, DType::F32),
        (&f64_input, DType::F64),
        (&f64_input, DType::I64),
        (&f64_input, DType::C32),
        (&f64_input, DType::C64),
        (&i64_input, DType::F32),
        (&i64_input, DType::F64),
        (&i64_input, DType::I64),
        (&i64_input, DType::C32),
        (&i64_input, DType::C64),
        (&c32_input, DType::F32),
        (&c32_input, DType::F64),
        (&c32_input, DType::I64),
        (&c32_input, DType::C32),
        (&c32_input, DType::C64),
        (&c64_input, DType::F32),
        (&c64_input, DType::F64),
        (&c64_input, DType::I64),
        (&c64_input, DType::C32),
        (&c64_input, DType::C64),
    ];

    for (input, to) in cases {
        let output = backend.convert(input, to).unwrap();
        assert_eq!(output.shape(), &[2]);
        assert_eq!(output.dtype(), to);

        match (input.dtype(), &output) {
            (DType::F32, Tensor::F32(inner)) => assert_eq!(inner.host_data(), &[1.25, -2.5]),
            (DType::F32, Tensor::F64(inner)) => assert_eq!(inner.host_data(), &[1.25, -2.5]),
            (DType::F32, Tensor::I64(inner)) => assert_eq!(inner.host_data(), &[1, -2]),
            (DType::F32, Tensor::C32(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex32::new(1.25, 0.0), Complex32::new(-2.5, 0.0)]
            ),
            (DType::F32, Tensor::C64(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex64::new(1.25, 0.0), Complex64::new(-2.5, 0.0)]
            ),
            (DType::F64, Tensor::F32(inner)) => assert_eq!(inner.host_data(), &[1.25, -2.5]),
            (DType::F64, Tensor::F64(inner)) => assert_eq!(inner.host_data(), &[1.25, -2.5]),
            (DType::F64, Tensor::I64(inner)) => assert_eq!(inner.host_data(), &[1, -2]),
            (DType::F64, Tensor::C32(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex32::new(1.25, 0.0), Complex32::new(-2.5, 0.0)]
            ),
            (DType::F64, Tensor::C64(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex64::new(1.25, 0.0), Complex64::new(-2.5, 0.0)]
            ),
            (DType::I64, Tensor::F32(inner)) => assert_eq!(inner.host_data(), &[1.0, -2.0]),
            (DType::I64, Tensor::F64(inner)) => assert_eq!(inner.host_data(), &[1.0, -2.0]),
            (DType::I64, Tensor::I64(inner)) => assert_eq!(inner.host_data(), &[1, -2]),
            (DType::I64, Tensor::C32(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex32::new(1.0, 0.0), Complex32::new(-2.0, 0.0)]
            ),
            (DType::I64, Tensor::C64(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)]
            ),
            (DType::C32, Tensor::F32(inner)) => assert_eq!(inner.host_data(), &[1.25, -2.5]),
            (DType::C32, Tensor::F64(inner)) => assert_eq!(inner.host_data(), &[1.25, -2.5]),
            (DType::C32, Tensor::I64(inner)) => assert_eq!(inner.host_data(), &[1, -2]),
            (DType::C32, Tensor::C32(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex32::new(1.25, -0.5), Complex32::new(-2.5, 4.0)]
            ),
            (DType::C32, Tensor::C64(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex64::new(1.25, -0.5), Complex64::new(-2.5, 4.0)]
            ),
            (DType::C64, Tensor::F32(inner)) => assert_eq!(inner.host_data(), &[1.25, -2.5]),
            (DType::C64, Tensor::F64(inner)) => assert_eq!(inner.host_data(), &[1.25, -2.5]),
            (DType::C64, Tensor::I64(inner)) => assert_eq!(inner.host_data(), &[1, -2]),
            (DType::C64, Tensor::C32(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex32::new(1.25, -0.5), Complex32::new(-2.5, 4.0)]
            ),
            (DType::C64, Tensor::C64(inner)) => assert_eq!(
                inner.host_data(),
                &[Complex64::new(1.25, -0.5), Complex64::new(-2.5, 4.0)]
            ),
            _ => unreachable!("unexpected conversion case"),
        }
    }
}

#[test]
fn test_cpu_supports_i32_and_bool_structural_paths() {
    let mut backend = CpuBackend::new();

    let i32_tensor = Tensor::from_vec_col_major(vec![2], vec![-1_i32, 0]);
    let i32_as_bool = backend.convert(&i32_tensor, DType::Bool).unwrap();
    assert_eq!(i32_as_bool.as_slice::<bool>().unwrap(), &[true, false]);

    let bool_tensor = Tensor::from_vec_col_major(vec![2], vec![true, false]);
    let bool_as_i64 = backend.convert(&bool_tensor, DType::I64).unwrap();
    assert_eq!(bool_as_i64.as_slice::<i64>().unwrap(), &[1, 0]);

    let bool_matrix =
        Tensor::from_vec_col_major(vec![2, 3], vec![true, false, false, true, true, false]);
    let transposed = transpose(&bool_matrix, &[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(
        transposed.as_slice::<bool>().unwrap(),
        &[true, false, true, false, true, false]
    );

    let padded = pad(
        &bool_tensor,
        &PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        },
    )
    .unwrap();
    assert_eq!(
        padded.as_slice::<bool>().unwrap(),
        &[false, true, false, false]
    );

    let starts = Tensor::from_vec_col_major(vec![1], vec![1_i32]);
    let sliced = dynamic_slice(
        &Tensor::from_vec_col_major(vec![3], vec![true, false, true]),
        &starts,
        &[2],
    )
    .unwrap();
    assert_eq!(sliced.as_slice::<bool>().unwrap(), &[false, true]);

    let upper = triu(
        &Tensor::from_vec_col_major(vec![2, 2], vec![true, true, false, true]),
        0,
    )
    .unwrap();
    assert_eq!(
        upper.as_slice::<bool>().unwrap(),
        &[true, false, false, true]
    );

    let i32_sum = reduce_sum(
        &Tensor::from_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4]),
        &[0],
    )
    .unwrap();
    assert_eq!(i32_sum.as_slice::<i32>().unwrap(), &[3, 7]);
}

#[test]
fn test_backend_linalg_returns_errors_for_unsupported_dtypes() {
    let mut backend = CpuBackend::new();
    let i64_matrix = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1_i64, 0, 0, 1],
    ));
    let i64_rhs = Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1_i64, 2]));

    assert!(backend.cholesky(&i64_matrix).is_err());
    assert!(backend.svd(&i64_matrix).is_err());
    assert!(backend.qr(&i64_matrix).is_err());
    assert!(backend.eigh(&i64_matrix).is_err());
    assert!(backend.eig(&i64_matrix).is_err());
    assert!(backend.solve(&i64_matrix, &i64_rhs).is_err());
    assert!(backend
        .triangular_solve(&i64_matrix, &i64_rhs, true, true, false, false)
        .is_err());
}

#[test]
fn test_backend_default_and_buffer_pool_len() {
    let backend = CpuBackend::default();
    assert!(backend.num_threads() >= 1);
    assert_eq!(backend.buffer_pool_len(), 0);
}

#[test]
fn test_backend_buffer_pool_controls_report_and_update_limits() {
    let ctx = Arc::new(CpuContext::with_threads(1));
    let mut backend = CpuBackend::from_context_with_buffer_pool_limit(ctx, 64);

    assert_eq!(backend.num_threads(), 1);
    assert_eq!(backend.buffer_pool_limit_bytes(), 64);
    assert_eq!(backend.buffer_pool_len(), 0);
    let stats = backend.buffer_pool_stats();
    assert_eq!(stats.buffers, 0);
    assert_eq!(stats.capacity_bytes, 0);
    let cache_stats = backend.buffer_pool_cache_stats();
    assert_eq!(cache_stats.entries, 0);
    assert_eq!(cache_stats.retained_bytes, 0);

    backend.set_buffer_pool_limit_bytes(0);
    assert_eq!(backend.buffer_pool_limit_bytes(), 0);
    backend.reset_buffer_pool();
    assert_eq!(backend.buffer_pool_len(), 0);
}

#[test]
fn test_backend_mul_neg_conj_dispatch() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, -2.0]));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]));
    let c = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
    ));
    let mut backend = CpuBackend::new();

    let prod = TensorBackend::mul(&mut backend, &a, &b).unwrap();
    assert_eq!(get_f64(&prod, &[0]), 3.0);
    assert_eq!(get_f64(&prod, &[1]), -8.0);

    let negated = backend.neg(&a).unwrap();
    assert_eq!(get_f64(&negated, &[0]), -1.0);
    assert_eq!(get_f64(&negated, &[1]), 2.0);

    let conjugated = backend.conj(&c).unwrap();
    assert_c64_close(get_c64(&conjugated, &[0]), Complex64::new(1.0, -2.0));
    assert_c64_close(get_c64(&conjugated, &[1]), Complex64::new(-3.0, -0.5));
}

#[test]
fn test_backend_structural_ops_dispatch() {
    let mut backend = CpuBackend::new();
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));

    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![5.0]));
    let broadcast = backend.broadcast_in_dim(&scalar, &[2, 2], &[]).unwrap();
    assert_eq!(broadcast.shape(), &[2, 2]);
    assert_eq!(get_f64(&broadcast, &[0, 0]), 5.0);
    assert_eq!(get_f64(&broadcast, &[1, 1]), 5.0);

    let diag = backend.extract_diagonal(&a, 0, 1).unwrap();
    assert_eq!(diag.shape(), &[2]);
    assert_eq!(get_f64(&diag, &[0]), 1.0);
    assert_eq!(get_f64(&diag, &[1]), 4.0);

    let d = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![10.0, 20.0]));
    let embedded = backend.embed_diagonal(&d, 0, 1).unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(get_f64(&embedded, &[0, 0]), 10.0);
    assert_eq!(get_f64(&embedded, &[1, 1]), 20.0);

    let tril_result = backend.tril(&a, 0).unwrap();
    assert_eq!(tril_result.shape(), &[2, 2]);
    assert_eq!(get_f64(&tril_result, &[0, 1]), 0.0);

    let triu_result = backend.triu(&a, 0).unwrap();
    assert_eq!(triu_result.shape(), &[2, 2]);
    assert_eq!(get_f64(&triu_result, &[1, 0]), 0.0);

    let summed = TensorBackend::reduce_sum(&mut backend, &a, &[0]).unwrap();
    assert_eq!(summed.shape(), &[2]);
    assert_eq!(get_f64(&summed, &[0]), 3.0);
    assert_eq!(get_f64(&summed, &[1]), 7.0);
}

#[test]
fn test_backend_dot_general_f32_c32_and_dtype_mismatch() {
    let mut backend = CpuBackend::new();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let a_f32 = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![1, 2],
        vec![1.0f32, 2.0],
    ));
    let b_f32 = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2, 1],
        vec![3.0f32, 4.0],
    ));
    let out_f32 = backend.dot_general(&a_f32, &b_f32, &config).unwrap();
    assert_eq!(out_f32.shape(), &[1, 1]);

    let a_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![1, 2],
        vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
    ));
    let b_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 1],
        vec![Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)],
    ));
    let out_c32 = backend.dot_general(&a_c32, &b_c32, &config).unwrap();
    assert_eq!(out_c32.shape(), &[1, 1]);

    let f64_t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    let f32_t = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]));
    let err = backend.dot_general(&f64_t, &f32_t, &config).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::DTypeMismatch {
            op: "dot_general",
            ..
        }
    ));
}

#[test]
fn test_backend_gather_scatter_dynamic_slice_dispatch() {
    let mut backend = CpuBackend::new();

    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 2, 4]);
    let gathered = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap();
    assert_eq!(gathered.shape(), &[3]);
    assert_eq!(get_f64(&gathered, &[0]), 10.0);
    assert_eq!(get_f64(&gathered, &[2]), 50.0);

    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3]));
    let scatter_indices = Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 1, 2, 0, 1, 2]);
    let updates = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![5.0, 6.0, 7.0],
    ));
    let scattered = backend
        .scatter(
            &operand,
            &scatter_indices,
            &updates,
            &diagonal_scatter_config(),
        )
        .unwrap();
    assert_eq!(get_f64(&scattered, &[0, 0]), 5.0);
    assert_eq!(get_f64(&scattered, &[1, 1]), 6.0);
    assert_eq!(get_f64(&scattered, &[2, 2]), 7.0);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    ));
    let starts = Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]);
    let ds = backend.dynamic_slice(&input, &starts, &[2, 2]).unwrap();
    assert_eq!(ds.shape(), &[2, 2]);
    assert_eq!(get_f64(&ds, &[0, 0]), 11.0);
    assert_eq!(get_f64(&ds, &[1, 1]), 16.0);
}

#[test]
fn test_solve_zero_dim_returns_zeros() {
    let mut backend = CpuBackend::new();
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 0], vec![]));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 1], vec![]));
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[0, 1]);
}

#[test]
fn test_solve_with_1d_vector_rhs() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![2.0, 1.0, 0.0, 3.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![5.0, 7.0]));
    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[2]);
    let expected = matmul_f64(
        &[2.0, 1.0, 0.0, 3.0],
        &[get_f64(&x, &[0]), get_f64(&x, &[1])],
        2,
        2,
        1,
    );
    assert_f64_close_tol(expected[0], 5.0, 1e-10);
    assert_f64_close_tol(expected[1], 7.0, 1e-10);
}

#[test]
fn test_solve_with_batched_vector_rhs() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![2.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![5.0, 7.0, 3.0, 4.0],
    ));
    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[2, 2]);
}

#[test]
fn test_triangular_solve_dtype_mismatch_and_unsupported() {
    let mut backend = CpuBackend::new();
    let a_f32 = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0f32, 0.0, 0.0, 1.0],
    ));
    let b_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]));
    let err = backend
        .triangular_solve(&a_f32, &b_f64, true, true, false, false)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::DTypeMismatch {
            op: "triangular_solve",
            ..
        }
    ));

    let a_i64 = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1_i64, 0, 0, 1],
    ));
    let b_i64 = Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1_i64, 2]));
    let err = backend
        .triangular_solve(&a_i64, &b_i64, true, true, false, false)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure {
            op: "triangular_solve",
            ..
        }
    ));
}

#[test]
fn test_reclaim_buffer_returns_host_buffer_to_pool() {
    let mut backend = CpuBackend::new();
    assert_eq!(backend.buffer_pool_len(), 0);
    let t = TensorBackend::add(
        &mut backend,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0])),
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0])),
    )
    .unwrap();
    backend.reclaim_buffer(t);
    assert!(backend.buffer_pool_len() > 0);
}

#[test]
fn test_elementwise_add_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![0.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![4.0, 3.0, 2.0, 1.0],
    ));
    let out = backend.add(&lhs, &rhs).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0]), 5.0);
    assert_eq!(get_f64(&out, &[3]), 5.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_structural_transpose_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![0.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let out = backend.transpose(&input, &[1, 0]).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0, 0]), 1.0);
    assert_eq!(get_f64(&out, &[1, 0]), 3.0);
    assert_eq!(get_f64(&out, &[0, 1]), 2.0);
    assert_eq!(get_f64(&out, &[1, 1]), 4.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_convert_acquires_output_from_dtype_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F32(TypedTensor::from_vec_col_major(
        vec![4],
        vec![0.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![1.25, 2.5, 3.75, 4.0],
    ));
    let out = backend.convert(&input, DType::F32).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f32(&out, &[0]), 1.25);
    assert_eq!(get_f32(&out, &[3]), 4.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_slice_acquires_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![0.0; 2],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let config = SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    };
    let out = backend.slice(&input, &config).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0]), 2.0);
    assert_eq!(get_f64(&out, &[1]), 3.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_pad_acquires_and_zeroes_output_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![9.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    let config = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![0],
    };
    let out = backend.pad(&input, &config).unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 1.0);
    assert_eq!(get_f64(&out, &[2]), 2.0);
    assert_eq!(get_f64(&out, &[3]), 0.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_dynamic_update_slice_acquires_clone_from_pool() {
    let mut backend = CpuBackend::new();
    backend.reclaim_buffer(Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![9.0; 4],
    )));
    assert_eq!(backend.buffer_pool_len(), 1);

    let operand = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4],
        vec![0.0, 1.0, 2.0, 3.0],
    ));
    let update = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![7.0, 8.0]));
    let starts = Tensor::I64(TypedTensor::from_vec_col_major(vec![1], vec![1]));
    let out = backend
        .dynamic_update_slice(&operand, &update, &starts)
        .unwrap();

    assert_eq!(backend.buffer_pool_len(), 0);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 7.0);
    assert_eq!(get_f64(&out, &[2]), 8.0);
    assert_eq!(get_f64(&out, &[3]), 3.0);
    backend.reclaim_buffer(out);
    assert_eq!(backend.buffer_pool_len(), 1);
}

#[test]
fn test_reclaim_buffer_covers_all_dtypes() {
    let mut backend = CpuBackend::new();
    let f32_t = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]));
    backend.reclaim_buffer(f32_t);
    let c32_t = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![1],
        vec![Complex32::new(1.0, 0.0)],
    ));
    backend.reclaim_buffer(c32_t);
    let c64_t = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![1],
        vec![Complex64::new(1.0, 0.0)],
    ));
    backend.reclaim_buffer(c64_t);
    assert!(backend.buffer_pool_len() >= 3);
}

#[test]
fn test_solve_zero_dim_rhs_returns_zeros() {
    let mut backend = CpuBackend::new();
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 0.0, 0.0, 1.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![0], vec![]));
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[0]);
}

#[test]
fn test_install_with_pool_preserves_buffers() {
    let mut backend = CpuBackend::with_threads(1);
    let t = TensorBackend::add(
        &mut backend,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0])),
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0])),
    )
    .unwrap();
    assert_eq!(get_f64(&t, &[0]), 4.0);
    assert_eq!(get_f64(&t, &[1]), 6.0);
    assert_eq!(backend.buffer_pool_len(), 0);
}

#[test]
fn test_solve_with_regular_matrix_rhs() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![2.0, 1.0, 0.0, 3.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![5.0, 7.0, 3.0, 4.0],
    ));
    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[2, 2]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&[2.0, 1.0, 0.0, 3.0], &x_data, 2, 2, 2);
    assert_f64_close_tol(recon[0], 5.0, 1e-10);
    assert_f64_close_tol(recon[1], 7.0, 1e-10);
    assert_f64_close_tol(recon[2], 3.0, 1e-10);
    assert_f64_close_tol(recon[3], 4.0, 1e-10);
}

#[test]
fn test_lu_unsupported_dtype_returns_error() {
    let input = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1_i64, 0, 0, 1],
    ));
    let mut backend = CpuBackend::new();
    assert!(backend.lu(&input).is_err());
}

#[test]
fn test_lu_zero_sized_batch_outputs_empty_parity() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2, 0], Vec::new()));
    let mut backend = CpuBackend::new();
    let outputs = backend.lu(&input).unwrap();

    assert_eq!(outputs.len(), 4);
    assert_eq!(outputs[0].shape(), &[2, 2, 0]);
    assert_eq!(outputs[1].shape(), &[2, 2, 0]);
    assert_eq!(outputs[2].shape(), &[2, 2, 0]);
    assert_eq!(outputs[3].shape(), &[0]);
    for output in outputs {
        match output {
            Tensor::F64(inner) => assert!(inner.host_data().is_empty()),
            other => panic!("expected f64 tensor, got {:?}", other.dtype()),
        }
    }
}

#[test]
fn test_svd_unsupported_dtype_returns_error() {
    let input = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1_i64, 0, 0, 1],
    ));
    let mut backend = CpuBackend::new();
    assert!(backend.svd(&input).is_err());
}

#[cfg(feature = "cpu-faer")]
#[test]
fn test_faer_svd_decomposition_failure_returns_error() {
    let ctx = CpuContext::with_threads(1);
    let mut buffers = BufferPool::new();
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![f64::NAN, 0.0, 0.0, 1.0]);

    let err = faer_linalg::svd(&ctx, &mut buffers, &input).unwrap_err();

    assert!(err.to_string().contains("svd"), "unexpected error: {err}");
}

#[cfg(feature = "cpu-faer")]
#[test]
fn test_faer_eig_decomposition_failure_returns_error() {
    let ctx = CpuContext::with_threads(1);
    let mut buffers = BufferPool::new();
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![f64::NAN, 0.0, 0.0, 1.0],
    ));

    let err = faer_linalg::eig(&ctx, &mut buffers, &input).unwrap_err();

    assert!(err.to_string().contains("eig"), "unexpected error: {err}");
}

#[test]
fn test_qr_unsupported_dtype_returns_error() {
    let input = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1_i64, 0, 0, 1],
    ));
    let mut backend = CpuBackend::new();
    assert!(backend.qr(&input).is_err());
}

#[test]
fn test_eig_returns_complex_outputs_for_real_input() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![0.0, -1.0, 1.0, 0.0],
    ));
    let mut backend = CpuBackend::new();
    let outputs = backend.eig(&input).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
}

#[test]
fn test_default_backend_session_methods_cover_cache_fallbacks() {
    struct DefaultOnlyBackend;

    macro_rules! panic_backend_methods {
        ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
            $(
                fn $name(&mut self, $($arg: $argty),*) -> $ret {
                    $(let _ = &$arg;)*
                    panic!(concat!(stringify!($name), " should not be called by this test"))
                }
            )+
        };
    }

    impl TensorBackend for DefaultOnlyBackend {
        type RuntimeCache = ();

        panic_backend_methods! {
            add(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            neg(input: &Tensor) -> crate::Result<Tensor>;
            div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            abs(input: &Tensor) -> crate::Result<Tensor>;
            sign(input: &Tensor) -> crate::Result<Tensor>;
            maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
            select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor>;
            clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
            exp(input: &Tensor) -> crate::Result<Tensor>;
            log(input: &Tensor) -> crate::Result<Tensor>;
            sin(input: &Tensor) -> crate::Result<Tensor>;
            cos(input: &Tensor) -> crate::Result<Tensor>;
            tanh(input: &Tensor) -> crate::Result<Tensor>;
            sqrt(input: &Tensor) -> crate::Result<Tensor>;
            rsqrt(input: &Tensor) -> crate::Result<Tensor>;
            pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            expm1(input: &Tensor) -> crate::Result<Tensor>;
            log1p(input: &Tensor) -> crate::Result<Tensor>;
            transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
            reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
            broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
            convert(input: &Tensor, to: DType) -> crate::Result<Tensor>;
            extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
            embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
            tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
            triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
            reduce_sum(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            reduce_prod(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            reduce_max(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            reduce_min(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> crate::Result<Tensor>;
            scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> crate::Result<Tensor>;
            slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
            dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> crate::Result<Tensor>;
            dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> crate::Result<Tensor>;
            pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
            concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
            reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            cholesky(input: &Tensor) -> crate::Result<Tensor>;
            triangular_solve(
                a: &Tensor,
                b: &Tensor,
                left_side: bool,
                lower: bool,
                transpose_a: bool,
                unit_diagonal: bool
            ) -> crate::Result<Tensor>;
            lu(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            full_piv_lu(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            full_piv_lu_solve(a: &Tensor, b: &Tensor, transpose_a: bool) -> crate::Result<Tensor>;
            svd(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            qr(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            eigh(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            eig(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            solve(a: &Tensor, b: &Tensor) -> crate::Result<Tensor>;
        }

        fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().conj(input)
        }

        fn dot_general(
            &mut self,
            lhs: &Tensor,
            rhs: &Tensor,
            config: &DotGeneralConfig,
        ) -> crate::Result<Tensor> {
            CpuBackend::new().dot_general(lhs, rhs, config)
        }
    }

    struct DefaultOnlyExec;

    impl crate::backend::BackendSession for DefaultOnlyExec {
        panic_backend_methods! {
            add(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            neg(input: &Tensor) -> crate::Result<Tensor>;
            div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            abs(input: &Tensor) -> crate::Result<Tensor>;
            sign(input: &Tensor) -> crate::Result<Tensor>;
            maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
            select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor>;
            clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
            exp(input: &Tensor) -> crate::Result<Tensor>;
            log(input: &Tensor) -> crate::Result<Tensor>;
            sin(input: &Tensor) -> crate::Result<Tensor>;
            cos(input: &Tensor) -> crate::Result<Tensor>;
            tanh(input: &Tensor) -> crate::Result<Tensor>;
            sqrt(input: &Tensor) -> crate::Result<Tensor>;
            rsqrt(input: &Tensor) -> crate::Result<Tensor>;
            pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
            expm1(input: &Tensor) -> crate::Result<Tensor>;
            log1p(input: &Tensor) -> crate::Result<Tensor>;
            transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
            reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
            broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
            convert(input: &Tensor, to: DType) -> crate::Result<Tensor>;
            extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
            embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
            tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
            triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
            reduce_sum(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            reduce_prod(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            reduce_max(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            reduce_min(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> crate::Result<Tensor>;
            scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> crate::Result<Tensor>;
            slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
            dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> crate::Result<Tensor>;
            dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> crate::Result<Tensor>;
            pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
            concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
            reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
            cholesky(input: &Tensor) -> crate::Result<Tensor>;
            triangular_solve(
                a: &Tensor,
                b: &Tensor,
                left_side: bool,
                lower: bool,
                transpose_a: bool,
                unit_diagonal: bool
            ) -> crate::Result<Tensor>;
            lu(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            full_piv_lu(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            full_piv_lu_solve(a: &Tensor, b: &Tensor, transpose_a: bool) -> crate::Result<Tensor>;
            svd(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            qr(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            eigh(input: &Tensor) -> crate::Result<Vec<Tensor>>;
            eig(input: &Tensor) -> crate::Result<Vec<Tensor>>;
        }

        fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            CpuBackend::new().conj(input)
        }

        fn dot_general(
            &mut self,
            lhs: &Tensor,
            rhs: &Tensor,
            config: &DotGeneralConfig,
        ) -> crate::Result<Tensor> {
            CpuBackend::new().dot_general(lhs, rhs, config)
        }

        fn reclaim_buffer(&mut self, _tensor: Tensor) {}
    }

    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]);
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]);
    let one_shape = [1usize, 1];
    let lhs_data = [2.0_f64];
    let rhs_data = [3.0_f64];
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut backend = DefaultOnlyBackend;
    let mut cache = ();

    let direct =
        TensorBackend::dot_general_cached(&mut backend, &mut cache, Some(0), &lhs, &rhs, &config)
            .unwrap();
    assert_eq!(direct.as_slice::<f64>().unwrap(), &[6.0]);

    let lhs_folded =
        TensorBackend::dot_general_with_conj(&mut backend, &lhs, &rhs, &config, true, false)
            .unwrap();
    assert_eq!(lhs_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let both_folded =
        TensorBackend::dot_general_with_conj(&mut backend, &lhs, &rhs, &config, true, true)
            .unwrap();
    assert_eq!(both_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let read_views = TensorBackend::dot_general_read(
        &mut backend,
        TensorRead::from_view(TensorView::f64(&one_shape, &lhs_data).unwrap()),
        TensorRead::from_view(TensorView::f64(&one_shape, &rhs_data).unwrap()),
        &config,
    )
    .unwrap();
    assert_eq!(read_views.as_slice::<f64>().unwrap(), &[6.0]);

    let rhs_folded = TensorBackend::dot_general_with_conj_cached(
        &mut backend,
        &mut cache,
        Some(1),
        &lhs,
        &rhs,
        &config,
        false,
        true,
    )
    .unwrap();
    assert_eq!(rhs_folded.as_slice::<f64>().unwrap(), &[6.0]);

    let uploaded = backend.upload_host_tensor(&lhs).unwrap();
    assert_eq!(uploaded.shape(), &[1, 1]);
    let downloaded = backend.download_to_host(&uploaded).unwrap();
    assert_eq!(downloaded.as_slice::<f64>().unwrap(), &[2.0]);
    backend.reclaim_buffer(downloaded);

    let fusion_plan = crate::backend::ElementwiseFusionPlan {
        dtype: DType::F64,
        n_inputs: 0,
        outputs: vec![],
        ops: vec![],
    };
    assert!(backend
        .execute_elementwise_fusion(&[], &fusion_plan)
        .unwrap()
        .is_none());

    let session_value =
        TensorBackend::with_backend_session_cached(&mut backend, &mut cache, |exec| {
            let cached = exec
                .dot_general_cached(Some(2), &lhs, &rhs, &config)
                .unwrap();
            let folded = exec
                .dot_general_with_conj_cached(Some(3), &lhs, &rhs, &config, true, false)
                .unwrap();
            cached.as_slice::<f64>().unwrap()[0] + folded.as_slice::<f64>().unwrap()[0]
        });
    assert_eq!(session_value, 12.0);

    let mut exec = DefaultOnlyExec;
    let exec_read_tensor = crate::backend::BackendSession::dot_general_read(
        &mut exec,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
    )
    .unwrap();
    assert_eq!(exec_read_tensor.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_read_views = crate::backend::BackendSession::dot_general_read(
        &mut exec,
        TensorRead::from_view(TensorView::f64(&one_shape, &lhs_data).unwrap()),
        TensorRead::from_view(TensorView::f64(&one_shape, &rhs_data).unwrap()),
        &config,
    )
    .unwrap();
    assert_eq!(exec_read_views.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_no_conj = crate::backend::BackendSession::dot_general_with_conj(
        &mut exec, &lhs, &rhs, &config, false, false,
    )
    .unwrap();
    assert_eq!(exec_no_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_lhs_conj = crate::backend::BackendSession::dot_general_with_conj(
        &mut exec, &lhs, &rhs, &config, true, false,
    )
    .unwrap();
    assert_eq!(exec_lhs_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_rhs_conj = crate::backend::BackendSession::dot_general_with_conj(
        &mut exec, &lhs, &rhs, &config, false, true,
    )
    .unwrap();
    assert_eq!(exec_rhs_conj.as_slice::<f64>().unwrap(), &[6.0]);
    let exec_both_conj = crate::backend::BackendSession::dot_general_with_conj(
        &mut exec, &lhs, &rhs, &config, true, true,
    )
    .unwrap();
    assert_eq!(exec_both_conj.as_slice::<f64>().unwrap(), &[6.0]);
}

#[test]
fn test_pool_backed_elementwise_public_paths_cover_dtypes_and_scalars() {
    let f32_scalar = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![2.0]));
    let c32_vec = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 1.0), Complex32::new(-3.0, 0.5)],
    ));
    assert_eq!(
        add(&f32_scalar, &c32_vec)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(3.0, 1.0)
    );
    assert_eq!(
        add(&c32_vec, &f32_scalar)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[1],
        Complex32::new(-1.0, 0.5)
    );
    assert_eq!(
        div(&f32_scalar, &c32_vec)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(1.0, -1.0)
    );
    assert_eq!(
        mul(&c32_vec, &f32_scalar)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap()[0],
        Complex32::new(2.0, 2.0)
    );

    let f64_scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![4.0]));
    let c64_vec = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 2.0)],
    ));
    assert_c64_close(
        div(&c64_vec, &f64_scalar)
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()[1],
        Complex64::new(0.0, 0.5),
    );

    assert!(neg(&Tensor::from_vec_col_major(vec![1], vec![1_i64]))
        .unwrap_err()
        .to_string()
        .contains("I64"));
    assert!(conj(&Tensor::from_vec_col_major(vec![1], vec![1_i64])).is_err());
    assert!(abs(&Tensor::from_vec_col_major(vec![1], vec![1_i64])).is_err());
    assert!(sign(&Tensor::from_vec_col_major(vec![1], vec![1_i64])).is_err());

    let a = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
    ));
    let b = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.0, 2.0), Complex64::new(5.0, 0.0)],
    ));
    assert_c64_close(
        get_c64(&maximum(&a, &b).unwrap(), &[0]),
        Complex64::new(3.0, 4.0),
    );
    assert_c64_close(
        get_c64(&minimum(&a, &b).unwrap(), &[0]),
        Complex64::new(0.0, 2.0),
    );
    assert!(get_bool(&compare(&a, &b, &CompareDir::Ge).unwrap(), &[0]));
    let pred = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, true]));
    assert_c64_close(
        get_c64(&select(&pred, &a, &b).unwrap(), &[1]),
        Complex64::new(1.0, 0.0),
    );
    assert_c64_close(
        get_c64(&clamp(&a, &b, &a).unwrap(), &[1]),
        Complex64::new(5.0, 0.0),
    );
}

#[test]
fn test_pool_backed_analytic_public_paths_cover_supported_dtypes() {
    let real = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 4.0]);
    assert_f64_close(
        crate::cpu::analytic::exp(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        1.0,
    );
    assert_f64_close(
        crate::cpu::analytic::sqrt(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[1],
        2.0,
    );
    assert_f64_close(
        crate::cpu::analytic::rsqrt(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[1],
        0.5,
    );
    assert_f64_close(
        crate::cpu::analytic::log1p(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        0.0,
    );
    assert_f64_close(
        crate::cpu::analytic::expm1(&real)
            .unwrap()
            .as_slice::<f64>()
            .unwrap()[0],
        0.0,
    );

    let complex = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![1],
        vec![Complex64::new(1.0, 0.0)],
    ));
    assert_c64_close(
        crate::cpu::analytic::log(&complex)
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()[0],
        Complex64::new(0.0, 0.0),
    );
    assert!(crate::cpu::analytic::sin(&complex).is_ok());
    assert!(crate::cpu::analytic::cos(&complex).is_ok());
    assert!(crate::cpu::analytic::tanh(&complex).is_ok());

    let base = Tensor::from_vec_col_major(vec![2], vec![2.0_f32, 3.0]);
    let exponent = Tensor::from_vec_col_major(vec![2], vec![3.0_f32, 2.0]);
    assert_eq!(
        crate::cpu::analytic::pow(&base, &exponent)
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[8.0, 9.0]
    );
    assert!(crate::cpu::analytic::exp(&Tensor::from_vec_col_major(vec![1], vec![1_i64])).is_err());
    assert!(crate::cpu::analytic::pow(&real, &base).is_err());
}

#[test]
fn test_pool_backed_structural_public_paths_cover_dispatch_and_helpers() {
    let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let transposed = transpose(&matrix, &[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);

    let typed = TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]);
    let typed_t = crate::cpu::structural::typed_transpose(&typed, &[1, 0]).unwrap();
    assert_eq!(typed_t.host_data(), &[1, 3, 2, 4]);

    let row = TypedTensor::from_vec_col_major(vec![1, 2], vec![5.0_f32, 6.0]);
    let typed_b = crate::cpu::structural::typed_broadcast_in_dim(&row, &[2, 2], &[0, 1]).unwrap();
    assert_eq!(typed_b.host_data(), &[5.0, 5.0, 6.0, 6.0]);

    let scalar = Tensor::from_vec_col_major(vec![], vec![7.0_f64]);
    let broadcasted = broadcast_in_dim(&scalar, &[2, 2], &[]).unwrap();
    assert_eq!(
        broadcasted.as_slice::<f64>().unwrap(),
        &[7.0, 7.0, 7.0, 7.0]
    );

    let i64_matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]);
    let as_c64 = crate::cpu::structural::convert(&i64_matrix, DType::C64).unwrap();
    assert_eq!(as_c64.dtype(), DType::C64);
    let as_f32 = crate::cpu::structural::convert(&as_c64, DType::F32).unwrap();
    assert_eq!(as_f32.dtype(), DType::F32);
    let as_c32 = crate::cpu::structural::convert(&matrix, DType::C32).unwrap();
    assert_eq!(as_c32.dtype(), DType::C32);
    let as_i64 = crate::cpu::structural::convert(&as_c32, DType::I64).unwrap();
    assert_eq!(as_i64.as_slice::<i64>().unwrap(), &[1, 2, 3, 4]);

    let diag = extract_diagonal(&matrix, 0, 1).unwrap();
    assert_eq!(diag.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    let embedded = embed_diagonal(&diag, 0, 1).unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(embedded.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 4.0]);

    let typed_diag = crate::cpu::structural::typed_extract_diagonal(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
        0,
        1,
    )
    .unwrap();
    assert_eq!(typed_diag.host_data(), &[1.0, 4.0]);
    let typed_embedded = crate::cpu::structural::typed_embed_diagonal(&typed_diag, 0, 1).unwrap();
    assert_eq!(typed_embedded.host_data(), &[1.0, 0.0, 0.0, 4.0]);

    let lower = tril(&matrix, 0).unwrap();
    assert_eq!(lower.as_slice::<f64>().unwrap(), &[1.0, 2.0, 0.0, 4.0]);
    let upper = triu(&matrix, 0).unwrap();
    assert_eq!(upper.as_slice::<f64>().unwrap(), &[1.0, 0.0, 3.0, 4.0]);
    let typed_lower = crate::cpu::structural::typed_tril(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]),
        0,
    )
    .unwrap();
    assert_eq!(typed_lower.host_data(), &[1, 2, 0, 4]);
    let typed_upper = crate::cpu::structural::typed_triu(
        &TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]),
        0,
    )
    .unwrap();
    assert_eq!(typed_upper.host_data(), &[1, 0, 3, 4]);
    assert!(crate::cpu::structural::typed_tril(
        &TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        0
    )
    .is_err());

    let c32_matrix = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ],
    ));
    assert_eq!(transpose(&c32_matrix, &[1, 0]).unwrap().dtype(), DType::C32);
    assert_eq!(tril(&c32_matrix, 0).unwrap().dtype(), DType::C32);
}
