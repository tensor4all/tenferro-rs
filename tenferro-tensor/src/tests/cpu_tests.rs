use num_complex::Complex64;

use crate::backend::TensorBackend;
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::cpu::{
    add, broadcast_in_dim, conj, dynamic_slice, embed_diagonal, extract_diagonal, gather, mul, neg,
    pad, reduce_max, reduce_min, reduce_prod, reduce_sum, reshape, scatter, transpose, CpuBackend,
};
use crate::types::{DType, Tensor, TypedTensor};

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
fn test_from_vec_col_major() {
    let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(*t.get(&[0, 0]), 1.0);
    assert_eq!(*t.get(&[1, 0]), 2.0);
    assert_eq!(*t.get(&[0, 1]), 3.0);
    assert_eq!(*t.get(&[1, 1]), 4.0);
    assert_eq!(*t.get(&[0, 2]), 5.0);
    assert_eq!(*t.get(&[1, 2]), 6.0);
}

#[test]
fn test_tensor_metadata() {
    let t = Tensor::F64(TypedTensor::from_vec(vec![2, 1], vec![1.0, 2.0]));
    assert_eq!(t.shape(), &[2, 1]);
    assert_eq!(t.dtype(), DType::F64);
}

#[test]
fn test_reshape() {
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let r = reshape(&t, &[3, 2]);
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
    let a = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let b = Tensor::F64(TypedTensor::from_vec(
        vec![2, 2],
        vec![10.0, 20.0, 30.0, 40.0],
    ));
    let sum = add(&a, &b);
    let prod = mul(&a, &b);

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
fn test_rank0_typed_tensor_behaves_like_scalar() {
    let mut zeros = TypedTensor::<f64>::zeros(vec![]);
    assert_eq!(zeros.shape, vec![]);
    assert_eq!(zeros.n_elements(), 1);
    assert_eq!(zeros.linear_offset(&[]), 0);
    assert_eq!(zeros.get(&[]), &0.0);

    *zeros.get_mut(&[]) = 2.5;
    assert_eq!(zeros.host_data(), &[2.5]);

    let ones = TypedTensor::<f64>::ones(vec![]);
    assert_eq!(ones.shape, vec![]);
    assert_eq!(ones.n_elements(), 1);
    assert_eq!(ones.get(&[]), &1.0);

    let scalar = TypedTensor::<f64>::from_vec(vec![], vec![7.0]);
    assert_eq!(scalar.shape, vec![]);
    assert_eq!(scalar.n_elements(), 1);
    assert_eq!(scalar.linear_offset(&[]), 0);
    assert_eq!(scalar.get(&[]), &7.0);
}

#[test]
fn test_reduce_sum() {
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let r = reduce_sum(&t, &[0]);
    assert_eq!(r.shape(), &[3]);
    assert_eq!(get_f64(&r, &[0]), 3.0);
    assert_eq!(get_f64(&r, &[1]), 7.0);
    assert_eq!(get_f64(&r, &[2]), 11.0);

    let all = reduce_sum(&t, &[0, 1]);
    assert!(all.shape().is_empty());
    assert_eq!(get_f64(&all, &[]), 21.0);
}

#[test]
fn test_reduce_prod() {
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let r = reduce_prod(&t, &[0]);
    assert_eq!(r.shape(), &[3]);
    assert_eq!(get_f64(&r, &[0]), 2.0);
    assert_eq!(get_f64(&r, &[1]), 12.0);
    assert_eq!(get_f64(&r, &[2]), 30.0);

    let all = reduce_prod(&t, &[0, 1]);
    assert!(all.shape().is_empty());
    assert_eq!(get_f64(&all, &[]), 720.0);
}

#[test]
fn test_reduce_max_and_min() {
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let max_cols = reduce_max(&t, &[0]);
    assert_eq!(max_cols.shape(), &[3]);
    assert_eq!(get_f64(&max_cols, &[0]), 2.0);
    assert_eq!(get_f64(&max_cols, &[1]), 4.0);
    assert_eq!(get_f64(&max_cols, &[2]), 6.0);

    let max_all = reduce_max(&t, &[0, 1]);
    assert!(max_all.shape().is_empty());
    assert_eq!(get_f64(&max_all, &[]), 6.0);

    let min_rows = reduce_min(&t, &[1]);
    assert_eq!(min_rows.shape(), &[2]);
    assert_eq!(get_f64(&min_rows, &[0]), 1.0);
    assert_eq!(get_f64(&min_rows, &[1]), 2.0);

    let min_all = reduce_min(&t, &[0, 1]);
    assert!(min_all.shape().is_empty());
    assert_eq!(get_f64(&min_all, &[]), 1.0);
}

#[test]
fn test_backend_reduce_prod_max_and_min_delegate_to_cpu_reduction_impls() {
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let mut backend = CpuBackend::new();

    let prod = backend.reduce_prod(&t, &[0]);
    assert_eq!(prod.shape(), &[3]);
    assert_eq!(get_f64(&prod, &[0]), 2.0);
    assert_eq!(get_f64(&prod, &[1]), 12.0);
    assert_eq!(get_f64(&prod, &[2]), 30.0);

    let max = backend.reduce_max(&t, &[1]);
    assert_eq!(max.shape(), &[2]);
    assert_eq!(get_f64(&max, &[0]), 5.0);
    assert_eq!(get_f64(&max, &[1]), 6.0);

    let min = backend.reduce_min(&t, &[0, 1]);
    assert!(min.shape().is_empty());
    assert_eq!(get_f64(&min, &[]), 1.0);
}

#[test]
fn test_slice() {
    let input = Tensor::F64(TypedTensor::from_vec(
        vec![4, 4],
        (1..=16).map(|value| value as f64).collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.slice(
        &input,
        &SliceConfig {
            starts: vec![1, 1],
            limits: vec![3, 3],
            strides: vec![1, 1],
        },
    );

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64(&out, &[0, 0]), 6.0);
    assert_eq!(get_f64(&out, &[1, 0]), 7.0);
    assert_eq!(get_f64(&out, &[0, 1]), 10.0);
    assert_eq!(get_f64(&out, &[1, 1]), 11.0);
}

#[test]
fn test_reverse_axis_zero() {
    let input = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let mut backend = CpuBackend::new();
    let out = backend.reverse(&input, &[0]);

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 2.0);
    assert_eq!(get_f64(&out, &[1, 0]), 1.0);
    assert_eq!(get_f64(&out, &[0, 1]), 4.0);
    assert_eq!(get_f64(&out, &[1, 1]), 3.0);
    assert_eq!(get_f64(&out, &[0, 2]), 6.0);
    assert_eq!(get_f64(&out, &[1, 2]), 5.0);
}

#[test]
fn test_concatenate_axis_zero() {
    let lhs = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let rhs = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ));
    let mut backend = CpuBackend::new();
    let out = backend.concatenate(&[&lhs, &rhs], 0);

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
    let a = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec(
        vec![3, 4],
        vec![
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
    ));
    let mut backend = CpuBackend::new();
    let c = backend.dot_general(
        &a,
        &b,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            lhs_rank: 2,
            rhs_rank: 2,
        },
    );
    assert_eq!(c.shape(), &[2, 4]);
    assert_eq!(get_f64(&c, &[0, 0]), 38.0);
    assert_eq!(get_f64(&c, &[1, 0]), 83.0);
    assert_eq!(get_f64(&c, &[0, 1]), 44.0);
    assert_eq!(get_f64(&c, &[1, 1]), 98.0);
    assert_eq!(get_f64(&c, &[0, 3]), 56.0);
    assert_eq!(get_f64(&c, &[1, 3]), 128.0);
}

#[test]
fn test_dot_general_inner_product_returns_rank0_scalar() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let b = Tensor::F64(TypedTensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]));
    let mut backend = CpuBackend::new();
    let c = backend.dot_general(
        &a,
        &b,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            lhs_rank: 1,
            rhs_rank: 1,
        },
    );
    assert!(c.shape().is_empty());
    assert_eq!(get_f64(&c, &[]), 32.0);
}

#[test]
fn test_transpose() {
    let t = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let tr = transpose(&t, &[1, 0]);
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
    let scalar = Tensor::F64(TypedTensor::from_vec(vec![], vec![5.0]));
    let broadcast = broadcast_in_dim(&scalar, &[3], &[]);
    assert_eq!(broadcast.shape(), &[3]);
    assert_eq!(get_f64(&broadcast, &[0]), 5.0);
    assert_eq!(get_f64(&broadcast, &[1]), 5.0);
    assert_eq!(get_f64(&broadcast, &[2]), 5.0);

    let v = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let m = broadcast_in_dim(&v, &[3, 2], &[0]);
    assert_eq!(m.shape(), &[3, 2]);
    for j in 0..2 {
        assert_eq!(get_f64(&m, &[0, j]), 1.0);
        assert_eq!(get_f64(&m, &[1, j]), 2.0);
        assert_eq!(get_f64(&m, &[2, j]), 3.0);
    }
}

#[test]
fn test_neg_and_conj() {
    let t = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, -7.0]));
    let n = neg(&t);
    assert_eq!(get_f64(&n, &[0]), -3.0);
    assert_eq!(get_f64(&n, &[1]), 7.0);

    let c = conj(&t);
    assert_eq!(get_f64(&c, &[0]), 3.0);
    assert_eq!(get_f64(&c, &[1]), -7.0);
}

#[test]
fn test_cpu_backend_analytic_ops_real() {
    let mut backend = CpuBackend::new();

    let exp_input = Tensor::F64(TypedTensor::from_vec(vec![2], vec![0.0, 1.0]));
    let exp_out = backend.exp(&exp_input);
    assert_f64_close(get_f64(&exp_out, &[0]), 1.0);
    assert_f64_close(get_f64(&exp_out, &[1]), std::f64::consts::E);

    let log_input = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 4.0]));
    let log_out = backend.log(&log_input);
    assert_f64_close(get_f64(&log_out, &[0]), 0.0);
    assert_f64_close(get_f64(&log_out, &[1]), 4.0_f64.ln());

    let trig_input = Tensor::F64(TypedTensor::from_vec(
        vec![2],
        vec![0.0, std::f64::consts::FRAC_PI_2],
    ));
    let sin_out = backend.sin(&trig_input);
    let cos_out = backend.cos(&trig_input);
    assert_f64_close(get_f64(&sin_out, &[0]), 0.0);
    assert_f64_close(get_f64(&sin_out, &[1]), 1.0);
    assert_f64_close(get_f64(&cos_out, &[0]), 1.0);
    assert_f64_close(get_f64(&cos_out, &[1]), 0.0);

    let tanh_input = Tensor::F64(TypedTensor::from_vec(vec![2], vec![0.0, 1.0]));
    let tanh_out = backend.tanh(&tanh_input);
    assert_f64_close(get_f64(&tanh_out, &[0]), 0.0);
    assert_f64_close(get_f64(&tanh_out, &[1]), 1.0_f64.tanh());

    let sqrt_input = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 4.0]));
    let sqrt_out = backend.sqrt(&sqrt_input);
    let rsqrt_out = backend.rsqrt(&sqrt_input);
    assert_f64_close(get_f64(&sqrt_out, &[0]), 1.0);
    assert_f64_close(get_f64(&sqrt_out, &[1]), 2.0);
    assert_f64_close(get_f64(&rsqrt_out, &[0]), 1.0);
    assert_f64_close(get_f64(&rsqrt_out, &[1]), 0.5);

    let expm1_out = backend.expm1(&exp_input);
    let log1p_out = backend.log1p(&log_input);
    assert_f64_close(get_f64(&expm1_out, &[0]), 0.0);
    assert_f64_close(get_f64(&expm1_out, &[1]), 1.0_f64.exp_m1());
    assert_f64_close(get_f64(&log1p_out, &[0]), 2.0_f64.ln());
    assert_f64_close(get_f64(&log1p_out, &[1]), 5.0_f64.ln());

    let pow_base = Tensor::F64(TypedTensor::from_vec(vec![2], vec![2.0, 9.0]));
    let pow_exp = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, 0.5]));
    let pow_out = backend.pow(&pow_base, &pow_exp);
    assert_f64_close(get_f64(&pow_out, &[0]), 8.0);
    assert_f64_close(get_f64(&pow_out, &[1]), 3.0);
}

#[test]
fn test_cpu_backend_analytic_ops_complex() {
    let mut backend = CpuBackend::new();

    let exp_input = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 1.0)],
    ));
    let exp_out = backend.exp(&exp_input);
    assert_c64_close(get_c64(&exp_out, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&exp_out, &[1]), Complex64::new(1.0, 1.0).exp());

    let log_input = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, -0.5)],
    ));
    let log_out = backend.log(&log_input);
    assert_c64_close(get_c64(&log_out, &[0]), Complex64::new(1.0, 0.0).ln());
    assert_c64_close(get_c64(&log_out, &[1]), Complex64::new(2.0, -0.5).ln());

    let trig_input = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, -0.25)],
    ));
    let sin_out = backend.sin(&trig_input);
    let cos_out = backend.cos(&trig_input);
    let tanh_out = backend.tanh(&trig_input);
    assert_c64_close(get_c64(&sin_out, &[0]), Complex64::new(0.0, 0.0).sin());
    assert_c64_close(get_c64(&sin_out, &[1]), Complex64::new(0.5, -0.25).sin());
    assert_c64_close(get_c64(&cos_out, &[0]), Complex64::new(0.0, 0.0).cos());
    assert_c64_close(get_c64(&cos_out, &[1]), Complex64::new(0.5, -0.25).cos());
    assert_c64_close(get_c64(&tanh_out, &[0]), Complex64::new(0.0, 0.0).tanh());
    assert_c64_close(get_c64(&tanh_out, &[1]), Complex64::new(0.5, -0.25).tanh());

    let sqrt_input = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(4.0, 3.0)],
    ));
    let sqrt_out = backend.sqrt(&sqrt_input);
    let rsqrt_out = backend.rsqrt(&sqrt_input);
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

    let expm1_out = backend.expm1(&exp_input);
    let log1p_out = backend.log1p(&log_input);
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

    let pow_base = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    ));
    let pow_exp = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.5, 0.25)],
    ));
    let pow_out = backend.pow(&pow_base, &pow_exp);
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
    let square = Tensor::F64(TypedTensor::from_vec(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let d = extract_diagonal(&square, 0, 1);
    assert_eq!(d.shape(), &[3]);
    assert_eq!(get_f64(&d, &[0]), 1.0);
    assert_eq!(get_f64(&d, &[1]), 5.0);
    assert_eq!(get_f64(&d, &[2]), 9.0);

    let cube = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3, 3],
        (1..=18).map(|x| x as f64).collect(),
    ));
    let diag = extract_diagonal(&cube, 1, 2);
    assert_eq!(diag.shape(), &[2, 3]);
    assert_eq!(get_f64(&diag, &[0, 0]), 1.0);
    assert_eq!(get_f64(&diag, &[1, 1]), 10.0);
    assert_eq!(get_f64(&diag, &[1, 2]), 18.0);
}

#[test]
fn test_embed_diagonal() {
    let v = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let m = embed_diagonal(&v, 0, 1);
    assert_eq!(m.shape(), &[3, 3]);
    assert_eq!(get_f64(&m, &[0, 0]), 1.0);
    assert_eq!(get_f64(&m, &[1, 1]), 2.0);
    assert_eq!(get_f64(&m, &[2, 2]), 3.0);
    assert_eq!(get_f64(&m, &[0, 1]), 0.0);
    assert_eq!(get_f64(&m, &[2, 0]), 0.0);
}

#[test]
fn test_cpu_backend_dispatches_tensor_backend_ops() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
    let b = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, 4.0]));
    let mut backend = CpuBackend::new();
    let out = TensorBackend::add(&mut backend, &a, &b);
    assert_eq!(get_f64(&out, &[0]), 4.0);
    assert_eq!(get_f64(&out, &[1]), 6.0);
}

#[test]
fn test_tier2_elementwise_ops_real() {
    let lhs = Tensor::F64(TypedTensor::from_vec(vec![3], vec![8.0, -2.0, 9.0]));
    let rhs = Tensor::F64(TypedTensor::from_vec(vec![3], vec![2.0, 5.0, 3.0]));
    let pred = Tensor::F64(TypedTensor::from_vec(vec![3], vec![0.0, -1.0, 2.0]));
    let on_true = Tensor::F64(TypedTensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]));
    let on_false = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let lower = Tensor::F64(TypedTensor::from_vec(vec![3], vec![-1.0, -1.0, 0.0]));
    let upper = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 0.25, 4.0]));
    let mut backend = CpuBackend::new();

    let div = backend.div(&lhs, &rhs);
    assert_eq!(get_f64(&div, &[0]), 4.0);
    assert_eq!(get_f64(&div, &[1]), -0.4);
    assert_eq!(get_f64(&div, &[2]), 3.0);

    let abs = backend.abs(&lhs);
    assert_eq!(get_f64(&abs, &[0]), 8.0);
    assert_eq!(get_f64(&abs, &[1]), 2.0);
    assert_eq!(get_f64(&abs, &[2]), 9.0);

    let sign = backend.sign(&lhs);
    assert_eq!(get_f64(&sign, &[0]), 1.0);
    assert_eq!(get_f64(&sign, &[1]), -1.0);
    assert_eq!(get_f64(&sign, &[2]), 1.0);

    let maximum = backend.maximum(&lhs, &rhs);
    assert_eq!(get_f64(&maximum, &[0]), 8.0);
    assert_eq!(get_f64(&maximum, &[1]), 5.0);
    assert_eq!(get_f64(&maximum, &[2]), 9.0);

    let minimum = backend.minimum(&lhs, &rhs);
    assert_eq!(get_f64(&minimum, &[0]), 2.0);
    assert_eq!(get_f64(&minimum, &[1]), -2.0);
    assert_eq!(get_f64(&minimum, &[2]), 3.0);

    let eq = backend.compare(&lhs, &rhs, &CompareDir::Eq);
    assert_eq!(get_f64(&eq, &[0]), 0.0);
    assert_eq!(get_f64(&eq, &[1]), 0.0);
    assert_eq!(get_f64(&eq, &[2]), 0.0);

    let lt = backend.compare(&lhs, &rhs, &CompareDir::Lt);
    assert_eq!(get_f64(&lt, &[0]), 0.0);
    assert_eq!(get_f64(&lt, &[1]), 1.0);
    assert_eq!(get_f64(&lt, &[2]), 0.0);

    let le = backend.compare(&lhs, &rhs, &CompareDir::Le);
    assert_eq!(get_f64(&le, &[0]), 0.0);
    assert_eq!(get_f64(&le, &[1]), 1.0);
    assert_eq!(get_f64(&le, &[2]), 0.0);

    let gt = backend.compare(&lhs, &rhs, &CompareDir::Gt);
    assert_eq!(get_f64(&gt, &[0]), 1.0);
    assert_eq!(get_f64(&gt, &[1]), 0.0);
    assert_eq!(get_f64(&gt, &[2]), 1.0);

    let ge = backend.compare(&lhs, &rhs, &CompareDir::Ge);
    assert_eq!(get_f64(&ge, &[0]), 1.0);
    assert_eq!(get_f64(&ge, &[1]), 0.0);
    assert_eq!(get_f64(&ge, &[2]), 1.0);

    let select = backend.select(&pred, &on_true, &on_false);
    assert_eq!(get_f64(&select, &[0]), 1.0);
    assert_eq!(get_f64(&select, &[1]), 20.0);
    assert_eq!(get_f64(&select, &[2]), 30.0);

    let clamp = backend.clamp(&lhs, &lower, &upper);
    assert_eq!(get_f64(&clamp, &[0]), 1.0);
    assert_eq!(get_f64(&clamp, &[1]), -1.0);
    assert_eq!(get_f64(&clamp, &[2]), 4.0);
}

#[test]
fn test_tier2_elementwise_ops_complex() {
    let input = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
    ));
    let lhs = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
    ));
    let rhs = Tensor::C64(TypedTensor::from_vec(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 2.0)],
    ));
    let mut backend = CpuBackend::new();

    let abs = backend.abs(&input);
    assert_c64_close(get_c64(&abs, &[0]), Complex64::new(5.0, 0.0));
    assert_c64_close(get_c64(&abs, &[1]), Complex64::new(0.0, 0.0));

    let sign = backend.sign(&input);
    assert_c64_close(get_c64(&sign, &[0]), Complex64::new(0.6, 0.8));
    assert_c64_close(get_c64(&sign, &[1]), Complex64::new(0.0, 0.0));

    let maximum = backend.maximum(&lhs, &rhs);
    assert_c64_close(get_c64(&maximum, &[0]), Complex64::new(3.0, 4.0));
    assert_c64_close(get_c64(&maximum, &[1]), Complex64::new(0.0, 2.0));

    let minimum = backend.minimum(&lhs, &rhs);
    assert_c64_close(get_c64(&minimum, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&minimum, &[1]), Complex64::new(1.0, 0.0));
}

#[test]
fn test_batched_cholesky() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);

    let input = Tensor::F64(TypedTensor::from_vec(
        vec![3, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.cholesky(&input);

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
    let input = Tensor::F64(TypedTensor::from_vec(
        vec![4, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.svd(&input);

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
fn test_batched_solve() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);
    let a = Tensor::F64(TypedTensor::from_vec(
        vec![3, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let b = Tensor::F64(TypedTensor::from_vec(
        vec![3, 1, 2],
        vec![1.0, 2.0, 3.0, -1.0, 4.0, 0.5],
    ));

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b);

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
    let a = Tensor::C64(TypedTensor::from_vec(
        vec![2, 2, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let b = Tensor::C64(TypedTensor::from_vec(
        vec![2, 1, 2],
        vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(0.5, 2.0),
            Complex64::new(-2.0, 0.25),
            Complex64::new(1.5, -0.75),
        ],
    ));

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b);

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
    let input = Tensor::C64(TypedTensor::from_vec(
        vec![2, 2, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));

    let mut backend = CpuBackend::new();
    let out = backend.eigh(&input);

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
fn test_complex_cholesky() {
    let l = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let a = matmul_c64(&l, &conjugate_transpose_c64(&l, 2, 2), 2, 2, 2);
    let input = Tensor::C64(TypedTensor::from_vec(vec![2, 2], a.clone()));

    let mut backend = CpuBackend::new();
    let out = backend.cholesky(&input);

    assert_eq!(out.shape(), &[2, 2]);
    let l_out = matrix_c64_from_tensor(&out, 2, 2);
    let recon = matmul_c64(&l_out, &conjugate_transpose_c64(&l_out, 2, 2), 2, 2, 2);
    for (actual, expected) in recon.iter().zip(a.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
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
    let input = Tensor::C64(TypedTensor::from_vec(vec![3, 2], input_data.clone()));
    let mut backend = CpuBackend::new();
    let out = backend.svd(&input);

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
fn test_gather_1d_indices() {
    let operand = Tensor::F64(TypedTensor::from_vec(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::F64(TypedTensor::from_vec(vec![3, 1], vec![0.0, 2.0, 4.0]));

    let out = gather(&operand, &start_indices, &simple_gather_config());

    assert_eq!(out.shape(), &[3]);
    assert_eq!(get_f64(&out, &[0]), 10.0);
    assert_eq!(get_f64(&out, &[1]), 30.0);
    assert_eq!(get_f64(&out, &[2]), 50.0);
}

#[test]
fn test_scatter_to_diagonal() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3]));
    let scatter_indices = Tensor::F64(TypedTensor::from_vec(
        vec![3, 2],
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    ));
    let updates = Tensor::F64(TypedTensor::from_vec(vec![3], vec![5.0, 6.0, 7.0]));

    let out = scatter(
        &operand,
        &scatter_indices,
        &updates,
        &diagonal_scatter_config(),
    );

    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 5.0);
    assert_eq!(get_f64(&out, &[1, 1]), 6.0);
    assert_eq!(get_f64(&out, &[2, 2]), 7.0);
    assert_eq!(get_f64(&out, &[1, 0]), 0.0);
    assert_eq!(get_f64(&out, &[0, 2]), 0.0);
}

#[test]
fn test_pad_adds_zero_edges() {
    let input = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let config = PadConfig {
        edge_padding_low: vec![1, 1],
        edge_padding_high: vec![1, 1],
        interior_padding: vec![0, 0],
    };

    let out = pad(&input, &config);

    assert_eq!(out.shape(), &[4, 5]);
    assert_eq!(get_f64(&out, &[1, 1]), 1.0);
    assert_eq!(get_f64(&out, &[2, 1]), 2.0);
    assert_eq!(get_f64(&out, &[1, 2]), 3.0);
    assert_eq!(get_f64(&out, &[2, 3]), 6.0);
    assert_eq!(get_f64(&out, &[0, 0]), 0.0);
    assert_eq!(get_f64(&out, &[3, 4]), 0.0);
}

#[test]
fn test_dynamic_slice_clamps_starts() {
    let input = Tensor::F64(TypedTensor::from_vec(
        vec![4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    ));
    let starts = Tensor::F64(TypedTensor::from_vec(vec![2], vec![2.0, 3.0]));

    let out = dynamic_slice(&input, &starts, &[2, 2]);

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64(&out, &[0, 0]), 11.0);
    assert_eq!(get_f64(&out, &[1, 0]), 12.0);
    assert_eq!(get_f64(&out, &[0, 1]), 15.0);
    assert_eq!(get_f64(&out, &[1, 1]), 16.0);
}
