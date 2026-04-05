use num_complex::Complex64;

use crate::backend::TensorBackend;
use crate::config::DotGeneralConfig;
use crate::cpu::{
    add, broadcast_in_dim, conj, embed_diagonal, extract_diagonal, mul, neg, reduce_sum, reshape,
    transpose, CpuBackend,
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

fn assert_c64_close(actual: Complex64, expected: Complex64) {
    assert_f64_close(actual.re, expected.re);
    assert_f64_close(actual.im, expected.im);
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
    assert_eq!(all.shape(), &[1]);
    assert_eq!(get_f64(&all, &[0]), 21.0);
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
    let scalar = Tensor::F64(TypedTensor::from_vec(vec![1], vec![5.0]));
    let broadcast = broadcast_in_dim(&scalar, &[3], &[0]);
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
