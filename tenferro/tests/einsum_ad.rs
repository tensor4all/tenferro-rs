use num_complex::Complex64;
use tenferro::einsum::einsum;
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{Tensor, TypedTensor};

const TOL: f64 = 1e-6;
const COMPLEX_TOL: f64 = 2e-5;

fn finite_diff_scalar(f: impl Fn(&[f64]) -> f64, x: &[f64], idx: usize, h: f64) -> f64 {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[idx] += h;
    xm[idx] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

fn finite_diff_complex(
    f: impl Fn(&[Complex64]) -> f64,
    x: &[Complex64],
    idx: usize,
    h: f64,
) -> Complex64 {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[idx] += Complex64::new(h, 0.0);
    xm[idx] -= Complex64::new(h, 0.0);
    let df_dre = (f(&xp) - f(&xm)) / (2.0 * h);

    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[idx] += Complex64::new(0.0, h);
    xm[idx] -= Complex64::new(0.0, h);
    let df_dim = (f(&xp) - f(&xm)) / (2.0 * h);

    Complex64::new(df_dre, df_dim)
}

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn c64_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected f64 tensor"),
    }
}

fn get_c64_data(tensor: &Tensor) -> &[Complex64] {
    match tensor {
        Tensor::C64(inner) => inner.host_data(),
        _ => panic!("expected c64 tensor"),
    }
}

fn eval_tensor(mut traced: TracedTensor) -> Tensor {
    let mut engine = Engine::new(CpuBackend::new());
    traced.eval(&mut engine).unwrap().clone()
}

fn eval_scalar(traced: TracedTensor) -> f64 {
    get_f64_data(&eval_tensor(traced))[0]
}

fn eval_real_c64_scalar(traced: TracedTensor) -> f64 {
    let tensor = eval_tensor(traced);
    let value = get_c64_data(&tensor)[0];
    assert!(
        value.im.abs() <= 1e-9,
        "expected real-valued complex scalar loss, got {value:?}"
    );
    value.re
}

fn traced_real_scalar(input: &TracedTensor) -> TracedTensor {
    let half = TracedTensor::from_tensor(c64_tensor(vec![], vec![Complex64::new(0.5, 0.0)]));
    input.traced_add(&input.traced_conj()).traced_mul(&half)
}

fn col_major_index(rows: usize, i: usize, j: usize) -> usize {
    i + rows * j
}

fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                sum += lhs[col_major_index(m, i, p)] * rhs[col_major_index(k, p, j)];
            }
            out[col_major_index(m, i, j)] = sum;
        }
    }
    out
}

fn transpose_f64(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = data[col_major_index(rows, i, j)];
        }
    }
    out
}

fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_grad_matches_finite_diff(actual: &[f64], base: &[f64], f: impl Fn(&[f64]) -> f64) {
    assert_eq!(actual.len(), base.len());
    for index in 0..base.len() {
        let expected = finite_diff_scalar(&f, base, index, 1e-6);
        assert!(
            (actual[index] - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            actual[index]
        );
    }
}

fn assert_grad_matches_complex_finite_diff(
    actual: &[Complex64],
    base: &[Complex64],
    f: impl Fn(&[Complex64]) -> f64,
) {
    assert_eq!(actual.len(), base.len());
    for index in 0..base.len() {
        let expected = finite_diff_complex(&f, base, index, 1e-6);
        assert!(
            (actual[index] - expected).norm() <= COMPLEX_TOL,
            "index {index}: expected {expected:?}, got {:?}",
            actual[index]
        );
    }
}

#[test]
fn grad_einsum_matmul_real() {
    let a_data = vec![1.0, -2.0, 0.5, 3.0, 1.25, -0.75];
    let b_data = vec![2.0, 0.25, -1.5, 4.0, 0.75, -0.5];

    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], a_data.clone()));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
    let mut engine = Engine::new(CpuBackend::new());
    let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
    let grad_a = y.traced_reduce_sum(&[0, 1]).grad(&a).unwrap();

    let grad_a_tensor = eval_tensor(grad_a);
    let grad_a_data = get_f64_data(&grad_a_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], xs.to_vec()));
        let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
        let mut engine = Engine::new(CpuBackend::new());
        let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
        eval_scalar(y.traced_reduce_sum(&[0, 1]))
    };
    assert_grad_matches_finite_diff(grad_a_data, &a_data, f);
}

#[test]
fn grad_einsum_trace_real() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let mut engine = Engine::new(CpuBackend::new());
    let loss = einsum(&mut engine, &[&a], "ii->").unwrap();
    let grad = loss.grad(&a).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn grad_einsum_matmul_complex() {
    let a_data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.25, 1.25),
        Complex64::new(0.75, -1.0),
        Complex64::new(2.0, 0.1),
    ];
    let b_data = vec![
        Complex64::new(0.5, -0.5),
        Complex64::new(1.5, 0.25),
        Complex64::new(-1.0, 0.75),
        Complex64::new(0.25, -1.5),
    ];

    let a = TracedTensor::from_tensor(c64_tensor(vec![2, 2], a_data.clone()));
    let b = TracedTensor::from_tensor(c64_tensor(vec![2, 2], b_data.clone()));
    let mut engine = Engine::new(CpuBackend::new());
    let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
    let norm_sq = y.traced_conj().traced_mul(&y).traced_reduce_sum(&[0, 1]);
    let loss = traced_real_scalar(&norm_sq);
    let grad_a = loss.grad(&a).unwrap();

    let grad_a_tensor = eval_tensor(grad_a);
    let grad_a_data = get_c64_data(&grad_a_tensor);

    let f = |xs: &[Complex64]| {
        let a = TracedTensor::from_tensor(c64_tensor(vec![2, 2], xs.to_vec()));
        let b = TracedTensor::from_tensor(c64_tensor(vec![2, 2], b_data.clone()));
        let mut engine = Engine::new(CpuBackend::new());
        let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
        let norm_sq = y.traced_conj().traced_mul(&y).traced_reduce_sum(&[0, 1]);
        eval_real_c64_scalar(traced_real_scalar(&norm_sq))
    };
    assert_grad_matches_complex_finite_diff(grad_a_data, &a_data, f);
}

#[test]
fn jvp_einsum_matmul() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b_data = vec![0.5, -1.0, 2.0, 1.5, -0.25, 3.0];
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
    let da_data = vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.0];
    let da = TracedTensor::from_tensor(f64_tensor(vec![2, 3], da_data.clone()));

    let mut engine = Engine::new(CpuBackend::new());
    let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
    let dy = y.jvp(&a, &da);

    let result = eval_tensor(dy);
    let expected = matmul_f64(&da_data, &b_data, 2, 3, 2);
    assert_close_slice(get_f64_data(&result), &expected);
}

#[test]
fn vjp_einsum_matmul() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b_data = vec![0.5, -1.0, 2.0, 1.5, -0.25, 3.0];
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
    let ct_data = vec![1.0, -0.5, 0.25, 2.0];
    let cotangent = TracedTensor::from_tensor(f64_tensor(vec![2, 2], ct_data.clone()));

    let mut engine = Engine::new(CpuBackend::new());
    let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
    let ct_a = y.vjp(&a, &cotangent);

    let result = eval_tensor(ct_a);
    let expected = matmul_f64(&ct_data, &transpose_f64(&b_data, 3, 2), 2, 2, 3);
    assert_close_slice(get_f64_data(&result), &expected);
}

#[test]
fn grad_einsum_three_way() {
    let a_data = vec![1.0, -0.5, 2.0, 0.75];
    let b_data = vec![0.5, 1.5, -1.0, 0.25];
    let c_data = vec![2.0, -1.5, 0.75, 1.25];

    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], a_data.clone()));
    let b = TracedTensor::from_tensor(f64_tensor(vec![2, 2], b_data.clone()));
    let c = TracedTensor::from_tensor(f64_tensor(vec![2, 2], c_data.clone()));
    let mut engine = Engine::new(CpuBackend::new());
    let y = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
    let grad_a = y.traced_reduce_sum(&[0, 1]).grad(&a).unwrap();

    let grad_a_tensor = eval_tensor(grad_a);
    let grad_a_data = get_f64_data(&grad_a_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], xs.to_vec()));
        let b = TracedTensor::from_tensor(f64_tensor(vec![2, 2], b_data.clone()));
        let c = TracedTensor::from_tensor(f64_tensor(vec![2, 2], c_data.clone()));
        let mut engine = Engine::new(CpuBackend::new());
        let y = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
        eval_scalar(y.traced_reduce_sum(&[0, 1]))
    };
    assert_grad_matches_finite_diff(grad_a_data, &a_data, f);
}
