use num_complex::Complex64;
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro::traced_tensor::{cholesky, qr, solve, svd, triangular_solve};
use tenferro_tensor::{cpu::CpuBackend, Tensor, TensorBackend, TypedTensor};

const TOL: f64 = 1.0e-9;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected f64 tensor"),
    }
}

fn c64_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec(shape, data))
}

fn get_c64_data(tensor: &Tensor) -> &[Complex64] {
    match tensor {
        Tensor::C64(inner) => inner.host_data(),
        _ => panic!("expected c64 tensor"),
    }
}

fn eval(traced: TracedTensor) -> Tensor {
    let mut engine = Engine::new(CpuBackend::new());
    let mut traced = traced;
    traced.eval(&mut engine).unwrap().clone()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= TOL,
            "index {index}: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

fn assert_close_c64(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).norm();
        assert!(
            diff <= TOL,
            "index {index}: actual={actual:?}, expected={expected:?}, diff={diff}"
        );
    }
}

fn finite_diff_scalar(f: impl Fn(&[f64]) -> f64, x: &[f64], index: usize, h: f64) -> f64 {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[index] += h;
    xm[index] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

fn wide_qr_r_sum(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 5], data.to_vec());
    let outputs = TensorBackend::qr(&mut backend, &input).unwrap();
    get_f64_data(&outputs[1]).iter().sum()
}

#[test]
fn solve_supports_batched_vector_rhs() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![
            2.0, 0.0, 0.0, 3.0, //
            4.0, 0.0, 0.0, 5.0,
        ],
    ));
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![4.0, 9.0, 8.0, 20.0]));

    let out = eval(solve(&a, &b));

    assert_eq!(out.shape(), &[2, 2]);
    assert_close(get_f64_data(&out), &[2.0, 3.0, 2.0, 4.0]);
}

#[test]
fn triangular_solve_supports_batched_vector_rhs() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![
            2.0, 0.0, 0.0, 3.0, //
            4.0, 0.0, 0.0, 5.0,
        ],
    ));
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![4.0, 9.0, 8.0, 20.0]));

    let out = eval(triangular_solve(&a, &b, true, true, false, false));

    assert_eq!(out.shape(), &[2, 2]);
    assert_close(get_f64_data(&out), &[2.0, 3.0, 2.0, 4.0]);
}

#[test]
fn batched_cholesky_jvp_on_diagonal_matrices() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![
            4.0, 0.0, 0.0, 9.0, //
            1.0, 0.0, 0.0, 16.0,
        ],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![
            2.0, 0.0, 0.0, 8.0, //
            6.0, 0.0, 0.0, 32.0,
        ],
    ));

    let dl = cholesky(&a).jvp(&a, &da);
    let out = eval(dl);

    assert_eq!(out.shape(), &[2, 2, 2]);
    assert_close(
        get_f64_data(&out),
        &[0.5, 0.0, 0.0, 4.0 / 3.0, 3.0, 0.0, 0.0, 4.0],
    );
}

#[test]
fn batched_qr_diag_r_sum_jvp_on_diagonal_matrices() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![
            4.0, 0.0, 0.0, 9.0, //
            1.0, 0.0, 0.0, 16.0,
        ],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![
            2.0, 0.0, 0.0, 8.0, //
            6.0, 0.0, 0.0, 32.0,
        ],
    ));

    let (_q, r) = qr(&a);
    let y = r.extract_diag(0, 1).reduce_sum(&[0]);
    let dy = y.jvp(&a, &da);
    let out = eval(dy);

    assert_eq!(out.shape(), &[2]);
    assert_close(get_f64_data(&out), &[10.0, 38.0]);
}

#[test]
fn batched_rectangular_svd_singular_value_sum_jvp() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2, 2],
        vec![
            9.0, 0.0, 0.0, 0.0, 4.0, 0.0, //
            16.0, 0.0, 0.0, 0.0, 25.0, 0.0,
        ],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2, 2],
        vec![
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, //
            3.0, 0.0, 0.0, 0.0, 5.0, 0.0,
        ],
    ));

    let (_u, s, _vt) = svd(&a);
    let y = s.reduce_sum(&[0]);
    let dy = y.jvp(&a, &da);
    let out = eval(dy);

    assert_eq!(out.shape(), &[2]);
    assert_close(get_f64_data(&out), &[3.0, 8.0]);
}

#[test]
fn unit_diagonal_triangular_solve_jvp_ignores_diagonal_tangent() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 3],
        vec![
            1.0, 0.0, 0.0, //
            2.0, 1.0, 0.0, //
            3.0, 4.0, 1.0,
        ],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 1], vec![1.0, 2.0, 3.0]));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 3],
        vec![
            7.0, 0.0, 0.0, //
            5.0, 11.0, 0.0, //
            17.0, 19.0, 13.0,
        ],
    ));

    let x = triangular_solve(&a, &b, true, true, false, true);
    let dx = x.jvp(&a, &da);
    let out = eval(dx);

    assert_eq!(out.shape(), &[3, 1]);
    assert_close(get_f64_data(&out), &[0.0, 0.0, 0.0]);
}

#[test]
fn transpose_triangular_solve_jvp_uses_transpose_not_adjoint_for_complex_inputs() {
    let a = TracedTensor::from_tensor_concrete_shape(c64_tensor(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(c64_tensor(
        vec![2, 1],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(c64_tensor(
        vec![2, 2],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    ));

    let x = triangular_solve(&a, &b, true, true, true, true);
    let dx = x.jvp(&a, &da);
    let out = eval(dx);

    assert_eq!(out.shape(), &[2, 1]);
    assert_close_c64(
        get_c64_data(&out),
        &[Complex64::new(0.0, -1.0), Complex64::new(0.0, 0.0)],
    );
}

#[test]
fn rectangular_qr_diag_r_sum_jvp_matches_diagonal_input() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5, 2],
        vec![
            4.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, 0.0,
        ],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5, 2],
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, 0.0,
        ],
    ));

    let (_q, r) = qr(&a);
    let y = r.extract_diag(0, 1).reduce_sum(&[0]);
    let dy = y.jvp(&a, &da);
    let out = eval(dy);

    assert!(out.shape().is_empty());
    assert_close(get_f64_data(&out), &[3.0]);
}

#[test]
fn wide_qr_r_sum_jvp_matches_diagonal_and_tail_input() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 5],
        vec![
            4.0, 0.0, //
            0.0, 3.0, //
            0.0, 0.0, //
            0.0, 0.0, //
            0.0, 0.0,
        ],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 5],
        vec![
            1.5, 0.0, //
            0.0, 2.5, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0,
        ],
    ));

    let (_q, r) = qr(&a);
    let y = r.reduce_sum(&[0, 1]);
    let dy = y.jvp(&a, &da);
    let out = eval(dy);

    assert!(out.shape().is_empty());
    assert_close(get_f64_data(&out), &[37.0]);
}

#[test]
fn wide_qr_r_sum_grad_matches_finite_difference() {
    let a_data = vec![
        -0.5122127092688165,
        -0.7943778587822452,
        -0.10413653078009572,
        -0.19003586984572776,
        -0.12419960983007988,
        -0.4892132325182914,
        -0.6516216941236995,
        -0.5179743517276878,
        -0.45892809719604577,
        0.7718872740654947,
    ];
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 5], a_data.clone()));
    let (_q, r) = qr(&a);
    let loss = r.reduce_sum(&[0, 1]);
    let grad = eval(loss.grad(&a).unwrap());
    let grad_data = get_f64_data(&grad);

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(wide_qr_r_sum, &a_data, index, 1.0e-6);
        let actual = grad_data[index];
        assert!(
            (actual - expected).abs() <= 1.0e-4,
            "index {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn zero_sized_qr_outputs_expected_shapes() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![0, 5], Vec::new()));
    let (q, r) = qr(&a);
    let q_out = eval(q);
    let r_out = eval(r);

    assert_eq!(q_out.shape(), &[0, 0]);
    assert!(get_f64_data(&q_out).is_empty());
    assert_eq!(r_out.shape(), &[0, 5]);
    assert!(get_f64_data(&r_out).is_empty());
}

#[test]
fn zero_sized_svd_outputs_expected_shapes() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5, 0], Vec::new()));
    let (u, s, vh) = svd(&a);
    let u_out = eval(u);
    let s_out = eval(s);
    let vh_out = eval(vh);

    assert_eq!(u_out.shape(), &[5, 0]);
    assert!(get_f64_data(&u_out).is_empty());
    assert_eq!(s_out.shape(), &[0]);
    assert!(get_f64_data(&s_out).is_empty());
    assert_eq!(vh_out.shape(), &[0, 0]);
    assert!(get_f64_data(&vh_out).is_empty());
}

#[test]
fn zero_sized_batched_qr_outputs_expected_shapes() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 5, 0], Vec::new()));
    let (q, r) = qr(&a);
    let q_out = eval(q);
    let r_out = eval(r);

    assert_eq!(q_out.shape(), &[2, 2, 0]);
    assert!(get_f64_data(&q_out).is_empty());
    assert_eq!(r_out.shape(), &[2, 5, 0]);
    assert!(get_f64_data(&r_out).is_empty());
}

#[test]
fn zero_sized_batched_svd_outputs_expected_shapes() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5, 2, 0], Vec::new()));
    let (u, s, vh) = svd(&a);
    let u_out = eval(u);
    let s_out = eval(s);
    let vh_out = eval(vh);

    assert_eq!(u_out.shape(), &[5, 2, 0]);
    assert!(get_f64_data(&u_out).is_empty());
    assert_eq!(s_out.shape(), &[2, 0]);
    assert!(get_f64_data(&s_out).is_empty());
    assert_eq!(vh_out.shape(), &[2, 2, 0]);
    assert!(get_f64_data(&vh_out).is_empty());
}

#[test]
fn zero_sized_cholesky_and_solve_outputs_expected_shapes() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![0, 0], Vec::new()));
    let l_out = eval(cholesky(&a));
    assert_eq!(l_out.shape(), &[0, 0]);
    assert!(get_f64_data(&l_out).is_empty());

    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![0, 3], Vec::new()));
    let x_out = eval(solve(&a, &b));
    assert_eq!(x_out.shape(), &[0, 3]);
    assert!(get_f64_data(&x_out).is_empty());

    let tx_out = eval(triangular_solve(&a, &b, true, true, false, false));
    assert_eq!(tx_out.shape(), &[0, 3]);
    assert!(get_f64_data(&tx_out).is_empty());
}

#[test]
fn zero_sized_batched_vector_rhs_solve_outputs_expected_shapes() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2, 0], Vec::new()));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 0], Vec::new()));

    let x_out = eval(solve(&a, &b));
    assert_eq!(x_out.shape(), &[2, 0]);
    assert!(get_f64_data(&x_out).is_empty());

    let tx_out = eval(triangular_solve(&a, &b, true, true, false, false));
    assert_eq!(tx_out.shape(), &[2, 0]);
    assert!(get_f64_data(&tx_out).is_empty());
}

#[test]
fn zero_sized_upper_cholesky_ad_outputs_expected_shapes() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![0, 0], Vec::new()));
    let direction = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![0, 0], Vec::new()));
    let cotangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![0, 0], Vec::new()));

    let factor = cholesky(&a).transpose(&[1, 0]);

    let jvp_out = eval(
        factor
            .try_jvp(&a, &direction)
            .expect("zero-sized cholesky JVP should remain active"),
    );
    assert_eq!(jvp_out.shape(), &[0, 0]);
    assert!(get_f64_data(&jvp_out).is_empty());

    let vjp = factor.vjp(&a, &cotangent);
    let vjp_out = eval(vjp.clone());
    assert_eq!(vjp_out.shape(), &[0, 0]);
    assert!(get_f64_data(&vjp_out).is_empty());

    let hvp_out = eval(
        vjp.try_jvp(&a, &direction)
            .expect("zero-sized cholesky HVP should remain active"),
    );
    assert_eq!(hvp_out.shape(), &[0, 0]);
    assert!(get_f64_data(&hvp_out).is_empty());
}

#[test]
fn zero_sized_batched_upper_cholesky_ad_outputs_expected_shapes() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2, 0], Vec::new()));
    let direction = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2, 0], Vec::new()));
    let cotangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2, 0], Vec::new()));

    let factor = cholesky(&a).transpose(&[1, 0, 2]);

    let jvp_out = eval(
        factor
            .try_jvp(&a, &direction)
            .expect("zero-sized batched cholesky JVP should remain active"),
    );
    assert_eq!(jvp_out.shape(), &[2, 2, 0]);
    assert!(get_f64_data(&jvp_out).is_empty());

    let vjp = factor.vjp(&a, &cotangent);
    let vjp_out = eval(vjp.clone());
    assert_eq!(vjp_out.shape(), &[2, 2, 0]);
    assert!(get_f64_data(&vjp_out).is_empty());

    let hvp_out = eval(
        vjp.try_jvp(&a, &direction)
            .expect("zero-sized batched cholesky HVP should remain active"),
    );
    assert_eq!(hvp_out.shape(), &[2, 2, 0]);
    assert!(get_f64_data(&hvp_out).is_empty());
}

#[test]
fn try_grad_returns_none_for_inactive_scalar_output() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![1], vec![3.0]));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![1], vec![4.0]));
    let loss = x.reduce_sum(&[0]);

    assert!(loss.try_grad(&y).unwrap().is_none());
}
