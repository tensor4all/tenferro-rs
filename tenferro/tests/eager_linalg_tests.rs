use num_complex::Complex64;
use std::sync::{Arc, OnceLock};
use tenferro::{
    CpuBackend, DType, DotGeneralConfig, EagerRuntime, EagerTensor, ScatterConfig, Tensor,
};

fn test_ctx() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::with_cpu_backend(CpuBackend::new()))
        .clone()
}

const LINALG_TOL: f64 = 1.0e-7;
const ELEMENTWISE_TOL: f64 = 1.0e-8;

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn c64_data(tensor: &Tensor) -> &[Complex64] {
    tensor.as_slice::<Complex64>().unwrap()
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_all_finite(values: &[f64]) {
    for &value in values {
        assert!(value.is_finite(), "encountered non-finite value: {value}");
    }
}

fn matmul(lhs: &EagerTensor, rhs: &EagerTensor) -> EagerTensor {
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    )
    .unwrap()
}

fn transpose(matrix: &EagerTensor) -> EagerTensor {
    matrix.transpose(&[1, 0]).unwrap()
}

fn reduce_all(tensor: &EagerTensor) -> EagerTensor {
    let axes: Vec<usize> = (0..tensor.data().shape().len()).collect();
    tensor.reduce_sum(&axes).unwrap()
}

#[test]
fn eager_linalg_svd_reconstructs_input() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![2, 2], vec![3.0_f64, 0.5, 1.0, 2.0]),
        test_ctx(),
    );
    let (u, s, vh) = a.svd().unwrap();

    let sigma = s.embed_diag(0, 1).unwrap();
    let reconstruction = matmul(&matmul(&u, &sigma), &vh);

    assert_close_slice(
        f64_data(reconstruction.data()),
        f64_data(a.data()),
        LINALG_TOL,
    );
}

#[test]
fn eager_linalg_svd_backward_is_finite() {
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![2, 2], vec![3.0_f64, 0.5, 1.0, 2.0]),
        test_ctx(),
    );
    let (_, s, _) = a.svd().unwrap();
    let loss = reduce_all(&s);

    let _ = loss.backward().unwrap();

    let grad = a.grad().unwrap();
    assert_eq!(grad.shape(), &[2, 2]);
    assert_all_finite(f64_data(grad.as_ref()));
}

#[test]
fn eager_linalg_qr_reconstructs_input() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3, 2], vec![3.0_f64, 1.0, 2.0, 2.0, 0.0, 1.0]),
        test_ctx(),
    );
    let (q, r) = a.qr().unwrap();

    let reconstruction = matmul(&q, &r);

    assert_close_slice(
        f64_data(reconstruction.data()),
        f64_data(a.data()),
        LINALG_TOL,
    );
}

#[test]
fn eager_linalg_qr_backward_is_finite() {
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![3, 2], vec![3.0_f64, 1.0, 2.0, 2.0, 0.0, 1.0]),
        test_ctx(),
    );
    let (_, r) = a.qr().unwrap();
    let loss = reduce_all(&r);

    let _ = loss.backward().unwrap();

    let grad = a.grad().unwrap();
    assert_eq!(grad.shape(), &[3, 2]);
    assert_all_finite(f64_data(grad.as_ref()));
}

#[test]
fn eager_linalg_lu_reconstructs_permuted_input() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]),
        test_ctx(),
    );
    let (p, l, u, _) = a.lu().unwrap();

    let pa = matmul(&p, &a);
    let lu = matmul(&l, &u);

    assert_close_slice(f64_data(pa.data()), f64_data(lu.data()), LINALG_TOL);
}

#[test]
fn eager_linalg_lu_backward_is_finite() {
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]),
        test_ctx(),
    );
    let (_, l, u, _) = a.lu().unwrap();
    let l_sum = reduce_all(&l);
    let u_sum = reduce_all(&u);
    let loss = &l_sum + &u_sum;

    let _ = loss.backward().unwrap();

    let grad = a.grad().unwrap();
    assert_eq!(grad.shape(), &[2, 2]);
    assert_all_finite(f64_data(grad.as_ref()));
}

#[test]
fn eager_linalg_cholesky_reconstructs_input() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]),
        test_ctx(),
    );
    let l = a.cholesky().unwrap();

    let reconstruction = matmul(&l, &transpose(&l));

    assert_close_slice(
        f64_data(reconstruction.data()),
        f64_data(a.data()),
        LINALG_TOL,
    );
}

#[test]
fn eager_linalg_cholesky_backward_is_finite() {
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]),
        test_ctx(),
    );
    let l = a.cholesky().unwrap();
    let loss = reduce_all(&l);

    let _ = loss.backward().unwrap();

    let grad = a.grad().unwrap();
    assert_eq!(grad.shape(), &[2, 2]);
    assert_all_finite(f64_data(grad.as_ref()));
}

#[test]
fn eager_linalg_eigh_reconstructs_input() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]),
        test_ctx(),
    );
    let (values, vectors) = a.eigh().unwrap();

    let diagonal = values.embed_diag(0, 1).unwrap();
    let reconstruction = matmul(&matmul(&vectors, &diagonal), &transpose(&vectors));

    assert_close_slice(
        f64_data(reconstruction.data()),
        f64_data(a.data()),
        LINALG_TOL,
    );
}

#[test]
fn eager_linalg_eigh_backward_is_finite() {
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]),
        test_ctx(),
    );
    let (values, _) = a.eigh().unwrap();
    let loss = reduce_all(&values);

    let _ = loss.backward().unwrap();

    let grad = a.grad().unwrap();
    assert_eq!(grad.shape(), &[2, 2]);
    assert_all_finite(f64_data(grad.as_ref()));
}

#[test]
fn eager_elementwise_forward_surface_smoke() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![8.0_f64, -6.0, 9.0]),
        test_ctx(),
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![2.0_f64, 3.0, 3.0]),
        test_ctx(),
    );
    assert_close_slice(
        f64_data(x.div(&y).unwrap().data()),
        &[4.0, -2.0, 3.0],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(x.abs().unwrap().data()),
        &[8.0, 6.0, 9.0],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(x.sign().unwrap().data()),
        &[1.0, -1.0, 1.0],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(x.maximum(&y).unwrap().data()),
        &[8.0, 3.0, 9.0],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(x.minimum(&y).unwrap().data()),
        &[2.0, -6.0, 3.0],
        ELEMENTWISE_TOL,
    );

    let positive = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![2], vec![1.0_f64, std::f64::consts::E]),
        test_ctx(),
    );
    assert_close_slice(
        f64_data(positive.log().unwrap().data()),
        &[0.0, 1.0],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(positive.sqrt().unwrap().data()),
        &[1.0, std::f64::consts::E.sqrt()],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(positive.rsqrt().unwrap().data()),
        &[1.0, 1.0 / std::f64::consts::E.sqrt()],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(positive.log1p().unwrap().data()),
        &[2.0_f64.ln(), (1.0 + std::f64::consts::E).ln()],
        ELEMENTWISE_TOL,
    );

    let angles = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![2], vec![0.0_f64, std::f64::consts::FRAC_PI_2]),
        test_ctx(),
    );
    assert_close_slice(
        f64_data(angles.sin().unwrap().data()),
        &[0.0, 1.0],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(angles.cos().unwrap().data()),
        &[1.0, 0.0],
        ELEMENTWISE_TOL,
    );

    let tanh_input =
        EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![0.0_f64, 1.0]), test_ctx());
    assert_close_slice(
        f64_data(tanh_input.tanh().unwrap().data()),
        &[0.0, 1.0_f64.tanh()],
        ELEMENTWISE_TOL,
    );
    assert_close_slice(
        f64_data(tanh_input.expm1().unwrap().data()),
        &[0.0, 1.0_f64.exp_m1()],
        ELEMENTWISE_TOL,
    );

    let pow_base = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![8.0_f64, 9.0, 4.0]),
        test_ctx(),
    );
    let exponents = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![3.0_f64, 0.5, 2.0]),
        test_ctx(),
    );
    assert_close_slice(
        f64_data(pow_base.pow(&exponents).unwrap().data()),
        &[512.0, 3.0, 16.0],
        ELEMENTWISE_TOL,
    );

    let condition = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![0.0_f64, -1.0, 2.0]),
        test_ctx(),
    );
    let on_true = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]),
        test_ctx(),
    );
    let on_false = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]),
        test_ctx(),
    );
    assert_close_slice(
        f64_data(
            EagerTensor::select(&condition, &on_true, &on_false)
                .unwrap()
                .data(),
        ),
        &[1.0, 20.0, 30.0],
        ELEMENTWISE_TOL,
    );

    let complex = EagerTensor::from_tensor_in(
        Tensor::from_vec(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        ),
        test_ctx(),
    );
    assert_eq!(
        c64_data(complex.conj().unwrap().data()),
        &[Complex64::new(1.0, -2.0), Complex64::new(-3.0, -0.5)]
    );

    let converted = positive.convert(DType::C64).unwrap();
    assert_eq!(converted.data().dtype(), DType::C64);
    assert_eq!(
        c64_data(converted.data()),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(std::f64::consts::E, 0.0)
        ]
    );
}

#[test]
fn eager_elementwise_scatter_forward_updates_operand() {
    let operand = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![4], vec![0.0_f64, 0.0, 0.0, 0.0]),
        test_ctx(),
    );
    let indices =
        EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 1], vec![1_i64, 3]), test_ctx());
    let updates =
        EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![5.0_f64, 4.0]), test_ctx());
    let result = operand
        .scatter(
            &indices,
            &updates,
            ScatterConfig {
                update_window_dims: vec![],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            },
        )
        .unwrap();

    assert_close_slice(
        f64_data(result.data()),
        &[0.0, 5.0, 0.0, 4.0],
        ELEMENTWISE_TOL,
    );
}
