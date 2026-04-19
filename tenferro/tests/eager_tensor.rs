use num_complex::Complex64;
use tenferro::{
    CpuBackend, DotGeneralConfig, EagerContext, EagerTensor, GatherConfig, PadConfig, SliceConfig,
    Tensor,
};

const FD_H: f64 = 1.0e-6;
const TOL: f64 = 1.0e-5;
const FD_TOL: f64 = 1.0e-4;

fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn c64_data(tensor: &Tensor) -> &[Complex64] {
    tensor.as_slice::<Complex64>().unwrap()
}

fn assert_send_sync<T: Send + Sync>() {}

fn finite_diff_scalar(f: impl Fn(&[f64]) -> f64, x: &[f64], index: usize) -> f64 {
    let mut plus = x.to_vec();
    let mut minus = x.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(&plus) - f(&minus)) / (2.0 * FD_H)
}

fn finite_diff_lhs(
    f: impl Fn(&[f64], &[f64]) -> f64,
    lhs: &[f64],
    rhs: &[f64],
    index: usize,
) -> f64 {
    let mut plus = lhs.to_vec();
    let mut minus = lhs.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(&plus, rhs) - f(&minus, rhs)) / (2.0 * FD_H)
}

fn finite_diff_rhs(
    f: impl Fn(&[f64], &[f64]) -> f64,
    lhs: &[f64],
    rhs: &[f64],
    index: usize,
) -> f64 {
    let mut plus = rhs.to_vec();
    let mut minus = rhs.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(lhs, &plus) - f(lhs, &minus)) / (2.0 * FD_H)
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn eager_matmul_sum(lhs: &[f64], rhs: &[f64]) -> f64 {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 3], lhs.to_vec()));
    let b = EagerTensor::from_tensor(Tensor::from_vec(vec![3, 2], rhs.to_vec()));
    let loss = a
        .dot_general(&b, matmul_config())
        .unwrap()
        .reduce_sum(&[0, 1])
        .unwrap();
    f64_data(loss.data())[0]
}

#[test]
fn eager_x_squared_gradient_matches_finite_difference() {
    let x_data = vec![1.0, 2.0, 3.0];
    let x = EagerTensor::requires_grad(Tensor::from_vec(vec![3], x_data.clone()));
    let loss = (&x * &x).reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();
    let grad = x.grad().unwrap();

    let grad_data = f64_data(grad.as_ref());
    let expected: Vec<f64> = (0..x_data.len())
        .map(|index| {
            finite_diff_scalar(
                |values| values.iter().map(|v| v * v).sum::<f64>(),
                &x_data,
                index,
            )
        })
        .collect();
    assert_close_slice(grad_data, &expected, FD_TOL);
}

#[test]
fn eager_repeated_backward_accumulates_across_calls() {
    let x = EagerTensor::requires_grad(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]));

    let loss = (&x * &x).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert_close_slice(f64_data(x.grad().unwrap().as_ref()), &[2.0, 4.0, 6.0], TOL);

    let loss = (&x * &x).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert_close_slice(f64_data(x.grad().unwrap().as_ref()), &[4.0, 8.0, 12.0], TOL);
}

#[test]
fn eager_matmul_gradients_match_finite_difference() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a = EagerTensor::requires_grad(Tensor::from_vec(vec![2, 3], a_data.clone()));
    let b = EagerTensor::requires_grad(Tensor::from_vec(vec![3, 2], b_data.clone()));
    let loss = a
        .dot_general(&b, matmul_config())
        .unwrap()
        .reduce_sum(&[0, 1])
        .unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();
    let grad_a_data = f64_data(grad_a.as_ref());
    let grad_b_data = f64_data(grad_b.as_ref());

    let expected_a: Vec<f64> = (0..a_data.len())
        .map(|index| finite_diff_lhs(eager_matmul_sum, &a_data, &b_data, index))
        .collect();
    let expected_b: Vec<f64> = (0..b_data.len())
        .map(|index| finite_diff_rhs(eager_matmul_sum, &a_data, &b_data, index))
        .collect();

    assert_close_slice(grad_a_data, &expected_a, FD_TOL);
    assert_close_slice(grad_b_data, &expected_b, FD_TOL);
}

#[test]
fn eager_exp_gradient_matches_primal() {
    let x = EagerTensor::requires_grad(Tensor::from_vec(vec![3], vec![0.0, 1.0, 2.0]));
    let loss = x.exp().unwrap().reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    let expected = vec![1.0, 1.0_f64.exp(), 2.0_f64.exp()];
    assert_close_slice(f64_data(grad.as_ref()), &expected, TOL);
}

#[test]
fn eager_fan_out_accumulates_gradient() {
    let x = EagerTensor::requires_grad(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let loss = (&x + &x).reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[2.0, 2.0, 2.0], TOL);
}

#[test]
fn eager_clear_grad_resets_only_one_leaf() {
    let ctx = EagerContext::with_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]),
        ctx.clone(),
    );

    let loss = (&x * &y).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    x.clear_grad();

    assert!(x.grad().is_none());
    assert_close_slice(f64_data(y.grad().unwrap().as_ref()), &[1.0, 2.0, 3.0], TOL);

    let loss = (&x * &x).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(f64_data(x.grad().unwrap().as_ref()), &[2.0, 4.0, 6.0], TOL);
    assert_close_slice(f64_data(y.grad().unwrap().as_ref()), &[1.0, 2.0, 3.0], TOL);
}

#[test]
fn eager_context_clear_grads_resets_all_live_leaves() {
    let ctx = EagerContext::with_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]),
        ctx.clone(),
    );

    let loss = (&x * &y).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    ctx.clear_grads();

    assert!(x.grad().is_none());
    assert!(y.grad().is_none());

    let loss = (&x * &y).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(f64_data(x.grad().unwrap().as_ref()), &[4.0, 5.0, 6.0], TOL);
    assert_close_slice(f64_data(y.grad().unwrap().as_ref()), &[1.0, 2.0, 3.0], TOL);
}

#[test]
fn eager_unrelated_backward_keeps_existing_leaf_grad() {
    let ctx = EagerContext::with_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]),
        ctx.clone(),
    );

    let loss_x = (&x * &x).reduce_sum(&[0]).unwrap();
    let _ = loss_x.backward().unwrap();
    assert_close_slice(f64_data(x.grad().unwrap().as_ref()), &[2.0, 4.0, 6.0], TOL);

    let loss_y = (&y * &y).reduce_sum(&[0]).unwrap();
    let _ = loss_y.backward().unwrap();

    assert_close_slice(f64_data(x.grad().unwrap().as_ref()), &[2.0, 4.0, 6.0], TOL);
    assert_close_slice(
        f64_data(y.grad().unwrap().as_ref()),
        &[8.0, 10.0, 12.0],
        TOL,
    );
}

#[test]
fn eager_tracks_grad_reports_leaf_state() {
    let ctx = EagerContext::with_backend(CpuBackend::new());
    let plain = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let leaf =
        EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]), ctx);

    assert!(!plain.tracks_grad());
    assert!(leaf.tracks_grad());
    assert!(!leaf.detach().tracks_grad());
}

#[test]
fn eager_send_sync_contracts_compile() {
    assert_send_sync::<EagerTensor<CpuBackend>>();
    assert_send_sync::<EagerContext<CpuBackend>>();
}

#[test]
fn eager_detach_cuts_one_gradient_path() {
    let x = EagerTensor::requires_grad(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let detached = x.detach();
    let loss = (&detached * &x).reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[1.0, 2.0, 3.0], TOL);
    assert!(detached.grad().is_none());
}

#[test]
fn eager_untracked_tensor_behaves_like_plain_tensor() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let y = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]));
    let z = &x * &y;

    assert_close_slice(f64_data(z.data()), &[4.0, 10.0, 18.0], TOL);
    assert!(x.grad().is_none());
    assert!(y.grad().is_none());
    assert!(z.grad().is_none());
}

#[test]
fn eager_structural_primal_ops_transpose_and_reshape() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let transposed = x.transpose(&[1, 0]).unwrap();
    assert_eq!(transposed.data().shape(), &[3, 2]);
    assert_close_slice(
        f64_data(transposed.data()),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
        TOL,
    );

    let reshaped = x.reshape(&[6]).unwrap();
    assert_eq!(reshaped.data().shape(), &[6]);
    assert_close_slice(
        f64_data(reshaped.data()),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        TOL,
    );
}

#[test]
fn eager_elementwise_primal_ops_div_abs_and_sin() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![8.0_f64, -6.0, 9.0]));
    let y = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![2.0_f64, 3.0, 3.0]));

    let div = x.div(&y).unwrap();
    assert_close_slice(f64_data(div.data()), &[4.0, -2.0, 3.0], TOL);

    let abs = x.abs().unwrap();
    assert_close_slice(f64_data(abs.data()), &[8.0, 6.0, 9.0], TOL);

    let angles = EagerTensor::from_tensor(Tensor::from_vec(
        vec![2],
        vec![0.0_f64, std::f64::consts::FRAC_PI_2],
    ));
    let sin = angles.sin().unwrap();
    assert_close_slice(f64_data(sin.data()), &[0.0, 1.0], TOL);
}

#[test]
fn eager_diagonal_primal_ops_extract_diag_and_tril() {
    let matrix = EagerTensor::from_tensor(Tensor::from_vec(
        vec![3, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let diag = matrix.extract_diag(0, 1).unwrap();
    assert_close_slice(f64_data(diag.data()), &[1.0, 5.0, 9.0], TOL);

    let lower =
        EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]))
            .tril(0)
            .unwrap();
    assert_close_slice(f64_data(lower.data()), &[1.0, 2.0, 0.0, 4.0], TOL);
}

#[test]
fn eager_reduction_primal_ops_reduce_prod() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));

    let prod = x.reduce_prod(&[0, 1]).unwrap();
    assert_close_slice(f64_data(prod.data()), &[24.0], TOL);

    let max = x.reduce_max(&[0, 1]).unwrap();
    assert_close_slice(f64_data(max.data()), &[4.0], TOL);

    let min = x.reduce_min(&[0, 1]).unwrap();
    assert_close_slice(f64_data(min.data()), &[1.0], TOL);
}

#[test]
fn eager_slice_primal() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(
        vec![4, 3],
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    ));

    let y = x
        .slice(SliceConfig {
            starts: vec![0, 0],
            limits: vec![4, 3],
            strides: vec![2, 2],
        })
        .unwrap();

    assert_eq!(y.data().shape(), &[2, 2]);
    assert_close_slice(f64_data(y.data()), &[1.0, 3.0, 9.0, 11.0], TOL);
}

#[test]
fn eager_broadcast_in_dim_primal() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]));
    let y = x.broadcast_in_dim(&[3, 2], &[0]).unwrap();

    assert_eq!(y.data().shape(), &[3, 2]);
    assert_close_slice(f64_data(y.data()), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], TOL);
}

#[test]
fn eager_pad_primal() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
    let y = x
        .pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![1],
        })
        .unwrap();

    assert_eq!(y.data().shape(), &[5]);
    assert_close_slice(f64_data(y.data()), &[0.0, 1.0, 0.0, 2.0, 0.0], TOL);
}

#[test]
fn eager_reverse_primal() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]));
    let y = x.reverse(&[0]).unwrap();

    assert_close_slice(f64_data(y.data()), &[4.0, 3.0, 2.0, 1.0], TOL);
}

#[test]
fn eager_concatenate_primal() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
    let y = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]));
    let z = EagerTensor::concatenate(&[&x, &y], 0).unwrap();

    assert_eq!(z.data().shape(), &[4]);
    assert_close_slice(f64_data(z.data()), &[1.0, 2.0, 3.0, 4.0], TOL);
}

#[test]
fn eager_gather_primal() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(
        vec![5],
        vec![10.0_f64, 20.0, 30.0, 40.0, 50.0],
    ));
    let indices = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![4.0_f64, 1.0, 0.0]));
    let y = x
        .gather(
            &indices,
            GatherConfig {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            },
        )
        .unwrap();

    assert_eq!(y.data().shape(), &[3]);
    assert_close_slice(f64_data(y.data()), &[50.0, 20.0, 10.0], TOL);
}

#[test]
fn eager_dynamic_slice_primal() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(
        vec![4, 4],
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
    ));
    let starts = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![2.0_f64, 3.0]));
    let y = x.dynamic_slice(&starts, &[2, 2]).unwrap();

    assert_eq!(y.data().shape(), &[2, 2]);
    assert_close_slice(f64_data(y.data()), &[11.0, 12.0, 15.0, 16.0], TOL);
}

#[test]
fn eager_conj_primal() {
    let x = EagerTensor::from_tensor(Tensor::from_vec(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
    ));
    let y = x.conj().unwrap();

    assert_eq!(
        c64_data(y.data()),
        &[Complex64::new(1.0, -2.0), Complex64::new(-3.0, -0.5)]
    );
}

#[test]
fn eager_analytic_primal_ops_sign_log_sqrt_rsqrt_cos_tanh_expm1_log1p() {
    let sign_input = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![-2.0_f64, 0.0, 3.0]));
    let sign = sign_input.sign().unwrap();
    assert_close_slice(f64_data(sign.data()), &[-1.0, 0.0, 1.0], TOL);

    let log_input = EagerTensor::from_tensor(Tensor::from_vec(
        vec![2],
        vec![1.0_f64, std::f64::consts::E],
    ));
    let log = log_input.log().unwrap();
    assert_close_slice(f64_data(log.data()), &[0.0, 1.0], TOL);

    let sqrt_input = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, 4.0]));
    let sqrt = sqrt_input.sqrt().unwrap();
    let rsqrt = sqrt_input.rsqrt().unwrap();
    assert_close_slice(f64_data(sqrt.data()), &[1.0, 2.0], TOL);
    assert_close_slice(f64_data(rsqrt.data()), &[1.0, 0.5], TOL);

    let angles = EagerTensor::from_tensor(Tensor::from_vec(
        vec![2],
        vec![0.0_f64, std::f64::consts::PI],
    ));
    let cos = angles.cos().unwrap();
    assert_close_slice(f64_data(cos.data()), &[1.0, -1.0], TOL);

    let tanh_input = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![0.0_f64, 1.0]));
    let tanh = tanh_input.tanh().unwrap();
    assert_close_slice(f64_data(tanh.data()), &[0.0, 1.0_f64.tanh()], TOL);

    let expm1_input = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![0.0_f64, 1.0]));
    let expm1 = expm1_input.expm1().unwrap();
    assert_close_slice(f64_data(expm1.data()), &[0.0, 1.0_f64.exp_m1()], TOL);

    let log1p_input = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, 4.0]));
    let log1p = log1p_input.log1p().unwrap();
    assert_close_slice(f64_data(log1p.data()), &[2.0_f64.ln(), 5.0_f64.ln()], TOL);
}

#[test]
fn eager_pow_maximum_and_minimum_primal() {
    let base = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![2.0_f64, 9.0]));
    let exp = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![3.0_f64, 0.5]));
    let pow = base.pow(&exp).unwrap();
    assert_close_slice(f64_data(pow.data()), &[8.0, 3.0], TOL);

    let x = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![8.0_f64, -2.0, 9.0]));
    let y = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![2.0_f64, 5.0, 3.0]));
    let maximum = x.maximum(&y).unwrap();
    let minimum = x.minimum(&y).unwrap();
    assert_close_slice(f64_data(maximum.data()), &[8.0, 5.0, 9.0], TOL);
    assert_close_slice(f64_data(minimum.data()), &[2.0, -2.0, 3.0], TOL);
}

#[test]
fn eager_select_primal() {
    let condition = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![0.0_f64, -1.0, 2.0]));
    let on_true = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]));
    let on_false = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]));
    let y = EagerTensor::select(&condition, &on_true, &on_false).unwrap();

    assert_close_slice(f64_data(y.data()), &[1.0, 20.0, 30.0], TOL);
}

#[test]
fn eager_embed_diag_and_triu_primal() {
    let diagonal = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]));
    let embedded = diagonal.embed_diag(0, 1).unwrap();
    assert_eq!(embedded.data().shape(), &[3, 3]);
    assert_close_slice(
        f64_data(embedded.data()),
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        TOL,
    );

    let matrix = EagerTensor::from_tensor(Tensor::from_vec(
        vec![3, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let upper = matrix.triu(0).unwrap();
    assert_close_slice(
        f64_data(upper.data()),
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
        TOL,
    );
}
