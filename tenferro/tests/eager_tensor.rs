#![cfg(feature = "autodiff")]

use std::sync::{Arc, OnceLock};

use num_complex::Complex64;
use tenferro::{
    CpuBackend, DType, DotGeneralConfig, EagerRuntime, EagerTensor, GatherConfig, PadConfig,
    SliceConfig, Tensor,
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

fn assert_close_c64_slice(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).norm() <= tol,
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
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], lhs.to_vec()),
        test_ctx(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], rhs.to_vec()),
        test_ctx(),
    );
    let loss = a
        .dot_general(&b, matmul_config())
        .unwrap()
        .reduce_sum(&[0, 1])
        .unwrap();
    f64_data(loss.data())[0]
}

fn test_ctx() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::with_cpu_backend(CpuBackend::new()))
        .clone()
}

#[test]
fn matrix_eager_input_uses_column_major_values() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]),
        test_ctx(),
    );
    let y = &x + &x;

    assert_eq!(y.data().shape(), &[2, 3]);
    assert_eq!(f64_data(y.data()), &[2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
}

#[test]
fn untracked_eager_intermediate_can_later_feed_tracked_ad() {
    let ctx = test_ctx();
    let plain = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let scale = &plain + &plain;
    assert!(!scale.tracks_grad());

    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]),
        ctx,
    );
    let loss = (&x * &scale).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(f64_data(x.grad().unwrap().as_ref()), &[2.0, 4.0, 6.0], TOL);
    x.clear_grad();
}

#[test]
fn eager_dot_general_with_conj_uses_untracked_fast_path() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, -0.25),
                Complex64::new(-1.0, 0.75),
                Complex64::new(0.5, 1.5),
            ],
        ),
        ctx.clone(),
    );
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(0.25, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(-2.0, 0.25),
                Complex64::new(1.5, -0.75),
            ],
        ),
        ctx,
    );
    let config = matmul_config();

    let fused = lhs
        .dot_general_with_conj(&rhs, &config, true, false)
        .unwrap();
    let explicit = lhs.conj().unwrap().dot_general(&rhs, config).unwrap();

    assert!(!fused.tracks_grad());
    assert_eq!(fused.data().shape(), explicit.data().shape());
    assert_close_c64_slice(c64_data(fused.data()), c64_data(explicit.data()), TOL);
}

#[test]
fn eager_gather_keeps_indices_integer_for_complex_operand() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
        ),
        test_ctx(),
    );
    let indices = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![2_i64, 0]),
        test_ctx(),
    );

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

    assert_eq!(y.data().shape(), &[2]);
    assert_eq!(
        c64_data(y.data()),
        &[Complex64::new(3.0, 0.5), Complex64::new(1.0, 1.0)]
    );
}

#[test]
fn eager_index_select_keeps_indices_integer_for_complex_operand() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
        ),
        test_ctx(),
    );

    let y = x.index_select(-1, &[2, 0]).unwrap();

    assert_eq!(y.data().shape(), &[2]);
    assert_eq!(
        c64_data(y.data()),
        &[Complex64::new(3.0, 0.5), Complex64::new(1.0, 1.0)]
    );
}

#[test]
fn eager_stack_trailing_axis_and_index_select_primal() {
    let x0 = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        test_ctx(),
    );
    let x1 = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]),
        test_ctx(),
    );

    let stacked = EagerTensor::stack(&[&x0, &x1], -1).unwrap();
    let selected = stacked.index_select(-1, &[1, 0, 1]).unwrap();

    assert_eq!(selected.data().shape(), &[2, 3]);
    assert_close_slice(
        f64_data(selected.data()),
        &[3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        TOL,
    );
}

#[test]
fn eager_index_select_rejects_invalid_axis_and_position() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        test_ctx(),
    );

    let axis_err = x.index_select(1, &[0]).err().unwrap().to_string();
    assert!(axis_err.contains("index_select"), "got: {axis_err}");
    assert!(axis_err.contains("axis"), "got: {axis_err}");

    let position_err = x.index_select(0, &[2]).err().unwrap().to_string();
    assert!(position_err.contains("index_select"), "got: {position_err}");
    assert!(
        position_err.contains("position 2 out of bounds"),
        "got: {position_err}"
    );
}

#[test]
fn eager_stack_rejects_empty_mismatched_shapes_and_invalid_axis() {
    let empty: [&EagerTensor; 0] = [];
    let empty_err = EagerTensor::stack(&empty, 0).err().unwrap().to_string();
    assert!(empty_err.contains("stack requires at least one input"));

    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        test_ctx(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]),
        test_ctx(),
    );
    let shape_err = EagerTensor::stack(&[&a, &b], -1).err().unwrap().to_string();
    assert!(shape_err.contains("shape mismatch"), "got: {shape_err}");

    let axis_err = EagerTensor::stack(&[&a], 2).err().unwrap().to_string();
    assert!(axis_err.contains("axis"), "got: {axis_err}");

    let c = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]),
        test_ctx(),
    );
    let out = EagerTensor::stack(&[&a, &c], 0).unwrap();
    assert_eq!(out.data().shape(), &[2, 2]);
    assert_close_slice(f64_data(out.data()), &[1.0, 3.0, 2.0, 4.0], TOL);
}

#[test]
fn eager_index_select_repeated_positions_accumulates_grad() {
    let ctx = test_ctx();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let weights = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]),
        ctx,
    );

    let selected = x.index_select(0, &[1, 1, 2]).unwrap();
    let loss = (&selected * &weights).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(
        f64_data(x.grad().unwrap().as_ref()),
        &[0.0, 30.0, 30.0],
        TOL,
    );
}

#[test]
fn eager_x_squared_gradient_matches_finite_difference() {
    let x_data = vec![1.0, 2.0, 3.0];
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], x_data.clone()),
        test_ctx(),
    );
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
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        test_ctx(),
    );

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

    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], a_data.clone()),
        test_ctx(),
    );
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], b_data.clone()),
        test_ctx(),
    );
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
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![0.0, 1.0, 2.0]),
        test_ctx(),
    );
    let loss = x.exp().unwrap().reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    let expected = vec![1.0, 1.0_f64.exp(), 2.0_f64.exp()];
    assert_close_slice(f64_data(grad.as_ref()), &expected, TOL);
}

#[test]
fn eager_fan_out_accumulates_gradient() {
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]),
        test_ctx(),
    );
    let loss = (&x + &x).reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[2.0, 2.0, 2.0], TOL);
}

#[test]
fn eager_clear_grad_resets_only_one_leaf() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]),
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
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]),
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
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]),
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
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let plain = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let leaf = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]),
        ctx,
    );

    assert!(!plain.tracks_grad());
    assert!(leaf.tracks_grad());
    assert!(!leaf.detach().tracks_grad());
}

#[test]
fn eager_send_sync_contracts_compile() {
    assert_send_sync::<EagerTensor>();
    assert_send_sync::<EagerRuntime>();
}

#[test]
fn eager_context_and_tensor_are_backend_erased_public_types() {
    assert_send_sync::<EagerTensor>();
    assert_send_sync::<EagerRuntime>();

    let ctx: Arc<EagerRuntime> = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1));
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
    let loss = (&x * &x).reduce_sum(&[0]).unwrap();
    loss.backward().unwrap();

    assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn eager_detach_cuts_one_gradient_path() {
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]),
        test_ctx(),
    );
    let detached = x.detach();
    let loss = (&detached * &x).reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[1.0, 2.0, 3.0], TOL);
    assert!(detached.grad().is_none());
}

#[test]
fn eager_untracked_tensor_behaves_like_plain_tensor() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]),
        test_ctx(),
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0, 5.0, 6.0]),
        test_ctx(),
    );
    let z = &x * &y;

    assert_close_slice(f64_data(z.data()), &[4.0, 10.0, 18.0], TOL);
    assert!(x.grad().is_none());
    assert!(y.grad().is_none());
    assert!(z.grad().is_none());
}

#[test]
fn eager_structural_primal_ops_transpose_and_reshape() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        test_ctx(),
    );

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
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![8.0_f64, -6.0, 9.0]),
        test_ctx(),
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 3.0, 3.0]),
        test_ctx(),
    );

    let div = x.div(&y).unwrap();
    assert_close_slice(f64_data(div.data()), &[4.0, -2.0, 3.0], TOL);

    let abs = x.abs().unwrap();
    assert_close_slice(f64_data(abs.data()), &[8.0, 6.0, 9.0], TOL);

    let angles = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::FRAC_PI_2]),
        test_ctx(),
    );
    let sin = angles.sin().unwrap();
    assert_close_slice(f64_data(sin.data()), &[0.0, 1.0], TOL);
}

#[test]
fn eager_diagonal_primal_ops_extract_diag_and_tril() {
    let matrix = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 3],
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ),
        test_ctx(),
    );
    let diag = matrix.extract_diag(0, 1).unwrap();
    assert_close_slice(f64_data(diag.data()), &[1.0, 5.0, 9.0], TOL);

    let lower = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
        test_ctx(),
    )
    .tril(0)
    .unwrap();
    assert_close_slice(f64_data(lower.data()), &[1.0, 2.0, 0.0, 4.0], TOL);
}

#[test]
fn eager_reduction_primal_ops_reduce_prod() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
        test_ctx(),
    );

    let prod = x.reduce_prod(&[0, 1]).unwrap();
    assert_close_slice(f64_data(prod.data()), &[24.0], TOL);

    let max = x.reduce_max(&[0, 1]).unwrap();
    assert_close_slice(f64_data(max.data()), &[4.0], TOL);

    let min = x.reduce_min(&[0, 1]).unwrap();
    assert_close_slice(f64_data(min.data()), &[1.0], TOL);
}

#[test]
fn eager_slice_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![4, 3],
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ),
        test_ctx(),
    );

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
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        test_ctx(),
    );
    let y = x.broadcast_in_dim(&[3, 2], &[0]).unwrap();

    assert_eq!(y.data().shape(), &[3, 2]);
    assert_close_slice(f64_data(y.data()), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], TOL);
}

#[test]
fn eager_pad_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        test_ctx(),
    );
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
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]),
        test_ctx(),
    );
    let y = x.reverse(&[0]).unwrap();

    assert_close_slice(f64_data(y.data()), &[4.0, 3.0, 2.0, 1.0], TOL);
}

#[test]
fn eager_concatenate_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        test_ctx(),
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]),
        test_ctx(),
    );
    let z = EagerTensor::concatenate(&[&x, &y], 0).unwrap();

    assert_eq!(z.data().shape(), &[4]);
    assert_close_slice(f64_data(z.data()), &[1.0, 2.0, 3.0, 4.0], TOL);
}

#[test]
fn eager_gather_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]),
        test_ctx(),
    );
    let indices = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![4_i64, 1, 0]),
        test_ctx(),
    );
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
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![4, 4],
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0,
            ],
        ),
        test_ctx(),
    );
    let starts = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]),
        test_ctx(),
    );
    let y = x.dynamic_slice(&starts, &[2, 2]).unwrap();

    assert_eq!(y.data().shape(), &[2, 2]);
    assert_close_slice(f64_data(y.data()), &[11.0, 12.0, 15.0, 16.0], TOL);
}

#[test]
fn eager_conj_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        ),
        test_ctx(),
    );
    let y = x.conj().unwrap();

    assert_eq!(
        c64_data(y.data()),
        &[Complex64::new(1.0, -2.0), Complex64::new(-3.0, -0.5)]
    );
}

#[test]
fn eager_analytic_primal_ops_sign_log_sqrt_rsqrt_cos_tanh_expm1_log1p() {
    let sign_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![-2.0_f64, 0.0, 3.0]),
        test_ctx(),
    );
    let sign = sign_input.sign().unwrap();
    assert_close_slice(f64_data(sign.data()), &[-1.0, 0.0, 1.0], TOL);

    let log_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, std::f64::consts::E]),
        test_ctx(),
    );
    let log = log_input.log().unwrap();
    assert_close_slice(f64_data(log.data()), &[0.0, 1.0], TOL);

    let sqrt_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]),
        test_ctx(),
    );
    let sqrt = sqrt_input.sqrt().unwrap();
    let rsqrt = sqrt_input.rsqrt().unwrap();
    assert_close_slice(f64_data(sqrt.data()), &[1.0, 2.0], TOL);
    assert_close_slice(f64_data(rsqrt.data()), &[1.0, 0.5], TOL);

    let angles = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::PI]),
        test_ctx(),
    );
    let cos = angles.cos().unwrap();
    assert_close_slice(f64_data(cos.data()), &[1.0, -1.0], TOL);

    let tanh_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]),
        test_ctx(),
    );
    let tanh = tanh_input.tanh().unwrap();
    assert_close_slice(f64_data(tanh.data()), &[0.0, 1.0_f64.tanh()], TOL);

    let expm1_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]),
        test_ctx(),
    );
    let expm1 = expm1_input.expm1().unwrap();
    assert_close_slice(f64_data(expm1.data()), &[0.0, 1.0_f64.exp_m1()], TOL);

    let log1p_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]),
        test_ctx(),
    );
    let log1p = log1p_input.log1p().unwrap();
    assert_close_slice(f64_data(log1p.data()), &[2.0_f64.ln(), 5.0_f64.ln()], TOL);
}

#[test]
fn eager_pow_maximum_and_minimum_primal() {
    let base = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 9.0]),
        test_ctx(),
    );
    let exp = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 0.5]),
        test_ctx(),
    );
    let pow = base.pow(&exp).unwrap();
    assert_close_slice(f64_data(pow.data()), &[8.0, 3.0], TOL);

    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![8.0_f64, -2.0, 9.0]),
        test_ctx(),
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 5.0, 3.0]),
        test_ctx(),
    );
    let maximum = x.maximum(&y).unwrap();
    let minimum = x.minimum(&y).unwrap();
    assert_close_slice(f64_data(maximum.data()), &[8.0, 5.0, 9.0], TOL);
    assert_close_slice(f64_data(minimum.data()), &[2.0, -2.0, 3.0], TOL);
}

#[test]
fn eager_select_primal() {
    let condition = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![false, true, true]),
        test_ctx(),
    );
    let on_true = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]),
        test_ctx(),
    );
    let on_false = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        test_ctx(),
    );
    let y = EagerTensor::select(&condition, &on_true, &on_false).unwrap();

    assert_close_slice(f64_data(y.data()), &[1.0, 20.0, 30.0], TOL);
}

#[test]
fn eager_embed_diag_and_triu_primal() {
    let diagonal = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]),
        test_ctx(),
    );
    let embedded = diagonal.embed_diag(0, 1).unwrap();
    assert_eq!(embedded.data().shape(), &[3, 3]);
    assert_close_slice(
        f64_data(embedded.data()),
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        TOL,
    );

    let matrix = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 3],
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ),
        test_ctx(),
    );
    let upper = matrix.triu(0).unwrap();
    assert_close_slice(
        f64_data(upper.data()),
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
        TOL,
    );
}

// --- Context boundary tests ---

#[test]
fn context_id_is_unique() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    assert_ne!(ctx_a.id(), ctx_b.id());
}

#[test]
fn same_context_true_for_shared_ctx() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]),
        ctx.clone(),
    );
    let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]), ctx);
    assert!(x.same_context(&y));
    assert_eq!(x.ctx_id(), y.ctx_id());
}

#[test]
fn same_context_false_for_different_ctx() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]), ctx_a);
    let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]), ctx_b);
    assert!(!x.same_context(&y));
    assert_ne!(x.ctx_id(), y.ctx_id());
}

#[test]
fn constant_from_creates_untracked_leaf() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let c = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
    assert_eq!(c.ctx_id(), ctx.id());
    assert!(!c.tracks_grad());
    assert_eq!(f64_data(c.data()), &[1.0, 2.0]);
}

#[test]
fn variable_from_creates_tracked_leaf() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let p = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
    assert_eq!(p.ctx_id(), ctx.id());
    assert!(p.tracks_grad());
    // backward should work on a tracked variable
    let loss = p.exp().unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert!(p.grad().is_some());
}

#[test]
fn cross_context_add_rejected() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx_a,
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]),
        ctx_b,
    );
    let msg = match x.add(&y) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(msg.contains("different eager AD contexts"), "got: {msg}");
}

#[test]
fn cross_context_mul_rejected() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx_a,
    );
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]),
        ctx_b,
    );
    let msg = match x.mul(&y) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(msg.contains("different eager AD contexts"), "got: {msg}");
}

#[test]
fn cross_context_tracked_tensors_rejected() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx_a,
    );
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]),
        ctx_b,
    );
    let msg = match x.add(&y) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(msg.contains("different eager AD contexts"), "got: {msg}");
}

#[test]
fn constant_from_can_cross_context() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx.clone(),
    );
    // Import a fixed mask from a raw tensor into the same context
    let c = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]));
    let z = x.add(&c).unwrap();
    assert_eq!(f64_data(z.data()), &[4.0, 6.0]);
}

#[test]
fn detach_into_different_context() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx_a,
    );
    let d = x.detach_into(&ctx_b);
    assert_eq!(d.ctx_id(), ctx_b.id());
    assert!(!d.tracks_grad());
    assert_eq!(f64_data(d.data()), &[1.0, 2.0]);
    // Can operate with tensors from ctx_b now
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]),
        ctx_b,
    );
    let z = d.add(&y).unwrap();
    assert_eq!(f64_data(z.data()), &[4.0, 6.0]);
}

#[test]
fn detach_into_still_accessible_in_original_context() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx_a.clone(),
    );
    let d = x.detach_into(&ctx_b);
    // Original tensor still in ctx_a, should work fine
    let loss = x.exp().unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert!(x.grad().is_some());
    // d is in ctx_b, x is in ctx_a
    assert_ne!(d.ctx_id(), x.ctx_id());
}

// --- dtype promotion tests ---

#[test]
fn promote_i64_add_f64_eager() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]),
        ctx.clone(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 1.0]),
        ctx.clone(),
    );
    // I64 + F64 should promote to F64
    let z = a.add(&b).unwrap();
    assert_eq!(z.data().dtype(), DType::F64);
    assert_eq!(z.data().as_slice::<f64>().unwrap(), &[1.5, 3.0]);
}

#[test]
fn promote_i64_mul_c64_eager() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![3_i64]),
        ctx.clone(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]),
        ctx,
    );
    // I64 * C64 should promote to C64
    let z = a.mul(&b).unwrap();
    assert_eq!(z.data().dtype(), DType::C64);
    assert_eq!(
        z.data().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(3.0, 0.0)]
    );
}

#[test]
fn promote_f32_add_f64_eager() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]),
        ctx.clone(),
    );
    let b =
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 1.0]), ctx);
    // F32 + F64 should promote to F64
    let z = a.add(&b).unwrap();
    assert_eq!(z.data().dtype(), DType::F64);
    assert_eq!(z.data().as_slice::<f64>().unwrap(), &[1.5, 3.0]);
}

#[test]
fn promote_same_dtype_no_conversion_penalty() {
    // Same-dtype ops should work without any conversion overhead
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx.clone(),
    );
    let b =
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx);
    let z = a.add(&b).unwrap();
    assert_eq!(z.data().dtype(), DType::F64);
    assert_eq!(z.data().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}
