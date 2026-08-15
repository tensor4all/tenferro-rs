use std::fmt::Debug;
use std::sync::Arc;

use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorScalar};

const FD_H: f64 = 1.0e-6;
const TOL: f64 = 1.0e-5;

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn test_ctx() -> Arc<EagerRuntime> {
    EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap()
}

fn eager_input<T: TensorScalar>(
    ctx: &Arc<EagerRuntime>,
    shape: Vec<usize>,
    data: &[T],
    tracked: bool,
) -> EagerTensor {
    let tensor = Tensor::from_vec_col_major(shape, data.to_vec()).unwrap();
    if tracked {
        EagerTensor::requires_grad_in(tensor, Arc::clone(ctx)).unwrap()
    } else {
        EagerTensor::from_tensor_in(tensor, Arc::clone(ctx)).unwrap()
    }
}

fn assert_values<T>(tensor: &EagerTensor, expected: &[T])
where
    T: TensorScalar + Debug + PartialEq,
{
    let actual = tensor.to_tensor().unwrap();
    assert_eq!(actual.as_slice::<T>().unwrap(), expected);
}

fn finite_diff_unary(f: impl Fn(&[f64]) -> f64, base: &[f64]) -> Vec<f64> {
    (0..base.len())
        .map(|index| {
            let mut plus = base.to_vec();
            let mut minus = base.to_vec();
            plus[index] += FD_H;
            minus[index] -= FD_H;
            (f(&plus) - f(&minus)) / (2.0 * FD_H)
        })
        .collect()
}

fn weighted_sum(values: impl Iterator<Item = f64>, weights: &[f64]) -> f64 {
    values
        .zip(weights.iter())
        .map(|(value, &weight)| value * weight)
        .sum()
}

#[test]
fn eager_maximum_and_minimum_gradients_match_finite_diff() {
    let ctx = test_ctx();
    let x_data = vec![3.0_f64, -2.0, 0.5, 4.0];
    let y_data = vec![1.0_f64, 5.0, -1.0, 2.0];
    let max_weights = vec![0.5_f64, -1.25, 2.0, -0.75];
    let min_weights = vec![1.5_f64, -0.25, 0.75, 2.25];

    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], x_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], y_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let max_weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], max_weights.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let min_weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], min_weights.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let max_loss = x
        .maximum(&y)
        .unwrap()
        .mul(&max_weights_tensor)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();
    let min_loss = x
        .minimum(&y)
        .unwrap()
        .mul(&min_weights_tensor)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();
    let loss = max_loss.add(&min_loss).unwrap();
    let _ = loss.backward().unwrap();

    let grad_x = x.grad().unwrap().unwrap();
    let grad_y = y.grad().unwrap().unwrap();

    let loss_for_x = |xs: &[f64]| {
        let max_part = weighted_sum(
            xs.iter()
                .zip(&y_data)
                .map(|(&x, &y)| if x >= y { x } else { y }),
            &max_weights,
        );
        let min_part = weighted_sum(
            xs.iter()
                .zip(&y_data)
                .map(|(&x, &y)| if x <= y { x } else { y }),
            &min_weights,
        );
        max_part + min_part
    };
    let loss_for_y = |ys: &[f64]| {
        let max_part = weighted_sum(
            x_data
                .iter()
                .zip(ys)
                .map(|(&x, &y)| if x >= y { x } else { y }),
            &max_weights,
        );
        let min_part = weighted_sum(
            x_data
                .iter()
                .zip(ys)
                .map(|(&x, &y)| if x <= y { x } else { y }),
            &min_weights,
        );
        max_part + min_part
    };

    assert_close(
        f64_data(&grad_x.to_tensor().unwrap()),
        &finite_diff_unary(loss_for_x, &x_data),
    );
    assert_close(
        f64_data(&grad_y.to_tensor().unwrap()),
        &finite_diff_unary(loss_for_y, &y_data),
    );
}

#[test]
fn eager_select_gradients_match_finite_diff() {
    let ctx = test_ctx();
    let condition_data = vec![false, true, true, false];
    let true_data = vec![10.0_f64, 20.0, 30.0, 40.0];
    let false_data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let weights = vec![0.5_f64, -1.0, 2.0, 1.25];

    let condition = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], condition_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let on_true = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], true_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let on_false = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], false_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], weights.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let loss = EagerTensor::select(&condition, &on_true, &on_false)
        .unwrap()
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();
    let _ = loss.backward().unwrap();

    let loss_for_true = |values: &[f64]| {
        weighted_sum(
            condition_data
                .iter()
                .zip(values.iter())
                .zip(false_data.iter())
                .map(|((&cond, &t), &f)| if cond { t } else { f }),
            &weights,
        )
    };
    let loss_for_false = |values: &[f64]| {
        weighted_sum(
            condition_data
                .iter()
                .zip(true_data.iter())
                .zip(values.iter())
                .map(|((&cond, &t), &f)| if cond { t } else { f }),
            &weights,
        )
    };

    assert_close(
        f64_data(&on_true.grad().unwrap().unwrap().to_tensor().unwrap()),
        &finite_diff_unary(loss_for_true, &true_data),
    );
    assert_close(
        f64_data(&on_false.grad().unwrap().unwrap().to_tensor().unwrap()),
        &finite_diff_unary(loss_for_false, &false_data),
    );
}

#[test]
fn eager_clamp_gradients_match_finite_diff() {
    let ctx = test_ctx();
    let input_data = vec![-2.0_f64, 0.5, 5.0, 2.0];
    let lower_data = vec![-1.0_f64, 0.0, 1.0, 0.0];
    let upper_data = vec![1.0_f64, 2.0, 4.0, 3.0];
    let weights = vec![0.5_f64, -1.25, 2.0, 0.75];

    let input = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], input_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let lower = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], lower_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let upper = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], upper_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], weights.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let loss = input
        .clamp(&lower, &upper)
        .unwrap()
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();
    let _ = loss.backward().unwrap();

    let loss_with = |xs: &[f64], lows: &[f64], highs: &[f64]| {
        weighted_sum(
            xs.iter()
                .zip(lows.iter())
                .zip(highs.iter())
                .map(|((&x, &lo), &hi)| lo.max(hi.min(x))),
            &weights,
        )
    };

    assert_close(
        f64_data(&input.grad().unwrap().unwrap().to_tensor().unwrap()),
        &finite_diff_unary(|xs| loss_with(xs, &lower_data, &upper_data), &input_data),
    );
    assert_close(
        f64_data(&lower.grad().unwrap().unwrap().to_tensor().unwrap()),
        &finite_diff_unary(
            |lows| loss_with(&input_data, lows, &upper_data),
            &lower_data,
        ),
    );
    assert_close(
        f64_data(&upper.grad().unwrap().unwrap().to_tensor().unwrap()),
        &finite_diff_unary(
            |highs| loss_with(&input_data, &lower_data, highs),
            &upper_data,
        ),
    );
}

#[test]
fn eager_extract_diag_rectangular_gradient_matches_finite_diff() {
    let ctx = test_ctx();
    let input_data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weights = vec![0.5_f64, -1.25];

    let input = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], input_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], weights.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let loss = input
        .extract_diag(0, 1)
        .unwrap()
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();
    let _ = loss.backward().unwrap();

    let loss_for_input = |values: &[f64]| weights[0] * values[0] + weights[1] * values[3];

    let grad = input.grad().unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);
    assert_close(
        f64_data(&grad.to_tensor().unwrap()),
        &finite_diff_unary(loss_for_input, &input_data),
    );
}

#[test]
fn eager_embed_diag_shifted_axis_gradient_matches_finite_diff() {
    let ctx = test_ctx();
    let input_shape = vec![2, 3];
    let input_data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weights = vec![
        0.5_f64, -1.0, 1.5, 2.0, -0.25, 0.75, 1.25, -1.5, 0.25, -0.5, 2.5, -2.0, 0.0, 3.0, -3.5,
        4.0, -4.5, 5.0,
    ];

    let input = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(input_shape.clone(), input_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2, 3], weights.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let loss = input
        .embed_diag(1, 0)
        .unwrap()
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(Some(&[0, 1, 2]))
        .unwrap();
    let _ = loss.backward().unwrap();

    let loss_for_input = |values: &[f64]| {
        (0..input_shape[1])
            .flat_map(|col| (0..input_shape[0]).map(move |row| (row, col)))
            .map(|(row, col)| {
                let input_flat = row + input_shape[0] * col;
                let output_flat = col + input_shape[1] * (row + input_shape[0] * col);
                values[input_flat] * weights[output_flat]
            })
            .sum()
    };

    let grad = input.grad().unwrap().unwrap();
    assert_eq!(grad.shape(), input_shape.as_slice());
    assert_close(
        f64_data(&grad.to_tensor().unwrap()),
        &finite_diff_unary(loss_for_input, &input_data),
    );
}

#[test]
fn eager_concatenate_gradients_match_finite_diff() {
    let ctx = test_ctx();
    let left_data = vec![1.0_f64, 2.0];
    let middle_data = vec![3.0_f64];
    let right_data = vec![4.0_f64, 5.0, 6.0];
    let weights = vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 0.75];

    let left = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], left_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let middle = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], middle_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let right = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], right_data.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![6], weights.clone()).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let concatenated = EagerTensor::concatenate(&[&left, &middle, &right], 0).unwrap();
    let loss = concatenated
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();
    let _ = loss.backward().unwrap();

    let loss_for_left = |values: &[f64]| {
        weighted_sum(
            values
                .iter()
                .chain(middle_data.iter())
                .chain(right_data.iter())
                .copied(),
            &weights,
        )
    };
    let loss_for_middle = |values: &[f64]| {
        weighted_sum(
            left_data
                .iter()
                .chain(values.iter())
                .chain(right_data.iter())
                .copied(),
            &weights,
        )
    };
    let loss_for_right = |values: &[f64]| {
        weighted_sum(
            left_data
                .iter()
                .chain(middle_data.iter())
                .chain(values.iter())
                .copied(),
            &weights,
        )
    };

    assert_close(
        f64_data(&left.grad().unwrap().unwrap().to_tensor().unwrap()),
        &finite_diff_unary(loss_for_left, &left_data),
    );
    assert_close(
        f64_data(&middle.grad().unwrap().unwrap().to_tensor().unwrap()),
        &finite_diff_unary(loss_for_middle, &middle_data),
    );
    assert_close(
        f64_data(&right.grad().unwrap().unwrap().to_tensor().unwrap()),
        &finite_diff_unary(loss_for_right, &right_data),
    );
}

// INVARIANT: Explicit input and oracle slices keep both dtype cases independently auditable.
#[allow(clippy::too_many_arguments)]
fn run_exact_concatenate_case<T>(
    left_data: &[T],
    middle_data: &[T],
    right_data: &[T],
    cotangent_data: &[T],
    middle_tangent_data: &[T],
    expected_output: &[T],
    expected_left_vjp: &[T],
    expected_middle_vjp: &[T],
    expected_right_vjp: &[T],
    expected_jvp: &[T],
) where
    T: TensorScalar + Debug + PartialEq,
{
    let ctx = test_ctx();
    let left = eager_input(&ctx, vec![2, 1], left_data, true);
    let middle = eager_input(&ctx, vec![2, 2], middle_data, true);
    let right = eager_input(&ctx, vec![2, 1], right_data, true);
    let cotangent = eager_input(&ctx, vec![2, 4], cotangent_data, false);

    let output = EagerTensor::concatenate(&[&left, &middle, &right], 1).unwrap();
    assert_values(&output, expected_output);
    assert_values(
        &ctx.vjp(&output, &left, &cotangent).unwrap(),
        expected_left_vjp,
    );
    assert_values(
        &ctx.vjp(&output, &middle, &cotangent).unwrap(),
        expected_middle_vjp,
    );
    assert_values(
        &ctx.vjp(&output, &right, &cotangent).unwrap(),
        expected_right_vjp,
    );

    let middle_tangent = eager_input(&ctx, vec![2, 2], middle_tangent_data, false);
    assert_values(
        &ctx.jvp(&output, &middle, &middle_tangent).unwrap(),
        expected_jvp,
    );
}

#[test]
fn eager_concatenate_semantic_vjp_accepts_distinct_tracked_shapes() {
    run_exact_concatenate_case(
        &[1.0_f64, 2.0],
        &[3.0, 4.0, 5.0, 6.0],
        &[7.0, 8.0],
        &[0.5, -1.25, 2.0, 1.5, -0.25, 0.75, 3.0, -2.0],
        &[0.25, -0.5, 1.25, -1.5],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[0.5, -1.25],
        &[2.0, 1.5, -0.25, 0.75],
        &[3.0, -2.0],
        &[0.0, 0.0, 0.25, -0.5, 1.25, -1.5, 0.0, 0.0],
    );
}

#[test]
fn eager_concatenate_semantic_vjp_supports_complex64_exact_weights() {
    let c = Complex64::new;
    run_exact_concatenate_case(
        &[c(1.0, 0.5), c(2.0, -0.25)],
        &[c(3.0, 1.0), c(4.0, -1.0), c(5.0, 0.5), c(6.0, -0.5)],
        &[c(7.0, 0.25), c(8.0, -0.75)],
        &[
            c(0.5, -1.0),
            c(-1.25, 0.25),
            c(2.0, 0.5),
            c(1.5, -0.75),
            c(-0.25, 1.0),
            c(0.75, -0.5),
            c(3.0, 1.25),
            c(-2.0, 0.25),
        ],
        &[c(0.25, 0.5), c(-0.5, -0.25), c(1.25, 1.0), c(-1.5, 0.75)],
        &[
            c(1.0, 0.5),
            c(2.0, -0.25),
            c(3.0, 1.0),
            c(4.0, -1.0),
            c(5.0, 0.5),
            c(6.0, -0.5),
            c(7.0, 0.25),
            c(8.0, -0.75),
        ],
        &[c(0.5, -1.0), c(-1.25, 0.25)],
        &[c(2.0, 0.5), c(1.5, -0.75), c(-0.25, 1.0), c(0.75, -0.5)],
        &[c(3.0, 1.25), c(-2.0, 0.25)],
        &[
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.25, 0.5),
            c(-0.5, -0.25),
            c(1.25, 1.0),
            c(-1.5, 0.75),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
    );
}

fn run_mixed_concatenate_case(tracked_first: bool) {
    let ctx = test_ctx();
    let tracked = eager_input(&ctx, vec![2, 1], &[10.0_f64, 20.0], true);
    let inactive = eager_input(&ctx, vec![2, 2], &[30.0_f64, 40.0, 50.0, 60.0], false);
    let cotangent = eager_input(
        &ctx,
        vec![2, 3],
        &[0.5_f64, -1.0, 2.0, 1.5, -0.25, 0.75],
        false,
    );
    let output = if tracked_first {
        EagerTensor::concatenate(&[&tracked, &inactive], 1).unwrap()
    } else {
        EagerTensor::concatenate(&[&inactive, &tracked], 1).unwrap()
    };

    let expected_output = if tracked_first {
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0][..]
    } else {
        &[30.0, 40.0, 50.0, 60.0, 10.0, 20.0][..]
    };
    let expected_vjp = if tracked_first {
        &[0.5, -1.0][..]
    } else {
        &[-0.25, 0.75][..]
    };
    assert_values(&output, expected_output);
    assert_values(
        &ctx.vjp(&output, &tracked, &cotangent).unwrap(),
        expected_vjp,
    );
}

#[test]
fn eager_concatenate_mixed_activity_preserves_tracked_gradients_and_offsets() {
    for tracked_first in [true, false] {
        run_mixed_concatenate_case(tracked_first);
    }
}

// INVARIANT: Explicit inputs and oracles cover the bounded order/axis test matrix without duplication.
#[allow(clippy::too_many_arguments)]
fn run_mixed_stack_case<T>(
    dim: isize,
    tracked_first: bool,
    tracked_data: &[T],
    inactive_data: &[T],
    cotangent_data: &[T],
    tangent_data: &[T],
    expected_output: Vec<T>,
    expected_vjp: Vec<T>,
    expected_jvp: Vec<T>,
) where
    T: TensorScalar + Debug + PartialEq,
{
    let ctx = test_ctx();
    let tracked = eager_input(&ctx, vec![2], tracked_data, true);
    let inactive = eager_input(&ctx, vec![2], inactive_data, false);
    let output = if tracked_first {
        EagerTensor::stack(&[&tracked, &inactive], dim).unwrap()
    } else {
        EagerTensor::stack(&[&inactive, &tracked], dim).unwrap()
    };
    let cotangent = eager_input(&ctx, vec![2, 2], cotangent_data, false);
    let tangent = eager_input(&ctx, vec![2], tangent_data, false);

    assert_values(&output, &expected_output);
    assert_values(
        &ctx.vjp(&output, &tracked, &cotangent).unwrap(),
        &expected_vjp,
    );
    assert!(ctx
        .vjp_optional(&output, &inactive, &cotangent)
        .unwrap()
        .is_none());
    assert_values(
        &ctx.jvp(&output, &tracked, &tangent).unwrap(),
        &expected_jvp,
    );
}

#[test]
fn eager_mixed_stack_vjp_and_jvp_cover_orders_and_axes() {
    for &(dim, tracked_first) in &[(0, true), (0, false), (-1, true), (-1, false)] {
        let (expected_output, expected_vjp, expected_jvp) = match (dim, tracked_first) {
            (0, true) => (
                vec![10.0, 30.0, 20.0, 40.0],
                vec![0.5, 2.0],
                vec![0.25, 0.0, -1.5, 0.0],
            ),
            (0, false) => (
                vec![30.0, 10.0, 40.0, 20.0],
                vec![-1.25, 1.5],
                vec![0.0, 0.25, 0.0, -1.5],
            ),
            (-1, true) => (
                vec![10.0, 20.0, 30.0, 40.0],
                vec![0.5, -1.25],
                vec![0.25, -1.5, 0.0, 0.0],
            ),
            (-1, false) => (
                vec![30.0, 40.0, 10.0, 20.0],
                vec![2.0, 1.5],
                vec![0.0, 0.0, 0.25, -1.5],
            ),
            _ => unreachable!(),
        };
        run_mixed_stack_case(
            dim,
            tracked_first,
            &[10.0_f64, 20.0],
            &[30.0_f64, 40.0],
            &[0.5_f64, -1.25, 2.0, 1.5],
            &[0.25_f64, -1.5],
            expected_output,
            expected_vjp,
            expected_jvp,
        );
    }

    let c = Complex64::new;
    run_mixed_stack_case(
        -1,
        false,
        &[c(10.0, 1.0), c(20.0, -2.0)],
        &[c(30.0, 3.0), c(40.0, -4.0)],
        &[c(0.5, -0.5), c(-1.25, 0.25), c(2.0, 1.0), c(1.5, -1.5)],
        &[c(0.25, 0.75), c(-1.5, -0.25)],
        vec![c(30.0, 3.0), c(40.0, -4.0), c(10.0, 1.0), c(20.0, -2.0)],
        vec![c(2.0, 1.0), c(1.5, -1.5)],
        vec![c(0.0, 0.0), c(0.0, 0.0), c(0.25, 0.75), c(-1.5, -0.25)],
    );
}

#[test]
fn eager_mixed_stack_vjp_then_jvp_remains_composable() {
    let ctx = test_ctx();
    let tracked = eager_input(&ctx, vec![2], &[2.0_f64, 3.0], true);
    let inactive = eager_input(&ctx, vec![2], &[5.0_f64, 7.0], false);
    let stacked = EagerTensor::stack(&[&tracked, &inactive], -1).unwrap();
    let loss = stacked
        .mul(&stacked)
        .unwrap()
        .reduce_sum(Some(&[0, 1]))
        .unwrap();
    let seed = eager_input(&ctx, vec![], &[1.0_f64], false);

    let gradient = ctx.vjp(&loss, &tracked, &seed).unwrap();
    assert_values(&gradient, &[4.0, 6.0]);

    let tangent = eager_input(&ctx, vec![2], &[0.5_f64, -1.25], false);
    let hvp = ctx.jvp(&gradient, &tracked, &tangent).unwrap();
    assert_values(&hvp, &[1.0, -2.5]);
}

#[test]
fn eager_concatenate_cache_isolated_for_compatible_shapes() {
    let ctx = test_ctx();
    ctx.clear_caches().unwrap();

    let first_left = eager_input(&ctx, vec![2, 1], &[1.0_f64, 2.0], true);
    let first_right = eager_input(&ctx, vec![2, 2], &[3.0_f64, 4.0, 5.0, 6.0], true);
    let first_output = EagerTensor::concatenate(&[&first_left, &first_right], 1).unwrap();
    let first_cotangent = eager_input(
        &ctx,
        vec![2, 3],
        &[0.5_f64, -1.0, 2.0, 1.5, -0.25, 0.75],
        false,
    );
    assert_values(
        &ctx.vjp(&first_output, &first_right, &first_cotangent)
            .unwrap(),
        &[2.0, 1.5, -0.25, 0.75],
    );
    assert_eq!(ctx.cache_stats().unwrap().prepared_derivatives.entries, 1);

    let second_left = eager_input(&ctx, vec![2, 2], &[7.0_f64, 8.0, 9.0, 10.0], true);
    let second_right = eager_input(&ctx, vec![2, 1], &[11.0_f64, 12.0], true);
    let second_output = EagerTensor::concatenate(&[&second_left, &second_right], 1).unwrap();
    let second_cotangent = eager_input(
        &ctx,
        vec![2, 3],
        &[1.25_f64, -0.5, 2.5, -1.75, 3.0, 0.25],
        false,
    );
    assert_values(
        &ctx.vjp(&second_output, &second_right, &second_cotangent)
            .unwrap(),
        &[3.0, 0.25],
    );
    assert_eq!(ctx.cache_stats().unwrap().prepared_derivatives.entries, 2);
}
