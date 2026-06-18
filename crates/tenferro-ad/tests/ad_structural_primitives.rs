use std::sync::Arc;
use tenferro_ad::{EagerRuntime, EagerTensor};

use tenferro_cpu::CpuBackend;
use tenferro_runtime::Tensor;

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
    EagerRuntime::with_cpu_backend(CpuBackend::new())
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
    );
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], y_data.clone()).unwrap(),
        ctx.clone(),
    );
    let max_weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], max_weights.clone()).unwrap(),
        ctx.clone(),
    );
    let min_weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], min_weights.clone()).unwrap(),
        ctx.clone(),
    );

    let max_loss = x
        .maximum(&y)
        .unwrap()
        .mul(&max_weights_tensor)
        .unwrap()
        .reduce_sum(&[0])
        .unwrap();
    let min_loss = x
        .minimum(&y)
        .unwrap()
        .mul(&min_weights_tensor)
        .unwrap()
        .reduce_sum(&[0])
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

    assert_close(f64_data(&grad_x), &finite_diff_unary(loss_for_x, &x_data));
    assert_close(f64_data(&grad_y), &finite_diff_unary(loss_for_y, &y_data));
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
    );
    let on_true = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], true_data.clone()).unwrap(),
        ctx.clone(),
    );
    let on_false = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], false_data.clone()).unwrap(),
        ctx.clone(),
    );
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], weights.clone()).unwrap(),
        ctx.clone(),
    );

    let loss = EagerTensor::select(&condition, &on_true, &on_false)
        .unwrap()
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(&[0])
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
        f64_data(on_true.grad().unwrap().unwrap().as_ref()),
        &finite_diff_unary(loss_for_true, &true_data),
    );
    assert_close(
        f64_data(on_false.grad().unwrap().unwrap().as_ref()),
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
    );
    let lower = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], lower_data.clone()).unwrap(),
        ctx.clone(),
    );
    let upper = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], upper_data.clone()).unwrap(),
        ctx.clone(),
    );
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], weights.clone()).unwrap(),
        ctx.clone(),
    );

    let loss = input
        .clamp(&lower, &upper)
        .unwrap()
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(&[0])
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
        f64_data(input.grad().unwrap().unwrap().as_ref()),
        &finite_diff_unary(|xs| loss_with(xs, &lower_data, &upper_data), &input_data),
    );
    assert_close(
        f64_data(lower.grad().unwrap().unwrap().as_ref()),
        &finite_diff_unary(
            |lows| loss_with(&input_data, lows, &upper_data),
            &lower_data,
        ),
    );
    assert_close(
        f64_data(upper.grad().unwrap().unwrap().as_ref()),
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
    );
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], weights.clone()).unwrap(),
        ctx.clone(),
    );

    let loss = input
        .extract_diag(0, 1)
        .unwrap()
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(&[0])
        .unwrap();
    let _ = loss.backward().unwrap();

    let loss_for_input = |values: &[f64]| weights[0] * values[0] + weights[1] * values[3];

    let grad = input.grad().unwrap().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);
    assert_close(
        f64_data(&grad),
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
    );
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2, 3], weights.clone()).unwrap(),
        ctx.clone(),
    );

    let loss = input
        .embed_diag(1, 0)
        .unwrap()
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(&[0, 1, 2])
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
        f64_data(&grad),
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
    );
    let middle = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], middle_data.clone()).unwrap(),
        ctx.clone(),
    );
    let right = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], right_data.clone()).unwrap(),
        ctx.clone(),
    );
    let weights_tensor = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![6], weights.clone()).unwrap(),
        ctx.clone(),
    );

    let concatenated = EagerTensor::concatenate(&[&left, &middle, &right], 0).unwrap();
    let loss = concatenated
        .mul(&weights_tensor)
        .unwrap()
        .reduce_sum(&[0])
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
        f64_data(left.grad().unwrap().unwrap().as_ref()),
        &finite_diff_unary(loss_for_left, &left_data),
    );
    assert_close(
        f64_data(middle.grad().unwrap().unwrap().as_ref()),
        &finite_diff_unary(loss_for_middle, &middle_data),
    );
    assert_close(
        f64_data(right.grad().unwrap().unwrap().as_ref()),
        &finite_diff_unary(loss_for_right, &right_data),
    );
}
