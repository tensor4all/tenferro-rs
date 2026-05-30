use super::*;

#[test]
fn test_reduce_prod_jvp() {
    let op = StdTensorOp::ReduceProd { axes: vec![0] };
    let input_shape = vec![4usize];
    let (fragment, input_key, output_key) =
        build_unary_fragment(op.clone(), tensor_input_key(60_000));
    let x_data = vec![1.5, -2.0, 0.75, 4.0];
    let dx_data = vec![0.25, -0.5, 1.0, 0.75];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        input_key.clone(),
        f64_tensor(input_shape.clone(), x_data.clone()),
    );

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(input_shape.clone(), dx_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &x_data, &dx_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
    });
}

#[test]
fn test_reduce_prod_vjp() {
    let op = StdTensorOp::ReduceProd { axes: vec![0] };
    let input_shape = vec![2usize, 3];
    let input_key = tensor_input_key(60_001);
    let x_data = vec![1.5, -2.0, 0.75, 4.0, -0.5, 2.0];
    let cotangent = vec![0.5, -1.0, 2.0];
    let grad = transpose_primal_unary_op_with_inputs(
        op.clone(),
        input_key,
        f64_tensor(input_shape.clone(), x_data.clone()),
        f64_tensor(vec![3], cotangent.clone()),
    );

    assert_grad_matches_finite_diff(get_f64_data(&grad), &x_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
            .iter()
            .zip(cotangent.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    });
}

#[test]
fn test_reduce_max_jvp() {
    let op = StdTensorOp::ReduceMax { axes: vec![0] };
    let input_shape = vec![2usize, 3];
    let (fragment, input_key, output_key) =
        build_unary_fragment(op.clone(), tensor_input_key(60_002));
    let x_data = vec![2.0, 2.0, 4.0, 1.0, -3.0, -3.0];
    let dx_data = vec![1.0, -0.5, 0.75, -1.25, 2.0, -1.0];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        input_key.clone(),
        f64_tensor(input_shape.clone(), x_data.clone()),
    );

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(input_shape.clone(), dx_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &x_data, &dx_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
    });
}

#[test]
fn test_reduce_max_vjp() {
    let op = StdTensorOp::ReduceMax { axes: vec![0] };
    let input_shape = vec![4usize];
    let input_key = tensor_input_key(60_003);
    let x_data = vec![1.0, 3.0, 3.0, -2.0];
    let cotangent = 2.5;
    let grad = transpose_primal_unary_op_with_inputs(
        op.clone(),
        input_key,
        f64_tensor(input_shape.clone(), x_data.clone()),
        scalar_f64_tensor(cotangent),
    );

    assert_grad_matches_finite_diff(get_f64_data(&grad), &x_data, |xs| {
        cotangent * eval_f64_reduction_op(&op, &input_shape, xs)[0]
    });
}

#[test]
fn test_reduce_min_jvp() {
    let op = StdTensorOp::ReduceMin { axes: vec![0] };
    let input_shape = vec![4usize];
    let (fragment, input_key, output_key) =
        build_unary_fragment(op.clone(), tensor_input_key(60_004));
    let x_data = vec![1.0, -2.0, 4.0, -2.0];
    let dx_data = vec![0.5, 1.0, -1.0, -0.5];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        input_key.clone(),
        f64_tensor(input_shape.clone(), x_data.clone()),
    );

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(input_shape.clone(), dx_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &x_data, &dx_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
    });
}

#[test]
fn test_reduce_min_vjp() {
    let op = StdTensorOp::ReduceMin { axes: vec![0] };
    let input_shape = vec![2usize, 3];
    let input_key = tensor_input_key(60_005);
    let x_data = vec![-4.0, -4.0, 0.5, 2.0, 1.0, 1.0];
    let cotangent = vec![1.5, -0.25, 0.75];
    let grad = transpose_primal_unary_op_with_inputs(
        op.clone(),
        input_key,
        f64_tensor(input_shape.clone(), x_data.clone()),
        f64_tensor(vec![3], cotangent.clone()),
    );

    assert_grad_matches_finite_diff(get_f64_data(&grad), &x_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
            .iter()
            .zip(cotangent.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    });
}

#[test]
fn grad_broadcast_reduce() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = x.broadcast_in_dim(&[3, 3], &[0]);
    let loss = y.reduce_sum(&[0, 1]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[3.0, 3.0, 3.0]);
}

#[test]
fn grad_broadcast_add_singleton_lhs() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![1], vec![1.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let loss = (&a + &b).sum(&[0]);
    let grad = loss.grad(&a).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[3.0]);
}

#[test]
fn grad_reshape() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let y = x.reshape(&[2, 2]);
    let loss = y.reduce_sum(&[0, 1]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[4]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_transpose() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let y = a.transpose(&[1, 0]);
    let loss = y.reduce_sum(&[0, 1]);
    let grad = loss.grad(&a).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[2, 3]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_exp() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.exp().sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| x.exp()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.exp().sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_log() {
    let x_data = vec![0.8, 1.5, 2.4];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.log().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 / x).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.log().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_sin_cos() {
    let x_data = vec![0.2, -0.7, 1.3];

    let x_sin = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let sin_loss = x_sin.sin().reduce_sum(&[0]);
    let sin_grad = sin_loss.grad(&x_sin).unwrap();
    let sin_grad_tensor = eval_tensor(sin_grad);
    let sin_grad_data = get_f64_data(&sin_grad_tensor);
    let expected_sin: Vec<f64> = x_data.iter().map(|x| x.cos()).collect();
    assert_close_slice(sin_grad_data, &expected_sin);

    let f_sin = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.sin().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(sin_grad_data, &x_data, f_sin);

    let x_cos = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let cos_loss = x_cos.cos().reduce_sum(&[0]);
    let cos_grad = cos_loss.grad(&x_cos).unwrap();
    let cos_grad_tensor = eval_tensor(cos_grad);
    let cos_grad_data = get_f64_data(&cos_grad_tensor);
    let expected_cos: Vec<f64> = x_data.iter().map(|x| -x.sin()).collect();
    assert_close_slice(cos_grad_data, &expected_cos);

    let f_cos = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.cos().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(cos_grad_data, &x_data, f_cos);
}

#[test]
fn grad_div() {
    let x_data = vec![1.2, -2.4, 3.6];
    let y_data = vec![0.5, -1.5, 2.0];

    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], y_data.clone()));
    let loss = (&x / &y).reduce_sum(&[0]);

    let grad_x = loss.grad(&x).unwrap();
    let grad_y = loss.grad(&y).unwrap();
    let grad_x_tensor = eval_tensor(grad_x);
    let grad_y_tensor = eval_tensor(grad_y);
    let grad_x_data = get_f64_data(&grad_x_tensor);
    let grad_y_data = get_f64_data(&grad_y_tensor);

    let expected_x: Vec<f64> = y_data.iter().map(|y| 1.0 / y).collect();
    let expected_y: Vec<f64> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(x, y)| -x / (y * y))
        .collect();
    assert_close_slice(grad_x_data, &expected_x);
    assert_close_slice(grad_y_data, &expected_y);

    let f = |lhs: &[f64], rhs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar((&x / &y).reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_lhs(grad_x_data, &x_data, &y_data, &f);
    assert_grad_matches_finite_diff_rhs(grad_y_data, &x_data, &y_data, &f);
}

#[test]
fn grad_sqrt() {
    let x_data = vec![0.8, 1.5, 3.2];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.sqrt().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 0.5 / x.sqrt()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.sqrt().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_tanh() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.tanh().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 - x.tanh().powi(2)).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.tanh().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_pow() {
    let x_data = vec![0.7, 1.3, 2.1];
    let y_data = vec![2.0, 2.0, 2.0];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], y_data.clone()));
    let loss = x.pow(&y).reduce_sum(&[0]);

    let grad_x = loss.grad(&x).unwrap();
    let grad_x_tensor = eval_tensor(grad_x);
    let grad_x_data = get_f64_data(&grad_x_tensor);
    let expected_x: Vec<f64> = x_data.iter().map(|x| 2.0 * x).collect();
    assert_close_slice(grad_x_data, &expected_x);

    let f = |lhs: &[f64], rhs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar(x.pow(&y).reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_lhs(grad_x_data, &x_data, &y_data, &f);
}

#[test]
fn grad_pow_wrt_exponent() {
    let x_data = vec![1.2, 1.8, 2.5];
    let y_data = vec![0.5, 1.5, 2.0];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], y_data.clone()));
    let loss = x.pow(&y).reduce_sum(&[0]);

    let grad_y = loss.grad(&y).unwrap();
    let grad_y_tensor = eval_tensor(grad_y);
    let grad_y_data = get_f64_data(&grad_y_tensor);
    let expected_y: Vec<f64> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(x, y)| x.ln() * x.powf(*y))
        .collect();
    assert_close_slice(grad_y_data, &expected_y);

    let f = |lhs: &[f64], rhs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar(x.pow(&y).reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_rhs(grad_y_data, &x_data, &y_data, &f);
}

#[test]
fn grad_abs() {
    let x_data = vec![-1.7, 0.8, 2.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.abs().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected = [-1.0, 1.0, 1.0];
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.abs().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_sign() {
    let x_data = vec![-1.7, 0.8, 2.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.sign().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    assert_close_slice(grad_data, &[0.0, 0.0, 0.0]);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.sign().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_rsqrt() {
    let x_data = vec![0.8, 1.5, 3.2];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.rsqrt().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| -0.5 / (x * x.sqrt())).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.rsqrt().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_expm1() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.expm1().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| x.exp()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.expm1().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_log1p() {
    let x_data = vec![0.2, 0.7, 1.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.log1p().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 / (x + 1.0)).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.log1p().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

fn build_gather_reduce_sum_fragment(
    operand_key: TensorInputKey,
    indices_key: TensorInputKey,
    config: GatherConfig,
    reduce_axes: Vec<usize>,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let operand = builder.add_input(operand_key);
    let indices = builder.add_input(indices_key);
    let gathered = builder.add_op(
        StdTensorOp::Gather(config),
        vec![ValRef::Local(operand), ValRef::Local(indices)],
        OpMode::Primal,
    )[0];
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: reduce_axes },
        vec![ValRef::Local(gathered)],
        OpMode::Primal,
    )[0];
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), loss_key)
}

#[test]
fn grad_gather_reduce_sum_accumulates_indices_correctly() {
    // operand is rank-1 with five distinct values; gather reads indices
    // [2, 4, 0, 2, 2] so the gradient of the summed output w.r.t. the
    // operand should count the number of times each index is read: index
    // 0 once, index 2 three times, index 4 once, and other slots zero.
    let operand_key = tensor_input_key(60_000);
    let indices_key = tensor_input_key(60_001);
    let config = GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let (fragment, loss_key) =
        build_gather_reduce_sum_fragment(operand_key.clone(), indices_key.clone(), config, vec![0]);

    let operand_data = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
    let indices_data = vec![2_i64, 4, 0, 2, 2];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        operand_key.clone(),
        f64_tensor(vec![5], operand_data.clone()),
    );
    inputs_map.insert(indices_key, i64_tensor(vec![5, 1], indices_data));

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, operand_key, inputs_map);
    assert_eq!(grad.shape(), &[5]);
    assert_close_slice(get_f64_data(&grad), &[1.0, 0.0, 3.0, 0.0, 1.0]);
}

#[test]
fn grad_traced_index_select_repeated_positions_accumulates() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let weights =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![10.0, 20.0, 30.0]));

    let selected = x.index_select(0, &[1, 1, 2]).unwrap();
    let loss = (&selected * &weights).reduce_sum(&[0]);
    let grad = eval_tensor(loss.grad(&x).unwrap());

    assert_eq!(grad.shape(), &[3]);
    assert_close_slice(get_f64_data(&grad), &[0.0, 30.0, 30.0]);
}

#[test]
fn jvp_traced_index_select_gathers_tangent() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![0.5, 1.5, 2.5, 3.5]));

    let y = x.index_select(0, &[3, 1, 3]).unwrap();
    let tangent_y = eval_tensor(y.jvp(&x, &tangent));

    assert_eq!(tangent_y.shape(), &[3]);
    assert_close_slice(get_f64_data(&tangent_y), &[3.5, 1.5, 3.5]);
}

/// Build `y = ReduceSum(Scatter(operand, indices, updates, config))`,
/// where the Scatter output has the same shape as `operand`.
fn build_scatter_reduce_sum_fragment(
    operand_key: TensorInputKey,
    indices_key: TensorInputKey,
    updates_key: TensorInputKey,
    config: ScatterConfig,
    reduce_axes: Vec<usize>,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let operand = builder.add_input(operand_key);
    let indices = builder.add_input(indices_key);
    let updates = builder.add_input(updates_key);
    let scattered = builder.add_op(
        StdTensorOp::Scatter(config),
        vec![
            ValRef::Local(operand),
            ValRef::Local(indices),
            ValRef::Local(updates),
        ],
        OpMode::Primal,
    )[0];
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: reduce_axes },
        vec![ValRef::Local(scattered)],
        OpMode::Primal,
    )[0];
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), loss_key)
}

fn build_weighted_unary_sum_fragment(
    input_key: TensorInputKey,
    weights_key: TensorInputKey,
    op: StdTensorOp,
    reduce_axes: Vec<usize>,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let input = builder.add_input(input_key);
    let weights = builder.add_input(weights_key);
    let output = builder.add_op(op, vec![ValRef::Local(input)], OpMode::Primal)[0];
    let weighted = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(output), ValRef::Local(weights)],
        OpMode::Primal,
    )[0];
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: reduce_axes },
        vec![ValRef::Local(weighted)],
        OpMode::Primal,
    )[0];
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), loss_key)
}

fn build_dynamic_slice_fragment(
    input_key: TensorInputKey,
    starts_key: TensorInputKey,
    slice_sizes: Vec<usize>,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let input = builder.add_input(input_key);
    let starts = builder.add_input(starts_key);
    let output = builder.add_op(
        StdTensorOp::DynamicSlice { slice_sizes },
        vec![ValRef::Local(input), ValRef::Local(starts)],
        OpMode::Primal,
    )[0];
    let output_key = builder.global_key(output).clone();
    builder.set_outputs(vec![output]);
    (Arc::new(builder.build()), output_key)
}

fn build_dynamic_update_slice_fragment(
    operand_key: TensorInputKey,
    update_key: TensorInputKey,
    starts_key: TensorInputKey,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let operand = builder.add_input(operand_key);
    let update = builder.add_input(update_key);
    let starts = builder.add_input(starts_key);
    let output = builder.add_op(
        StdTensorOp::DynamicUpdateSlice,
        vec![
            ValRef::Local(operand),
            ValRef::Local(update),
            ValRef::Local(starts),
        ],
        OpMode::Primal,
    )[0];
    let output_key = builder.global_key(output).clone();
    builder.set_outputs(vec![output]);
    (Arc::new(builder.build()), output_key)
}

#[test]
fn grad_scatter_reduce_sum_wrt_updates_is_ones() {
    // `y = reduce_sum(scatter(operand, indices, updates, config))`. The
    // scatter backward feeds `cot_out` (a ones tensor of operand shape)
    // into the inverse Gather at each `scatter_indices` entry. With all
    // indices in range, the updates gradient is ones of the updates shape.
    let operand_key = tensor_input_key(61_000);
    let indices_key = tensor_input_key(61_001);
    let updates_key = tensor_input_key(61_002);
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let (fragment, loss_key) = build_scatter_reduce_sum_fragment(
        operand_key.clone(),
        indices_key.clone(),
        updates_key.clone(),
        config,
        vec![0],
    );

    let operand_data = vec![0.0_f64, 0.0, 0.0, 0.0];
    let indices_data = vec![1_i64, 3, 0];
    let updates_data = vec![5.0_f64, 7.0, 9.0];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(operand_key, f64_tensor(vec![4], operand_data));
    inputs_map.insert(indices_key, i64_tensor(vec![3, 1], indices_data));
    inputs_map.insert(
        updates_key.clone(),
        f64_tensor(vec![3], updates_data.clone()),
    );

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, updates_key, inputs_map);
    assert_eq!(grad.shape(), &[3]);
    // reduce_sum over the scatter output contributes a 1 to each updated
    // slot; the inverse Gather reads one value for each `indices` entry.
    assert_close_slice(get_f64_data(&grad), &[1.0, 1.0, 1.0]);
}

#[test]
fn jvp_dynamic_slice_matches_finite_diff() {
    let input_key = tensor_input_key(61_500);
    let starts_key = tensor_input_key(61_501);
    let (fragment, output_key) =
        build_dynamic_slice_fragment(input_key.clone(), starts_key.clone(), vec![3]);

    let input_data = vec![0.5_f64, -1.0, 2.5, 4.0, -3.0];
    let starts_data = vec![1_i64];
    let tangent_data = vec![1.25_f64, -0.75, 3.0, 2.5, -1.0];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![5], input_data.clone())),
        (starts_key, i64_tensor(vec![1], starts_data)),
    ]);

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(vec![5], tangent_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &input_data, &tangent_data, |xs| {
        xs[1..4].to_vec()
    });
}

#[test]
fn grad_dynamic_slice_clamped_start_matches_finite_diff() {
    let input_key = tensor_input_key(61_510);
    let starts_key = tensor_input_key(61_511);
    let (fragment, output_key) =
        build_dynamic_slice_fragment(input_key.clone(), starts_key.clone(), vec![3]);

    let input_data = vec![0.5_f64, -1.0, 2.5, 4.0, -3.0];
    let starts_data = vec![4_i64];
    let cotangent_data = vec![0.5_f64, -1.0, 2.0];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![5], input_data.clone())),
        (starts_key, i64_tensor(vec![1], starts_data)),
    ]);

    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(vec![3], cotangent_data.clone()),
    );

    let loss = |xs: &[f64]| {
        xs[2..5]
            .iter()
            .zip(cotangent_data.iter())
            .map(|(&value, &weight)| value * weight)
            .sum()
    };
    let expected: Vec<f64> = (0..input_data.len())
        .map(|idx| finite_diff_scalar(&loss, &input_data, idx, 1.0e-6))
        .collect();
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn jvp_dynamic_update_slice_update_matches_finite_diff() {
    let operand_key = tensor_input_key(61_520);
    let update_key = tensor_input_key(61_521);
    let starts_key = tensor_input_key(61_522);
    let (fragment, output_key) = build_dynamic_update_slice_fragment(
        operand_key.clone(),
        update_key.clone(),
        starts_key.clone(),
    );

    let operand_data = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0];
    let update_data = vec![1.0_f64, 2.0, 3.0];
    let starts_data = vec![4_i64];
    let tangent_data = vec![0.5_f64, -1.0, 2.0];
    let inputs_map = HashMap::from([
        (operand_key, f64_tensor(vec![5], operand_data.clone())),
        (update_key.clone(), f64_tensor(vec![3], update_data.clone())),
        (starts_key, i64_tensor(vec![1], starts_data)),
    ]);

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        update_key,
        inputs_map,
        f64_tensor(vec![3], tangent_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &update_data, &tangent_data, |upd| {
        let mut out = operand_data.clone();
        out[2..5].copy_from_slice(upd);
        out
    });
}

#[test]
fn grad_dynamic_update_slice_matches_finite_diff() {
    let operand_key = tensor_input_key(61_530);
    let update_key = tensor_input_key(61_531);
    let starts_key = tensor_input_key(61_532);
    let (fragment, output_key) = build_dynamic_update_slice_fragment(
        operand_key.clone(),
        update_key.clone(),
        starts_key.clone(),
    );

    let operand_data = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0];
    let update_data = vec![1.0_f64, 2.0, 3.0];
    let starts_data = vec![4_i64];
    let cotangent_data = vec![0.5_f64, -0.25, 1.0, 2.0, -1.5];
    let inputs_map = HashMap::from([
        (
            operand_key.clone(),
            f64_tensor(vec![5], operand_data.clone()),
        ),
        (update_key.clone(), f64_tensor(vec![3], update_data.clone())),
        (starts_key, i64_tensor(vec![1], starts_data)),
    ]);

    let grad_operand = grad_from_fragment_with_inputs_and_cotangent(
        fragment.clone(),
        output_key.clone(),
        operand_key,
        inputs_map.clone(),
        f64_tensor(vec![5], cotangent_data.clone()),
    );
    let grad_update = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        output_key,
        update_key,
        inputs_map,
        f64_tensor(vec![5], cotangent_data.clone()),
    );

    let loss_with = |operand: &[f64], update: &[f64]| {
        let mut out = operand.to_vec();
        out[2..5].copy_from_slice(update);
        out.iter()
            .zip(cotangent_data.iter())
            .map(|(&value, &weight)| value * weight)
            .sum()
    };
    let expected_operand: Vec<f64> = (0..operand_data.len())
        .map(|idx| finite_diff_scalar_lhs(&loss_with, &operand_data, &update_data, idx, 1.0e-6))
        .collect();
    let expected_update: Vec<f64> = (0..update_data.len())
        .map(|idx| finite_diff_scalar_rhs(&loss_with, &operand_data, &update_data, idx, 1.0e-6))
        .collect();

    assert_close_slice(get_f64_data(&grad_operand), &expected_operand);
    assert_close_slice(get_f64_data(&grad_update), &expected_update);
}

#[test]
fn grad_slice_weighted_sum_matches_finite_diff() {
    let input_key = tensor_input_key(62_000);
    let weights_key = tensor_input_key(62_001);
    let config = SliceConfig {
        starts: vec![1],
        limits: vec![5],
        strides: vec![2],
    };
    let (fragment, loss_key) = build_weighted_unary_sum_fragment(
        input_key.clone(),
        weights_key.clone(),
        StdTensorOp::Slice(config),
        vec![0],
    );

    let input_data = vec![0.5_f64, -1.0, 2.5, 4.0, -3.0];
    let weights_data = vec![1.25_f64, -0.75];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![5], input_data.clone())),
        (weights_key, f64_tensor(vec![2], weights_data.clone())),
    ]);

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, input_key, inputs_map);
    assert_grad_matches_finite_diff(get_f64_data(&grad), &input_data, |xs| {
        xs[1] * weights_data[0] + xs[3] * weights_data[1]
    });
}

#[test]
fn grad_pad_weighted_sum_matches_finite_diff() {
    let input_key = tensor_input_key(63_000);
    let weights_key = tensor_input_key(63_001);
    let config = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![2],
        interior_padding: vec![1],
    };
    let (fragment, loss_key) = build_weighted_unary_sum_fragment(
        input_key.clone(),
        weights_key.clone(),
        StdTensorOp::Pad(config),
        vec![0],
    );

    let input_data = vec![2.0_f64, -1.5, 0.25];
    let weights_data = vec![0.5_f64, 1.25, -0.5, 2.0, 0.75, -1.0, 3.0, -2.5];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![3], input_data.clone())),
        (weights_key, f64_tensor(vec![8], weights_data.clone())),
    ]);

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, input_key, inputs_map);
    assert_grad_matches_finite_diff(get_f64_data(&grad), &input_data, |xs| {
        xs[0] * weights_data[1] + xs[1] * weights_data[3] + xs[2] * weights_data[5]
    });
}

#[test]
fn grad_reverse_weighted_sum_matches_finite_diff() {
    let input_key = tensor_input_key(64_000);
    let weights_key = tensor_input_key(64_001);
    let (fragment, loss_key) = build_weighted_unary_sum_fragment(
        input_key.clone(),
        weights_key.clone(),
        StdTensorOp::Reverse { axes: vec![0] },
        vec![0],
    );

    let input_data = vec![1.0_f64, -2.0, 3.5, 0.25];
    let weights_data = vec![0.5_f64, -1.0, 2.0, 1.5];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![4], input_data.clone())),
        (weights_key, f64_tensor(vec![4], weights_data.clone())),
    ]);

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, input_key, inputs_map);
    assert_grad_matches_finite_diff(get_f64_data(&grad), &input_data, |xs| {
        xs[0] * weights_data[3]
            + xs[1] * weights_data[2]
            + xs[2] * weights_data[1]
            + xs[3] * weights_data[0]
    });
}

#[test]
fn dropped_traced_graph_releases_registered_metadata() {
    let leaf_key;
    let derived_key;
    let y;

    {
        let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]);
        leaf_key = GlobalValKey::Input(x.input_key().expect("leaf input key"));

        y = &x + &x;
        derived_key = y.fragment.vals()[y.val].key.clone();

        assert!(lookup_global_metadata(&leaf_key).is_some());
        assert!(lookup_global_metadata(&derived_key).is_some());
    }

    assert!(lookup_global_metadata(&leaf_key).is_some());
    assert!(lookup_global_metadata(&derived_key).is_some());

    drop(y);

    assert!(lookup_global_metadata(&leaf_key).is_none());
    assert!(lookup_global_metadata(&derived_key).is_none());
}
