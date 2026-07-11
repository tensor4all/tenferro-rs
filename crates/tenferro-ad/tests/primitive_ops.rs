mod support;
use support::{run_many_traced_with, RunTraced};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::traced::TracedTensor;
use tenferro_runtime::DType;
use tenferro_runtime::GraphExecutor;
use tenferro_tensor::{DotGeneralConfig, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn i64_tensor(shape: Vec<usize>, data: Vec<i64>) -> Tensor {
    Tensor::I64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn get_f64_data(t: &Tensor) -> &[f64] {
    match t {
        Tensor::F64(inner) => inner.host_data().unwrap(),
        _ => panic!("expected F64"),
    }
}

fn get_i64_data(t: &Tensor) -> &[i64] {
    match t {
        Tensor::I64(inner) => inner.host_data().unwrap(),
        _ => panic!("expected I64"),
    }
}

#[test]
fn test_add() {
    let a = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb = TracedTensor::from_tensor_concrete_shape(b).unwrap();
    let tc = (&ta + &tb).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_add_broadcast_scalar_plus_vector() {
    let scalar = f64_tensor(vec![], vec![1.0]);
    let vector = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(scalar).unwrap();
    let tb = TracedTensor::from_tensor_concrete_shape(vector).unwrap();
    let tc = (&ta + &tb).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(get_f64_data(&result), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_mul() {
    let a = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb = TracedTensor::from_tensor_concrete_shape(b).unwrap();
    let tc = (&ta * &tb).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[4.0, 10.0, 18.0]);
}

#[test]
fn test_mul_broadcast_column_times_row() {
    let column = f64_tensor(vec![3, 1], vec![1.0, 2.0, 3.0]);
    let row = f64_tensor(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(column).unwrap();
    let tb = TracedTensor::from_tensor_concrete_shape(row).unwrap();
    let tc = (&ta * &tb).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[3, 4]);
    assert_eq!(
        get_f64_data(&result),
        &[10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0, 40.0, 80.0, 120.0]
    );
}

#[test]
fn test_div_broadcast_vector_by_scalar() {
    let vector = f64_tensor(vec![3], vec![2.0, 4.0, 8.0]);
    let scalar = f64_tensor(vec![], vec![2.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(vector).unwrap();
    let tb = TracedTensor::from_tensor_concrete_shape(scalar).unwrap();
    let tc = (&ta / &tb).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[1.0, 2.0, 4.0]);
}

#[test]
fn test_scale_real() {
    let x =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let y = x.scale_real(2.0).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[2.0, 4.0, 6.0]);
}

#[test]
fn test_scale_real_operator_overload() {
    let x =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let y = (&x * 3.0).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[3.0, 6.0, 9.0]);
}

#[test]
fn test_scale_real_i64_rounds_factor() {
    let x = TracedTensor::from_tensor_concrete_shape(i64_tensor(vec![3], vec![1, 2, -3])).unwrap();
    let y = x.scale_real(2.7).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(get_i64_data(&result), &[3, 6, -9]);
}

#[test]
fn test_pow_broadcast_vector_with_scalar_exponent() {
    let base =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![2.0, 3.0, 4.0])).unwrap();
    let exp = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![], vec![2.0])).unwrap();
    let out = base.pow(&exp).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = out.run_with(&mut engine).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(get_f64_data(&result), &[4.0, 9.0, 16.0]);
}

#[test]
fn test_neg() {
    let a = f64_tensor(vec![3], vec![1.0, -2.0, 3.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb = (-&ta).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tb.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[-1.0, 2.0, -3.0]);
}

#[test]
fn traced_abs_of_complex_tensor_returns_real_tensor() {
    use num_complex::Complex64;

    let x = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(3.0, 4.0), Complex64::new(5.0, 12.0)],
        )
        .unwrap(),
    )
    .unwrap();
    let y = x.abs().unwrap();
    assert_eq!(y.dtype, DType::F64);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = y.run_with(&mut engine).unwrap();
    assert_eq!(result.dtype(), DType::F64);
    assert_eq!(get_f64_data(&result), &[5.0, 13.0]);
}

#[test]
fn test_dot_general() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb = TracedTensor::from_tensor_concrete_shape(b).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let tc = ta.dot_general(&tb, config).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tc.run_with(&mut engine).unwrap();
    // C = [[22, 49], [28, 64]] col-major: [22, 28, 49, 64]
    assert_eq!(get_f64_data(&result), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_broadcast_reduce() {
    let v = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let tv = TracedTensor::from_tensor_concrete_shape(v).unwrap();
    let tb = tv.broadcast_in_dim(&[3, 2], &[0]).unwrap();
    let tr = tb.reduce_sum(&[1]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tr.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[2.0, 4.0, 6.0]);
}

#[test]
fn test_transpose() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb = ta.transpose(&[1, 0]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tb.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_reshape() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb = ta.reshape(&[6]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = tb.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&result), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn traced_index_select_keeps_indices_integer_for_complex_operand() {
    use num_complex::Complex64;

    let x = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let y = x.index_select(-1, &[2, 0]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let out = y.run_with(&mut engine).unwrap();

    assert_eq!(out.shape(), &[2]);
    assert_eq!(
        out.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(3.0, 0.5), Complex64::new(1.0, 1.0)]
    );
}

#[test]
fn traced_stack_trailing_axis_and_index_select_feed_batched_dot_general() {
    let a0 =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]))
            .unwrap();
    let a1 =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]))
            .unwrap();
    let b0 =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![2.0, 3.0])).unwrap();
    let b1 =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 5.0])).unwrap();

    let a = TracedTensor::stack(&[&a0, &a1], -1).unwrap();
    let b = TracedTensor::stack(&[&b0, &b1], -1)
        .unwrap()
        .index_select(-1, &[1, 0])
        .unwrap();
    let c = a
        .dot_general(
            &b,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![2],
                rhs_batch_dims: vec![2],
            },
        )
        .unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let out = c.run_with(&mut engine).unwrap();

    assert_eq!(out.shape(), &[2, 1, 2]);
    assert_eq!(get_f64_data(&out), &[19.0, 28.0, 31.0, 36.0]);
}

#[test]
fn traced_index_select_rejects_invalid_axis_position_and_symbolic_shape() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![1.0, 2.0])).unwrap();

    let axis_err = x.index_select(1, &[0]).err().unwrap().to_string();
    assert!(axis_err.contains("index_select"), "got: {axis_err}");
    assert!(axis_err.contains("axis"), "got: {axis_err}");

    let position_err = x.index_select(0, &[2]).err().unwrap().to_string();
    assert!(
        position_err.contains("position 2 out of bounds"),
        "got: {position_err}"
    );

    let symbolic = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let shape_err = symbolic.index_select(0, &[0]).err().unwrap().to_string();
    assert!(
        shape_err.contains("concrete shape hint"),
        "got: {shape_err}"
    );
}

#[test]
fn traced_stack_rejects_empty_mismatched_invalid_axis_and_symbolic_shapes() {
    let empty: [&TracedTensor; 0] = [];
    let empty_err = TracedTensor::stack(&empty, 0).err().unwrap().to_string();
    assert!(empty_err.contains("stack requires at least one input"));

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![1.0, 2.0])).unwrap();
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![3.0, 4.0, 5.0])).unwrap();
    let shape_err = TracedTensor::stack(&[&a, &b], -1)
        .err()
        .unwrap()
        .to_string();
    assert!(shape_err.contains("shape mismatch"), "got: {shape_err}");

    let axis_err = TracedTensor::stack(&[&a], 2).err().unwrap().to_string();
    assert!(axis_err.contains("axis"), "got: {axis_err}");

    let symbolic_first = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let first_shape_err = TracedTensor::stack(&[&symbolic_first], -1)
        .err()
        .unwrap()
        .to_string();
    assert!(
        first_shape_err.contains("concrete shape hints"),
        "got: {first_shape_err}"
    );

    let symbolic_second = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let second_shape_err = TracedTensor::stack(&[&a, &symbolic_second], -1)
        .err()
        .unwrap()
        .to_string();
    assert!(
        second_shape_err.contains("concrete shape hints"),
        "got: {second_shape_err}"
    );
}

#[test]
fn traced_stack_positive_axis_promotes_mixed_input_dtypes() {
    let a = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap(),
    )
    .unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![3.0, 4.0])).unwrap();

    let out = TracedTensor::stack(&[&a, &b], 0).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let result = out.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(get_f64_data(&result), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_run_many_traced() {
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![3.0, 4.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb = TracedTensor::from_tensor_concrete_shape(b).unwrap();
    let sum = (&ta + &tb).unwrap();
    let prod = (&ta * &tb).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let results = run_many_traced_with(&mut engine, &[&sum, &prod]).unwrap();
    assert_eq!(get_f64_data(&results[0]), &[4.0, 6.0]);
    assert_eq!(get_f64_data(&results[1]), &[3.0, 8.0]);
}

#[test]
fn test_chained_ops() {
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![3.0, 4.0]);
    let c = f64_tensor(vec![2], vec![10.0, 20.0]);
    let ta = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb = TracedTensor::from_tensor_concrete_shape(b).unwrap();
    let tc = TracedTensor::from_tensor_concrete_shape(c).unwrap();
    let sum = (&ta + &tb).unwrap();
    let result = (&sum * &tc).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let out = result.run_with(&mut engine).unwrap();
    assert_eq!(get_f64_data(&out), &[40.0, 120.0]);
}
