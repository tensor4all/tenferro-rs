use super::*;

#[test]
fn pooled_engine_runs_operations() {
    let mut engine = crate::test_support::cpu_engine(2).unwrap();
    let result = engine
        .add_read(
            TensorRead::from_tensor(&Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap(),
            )),
            TensorRead::from_tensor(&Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap(),
            )),
        )
        .unwrap();
    assert_eq!(get_f64(&result, &[0]), 4.0);
    assert_eq!(get_f64(&result, &[1]), 6.0);
}

#[test]
fn operations_are_called_from_the_caller_thread() {
    let mut engine = crate::test_support::cpu_engine(2).unwrap();
    let lhs = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    );
    let rhs = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
    );
    let caller = std::thread::current().id();
    engine
        .add_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .unwrap();
    engine.neg_read(TensorRead::from_tensor(&lhs)).unwrap();
    engine
        .mul_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .unwrap();
    engine
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &DotGeneralConfig {
                lhs_contracting_dims: [0].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        )
        .unwrap();
    assert_eq!(std::thread::current().id(), caller);
}
