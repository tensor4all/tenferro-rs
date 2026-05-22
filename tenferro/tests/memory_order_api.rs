use tenferro::{Tensor, TracedTensor};

#[test]
fn traced_tensor_row_major_constructor_stores_column_major_input() {
    let traced =
        TracedTensor::from_vec_row_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let compiled = traced.compile_with_inputs(&[]).unwrap();
    assert_eq!(compiled.inputs.len(), 1);
    assert_eq!(compiled.inputs[0].shape(), &[2, 3]);
    assert_eq!(
        compiled.inputs[0].as_slice::<f64>().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );
}

#[test]
fn traced_tensor_col_major_constructor_keeps_physical_order() {
    let traced = TracedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 3, 2, 4]);

    let compiled = traced.compile_with_inputs(&[]).unwrap();
    assert_eq!(
        compiled.inputs[0]
            .clone()
            .try_into_vec_col_major::<i64>()
            .unwrap(),
        (vec![2, 2], vec![1, 3, 2, 4]),
    );
}

#[test]
fn tensor_row_major_constructor_remains_available_through_facade() {
    let tensor = Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
}
