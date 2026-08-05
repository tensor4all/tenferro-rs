use tenferro_tensor::{Tensor, TensorValue};

#[test]
fn tensor_value_can_consume_its_owned_bundle_without_copying() -> tenferro_tensor::Result<()> {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let value = TensorValue::from_tensor(tensor);
    let tensor = value.into_tensor()?;

    assert_eq!(tensor.as_slice::<f64>()?, &[1.0, 2.0]);
    Ok(())
}

#[test]
fn tensor_value_view_requires_explicit_materialization() -> tenferro_tensor::Result<()> {
    let tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    let value = TensorValue::from_parts(tensor, vec![2, 2], vec![2, 1], 0)?;

    assert!(value.is_view());
    assert!(value.into_tensor().is_err());
    Ok(())
}
