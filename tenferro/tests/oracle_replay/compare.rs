use tenferro::Tensor;

use crate::decode::{tensor_data_as_col_major, TensorData};

pub fn allclose(actual: &[f64], expected: &[f64], rtol: f64, atol: f64) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: actual {} vs expected {}",
            actual.len(),
            expected.len()
        ));
    }

    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        if actual_value == expected_value {
            continue;
        }
        if actual_value.is_nan() && expected_value.is_nan() {
            continue;
        }

        let diff = (actual_value - expected_value).abs();
        let limit = atol + rtol * expected_value.abs();
        if diff > limit {
            return Err(format!(
                "mismatch at flat index {index}: actual={actual_value}, expected={expected_value}, diff={diff}, limit={limit}"
            ));
        }
    }

    Ok(())
}

pub fn compare_tensor(
    actual: &Tensor,
    expected: &TensorData,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if actual.shape() != expected.shape.as_slice() {
        return Err(format!(
            "shape mismatch: actual {:?} vs expected {:?}",
            actual.shape(),
            expected.shape
        ));
    }

    let expected_data = tensor_data_as_col_major(expected)?;
    match actual {
        Tensor::F64(inner) => allclose(inner.host_data(), &expected_data, rtol, atol),
        _ => Err(format!("expected F64 tensor, got {:?}", actual.dtype())),
    }
}
