use num_complex::Complex64;
use tenferro::Tensor;

use crate::decode::{complex_tensor_data_as_col_major, tensor_data_as_col_major, TensorData};

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

    match actual {
        Tensor::F64(inner) => {
            let expected_data = tensor_data_as_col_major(expected)?;
            allclose(inner.host_data(), &expected_data, rtol, atol)
        }
        Tensor::C64(inner) => compare_complex64(inner.host_data(), expected, rtol, atol),
        _ => Err(format!(
            "unsupported actual tensor dtype {:?}",
            actual.dtype()
        )),
    }
}

fn compare_complex64(
    actual: &[Complex64],
    expected: &TensorData,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    match expected.dtype.as_str() {
        "complex128" => {
            let expected_data = complex_tensor_data_as_col_major(expected)?;
            if actual.len() != expected_data.len() {
                return Err(format!(
                    "length mismatch: actual {} vs expected {}",
                    actual.len(),
                    expected_data.len()
                ));
            }
            for (index, (actual_value, expected_value)) in
                actual.iter().zip(expected_data.iter()).enumerate()
            {
                compare_component(actual_value.re, expected_value.re, rtol, atol, index, "re")?;
                compare_component(actual_value.im, expected_value.im, rtol, atol, index, "im")?;
            }
            Ok(())
        }
        "float64" => {
            let expected_data = tensor_data_as_col_major(expected)?;
            if actual.len() != expected_data.len() {
                return Err(format!(
                    "length mismatch: actual {} vs expected {}",
                    actual.len(),
                    expected_data.len()
                ));
            }
            for (index, (actual_value, expected_value)) in
                actual.iter().zip(expected_data.iter()).enumerate()
            {
                compare_component(actual_value.re, *expected_value, rtol, atol, index, "re")?;
                compare_component(actual_value.im, 0.0, rtol, atol, index, "im")?;
            }
            Ok(())
        }
        other => Err(format!("unsupported expected tensor dtype {other}")),
    }
}

fn compare_component(
    actual: f64,
    expected: f64,
    rtol: f64,
    atol: f64,
    index: usize,
    component: &str,
) -> Result<(), String> {
    if actual == expected {
        return Ok(());
    }
    if actual.is_nan() && expected.is_nan() {
        return Ok(());
    }

    let diff = (actual - expected).abs();
    let limit = atol + rtol * expected.abs();
    if diff > limit {
        return Err(format!(
            "mismatch at flat index {index} component {component}: actual={actual}, expected={expected}, diff={diff}, limit={limit}"
        ));
    }
    Ok(())
}
