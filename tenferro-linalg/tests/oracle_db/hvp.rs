use std::collections::BTreeMap;

use tenferro_tensor::Tensor;

use crate::decode::{tensor_data_col_major, tensor_from_col_major, OracleDbScalar};

fn tensor_add_scaled(
    base: &Tensor<f64>,
    direction: &Tensor<f64>,
    scale: f64,
) -> Result<Tensor<f64>, String> {
    if base.dims() != direction.dims() {
        return Err(format!(
            "tensor perturbation shape mismatch: base {:?}, direction {:?}",
            base.dims(),
            direction.dims()
        ));
    }
    let base_data = tensor_data_col_major(base);
    let direction_data = tensor_data_col_major(direction);
    let data: Vec<f64> = base_data
        .iter()
        .zip(direction_data.iter())
        .map(|(x, dx)| x + scale * dx)
        .collect();
    Ok(tensor_from_col_major(data, base.dims()))
}

pub fn perturb_input_map(
    base_inputs: &BTreeMap<String, Tensor<f64>>,
    direction: &BTreeMap<String, Tensor<f64>>,
    scale: f64,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    if base_inputs.keys().collect::<Vec<_>>() != direction.keys().collect::<Vec<_>>() {
        return Err(format!(
            "input perturbation key mismatch: base {:?}, direction {:?}",
            base_inputs.keys().collect::<Vec<_>>(),
            direction.keys().collect::<Vec<_>>()
        ));
    }
    let mut out = BTreeMap::new();
    for (name, tensor) in base_inputs {
        out.insert(
            name.clone(),
            tensor_add_scaled(tensor, direction.get(name).unwrap(), scale)?,
        );
    }
    Ok(out)
}

pub fn central_diff_tensor_maps(
    plus: &BTreeMap<String, Tensor<f64>>,
    minus: &BTreeMap<String, Tensor<f64>>,
    step: f64,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    if step <= 0.0 {
        return Err(format!(
            "central difference requires positive step, got {step}"
        ));
    }
    if plus.keys().collect::<Vec<_>>() != minus.keys().collect::<Vec<_>>() {
        return Err(format!(
            "central-diff key mismatch: plus {:?}, minus {:?}",
            plus.keys().collect::<Vec<_>>(),
            minus.keys().collect::<Vec<_>>()
        ));
    }
    let mut out = BTreeMap::new();
    for (name, plus_tensor) in plus {
        let minus_tensor = minus.get(name).unwrap();
        if plus_tensor.dims() != minus_tensor.dims() {
            return Err(format!(
                "central-diff shape mismatch for {name}: plus {:?}, minus {:?}",
                plus_tensor.dims(),
                minus_tensor.dims()
            ));
        }
        let plus_data = tensor_data_col_major(plus_tensor);
        let minus_data = tensor_data_col_major(minus_tensor);
        let data: Vec<f64> = plus_data
            .iter()
            .zip(minus_data.iter())
            .map(|(p, m)| (p - m) / (2.0 * step))
            .collect();
        out.insert(
            name.clone(),
            tensor_from_col_major(data, plus_tensor.dims()),
        );
    }
    Ok(out)
}

fn tensor_add_scaled_typed<T: OracleDbScalar>(
    base: &Tensor<T>,
    direction: &Tensor<T>,
    scale: f64,
) -> Result<Tensor<T>, String> {
    if base.dims() != direction.dims() {
        return Err(format!(
            "tensor perturbation shape mismatch: base {:?}, direction {:?}",
            base.dims(),
            direction.dims()
        ));
    }
    let scale_real = num_traits::NumCast::from(scale)
        .ok_or_else(|| format!("failed to cast perturbation scale {scale}"))?;
    let scale_t = T::from_real(scale_real);
    let base_data = tensor_data_col_major(base);
    let direction_data = tensor_data_col_major(direction);
    let data: Vec<T> = base_data
        .iter()
        .zip(direction_data.iter())
        .map(|(x, dx)| *x + *dx * scale_t)
        .collect();
    Ok(tensor_from_col_major(data, base.dims()))
}

pub fn perturb_input_map_typed<T: OracleDbScalar>(
    base_inputs: &BTreeMap<String, Tensor<T>>,
    direction: &BTreeMap<String, Tensor<T>>,
    scale: f64,
) -> Result<BTreeMap<String, Tensor<T>>, String> {
    if base_inputs.keys().collect::<Vec<_>>() != direction.keys().collect::<Vec<_>>() {
        return Err(format!(
            "input perturbation key mismatch: base {:?}, direction {:?}",
            base_inputs.keys().collect::<Vec<_>>(),
            direction.keys().collect::<Vec<_>>()
        ));
    }
    let mut out = BTreeMap::new();
    for (name, tensor) in base_inputs {
        out.insert(
            name.clone(),
            tensor_add_scaled_typed(tensor, direction.get(name).unwrap(), scale)?,
        );
    }
    Ok(out)
}

pub fn central_diff_tensor_maps_typed<T: OracleDbScalar>(
    plus: &BTreeMap<String, Tensor<T>>,
    minus: &BTreeMap<String, Tensor<T>>,
    step: f64,
) -> Result<BTreeMap<String, Tensor<T>>, String> {
    if step <= 0.0 {
        return Err(format!(
            "central difference requires positive step, got {step}"
        ));
    }
    if plus.keys().collect::<Vec<_>>() != minus.keys().collect::<Vec<_>>() {
        return Err(format!(
            "central-diff key mismatch: plus {:?}, minus {:?}",
            plus.keys().collect::<Vec<_>>(),
            minus.keys().collect::<Vec<_>>()
        ));
    }
    let scale_real = num_traits::NumCast::from(1.0f64 / (2.0 * step))
        .ok_or_else(|| format!("failed to cast central-diff scale from step {step}"))?;
    let scale_t = T::from_real(scale_real);
    let mut out = BTreeMap::new();
    for (name, plus_tensor) in plus {
        let minus_tensor = minus.get(name).unwrap();
        if plus_tensor.dims() != minus_tensor.dims() {
            return Err(format!(
                "central-diff shape mismatch for {name}: plus {:?}, minus {:?}",
                plus_tensor.dims(),
                minus_tensor.dims()
            ));
        }
        let plus_data = tensor_data_col_major(plus_tensor);
        let minus_data = tensor_data_col_major(minus_tensor);
        let data: Vec<T> = plus_data
            .iter()
            .zip(minus_data.iter())
            .map(|(p, m)| (*p - *m) * scale_t)
            .collect();
        out.insert(
            name.clone(),
            tensor_from_col_major(data, plus_tensor.dims()),
        );
    }
    Ok(out)
}
