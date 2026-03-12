use num_traits::Zero;
use tenferro_device::{Error, Result};
use tenferro_einsum::Subscripts;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::argmax::ArgmaxTracker;
use crate::prims::{for_each_index, tensor_to_view, unflatten_index};

use super::common::{
    build_mode_values, col_major_flat_index, dims_for_modes, operand_index_from_mode_values,
    validate_modes_in_scope,
};
use super::TropicalScalar;

pub(crate) fn tropical_backward<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    match operands.len() {
        1 => tropical_backward_unary(operands[0], cotangent, tracker, subs, contracted),
        2 => tropical_backward_binary(operands, cotangent, tracker, subs, contracted),
        n => Err(Error::InvalidArgument(format!(
            "tropical backward supports 1 or 2 operands, got {n}"
        ))),
    }
}

fn tropical_backward_unary<T: TropicalScalar>(
    operand: &Tensor<T>,
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    let input_modes = &subs.inputs[0];
    validate_modes_in_scope(
        input_modes,
        &subs.output,
        contracted,
        "unary backward operand",
    )?;
    let cot_view = tensor_to_view(cotangent)?;
    let contracted_dims = contracted
        .iter()
        .map(|m| {
            let pos = input_modes.iter().position(|x| x == m).ok_or_else(|| {
                Error::InvalidArgument(format!("contracted mode {m} not in input"))
            })?;
            Ok(operand.dims()[pos])
        })
        .collect::<Result<Vec<_>>>()?;

    let output_shape = tracker.output_shape();
    let mut grad_data = vec![T::Inner::zero(); operand.len()];
    let mut backward_error: Option<Error> = None;

    for_each_index(output_shape, |out_idx| {
        if backward_error.is_some() {
            return;
        }
        let dout = cot_view.get(out_idx);
        let out_flat = col_major_flat_index(output_shape, out_idx);
        let winner = tracker.indices()[out_flat];
        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(winner, &contracted_dims)
        };
        let mode_values = build_mode_values(&subs.output, out_idx, contracted, &k_idx);
        let input_idx =
            match operand_index_from_mode_values(input_modes, &mode_values, "unary backward input")
            {
                Ok(idx) => idx,
                Err(err) => {
                    backward_error = Some(err);
                    return;
                }
            };
        let input_flat = col_major_flat_index(operand.dims(), &input_idx);
        if let Some(slot) = grad_data.get_mut(input_flat) {
            *slot += dout;
        } else {
            backward_error = Some(Error::InvalidArgument(format!(
                "flat index {input_flat} out of bounds for gradient buffer of size {}",
                grad_data.len()
            )));
        }
    });

    if let Some(err) = backward_error {
        return Err(err);
    }

    let grad = Tensor::<T::Inner>::from_slice(&grad_data, operand.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    Ok(vec![grad])
}

fn tropical_backward_binary<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    cotangent: &Tensor<T::Inner>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<Vec<Tensor<T::Inner>>> {
    let a = operands[0];
    let b = operands[1];
    let a_view = tensor_to_view(a)?;
    let b_view = tensor_to_view(b)?;
    let cot_view = tensor_to_view(cotangent)?;
    let input_modes_a = &subs.inputs[0];
    let input_modes_b = &subs.inputs[1];
    validate_modes_in_scope(
        input_modes_a,
        &subs.output,
        contracted,
        "binary backward operand A",
    )?;
    validate_modes_in_scope(
        input_modes_b,
        &subs.output,
        contracted,
        "binary backward operand B",
    )?;
    let contracted_dims = dims_for_modes(operands, subs, contracted, "contracted")?;

    let output_shape = tracker.output_shape();
    let mut da_data = vec![T::Inner::zero(); a.len()];
    let mut db_data = vec![T::Inner::zero(); b.len()];
    let mut backward_error: Option<Error> = None;

    for_each_index(output_shape, |out_idx| {
        if backward_error.is_some() {
            return;
        }
        let dout = cot_view.get(out_idx);
        let out_flat = col_major_flat_index(output_shape, out_idx);
        let winner = tracker.indices()[out_flat];
        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(winner, &contracted_dims)
        };
        let mode_values = build_mode_values(&subs.output, out_idx, contracted, &k_idx);
        let a_idx = match operand_index_from_mode_values(
            input_modes_a,
            &mode_values,
            "binary backward a",
        ) {
            Ok(idx) => idx,
            Err(err) => {
                backward_error = Some(err);
                return;
            }
        };
        let b_idx = match operand_index_from_mode_values(
            input_modes_b,
            &mode_values,
            "binary backward b",
        ) {
            Ok(idx) => idx,
            Err(err) => {
                backward_error = Some(err);
                return;
            }
        };

        let a_inner = a_view.get(&a_idx).inner();
        let b_inner = b_view.get(&b_idx).inner();
        let da_contrib = T::mul_backward_a(a_inner, b_inner, dout);
        let db_contrib = T::mul_backward_b(a_inner, b_inner, dout);

        let a_flat = col_major_flat_index(a.dims(), &a_idx);
        let b_flat = col_major_flat_index(b.dims(), &b_idx);

        if let Some(slot) = da_data.get_mut(a_flat) {
            *slot += da_contrib;
        } else {
            backward_error = Some(Error::InvalidArgument(format!(
                "flat index {a_flat} out of bounds for da buffer of size {}",
                da_data.len()
            )));
            return;
        }
        if let Some(slot) = db_data.get_mut(b_flat) {
            *slot += db_contrib;
        } else {
            backward_error = Some(Error::InvalidArgument(format!(
                "flat index {b_flat} out of bounds for db buffer of size {}",
                db_data.len()
            )));
        }
    });

    if let Some(err) = backward_error {
        return Err(err);
    }

    let da = Tensor::<T::Inner>::from_slice(&da_data, a.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    let db = Tensor::<T::Inner>::from_slice(&db_data, b.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    Ok(vec![da, db])
}
