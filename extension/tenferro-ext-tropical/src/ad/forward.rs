use num_traits::Zero;
use tenferro_device::{Error, Result};
use tenferro_einsum::Subscripts;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::argmax::ArgmaxTracker;
use crate::prims::{for_each_index, tensor_to_view, unflatten_index};

use super::common::{
    build_mode_values, col_major_flat_index, dims_for_modes, operand_index_from_mode_values,
    validate_mode_dimensions, validate_modes_in_scope,
};
use super::TropicalScalar;

pub(crate) fn tropical_forward_with_argmax<T: TropicalScalar>(
    operands: &[&Tensor<T>],
    subs: &Subscripts,
    contracted: &[u32],
) -> Result<(Tensor<T>, ArgmaxTracker)> {
    let output_shape = dims_for_modes(operands, subs, &subs.output, "output")?;
    let contracted_dims = dims_for_modes(operands, subs, contracted, "contracted")?;
    validate_mode_dimensions(operands, subs, &output_shape, contracted, &contracted_dims)?;
    for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
        validate_modes_in_scope(
            input_modes,
            &subs.output,
            contracted,
            &format!("operand {op_idx}"),
        )?;
    }

    let views: Vec<_> = operands
        .iter()
        .map(|op| tensor_to_view(*op))
        .collect::<Result<_>>()?;
    let contracted_total = contracted_dims.iter().product::<usize>().max(1);
    let total_output = output_shape.iter().product::<usize>().max(1);
    let mut output_data = vec![T::zero(); total_output];
    let mut tracker = ArgmaxTracker::new(&output_shape);
    let mut forward_error: Option<Error> = None;

    for_each_index(&output_shape, |out_idx| {
        if forward_error.is_some() {
            return;
        }

        let out_flat = col_major_flat_index(&output_shape, out_idx);
        let mut best = T::zero();
        let mut best_k = 0usize;

        for k_flat in 0..contracted_total {
            let k_idx = if contracted_dims.is_empty() {
                vec![]
            } else {
                unflatten_index(k_flat, &contracted_dims)
            };
            let mode_values = build_mode_values(&subs.output, out_idx, contracted, &k_idx);

            let mut product = T::one();
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                let idx = match operand_index_from_mode_values(
                    input_modes,
                    &mode_values,
                    &format!("operand {op_idx} forward index"),
                ) {
                    Ok(idx) => idx,
                    Err(err) => {
                        forward_error = Some(err);
                        return;
                    }
                };
                product = product * views[op_idx].get(&idx);
            }

            let new_sum = best + product;
            if k_flat == 0 || new_sum.inner() != best.inner() {
                best_k = k_flat;
            }
            best = new_sum;
        }

        output_data[out_flat] = best;
        tracker.indices_mut()[out_flat] = best_k;
    });

    if let Some(err) = forward_error {
        return Err(err);
    }

    let output = Tensor::<T>::from_slice(&output_data, &output_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))?;
    Ok((output, tracker))
}

pub(crate) fn tropical_forward_tangent<T: TropicalScalar>(
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T::Inner>>],
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
    output_shape: &[usize],
) -> Result<Tensor<T::Inner>> {
    match primals.len() {
        1 => tropical_forward_tangent_unary(
            primals[0],
            tangents[0],
            tracker,
            subs,
            contracted,
            output_shape,
        ),
        2 => tropical_forward_tangent_binary(
            primals,
            tangents,
            tracker,
            subs,
            contracted,
            output_shape,
        ),
        n => Err(Error::InvalidArgument(format!(
            "tropical forward tangent supports 1 or 2 operands, got {n}"
        ))),
    }
}

fn tropical_forward_tangent_unary<T: TropicalScalar>(
    primal: &Tensor<T>,
    tangent: Option<&Tensor<T::Inner>>,
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
    output_shape: &[usize],
) -> Result<Tensor<T::Inner>> {
    let input_modes = &subs.inputs[0];
    validate_modes_in_scope(
        input_modes,
        &subs.output,
        contracted,
        "unary tangent operand",
    )?;
    let tangent_view = tangent.map(tensor_to_view).transpose()?;
    let contracted_dims = contracted
        .iter()
        .map(|m| {
            let pos = input_modes.iter().position(|x| x == m).ok_or_else(|| {
                Error::InvalidArgument(format!("contracted mode {m} not in input"))
            })?;
            Ok(primal.dims()[pos])
        })
        .collect::<Result<Vec<_>>>()?;

    let mut output_data = vec![T::Inner::zero(); output_shape.iter().product::<usize>().max(1)];
    let mut forward_error: Option<Error> = None;

    for_each_index(output_shape, |out_idx| {
        if forward_error.is_some() {
            return;
        }
        let out_flat = col_major_flat_index(output_shape, out_idx);
        let winner = tracker.indices()[out_flat];
        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(winner, &contracted_dims)
        };
        let mode_values = build_mode_values(&subs.output, out_idx, contracted, &k_idx);
        let input_idx = match operand_index_from_mode_values(
            input_modes,
            &mode_values,
            "unary tangent input index",
        ) {
            Ok(idx) => idx,
            Err(err) => {
                forward_error = Some(err);
                return;
            }
        };
        output_data[out_flat] = tangent_view
            .as_ref()
            .map(|view| view.get(&input_idx))
            .unwrap_or_else(T::Inner::zero);
    });

    if let Some(err) = forward_error {
        return Err(err);
    }

    Tensor::<T::Inner>::from_slice(&output_data, output_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}

fn tropical_forward_tangent_binary<T: TropicalScalar>(
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T::Inner>>],
    tracker: &ArgmaxTracker,
    subs: &Subscripts,
    contracted: &[u32],
    output_shape: &[usize],
) -> Result<Tensor<T::Inner>> {
    let a = primals[0];
    let b = primals[1];
    let a_view = tensor_to_view(a)?;
    let b_view = tensor_to_view(b)?;
    let da_view = tangents[0].map(tensor_to_view).transpose()?;
    let db_view = tangents[1].map(tensor_to_view).transpose()?;
    let input_modes_a = &subs.inputs[0];
    let input_modes_b = &subs.inputs[1];
    validate_modes_in_scope(
        input_modes_a,
        &subs.output,
        contracted,
        "binary tangent operand A",
    )?;
    validate_modes_in_scope(
        input_modes_b,
        &subs.output,
        contracted,
        "binary tangent operand B",
    )?;
    let contracted_dims = dims_for_modes(primals, subs, contracted, "contracted")?;

    let mut output_data = vec![T::Inner::zero(); output_shape.iter().product::<usize>().max(1)];
    let mut forward_error: Option<Error> = None;

    for_each_index(output_shape, |out_idx| {
        if forward_error.is_some() {
            return;
        }
        let out_flat = col_major_flat_index(output_shape, out_idx);
        let winner = tracker.indices()[out_flat];
        let k_idx = if contracted_dims.is_empty() {
            vec![]
        } else {
            unflatten_index(winner, &contracted_dims)
        };
        let mode_values = build_mode_values(&subs.output, out_idx, contracted, &k_idx);
        let a_idx =
            match operand_index_from_mode_values(input_modes_a, &mode_values, "binary tangent a") {
                Ok(idx) => idx,
                Err(err) => {
                    forward_error = Some(err);
                    return;
                }
            };
        let b_idx =
            match operand_index_from_mode_values(input_modes_b, &mode_values, "binary tangent b") {
                Ok(idx) => idx,
                Err(err) => {
                    forward_error = Some(err);
                    return;
                }
            };

        let a_inner = a_view.get(&a_idx).inner();
        let b_inner = b_view.get(&b_idx).inner();
        let mut dout = T::Inner::zero();
        if let Some(view) = da_view.as_ref() {
            dout += T::mul_backward_a(a_inner, b_inner, view.get(&a_idx));
        }
        if let Some(view) = db_view.as_ref() {
            dout += T::mul_backward_b(a_inner, b_inner, view.get(&b_idx));
        }
        output_data[out_flat] = dout;
    });

    if let Some(err) = forward_error {
        return Err(err);
    }

    Tensor::<T::Inner>::from_slice(&output_data, output_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}
