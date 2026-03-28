pub(super) use chainrules::ScalarAd;
pub(super) use num_traits::NumCast;
pub(super) use tenferro_algebra::Scalar;
pub(super) use tenferro_prims::{
    AnalyticBinaryOp, AnalyticReductionOp, AnalyticUnaryOp, ScalarBinaryOp, ScalarReductionOp,
    ScalarUnaryOp,
};
pub(super) use tenferro_tensor::Tensor;

pub(super) use crate::{tape, AdTensor, Error, Result};

pub(super) use crate::ops::common::{
    broadcast_scalar_like, collect_reverse_input_specs, compress_structured_pullback_like,
    dense_input_snapshot_in_runtime, has_any_tangent, has_forward, scalar_from_rank0_tensor,
    wrap_same_type_dense_ad_output,
};
pub(super) use crate::ops::scalar::primal::{
    analytic_binary_primal, analytic_full_reduction_primal, analytic_unary_primal,
    scalar_binary_primal, scalar_full_reduction_primal, scalar_unary_primal,
};
pub(super) use crate::runtime::contracts::{GenericAdRuntimeValue, RealAdRuntimeValue};

fn dense_host_values<T>(tensor: &Tensor<T>, op_name: &'static str) -> Result<Vec<T>>
where
    T: Scalar + Copy,
{
    let contiguous = tensor.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
    let slice = contiguous
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} could not materialize dense tensor on host memory"),
        })?;
    let start = usize::try_from(contiguous.offset()).map_err(|_| Error::InvalidAdTensor {
        message: format!(
            "{op_name} computed negative dense tensor offset {}",
            contiguous.offset()
        ),
    })?;
    let end = start
        .checked_add(contiguous.len())
        .ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} dense tensor offset overflow"),
        })?;
    slice
        .get(start..end)
        .map(|values| values.to_vec())
        .ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} dense tensor slice [{start}..{end}) is out of bounds"),
        })
}

fn elementwise_unary_primal_tensor<T>(
    op_name: &'static str,
    input: &Tensor<T>,
    mut primal: impl FnMut(T) -> T,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let values = dense_host_values(input, op_name)?;
    let output: Vec<T> = values.into_iter().map(&mut primal).collect();
    Tensor::from_slice(
        &output,
        input.dims(),
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .map_err(Error::from)
}

fn elementwise_unary_tangent_tensor<T>(
    op_name: &'static str,
    input: &Tensor<T>,
    tangent: &Tensor<T>,
    mut frule: impl FnMut(T, T) -> (T, T),
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let input_values = dense_host_values(input, op_name)?;
    let tangent_values = dense_host_values(tangent, op_name)?;
    if input_values.len() != tangent_values.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} elementwise tangent length mismatch: {} vs {}",
                input_values.len(),
                tangent_values.len()
            ),
        });
    }
    let output: Vec<T> = input_values
        .into_iter()
        .zip(tangent_values)
        .map(|(x, dx)| frule(x, dx).1)
        .collect();
    Tensor::from_slice(
        &output,
        input.dims(),
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .map_err(Error::from)
}

fn elementwise_unary_pullback_tensor<T>(
    op_name: &'static str,
    input: &Tensor<T>,
    primal: &Tensor<T>,
    cotangent: &Tensor<T>,
    mut rrule: impl FnMut(T, T, T) -> T,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let input_values = dense_host_values(input, op_name)?;
    let primal_values = dense_host_values(primal, op_name)?;
    let cotangent_values = dense_host_values(cotangent, op_name)?;
    if input_values.len() != primal_values.len() || input_values.len() != cotangent_values.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} elementwise pullback length mismatch: input={}, primal={}, cotangent={}",
                input_values.len(),
                primal_values.len(),
                cotangent_values.len()
            ),
        });
    }
    let output: Vec<T> = input_values
        .into_iter()
        .zip(primal_values)
        .zip(cotangent_values)
        .map(|((x, y), cot)| rrule(x, y, cot))
        .collect();
    Tensor::from_slice(
        &output,
        input.dims(),
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .map_err(Error::from)
}

pub(super) fn scalar_from_usize<T: ScalarAd>(value: usize) -> Result<T> {
    let Some(real) = <T::Real as NumCast>::from(value) else {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "cannot represent reduction size {value} for scalar type {}",
                std::any::type_name::<T>()
            ),
        });
    };
    Ok(T::from_real(real))
}

pub(super) fn two_tensor_like<T>(template: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Scalar + ScalarAd + Copy,
{
    broadcast_scalar_like(T::one() + T::one(), template)
}

pub(super) fn one_tensor_like<T>(template: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Scalar + ScalarAd + Copy,
{
    broadcast_scalar_like(T::one(), template)
}

pub(super) fn centered_input_tensor<T>(
    op_name: &'static str,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let mean = scalar_full_reduction_primal(op_name, ScalarReductionOp::Mean, input)?;
    let mean_scalar = scalar_from_rank0_tensor(&mean, op_name)?;
    let mean_broadcast = broadcast_scalar_like(mean_scalar, input)?;
    scalar_binary_primal(op_name, ScalarBinaryOp::Sub, input, &mean_broadcast)
}

pub(super) fn variance_reduction_tangent<T>(
    op_name: &'static str,
    input: &Tensor<T>,
    tangent: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let centered = centered_input_tensor(op_name, input)?;
    let centered_dt = scalar_binary_primal(op_name, ScalarBinaryOp::Mul, &centered, tangent)?;
    let mean_dt = scalar_full_reduction_primal(op_name, ScalarReductionOp::Mean, &centered_dt)?;
    let two = two_tensor_like(&mean_dt)?;
    scalar_binary_primal(op_name, ScalarBinaryOp::Mul, &mean_dt, &two)
}

pub(super) fn variance_reduction_pullback<T>(
    op_name: &'static str,
    input: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let centered = centered_input_tensor(op_name, input)?;
    let cotangent_scalar = scalar_from_rank0_tensor(cotangent, op_name)?;
    let coeff = (T::one() + T::one()) * cotangent_scalar / scalar_from_usize::<T>(input.len())?;
    let coeff_tensor = broadcast_scalar_like(coeff, input)?;
    scalar_binary_primal(op_name, ScalarBinaryOp::Mul, &centered, &coeff_tensor)
}

pub(super) fn run_scalar_unary_ad<T, FPrimal, FTangent, FPullback>(
    op_name: &'static str,
    _pullback_op_name: &'static str,
    input: &AdTensor<T>,
    primal_fn: FPrimal,
    tangent_fn: FTangent,
    pullback_fn: FPullback,
) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
    FPrimal: Fn(&Tensor<T>) -> Result<Tensor<T>>,
    FTangent: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FPullback: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
{
    let operands = [input];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
    let (input_primal, input_tangent) =
        dense_input_snapshot_in_runtime(op_name, input, needs_tangent)?;
    let primal = primal_fn(&input_primal)?;
    let tangent = if needs_tangent {
        let tangent = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized tangent"),
        })?;
        Some(tangent_fn(&input_primal, &primal, &tangent)?)
    } else {
        None
    };

    let out = wrap_same_type_dense_ad_output(op_name, &operands, primal.clone(), tangent)?;

    if let Some((node, tape)) = out.reverse_handle() {
        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();

        let input_node_ids: Vec<_> = input_spec.iter().map(|s| s.node).collect();
        tape::register_closure_rule::<T>(
            &tape,
            node,
            input_node_ids,
            Box::new(move |cotangent| {
                let grad = pullback_fn(&input_primal, &primal, cotangent.payload())?;
                let Some(spec) = &input_spec else {
                    return Ok(Vec::new());
                };
                let grad = compress_structured_pullback_like(op_name, grad, &spec.layout)?;
                Ok(vec![(spec.node, grad)])
            }),
        );
    }

    Ok(out)
}

pub(super) fn run_elementwise_scalar_unary_ad<T, FPrimalElem, FFruleElem, FRruleElem>(
    op_name: &'static str,
    pullback_op_name: &'static str,
    input: &AdTensor<T>,
    primal_elem: FPrimalElem,
    frule_elem: FFruleElem,
    rrule_elem: FRruleElem,
) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
    FPrimalElem: Fn(T) -> T + Copy,
    FFruleElem: Fn(T, T) -> (T, T) + Copy,
    FRruleElem: Fn(T, T, T) -> T + Send + Sync + Copy + 'static,
{
    run_scalar_unary_ad(
        op_name,
        pullback_op_name,
        input,
        move |tensor| elementwise_unary_primal_tensor(op_name, tensor, primal_elem),
        move |input_primal, _output_primal, tangent| {
            elementwise_unary_tangent_tensor(op_name, input_primal, tangent, frule_elem)
        },
        move |input_primal, output_primal, cotangent| {
            elementwise_unary_pullback_tensor(
                pullback_op_name,
                input_primal,
                output_primal,
                cotangent,
                rrule_elem,
            )
        },
    )
}

pub(super) fn run_scalar_binary_ad<T, FPrimal, FTangent, FPullback>(
    op_name: &'static str,
    _pullback_op_name: &'static str,
    lhs: &AdTensor<T>,
    rhs: &AdTensor<T>,
    primal_fn: FPrimal,
    tangent_fn: FTangent,
    pullback_fn: FPullback,
) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
    FPrimal: Fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FTangent: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FPullback: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)>
        + Send
        + Sync
        + 'static,
{
    let operands = [lhs, rhs];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
    let (lhs_snapshot, rhs_snapshot) = (
        dense_input_snapshot_in_runtime(op_name, lhs, needs_tangent)?,
        dense_input_snapshot_in_runtime(op_name, rhs, needs_tangent)?,
    );
    let (lhs_primal, lhs_tangent) = lhs_snapshot;
    let (rhs_primal, rhs_tangent) = rhs_snapshot;
    let primal = primal_fn(&lhs_primal, &rhs_primal)?;
    let tangent = if needs_tangent {
        let lhs_tangent = lhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized lhs tangent"),
        })?;
        let rhs_tangent = rhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized rhs tangent"),
        })?;
        Some(tangent_fn(
            &lhs_primal,
            &rhs_primal,
            &primal,
            &lhs_tangent,
            &rhs_tangent,
        )?)
    } else {
        None
    };

    let out = wrap_same_type_dense_ad_output(op_name, &operands, primal.clone(), tangent)?;

    if let Some((node, tape)) = out.reverse_handle() {
        let reverse_specs = collect_reverse_input_specs(&operands);

        let input_node_ids: Vec<_> = reverse_specs
            .iter()
            .filter_map(|s| s.as_ref().map(|s| s.node))
            .collect();
        tape::register_closure_rule::<T>(
            &tape,
            node,
            input_node_ids,
            Box::new(move |cotangent| {
                let (grad_lhs, grad_rhs) =
                    pullback_fn(&lhs_primal, &rhs_primal, &primal, cotangent.payload())?;
                let mut input_grads = Vec::new();
                if let Some(spec) = &reverse_specs[0] {
                    let grad = compress_structured_pullback_like(op_name, grad_lhs, &spec.layout)?;
                    input_grads.push((spec.node, grad));
                }
                if let Some(spec) = &reverse_specs[1] {
                    let grad = compress_structured_pullback_like(op_name, grad_rhs, &spec.layout)?;
                    input_grads.push((spec.node, grad));
                }
                Ok(input_grads)
            }),
        );
    }

    Ok(out)
}
