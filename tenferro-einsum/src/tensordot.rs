use std::collections::HashSet;

use tenferro::error::{Error, Result};
use tenferro::{DotGeneralConfig, TracedTensor};

/// Axis specification for [`tensordot`](crate::traced_tensor::tensordot)
/// contraction sugar.
///
/// `Count(n)` contracts the last `n` axes of the left operand with the first
/// `n` axes of the right operand. `Axes` contracts the explicitly paired axes
/// in the order they are provided, accepting negative indices relative to each
/// operand rank.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::TensorDotAxes;
///
/// let count = TensorDotAxes::Count(2);
/// let explicit = TensorDotAxes::Axes {
///     lhs: &[-1],
///     rhs: &[0],
/// };
///
/// assert_eq!(count, TensorDotAxes::Count(2));
/// assert_ne!(count, explicit);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDotAxes<'a> {
    /// Contract the last `n` left axes with the first `n` right axes.
    Count(usize),
    /// Contract explicit left/right axis pairs.
    Axes {
        /// Left operand axes. Negative axes are normalized against the left rank.
        lhs: &'a [isize],
        /// Right operand axes. Negative axes are normalized against the right rank.
        rhs: &'a [isize],
    },
}

pub(crate) fn dot_general_config(
    axes: TensorDotAxes<'_>,
    lhs_rank: usize,
    rhs_rank: usize,
) -> Result<DotGeneralConfig> {
    let (lhs_contracting_dims, rhs_contracting_dims) = match axes {
        TensorDotAxes::Count(count) => {
            if count > lhs_rank || count > rhs_rank {
                return Err(contraction_error(format!(
                    "TensorDotAxes::Count({count}) cannot contract {count} axes \
                     for lhs rank {lhs_rank} and rhs rank {rhs_rank}"
                )));
            }
            ((lhs_rank - count..lhs_rank).collect(), (0..count).collect())
        }
        TensorDotAxes::Axes { lhs, rhs } => {
            if lhs.len() != rhs.len() {
                return Err(contraction_error(format!(
                    "tensordot explicit axes must have matching lengths, got lhs {} and rhs {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            (
                normalize_axes(lhs, lhs_rank, "lhs")?,
                normalize_axes(rhs, rhs_rank, "rhs")?,
            )
        }
    };

    let config = DotGeneralConfig {
        lhs_contracting_dims,
        rhs_contracting_dims,
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    config
        .validate_dims_with_ranks(lhs_rank, rhs_rank)
        .map_err(contraction_error)?;
    Ok(config)
}

#[cfg(feature = "autodiff")]
pub(crate) fn validate_concrete_contract_dims(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    config: &DotGeneralConfig,
) -> Result<()> {
    config
        .validate_dims_with_ranks(lhs_shape.len(), rhs_shape.len())
        .map_err(contraction_error)?;
    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(config.rhs_contracting_dims.iter())
    {
        let lhs_dim = lhs_shape[lhs_axis];
        let rhs_dim = rhs_shape[rhs_axis];
        if lhs_dim != rhs_dim {
            return Err(contracted_dims_error(lhs_axis, lhs_dim, rhs_axis, rhs_dim));
        }
    }
    Ok(())
}

pub(crate) fn validate_traced_contract_dims(
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    config: &DotGeneralConfig,
) -> Result<()> {
    config
        .validate_dims_with_ranks(lhs.rank, rhs.rank)
        .map_err(contraction_error)?;
    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(config.rhs_contracting_dims.iter())
    {
        let lhs_dim = lhs.axis_sym_dim(lhs_axis);
        let rhs_dim = rhs.axis_sym_dim(rhs_axis);
        if lhs_dim == rhs_dim {
            continue;
        }
        if let (Some(lhs_value), Some(rhs_value)) =
            (lhs_dim.constant_value(), rhs_dim.constant_value())
        {
            if lhs_value != rhs_value {
                return Err(contracted_dims_error(
                    lhs_axis, lhs_value, rhs_axis, rhs_value,
                ));
            }
        }
    }
    Ok(())
}

fn normalize_axes(axes: &[isize], rank: usize, operand: &str) -> Result<Vec<usize>> {
    let mut normalized = Vec::with_capacity(axes.len());
    let mut seen = HashSet::with_capacity(axes.len());
    for &axis in axes {
        let normalized_axis = normalize_axis(axis, rank, operand)?;
        if !seen.insert(normalized_axis) {
            return Err(contraction_error(format!(
                "duplicate {operand} axis {normalized_axis} in tensordot axes"
            )));
        }
        normalized.push(normalized_axis);
    }
    Ok(normalized)
}

fn normalize_axis(axis: isize, rank: usize, operand: &str) -> Result<usize> {
    let rank_isize = isize::try_from(rank).map_err(|_| {
        contraction_error(format!(
            "{operand} rank {rank} is too large to normalize tensordot axes"
        ))
    })?;
    let normalized = if axis < 0 { rank_isize + axis } else { axis };
    if normalized < 0 || normalized >= rank_isize {
        return Err(contraction_error(format!(
            "{operand} axis {axis} out of bounds for rank {rank}"
        )));
    }
    usize::try_from(normalized).map_err(|_| {
        contraction_error(format!(
            "{operand} axis {axis} could not be normalized for rank {rank}"
        ))
    })
}

fn contracted_dims_error(
    lhs_axis: usize,
    lhs_dim: usize,
    rhs_axis: usize,
    rhs_dim: usize,
) -> Error {
    contraction_error(format!(
        "contracted dimensions differ for lhs axis {lhs_axis} ({lhs_dim}) \
         and rhs axis {rhs_axis} ({rhs_dim})"
    ))
}

fn contraction_error(message: impl Into<String>) -> Error {
    Error::ContractionError(message.into())
}
