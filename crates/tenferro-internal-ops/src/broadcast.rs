use thiserror::Error;

/// Lowering plan for broadcasting one input to an output shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastInputPlan {
    /// Shape to use before `BroadcastInDim`.
    pub source_shape: Vec<usize>,
    /// Source axes retained in `source_shape` and their output-axis positions.
    pub dims: Vec<usize>,
}

/// Error returned when NumPy-style broadcast planning fails.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BroadcastError {
    /// Two shapes cannot be broadcast together.
    #[error("cannot broadcast shapes {lhs:?} and {rhs:?}")]
    IncompatibleBinary { lhs: Vec<usize>, rhs: Vec<usize> },
    /// One input cannot be broadcast to the requested output shape.
    #[error("cannot broadcast shape {input:?} to {output:?}")]
    IncompatibleInput {
        input: Vec<usize>,
        output: Vec<usize>,
    },
    /// A higher-rank input cannot broadcast to a lower-rank output.
    #[error("cannot broadcast higher-rank shape {input:?} to {output:?}")]
    RankTooLarge {
        input: Vec<usize>,
        output: Vec<usize>,
    },
}

/// Compute the NumPy-style broadcast shape for two concrete shapes.
///
/// # Examples
///
/// ```
/// use tenferro_ops::broadcast::broadcast_shape;
///
/// assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
/// assert_eq!(broadcast_shape(&[], &[2, 3]).unwrap(), vec![2, 3]);
/// ```
///
/// # Errors
///
/// Returns [`BroadcastError::IncompatibleBinary`] when a pair of aligned
/// dimensions is incompatible.
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, BroadcastError> {
    let rank = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(rank);
    for axis in 0..rank {
        let lhs_dim = aligned_dim(lhs, rank, axis);
        let rhs_dim = aligned_dim(rhs, rank, axis);
        if lhs_dim == rhs_dim {
            out.push(lhs_dim);
        } else if lhs_dim == 1 {
            out.push(rhs_dim);
        } else if rhs_dim == 1 {
            out.push(lhs_dim);
        } else {
            return Err(BroadcastError::IncompatibleBinary {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        }
    }
    Ok(out)
}

/// Compute the common NumPy-style broadcast shape for zero or more shapes.
///
/// # Examples
///
/// ```
/// use tenferro_ops::broadcast::broadcast_shapes;
///
/// let shape = broadcast_shapes([&[3, 1][..], &[1, 4][..], &[3, 4][..]]).unwrap();
/// assert_eq!(shape, vec![3, 4]);
/// ```
///
/// # Errors
///
/// Returns [`BroadcastError::IncompatibleBinary`] when any pair of input
/// shapes cannot be broadcast together.
pub fn broadcast_shapes<'a>(
    shapes: impl IntoIterator<Item = &'a [usize]>,
) -> Result<Vec<usize>, BroadcastError> {
    let mut iter = shapes.into_iter();
    let Some(first) = iter.next() else {
        return Ok(Vec::new());
    };
    let mut out = first.to_vec();
    for shape in iter {
        out = broadcast_shape(&out, shape)?;
    }
    Ok(out)
}

/// Plan how one input should lower to `BroadcastInDim`.
///
/// Expanding singleton axes are omitted from `source_shape` so downstream VJP
/// rules reduce those axes explicitly.
///
/// # Examples
///
/// ```
/// use tenferro_ops::broadcast::broadcast_input_plan;
///
/// let plan = broadcast_input_plan(&[3, 1], &[3, 4]).unwrap();
/// assert_eq!(plan.source_shape, vec![3]);
/// assert_eq!(plan.dims, vec![0]);
/// ```
///
/// # Errors
///
/// Returns [`BroadcastError::RankTooLarge`] when `input` has higher rank than
/// `output`, or [`BroadcastError::IncompatibleInput`] for incompatible axes.
pub fn broadcast_input_plan(
    input: &[usize],
    output: &[usize],
) -> Result<BroadcastInputPlan, BroadcastError> {
    if input.len() > output.len() {
        return Err(BroadcastError::RankTooLarge {
            input: input.to_vec(),
            output: output.to_vec(),
        });
    }
    let rank_diff = output.len() - input.len();
    let mut source_shape = Vec::with_capacity(input.len());
    let mut dims = Vec::with_capacity(input.len());
    for (src_axis, &src_dim) in input.iter().enumerate() {
        let dst_axis = src_axis + rank_diff;
        let dst_dim = output[dst_axis];
        if src_dim != dst_dim && src_dim != 1 {
            return Err(BroadcastError::IncompatibleInput {
                input: input.to_vec(),
                output: output.to_vec(),
            });
        }
        if src_dim == 1 && dst_dim != 1 {
            continue;
        }
        source_shape.push(src_dim);
        dims.push(dst_axis);
    }
    Ok(BroadcastInputPlan { source_shape, dims })
}

fn aligned_dim(shape: &[usize], output_rank: usize, output_axis: usize) -> usize {
    if output_axis < output_rank - shape.len() {
        1
    } else {
        shape[output_axis - (output_rank - shape.len())]
    }
}
