use crate::{Error, Result};

/// Broadcast-aware batch index mapping in column-major order.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::BroadcastBatchIndexer;
///
/// let indexer = BroadcastBatchIndexer::new(&[1, 3], &[2, 3], "solve", "b").unwrap();
/// assert_eq!(indexer.source_linear_batch_index(0), 0);
/// assert_eq!(indexer.source_linear_batch_index(5), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastBatchIndexer {
    output_batch_dims: Vec<usize>,
    normalized_source_batch_dims: Vec<usize>,
    source_strides: Vec<usize>,
    identity: bool,
}

impl BroadcastBatchIndexer {
    /// Builds a broadcast-aware indexer from source batch dims to output batch dims.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::BroadcastBatchIndexer;
    ///
    /// let indexer = BroadcastBatchIndexer::new(&[2, 3], &[2, 3], "solve", "a").unwrap();
    /// assert_eq!(indexer.output_batch_dims(), &[2, 3]);
    /// ```
    pub fn new(
        source_batch_dims: &[usize],
        output_batch_dims: &[usize],
        op_name: &str,
        arg_name: &str,
    ) -> Result<Self> {
        if source_batch_dims.len() > output_batch_dims.len() {
            return Err(Error::InvalidArgument(format!(
                "{op_name} {arg_name} batch rank {} exceeds target batch rank {}",
                source_batch_dims.len(),
                output_batch_dims.len()
            )));
        }

        let missing = output_batch_dims.len() - source_batch_dims.len();
        let mut normalized_source_batch_dims = vec![1; output_batch_dims.len()];
        normalized_source_batch_dims[missing..].copy_from_slice(source_batch_dims);

        for (axis, (&source_dim, &output_dim)) in normalized_source_batch_dims
            .iter()
            .zip(output_batch_dims.iter())
            .enumerate()
        {
            if source_dim != 1 && source_dim != output_dim {
                return Err(Error::InvalidArgument(format!(
                    "{op_name} {arg_name} batch dims are not broadcastable to {:?}: source axis {axis} has {source_dim}, target has {output_dim}",
                    output_batch_dims
                )));
            }
        }

        let identity = normalized_source_batch_dims == output_batch_dims;
        Ok(Self {
            output_batch_dims: output_batch_dims.to_vec(),
            source_strides: col_major_strides(&normalized_source_batch_dims)?,
            normalized_source_batch_dims,
            identity,
        })
    }

    /// Returns the broadcasted output batch dims.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::BroadcastBatchIndexer;
    ///
    /// let indexer = BroadcastBatchIndexer::new(&[2, 3], &[2, 3], "solve", "a").unwrap();
    /// assert_eq!(indexer.output_batch_dims(), &[2, 3]);
    /// ```
    pub fn output_batch_dims(&self) -> &[usize] {
        &self.output_batch_dims
    }

    /// Returns `true` when source and output batch layouts are identical.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::BroadcastBatchIndexer;
    ///
    /// let indexer = BroadcastBatchIndexer::new(&[2, 3], &[2, 3], "solve", "a").unwrap();
    /// assert!(indexer.is_identity());
    /// ```
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Maps a flat output batch index to the corresponding flat source batch index.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::BroadcastBatchIndexer;
    ///
    /// let indexer = BroadcastBatchIndexer::new(&[1, 3], &[2, 3], "solve", "b").unwrap();
    /// let mapped: Vec<_> = (0..6).map(|i| indexer.source_linear_batch_index(i)).collect();
    /// assert_eq!(mapped, vec![0, 0, 1, 1, 2, 2]);
    /// ```
    pub fn source_linear_batch_index(&self, mut output_linear_batch_index: usize) -> usize {
        if self.output_batch_dims.is_empty() {
            return 0;
        }

        let mut source_linear_batch_index = 0usize;
        for axis in 0..self.output_batch_dims.len() {
            let output_dim = self.output_batch_dims[axis];
            let coord = output_linear_batch_index % output_dim;
            output_linear_batch_index /= output_dim;
            if self.normalized_source_batch_dims[axis] != 1 {
                source_linear_batch_index += coord * self.source_strides[axis];
            }
        }
        source_linear_batch_index
    }
}

/// Computes the broadcasted batch shape for two operands.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::broadcast_batch_dims;
///
/// let dims = broadcast_batch_dims(&[2, 1], &[1, 3], "solve", "a", "b").unwrap();
/// assert_eq!(dims, vec![2, 3]);
/// ```
pub fn broadcast_batch_dims(
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    op_name: &str,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<Vec<usize>> {
    let rank = lhs_batch_dims.len().max(rhs_batch_dims.len());
    let lhs_pad = rank - lhs_batch_dims.len();
    let rhs_pad = rank - rhs_batch_dims.len();
    let mut output_batch_dims = Vec::with_capacity(rank);

    for axis in 0..rank {
        let lhs_dim = if axis < lhs_pad {
            1
        } else {
            lhs_batch_dims[axis - lhs_pad]
        };
        let rhs_dim = if axis < rhs_pad {
            1
        } else {
            rhs_batch_dims[axis - rhs_pad]
        };
        if lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1 {
            output_batch_dims.push(lhs_dim.max(rhs_dim));
        } else {
            return Err(Error::InvalidArgument(format!(
                "{op_name} batch dims are not broadcastable: {lhs_name} has {:?}, {rhs_name} has {:?}",
                lhs_batch_dims, rhs_batch_dims
            )));
        }
    }

    Ok(output_batch_dims)
}

/// Computes the total number of batch iterations with overflow checking.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::checked_batch_count;
///
/// assert_eq!(checked_batch_count(&[]).unwrap(), 1);
/// assert_eq!(checked_batch_count(&[2, 3]).unwrap(), 6);
/// ```
pub fn checked_batch_count(batch_dims: &[usize]) -> Result<usize> {
    batch_dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidArgument("batch iteration count overflow".into()))
    })
}

/// Flattens a column-major multi-index into a linear index.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::flatten_col_major_index;
///
/// assert_eq!(flatten_col_major_index(&[2, 3, 4], &[1, 0, 2]).unwrap(), 13);
/// ```
pub fn flatten_col_major_index(dims: &[usize], index: &[usize]) -> Result<usize> {
    if dims.len() != index.len() {
        return Err(Error::InvalidArgument(format!(
            "flatten rank mismatch: dims {:?}, index {:?}",
            dims, index
        )));
    }

    let mut flat = 0usize;
    let mut stride = 1usize;
    for (axis, (&dim, &coord)) in dims.iter().zip(index.iter()).enumerate() {
        if coord >= dim {
            return Err(Error::InvalidArgument(format!(
                "flatten index {coord} out of range for axis {axis} with dim {dim}"
            )));
        }
        let term = coord
            .checked_mul(stride)
            .ok_or_else(|| Error::InvalidArgument("flatten linear index overflow".into()))?;
        flat = flat
            .checked_add(term)
            .ok_or_else(|| Error::InvalidArgument("flatten linear index overflow".into()))?;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| Error::InvalidArgument("flatten stride overflow".into()))?;
    }
    Ok(flat)
}

/// Unflattens a column-major linear index into a preallocated coordinate buffer.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::unflatten_col_major_index_into;
///
/// let mut out = [0usize; 3];
/// unflatten_col_major_index_into(13, &[2, 3, 4], &mut out).unwrap();
/// assert_eq!(out, [1, 0, 2]);
/// ```
pub fn unflatten_col_major_index_into(
    mut flat: usize,
    dims: &[usize],
    out: &mut [usize],
) -> Result<()> {
    if out.len() != dims.len() {
        return Err(Error::InvalidArgument(format!(
            "unflatten output rank mismatch: dims {:?}, out len {}",
            dims,
            out.len()
        )));
    }
    let total = checked_batch_count(dims)?;
    if flat >= total {
        return Err(Error::InvalidArgument(format!(
            "flat index {flat} out of range for dims {dims:?}"
        )));
    }
    for d in 0..dims.len() {
        out[d] = flat % dims[d];
        flat /= dims[d];
    }
    Ok(())
}

fn col_major_strides(dims: &[usize]) -> Result<Vec<usize>> {
    let mut strides = vec![0usize; dims.len()];
    if dims.is_empty() {
        return Ok(strides);
    }
    strides[0] = 1;
    for axis in 1..dims.len() {
        strides[axis] = strides[axis - 1]
            .checked_mul(dims[axis - 1])
            .ok_or_else(|| Error::InvalidArgument("batch stride overflow".into()))?;
    }
    Ok(strides)
}

#[cfg(test)]
mod tests;
