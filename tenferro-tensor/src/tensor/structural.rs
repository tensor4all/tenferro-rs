use num_traits::ToPrimitive;
use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};

use super::Tensor;
use crate::layout::compute_contiguous_strides;
use crate::MemoryOrder;

/// Scalar types accepted as `keep_counts` in [`Tensor::zero_trailing_by_counts`].
///
/// Implemented for `f32` and `f64`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::KeepCountScalar;
///
/// fn accepts_keep_counts<T: KeepCountScalar>() {}
/// accepts_keep_counts::<f32>();
/// accepts_keep_counts::<f64>();
/// ```
#[cfg(not(feature = "cuda"))]
pub trait KeepCountScalar: Scalar + ToPrimitive {}

#[cfg(not(feature = "cuda"))]
impl KeepCountScalar for f32 {}
#[cfg(not(feature = "cuda"))]
impl KeepCountScalar for f64 {}

/// Scalar types accepted as `keep_counts` in [`Tensor::zero_trailing_by_counts`].
///
/// Implemented for `f32` and `f64`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::KeepCountScalar;
///
/// fn accepts_keep_counts<T: KeepCountScalar>() {}
/// accepts_keep_counts::<f32>();
/// accepts_keep_counts::<f64>();
/// ```
#[cfg(feature = "cuda")]
pub trait KeepCountScalar:
    Scalar + ToPrimitive + tenferro_device::cuda::runtime::RuntimeKeepCountScalar
{
}

#[cfg(feature = "cuda")]
impl KeepCountScalar for f32 {}
#[cfg(feature = "cuda")]
impl KeepCountScalar for f64 {}

fn parse_keep_count<R: ToPrimitive>(value: &R, axis_len: usize, index: usize) -> Result<usize> {
    let raw = value.to_f64().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "keep_counts[{index}] must be representable as a real scalar"
        ))
    })?;
    if !raw.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "keep_counts[{index}] must be finite"
        )));
    }
    if raw < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "keep_counts[{index}] must be non-negative"
        )));
    }
    if raw.fract() != 0.0 {
        return Err(Error::InvalidArgument(format!(
            "keep_counts[{index}] must be integer-valued"
        )));
    }
    let count = raw as usize;
    if count > axis_len {
        return Err(Error::InvalidArgument(format!(
            "keep_counts[{index}]={count} exceeds axis length {axis_len}"
        )));
    }
    Ok(count)
}

impl<T: Scalar> Tensor<T> {
    /// Return a contiguous tensor with trailing elements zeroed according to
    /// batch-local keep counts.
    ///
    /// `structural_rank` splits the payload dims from the trailing batch dims.
    /// `axis` is interpreted within the structural prefix `[0, structural_rank)`.
    ///
    /// Phase 1 supports main-memory tensors only.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)?;
    /// let keep_counts = Tensor::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor)?;
    /// let trimmed = payload.zero_trailing_by_counts(&keep_counts, 1, 2)?;
    /// assert_eq!(trimmed.buffer().as_slice().unwrap(), &[1.0, 2.0, 0.0, 0.0]);
    /// # Ok::<(), tenferro_device::Error>(())
    /// ```
    pub fn zero_trailing_by_counts<R>(
        &self,
        keep_counts: &Tensor<R>,
        axis: usize,
        structural_rank: usize,
    ) -> Result<Tensor<T>>
    where
        R: KeepCountScalar,
    {
        self.wait();
        keep_counts.wait();

        if self.logical_memory_space != keep_counts.logical_memory_space() {
            return Err(Error::InvalidArgument(format!(
                "zero_trailing_by_counts requires matching memory spaces, got {:?} and {:?}",
                self.logical_memory_space,
                keep_counts.logical_memory_space()
            )));
        }
        #[cfg(feature = "cuda")]
        if matches!(
            self.logical_memory_space,
            LogicalMemorySpace::GpuMemory { .. }
        ) {
            return crate::cuda_runtime::zero_trailing_by_counts_tensor(
                self,
                keep_counts,
                axis,
                structural_rank,
            );
        }

        if self.logical_memory_space != LogicalMemorySpace::MainMemory {
            return Err(Error::DeviceError(
                "zero_trailing_by_counts only supports main-memory tensors in Phase 1".into(),
            ));
        }
        if structural_rank == 0 {
            return Err(Error::InvalidArgument(
                "zero_trailing_by_counts requires structural_rank >= 1".into(),
            ));
        }
        if structural_rank > self.ndim() {
            return Err(Error::RankMismatch {
                expected: self.ndim(),
                got: structural_rank,
            });
        }
        if axis >= structural_rank {
            return Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for structural_rank {structural_rank}"
            )));
        }

        let expected_batch_dims = &self.dims[structural_rank..];
        if keep_counts.dims() != expected_batch_dims {
            return Err(Error::ShapeMismatch {
                expected: expected_batch_dims.to_vec(),
                got: keep_counts.dims().to_vec(),
            });
        }

        let keep_counts = keep_counts.contiguous(MemoryOrder::ColumnMajor);
        let keep_counts_slice = keep_counts.buffer().as_slice().ok_or_else(|| {
            Error::DeviceError(
                "zero_trailing_by_counts requires CPU-accessible keep_counts in Phase 1".into(),
            )
        })?;
        let axis_len = self.dims[axis];
        let parsed_counts = keep_counts_slice
            .iter()
            .enumerate()
            .map(|(index, value)| parse_keep_count(value, axis_len, index))
            .collect::<Result<Vec<_>>>()?;

        let src = self.buffer().as_slice().ok_or_else(|| {
            Error::DeviceError(
                "zero_trailing_by_counts requires a CPU-accessible payload in Phase 1".into(),
            )
        })?;
        let mut out = vec![T::zero(); self.len()];
        if out.is_empty() {
            return Ok(self.materialized_from_vec(out, MemoryOrder::ColumnMajor));
        }

        let batch_strides =
            compute_contiguous_strides(expected_batch_dims, MemoryOrder::ColumnMajor);
        let out_len = out.len();
        let mut index = vec![0usize; self.ndim()];
        for (dst_pos, dst) in out.iter_mut().enumerate() {
            let batch_pos = if expected_batch_dims.is_empty() {
                0usize
            } else {
                index[structural_rank..]
                    .iter()
                    .zip(batch_strides.iter())
                    .try_fold(0usize, |acc, (&coord, &stride)| {
                        let stride = usize::try_from(stride).ok()?;
                        coord
                            .checked_mul(stride)
                            .and_then(|term| acc.checked_add(term))
                    })
                    .ok_or_else(|| {
                        Error::StrideError(format!(
                            "batch offset overflow for index {:?} and batch strides {:?}",
                            &index[structural_rank..],
                            batch_strides
                        ))
                    })?
            };
            let keep = parsed_counts.get(batch_pos).copied().ok_or_else(|| {
                Error::StrideError(format!("batch position {batch_pos} out of bounds"))
            })?;
            if index[axis] < keep {
                let src_pos = self
                    .offset
                    .checked_add(
                        index
                            .iter()
                            .zip(self.strides.iter())
                            .try_fold(0isize, |acc, (&coord, &stride)| {
                                (coord as isize)
                                    .checked_mul(stride)
                                    .and_then(|term| acc.checked_add(term))
                            })
                            .ok_or_else(|| {
                                Error::StrideError(format!(
                                    "source offset overflow for index {:?} and strides {:?}",
                                    index, self.strides
                                ))
                            })?,
                    )
                    .and_then(|pos| usize::try_from(pos).ok())
                    .ok_or_else(|| {
                        Error::StrideError(format!(
                            "source position overflow for index {:?} and offset {}",
                            index, self.offset
                        ))
                    })?;
                *dst = src[src_pos];
            }

            if dst_pos + 1 == out_len {
                continue;
            }
            for axis_index in 0..self.ndim() {
                index[axis_index] += 1;
                if index[axis_index] < self.dims[axis_index] {
                    break;
                }
                index[axis_index] = 0;
            }
        }

        Ok(self.materialized_from_vec(out, MemoryOrder::ColumnMajor))
    }
}
