use num_traits::{Float, NumCast};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};

use super::Tensor;
use crate::layout::compute_contiguous_strides;
use crate::MemoryOrder;

impl<T> Tensor<T>
where
    T: Scalar + Float + NumCast,
{
    /// Create an identity matrix.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let id = Tensor::<f64>::eye(
    ///     3,
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// assert_eq!(id.dims(), &[3, 3]);
    /// ```
    pub fn eye(n: usize, memory_space: LogicalMemorySpace, order: MemoryOrder) -> Result<Self> {
        let dims = [n, n];
        let strides = compute_contiguous_strides(&dims, order);
        let storage_len = n
            .checked_mul(n)
            .ok_or_else(|| Error::StrideError(format!("eye: storage length overflow for n={n}")))?;
        let mut data = vec![T::zero(); storage_len];
        for i in 0..n {
            let i = isize::try_from(i).map_err(|_| {
                Error::StrideError(format!("eye: index {i} cannot be represented as isize"))
            })?;
            let pos = i
                .checked_mul(strides[0])
                .and_then(|a| i.checked_mul(strides[1]).and_then(|b| a.checked_add(b)))
                .ok_or_else(|| {
                    Error::StrideError(format!(
                        "eye: position overflow for index {i} with strides {:?}",
                        strides
                    ))
                })?;
            let pos = usize::try_from(pos).map_err(|_| {
                Error::StrideError(format!(
                    "eye: position {pos} cannot be represented as usize for strides {:?}",
                    strides
                ))
            })?;
            data[pos] = T::one();
        }
        Self::finish_allocation(
            Self::main_memory_contiguous(data, &dims, order),
            memory_space,
        )
    }

    /// Create a regularly spaced 1-D tensor from `start` toward `end`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let xs = Tensor::<f64>::arange(
    ///     0.0,
    ///     5.0,
    ///     1.0,
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// assert_eq!(xs.dims(), &[5]);
    /// ```
    pub fn arange(
        start: T,
        end: T,
        step: T,
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Result<Self> {
        if step.is_zero() {
            return Err(Error::InvalidArgument(
                "arange: step must be non-zero".into(),
            ));
        }

        let mut data = Vec::new();
        let zero = T::zero();
        if step > zero {
            let mut current = start;
            while current < end {
                data.push(current);
                current = current + step;
            }
        } else {
            let mut current = start;
            while current > end {
                data.push(current);
                current = current + step;
            }
        }

        let dims = [data.len()];
        let tensor = Self::main_memory_contiguous(data, &dims, order);
        Self::finish_allocation(tensor, memory_space)
    }

    /// Create a 1-D tensor containing `n_samples` evenly spaced values.
    ///
    /// Returns an error if `n_samples` is negative.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let xs = Tensor::<f64>::linspace(
    ///     0.0,
    ///     1.0,
    ///     5,
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// assert_eq!(xs.dims(), &[5]);
    /// ```
    pub fn linspace(
        start: T,
        end: T,
        n_samples: isize,
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Result<Self> {
        if n_samples < 0 {
            return Err(Error::InvalidArgument(format!(
                "linspace: steps must be non-negative, got {n_samples}"
            )));
        }

        let n_samples = usize::try_from(n_samples).map_err(|_| {
            Error::InvalidArgument(format!(
                "linspace: steps {n_samples} cannot be represented as usize"
            ))
        })?;
        let mut data = Vec::with_capacity(n_samples);
        match n_samples {
            0 => {}
            1 => data.push(start),
            _ => {
                let Some(denom) = <T as NumCast>::from(n_samples - 1) else {
                    return Err(Error::InvalidArgument(format!(
                        "linspace: sample count {} cannot be represented in target scalar type",
                        n_samples
                    )));
                };
                let step = (end - start) / denom;
                let mut current = start;
                for _ in 0..n_samples {
                    data.push(current);
                    current = current + step;
                }
                if let Some(last) = data.last_mut() {
                    *last = end;
                }
            }
        }

        let dims = [data.len()];
        let tensor = Self::main_memory_contiguous(data, &dims, order);
        Self::finish_allocation(tensor, memory_space)
    }
}
