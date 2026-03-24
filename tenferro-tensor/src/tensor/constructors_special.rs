use num_traits::{Float, NumCast};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};

use super::Tensor;
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
        let storage_len = n
            .checked_mul(n)
            .ok_or_else(|| Error::StrideError(format!("eye: storage length overflow for n={n}")))?;
        let mut data = vec![T::zero(); storage_len];
        let diag_stride = n + 1;
        for i in 0..n {
            data[i * diag_stride] = T::one();
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

        let n_samples = n_samples as usize;
        let mut data = Vec::with_capacity(n_samples);
        match n_samples {
            0 => {}
            1 => data.push(start),
            _ => {
                let denom = <T as NumCast>::from(n_samples - 1).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "linspace: sample count {} cannot be represented in target scalar type",
                        n_samples
                    ))
                })?;
                let step = (end - start) / denom;
                let mut current = start;
                for _ in 0..n_samples {
                    data.push(current);
                    current = current + step;
                }
                data[n_samples - 1] = end;
            }
        }

        let dims = [data.len()];
        let tensor = Self::main_memory_contiguous(data, &dims, order);
        Self::finish_allocation(tensor, memory_space)
    }
}

#[cfg(test)]
mod tests;
