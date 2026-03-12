use tenferro_algebra::{Conjugate, Scalar};

use super::Tensor;
use crate::layout::{compute_contiguous_strides, copy_strided, is_contiguous_in_order};
use crate::{DataBuffer, MemoryOrder};

enum TriangularHalf {
    Lower,
    Upper,
}

impl<T: Scalar> Tensor<T> {
    /// Return a contiguous copy of this tensor in the given memory order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    /// let c = t.contiguous(MemoryOrder::RowMajor);
    /// assert!(c.is_contiguous());
    /// ```
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T> {
        self.wait();
        if is_contiguous_in_order(&self.dims, &self.strides, order) && self.offset == 0 {
            return self.clone();
        }

        let mut data = vec![T::zero(); self.len()];
        if !data.is_empty() {
            let dst_strides = compute_contiguous_strides(&self.dims, order);
            copy_strided(
                self.cpu_backed_slice_or_panic("contiguous"),
                &self.dims,
                &self.strides,
                self.offset,
                &mut data,
                &dst_strides,
            );
        }
        self.materialized_from_vec(data, order)
    }

    /// Consume this tensor and return a contiguous version.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let c = t.into_contiguous(MemoryOrder::ColumnMajor);
    /// assert!(c.is_contiguous());
    /// ```
    pub fn into_contiguous(self, order: MemoryOrder) -> Tensor<T> {
        if is_contiguous_in_order(&self.dims, &self.strides, order) && self.offset == 0 {
            return self;
        }
        self.contiguous(order)
    }

    /// Returns `true` if the tensor data is contiguous in memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// assert!(t.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        is_contiguous_in_order(&self.dims, &self.strides, MemoryOrder::ColumnMajor)
            || is_contiguous_in_order(&self.dims, &self.strides, MemoryOrder::RowMajor)
    }

    /// Check if the tensor has column-major contiguous layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// assert!(t.is_col_major_contiguous());
    /// ```
    pub fn is_col_major_contiguous(&self) -> bool {
        is_contiguous_in_order(&self.dims, &self.strides, MemoryOrder::ColumnMajor)
    }

    /// Return a lazily-conjugated tensor (shared buffer, flag flip).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let a = Tensor::from_slice(&data, &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let a_conj = a.conj();
    /// assert!(a_conj.is_conjugated());
    /// ```
    pub fn conj(&self) -> Tensor<T>
    where
        T: Conjugate,
    {
        Tensor::from_parts(
            self.buffer.clone(),
            self.dims.clone(),
            self.strides.clone(),
            self.offset,
            self.logical_memory_space,
            self.preferred_compute_device,
            self.event.clone(),
            !self.conjugated,
            None,
        )
    }

    /// Consume this tensor and return a lazily-conjugated version.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let tc = t.into_conj();
    /// assert!(tc.is_conjugated());
    /// ```
    pub fn into_conj(self) -> Tensor<T>
    where
        T: Conjugate,
    {
        Tensor::from_parts(
            self.buffer,
            self.dims,
            self.strides,
            self.offset,
            self.logical_memory_space,
            self.preferred_compute_device,
            self.event,
            !self.conjugated,
            None,
        )
    }

    fn triangular_part(&self, diagonal: isize, half: TriangularHalf) -> Tensor<T> {
        self.wait();
        if self.ndim() <= 1 {
            return self.contiguous(MemoryOrder::ColumnMajor);
        }

        let m = self.dims[0];
        let n = self.dims[1];
        let out_strides = compute_contiguous_strides(&self.dims, MemoryOrder::ColumnMajor);
        let mut data = vec![T::zero(); self.len()];
        if data.is_empty() {
            return self.materialized_from_vec(data, MemoryOrder::ColumnMajor);
        }

        let src = self.cpu_backed_slice_or_panic(match half {
            TriangularHalf::Lower => "tril",
            TriangularHalf::Upper => "triu",
        });
        let batch_dims = &self.dims[2..];
        let mut batch_index = vec![0usize; batch_dims.len()];
        let n_batch = batch_dims.iter().product::<usize>().max(1);

        for _ in 0..n_batch {
            let src_batch_off: isize = batch_index
                .iter()
                .enumerate()
                .map(|(axis, &idx)| idx as isize * self.strides[axis + 2])
                .sum();
            let dst_batch_off: isize = batch_index
                .iter()
                .enumerate()
                .map(|(axis, &idx)| idx as isize * out_strides[axis + 2])
                .sum();

            for j in 0..n {
                for i in 0..m {
                    let keep = match half {
                        TriangularHalf::Lower => (j as isize - i as isize) <= diagonal,
                        TriangularHalf::Upper => (j as isize - i as isize) >= diagonal,
                    };
                    if !keep {
                        continue;
                    }

                    let src_pos = (self.offset
                        + src_batch_off
                        + i as isize * self.strides[0]
                        + j as isize * self.strides[1]) as usize;
                    let dst_pos = (dst_batch_off
                        + i as isize * out_strides[0]
                        + j as isize * out_strides[1]) as usize;
                    data[dst_pos] = src[src_pos];
                }
            }

            for axis in 0..batch_dims.len() {
                batch_index[axis] += 1;
                if batch_index[axis] < batch_dims[axis] {
                    break;
                }
                batch_index[axis] = 0;
            }
        }

        Tensor::from_parts(
            DataBuffer::from_vec(data),
            self.dims.clone(),
            std::sync::Arc::from(out_strides),
            0,
            self.logical_memory_space,
            self.preferred_compute_device,
            None,
            self.conjugated,
            None,
        )
    }

    /// Extract the lower triangular part of a matrix.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::<f64>::ones(&[3, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let lower = a.tril(0);
    /// assert_eq!(lower.dims(), &[3, 3]);
    /// ```
    pub fn tril(&self, diagonal: isize) -> Tensor<T> {
        self.triangular_part(diagonal, TriangularHalf::Lower)
    }

    /// Extract the upper triangular part of a matrix.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::<f64>::ones(&[3, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let upper = a.triu(0);
    /// assert_eq!(upper.dims(), &[3, 3]);
    /// ```
    pub fn triu(&self, diagonal: isize) -> Tensor<T> {
        self.triangular_part(diagonal, TriangularHalf::Upper)
    }
}
