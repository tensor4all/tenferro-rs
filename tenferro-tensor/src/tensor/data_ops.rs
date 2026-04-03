use std::any::TypeId;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
#[cfg(feature = "cuda")]
use tenferro_device::LogicalMemorySpace;
use tenferro_device::{checked_batch_count, unflatten_col_major_index_into, Error, Result};

use super::{Tensor, TensorParts};
use crate::layout::{compute_contiguous_strides, copy_strided, is_contiguous_in_order};
use crate::{DataBuffer, MemoryOrder};

#[inline]
fn checked_pos(offset: isize, i: isize, stride0: isize, j: isize, stride1: isize) -> Option<usize> {
    (i as isize)
        .checked_mul(stride0)
        .and_then(|d| offset.checked_add(d))
        .and_then(|off| {
            (j as isize)
                .checked_mul(stride1)
                .and_then(|d| off.checked_add(d))
        })
        .and_then(|pos| usize::try_from(pos).ok())
}

#[inline]
fn checked_batch_offset(
    batch_index: &[usize],
    strides: &[isize],
    offset_axis: usize,
) -> Option<isize> {
    batch_index
        .iter()
        .enumerate()
        .try_fold(0isize, |acc, (axis, &idx)| {
            (idx as isize)
                .checked_mul(strides[axis + offset_axis])
                .and_then(|v| acc.checked_add(v))
        })
}

enum TriangularHalf {
    Lower,
    Upper,
}

impl<T> Tensor<T> {
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
        T: tenferro_algebra::Conjugate,
    {
        Tensor::from_parts(TensorParts {
            buffer: self.buffer.clone(),
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: self.event.clone(),
            conjugated: !self.conjugated,
            fw_grad: None,
        })
    }

    /// Consume this tensor and return a lazily-conjugated version.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let tc = t.into_conj();
    /// assert!(tc.is_conjugated());
    /// ```
    pub fn into_conj(self) -> Tensor<T>
    where
        T: tenferro_algebra::Conjugate,
    {
        Tensor::from_parts(TensorParts {
            buffer: self.buffer,
            dims: self.dims,
            strides: self.strides,
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: self.event,
            conjugated: !self.conjugated,
            fw_grad: None,
        })
    }
}

impl<T: Scalar> Tensor<T> {
    /// Create a deep copy with an exclusively-owned contiguous buffer.
    ///
    /// Unlike [`clone`](Clone::clone) (which is a shallow `Arc` refcount
    /// bump), this always allocates a fresh buffer and copies element data.
    /// The returned tensor is contiguous in column-major order and has
    /// `buffer.is_unique() == true`, so [`set`](Tensor::set) and
    /// [`get_mut`](Tensor::get_mut) are guaranteed to succeed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let a = Tensor::<f64>::from_slice(
    ///     &[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// let b = a.clone(); // shallow — shares buffer
    ///
    /// let mut c = a.deep_clone(); // deep — independent buffer
    /// c.set(&[0, 0], 99.0).unwrap();
    /// assert_eq!(c.get(&[0, 0]), Some(&99.0));
    /// assert_eq!(a.get(&[0, 0]), Some(&1.0)); // original unchanged
    /// ```
    pub fn deep_clone(&self) -> Tensor<T> {
        self.wait();
        let order = MemoryOrder::ColumnMajor;
        let mut data = vec![T::zero(); self.len()];
        if !data.is_empty() {
            let dst_strides = compute_contiguous_strides(&self.dims, order);
            copy_strided(
                self.cpu_backed_slice_or_panic("deep_clone"),
                &self.dims,
                &self.strides,
                self.offset,
                &mut data,
                &dst_strides,
            );
        }
        self.materialized_from_vec(data, order)
    }

    /// Return a contiguous copy of this tensor in the given memory order.
    ///
    /// `order` controls the materialized output buffer only. It does not change
    /// the internal column-major semantics used by view operations such as
    /// [`reshape`](Tensor::reshape).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor).unwrap();
    /// let c = t.contiguous(MemoryOrder::RowMajor);
    /// assert!(c.is_contiguous());
    /// ```
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T> {
        self.wait();
        if is_contiguous_in_order(&self.dims, &self.strides, order) && self.offset == 0 {
            return Tensor::from_parts(TensorParts {
                buffer: self.buffer.clone(),
                dims: self.dims.clone(),
                strides: self.strides.clone(),
                offset: self.offset,
                logical_memory_space: self.logical_memory_space,
                preferred_compute_device: self.preferred_compute_device,
                event: self.event.clone(),
                conjugated: self.conjugated,
                fw_grad: self.fw_grad.clone(),
            });
        }

        #[cfg(feature = "cuda")]
        if matches!(
            self.logical_memory_space,
            LogicalMemorySpace::GpuMemory { .. }
        ) {
            return crate::cuda_runtime::contiguous_tensor(self, order)
                .unwrap_or_else(|err| panic!("contiguous: GPU materialization failed: {err}"));
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
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let c = t.into_contiguous(MemoryOrder::ColumnMajor);
    /// assert!(c.is_contiguous());
    /// ```
    pub fn into_contiguous(self, order: MemoryOrder) -> Tensor<T> {
        if is_contiguous_in_order(&self.dims, &self.strides, order) && self.offset == 0 {
            return Tensor::from_parts(TensorParts {
                buffer: self.buffer,
                dims: self.dims,
                strides: self.strides,
                offset: self.offset,
                logical_memory_space: self.logical_memory_space,
                preferred_compute_device: self.preferred_compute_device,
                event: self.event,
                conjugated: self.conjugated,
                fw_grad: self.fw_grad,
            });
        }
        self.contiguous(order)
    }

    /// Consume this tensor and return a contiguous column-major version.
    ///
    /// This is a convenience wrapper around `into_contiguous(MemoryOrder::ColumnMajor)`
    /// since column-major is tenferro's canonical internal layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor).unwrap();
    /// let col_major = t.into_column_major();
    /// assert!(col_major.is_col_major_contiguous());
    /// ```
    pub fn into_column_major(self) -> Tensor<T> {
        self.into_contiguous(MemoryOrder::ColumnMajor)
    }

    /// Copy tensor data into a flat `Vec<T>` in column-major order.
    ///
    /// The returned vector has length `self.len()` with elements laid out
    /// in column-major (Fortran) order. For a 2-D tensor with shape
    /// `[m, n]`, the first `m` elements are column 0, the next `m` are
    /// column 1, and so on.
    ///
    /// This method internally materializes a contiguous copy when the
    /// tensor is not already column-major contiguous, so it always
    /// returns owned data regardless of the original layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_row_major_slice(
    ///     &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3],
    /// ).unwrap();
    /// // Matrix (row-major input):
    /// //   [[1, 2, 3],
    /// //    [4, 5, 6]]
    /// // Column-major output: col0=[1,4], col1=[2,5], col2=[3,6]
    /// assert_eq!(t.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn to_vec(&self) -> Vec<T> {
        let c = self.contiguous(MemoryOrder::ColumnMajor);
        let slice = c
            .buffer()
            .as_slice()
            .expect("to_vec: CPU-only operation; GPU tensors are not supported");
        slice.to_vec()
    }

    fn triangular_part(&self, diagonal: isize, half: TriangularHalf) -> Result<Tensor<T>> {
        self.wait();
        if self.ndim() <= 1 {
            return Ok(self.contiguous(MemoryOrder::ColumnMajor));
        }

        #[cfg(feature = "cuda")]
        if matches!(
            self.logical_memory_space,
            LogicalMemorySpace::GpuMemory { .. }
        ) {
            return crate::cuda_runtime::triangular_part_tensor(
                self,
                diagonal,
                matches!(half, TriangularHalf::Lower),
            )
            .map_err(|err| Error::Backend(err.to_string()));
        }

        let m = self.dims[0];
        let n = self.dims[1];
        let out_strides = compute_contiguous_strides(&self.dims, MemoryOrder::ColumnMajor);
        let mut data = vec![T::zero(); self.len()];
        if data.is_empty() {
            return Ok(self.materialized_from_vec(data, MemoryOrder::ColumnMajor));
        }

        let src = self.cpu_backed_slice_or_panic(match half {
            TriangularHalf::Lower => "tril",
            TriangularHalf::Upper => "triu",
        });
        let batch_dims = &self.dims[2..];
        let n_batch = checked_batch_count(batch_dims).map_err(|err| {
            Error::InvalidArgument(format!(
                "triangular_part: invalid batch dims {:?}: {err}",
                batch_dims
            ))
        })?;
        let mut batch_index = vec![0usize; batch_dims.len()];

        for batch in 0..n_batch {
            if !batch_dims.is_empty() {
                unflatten_col_major_index_into(batch, batch_dims, &mut batch_index).map_err(
                    |err| {
                        Error::InvalidArgument(format!(
                            "triangular_part: failed to unflatten batch index {batch} for dims {:?}: {err}",
                            batch_dims
                        ))
                    },
                )?;
            }
            let src_batch_off = checked_batch_offset(&batch_index, &self.strides, 2).ok_or_else(
                || {
                    Error::InvalidArgument(format!(
                        "triangular_part: source batch offset overflow with batch_index {:?}, strides {:?}",
                        batch_index, self.strides
                    ))
                },
            )?;
            let dst_batch_off = checked_batch_offset(&batch_index, &out_strides, 2).ok_or_else(
                || {
                    Error::InvalidArgument(format!(
                        "triangular_part: destination batch offset overflow with batch_index {:?}, strides {:?}",
                        batch_index, out_strides
                    ))
                },
            )?;

            for j in 0..n {
                for i in 0..m {
                    let keep = match half {
                        TriangularHalf::Lower => (j as isize - i as isize) <= diagonal,
                        TriangularHalf::Upper => (j as isize - i as isize) >= diagonal,
                    };
                    if !keep {
                        continue;
                    }

                    let src_pos = self
                        .offset
                        .checked_add(src_batch_off)
                        .and_then(|off| {
                            checked_pos(off, i as isize, self.strides[0], j as isize, self.strides[1])
                        })
                        .ok_or_else(|| {
                            Error::InvalidArgument(format!(
                                "triangular_part: source position overflow at ({}, {}) with offset {}, batch_off {}, strides {:?}",
                                i, j, self.offset, src_batch_off, self.strides
                            ))
                        })?;
                    let dst_pos = checked_pos(
                        dst_batch_off,
                        i as isize,
                        out_strides[0],
                        j as isize,
                        out_strides[1],
                    )
                    .ok_or_else(|| {
                        Error::InvalidArgument(format!(
                            "triangular_part: destination position overflow at ({}, {}) with batch_off {}, strides {:?}",
                            i, j, dst_batch_off, out_strides
                        ))
                    })?;
                    data[dst_pos] = src[src_pos];
                }
            }
        }

        Ok(Tensor::from_parts(TensorParts {
            buffer: DataBuffer::from_vec(data),
            dims: self.dims.clone(),
            strides: std::sync::Arc::from(out_strides),
            offset: 0,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        }))
    }

    /// Extract the lower triangular part of a matrix.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::<f64>::ones(&[3, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let lower = a.tril(0).unwrap();
    /// assert_eq!(lower.dims(), &[3, 3]);
    /// ```
    pub fn tril(&self, diagonal: isize) -> Result<Tensor<T>> {
        self.triangular_part(diagonal, TriangularHalf::Lower)
    }

    /// Extract the upper triangular part of a matrix.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor::<f64>::ones(&[3, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let upper = a.triu(0).unwrap();
    /// assert_eq!(upper.dims(), &[3, 3]);
    /// ```
    pub fn triu(&self, diagonal: isize) -> Result<Tensor<T>> {
        self.triangular_part(diagonal, TriangularHalf::Upper)
    }
}

impl<T: Scalar> Tensor<T> {
    /// Materialize the logical tensor values into a fresh contiguous tensor.
    ///
    /// The returned tensor is resolved (`conjugated = false`) and clears any
    /// preferred compute-device hint. GPU tensors use the Layer 0 logical-copy
    /// substrate when available.
    pub(crate) fn materialize_logical_contiguous(&self, order: MemoryOrder) -> Tensor<T> {
        self.wait();

        #[cfg(feature = "cuda")]
        if matches!(
            self.logical_memory_space,
            LogicalMemorySpace::GpuMemory { .. }
        ) {
            return crate::cuda_runtime::materialize_logical_contiguous_tensor(self, order)
                .unwrap_or_else(|err| {
                    panic!("materialize_logical_contiguous: GPU materialization failed: {err}")
                });
        }

        let mut data = vec![T::zero(); self.len()];
        if !data.is_empty() {
            let dst_strides = compute_contiguous_strides(&self.dims, order);
            copy_strided(
                self.cpu_backed_slice_or_panic("materialize_logical_contiguous"),
                &self.dims,
                &self.strides,
                self.offset,
                &mut data,
                &dst_strides,
            );
            apply_logical_conjugation_if_needed(&mut data, self.conjugated);
        }

        Tensor::from_owned_contiguous_data(
            data,
            self.dims.clone(),
            order,
            self.logical_memory_space,
            None,
            false,
        )
    }
}

/// Apply logical conjugation in place when the element type supports it.
///
/// This uses a narrow private runtime dispatch so `Tensor::cat` / `Tensor::stack`
/// can stay generic over `Scalar` while built-in complex tensors still materialize
/// their logical values correctly.
fn apply_logical_conjugation_if_needed<T: Scalar + 'static>(data: &mut [T], conjugated: bool) {
    if !conjugated || data.is_empty() {
        return;
    }

    if TypeId::of::<T>() == TypeId::of::<Complex32>() {
        // SAFETY: the type check above guarantees `T` really is `Complex32`.
        let data = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<Complex32>(), data.len())
        };
        for value in data {
            *value = value.conj();
        }
        return;
    }

    if TypeId::of::<T>() == TypeId::of::<Complex64>() {
        // SAFETY: the type check above guarantees `T` really is `Complex64`.
        let data = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<Complex64>(), data.len())
        };
        for value in data {
            *value = value.conj();
        }
    }
}
