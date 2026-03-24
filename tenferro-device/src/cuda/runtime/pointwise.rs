use super::*;
use crate::{Error, Result};
use std::ffi::c_void;

mod pointwise_complex;
mod pointwise_metadata;
mod pointwise_metadata_cast;
mod pointwise_real;

impl CudaRuntime {
    pub fn alloc<T>(&self, len: usize) -> Result<CudaBuffer<T>> {
        let ptr = self.alloc_raw::<T>(len)?;
        Ok(CudaBuffer::new(self.context(), ptr.cast::<c_void>(), len))
    }

    /// Copies a host slice into a device buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// runtime.copy_htod(&[1.0_f32, 2.0, 3.0, 4.0], &buffer).unwrap();
    /// ```
    pub fn copy_htod<T>(&self, src: &[T], dst: &CudaBuffer<T>) -> Result<()> {
        self.ensure_same_device(dst.device_id())?;
        unsafe { self.copy_htod_raw(src, dst.device_ptr(), dst.len()) }
    }

    /// Copies a device buffer into a host vector.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// let host = runtime.copy_dtoh::<f32>(&buffer).unwrap();
    /// assert_eq!(host.len(), 4);
    /// ```
    pub fn copy_dtoh<T>(&self, src: &CudaBuffer<T>) -> Result<Vec<T>> {
        self.ensure_same_device(src.device_id())?;
        unsafe { self.copy_dtoh_raw(src.device_ptr(), src.len()) }
    }

    /// Copies the contents of one device buffer into another.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// runtime.copy_dtod(&src, &dst).unwrap();
    /// ```
    pub fn copy_dtod<T>(&self, src: &CudaBuffer<T>, dst: &CudaBuffer<T>) -> Result<()> {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        if src.len() != dst.len() {
            return Err(Error::InvalidArgument(format!(
                "device/device length mismatch: src={} dst={}",
                src.len(),
                dst.len()
            )));
        }

        unsafe { self.copy_dtod_raw(src.device_ptr(), dst.device_ptr(), src.len()) }
    }

    /// Launches the generic strided-copy kernel from one device buffer to another.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(24).unwrap();
    /// let dst = runtime.alloc::<f32>(24).unwrap();
    /// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// runtime.copy_strided(&src, &dst, &spec).unwrap();
    /// ```
    pub fn copy_strided<T>(
        &self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        spec: &StridedCopySpec,
    ) -> Result<()> {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        unsafe { self.copy_strided_raw(src.device_ptr(), dst.device_ptr(), spec) }
    }

    /// Launches the generic strided-copy kernel while applying a source-side transform.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec, StridedCopyTransform};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<num_complex::Complex64>(24).unwrap();
    /// let dst = runtime.alloc::<num_complex::Complex64>(24).unwrap();
    /// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// runtime.copy_strided_with_transform(&src, &dst, &spec, StridedCopyTransform::Conj).unwrap();
    /// ```
    pub fn copy_strided_with_transform<T: 'static>(
        &self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        spec: &StridedCopySpec,
        transform: StridedCopyTransform,
    ) -> Result<()> {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        unsafe {
            self.copy_strided_raw_with_transform(
                src.device_ptr(),
                dst.device_ptr(),
                spec,
                transform,
            )
        }
    }

    /// Packs two source views into a freshly allocated contiguous destination buffer.
    ///
    /// The source views must live on the same device, have the same rank, and match on every
    /// dimension except `axis`. The destination is allocated on the same device and is laid out
    /// contiguously in the requested order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let left = runtime.alloc::<f32>(2).unwrap();
    /// let right = runtime.alloc::<f32>(4).unwrap();
    /// let left_spec = StridedCopySpec::to_contiguous(&[1, 2], &[1, 1], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// let right_spec = StridedCopySpec::to_contiguous(&[2, 2], &[1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// let packed = runtime.pack_concat_sources(&left, &left_spec, &right, &right_spec, 0, ContiguousOrder::ColumnMajor).unwrap();
    /// assert_eq!(packed.len(), 6);
    /// ```
    pub fn pack_concat_sources<T>(
        &self,
        left: &CudaBuffer<T>,
        left_spec: &StridedCopySpec,
        right: &CudaBuffer<T>,
        right_spec: &StridedCopySpec,
        axis: usize,
        order: ContiguousOrder,
    ) -> Result<CudaBuffer<T>> {
        self.ensure_same_device(left.device_id())?;
        self.ensure_same_device(right.device_id())?;
        if left_spec.dims.len() != left_spec.src_strides.len()
            || right_spec.dims.len() != right_spec.src_strides.len()
        {
            if left_spec.dims.len() != left_spec.src_strides.len() {
                return Err(Error::InvalidArgument(format!(
                    "concat pack left spec rank mismatch: dims={} src_strides={}",
                    left_spec.dims.len(),
                    left_spec.src_strides.len()
                )));
            }
            return Err(Error::InvalidArgument(format!(
                "concat pack right spec rank mismatch: dims={} src_strides={}",
                right_spec.dims.len(),
                right_spec.src_strides.len()
            )));
        }
        if left_spec.dims.len() != right_spec.dims.len() {
            return Err(Error::InvalidArgument(format!(
                "concat pack source rank mismatch: left={} right={}",
                left_spec.dims.len(),
                right_spec.dims.len()
            )));
        }
        if axis >= left_spec.dims.len() {
            return Err(Error::InvalidArgument(format!(
                "concat axis {axis} out of range for rank {}",
                left_spec.dims.len()
            )));
        }
        for dim_axis in 0..left_spec.dims.len() {
            if dim_axis != axis && left_spec.dims[dim_axis] != right_spec.dims[dim_axis] {
                return Err(Error::InvalidArgument(format!(
                    "concat dimension mismatch at axis {dim_axis}: left={} right={}",
                    left_spec.dims[dim_axis], right_spec.dims[dim_axis]
                )));
            }
        }

        let mut dst_dims = left_spec.dims.clone();
        dst_dims[axis] = dst_dims[axis]
            .checked_add(right_spec.dims[axis])
            .ok_or_else(|| Error::InvalidArgument("concat dimension overflow".into()))?;
        let dst_len = checked_numel(&dst_dims)?;
        let dst = self.alloc::<T>(dst_len)?;
        if dst_len == 0 {
            return Ok(dst);
        }
        let dst_strides = contiguous_strides(&dst_dims, order)?;
        let axis_stride = dst_strides[axis];
        let right_axis_len = isize::try_from(left_spec.dims[axis]).map_err(|_| {
            Error::InvalidArgument(format!(
                "concat axis length {} exceeds isize range",
                left_spec.dims[axis]
            ))
        })?;
        let right_dst_offset = right_axis_len
            .checked_mul(axis_stride)
            .ok_or_else(|| Error::InvalidArgument("concat destination offset overflow".into()))?;

        let left_dst_spec = StridedCopySpec {
            dims: left_spec.dims.clone(),
            src_strides: left_spec.src_strides.clone(),
            src_offset: left_spec.src_offset,
            dst_strides: dst_strides.clone(),
            dst_offset: 0,
        };
        let right_dst_spec = StridedCopySpec {
            dims: right_spec.dims.clone(),
            src_strides: right_spec.src_strides.clone(),
            src_offset: right_spec.src_offset,
            dst_strides,
            dst_offset: right_dst_offset,
        };

        let stream = unsafe {
            self.launch_strided_copy_raw(left.device_ptr(), dst.device_ptr(), &left_dst_spec)?
        };
        unsafe {
            self.launch_strided_copy_raw(right.device_ptr(), dst.device_ptr(), &right_dst_spec)?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))?;
        Ok(dst)
    }

    /// Launches the keep-count-driven trailing zero-fill kernel from one device buffer to another.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ZeroTrailingByCountsSpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(8).unwrap();
    /// let dst = runtime.alloc::<f32>(8).unwrap();
    /// let keep_counts = runtime.alloc::<f32>(2).unwrap();
    /// let spec = ZeroTrailingByCountsSpec::new(&[2, 2, 2], &[1, 2, 4], 0, &[1, 2, 4], 0, &[1], 0, 1, 2).unwrap();
    /// runtime.zero_trailing_by_counts(&src, &dst, &keep_counts, &spec).unwrap();
    /// ```
    pub fn zero_trailing_by_counts<T, R>(
        &self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        keep_counts: &CudaBuffer<R>,
        spec: &ZeroTrailingByCountsSpec,
    ) -> Result<()>
    where
        R: RuntimeKeepCountScalar,
    {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        self.ensure_same_device(keep_counts.device_id())?;
        let expected_keep_count_len = checked_numel(&spec.dims[spec.structural_rank..])?;
        if keep_counts.len() != expected_keep_count_len {
            return Err(Error::InvalidArgument(format!(
                "keep-count buffer length mismatch: expected {} got {}",
                expected_keep_count_len,
                keep_counts.len()
            )));
        }
        unsafe {
            self.zero_trailing_by_counts_raw(
                src.device_ptr(),
                dst.device_ptr(),
                keep_counts.device_ptr(),
                spec,
            )
        }
    }

    /// Launches the triangular-copy kernel from one device buffer to another.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, TriangularHalf, TriangularPartSpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(24).unwrap();
    /// let dst = runtime.alloc::<f32>(24).unwrap();
    /// let spec = TriangularPartSpec::new(&[3, 2, 4], &[1, 3, 6], 0, &[1, 3, 6], 0, 0, TriangularHalf::Upper).unwrap();
    /// runtime.triangular_part(&src, &dst, &spec).unwrap();
    /// ```
    pub fn triangular_part<T>(
        &self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        spec: &TriangularPartSpec,
    ) -> Result<()> {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        unsafe { self.triangular_part_raw(src.device_ptr(), dst.device_ptr(), spec) }
    }

    /// Launches the triangular-merge kernel from two device buffers into a destination buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, TriangularMergeSpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lower = runtime.alloc::<f32>(24).unwrap();
    /// let upper = runtime.alloc::<f32>(24).unwrap();
    /// let dst = runtime.alloc::<f32>(24).unwrap();
    /// let spec = TriangularMergeSpec::new(
    ///     &[3, 2, 4],
    ///     &[1, 3, 6],
    ///     0,
    ///     &[1, 3, 6],
    ///     0,
    ///     &[1, 3, 6],
    ///     0,
    /// ).unwrap();
    /// runtime.triangular_merge(&lower, &upper, &dst, &spec).unwrap();
    /// ```
    pub fn triangular_merge<T>(
        &self,
        lower_src: &CudaBuffer<T>,
        upper_src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        spec: &TriangularMergeSpec,
    ) -> Result<()> {
        self.ensure_same_device(lower_src.device_id())?;
        self.ensure_same_device(upper_src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        unsafe {
            self.triangular_merge_raw(
                lower_src.device_ptr(),
                upper_src.device_ptr(),
                dst.device_ptr(),
                spec,
            )
        }
    }
}
