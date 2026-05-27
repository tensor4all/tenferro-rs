use super::super::{CudaRuntime, MetadataCastSpec};
use super::validate_len;
use crate::Result;

impl CudaRuntime {
    /// Cast logical-bool metadata into an `f32` tensor with affine blending.
    ///
    /// The output contract is `dst <- alpha * cast(input) + beta * dst`.
    ///
    /// # Safety
    ///
    /// `input` and `dst` must point to live CUDA allocations on this runtime's
    /// device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc_raw::<u8>(4).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_cast_bool_f32(input, 4, dst, 4, &[4], &[1], 0, &[1], 0, 1.0, 0.0).unwrap();
    ///     runtime.free_raw(input).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_cast_bool_f32(
        &self,
        input: *const u8,
        input_len: usize,
        dst: *mut f32,
        dst_len: usize,
        dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        validate_len(
            input_len,
            dims,
            input_strides,
            input_offset,
            "metadata cast input",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata cast destination",
        )?;
        let spec =
            MetadataCastSpec::new(dims, input_strides, input_offset, dst_strides, dst_offset)?;
        unsafe { self.metadata_cast_bool_f32_raw(input, dst, &spec, alpha, beta) }
    }

    /// Cast `i32` metadata into an `f32` tensor with affine blending.
    ///
    /// The output contract is `dst <- alpha * cast(input) + beta * dst`.
    ///
    /// # Safety
    ///
    /// `input` and `dst` must point to live CUDA allocations on this runtime's
    /// device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc_raw::<i32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_cast_i32_f32(input, 4, dst, 4, &[4], &[1], 0, &[1], 0, 1.0, 0.0).unwrap();
    ///     runtime.free_raw(input).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_cast_i32_f32(
        &self,
        input: *const i32,
        input_len: usize,
        dst: *mut f32,
        dst_len: usize,
        dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        validate_len(
            input_len,
            dims,
            input_strides,
            input_offset,
            "metadata cast input",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata cast destination",
        )?;
        let spec =
            MetadataCastSpec::new(dims, input_strides, input_offset, dst_strides, dst_offset)?;
        unsafe { self.metadata_cast_i32_f32_raw(input, dst, &spec, alpha, beta) }
    }

    /// Cast logical-bool metadata into an `f64` tensor with affine blending.
    ///
    /// The output contract is `dst <- alpha * cast(input) + beta * dst`.
    ///
    /// # Safety
    ///
    /// `input` and `dst` must point to live CUDA allocations on this runtime's
    /// device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc_raw::<u8>(4).unwrap();
    /// let dst = runtime.alloc_raw::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_cast_bool_f64(input, 4, dst, 4, &[4], &[1], 0, &[1], 0, 1.0, 0.0).unwrap();
    ///     runtime.free_raw(input).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_cast_bool_f64(
        &self,
        input: *const u8,
        input_len: usize,
        dst: *mut f64,
        dst_len: usize,
        dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        validate_len(
            input_len,
            dims,
            input_strides,
            input_offset,
            "metadata cast input",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata cast destination",
        )?;
        let spec =
            MetadataCastSpec::new(dims, input_strides, input_offset, dst_strides, dst_offset)?;
        unsafe { self.metadata_cast_bool_f64_raw(input, dst, &spec, alpha, beta) }
    }

    /// Cast `i32` metadata into an `f64` tensor with affine blending.
    ///
    /// The output contract is `dst <- alpha * cast(input) + beta * dst`.
    ///
    /// # Safety
    ///
    /// `input` and `dst` must point to live CUDA allocations on this runtime's
    /// device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc_raw::<i32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_cast_i32_f64(input, 4, dst, 4, &[4], &[1], 0, &[1], 0, 1.0, 0.0).unwrap();
    ///     runtime.free_raw(input).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_cast_i32_f64(
        &self,
        input: *const i32,
        input_len: usize,
        dst: *mut f64,
        dst_len: usize,
        dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        validate_len(
            input_len,
            dims,
            input_strides,
            input_offset,
            "metadata cast input",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata cast destination",
        )?;
        let spec =
            MetadataCastSpec::new(dims, input_strides, input_offset, dst_strides, dst_offset)?;
        unsafe { self.metadata_cast_i32_f64_raw(input, dst, &spec, alpha, beta) }
    }
}
