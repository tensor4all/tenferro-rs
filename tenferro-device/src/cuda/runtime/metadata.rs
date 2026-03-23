use super::*;
use crate::{Error, Result};

fn required_storage_len(
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    label: &str,
) -> Result<usize> {
    if dims.len() != strides.len() {
        return Err(Error::InvalidArgument(format!(
            "{label} rank mismatch: dims={} strides={}",
            dims.len(),
            strides.len()
        )));
    }
    if dims.contains(&0) {
        return Ok(0);
    }

    let mut min_pos = offset;
    let mut max_pos = offset;
    for (axis, (&dim, &stride)) in dims.iter().zip(strides).enumerate() {
        let extent = isize::try_from(dim - 1)
            .ok()
            .and_then(|d| d.checked_mul(stride))
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{label} extent overflow for dimension {axis} (size={dim}, stride={stride})"
                ))
            })?;
        if extent >= 0 {
            max_pos = max_pos.checked_add(extent).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{label} maximum offset overflow for dimension {axis}"
                ))
            })?;
        } else {
            min_pos = min_pos.checked_add(extent).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{label} minimum offset overflow for dimension {axis}"
                ))
            })?;
        }
    }

    if min_pos < 0 {
        return Err(Error::InvalidArgument(format!(
            "{label} accesses negative buffer positions {}..={}",
            min_pos, max_pos
        )));
    }

    let max_pos = usize::try_from(max_pos).map_err(|_| {
        Error::InvalidArgument(format!(
            "{label} maximum position {max_pos} exceeds usize range"
        ))
    })?;
    max_pos
        .checked_add(1)
        .ok_or_else(|| Error::InvalidArgument(format!("{label} storage length overflow")))
}

fn validate_len(
    actual_len: usize,
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    label: &str,
) -> Result<()> {
    let required = required_storage_len(dims, strides, offset, label)?;
    if actual_len < required {
        return Err(Error::InvalidArgument(format!(
            "{label} length mismatch: actual={actual_len} required={required}"
        )));
    }
    Ok(())
}

impl CudaRuntime {
    /// Materialize a zero-based `iota` metadata tensor into a raw CUDA buffer.
    ///
    /// The output is integer metadata; bool metadata is intentionally not
    /// supported here because bool is `u8`-backed in tenferro today and the
    /// Phase 1 metadata family only needs the integer iota primitive.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live CUDA allocation on this runtime's device and
    /// be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let dst = runtime.alloc_raw::<i32>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_generate_iota_i32(dst, 4, &[4], &[1], 0).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_generate_iota_i32(
        &self,
        dst: *mut i32,
        dst_len: usize,
        dims: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata iota destination",
        )?;
        let spec = MetadataGenerateSpec::new(dims, dst_strides, dst_offset)?;
        unsafe { self.metadata_generate_iota_i32_raw(dst, &spec) }
    }

    /// Materialize a constant `i32` metadata tensor into a raw CUDA buffer.
    ///
    /// The tensor layout follows the same storage rules as
    /// [`metadata_generate_iota_i32`](Self::metadata_generate_iota_i32).
    ///
    /// # Safety
    ///
    /// `dst` must point to a live CUDA allocation on this runtime's device and
    /// be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let dst = runtime.alloc_raw::<i32>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_generate_constant_i32(dst, 4, &[4], &[1], 0, 7).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_generate_constant_i32(
        &self,
        dst: *mut i32,
        dst_len: usize,
        dims: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        value: i32,
    ) -> Result<()> {
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata constant destination",
        )?;
        let spec = MetadataGenerateSpec::new(dims, dst_strides, dst_offset)?;
        unsafe { self.metadata_generate_constant_i32_raw(dst, value, &spec) }
    }

    /// Materialize a constant logical-bool metadata tensor into a raw CUDA buffer.
    ///
    /// The tensor is stored in tenferro's current `u8`-backed logical bool
    /// representation, so `value` is written as `0` or `1` in device storage.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live CUDA allocation on this runtime's device and
    /// be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let dst = runtime.alloc_raw::<u8>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_generate_constant_bool(dst, 4, &[4], &[1], 0, true).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_generate_constant_bool(
        &self,
        dst: *mut u8,
        dst_len: usize,
        dims: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        value: bool,
    ) -> Result<()> {
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata constant destination",
        )?;
        let spec = MetadataGenerateSpec::new(dims, dst_strides, dst_offset)?;
        unsafe { self.metadata_generate_constant_bool_raw(dst, value as u8, &spec) }
    }

    /// Materialize an equality/inequality metadata comparison over two raw CUDA buffers.
    ///
    /// `equal = true` selects equality; `equal = false` selects inequality.
    /// The result is stored as logical bool metadata (`u8`-backed today).
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live CUDA allocations on this
    /// runtime's device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc_raw::<i32>(4).unwrap();
    /// let rhs = runtime.alloc_raw::<i32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<u8>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_binary_i32_bool(true, lhs, 4, rhs, 4, dst, 4, &[4], &[1], 0, &[1], 0, &[1], 0).unwrap();
    ///     runtime.free_raw(lhs).unwrap();
    ///     runtime.free_raw(rhs).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_binary_i32_bool(
        &self,
        equal: bool,
        lhs: *const i32,
        lhs_len: usize,
        rhs: *const i32,
        rhs_len: usize,
        dst: *mut u8,
        dst_len: usize,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_len(
            lhs_len,
            dims,
            lhs_strides,
            lhs_offset,
            "metadata binary lhs",
        )?;
        validate_len(
            rhs_len,
            dims,
            rhs_strides,
            rhs_offset,
            "metadata binary rhs",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata binary dst",
        )?;
        let spec = MetadataBinarySpec::new(
            dims,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            dst_strides,
            dst_offset,
        )?;
        let op = if equal {
            MetadataBinaryOp::Equal
        } else {
            MetadataBinaryOp::NotEqual
        };
        unsafe {
            self.metadata_binary_i32_bool_raw(
                if matches!(op, MetadataBinaryOp::Equal) {
                    0
                } else {
                    1
                },
                lhs,
                rhs,
                dst,
                &spec,
            )
        }
    }

    /// Materialize integer metadata arithmetic over two raw CUDA buffers.
    ///
    /// `op_code` follows the metadata family encoding used by tenferro:
    /// `0 = Equal`, `1 = NotEqual`, `2 = Add`, `3 = Sub`, `4 = Mul`.
    /// Equality and inequality are accepted so the same path can also be used
    /// for integer comparison tests.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live CUDA allocations on this
    /// runtime's device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc_raw::<i32>(4).unwrap();
    /// let rhs = runtime.alloc_raw::<i32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<i32>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_binary_i32_i32(
    ///         2,
    ///         lhs,
    ///         4,
    ///         rhs,
    ///         4,
    ///         dst,
    ///         4,
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         0,
    ///     )
    ///     .unwrap();
    ///     runtime.free_raw(lhs).unwrap();
    ///     runtime.free_raw(rhs).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_binary_i32_i32(
        &self,
        op_code: i32,
        lhs: *const i32,
        lhs_len: usize,
        rhs: *const i32,
        rhs_len: usize,
        dst: *mut i32,
        dst_len: usize,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_len(
            lhs_len,
            dims,
            lhs_strides,
            lhs_offset,
            "metadata binary lhs",
        )?;
        validate_len(
            rhs_len,
            dims,
            rhs_strides,
            rhs_offset,
            "metadata binary rhs",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata binary dst",
        )?;
        let spec = MetadataBinarySpec::new(
            dims,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            dst_strides,
            dst_offset,
        )?;
        unsafe { self.metadata_binary_i32_i32_raw(op_code, lhs, rhs, dst, &spec) }
    }

    /// Materialize an equality/inequality/bitand comparison over two bool metadata buffers.
    ///
    /// `op_code` follows the metadata family encoding used by tenferro:
    /// `0 = Equal`, `1 = NotEqual`, `2 = BitAnd`.
    ///
    /// Bool metadata is `u8`-backed today; the output is also `u8`-backed.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live CUDA allocations on this
    /// runtime's device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc_raw::<u8>(4).unwrap();
    /// let rhs = runtime.alloc_raw::<u8>(4).unwrap();
    /// let dst = runtime.alloc_raw::<u8>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_binary_bool_bool(
    ///         2,
    ///         lhs,
    ///         4,
    ///         rhs,
    ///         4,
    ///         dst,
    ///         4,
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         0,
    ///     )
    ///     .unwrap();
    ///     runtime.free_raw(lhs).unwrap();
    ///     runtime.free_raw(rhs).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_binary_bool_bool(
        &self,
        op_code: i32,
        lhs: *const u8,
        lhs_len: usize,
        rhs: *const u8,
        rhs_len: usize,
        dst: *mut u8,
        dst_len: usize,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_len(
            lhs_len,
            dims,
            lhs_strides,
            lhs_offset,
            "metadata binary lhs",
        )?;
        validate_len(
            rhs_len,
            dims,
            rhs_strides,
            rhs_offset,
            "metadata binary rhs",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata binary dst",
        )?;
        let spec = MetadataBinarySpec::new(
            dims,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            dst_strides,
            dst_offset,
        )?;
        unsafe { self.metadata_binary_bool_bool_raw(op_code, lhs, rhs, dst, &spec) }
    }

    /// Materialize a `where` ternary metadata operation for integer outputs.
    ///
    /// Bool metadata is `u8`-backed today.
    ///
    /// # Safety
    ///
    /// `cond`, `on_true`, `on_false`, and `dst` must point to live CUDA
    /// allocations on this runtime's device and be compatible with the provided
    /// layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let cond = runtime.alloc_raw::<u8>(4).unwrap();
    /// let on_true = runtime.alloc_raw::<i32>(4).unwrap();
    /// let on_false = runtime.alloc_raw::<i32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<i32>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_where_i32(cond, 4, on_true, 4, on_false, 4, dst, 4, &[4], &[1], 0, &[1], 0, &[1], 0, &[1], 0).unwrap();
    ///     runtime.free_raw(cond).unwrap();
    ///     runtime.free_raw(on_true).unwrap();
    ///     runtime.free_raw(on_false).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_where_i32(
        &self,
        cond: *const u8,
        cond_len: usize,
        on_true: *const i32,
        true_len: usize,
        on_false: *const i32,
        false_len: usize,
        dst: *mut i32,
        dst_len: usize,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        true_strides: &[isize],
        true_offset: isize,
        false_strides: &[isize],
        false_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_len(
            cond_len,
            dims,
            cond_strides,
            cond_offset,
            "metadata ternary cond",
        )?;
        validate_len(
            true_len,
            dims,
            true_strides,
            true_offset,
            "metadata ternary true",
        )?;
        validate_len(
            false_len,
            dims,
            false_strides,
            false_offset,
            "metadata ternary false",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata ternary dst",
        )?;
        let spec = MetadataTernarySpec::new(
            dims,
            cond_strides,
            cond_offset,
            true_strides,
            true_offset,
            false_strides,
            false_offset,
            dst_strides,
            dst_offset,
        )?;
        unsafe { self.metadata_where_i32_raw(cond, on_true, on_false, dst, &spec) }
    }

    /// Materialize a `where` ternary metadata operation for bool outputs.
    ///
    /// Bool metadata is `u8`-backed today.
    ///
    /// # Safety
    ///
    /// `cond`, `on_true`, `on_false`, and `dst` must point to live CUDA
    /// allocations on this runtime's device and be compatible with the provided
    /// layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let cond = runtime.alloc_raw::<u8>(4).unwrap();
    /// let on_true = runtime.alloc_raw::<u8>(4).unwrap();
    /// let on_false = runtime.alloc_raw::<u8>(4).unwrap();
    /// let dst = runtime.alloc_raw::<u8>(4).unwrap();
    /// unsafe {
    ///     runtime.metadata_where_bool(cond, 4, on_true, 4, on_false, 4, dst, 4, &[4], &[1], 0, &[1], 0, &[1], 0, &[1], 0).unwrap();
    ///     runtime.free_raw(cond).unwrap();
    ///     runtime.free_raw(on_true).unwrap();
    ///     runtime.free_raw(on_false).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_where_bool(
        &self,
        cond: *const u8,
        cond_len: usize,
        on_true: *const u8,
        true_len: usize,
        on_false: *const u8,
        false_len: usize,
        dst: *mut u8,
        dst_len: usize,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        true_strides: &[isize],
        true_offset: isize,
        false_strides: &[isize],
        false_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_len(
            cond_len,
            dims,
            cond_strides,
            cond_offset,
            "metadata ternary cond",
        )?;
        validate_len(
            true_len,
            dims,
            true_strides,
            true_offset,
            "metadata ternary true",
        )?;
        validate_len(
            false_len,
            dims,
            false_strides,
            false_offset,
            "metadata ternary false",
        )?;
        validate_len(
            dst_len,
            dims,
            dst_strides,
            dst_offset,
            "metadata ternary dst",
        )?;
        let spec = MetadataTernarySpec::new(
            dims,
            cond_strides,
            cond_offset,
            true_strides,
            true_offset,
            false_strides,
            false_offset,
            dst_strides,
            dst_offset,
        )?;
        unsafe { self.metadata_where_bool_raw(cond, on_true, on_false, dst, &spec) }
    }

    /// Reduce integer metadata with a sum reduction.
    ///
    /// # Safety
    ///
    /// `input` and `dst` must point to live CUDA allocations on this runtime's
    /// device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc_raw::<i32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<i32>(2).unwrap();
    /// unsafe {
    ///     runtime.metadata_reduce_sum_i32(input, 4, dst, 2, &[2, 2], &[1, 2], 0, &[2], &[2], 0, &[0], &[1]).unwrap();
    ///     runtime.free_raw(input).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_reduce_sum_i32(
        &self,
        input: *const i32,
        input_len: usize,
        dst: *mut i32,
        dst_len: usize,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        validate_len(
            input_len,
            input_dims,
            input_strides,
            input_offset,
            "metadata reduction input",
        )?;
        validate_len(
            dst_len,
            output_dims,
            output_strides,
            output_offset,
            "metadata reduction dst",
        )?;
        let spec = MetadataReductionSpec::new(
            input_dims,
            input_strides,
            input_offset,
            output_dims,
            output_strides,
            output_offset,
            kept_axes,
            reduced_axes,
        )?;
        unsafe { self.metadata_reduce_sum_i32_raw(input, dst, &spec) }
    }

    /// Reduce bool metadata with a sum reduction, returning integer metadata.
    ///
    /// Bool metadata is `u8`-backed today.
    ///
    /// # Safety
    ///
    /// `input` and `dst` must point to live CUDA allocations on this runtime's
    /// device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc_raw::<u8>(4).unwrap();
    /// let dst = runtime.alloc_raw::<i32>(2).unwrap();
    /// unsafe {
    ///     runtime.metadata_reduce_sum_bool(input, 4, dst, 2, &[2, 2], &[1, 2], 0, &[2], &[2], 0, &[0], &[1]).unwrap();
    ///     runtime.free_raw(input).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_reduce_sum_bool(
        &self,
        input: *const u8,
        input_len: usize,
        dst: *mut i32,
        dst_len: usize,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        validate_len(
            input_len,
            input_dims,
            input_strides,
            input_offset,
            "metadata reduction input",
        )?;
        validate_len(
            dst_len,
            output_dims,
            output_strides,
            output_offset,
            "metadata reduction dst",
        )?;
        let spec = MetadataReductionSpec::new(
            input_dims,
            input_strides,
            input_offset,
            output_dims,
            output_strides,
            output_offset,
            kept_axes,
            reduced_axes,
        )?;
        unsafe { self.metadata_reduce_sum_bool_raw(input, dst, &spec) }
    }

    /// Reduce bool metadata with a logical-all reduction.
    ///
    /// Bool metadata is `u8`-backed today.
    ///
    /// # Safety
    ///
    /// `input` and `dst` must point to live CUDA allocations on this runtime's
    /// device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc_raw::<u8>(4).unwrap();
    /// let dst = runtime.alloc_raw::<u8>(2).unwrap();
    /// unsafe {
    ///     runtime.metadata_reduce_all_bool(input, 4, dst, 2, &[2, 2], &[1, 2], 0, &[2], &[2], 0, &[0], &[1]).unwrap();
    ///     runtime.free_raw(input).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_reduce_all_bool(
        &self,
        input: *const u8,
        input_len: usize,
        dst: *mut u8,
        dst_len: usize,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        validate_len(
            input_len,
            input_dims,
            input_strides,
            input_offset,
            "metadata reduction input",
        )?;
        validate_len(
            dst_len,
            output_dims,
            output_strides,
            output_offset,
            "metadata reduction dst",
        )?;
        let spec = MetadataReductionSpec::new(
            input_dims,
            input_strides,
            input_offset,
            output_dims,
            output_strides,
            output_offset,
            kept_axes,
            reduced_axes,
        )?;
        unsafe { self.metadata_reduce_all_bool_raw(input, dst, &spec) }
    }

    /// Reduce bool metadata with a logical-any reduction.
    ///
    /// Bool metadata is `u8`-backed today.
    ///
    /// # Safety
    ///
    /// `input` and `dst` must point to live CUDA allocations on this runtime's
    /// device and be compatible with the provided layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc_raw::<u8>(4).unwrap();
    /// let dst = runtime.alloc_raw::<u8>(2).unwrap();
    /// unsafe {
    ///     runtime.metadata_reduce_any_bool(input, 4, dst, 2, &[2, 2], &[1, 2], 0, &[2], &[2], 0, &[0], &[1]).unwrap();
    ///     runtime.free_raw(input).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn metadata_reduce_any_bool(
        &self,
        input: *const u8,
        input_len: usize,
        dst: *mut u8,
        dst_len: usize,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        validate_len(
            input_len,
            input_dims,
            input_strides,
            input_offset,
            "metadata reduction input",
        )?;
        validate_len(
            dst_len,
            output_dims,
            output_strides,
            output_offset,
            "metadata reduction dst",
        )?;
        let spec = MetadataReductionSpec::new(
            input_dims,
            input_strides,
            input_offset,
            output_dims,
            output_strides,
            output_offset,
            kept_axes,
            reduced_axes,
        )?;
        unsafe { self.metadata_reduce_any_bool_raw(input, dst, &spec) }
    }
}
