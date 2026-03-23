use super::*;
use crate::{Error, Result};
use cudarc::driver::{LaunchConfig, PushKernelArg};

impl CudaRuntime {
    fn pointwise_unary_real_raw_impl<T: RuntimeRealScalar>(
        &self,
        op: RealUnaryOp,
        alpha: T,
        src: *const T,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: T,
        dst: *mut T,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_pointwise_rank(dims, src_strides, None, dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_real_scalar_kernel(self, T::UNARY_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("pointwise unary rank exceeds i32 range".into()))?;
        let src_offset = i64::try_from(src_offset).map_err(|_| {
            Error::InvalidArgument("pointwise unary source offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("pointwise unary destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise unary numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise unary currently requires len <= u32::MAX".into())
        })?;
        let opcode = unary_opcode(op);
        let src_ptr = src as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&src_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&src_strides_dev)
                .arg(&src_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA real unary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_binary_real_raw_impl<T: RuntimeRealScalar>(
        &self,
        op: RealBinaryOp,
        alpha: T,
        lhs: *const T,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const T,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: T,
        dst: *mut T,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_pointwise_rank(dims, lhs_strides, Some(rhs_strides), dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_real_scalar_kernel(self, T::BINARY_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let lhs_strides_dev = stream
            .clone_htod(&to_i64_vec(lhs_strides, "lhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD lhs strides", err))?;
        let rhs_strides_dev = stream
            .clone_htod(&to_i64_vec(rhs_strides, "rhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD rhs strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len()).map_err(|_| {
            Error::InvalidArgument("pointwise binary rank exceeds i32 range".into())
        })?;
        let lhs_offset = i64::try_from(lhs_offset).map_err(|_| {
            Error::InvalidArgument("pointwise binary lhs offset exceeds i64 range".into())
        })?;
        let rhs_offset = i64::try_from(rhs_offset).map_err(|_| {
            Error::InvalidArgument("pointwise binary rhs offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("pointwise binary destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise binary numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise binary currently requires len <= u32::MAX".into())
        })?;
        let opcode = binary_opcode(op);
        let lhs_ptr = lhs as u64;
        let rhs_ptr = rhs as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&lhs_ptr)
                .arg(&rhs_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&lhs_strides_dev)
                .arg(&lhs_offset)
                .arg(&rhs_strides_dev)
                .arg(&rhs_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA real binary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_ternary_real_raw_impl<T: RuntimeRealScalar>(
        &self,
        op: RealTernaryOp,
        alpha: T,
        cond: *const T,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const T,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const T,
        false_strides: &[isize],
        false_offset: isize,
        beta: T,
        dst: *mut T,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_ternary_pointwise_rank(
            dims,
            cond_strides,
            true_strides,
            false_strides,
            dst_strides,
        )?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_real_scalar_kernel(self, T::TERNARY_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let cond_strides_dev = stream
            .clone_htod(&to_i64_vec(cond_strides, "cond stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD cond strides", err))?;
        let true_strides_dev = stream
            .clone_htod(&to_i64_vec(true_strides, "true stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD true strides", err))?;
        let false_strides_dev = stream
            .clone_htod(&to_i64_vec(false_strides, "false stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD false strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len()).map_err(|_| {
            Error::InvalidArgument("pointwise ternary rank exceeds i32 range".into())
        })?;
        let cond_offset = i64::try_from(cond_offset).map_err(|_| {
            Error::InvalidArgument("pointwise ternary condition offset exceeds i64 range".into())
        })?;
        let true_offset = i64::try_from(true_offset).map_err(|_| {
            Error::InvalidArgument("pointwise ternary true offset exceeds i64 range".into())
        })?;
        let false_offset = i64::try_from(false_offset).map_err(|_| {
            Error::InvalidArgument("pointwise ternary false offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("pointwise ternary destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise ternary numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise ternary currently requires len <= u32::MAX".into())
        })?;
        let opcode = ternary_opcode(op);
        let cond_ptr = cond as u64;
        let true_ptr = on_true as u64;
        let false_ptr = on_false as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&cond_ptr)
                .arg(&true_ptr)
                .arg(&false_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&cond_strides_dev)
                .arg(&cond_offset)
                .arg(&true_strides_dev)
                .arg(&true_offset)
                .arg(&false_strides_dev)
                .arg(&false_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA real ternary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn reduce_real_raw_impl<T: RuntimeRealScalar>(
        &self,
        op: RealReductionOp,
        alpha: T,
        input: *const T,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: T,
        output: *mut T,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        if input_dims.len() != input_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "reduction rank mismatch: input dims={} input strides={}",
                input_dims.len(),
                input_strides.len()
            )));
        }
        if output_dims.len() != output_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "reduction rank mismatch: output dims={} output strides={}",
                output_dims.len(),
                output_strides.len()
            )));
        }
        if output_dims.len() != kept_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "reduction kept-axis mismatch: output dims={} kept_axes={}",
                output_dims.len(),
                kept_axes.len()
            )));
        }
        for (output_axis, &input_axis) in kept_axes.iter().enumerate() {
            let Some(&expected_dim) = input_dims.get(input_axis) else {
                return Err(Error::InvalidArgument(format!(
                    "reduction kept axis {input_axis} out of bounds"
                )));
            };
            if output_dims[output_axis] != expected_dim {
                return Err(Error::InvalidArgument(format!(
                    "reduction output dim mismatch at axis {output_axis}: expected {expected_dim}, got {}",
                    output_dims[output_axis]
                )));
            }
        }

        let output_numel = checked_numel(output_dims)?;
        if output_numel == 0 {
            return Ok(());
        }

        let reduced_dims: Vec<usize> = reduced_axes
            .iter()
            .map(|&axis| {
                input_dims.get(axis).copied().ok_or_else(|| {
                    Error::InvalidArgument(format!("reduction axis {axis} out of bounds"))
                })
            })
            .collect::<Result<_>>()?;
        let reduced_total = checked_numel(&reduced_dims)?;
        if reduced_total == 0 && matches!(op, RealReductionOp::Max | RealReductionOp::Min) {
            return Err(Error::InvalidArgument(
                "extrema reduction requires a non-empty reduction domain".into(),
            ));
        }
        let (kernel, stream) = load_real_scalar_kernel(self, T::REDUCTION_KERNEL_NAME)?;
        let input_strides_dev = stream
            .clone_htod(&to_i64_vec(input_strides, "input stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD input strides", err))?;
        let output_dims_dev = stream
            .clone_htod(&dims_to_i64(output_dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD output dims", err))?;
        let output_strides_dev = stream
            .clone_htod(&to_i64_vec(output_strides, "output stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD output strides", err))?;
        let kept_axes_dev = stream
            .clone_htod(&axes_to_i32(kept_axes, "kept")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD kept axes", err))?;
        let reduced_axes_dev = stream
            .clone_htod(&axes_to_i32(reduced_axes, "reduced")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD reduced axes", err))?;
        let reduced_dims_dev = stream
            .clone_htod(&dims_to_i64(&reduced_dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD reduced dims", err))?;
        let kept_rank = i32::try_from(kept_axes.len())
            .map_err(|_| Error::InvalidArgument("reduction kept rank exceeds i32 range".into()))?;
        let reduced_rank = i32::try_from(reduced_axes.len()).map_err(|_| {
            Error::InvalidArgument("reduction reduced rank exceeds i32 range".into())
        })?;
        let input_offset = i64::try_from(input_offset).map_err(|_| {
            Error::InvalidArgument("reduction input offset exceeds i64 range".into())
        })?;
        let output_offset = i64::try_from(output_offset).map_err(|_| {
            Error::InvalidArgument("reduction output offset exceeds i64 range".into())
        })?;
        let output_numel_u64 = u64::try_from(output_numel).map_err(|_| {
            Error::InvalidArgument("reduction output numel exceeds u64 range".into())
        })?;
        let output_numel_u32 = u32::try_from(output_numel).map_err(|_| {
            Error::InvalidArgument("reduction currently requires len <= u32::MAX".into())
        })?;
        let reduced_total_u64 = u64::try_from(reduced_total).map_err(|_| {
            Error::InvalidArgument("reduction reduced total exceeds u64 range".into())
        })?;
        let opcode = reduction_opcode(op);
        let input_ptr = input as u64;
        let output_ptr = output as u64;
        let config = LaunchConfig {
            grid_dim: (output_numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&input_ptr)
                .arg(&output_ptr)
                .arg(&input_strides_dev)
                .arg(&input_offset)
                .arg(&output_dims_dev)
                .arg(&output_strides_dev)
                .arg(&output_offset)
                .arg(&kept_axes_dev)
                .arg(&kept_rank)
                .arg(&reduced_axes_dev)
                .arg(&reduced_dims_dev)
                .arg(&reduced_rank)
                .arg(&output_numel_u64)
                .arg(&reduced_total_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA real reduction kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the Layer 0 real unary kernel for `f32` data.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealUnaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_unary_real_f32_raw(
    ///         RealUnaryOp::Abs,
    ///         1.0,
    ///         src.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_unary_real_f32_raw(
        &self,
        op: RealUnaryOp,
        alpha: f32,
        src: *const f32,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: f32,
        dst: *mut f32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_unary_real_raw_impl(
            op,
            alpha,
            src,
            dims,
            src_strides,
            src_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real unary kernel for `f64` data.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealUnaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f64>(4).unwrap();
    /// let dst = runtime.alloc::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_unary_real_f64_raw(
    ///         RealUnaryOp::Abs,
    ///         1.0,
    ///         src.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_unary_real_f64_raw(
        &self,
        op: RealUnaryOp,
        alpha: f64,
        src: *const f64,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: f64,
        dst: *mut f64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_unary_real_raw_impl(
            op,
            alpha,
            src,
            dims,
            src_strides,
            src_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real binary kernel for `f32` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealBinaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc::<f32>(4).unwrap();
    /// let rhs = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_binary_real_f32_raw(
    ///         RealBinaryOp::Add,
    ///         1.0,
    ///         lhs.device_ptr().cast_const(),
    ///         rhs.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_binary_real_f32_raw(
        &self,
        op: RealBinaryOp,
        alpha: f32,
        lhs: *const f32,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const f32,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: f32,
        dst: *mut f32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_binary_real_raw_impl(
            op,
            alpha,
            lhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real binary kernel for `f64` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealBinaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc::<f64>(4).unwrap();
    /// let rhs = runtime.alloc::<f64>(4).unwrap();
    /// let dst = runtime.alloc::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_binary_real_f64_raw(
    ///         RealBinaryOp::Add,
    ///         1.0,
    ///         lhs.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         rhs.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_binary_real_f64_raw(
        &self,
        op: RealBinaryOp,
        alpha: f64,
        lhs: *const f64,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const f64,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: f64,
        dst: *mut f64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_binary_real_raw_impl(
            op,
            alpha,
            lhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real ternary kernel for `f32` data.
    ///
    /// # Safety
    ///
    /// `cond`, `on_true`, `on_false`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealTernaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let cond = runtime.alloc::<f32>(4).unwrap();
    /// let on_true = runtime.alloc::<f32>(4).unwrap();
    /// let on_false = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_ternary_real_f32_raw(
    ///         RealTernaryOp::Where,
    ///         1.0,
    ///         cond.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         on_true.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         on_false.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_ternary_real_f32_raw(
        &self,
        op: RealTernaryOp,
        alpha: f32,
        cond: *const f32,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const f32,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const f32,
        false_strides: &[isize],
        false_offset: isize,
        beta: f32,
        dst: *mut f32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_ternary_real_raw_impl(
            op,
            alpha,
            cond,
            dims,
            cond_strides,
            cond_offset,
            on_true,
            true_strides,
            true_offset,
            on_false,
            false_strides,
            false_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real ternary kernel for `f64` data.
    ///
    /// # Safety
    ///
    /// `cond`, `on_true`, `on_false`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealTernaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let cond = runtime.alloc::<f64>(4).unwrap();
    /// let on_true = runtime.alloc::<f64>(4).unwrap();
    /// let on_false = runtime.alloc::<f64>(4).unwrap();
    /// let dst = runtime.alloc::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_ternary_real_f64_raw(
    ///         RealTernaryOp::Where,
    ///         1.0,
    ///         cond.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         on_true.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         on_false.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_ternary_real_f64_raw(
        &self,
        op: RealTernaryOp,
        alpha: f64,
        cond: *const f64,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const f64,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const f64,
        false_strides: &[isize],
        false_offset: isize,
        beta: f64,
        dst: *mut f64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_ternary_real_raw_impl(
            op,
            alpha,
            cond,
            dims,
            cond_strides,
            cond_offset,
            on_true,
            true_strides,
            true_offset,
            on_false,
            false_strides,
            false_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real reduction kernel for `f32` data.
    ///
    /// # Safety
    ///
    /// `input` and `output` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealReductionOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc::<f32>(4).unwrap();
    /// let output = runtime.alloc::<f32>(2).unwrap();
    /// unsafe {
    ///     runtime.reduce_real_f32_raw(
    ///         RealReductionOp::Sum,
    ///         1.0,
    ///         input.device_ptr().cast_const(),
    ///         &[2, 2],
    ///         &[1, 2],
    ///         0,
    ///         0.0,
    ///         output.device_ptr(),
    ///         &[2],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         &[0],
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn reduce_real_f32_raw(
        &self,
        op: RealReductionOp,
        alpha: f32,
        input: *const f32,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: f32,
        output: *mut f32,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        self.reduce_real_raw_impl(
            op,
            alpha,
            input,
            input_dims,
            input_strides,
            input_offset,
            beta,
            output,
            output_dims,
            output_strides,
            output_offset,
            kept_axes,
            reduced_axes,
        )
    }

    /// Launches the Layer 0 real reduction kernel for `f64` data.
    ///
    /// # Safety
    ///
    /// `input` and `output` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealReductionOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc::<f64>(4).unwrap();
    /// let output = runtime.alloc::<f64>(2).unwrap();
    /// unsafe {
    ///     runtime.reduce_real_f64_raw(
    ///         RealReductionOp::Sum,
    ///         1.0,
    ///         input.device_ptr().cast_const(),
    ///         &[2, 2],
    ///         &[1, 2],
    ///         0,
    ///         0.0,
    ///         output.device_ptr(),
    ///         &[2],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         &[0],
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn reduce_real_f64_raw(
        &self,
        op: RealReductionOp,
        alpha: f64,
        input: *const f64,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: f64,
        output: *mut f64,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        self.reduce_real_raw_impl(
            op,
            alpha,
            input,
            input_dims,
            input_strides,
            input_offset,
            beta,
            output,
            output_dims,
            output_strides,
            output_offset,
            kept_axes,
            reduced_axes,
        )
    }
}
