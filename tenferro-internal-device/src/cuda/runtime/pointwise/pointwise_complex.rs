use super::*;
use crate::{Error, Result};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use num_complex::{Complex32, Complex64};

impl CudaRuntime {
    pub unsafe fn pointwise_scale_complex32_real_f32_raw(
        &self,
        alpha: KernelComplex32,
        lhs: *const Complex32,
        rhs: *const f32,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Complex32,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: KernelComplex32,
    ) -> Result<()> {
        self.pointwise_scale_complex_real_raw_impl(
            COMPLEX_SCALE_KERNEL_NAME_F32,
            alpha,
            lhs,
            rhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            dst,
            dst_strides,
            dst_offset,
            beta,
        )
    }

    /// Launches the Layer 0 complex-by-real pointwise multiply kernel for `Complex64 * f64` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_device::cuda::runtime::{self, KernelComplex64};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc::<Complex64>(4).unwrap();
    /// let rhs = runtime.alloc::<f64>(4).unwrap();
    /// let dst = runtime.alloc::<Complex64>(4).unwrap();
    /// let alpha = KernelComplex64 { re: 1.0, im: 0.0 };
    /// let beta = KernelComplex64 { re: 0.0, im: 0.0 };
    /// unsafe {
    ///     runtime.pointwise_scale_complex64_real_f64_raw(
    ///         alpha,
    ///         lhs.device_ptr().cast_const(),
    ///         rhs.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///         beta,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_scale_complex64_real_f64_raw(
        &self,
        alpha: KernelComplex64,
        lhs: *const Complex64,
        rhs: *const f64,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Complex64,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: KernelComplex64,
    ) -> Result<()> {
        self.pointwise_scale_complex_real_raw_impl(
            COMPLEX_SCALE_KERNEL_NAME_F64,
            alpha,
            lhs,
            rhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            dst,
            dst_strides,
            dst_offset,
            beta,
        )
    }

    fn pointwise_scale_complex_real_raw_impl<Dst, Src>(
        &self,
        kernel_name: &str,
        alpha: Dst,
        lhs: *const Src,
        rhs: *const <Src as ComplexScaleSrc>::Real,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Src,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: Dst,
    ) -> Result<()>
    where
        Dst: cudarc::driver::DeviceRepr,
        Src: ComplexScaleSrc,
    {
        validate_pointwise_rank(dims, lhs_strides, Some(rhs_strides), dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_complex_real_kernel(self, kernel_name)?;
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
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("complex scale rank exceeds i32 range".into()))?;
        let lhs_offset = i64::try_from(lhs_offset).map_err(|_| {
            Error::InvalidArgument("complex scale lhs offset exceeds i64 range".into())
        })?;
        let rhs_offset = i64::try_from(rhs_offset).map_err(|_| {
            Error::InvalidArgument("complex scale rhs offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("complex scale destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("complex scale numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("complex scale currently requires len <= u32::MAX".into())
        })?;
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
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA complex scale kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_unary_complex_real_raw_impl<Dst, Src>(
        &self,
        kernel_name: &str,
        op: ComplexRealUnaryOp,
        alpha: Dst,
        src: *const Src,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: Dst,
        dst: *mut Dst,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()>
    where
        Dst: RuntimeRealScalar,
        Src: Copy + 'static,
    {
        validate_pointwise_rank(dims, src_strides, None, dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_complex_real_kernel(self, kernel_name)?;
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
        let opcode = complex_real_opcode(op);
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
                .map_err(|err| cuda_error("CUDA complex-real unary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_complex_scale_raw_impl<KernelComplex, Complex>(
        &self,
        kernel_name: &str,
        alpha: KernelComplex,
        lhs: *const Complex,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const <Complex as ComplexScaleSrc>::Real,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: KernelComplex,
        dst: *mut Complex,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()>
    where
        KernelComplex: Copy + cudarc::driver::DeviceRepr + 'static,
        Complex: ComplexScaleSrc + Copy + 'static,
    {
        validate_pointwise_rank(dims, lhs_strides, Some(rhs_strides), dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_complex_scale_kernel(self, kernel_name)?;
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
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("pointwise rank exceeds i32 range".into()))?;
        let lhs_offset = i64::try_from(lhs_offset)
            .map_err(|_| Error::InvalidArgument("pointwise lhs offset exceeds i64 range".into()))?;
        let rhs_offset = i64::try_from(rhs_offset)
            .map_err(|_| Error::InvalidArgument("pointwise rhs offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(dst_offset)
            .map_err(|_| Error::InvalidArgument("pointwise dst offset exceeds i64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("pointwise numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise currently requires len <= u32::MAX".into())
        })?;
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
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA complex-scale kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the Layer 0 complex-to-real unary kernel for `Complex32 -> f32` data.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex32;
    /// use tenferro_device::cuda::runtime::{self, ComplexRealUnaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<Complex32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_unary_complex32_to_real_f32_raw(
    ///         ComplexRealUnaryOp::Abs,
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
    pub unsafe fn pointwise_unary_complex32_to_real_f32_raw(
        &self,
        op: ComplexRealUnaryOp,
        alpha: f32,
        src: *const Complex32,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: f32,
        dst: *mut f32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_unary_complex_real_raw_impl(
            COMPLEX_REAL_UNARY_KERNEL_NAME_F32,
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

    /// Launches the Layer 0 complex-to-real unary kernel for `Complex64 -> f64` data.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_device::cuda::runtime::{self, ComplexRealUnaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<Complex64>(4).unwrap();
    /// let dst = runtime.alloc::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_unary_complex64_to_real_f64_raw(
    ///         ComplexRealUnaryOp::Abs,
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
    pub unsafe fn pointwise_unary_complex64_to_real_f64_raw(
        &self,
        op: ComplexRealUnaryOp,
        alpha: f64,
        src: *const Complex64,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: f64,
        dst: *mut f64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_unary_complex_real_raw_impl(
            COMPLEX_REAL_UNARY_KERNEL_NAME_F64,
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

    /// Launches the Layer 0 complex-scale kernel for `Complex32 × f32 -> Complex32` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible
    /// with the provided layout metadata.
    #[allow(private_interfaces)]
    pub unsafe fn pointwise_mul_complex32_real_f32_raw(
        &self,
        alpha: Complex32,
        lhs: *const Complex32,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const f32,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: Complex32,
        dst: *mut Complex32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_complex_scale_raw_impl::<KernelComplex32, Complex32>(
            COMPLEX_SCALE_KERNEL_NAME_F32,
            alpha.into(),
            lhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            beta.into(),
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 complex-scale kernel for `Complex64 × f64 -> Complex64` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible
    /// with the provided layout metadata.
    #[allow(private_interfaces)]
    pub unsafe fn pointwise_mul_complex64_real_f64_raw(
        &self,
        alpha: Complex64,
        lhs: *const Complex64,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const f64,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: Complex64,
        dst: *mut Complex64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_complex_scale_raw_impl::<KernelComplex64, Complex64>(
            COMPLEX_SCALE_KERNEL_NAME_F64,
            alpha.into(),
            lhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            beta.into(),
            dst,
            dst_strides,
            dst_offset,
        )
    }
}
