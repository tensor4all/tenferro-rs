use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use super::{kernels::*, shared::*, state::CudaRuntime};
use crate::{Error, Result};

impl CudaRuntime {
    /// Launches the generic strided-copy kernel from a raw device source to a raw device destination.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with `spec`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<f32>(24).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(24).unwrap();
    /// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// unsafe {
    ///     runtime.copy_strided_raw(src, dst, &spec).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_strided_raw<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
    ) -> Result<()> {
        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let stream =
            self.launch_strided_copy_raw_impl(src, dst, spec, STRIDED_COPY_TRANSFORM_NONE)?;
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the generic strided-copy kernel with a source-side transform from a raw device source to a raw device destination.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with `spec`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec, StridedCopyTransform};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<num_complex::Complex64>(24).unwrap();
    /// let dst = runtime.alloc_raw::<num_complex::Complex64>(24).unwrap();
    /// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// unsafe {
    ///     runtime.copy_strided_raw_with_transform(src, dst, &spec, StridedCopyTransform::Conj).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_strided_raw_with_transform<T: StridedCopyTransformElement>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
        transform: StridedCopyTransform,
    ) -> Result<()> {
        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let stream = self.launch_strided_copy_raw_with_transform(src, dst, spec, transform)?;
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    unsafe fn launch_strided_copy_raw_impl<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
        source_transform: i32,
    ) -> Result<std::sync::Arc<CudaStream>> {
        if spec.dims.len() != spec.src_strides.len() || spec.dims.len() != spec.dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "strided copy rank mismatch: dims={} src_strides={} dst_strides={}",
                spec.dims.len(),
                spec.src_strides.len(),
                spec.dst_strides.len()
            )));
        }

        let numel = checked_numel(&spec.dims)?;

        self.bind_context()?;
        let ctx = self.context();
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(strided_copy_ptx()?)
            .map_err(|err| cuda_error("CUDA module load", err))?;
        let kernel = module
            .load_function(STRIDED_COPY_KERNEL_NAME)
            .map_err(|err| cuda_error("CUDA load function", err))?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("strided copy rank exceeds i32 range".into()))?;
        let src_offset = i64::try_from(spec.src_offset)
            .map_err(|_| Error::InvalidArgument("source offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("destination offset exceeds i64 range".into()))?;
        let elem_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| Error::InvalidArgument("element size exceeds u64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("strided copy numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("strided copy currently requires len <= u32::MAX".into())
        })?;
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
                .arg(&source_transform)
                .arg(&ndim)
                .arg(&elem_size)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA strided-copy kernel launch", err))?;
        }

        Ok(stream)
    }

    pub(super) unsafe fn launch_strided_copy_raw<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
    ) -> Result<std::sync::Arc<CudaStream>> {
        self.launch_strided_copy_raw_impl(src, dst, spec, STRIDED_COPY_TRANSFORM_NONE)
    }

    pub(super) unsafe fn launch_strided_copy_raw_with_transform<T: StridedCopyTransformElement>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
        transform: StridedCopyTransform,
    ) -> Result<std::sync::Arc<CudaStream>> {
        if matches!(transform, StridedCopyTransform::Conj) && !T::SUPPORTS_CONJ_STRIDED_COPY {
            return Err(Error::InvalidArgument(
                "strided copy conj transform requires Complex32 or Complex64 element type".into(),
            ));
        }

        self.launch_strided_copy_raw_impl(src, dst, spec, strided_copy_transform_code(transform))
    }

    /// Launches the keep-count-driven trailing zero-fill kernel on raw device allocations.
    ///
    /// # Safety
    ///
    /// `src`, `dst`, and `keep_counts` must point to live device allocations compatible
    /// with `spec`.
    pub unsafe fn zero_trailing_by_counts_raw<T, R>(
        &self,
        src: *const T,
        dst: *mut T,
        keep_counts: *const R,
        spec: &ZeroTrailingByCountsSpec,
    ) -> Result<()>
    where
        R: RuntimeKeepCountScalar,
    {
        if spec.dims.len() != spec.src_strides.len() || spec.dims.len() != spec.dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "zero-trailing rank mismatch: dims={} src_strides={} dst_strides={}",
                spec.dims.len(),
                spec.src_strides.len(),
                spec.dst_strides.len()
            )));
        }

        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let batch_dims = &spec.dims[spec.structural_rank..];
        let count_numel = checked_numel(batch_dims)?;
        let batch_rank = i32::try_from(batch_dims.len())
            .map_err(|_| Error::InvalidArgument("batch rank exceeds i32 range".into()))?;
        let axis_len = i64::try_from(spec.dims[spec.axis])
            .map_err(|_| Error::InvalidArgument("axis length exceeds i64 range".into()))?;
        let keep_count_offset = i64::try_from(spec.keep_count_offset)
            .map_err(|_| Error::InvalidArgument("keep-count offset exceeds i64 range".into()))?;
        let src_offset = i64::try_from(spec.src_offset)
            .map_err(|_| Error::InvalidArgument("source offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("destination offset exceeds i64 range".into()))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("zero-trailing rank exceeds i32 range".into()))?;
        let axis = i32::try_from(spec.axis)
            .map_err(|_| Error::InvalidArgument("axis exceeds i32 range".into()))?;
        let structural_rank = i32::try_from(spec.structural_rank)
            .map_err(|_| Error::InvalidArgument("structural_rank exceeds i32 range".into()))?;
        let elem_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| Error::InvalidArgument("element size exceeds u64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("zero-trailing numel exceeds u64 range".into()))?;
        let count_numel_u64 = u64::try_from(count_numel)
            .map_err(|_| Error::InvalidArgument("keep-count numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("zero-trailing currently requires len <= u32::MAX".into())
        })?;
        let count_numel_u32 = u32::try_from(count_numel).map_err(|_| {
            Error::InvalidArgument(
                "keep-count validation currently requires len <= u32::MAX".into(),
            )
        })?;

        let (validate_kernel, stream) = load_zero_trailing_kernel(self, R::VALIDATE_KERNEL_NAME)?;
        let batch_dims_dev = stream
            .clone_htod(&dims_to_i64(batch_dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD batch dims", err))?;
        let keep_count_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.keep_count_strides, "keep-count stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD keep-count strides", err))?;
        let status = self.alloc::<i32>(1)?;
        self.copy_htod(&[0i32], &status)?;
        let keep_counts_ptr = keep_counts as u64;
        let status_ptr = status.device_ptr() as u64;
        let validate_config = LaunchConfig {
            grid_dim: (count_numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&validate_kernel)
                .arg(&keep_counts_ptr)
                .arg(&batch_dims_dev)
                .arg(&keep_count_strides_dev)
                .arg(&keep_count_offset)
                .arg(&batch_rank)
                .arg(&axis_len)
                .arg(&count_numel_u64)
                .arg(&status_ptr)
                .launch(validate_config)
                .map_err(|err| cuda_error("CUDA keep-count validation launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))?;
        let status_host = self.copy_dtoh(&status)?;
        map_keep_count_status(status_host[0])?;

        let (zero_kernel, stream) = load_zero_trailing_kernel(self, R::ZERO_TRAILING_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let src_ptr = src as u64;
        let dst_ptr = dst as u64;
        let launch_config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&zero_kernel)
                .arg(&src_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&src_strides_dev)
                .arg(&src_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&keep_counts_ptr)
                .arg(&keep_count_strides_dev)
                .arg(&keep_count_offset)
                .arg(&ndim)
                .arg(&axis)
                .arg(&structural_rank)
                .arg(&elem_size)
                .arg(&numel_u64)
                .launch(launch_config)
                .map_err(|err| cuda_error("CUDA zero-trailing launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the triangular-copy kernel on raw device allocations.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with `spec`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime::{self, TriangularHalf, TriangularPartSpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<f32>(24).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(24).unwrap();
    /// let spec = TriangularPartSpec::new(&[3, 2, 4], &[1, 3, 6], 0, &[1, 3, 6], 0, 0, TriangularHalf::Lower).unwrap();
    /// unsafe {
    ///     runtime.triangular_part_raw(src, dst, &spec).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn triangular_part_raw<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &TriangularPartSpec,
    ) -> Result<()> {
        if spec.dims.len() != spec.src_strides.len() || spec.dims.len() != spec.dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "triangular copy rank mismatch: dims={} src_strides={} dst_strides={}",
                spec.dims.len(),
                spec.src_strides.len(),
                spec.dst_strides.len()
            )));
        }

        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        self.bind_context()?;
        let ctx = self.context();
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(triangular_part_ptx()?)
            .map_err(|err| cuda_error("CUDA module load", err))?;
        let kernel = module
            .load_function(TRIANGULAR_PART_KERNEL_NAME)
            .map_err(|err| cuda_error("CUDA load function", err))?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("triangular copy rank exceeds i32 range".into()))?;
        let src_offset = i64::try_from(spec.src_offset)
            .map_err(|_| Error::InvalidArgument("source offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("destination offset exceeds i64 range".into()))?;
        let diagonal = i64::try_from(spec.diagonal)
            .map_err(|_| Error::InvalidArgument("diagonal exceeds i64 range".into()))?;
        let half = spec.half.as_i32();
        let elem_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| Error::InvalidArgument("element size exceeds u64 range".into()))?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("triangular copy numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("triangular copy currently requires len <= u32::MAX".into())
        })?;
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
                .arg(&diagonal)
                .arg(&half)
                .arg(&elem_size)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA triangular-part kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the triangular-merge kernel on raw device allocations.
    ///
    /// # Safety
    ///
    /// `lower_src`, `upper_src`, and `dst` must point to live device allocations
    /// compatible with `spec`.
    pub unsafe fn triangular_merge_raw<T>(
        &self,
        lower_src: *const T,
        upper_src: *const T,
        dst: *mut T,
        spec: &TriangularMergeSpec,
    ) -> Result<()> {
        if spec.dims.len() != spec.lower_strides.len()
            || spec.dims.len() != spec.upper_strides.len()
            || spec.dims.len() != spec.dst_strides.len()
        {
            return Err(Error::InvalidArgument(format!(
                "triangular merge rank mismatch: dims={} lower_strides={} upper_strides={} dst_strides={}",
                spec.dims.len(),
                spec.lower_strides.len(),
                spec.upper_strides.len(),
                spec.dst_strides.len()
            )));
        }

        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        self.bind_context()?;
        let ctx = self.context();
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(triangular_merge_ptx()?)
            .map_err(|err| cuda_error("CUDA module load", err))?;
        let kernel = module
            .load_function(TRIANGULAR_MERGE_KERNEL_NAME)
            .map_err(|err| cuda_error("CUDA load function", err))?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let lower_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.lower_strides, "lower stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD lower strides", err))?;
        let upper_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.upper_strides, "upper stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD upper strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len()).map_err(|_| {
            Error::InvalidArgument("triangular merge rank exceeds i32 range".into())
        })?;
        let lower_offset = i64::try_from(spec.lower_offset)
            .map_err(|_| Error::InvalidArgument("lower offset exceeds i64 range".into()))?;
        let upper_offset = i64::try_from(spec.upper_offset)
            .map_err(|_| Error::InvalidArgument("upper offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("destination offset exceeds i64 range".into()))?;
        let elem_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| Error::InvalidArgument("element size exceeds u64 range".into()))?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("triangular merge numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("triangular merge currently requires len <= u32::MAX".into())
        })?;
        let lower_src_ptr = lower_src as u64;
        let upper_src_ptr = upper_src as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        stream
            .launch_builder(&kernel)
            .arg(&lower_src_ptr)
            .arg(&upper_src_ptr)
            .arg(&dst_ptr)
            .arg(&dims_dev)
            .arg(&lower_strides_dev)
            .arg(&lower_offset)
            .arg(&upper_strides_dev)
            .arg(&upper_offset)
            .arg(&dst_strides_dev)
            .arg(&dst_offset)
            .arg(&ndim)
            .arg(&elem_size)
            .arg(&numel_u64)
            .launch(config)
            .map_err(|err| cuda_error("CUDA triangular-merge kernel launch", err))?;
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }
}
