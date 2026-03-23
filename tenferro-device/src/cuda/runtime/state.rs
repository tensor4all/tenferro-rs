use std::{ffi::c_void, marker::PhantomData, mem::MaybeUninit, sync::Arc};

use cudarc::{
    driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg},
    runtime::result as cuda_result,
};
use num_complex::{Complex32, Complex64};

use super::{kernels::*, shared::*};
use crate::{Error, Result};

/// Shared CUDA runtime handle for one device ordinal.
///
/// The handle retains the CUDA primary context and exposes low-level memory
/// allocation and copy primitives that higher-level crates can reuse.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// assert_eq!(runtime.device_id(), 0);
/// ```
#[derive(Debug)]
pub struct CudaRuntime {
    context: Arc<CudaContext>,
}

impl CudaRuntime {
    fn new(device_id: usize) -> Result<Arc<Self>> {
        let context =
            CudaContext::new(device_id).map_err(|err| cuda_error("CUDA device init", err))?;
        Ok(Arc::new(Self { context }))
    }

    pub(super) fn bind_context(&self) -> Result<()> {
        self.context
            .bind_to_thread()
            .map_err(|err| cuda_error("CUDA context bind", err))
    }

    /// Returns the CUDA device ordinal this runtime is bound to.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// assert_eq!(runtime.device_id(), 0);
    /// ```
    pub fn device_id(&self) -> usize {
        self.context.ordinal()
    }

    /// Returns a clone of the shared CUDA context handle.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ctx = runtime.context();
    /// assert_eq!(ctx.ordinal(), 0);
    /// ```
    pub fn context(&self) -> Arc<CudaContext> {
        Arc::clone(&self.context)
    }

    /// Allocates a raw device pointer for `len` elements of `T`.
    ///
    /// # Safety
    ///
    /// The returned pointer must eventually be passed to [`CudaRuntime::free_raw`]
    /// on the same runtime.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub fn alloc_raw<T>(&self, len: usize) -> Result<*mut T> {
        self.bind_context()?;
        if len == 0 {
            return Ok(std::ptr::null_mut());
        }

        let ptr = unsafe { cuda_result::malloc_sync(checked_num_bytes::<T>(len)?) }
            .map_err(|err| cuda_error("cudaMalloc", err))?;
        Ok(ptr.cast::<T>())
    }

    /// Frees a raw device pointer previously allocated by [`CudaRuntime::alloc_raw`].
    ///
    /// # Safety
    ///
    /// `ptr` must either be null or a live allocation returned by this runtime.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub unsafe fn free_raw<T>(&self, ptr: *mut T) -> Result<()> {
        self.bind_context()?;
        if ptr.is_null() {
            return Ok(());
        }

        unsafe { cuda_result::free_sync(ptr.cast::<c_void>()) }
            .map_err(|err| cuda_error("cudaFree", err))
    }

    /// Copies a host slice into a raw device allocation.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live device allocation holding at least `dst_len` elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.copy_htod_raw(&[1.0_f32, 2.0, 3.0, 4.0], ptr, 4).unwrap();
    ///     runtime.free_raw(ptr).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_htod_raw<T>(&self, src: &[T], dst: *mut T, dst_len: usize) -> Result<()> {
        if src.len() != dst_len {
            return Err(Error::InvalidArgument(format!(
                "host/device length mismatch: src={} dst={dst_len}",
                src.len()
            )));
        }

        self.bind_context()?;
        if src.is_empty() {
            return Ok(());
        }

        unsafe { cuda_result::memcpy_htod_sync(dst.cast::<c_void>(), as_byte_slice(src)) }
            .map_err(|err| cuda_error("cudaMemcpyHtoD", err))
    }

    /// Copies a raw device allocation into a host vector.
    ///
    /// # Safety
    ///
    /// `src` must point to a live device allocation holding at least `len` elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// let host = unsafe { runtime.copy_dtoh_raw(ptr, 4).unwrap() };
    /// assert_eq!(host.len(), 4);
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub unsafe fn copy_dtoh_raw<T>(&self, src: *const T, len: usize) -> Result<Vec<T>> {
        self.bind_context()?;
        if len == 0 {
            return Ok(Vec::new());
        }

        let num_bytes = checked_num_bytes::<T>(len)?;
        let mut host = Vec::<MaybeUninit<T>>::with_capacity(len);
        unsafe { host.set_len(len) };
        let host_bytes =
            unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), num_bytes) };

        unsafe { cuda_result::memcpy_dtoh_sync(host_bytes, src.cast::<c_void>()) }
            .map_err(|err| cuda_error("cudaMemcpyDtoH", err))?;

        let ptr = host.as_mut_ptr().cast::<T>();
        let len = host.len();
        let cap = host.capacity();
        std::mem::forget(host);
        Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) })
    }

    /// Copies one raw device allocation into another.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations holding at least `len` elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<f32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.copy_dtod_raw(src, dst, 4).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_dtod_raw<T>(&self, src: *const T, dst: *mut T, len: usize) -> Result<()> {
        self.bind_context()?;
        if len == 0 {
            return Ok(());
        }

        unsafe {
            cuda_result::memcpy_dtod_sync(
                dst.cast::<c_void>(),
                src.cast::<c_void>(),
                checked_num_bytes::<T>(len)?,
            )
        }
        .map_err(|err| cuda_error("cudaMemcpyDtoD", err))
    }

    /// Launches the generic strided-copy kernel from a raw device source to a raw device destination.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with `spec`.
    ///
    /// # Examples
    ///
    /// ```ignore
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
    /// ```ignore
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
    pub unsafe fn copy_strided_raw_with_transform<T: 'static>(
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
    ) -> Result<Arc<CudaStream>> {
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

    unsafe fn launch_strided_copy_raw<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
    ) -> Result<Arc<CudaStream>> {
        self.launch_strided_copy_raw_impl(src, dst, spec, STRIDED_COPY_TRANSFORM_NONE)
    }

    unsafe fn launch_strided_copy_raw_with_transform<T: 'static>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
        transform: StridedCopyTransform,
    ) -> Result<Arc<CudaStream>> {
        if matches!(transform, StridedCopyTransform::Conj) && !supports_conj_strided_copy::<T>() {
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
    /// ```ignore
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

    /// Launches the Layer 0 complex-by-real pointwise multiply kernel for `Complex32 * f32` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex32;
    /// use tenferro_device::cuda::runtime::{self, KernelComplex32};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc::<Complex32>(4).unwrap();
    /// let rhs = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<Complex32>(4).unwrap();
    /// let alpha = KernelComplex32 { re: 1.0, im: 0.0 };
    /// let beta = KernelComplex32 { re: 0.0, im: 0.0 };
    /// unsafe {
    ///     runtime.pointwise_scale_complex32_real_f32_raw(
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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

    /// Allocates a device buffer for `len` elements of `T`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert_eq!(buffer.len(), 4);
    /// ```
    pub fn alloc<T>(&self, len: usize) -> Result<CudaBuffer<T>> {
        let ptr = self.alloc_raw::<T>(len)?;
        Ok(CudaBuffer::new(
            Arc::clone(&self.context),
            ptr.cast::<c_void>(),
            len,
        ))
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
        unsafe { self.copy_htod_raw(src, dst.ptr.cast::<T>(), dst.len()) }
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
        unsafe { self.copy_dtoh_raw(src.ptr.cast::<T>(), src.len()) }
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

        unsafe { self.copy_dtod_raw(src.ptr.cast::<T>(), dst.ptr.cast::<T>(), src.len()) }
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
        unsafe { self.copy_strided_raw(src.ptr.cast::<T>(), dst.ptr.cast::<T>(), spec) }
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
                src.ptr.cast::<T>(),
                dst.ptr.cast::<T>(),
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
            self.launch_strided_copy_raw(left.ptr.cast::<T>(), dst.ptr.cast::<T>(), &left_dst_spec)?
        };
        unsafe {
            self.launch_strided_copy_raw(
                right.ptr.cast::<T>(),
                dst.ptr.cast::<T>(),
                &right_dst_spec,
            )?;
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
                src.ptr.cast::<T>(),
                dst.ptr.cast::<T>(),
                keep_counts.ptr.cast::<R>(),
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
        unsafe { self.triangular_part_raw(src.ptr.cast::<T>(), dst.ptr.cast::<T>(), spec) }
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
                lower_src.ptr.cast::<T>(),
                upper_src.ptr.cast::<T>(),
                dst.ptr.cast::<T>(),
                spec,
            )
        }
    }

    fn ensure_same_device(&self, device_id: usize) -> Result<()> {
        if self.device_id() == device_id {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "buffer belongs to device {device_id}, runtime is bound to device {}",
                self.device_id()
            )))
        }
    }
}

/// Owning CUDA device buffer allocated by [`CudaRuntime`].
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// let buffer = runtime.alloc::<f32>(4).unwrap();
/// assert_eq!(buffer.len(), 4);
/// ```
#[derive(Debug)]
pub struct CudaBuffer<T> {
    context: Arc<CudaContext>,
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Sync> Sync for CudaBuffer<T> {}

impl<T> CudaBuffer<T> {
    fn new(context: Arc<CudaContext>, ptr: *mut c_void, len: usize) -> Self {
        Self {
            context,
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Returns the number of elements in this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert_eq!(buffer.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the raw device pointer for this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert!(!buffer.device_ptr().is_null());
    /// ```
    pub fn device_ptr(&self) -> *mut T {
        self.ptr.cast::<T>()
    }

    /// Returns the device ordinal that owns this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert_eq!(buffer.device_id(), 0);
    /// ```
    pub fn device_id(&self) -> usize {
        self.context.ordinal()
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        if !self.ptr.is_null() {
            let _ = unsafe { cuda_result::free_sync(self.ptr) };
        }
    }
}

/// Returns the shared runtime handle for one CUDA device ordinal.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// assert_eq!(runtime.device_id(), 0);
/// ```
pub fn get_or_init(device_id: usize) -> Result<Arc<CudaRuntime>> {
    let mut cache = runtime_cache()
        .lock()
        .map_err(|_| Error::DeviceError("CUDA runtime cache mutex poisoned".into()))?;
    if let Some(runtime) = cache.get(&device_id) {
        return Ok(Arc::clone(runtime));
    }

    let runtime = CudaRuntime::new(device_id)?;
    cache.insert(device_id, Arc::clone(&runtime));
    Ok(runtime)
}
