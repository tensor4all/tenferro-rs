//! CubeCL-based GPU backend for tenferro tensors.
//!
//! This module provides GPU acceleration via [CubeCL](https://github.com/tracel-ai/cubecl)
//! running on NVIDIA CUDA devices. It is gated behind the `cuda` feature flag and
//! requires **CUDA 12.8+** with a compatible NVIDIA GPU.
//!
//! # Enabling the feature
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! tenferro-gpu = { version = "...", features = ["cuda"] }
//! ```
//!
//! You must also enable a CPU backend (`cpu-faer` or `cpu-blas`); the CubeCL backend
//! complements the CPU path but does not replace it.
//!
//! # Prerequisites
//!
//! - NVIDIA GPU with CUDA compute capability ≥ 7.0
//! - CUDA Toolkit 12.8 or newer installed (provides NVRTC for JIT kernel compilation)
//! - cuTENSOR shared library available on `LD_LIBRARY_PATH`
//!
//! ## Environment variables
//!
//! | Variable | Purpose |
//! |----------|---------|
//! | `CUDA_PATH` | CUDA toolkit root (e.g. `/usr/local/cuda-12.8`) |
//! | `CUBECL_DEBUG_LOG` | Set to `0` to suppress verbose JIT logs |
//! | `TENFERRO_CUTENSOR_PATH` | Override cuTENSOR library search path |
//!
//! # Basic usage
//!
//! GPU tensors must be explicitly uploaded before use on the device and downloaded
//! back to the host afterwards (no implicit CPU↔GPU transfer, following the PyTorch
//! convention).
//!
//! ```rust
//! use tenferro_gpu::{download_tensor, gpu_available, upload_tensor, CudaBackend};
//! use tenferro_tensor::{Tensor, TensorElementwise, TypedTensor};
//!
//! fn main() -> tenferro_tensor::Result<()> {
//! if !gpu_available() {
//!     return Ok(());
//! }
//!
//! // 1. Create the GPU backend (device ordinal 0)
//! let mut backend = CudaBackend::new(0)?;
//!
//! // 2. Create tensors on the CPU
//! let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
//! let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]));
//!
//! // 3. Upload to GPU
//! let gpu_a = upload_tensor(backend.runtime(), &a)?;
//! let gpu_b = upload_tensor(backend.runtime(), &b)?;
//!
//! // 4. Compute on GPU
//! let gpu_c = backend.add(&gpu_a, &gpu_b)?;
//!
//! // 5. Download result back to CPU
//! let cpu_c = download_tensor(backend.runtime(), &gpu_c)?;
//! assert_eq!(cpu_c.shape(), &[2]);
//! Ok(())
//! }
//! ```
//!
//! # Running GPU tests
//!
//! All GPU tests are marked `#[ignore]` so that `cargo test --features cuda`
//! passes on machines without a GPU. To actually run them:
//!
//! ```sh
//! CUBECL_DEBUG_LOG=0 \
//! CUDA_PATH=/usr/local/cuda-12.8 \
//! cargo test -p tenferro-gpu --features cuda -- --ignored
//! ```

use std::any::{Any, TypeId};
use std::cell::OnceCell;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};

use cubecl::client::ComputeClient;
use cubecl::features::AtomicUsage;
use cubecl::prelude::{
    ArrayArg, ComplexCore as CubeComplex, CubeElement, CubePrimitive, Float as CubeFloat,
    Numeric as CubeNumeric,
};
use cubecl::prelude::{Int as CubeInt, StorageType, TensorBinding, Type};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use num_complex::{Complex32, Complex64};
use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::CacheStats;

use crate::backend::{
    BackendCachedDot, BackendRuntimeCache, BackendSessionHost, TensorAnalytic, TensorBackend,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::kernels::reduce::{self as cubecl_reduce, ReduceStrategy};
use crate::kernels::{diagonal, elementwise, indexing, structural};
use crate::{
    Buffer, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, Tensor, TensorRank,
    TensorViewCanonicalization, TypedTensor, TypedTensorView, TypedTensorViewMut,
};

mod dispatch;
mod ffi;
mod fusion;
mod gemm;
pub(crate) mod interop;
mod memory;
pub(crate) mod op_descriptor;
mod runtime;

use dispatch::{
    alloc_output, comptime_sequence, cube_count_for_len, cube_dim_1d, dtype_mismatch,
    ensure_axes_unique, ensure_axis, ensure_rank, ensure_resident_on_runtime,
    ensure_view_mut_resident_on_runtime, ensure_view_resident_on_runtime, launch_binary,
    launch_binary_tensor, launch_compare_bool, launch_nullary_into, launch_select_bool,
    launch_ternary, launch_unary, launch_unary_tensor, launch_unary_tensor_into,
    ternary_dtype_mismatch, typed_tensor_array_arg, typed_tensor_array_arg_as,
    typed_tensor_binding, typed_view_array_arg, typed_view_mut_array_arg,
};

pub use memory::{device_ptr, download_tensor, upload_tensor};
pub use runtime::{gpu_available, CudaRuntime};

fn unsupported_dtype(op: &'static str, dtype: crate::DType) -> crate::Error {
    crate::Error::backend_failure(op, format!("unsupported dtype {dtype:?}"))
}

fn op_name(
    kind: PrimitiveOpKind,
    launch: op_descriptor::GpuLaunchKind,
) -> crate::Result<&'static str> {
    op_descriptor::require_gpu_descriptor(kind, launch).map(|descriptor| descriptor.name)
}

fn ensure_atomic_add_supported<T: CubePrimitive>(
    client: &ComputeClient<CubeclCudaRuntime>,
    op: &'static str,
) -> crate::Result<()> {
    let elem = T::as_type_native_unchecked().elem_type();
    let atomic_ty = Type::new(StorageType::Atomic(elem));
    if client
        .properties()
        .atomic_type_usage(atomic_ty)
        .contains(AtomicUsage::Add)
    {
        Ok(())
    } else {
        Err(crate::Error::backend_failure(
            op,
            format!("CubeCL runtime does not support atomic add for {elem:?}"),
        ))
    }
}

fn checked_dim_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            crate::Error::backend_failure(
                op,
                format!("{role} product overflow for shape {shape:?}"),
            )
        })
    })
}

fn view_strides_i64(strides: &[isize], op: &'static str) -> crate::Result<Vec<i64>> {
    strides
        .iter()
        .map(|&stride| {
            i64::try_from(stride).map_err(|_| {
                crate::Error::backend_failure(
                    op,
                    format!("view stride {stride} exceeds CubeCL i64 metadata limit"),
                )
            })
        })
        .collect()
}

fn view_offset_i64(offset: isize, op: &'static str) -> crate::Result<i64> {
    i64::try_from(offset).map_err(|_| {
        crate::Error::backend_failure(
            op,
            format!("view offset {offset} exceeds CubeCL i64 metadata limit"),
        )
    })
}

fn scatter_update_len(meta: &ScatterLaunchMeta) -> crate::Result<usize> {
    let batch_len = checked_dim_product("scatter", "batch shape", &meta.batch_shape)?;
    let window_len =
        checked_dim_product("scatter", "window update shape", &meta.window_shape_updates)?;
    batch_len.checked_mul(window_len).ok_or_else(|| {
        crate::Error::backend_failure(
            "scatter",
            format!(
                "scatter update domain product overflow for batch {:?} and window {:?}",
                meta.batch_shape, meta.window_shape_updates
            ),
        )
    })
}

/// CubeCL-based GPU backend.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::CudaBackend;
///
/// let _ctor: fn(usize) -> tenferro_tensor::Result<CudaBackend> = CudaBackend::new;
/// ```
pub struct CudaBackend {
    // CUDA library handles are dropped before `rt`; Rust drops fields in
    // declaration order, so cache-owned handles release while the CUDA primary
    // context is still retained by `CudaRuntime`.
    cutensor: OnceCell<crate::Result<ffi::cutensor::CutensorHandle>>,
    extension_cache: CudaExtensionCache,
    rt: CudaRuntime,
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend")
            .field("runtime", &self.rt)
            .field("cuda_extension_cache", &self.extension_cache)
            .field("cutensor_initialized", &self.cutensor.get().is_some())
            .finish_non_exhaustive()
    }
}

/// Type-indexed cache for CUDA extension-owned backend state.
#[doc(hidden)]
pub struct CudaExtensionCache {
    inner: Mutex<CudaExtensionCacheInner>,
}

impl fmt::Debug for CudaExtensionCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaExtensionCache")
            .field("max_entries", &self.max_entries())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

const DEFAULT_CUDA_EXTENSION_CACHE_MAX_ENTRIES: usize = 16;

struct CudaExtensionCacheEntry {
    value: Box<dyn Any + Send>,
    retained_bytes: usize,
}

struct CudaExtensionCacheInner {
    max_entries: NonZeroUsize,
    entries: HashMap<TypeId, CudaExtensionCacheEntry>,
    order: VecDeque<TypeId>,
    retained_bytes: usize,
}

impl CudaExtensionCacheInner {
    fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            max_entries,
            entries: HashMap::new(),
            order: VecDeque::new(),
            retained_bytes: 0,
        }
    }

    fn evict_to_limit(&mut self) {
        while self.entries.len() > self.max_entries.get() {
            let Some(type_id) = self.order.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&type_id) {
                self.retained_bytes = self.retained_bytes.saturating_sub(entry.retained_bytes);
            }
        }
    }

    fn insert<T: Send + 'static>(&mut self, type_id: TypeId, value: T, retained_bytes: usize) {
        self.entries.insert(
            type_id,
            CudaExtensionCacheEntry {
                value: Box::new(value),
                retained_bytes,
            },
        );
        self.order.retain(|&existing| existing != type_id);
        self.order.push_back(type_id);
        self.retained_bytes = self
            .entries
            .values()
            .map(|entry| entry.retained_bytes)
            .sum();
        self.evict_to_limit();
    }
}

impl CudaExtensionCache {
    fn poisoned_lock_error() -> crate::Error {
        crate::Error::backend_failure("cuda_extension_cache", "extension cache lock poisoned")
    }

    fn lock_inner(&self) -> crate::Result<MutexGuard<'_, CudaExtensionCacheInner>> {
        self.inner.lock().map_err(|_| Self::poisoned_lock_error())
    }

    /// Create an empty extension cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaExtensionCache;
    ///
    /// let cache = CudaExtensionCache::new();
    /// assert!(cache.is_empty());
    /// ```
    pub fn new() -> Self {
        let max_entries = NonZeroUsize::new(DEFAULT_CUDA_EXTENSION_CACHE_MAX_ENTRIES)
            .unwrap_or(NonZeroUsize::MIN);
        Self::with_max_entries(max_entries)
    }

    /// Create an empty extension cache with an explicit entry bound.
    pub fn with_max_entries(max_entries: NonZeroUsize) -> Self {
        Self {
            inner: Mutex::new(CudaExtensionCacheInner::new(max_entries)),
        }
    }

    /// Returns `true` when no extension state has been initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaExtensionCache;
    ///
    /// assert!(CudaExtensionCache::new().is_empty()?);
    /// # Ok::<(), tenferro_gpu::Error>(())
    /// ```
    pub fn is_empty(&self) -> crate::Result<bool> {
        Ok(self.lock_inner()?.entries.is_empty())
    }

    /// Remove every cached CUDA extension state value.
    pub fn clear(&self) -> crate::Result<()> {
        let mut inner = self.lock_inner()?;
        inner.entries.clear();
        inner.order.clear();
        inner.retained_bytes = 0;
        Ok(())
    }

    /// Snapshot the number of retained entries and logical retained bytes.
    pub fn stats(&self) -> crate::Result<CacheStats> {
        let inner = self.lock_inner()?;
        Ok(CacheStats {
            entries: inner.entries.len(),
            retained_bytes: inner.retained_bytes,
        })
    }

    /// Return the configured entry bound.
    pub fn max_entries(&self) -> crate::Result<NonZeroUsize> {
        Ok(self.lock_inner()?.max_entries)
    }

    /// Replace the entry bound and evict oldest entries if needed.
    pub fn set_max_entries(&self, max_entries: NonZeroUsize) -> crate::Result<()> {
        let mut inner = self.lock_inner()?;
        inner.max_entries = max_entries;
        inner.evict_to_limit();
        Ok(())
    }

    /// Get or lazily initialize one cache entry keyed by `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaExtensionCache;
    ///
    /// let cache = CudaExtensionCache::new();
    /// let value = cache.get_or_try_init::<usize>(|| Ok(3)).unwrap();
    /// assert_eq!(*value, 3);
    /// ```
    pub fn get_or_try_init<T>(
        &self,
        init: impl FnOnce() -> crate::Result<T>,
    ) -> crate::Result<CudaExtensionCacheGuard<'_, T>>
    where
        T: Send + 'static,
    {
        let type_id = TypeId::of::<T>();
        let mut inner = self.lock_inner()?;
        if !inner.entries.contains_key(&type_id) {
            inner.insert(type_id, init()?, std::mem::size_of::<T>());
        }
        let value = inner
            .entries
            .get(&type_id)
            .and_then(|entry| entry.value.downcast_ref::<T>())
            .map(NonNull::from)
            .ok_or_else(|| {
                crate::Error::backend_failure(
                    "cuda_extension_cache",
                    format!(
                        "stored entry for {} is missing or has the wrong type",
                        std::any::type_name::<T>()
                    ),
                )
            })?;
        Ok(CudaExtensionCacheGuard {
            inner,
            type_id,
            value,
            _marker: std::marker::PhantomData,
        })
    }
}

impl Default for CudaExtensionCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrow guard for one cached CUDA extension state value.
#[doc(hidden)]
pub struct CudaExtensionCacheGuard<'a, T> {
    inner: MutexGuard<'a, CudaExtensionCacheInner>,
    type_id: TypeId,
    value: NonNull<T>,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<T: 'static> fmt::Debug for CudaExtensionCacheGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let retained_bytes = self
            .inner
            .entries
            .get(&self.type_id)
            .map(|entry| entry.retained_bytes)
            .unwrap_or(0);
        f.debug_struct("CudaExtensionCacheGuard")
            .field("value_type", &std::any::type_name::<T>())
            .field("retained_bytes", &retained_bytes)
            .finish_non_exhaustive()
    }
}

impl<T: 'static> Deref for CudaExtensionCacheGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: get_or_try_init validates the downcast while holding this
        // same mutex guard. The entry cannot move or be evicted while this
        // guard owns the mutex.
        unsafe { self.value.as_ref() }
    }
}

impl CudaBackend {
    /// Create a new CubeCL backend for the given CUDA device ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaBackend;
    ///
    /// let _ctor: fn(usize) -> tenferro_tensor::Result<CudaBackend> = CudaBackend::new;
    /// ```
    pub fn new(device_ordinal: usize) -> crate::Result<Self> {
        Ok(Self {
            cutensor: OnceCell::new(),
            extension_cache: CudaExtensionCache::new(),
            rt: CudaRuntime::new(device_ordinal)?,
        })
    }

    /// Borrow the underlying CubeCL runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{CudaBackend, CudaRuntime};
    ///
    /// let _runtime: fn(&CudaBackend) -> &CudaRuntime = CudaBackend::runtime;
    /// ```
    pub fn runtime(&self) -> &CudaRuntime {
        &self.rt
    }

    fn cutensor_handle(&self) -> crate::Result<&ffi::cutensor::CutensorHandle> {
        match self
            .cutensor
            .get_or_init(ffi::cutensor::CutensorHandle::load)
        {
            Ok(handle) => Ok(handle),
            Err(err) => Err(err.clone()),
        }
    }

    #[doc(hidden)]
    pub fn cuda_extension_cache(&self) -> &CudaExtensionCache {
        &self.extension_cache
    }

    /// Clear CUDA extension-owned backend state.
    pub fn clear_cuda_extension_cache(&self) -> crate::Result<()> {
        self.extension_cache.clear()
    }

    /// Return CUDA extension cache stats.
    pub fn cuda_extension_cache_stats(&self) -> crate::Result<CacheStats> {
        self.extension_cache.stats()
    }

    /// Return the CUDA extension cache entry bound.
    pub fn cuda_extension_cache_max_entries(&self) -> crate::Result<NonZeroUsize> {
        self.extension_cache.max_entries()
    }

    /// Configure the CUDA extension cache entry bound.
    pub fn set_cuda_extension_cache_max_entries(
        &self,
        max_entries: NonZeroUsize,
    ) -> crate::Result<()> {
        self.extension_cache.set_max_entries(max_entries)
    }

    fn transpose_typed<T>(
        &self,
        input: &TypedTensor<T>,
        perm: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        validate_permutation("transpose", perm, input.shape().len())?;
        let output_shape: Vec<usize> = perm.iter().map(|&axis| input.shape()[axis]).collect();
        launch_unary_tensor(
            self.runtime(),
            input,
            &output_shape,
            "transpose",
            |client, count, dim, out, input_arg| unsafe {
                structural::transpose_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(perm),
                );
            },
        )
    }

    fn broadcast_typed<T>(
        &self,
        input: &TypedTensor<T>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        validate_broadcast_in_dim(input.shape(), shape, dims)?;
        launch_unary_tensor(
            self.runtime(),
            input,
            shape,
            "broadcast_in_dim",
            |client, count, dim, out, input_arg| unsafe {
                structural::broadcast_in_dim_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(dims),
                    shape.len(),
                );
            },
        )
    }

    fn reverse_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        ensure_axes_unique("reverse", "axes", axes, input.shape().len())?;
        launch_unary_tensor(
            self.runtime(),
            input,
            input.shape(),
            "reverse",
            |client, count, dim, out, input_arg| unsafe {
                structural::reverse_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(axes),
                    input.shape().len(),
                );
            },
        )
    }

    fn alloc_ranked_output<T, R>(
        &self,
        shape: &[usize],
        op: &'static str,
    ) -> crate::Result<TypedTensor<T, R>>
    where
        T: CubeElement + Clone + Send + Sync + 'static,
        R: TensorRank,
    {
        let len = checked_dim_product(op, "output shape", shape)?;
        let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
            crate::Error::backend_failure(
                op,
                format!("CubeCL output byte length overflow for shape {shape:?}"),
            )
        })?;
        let handle = self.runtime().client().empty(bytes);
        let shape = R::shape_from_vec(shape.to_vec().into()).map_err(|err| {
            crate::Error::InvalidConfig {
                op,
                message: format!("output rank mismatch: {err}"),
            }
        })?;
        Ok(TypedTensor::from_buffer_col_major(
            shape,
            Buffer::Backend(Arc::new(crate::CubeclBuffer::new(handle, len))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: self.runtime().device_ordinal(),
                }),
            },
        )?)
    }

    fn to_contiguous_view_typed<T, R>(
        &self,
        view: &TypedTensorView<'_, T, R>,
        op: &'static str,
    ) -> crate::Result<TypedTensor<T, R>>
    where
        T: CubeElement + CubePrimitive + Clone + Send + Sync + 'static,
        R: TensorRank,
    {
        ensure_view_resident_on_runtime(self.runtime(), view, op)?;
        let output = self.alloc_ranked_output::<T, R>(view.shape(), op)?;
        let len = output.n_elements();
        if len == 0 {
            return Ok(output);
        }
        let strides = view_strides_i64(view.strides(), op)?;
        let base_offset = view_offset_i64(view.offset(), op)?;
        let output_arg = typed_tensor_binding(&output, op)?;
        let input_arg = typed_view_array_arg(view, op)?;
        let rank = view.shape().len();
        unsafe {
            // SAFETY: The view constructor validated reachable offsets against
            // the backing allocation, and `ensure_view_resident_on_runtime`
            // proves this is a CubeCL buffer on this CUDA runtime. The launch
            // domain covers every logical output element exactly once.
            structural::view_to_contiguous_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                self.runtime().client(),
                cube_count_for_len(len)?,
                cube_dim_1d(),
                output_arg.into_tensor_arg(),
                input_arg,
                comptime_sequence(&strides),
                base_offset,
                rank,
            );
        }
        Ok(output)
    }

    fn copy_contiguous_to_view_typed<T, R>(
        &self,
        src: &TypedTensor<T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
        op: &'static str,
    ) -> crate::Result<()>
    where
        T: CubeElement + CubePrimitive + Clone + Send + Sync + 'static,
        R: TensorRank,
    {
        ensure_resident_on_runtime(self.runtime(), src, op)?;
        ensure_view_mut_resident_on_runtime(self.runtime(), dst, op)?;
        if src.shape() != dst.shape() {
            return Err(crate::Error::InvalidConfig {
                op,
                message: format!(
                    "shape mismatch: source {:?} does not match destination {:?}",
                    src.shape(),
                    dst.shape()
                ),
            });
        }
        let len = src.n_elements();
        if len == 0 {
            return Ok(());
        }
        let strides = view_strides_i64(dst.strides(), op)?;
        let base_offset = view_offset_i64(dst.offset(), op)?;
        let src_arg = typed_tensor_binding(src, op)?;
        let dst_arg = typed_view_mut_array_arg(dst, op)?;
        let rank = dst.shape().len();
        unsafe {
            // SAFETY: The source is an owned compact CubeCL tensor on this
            // runtime. The destination view has validated reachable offsets
            // and no overlap, and the launch domain covers each source element
            // and destination logical coordinate exactly once.
            structural::contiguous_to_view_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                self.runtime().client(),
                cube_count_for_len(len)?,
                cube_dim_1d(),
                dst_arg,
                src_arg.into_tensor_arg(),
                comptime_sequence(&strides),
                base_offset,
                rank,
            );
        }
        Ok(())
    }

    fn convert_float_to_float<In, Out>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + CubeFloat + Clone,
        Out: CubeElement + CubeFloat + Clone,
    {
        launch_unary(
            self.runtime(),
            input,
            input.shape(),
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_float_to_float::launch_unchecked::<Out, In, CubeclCudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_f32_to_c32(
        &self,
        input: &TypedTensor<f32>,
    ) -> crate::Result<TypedTensor<Complex32>> {
        self.convert_float_to_complex_raw::<f32, Complex32, f32>(input, |client, out, input, n| {
            unsafe {
                // SAFETY: `convert_float_to_complex_raw` validated that
                // `input` has `n` elements and `out` has `2 * n` scalar
                // components. The kernel launches exactly `n` logical input
                // positions and guards with `ABSOLUTE_POS < input.len()`.
                structural::convert_f32_to_c32_raw::launch_unchecked::<CubeclCudaRuntime>(
                    client,
                    cube_count_for_len(n)?,
                    cube_dim_1d(),
                    out,
                    input,
                );
            }
            Ok(())
        })
    }

    fn convert_f32_to_c64(
        &self,
        input: &TypedTensor<f32>,
    ) -> crate::Result<TypedTensor<Complex64>> {
        self.convert_float_to_complex_raw::<f32, Complex64, f64>(input, |client, out, input, n| {
            unsafe {
                // SAFETY: `convert_float_to_complex_raw` validated that
                // `input` has `n` elements and `out` has `2 * n` scalar
                // components. The kernel launches exactly `n` logical input
                // positions and guards with `ABSOLUTE_POS < input.len()`.
                structural::convert_f32_to_c64_raw::launch_unchecked::<CubeclCudaRuntime>(
                    client,
                    cube_count_for_len(n)?,
                    cube_dim_1d(),
                    out,
                    input,
                );
            }
            Ok(())
        })
    }

    fn convert_f64_to_c32(
        &self,
        input: &TypedTensor<f64>,
    ) -> crate::Result<TypedTensor<Complex32>> {
        self.convert_float_to_complex_raw::<f64, Complex32, f32>(input, |client, out, input, n| {
            unsafe {
                // SAFETY: `convert_float_to_complex_raw` validated that
                // `input` has `n` elements and `out` has `2 * n` scalar
                // components. The kernel launches exactly `n` logical input
                // positions and guards with `ABSOLUTE_POS < input.len()`.
                structural::convert_f64_to_c32_raw::launch_unchecked::<CubeclCudaRuntime>(
                    client,
                    cube_count_for_len(n)?,
                    cube_dim_1d(),
                    out,
                    input,
                );
            }
            Ok(())
        })
    }

    fn convert_f64_to_c64(
        &self,
        input: &TypedTensor<f64>,
    ) -> crate::Result<TypedTensor<Complex64>> {
        self.convert_float_to_complex_raw::<f64, Complex64, f64>(input, |client, out, input, n| {
            unsafe {
                // SAFETY: `convert_float_to_complex_raw` validated that
                // `input` has `n` elements and `out` has `2 * n` scalar
                // components. The kernel launches exactly `n` logical input
                // positions and guards with `ABSOLUTE_POS < input.len()`.
                structural::convert_f64_to_c64_raw::launch_unchecked::<CubeclCudaRuntime>(
                    client,
                    cube_count_for_len(n)?,
                    cube_dim_1d(),
                    out,
                    input,
                );
            }
            Ok(())
        })
    }

    /// Generic float-to-complex conversion via raw interleaved kernel.
    ///
    /// The kernel writes `(re, 0, re, 0, ...)` into a raw float buffer that
    /// is then reinterpreted as complex.
    fn convert_float_to_complex_raw<InFloat, OutComplex, OutFloat>(
        &self,
        input: &TypedTensor<InFloat>,
        launch: impl FnOnce(
            &cubecl::client::ComputeClient<CubeclCudaRuntime>,
            ArrayArg<CubeclCudaRuntime>,
            ArrayArg<CubeclCudaRuntime>,
            usize,
        ) -> crate::Result<()>,
    ) -> crate::Result<TypedTensor<OutComplex>>
    where
        InFloat: CubeElement + Clone,
        OutComplex: CubeElement + Clone,
        OutFloat: CubeElement + Clone,
    {
        let n = input.n_elements();
        let output = alloc_output::<OutComplex>(self.runtime(), input.shape())?;
        if n == 0 {
            return Ok(output);
        }
        let output_part_len = n.checked_mul(2).ok_or_else(|| {
            crate::Error::backend_failure("convert", "complex output part length overflow")
        })?;
        let output_parts =
            typed_tensor_array_arg_as::<OutComplex, OutFloat>(&output, output_part_len, "convert")?;
        let input_arg = typed_tensor_array_arg(input, "convert")?;
        // SAFETY: The checked raw-array helpers prove that `input_arg` covers
        // exactly the dense input shape and `output_parts` covers the complete
        // real/imaginary scalar representation of the output allocation.
        launch(self.runtime().client(), output_parts, input_arg, n)?;
        Ok(output)
    }

    fn convert_c32_to_f32(
        &self,
        input: &TypedTensor<Complex32>,
    ) -> crate::Result<TypedTensor<f32>> {
        launch_unary(
            self.runtime(),
            input,
            input.shape(),
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_c32_to_f32::launch_unchecked::<CubeclCudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_c32_to_f64(
        &self,
        input: &TypedTensor<Complex32>,
    ) -> crate::Result<TypedTensor<f64>> {
        launch_unary(
            self.runtime(),
            input,
            input.shape(),
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_c32_to_f64::launch_unchecked::<CubeclCudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_c64_to_f32(
        &self,
        input: &TypedTensor<Complex64>,
    ) -> crate::Result<TypedTensor<f32>> {
        launch_unary(
            self.runtime(),
            input,
            input.shape(),
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_c64_to_f32::launch_unchecked::<CubeclCudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_c64_to_f64(
        &self,
        input: &TypedTensor<Complex64>,
    ) -> crate::Result<TypedTensor<f64>> {
        launch_unary(
            self.runtime(),
            input,
            input.shape(),
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_c64_to_f64::launch_unchecked::<CubeclCudaRuntime>(
                    client, count, dim, out, input_arg,
                );
            },
        )
    }

    fn convert_complex_to_complex<In, Out>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + CubeComplex + Clone,
        Out: CubeElement + CubeComplex + Clone,
    {
        launch_unary(
            self.runtime(),
            input,
            input.shape(),
            "convert",
            |client, count, dim, out, input_arg| unsafe {
                structural::convert_complex_to_complex::launch_unchecked::<
                    Out,
                    In,
                    CubeclCudaRuntime,
                >(client, count, dim, out, input_arg);
            },
        )
    }

    fn extract_diagonal_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let (output_shape, diag_output_axis) =
            extract_diagonal_shape(input.shape(), axis_a, axis_b)?;
        launch_unary_tensor(
            self.runtime(),
            input,
            &output_shape,
            "extract_diagonal",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::extract_diagonal_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    axis_a,
                    axis_b,
                    diag_output_axis,
                    input.shape().len(),
                    output_shape.len(),
                );
            },
        )
    }

    fn embed_diagonal_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let output_shape = embed_diagonal_shape(input.shape(), axis_a, axis_b)?;
        let output = alloc_output::<T>(self.runtime(), &output_shape)?;
        launch_nullary_into(
            self.runtime(),
            &output,
            "embed_diagonal",
            cube_count_for_len(output.n_elements())?,
            cube_dim_1d(),
            |client, count, dim, out| unsafe {
                structural::fill_zero_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client, count, dim, out,
                );
            },
        )?;
        launch_unary_tensor_into(
            self.runtime(),
            &output,
            input,
            "embed_diagonal",
            cube_count_for_len(input.n_elements())?,
            cube_dim_1d(),
            |client, count, dim, out, input_arg| unsafe {
                diagonal::embed_diagonal_copy_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    axis_a,
                    axis_b,
                    input.shape().len(),
                    output_shape.len(),
                );
            },
        )?;
        Ok(output)
    }

    #[doc(hidden)]
    pub fn tril_typed<T>(&self, input: &TypedTensor<T>, k: i64) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        if input.shape().len() < 2 {
            return Err(crate::Error::RankMismatch {
                op: "tril",
                expected: 2,
                actual: input.shape().len(),
            });
        }
        launch_unary_tensor(
            self.runtime(),
            input,
            input.shape(),
            "tril",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::tril_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    k,
                );
            },
        )
    }

    #[doc(hidden)]
    pub fn triu_typed<T>(&self, input: &TypedTensor<T>, k: i64) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        if input.shape().len() < 2 {
            return Err(crate::Error::RankMismatch {
                op: "triu",
                expected: 2,
                actual: input.shape().len(),
            });
        }
        launch_unary_tensor(
            self.runtime(),
            input,
            input.shape(),
            "triu",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::triu_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    k,
                );
            },
        )
    }

    fn launch_reduce_axis_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axis: usize,
        op: &'static str,
        launch: impl FnOnce(
            &ComputeClient<CubeclCudaRuntime>,
            TensorBinding<CubeclCudaRuntime>,
            TensorBinding<CubeclCudaRuntime>,
        ) -> crate::kernels::Result<()>,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + Clone,
    {
        let output_shape = reduction_keepdims_shape(input.shape(), axis);
        let output = alloc_output::<T>(self.runtime(), &output_shape)?;
        if output.n_elements() == 0 {
            return Ok(output);
        }

        let input_binding = typed_tensor_binding(input, op)?;
        let output_binding = typed_tensor_binding(&output, op)?;
        launch(self.runtime().client(), input_binding, output_binding)
            .map_err(|err| crate::Error::backend_failure(op, err.to_string()))?;
        Ok(output)
    }

    fn reduce_axes_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axes: &[usize],
        op: &'static str,
        mut launch_axis: impl FnMut(&Self, &TypedTensor<T>, usize) -> crate::Result<TypedTensor<T>>,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + Clone,
    {
        ensure_axes_unique(op, "axes", axes, input.shape().len())?;
        if axes.is_empty() {
            return Ok(input.clone());
        }

        let final_shape = reduction_output_shape(input.shape(), axes);
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();

        let mut current = input.clone();
        for axis in sorted_axes {
            current = launch_axis(self, &current, axis)?;
        }

        cubecl_reshape_metadata(current, final_shape, op)
    }

    fn reduce_sum_float_typed<F: CubeElement + CubeFloat + Clone>(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        let op = op_name(
            PrimitiveOpKind::ReduceSum,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_sum_float::<CubeclCudaRuntime, F>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    fn reduce_sum_complex_typed<C: CubeElement + CubeComplex + Clone>(
        &self,
        input: &TypedTensor<C>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<C>> {
        let op = op_name(
            PrimitiveOpKind::ReduceSum,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_sum_complex::<CubeclCudaRuntime, C>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    fn reduce_sum_int_typed<I: CubeElement + CubeInt + Clone>(
        &self,
        input: &TypedTensor<I>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<I>> {
        let op = op_name(
            PrimitiveOpKind::ReduceSum,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_sum_int::<CubeclCudaRuntime, I>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    fn reduce_prod_float_typed<F: CubeElement + CubeFloat + Clone>(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        let op = op_name(
            PrimitiveOpKind::ReduceProd,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_prod_float::<CubeclCudaRuntime, F>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    fn reduce_prod_complex_typed<C: CubeElement + CubeComplex + Clone>(
        &self,
        input: &TypedTensor<C>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<C>> {
        let op = op_name(
            PrimitiveOpKind::ReduceProd,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_prod_complex::<CubeclCudaRuntime, C>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    fn reduce_prod_int_typed<I: CubeElement + CubeInt + Clone>(
        &self,
        input: &TypedTensor<I>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<I>> {
        let op = op_name(
            PrimitiveOpKind::ReduceProd,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_prod_int::<CubeclCudaRuntime, I>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    fn reduce_max_typed<F: CubeElement + CubeFloat + Clone>(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        let op = op_name(
            PrimitiveOpKind::ReduceMax,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_max_float::<CubeclCudaRuntime, F>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    fn reduce_min_typed<F: CubeElement + CubeFloat + Clone>(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        let op = op_name(
            PrimitiveOpKind::ReduceMin,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_min_float::<CubeclCudaRuntime, F>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    #[doc(hidden)]
    pub fn slice_typed<T>(
        &self,
        input: &TypedTensor<T>,
        config: &SliceConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let output_shape = validate_slice(input.shape(), config)?;
        launch_unary_tensor(
            self.runtime(),
            input,
            &output_shape,
            "slice",
            |client, count, dim, out, input_arg| unsafe {
                indexing::slice_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(&config.starts),
                    comptime_sequence(&config.strides),
                );
            },
        )
    }

    fn dynamic_slice_typed<T, I>(
        &self,
        input: &TypedTensor<T>,
        starts: &TypedTensor<I>,
        slice_sizes: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
        I: CubeElement + CubePrimitive + CubeNumeric + Clone,
    {
        ensure_rank("dynamic_slice", input.shape().len(), slice_sizes.len())?;
        ensure_rank("dynamic_slice", 1, starts.shape().len())?;
        if starts.shape()[0] != input.shape().len() {
            return Err(crate::Error::RankMismatch {
                op: "dynamic_slice",
                expected: input.shape().len(),
                actual: starts.shape()[0],
            });
        }
        for (axis, (&window, &dim)) in slice_sizes.iter().zip(input.shape()).enumerate() {
            if window > dim {
                return Err(crate::Error::InvalidConfig {
                    op: "dynamic_slice",
                    message: format!("slice size exceeds dimension on axis {axis}"),
                });
            }
        }
        launch_binary_tensor(
            self.runtime(),
            input,
            starts,
            slice_sizes,
            "dynamic_slice",
            |client, count, dim, out, input_arg, starts_arg| unsafe {
                indexing::dynamic_slice_kernel::launch_unchecked::<T, I, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    starts_arg.into_tensor_arg(),
                    comptime_sequence(slice_sizes),
                );
            },
        )
    }

    fn pad_typed<T>(
        &self,
        input: &TypedTensor<T>,
        config: &PadConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let output_shape = pad_output_shape(input.shape(), config)?;
        launch_unary_tensor(
            self.runtime(),
            input,
            &output_shape,
            "pad",
            |client, count, dim, out, input_arg| unsafe {
                indexing::pad_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    input_arg.into_tensor_arg(),
                    comptime_sequence(&config.edge_padding_low),
                    comptime_sequence(&config.interior_padding),
                );
            },
        )
    }

    fn concatenate_typed<T>(
        &self,
        inputs: &[&TypedTensor<T>],
        axis: usize,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
    {
        let output_shape = concatenate_output_shape(inputs, axis)?;
        let output = alloc_output::<T>(self.runtime(), &output_shape)?;
        let mut offset = 0usize;
        for input in inputs {
            launch_unary_tensor_into(
                self.runtime(),
                &output,
                input,
                "concatenate",
                cube_count_for_len(input.n_elements())?,
                cube_dim_1d(),
                |client, count, dim, out, input_arg| unsafe {
                    structural::concatenate_copy_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                        client,
                        count,
                        dim,
                        out.into_tensor_arg(),
                        input_arg.into_tensor_arg(),
                        axis,
                        offset,
                        input.shape().len(),
                    );
                },
            )?;
            // INVARIANT: `concatenate_output_shape(inputs, axis)?` above checks
            // the total axis extent, so every partial offset stays bounded.
            offset += input.shape()[axis];
        }
        Ok(output)
    }

    fn gather_typed<T, I>(
        &self,
        operand: &TypedTensor<T>,
        start_indices: &TypedTensor<I>,
        config: &GatherConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + Clone,
        I: CubeElement + CubePrimitive + CubeNumeric + Clone,
    {
        let meta = gather_launch_meta(operand.shape(), start_indices.shape(), config)?;
        launch_binary_tensor(
            self.runtime(),
            operand,
            start_indices,
            &meta.output_shape,
            "gather",
            |client, count, dim, out, operand_arg, indices_arg| unsafe {
                indexing::gather_kernel::launch_unchecked::<T, I, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_tensor_arg(),
                    operand_arg.into_tensor_arg(),
                    indices_arg.into_tensor_arg(),
                    comptime_sequence(&meta.window_dims),
                    comptime_sequence(&config.offset_dims),
                    comptime_sequence(&config.start_index_map),
                    comptime_sequence(&config.slice_sizes),
                    config.index_vector_dim,
                    operand.shape().len(),
                    meta.output_shape.len(),
                    start_indices.shape().len(),
                );
            },
        )
    }

    fn scatter_float_typed<T, I>(
        &self,
        operand: &TypedTensor<T>,
        scatter_indices: &TypedTensor<I>,
        updates: &TypedTensor<T>,
        config: &ScatterConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubeFloat + Clone,
        I: CubeElement + CubePrimitive + CubeNumeric + Clone,
    {
        let meta = scatter_launch_meta(
            operand.shape(),
            scatter_indices.shape(),
            updates.shape(),
            config,
        )?;
        let output = alloc_output::<T>(self.runtime(), operand.shape())?;
        if output.n_elements() == 0 {
            return Ok(output);
        }

        launch_unary_tensor_into(
            self.runtime(),
            &output,
            operand,
            "scatter",
            cube_count_for_len(output.n_elements())?,
            cube_dim_1d(),
            |client, count, dim, out_arg, operand_arg| unsafe {
                indexing::scatter_copy_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out_arg.into_tensor_arg(),
                    operand_arg.into_tensor_arg(),
                );
            },
        )?;

        let update_len = scatter_update_len(&meta)?;
        if update_len == 0 {
            return Ok(output);
        }
        let client = self.runtime().client();
        ensure_resident_on_runtime(self.runtime(), scatter_indices, "scatter")?;
        ensure_resident_on_runtime(self.runtime(), updates, "scatter")?;
        ensure_atomic_add_supported::<T>(client, "scatter")?;
        let output_parts =
            typed_tensor_array_arg_as::<T, T>(&output, output.n_elements(), "scatter")?;
        let operand_arg = typed_tensor_binding(operand, "scatter")?;
        let scatter_arg = typed_tensor_binding(scatter_indices, "scatter")?;
        let updates_arg = typed_tensor_binding(updates, "scatter")?;
        unsafe {
            // SAFETY: `scatter_launch_meta` validates the scatter/update
            // shapes and dimension-number mappings. `typed_tensor_binding`
            // validates input logical tensor buffers, while
            // `typed_tensor_array_arg_as` proves the atomic output view stays
            // within its backing allocation. The launch domain is
            // `scatter_update_len(meta)`, and the kernel maps each launched
            // update through the validated metadata before indexing.
            indexing::scatter_float_kernel::launch_unchecked::<T, I, CubeclCudaRuntime>(
                client,
                cube_count_for_len(update_len)?,
                cube_dim_1d(),
                output_parts,
                operand_arg.into_tensor_arg(),
                scatter_arg.into_tensor_arg(),
                updates_arg.into_tensor_arg(),
                comptime_sequence(&meta.window_dims),
                comptime_sequence(&config.update_window_dims),
                comptime_sequence(&config.scatter_dims_to_operand_dims),
                config.index_vector_dim,
                operand.shape().len(),
                updates.shape().len(),
                scatter_indices.shape().len(),
            );
        }
        Ok(output)
    }

    fn scatter_complex_typed<T, F, I>(
        &self,
        operand: &TypedTensor<T>,
        scatter_indices: &TypedTensor<I>,
        updates: &TypedTensor<T>,
        config: &ScatterConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubeComplex + Clone,
        F: CubeElement + CubeFloat + Clone,
        I: CubeElement + CubePrimitive + CubeNumeric + Clone,
    {
        let meta = scatter_launch_meta(
            operand.shape(),
            scatter_indices.shape(),
            updates.shape(),
            config,
        )?;
        let output = alloc_output::<T>(self.runtime(), operand.shape())?;
        if output.n_elements() == 0 {
            return Ok(output);
        }

        launch_unary_tensor_into(
            self.runtime(),
            &output,
            operand,
            "scatter",
            cube_count_for_len(output.n_elements())?,
            cube_dim_1d(),
            |client, count, dim, out_arg, operand_arg| unsafe {
                indexing::scatter_copy_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out_arg.into_tensor_arg(),
                    operand_arg.into_tensor_arg(),
                );
            },
        )?;

        let update_len = scatter_update_len(&meta)?;
        if update_len == 0 {
            return Ok(output);
        }
        let client = self.runtime().client();
        ensure_resident_on_runtime(self.runtime(), scatter_indices, "scatter")?;
        ensure_resident_on_runtime(self.runtime(), updates, "scatter")?;
        ensure_atomic_add_supported::<F>(client, "scatter")?;
        let output_part_len = output.n_elements().checked_mul(2).ok_or_else(|| {
            crate::Error::backend_failure("scatter", "complex output part length overflow")
        })?;
        let update_part_len = updates.n_elements().checked_mul(2).ok_or_else(|| {
            crate::Error::backend_failure("scatter", "complex update part length overflow")
        })?;
        // num_complex::Complex<T> is repr(C) as { re: T, im: T }, so the
        // complex buffers can be viewed as real scalar parts for atomic add.
        let output_parts = typed_tensor_array_arg_as::<T, F>(&output, output_part_len, "scatter")?;
        let update_parts = typed_tensor_array_arg_as::<T, F>(updates, update_part_len, "scatter")?;
        let operand_arg = typed_tensor_binding(operand, "scatter")?;
        let scatter_arg = typed_tensor_binding(scatter_indices, "scatter")?;
        let updates_arg = typed_tensor_binding(updates, "scatter")?;
        unsafe {
            // SAFETY: `scatter_launch_meta` validates the scatter/update
            // shapes and dimension-number mappings. `typed_tensor_binding`
            // validates logical tensor buffers, while `typed_tensor_array_arg_as`
            // proves complex real/imaginary part arrays stay within their
            // backing allocations. The launch domain is
            // `scatter_update_len(meta)` and the kernel indexes via the
            // validated metadata.
            indexing::scatter_complex_kernel::launch_unchecked::<T, F, I, CubeclCudaRuntime>(
                client,
                cube_count_for_len(update_len)?,
                cube_dim_1d(),
                output_parts,
                operand_arg.into_tensor_arg(),
                scatter_arg.into_tensor_arg(),
                updates_arg.into_tensor_arg(),
                update_parts,
                comptime_sequence(&meta.window_dims),
                comptime_sequence(&config.update_window_dims),
                comptime_sequence(&config.scatter_dims_to_operand_dims),
                config.index_vector_dim,
                operand.shape().len(),
                updates.shape().len(),
                scatter_indices.shape().len(),
            );
        }
        Ok(output)
    }
}

impl BackendRuntimeCache for CudaBackend {
    type RuntimeCache = ();
}

impl TensorElementwise for CudaBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_binary_float_complex!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Add,
            add_float,
            add_complex
        )
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_binary_float_complex!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Mul,
            mul_float,
            mul_complex
        )
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_complex!(
            self,
            input,
            PrimitiveOpKind::Neg,
            neg_float,
            neg_complex
        )
    }

    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Conj,
            op_descriptor::GpuLaunchKind::UnaryFloatComplex,
        )?;
        match input {
            Tensor::F32(tensor) => {
                ensure_resident_on_runtime(self.runtime(), tensor, op)?;
                Ok(Tensor::F32(tensor.clone()))
            }
            Tensor::F64(tensor) => {
                ensure_resident_on_runtime(self.runtime(), tensor, op)?;
                Ok(Tensor::F64(tensor.clone()))
            }
            Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
                Err(unsupported_dtype(op, input.dtype()))
            }
            Tensor::C32(tensor) => launch_unary(
                self.runtime(),
                tensor,
                tensor.shape(),
                op,
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::conj_complex::launch_unchecked::<Complex32, CubeclCudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::C32),
            Tensor::C64(tensor) => launch_unary(
                self.runtime(),
                tensor,
                tensor.shape(),
                op,
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::conj_complex::launch_unchecked::<Complex64, CubeclCudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::C64),
        }
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_binary_float_complex!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Div,
            div_float,
            div_complex
        )
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Abs, abs_float)
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Sign, sign_float)
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_binary_float_only!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Maximum,
            maximum_float
        )
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_binary_float_only!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Minimum,
            minimum_float
        )
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Compare,
            op_descriptor::GpuLaunchKind::CompareFloatToBool,
        )?;
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_compare_bool(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::compare_float_bool::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client,
                        count,
                        dim,
                        out,
                        lhs_arg,
                        rhs_arg,
                        dispatch::compare_mode(dir),
                    );
                },
            )
            .map(Tensor::Bool),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_compare_bool(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::compare_float_bool::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client,
                        count,
                        dim,
                        out,
                        lhs_arg,
                        rhs_arg,
                        dispatch::compare_mode(dir),
                    );
                },
            )
            .map(Tensor::Bool),
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => Err(
                crate::Error::backend_failure(op, format!("unsupported dtype {:?}", lhs.dtype())),
            ),
            _ => Err(dtype_mismatch(op, lhs, rhs)),
        }
    }

    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Select,
            op_descriptor::GpuLaunchKind::SelectBoolFloat,
        )?;
        match (pred, on_true, on_false) {
            (Tensor::Bool(pred), Tensor::F32(on_true), Tensor::F32(on_false)) => {
                launch_select_bool(
                    self.runtime(),
                    pred,
                    on_true,
                    on_false,
                    pred.shape(),
                    op,
                    |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                        elementwise::select_bool_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                            client, count, dim, out, pred_arg, true_arg, false_arg,
                        );
                    },
                )
                .map(Tensor::F32)
            }
            (Tensor::Bool(pred), Tensor::F64(on_true), Tensor::F64(on_false)) => {
                launch_select_bool(
                    self.runtime(),
                    pred,
                    on_true,
                    on_false,
                    pred.shape(),
                    op,
                    |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                        elementwise::select_bool_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                            client, count, dim, out, pred_arg, true_arg, false_arg,
                        );
                    },
                )
                .map(Tensor::F64)
            }
            (Tensor::C32(_), Tensor::C32(_), Tensor::C32(_))
            | (Tensor::C64(_), Tensor::C64(_), Tensor::C64(_)) => Err(
                crate::Error::backend_failure(op, format!("unsupported dtype {:?}", pred.dtype())),
            ),
            _ => Err(ternary_dtype_mismatch(op, pred, on_true, on_false)),
        }
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Clamp,
            op_descriptor::GpuLaunchKind::ClampFloat,
        )?;
        match (input, lower, upper) {
            (Tensor::F32(input), Tensor::F32(lower), Tensor::F32(upper)) => launch_ternary(
                self.runtime(),
                input,
                lower,
                upper,
                input.shape(),
                op,
                |client, count, dim, out, input_arg, lower_arg, upper_arg| unsafe {
                    elementwise::clamp_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, input_arg, lower_arg, upper_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(input), Tensor::F64(lower), Tensor::F64(upper)) => launch_ternary(
                self.runtime(),
                input,
                lower,
                upper,
                input.shape(),
                op,
                |client, count, dim, out, input_arg, lower_arg, upper_arg| unsafe {
                    elementwise::clamp_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, input_arg, lower_arg, upper_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::C32(_), Tensor::C32(_), Tensor::C32(_))
            | (Tensor::C64(_), Tensor::C64(_), Tensor::C64(_)) => Err(
                crate::Error::backend_failure(op, format!("unsupported dtype {:?}", input.dtype())),
            ),
            _ => Err(ternary_dtype_mismatch(op, input, lower, upper)),
        }
    }
}

impl TensorAnalytic for CudaBackend {
    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Exp, exp_float)
    }

    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Log, log_float)
    }

    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Sin, sin_float)
    }

    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Cos, cos_float)
    }

    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Tanh, tanh_float)
    }

    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Sqrt, sqrt_float)
    }

    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Rsqrt, rsqrt_float)
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_binary_float_only!(self, lhs, rhs, PrimitiveOpKind::Pow, pow_float)
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Expm1, expm1_float)
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Log1p, log1p_float)
    }
}

impl TensorStructural for CudaBackend {
    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.transpose_typed(t, perm).map(Tensor::F32),
            Tensor::F64(t) => self.transpose_typed(t, perm).map(Tensor::F64),
            Tensor::I32(t) => self.transpose_typed(t, perm).map(Tensor::I32),
            Tensor::I64(t) => self.transpose_typed(t, perm).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("transpose", input.dtype())),
            Tensor::C32(t) => self.transpose_typed(t, perm).map(Tensor::C32),
            Tensor::C64(t) => self.transpose_typed(t, perm).map(Tensor::C64),
        }
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        let old_n = checked_dim_product("reshape", "input shape", input.shape())?;
        let new_n = checked_dim_product("reshape", "output shape", shape)?;
        if old_n != new_n {
            return Err(crate::Error::ShapeMismatch {
                op: "reshape",
                lhs: input.shape().to_vec(),
                rhs: shape.to_vec(),
            });
        }
        match input {
            Tensor::F32(t) => Ok(Tensor::F32(TypedTensor::from_buffer_col_major(
                shape.to_vec(),
                t.buffer().clone(),
                t.placement().clone(),
            )?)),
            Tensor::F64(t) => Ok(Tensor::F64(TypedTensor::from_buffer_col_major(
                shape.to_vec(),
                t.buffer().clone(),
                t.placement().clone(),
            )?)),
            Tensor::I32(t) => Ok(Tensor::I32(TypedTensor::from_buffer_col_major(
                shape.to_vec(),
                t.buffer().clone(),
                t.placement().clone(),
            )?)),
            Tensor::I64(t) => Ok(Tensor::I64(TypedTensor::from_buffer_col_major(
                shape.to_vec(),
                t.buffer().clone(),
                t.placement().clone(),
            )?)),
            Tensor::Bool(t) => Ok(Tensor::Bool(TypedTensor::from_buffer_col_major(
                shape.to_vec(),
                t.buffer().clone(),
                t.placement().clone(),
            )?)),
            Tensor::C32(t) => Ok(Tensor::C32(TypedTensor::from_buffer_col_major(
                shape.to_vec(),
                t.buffer().clone(),
                t.placement().clone(),
            )?)),
            Tensor::C64(t) => Ok(Tensor::C64(TypedTensor::from_buffer_col_major(
                shape.to_vec(),
                t.buffer().clone(),
                t.placement().clone(),
            )?)),
        }
    }

    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.broadcast_typed(t, shape, dims).map(Tensor::F32),
            Tensor::F64(t) => self.broadcast_typed(t, shape, dims).map(Tensor::F64),
            Tensor::I32(t) => self.broadcast_typed(t, shape, dims).map(Tensor::I32),
            Tensor::I64(t) => self.broadcast_typed(t, shape, dims).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("broadcast_in_dim", input.dtype())),
            Tensor::C32(t) => self.broadcast_typed(t, shape, dims).map(Tensor::C32),
            Tensor::C64(t) => self.broadcast_typed(t, shape, dims).map(Tensor::C64),
        }
    }

    fn cast(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        match (input, to) {
            (Tensor::F32(t), crate::DType::F32) => Ok(Tensor::F32(t.clone())),
            (Tensor::F32(t), crate::DType::F64) => {
                self.convert_float_to_float::<f32, f64>(t).map(Tensor::F64)
            }
            (Tensor::F32(_), crate::DType::I32 | crate::DType::Bool) => {
                Err(unsupported_dtype("cast", to))
            }
            (Tensor::F32(_), crate::DType::I64) => Err(unsupported_dtype("cast", to)),
            (Tensor::F32(t), crate::DType::C32) => self.convert_f32_to_c32(t).map(Tensor::C32),
            (Tensor::F32(t), crate::DType::C64) => self.convert_f32_to_c64(t).map(Tensor::C64),
            (Tensor::F64(t), crate::DType::F32) => {
                self.convert_float_to_float::<f64, f32>(t).map(Tensor::F32)
            }
            (Tensor::F64(t), crate::DType::F64) => Ok(Tensor::F64(t.clone())),
            (Tensor::F64(_), crate::DType::I32 | crate::DType::Bool) => {
                Err(unsupported_dtype("cast", to))
            }
            (Tensor::F64(_), crate::DType::I64) => Err(unsupported_dtype("cast", to)),
            (Tensor::F64(t), crate::DType::C32) => self.convert_f64_to_c32(t).map(Tensor::C32),
            (Tensor::F64(t), crate::DType::C64) => self.convert_f64_to_c64(t).map(Tensor::C64),
            (Tensor::I32(t), crate::DType::I32) => Ok(Tensor::I32(t.clone())),
            (Tensor::I32(_), _) => Err(unsupported_dtype("cast", input.dtype())),
            (Tensor::I64(_), crate::DType::I64) => Ok(input.clone()),
            (Tensor::I64(_), _) => Err(unsupported_dtype("cast", input.dtype())),
            (Tensor::Bool(t), crate::DType::Bool) => Ok(Tensor::Bool(t.clone())),
            (Tensor::Bool(_), _) => Err(unsupported_dtype("cast", input.dtype())),
            (Tensor::C32(t), crate::DType::F32) => self.convert_c32_to_f32(t).map(Tensor::F32),
            (Tensor::C32(t), crate::DType::F64) => self.convert_c32_to_f64(t).map(Tensor::F64),
            (Tensor::C32(_), crate::DType::I32 | crate::DType::Bool) => {
                Err(unsupported_dtype("cast", to))
            }
            (Tensor::C32(_), crate::DType::I64) => Err(unsupported_dtype("cast", to)),
            (Tensor::C32(t), crate::DType::C32) => Ok(Tensor::C32(t.clone())),
            (Tensor::C32(t), crate::DType::C64) => self
                .convert_complex_to_complex::<Complex32, Complex64>(t)
                .map(Tensor::C64),
            (Tensor::C64(t), crate::DType::F32) => self.convert_c64_to_f32(t).map(Tensor::F32),
            (Tensor::C64(t), crate::DType::F64) => self.convert_c64_to_f64(t).map(Tensor::F64),
            (Tensor::C64(_), crate::DType::I32 | crate::DType::Bool) => {
                Err(unsupported_dtype("cast", to))
            }
            (Tensor::C64(_), crate::DType::I64) => Err(unsupported_dtype("cast", to)),
            (Tensor::C64(t), crate::DType::C32) => self
                .convert_complex_to_complex::<Complex64, Complex32>(t)
                .map(Tensor::C32),
            (Tensor::C64(t), crate::DType::C64) => Ok(Tensor::C64(t.clone())),
        }
    }

    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::F32),
            Tensor::F64(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::F64),
            Tensor::I32(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::I32),
            Tensor::I64(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("extract_diagonal", input.dtype())),
            Tensor::C32(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::C32),
            Tensor::C64(t) => self
                .extract_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::C64),
        }
    }

    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::F32),
            Tensor::F64(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::F64),
            Tensor::I32(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::I32),
            Tensor::I64(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("embed_diagonal", input.dtype())),
            Tensor::C32(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::C32),
            Tensor::C64(t) => self
                .embed_diagonal_typed(t, axis_a, axis_b)
                .map(Tensor::C64),
        }
    }

    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.tril_typed(t, k).map(Tensor::F32),
            Tensor::F64(t) => self.tril_typed(t, k).map(Tensor::F64),
            Tensor::I32(t) => self.tril_typed(t, k).map(Tensor::I32),
            Tensor::I64(t) => self.tril_typed(t, k).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("tril", input.dtype())),
            Tensor::C32(t) => self.tril_typed(t, k).map(Tensor::C32),
            Tensor::C64(t) => self.tril_typed(t, k).map(Tensor::C64),
        }
    }

    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.triu_typed(t, k).map(Tensor::F32),
            Tensor::F64(t) => self.triu_typed(t, k).map(Tensor::F64),
            Tensor::I32(t) => self.triu_typed(t, k).map(Tensor::I32),
            Tensor::I64(t) => self.triu_typed(t, k).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("triu", input.dtype())),
            Tensor::C32(t) => self.triu_typed(t, k).map(Tensor::C32),
            Tensor::C64(t) => self.triu_typed(t, k).map(Tensor::C64),
        }
    }
}

impl TensorReduction for CudaBackend {
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceSum,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input {
            Tensor::F32(t) => self.reduce_sum_float_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_sum_float_typed(t, axes).map(Tensor::F64),
            Tensor::I32(t) => self.reduce_sum_int_typed(t, axes).map(Tensor::I32),
            Tensor::I64(t) => self.reduce_sum_int_typed(t, axes).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype(op, input.dtype())),
            Tensor::C32(t) => self.reduce_sum_complex_typed(t, axes).map(Tensor::C32),
            Tensor::C64(t) => self.reduce_sum_complex_typed(t, axes).map(Tensor::C64),
        }
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceProd,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input {
            Tensor::F32(t) => self.reduce_prod_float_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_prod_float_typed(t, axes).map(Tensor::F64),
            Tensor::I32(t) => self.reduce_prod_int_typed(t, axes).map(Tensor::I32),
            Tensor::I64(t) => self.reduce_prod_int_typed(t, axes).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype(op, input.dtype())),
            Tensor::C32(t) => self.reduce_prod_complex_typed(t, axes).map(Tensor::C32),
            Tensor::C64(t) => self.reduce_prod_complex_typed(t, axes).map(Tensor::C64),
        }
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceMax,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input {
            Tensor::F32(t) => self.reduce_max_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_max_typed(t, axes).map(Tensor::F64),
            Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => {
                Err(crate::Error::backend_failure(
                    op,
                    format!("unsupported dtype {:?}", input.dtype()),
                ))
            }
        }
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceMin,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input {
            Tensor::F32(t) => self.reduce_min_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_min_typed(t, axes).map(Tensor::F64),
            Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => {
                Err(crate::Error::backend_failure(
                    op,
                    format!("unsupported dtype {:?}", input.dtype()),
                ))
            }
        }
    }
}

impl TensorDot for CudaBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        gemm::dot_general(self, lhs, rhs, config)
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        gemm::dot_general_with_conj(self, lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

impl TensorIndexing for CudaBackend {
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        match (operand, start_indices) {
            (Tensor::F32(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F32)
            }
            (Tensor::F64(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F64)
            }
            (Tensor::C32(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C32)
            }
            (Tensor::C64(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C64)
            }
            (Tensor::I32(operand), Tensor::F32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::I32)
            }
            (Tensor::F32(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F32)
            }
            (Tensor::F64(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F64)
            }
            (Tensor::C32(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C32)
            }
            (Tensor::C64(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C64)
            }
            (Tensor::I32(operand), Tensor::F64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::I32)
            }
            (Tensor::F32(operand), Tensor::I32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F32)
            }
            (Tensor::F64(operand), Tensor::I32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F64)
            }
            (Tensor::C32(operand), Tensor::I32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C32)
            }
            (Tensor::C64(operand), Tensor::I32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C64)
            }
            (Tensor::I32(operand), Tensor::I32(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::I32)
            }
            (Tensor::F32(operand), Tensor::I64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F32)
            }
            (Tensor::F64(operand), Tensor::I64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::F64)
            }
            (Tensor::C32(operand), Tensor::I64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C32)
            }
            (Tensor::C64(operand), Tensor::I64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::C64)
            }
            (Tensor::I32(operand), Tensor::I64(indices)) => {
                self.gather_typed(operand, indices, config).map(Tensor::I32)
            }
            (_, Tensor::Bool(_)) => Err(unsupported_dtype("gather", start_indices.dtype())),
            (_, Tensor::C32(_) | Tensor::C64(_)) => Err(crate::Error::backend_failure(
                "gather",
                "complex index tensors are not supported",
            )),
            (Tensor::I64(_) | Tensor::Bool(_), _) => {
                Err(unsupported_dtype("gather", operand.dtype()))
            }
        }
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        match (operand, scatter_indices, updates) {
            (Tensor::F32(operand), Tensor::F32(indices), Tensor::F32(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F32),
            (Tensor::F64(operand), Tensor::F32(indices), Tensor::F64(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F64),
            (Tensor::C32(operand), Tensor::F32(indices), Tensor::C32(updates)) => self
                .scatter_complex_typed::<_, f32, _>(operand, indices, updates, config)
                .map(Tensor::C32),
            (Tensor::C64(operand), Tensor::F32(indices), Tensor::C64(updates)) => self
                .scatter_complex_typed::<_, f64, _>(operand, indices, updates, config)
                .map(Tensor::C64),
            (Tensor::F32(operand), Tensor::F64(indices), Tensor::F32(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F32),
            (Tensor::F64(operand), Tensor::F64(indices), Tensor::F64(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F64),
            (Tensor::C32(operand), Tensor::F64(indices), Tensor::C32(updates)) => self
                .scatter_complex_typed::<_, f32, _>(operand, indices, updates, config)
                .map(Tensor::C32),
            (Tensor::C64(operand), Tensor::F64(indices), Tensor::C64(updates)) => self
                .scatter_complex_typed::<_, f64, _>(operand, indices, updates, config)
                .map(Tensor::C64),
            (Tensor::F32(operand), Tensor::I32(indices), Tensor::F32(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F32),
            (Tensor::F64(operand), Tensor::I32(indices), Tensor::F64(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F64),
            (Tensor::C32(operand), Tensor::I32(indices), Tensor::C32(updates)) => self
                .scatter_complex_typed::<_, f32, _>(operand, indices, updates, config)
                .map(Tensor::C32),
            (Tensor::C64(operand), Tensor::I32(indices), Tensor::C64(updates)) => self
                .scatter_complex_typed::<_, f64, _>(operand, indices, updates, config)
                .map(Tensor::C64),
            (Tensor::F32(operand), Tensor::I64(indices), Tensor::F32(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F32),
            (Tensor::F64(operand), Tensor::I64(indices), Tensor::F64(updates)) => self
                .scatter_float_typed(operand, indices, updates, config)
                .map(Tensor::F64),
            (Tensor::C32(operand), Tensor::I64(indices), Tensor::C32(updates)) => self
                .scatter_complex_typed::<_, f32, _>(operand, indices, updates, config)
                .map(Tensor::C32),
            (Tensor::C64(operand), Tensor::I64(indices), Tensor::C64(updates)) => self
                .scatter_complex_typed::<_, f64, _>(operand, indices, updates, config)
                .map(Tensor::C64),
            (_, Tensor::Bool(_), _) => Err(unsupported_dtype("scatter", scatter_indices.dtype())),
            (_, Tensor::C32(_) | Tensor::C64(_), _) => Err(crate::Error::backend_failure(
                "scatter",
                "complex index tensors are not supported",
            )),
            (Tensor::I32(_), _, _) | (Tensor::I64(_), _, _) | (Tensor::Bool(_), _, _) => {
                Err(unsupported_dtype("scatter", operand.dtype()))
            }
            (_, _, _) => Err(ternary_dtype_mismatch(
                "scatter",
                operand,
                scatter_indices,
                updates,
            )),
        }
    }

    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.slice_typed(t, config).map(Tensor::F32),
            Tensor::F64(t) => self.slice_typed(t, config).map(Tensor::F64),
            Tensor::I32(t) => self.slice_typed(t, config).map(Tensor::I32),
            Tensor::I64(t) => self.slice_typed(t, config).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("slice", input.dtype())),
            Tensor::C32(t) => self.slice_typed(t, config).map(Tensor::C32),
            Tensor::C64(t) => self.slice_typed(t, config).map(Tensor::C64),
        }
    }

    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        match (input, starts) {
            (Tensor::F32(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F32),
            (Tensor::F64(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F64),
            (Tensor::C32(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C32),
            (Tensor::C64(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C64),
            (Tensor::I32(input), Tensor::F32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::I32),
            (Tensor::F32(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F32),
            (Tensor::F64(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F64),
            (Tensor::C32(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C32),
            (Tensor::C64(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C64),
            (Tensor::I32(input), Tensor::F64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::I32),
            (Tensor::F32(input), Tensor::I32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F32),
            (Tensor::F64(input), Tensor::I32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F64),
            (Tensor::C32(input), Tensor::I32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C32),
            (Tensor::C64(input), Tensor::I32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C64),
            (Tensor::I32(input), Tensor::I32(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::I32),
            (Tensor::F32(input), Tensor::I64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F32),
            (Tensor::F64(input), Tensor::I64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::F64),
            (Tensor::C32(input), Tensor::I64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C32),
            (Tensor::C64(input), Tensor::I64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::C64),
            (Tensor::I32(input), Tensor::I64(starts)) => self
                .dynamic_slice_typed(input, starts, slice_sizes)
                .map(Tensor::I32),
            (_, Tensor::Bool(_)) => Err(unsupported_dtype("dynamic_slice", starts.dtype())),
            (_, Tensor::C32(_) | Tensor::C64(_)) => Err(crate::Error::backend_failure(
                "dynamic_slice",
                "complex index tensors are not supported",
            )),
            (Tensor::I64(_) | Tensor::Bool(_), _) => {
                Err(unsupported_dtype("dynamic_slice", input.dtype()))
            }
        }
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> crate::Result<Tensor> {
        Err(crate::Error::backend_failure(
            "dynamic_update_slice",
            "dynamic_update_slice is not implemented for the CubeCL backend",
        ))
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.pad_typed(t, config).map(Tensor::F32),
            Tensor::F64(t) => self.pad_typed(t, config).map(Tensor::F64),
            Tensor::I32(t) => self.pad_typed(t, config).map(Tensor::I32),
            Tensor::I64(t) => self.pad_typed(t, config).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("pad", input.dtype())),
            Tensor::C32(t) => self.pad_typed(t, config).map(Tensor::C32),
            Tensor::C64(t) => self.pad_typed(t, config).map(Tensor::C64),
        }
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        let first = inputs
            .first()
            .copied()
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "concatenate",
                message: "concatenate requires at least one input".into(),
            })?;
        match first {
            Tensor::F32(_) => {
                let typed: crate::Result<Vec<&TypedTensor<f32>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::F32(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::F32)
            }
            Tensor::F64(_) => {
                let typed: crate::Result<Vec<&TypedTensor<f64>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::F64(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::F64)
            }
            Tensor::I32(_) => {
                let typed: crate::Result<Vec<&TypedTensor<i32>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::I32(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::I32)
            }
            Tensor::I64(_) => {
                let typed: crate::Result<Vec<&TypedTensor<i64>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::I64(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::I64)
            }
            Tensor::Bool(_) => Err(unsupported_dtype("concatenate", first.dtype())),
            Tensor::C32(_) => {
                let typed: crate::Result<Vec<&TypedTensor<Complex32>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::C32(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::C32)
            }
            Tensor::C64(_) => {
                let typed: crate::Result<Vec<&TypedTensor<Complex64>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::C64(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis).map(Tensor::C64)
            }
        }
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.reverse_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reverse_typed(t, axes).map(Tensor::F64),
            Tensor::I32(t) => self.reverse_typed(t, axes).map(Tensor::I32),
            Tensor::I64(t) => self.reverse_typed(t, axes).map(Tensor::I64),
            Tensor::Bool(_) => Err(unsupported_dtype("reverse", input.dtype())),
            Tensor::C32(t) => self.reverse_typed(t, axes).map(Tensor::C32),
            Tensor::C64(t) => self.reverse_typed(t, axes).map(Tensor::C64),
        }
    }
}

impl TensorDeviceTransfer for CudaBackend {
    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        download_tensor(self.runtime(), tensor)
    }

    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        upload_tensor(self.runtime(), tensor)
    }
}

macro_rules! impl_cubecl_view_canonicalization {
    ($($ty:ty),* $(,)?) => {
        $(
            impl<R> TensorViewCanonicalization<$ty, R> for CudaBackend
            where
                R: TensorRank,
            {
                fn to_contiguous(
                    &mut self,
                    view: &TypedTensorView<'_, $ty, R>,
                ) -> crate::Result<TypedTensor<$ty, R>> {
                    self.to_contiguous_view_typed(view, "CudaBackend::to_contiguous")
                }

                fn copy_from_contiguous(
                    &mut self,
                    src: &TypedTensor<$ty, R>,
                    dst: &mut TypedTensorViewMut<'_, $ty, R>,
                ) -> crate::Result<()> {
                    self.copy_contiguous_to_view_typed(
                        src,
                        dst,
                        "CudaBackend::copy_from_contiguous",
                    )
                }
            }
        )*
    };
}

impl_cubecl_view_canonicalization!(f32, f64, i32, i64, Complex32, Complex64);

impl<R> TensorViewCanonicalization<bool, R> for CudaBackend
where
    R: TensorRank,
{
    fn to_contiguous(
        &mut self,
        _view: &TypedTensorView<'_, bool, R>,
    ) -> crate::Result<TypedTensor<bool, R>> {
        Err(unsupported_dtype(
            "CudaBackend::to_contiguous",
            crate::DType::Bool,
        ))
    }

    fn copy_from_contiguous(
        &mut self,
        _src: &TypedTensor<bool, R>,
        _dst: &mut TypedTensorViewMut<'_, bool, R>,
    ) -> crate::Result<()> {
        Err(unsupported_dtype(
            "CudaBackend::copy_from_contiguous",
            crate::DType::Bool,
        ))
    }
}

impl TensorFusion for CudaBackend {
    fn execute_elementwise_fusion(
        &mut self,
        inputs: &[&Tensor],
        plan: &crate::backend::ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        fusion::execute_elementwise_fusion(self, inputs, plan)
    }
}

impl BackendCachedDot for CudaBackend {}

impl BackendSessionHost for CudaBackend {}

impl TensorBuffer for CudaBackend {}

impl TensorBackend for CudaBackend {}

fn validate_permutation(op: &'static str, perm: &[usize], rank: usize) -> crate::Result<()> {
    ensure_rank(op, rank, perm.len())?;
    ensure_axes_unique(op, "perm", perm, rank)
}

fn validate_broadcast_in_dim(
    input_shape: &[usize],
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<()> {
    ensure_rank("broadcast_in_dim", input_shape.len(), dims.len())?;
    let mut seen = vec![false; shape.len()];
    for (src_axis, &dst_axis) in dims.iter().enumerate() {
        ensure_axis("broadcast_in_dim", dst_axis, shape.len())?;
        if seen[dst_axis] {
            return Err(crate::Error::DuplicateAxis {
                op: "broadcast_in_dim",
                axis: dst_axis,
                role: "dims",
            });
        }
        seen[dst_axis] = true;
        let src = input_shape[src_axis];
        let dst = shape[dst_axis];
        if src != dst && src != 1 {
            return Err(crate::Error::ShapeMismatch {
                op: "broadcast_in_dim",
                lhs: input_shape.to_vec(),
                rhs: shape.to_vec(),
            });
        }
    }
    Ok(())
}

fn extract_diagonal_shape(
    input_shape: &[usize],
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<(Vec<usize>, usize)> {
    ensure_axis("extract_diagonal", axis_a, input_shape.len())?;
    ensure_axis("extract_diagonal", axis_b, input_shape.len())?;
    if axis_a == axis_b {
        return Err(crate::Error::DuplicateAxis {
            op: "extract_diagonal",
            axis: axis_a,
            role: "axes",
        });
    }
    let diag_output_axis = if axis_a < axis_b { axis_a } else { axis_a - 1 };
    let diag_dim = input_shape[axis_a].min(input_shape[axis_b]);
    let mut output_shape = input_shape.to_vec();
    output_shape.remove(axis_b);
    output_shape[diag_output_axis] = diag_dim;
    Ok((output_shape, diag_output_axis))
}

fn embed_diagonal_shape(
    input_shape: &[usize],
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<Vec<usize>> {
    ensure_axis("embed_diagonal", axis_a, input_shape.len())?;
    if axis_b > input_shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op: "embed_diagonal",
            axis: axis_b,
            rank: input_shape.len(),
        });
    }
    let mut output_shape = input_shape.to_vec();
    output_shape.insert(axis_b, input_shape[axis_a]);
    Ok(output_shape)
}

fn reduction_output_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (!axes.contains(&axis)).then_some(dim))
        .collect();
    // cubecl Array::new(0) generates uint32 arr[0] which is invalid CUDA.
    // When all axes are reduced (scalar output), use shape [1] instead.
    if shape.is_empty() {
        vec![1]
    } else {
        shape
    }
}

fn reduction_keepdims_shape(input_shape: &[usize], axis: usize) -> Vec<usize> {
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = 1;
    output_shape
}

fn cubecl_reshape_metadata<T: CubeElement + Clone>(
    tensor: TypedTensor<T>,
    shape: Vec<usize>,
    op: &'static str,
) -> crate::Result<TypedTensor<T>> {
    let len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            crate::Error::backend_failure(
                op,
                format!("shape product overflow for CubeCL reshape shape {shape:?}"),
            )
        })?;
    let tensor_len = tensor.n_elements();
    if len != tensor_len {
        return Err(crate::Error::backend_failure(op, format!(
                "cannot reshape CubeCL output metadata from {:?} ({tensor_len} elements) to {:?} ({len} elements)",
                tensor.shape(), shape
            )));
    }

    let (buffer, _, placement) = tensor.into_parts();
    Ok(TypedTensor::from_buffer_col_major(
        shape, buffer, placement,
    )?)
}

fn validate_slice(input_shape: &[usize], config: &SliceConfig) -> crate::Result<Vec<usize>> {
    let rank = input_shape.len();
    ensure_rank("slice", rank, config.starts.len())?;
    ensure_rank("slice", rank, config.limits.len())?;
    ensure_rank("slice", rank, config.strides.len())?;
    input_shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            let start = config.starts[axis];
            let limit = config.limits[axis];
            let stride = config.strides[axis];
            if start > limit {
                return Err(crate::Error::InvalidConfig {
                    op: "slice",
                    message: format!("start exceeds limit on axis {axis}"),
                });
            }
            if limit > dim {
                return Err(crate::Error::AxisOutOfBounds {
                    op: "slice",
                    axis,
                    rank,
                });
            }
            if stride == 0 {
                return Err(crate::Error::InvalidConfig {
                    op: "slice",
                    message: format!("stride must be positive on axis {axis}"),
                });
            }
            let span = limit - start;
            Ok(span.div_ceil(stride))
        })
        .collect()
}

fn pad_output_shape(input_shape: &[usize], config: &PadConfig) -> crate::Result<Vec<usize>> {
    let rank = input_shape.len();
    ensure_rank("pad", rank, config.edge_padding_low.len())?;
    ensure_rank("pad", rank, config.edge_padding_high.len())?;
    ensure_rank("pad", rank, config.interior_padding.len())?;
    let mut out_shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        if config.interior_padding[axis] < 0 {
            return Err(crate::Error::InvalidConfig {
                op: "pad",
                message: format!("interior padding must be non-negative on axis {axis}"),
            });
        }
        let input_dim =
            i64::try_from(input_shape[axis]).map_err(|_| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("input dimension on axis {axis} must fit in i64"),
            })?;
        let base = if input_dim == 0 {
            0
        } else {
            let spacing = config.interior_padding[axis]
                .checked_add(1)
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op: "pad",
                    message: format!("interior padding overflow on axis {axis}"),
                })?;
            input_dim
                .checked_sub(1)
                .and_then(|extent| extent.checked_mul(spacing))
                .and_then(|extent| extent.checked_add(1))
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op: "pad",
                    message: format!("padded interior extent overflow on axis {axis}"),
                })?
        };
        let dim = config.edge_padding_low[axis]
            .checked_add(config.edge_padding_high[axis])
            .and_then(|edge| edge.checked_add(base))
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("output dimension overflow on axis {axis}"),
            })?;
        out_shape.push(
            usize::try_from(dim).map_err(|_| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("negative output dimension on axis {axis}"),
            })?,
        );
    }
    Ok(out_shape)
}

fn validate_slice_sizes_within_operand(
    op: &'static str,
    operand_shape: &[usize],
    slice_sizes: &[usize],
) -> crate::Result<()> {
    ensure_rank(op, operand_shape.len(), slice_sizes.len())?;
    for (axis, (&slice_size, &dim_size)) in slice_sizes.iter().zip(operand_shape).enumerate() {
        if slice_size > dim_size {
            return Err(crate::Error::InvalidConfig {
                op,
                message: format!(
                    "slice_sizes[{axis}]={slice_size} exceeds operand dimension {dim_size}"
                ),
            });
        }
    }
    Ok(())
}

fn index_vector_size(shape: &[usize], index_vector_dim: usize) -> usize {
    if index_vector_dim == shape.len() {
        1
    } else {
        shape[index_vector_dim]
    }
}

fn index_batch_shape(shape: &[usize], index_vector_dim: usize) -> Vec<usize> {
    if index_vector_dim == shape.len() {
        return shape.to_vec();
    }
    shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (axis != index_vector_dim).then_some(dim))
        .collect()
}

fn operand_window_dims(rank: usize, collapsed_or_inserted: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|dim| !collapsed_or_inserted.contains(dim))
        .collect()
}

#[derive(Debug)]
struct GatherLaunchMeta {
    output_shape: Vec<usize>,
    window_dims: Vec<usize>,
}

fn gather_launch_meta(
    operand_shape: &[usize],
    start_indices_shape: &[usize],
    config: &GatherConfig,
) -> crate::Result<GatherLaunchMeta> {
    ensure_rank("gather", operand_shape.len(), config.slice_sizes.len())?;
    validate_slice_sizes_within_operand("gather", operand_shape, &config.slice_sizes)?;
    if config.index_vector_dim > start_indices_shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op: "gather",
            axis: config.index_vector_dim,
            rank: start_indices_shape.len(),
        });
    }
    let index_size = index_vector_size(start_indices_shape, config.index_vector_dim);
    if index_size != config.start_index_map.len() {
        return Err(crate::Error::InvalidConfig {
            op: "gather",
            message: "start_index_map length mismatch".into(),
        });
    }
    ensure_axes_unique(
        "gather",
        "collapsed_slice_dims",
        &config.collapsed_slice_dims,
        operand_shape.len(),
    )?;
    for &dim in &config.collapsed_slice_dims {
        if config.slice_sizes[dim] != 1 {
            return Err(crate::Error::InvalidConfig {
                op: "gather",
                message: format!(
                    "collapsed slice dimension {dim} must have slice_size == 1, got {}",
                    config.slice_sizes[dim]
                ),
            });
        }
    }
    ensure_axes_unique(
        "gather",
        "start_index_map",
        &config.start_index_map,
        operand_shape.len(),
    )?;
    let window_dims = operand_window_dims(operand_shape.len(), &config.collapsed_slice_dims);
    if config.offset_dims.len() != window_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "gather",
            message: "offset_dims length mismatch".into(),
        });
    }
    let batch_shape = index_batch_shape(start_indices_shape, config.index_vector_dim);
    let out_rank = batch_shape.len() + config.offset_dims.len();
    ensure_axes_unique("gather", "offset_dims", &config.offset_dims, out_rank)?;
    let mut output_shape = vec![0usize; out_rank];
    let mut out_axis_to_operand_dim = vec![None; out_rank];
    for (offset_axis, &out_axis) in config.offset_dims.iter().enumerate() {
        out_axis_to_operand_dim[out_axis] = Some(window_dims[offset_axis]);
    }
    let mut batch_axis = 0usize;
    for out_axis in 0..out_rank {
        if let Some(operand_dim) = out_axis_to_operand_dim[out_axis] {
            output_shape[out_axis] = config.slice_sizes[operand_dim];
        } else {
            output_shape[out_axis] = batch_shape[batch_axis];
            batch_axis += 1;
        }
    }
    Ok(GatherLaunchMeta {
        output_shape,
        window_dims,
    })
}

#[derive(Debug)]
struct ScatterLaunchMeta {
    batch_shape: Vec<usize>,
    window_dims: Vec<usize>,
    window_shape_updates: Vec<usize>,
}

fn scatter_launch_meta(
    operand_shape: &[usize],
    scatter_indices_shape: &[usize],
    updates_shape: &[usize],
    config: &ScatterConfig,
) -> crate::Result<ScatterLaunchMeta> {
    if config.index_vector_dim > scatter_indices_shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op: "scatter",
            axis: config.index_vector_dim,
            rank: scatter_indices_shape.len(),
        });
    }
    let index_size = index_vector_size(scatter_indices_shape, config.index_vector_dim);
    if index_size != config.scatter_dims_to_operand_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: "scatter_dims_to_operand_dims length mismatch".into(),
        });
    }
    ensure_axes_unique(
        "scatter",
        "inserted_window_dims",
        &config.inserted_window_dims,
        operand_shape.len(),
    )?;
    ensure_axes_unique(
        "scatter",
        "scatter_dims_to_operand_dims",
        &config.scatter_dims_to_operand_dims,
        operand_shape.len(),
    )?;
    ensure_axes_unique(
        "scatter",
        "update_window_dims",
        &config.update_window_dims,
        updates_shape.len(),
    )?;
    let batch_shape = index_batch_shape(scatter_indices_shape, config.index_vector_dim);
    let window_dims = operand_window_dims(operand_shape.len(), &config.inserted_window_dims);
    if config.update_window_dims.len() != window_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: "update_window_dims length mismatch".into(),
        });
    }
    if updates_shape.len() - config.update_window_dims.len() != batch_shape.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: "updates batch rank mismatch".into(),
        });
    }
    let mut is_update_window_dim = vec![false; updates_shape.len()];
    for &axis in &config.update_window_dims {
        is_update_window_dim[axis] = true;
    }
    let mut batch_axis = 0usize;
    for (axis, &actual) in updates_shape.iter().enumerate() {
        if is_update_window_dim[axis] {
            continue;
        }
        let expected = batch_shape[batch_axis];
        if actual != expected {
            return Err(crate::Error::InvalidConfig {
                op: "scatter",
                message: format!(
                    "updates batch dim {batch_axis} extent {actual} does not match \
                     scatter batch extent {expected}"
                ),
            });
        }
        batch_axis += 1;
    }
    let window_shape_updates = config
        .update_window_dims
        .iter()
        .map(|&axis| updates_shape[axis])
        .collect();
    Ok(ScatterLaunchMeta {
        batch_shape,
        window_dims,
        window_shape_updates,
    })
}

fn concatenate_output_shape<T>(
    inputs: &[&TypedTensor<T>],
    axis: usize,
) -> crate::Result<Vec<usize>> {
    let first = inputs[0];
    let rank = first.shape().len();
    ensure_axis("concatenate", axis, rank)?;
    let mut out_shape = first.shape().to_vec();
    let mut axis_extent = 0usize;
    for input in inputs {
        ensure_rank("concatenate", rank, input.shape().len())?;
        for dim in 0..rank {
            if dim == axis {
                axis_extent = axis_extent.checked_add(input.shape()[dim]).ok_or_else(|| {
                    crate::Error::backend_failure(
                        "concatenate",
                        "concatenate axis extent overflows usize",
                    )
                })?;
            } else if input.shape()[dim] != first.shape()[dim] {
                return Err(crate::Error::ShapeMismatch {
                    op: "concatenate",
                    lhs: first.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                });
            }
        }
    }
    out_shape[axis] = axis_extent;
    Ok(out_shape)
}

#[cfg(test)]
mod tests;
