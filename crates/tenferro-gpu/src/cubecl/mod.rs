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
//! use tenferro_gpu::{cuda::cuda_devices, cuda::CudaBackend, cuda::CudaDeviceError};
//!
//! fn first_cuda_backend() -> Result<Option<CudaBackend>, CudaDeviceError> {
//!     let devices = cuda_devices()?;
//!     let Some(device) = devices.first() else {
//!         return Ok(None);
//!     };
//!     Ok(Some(CudaBackend::new(device.id())?))
//! }
//!
//! let _example: fn() -> Result<Option<CudaBackend>, CudaDeviceError> = first_cuda_backend;
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
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use cubecl::client::ComputeClient;
use cubecl::features::AtomicUsage;
use cubecl::prelude::{
    ArrayArg, ComplexCore as CubeComplex, CubeDim, CubeElement, CubePrimitive, Float as CubeFloat,
    Numeric as CubeNumeric,
};
use cubecl::prelude::{CubeCount, Int as CubeInt, StorageType, TensorBinding, Type};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use num_complex::{Complex32, Complex64};
use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::CacheStats;
use tenferro_tensor::{DotGeneralAccumulation, TensorRead, TensorWrite};

use crate::backend::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, TensorAnalytic, TensorBackend,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::kernels::reduce::{self as cubecl_reduce, ReduceStrategy};
use crate::kernels::{diagonal, elementwise, indexing, structural};
use crate::native_permutation::{
    NativePermutationKind, NativePermutationPlan, NativeTransposeTile,
};
use crate::{
    DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, StorageBuffer, Tensor, TensorRank,
    TensorScalar, TensorView, TensorViewCanonicalization, TensorViewMut, TypedTensor,
    TypedTensorView, TypedTensorViewMut,
};

mod capability;
mod device;
pub(crate) mod dispatch;
mod error;
mod event_domain;
mod exec_session;
mod ffi;
mod fusion;
mod gemm;
mod identity;
pub(crate) mod interop;
mod memory;
pub(crate) mod op_descriptor;
mod permutation;
pub(crate) mod raw;
mod runtime;
mod runtime_adapter;
pub(crate) mod session_cubecl;

use dispatch::{
    alloc_bool_output, alloc_output, bool_tensor_array_arg, comptime_sequence, cube_count_for_len,
    cube_dim_1d, dtype_mismatch, ensure_axes_unique, ensure_axis, ensure_rank,
    ensure_resident_on_runtime, ensure_view_mut_resident_on_runtime,
    ensure_view_resident_on_runtime, launch_binary, launch_binary_bool_tensor,
    launch_binary_tensor, launch_bool_tensor_into, launch_compare_bool, launch_nullary_bool_into,
    launch_nullary_into, launch_select_bool, launch_ternary, launch_unary,
    launch_unary_bool_tensor, launch_unary_tensor, launch_unary_tensor_into,
    ternary_dtype_mismatch, typed_tensor_array_arg, typed_tensor_array_arg_as,
    typed_tensor_binding, typed_view_array_arg, typed_view_binding, typed_view_mut_array_arg,
};
use error::{unsupported_dtype, unsupported_operation};

pub use capability::cuda_capabilities;
pub use device::{cuda_devices, CudaDeviceError, CudaDeviceId, CudaDeviceInfo};
#[doc(hidden)]
pub use exec_session::{with_cuda_exec_session, CudaExecSession};
pub use identity::{CudaComputeCapability, CudaDeviceUuid, GpuExtensionCapability};
pub use memory::{download_tensor, upload_tensor};
pub use runtime::{gpu_available, CudaRuntime, CudaRuntimeIdentity};
pub use runtime_adapter::{cuda_runtime_engine_registration, cuda_runtime_hardware_class};

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
        Err(unsupported_operation(
            op,
            "CubeCL runtime does not support atomic add",
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
            crate::Error::invalid_argument(
                op,
                role,
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
                crate::Error::invalid_argument(
                    op,
                    "layout",
                    format!("view stride {stride} exceeds CubeCL i64 metadata limit"),
                )
            })
        })
        .collect()
}

fn view_offset_i64(offset: isize, op: &'static str) -> crate::Result<i64> {
    i64::try_from(offset).map_err(|_| {
        crate::Error::invalid_argument(
            op,
            "layout",
            format!("view offset {offset} exceeds CubeCL i64 metadata limit"),
        )
    })
}

fn launch_native_materialization<E: CubePrimitive>(
    backend: &CudaBackend,
    output: ArrayArg<CubeclCudaRuntime>,
    input: ArrayArg<CubeclCudaRuntime>,
    plan: &NativePermutationPlan,
    op: &'static str,
) -> crate::Result<()> {
    if plan.len == 0 {
        return Ok(());
    }
    if plan.kind == NativePermutationKind::TiledTranspose {
        if let Some(config) = NativeTransposeTile::selected(op)? {
            let block_rows = config.block_rows as usize;
            let padding = config.padding as usize;
            let vector_width = config.vector_width as usize;
            let src_offset = usize::try_from(plan.src_offset).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "offset",
                    "tiled transpose requires a non-negative source offset",
                )
            })?;
            if let Some((cubes_x, cubes_y, cubes_z)) =
                config.dispatch_grid(op, &plan.dims, 65_535)?
            {
                let batch_stride = plan.tiled_matrix_len(op)?;
                unsafe {
                    // SAFETY: The tiled classification proves a compact 2D
                    // transpose. Bounds guards cover edge tiles and every unit
                    // reaches the shared-memory barrier.
                    structural::tiled_transpose_kernel::launch_unchecked::<E, CubeclCudaRuntime>(
                        backend.runtime().client(),
                        CubeCount::Static(cubes_x, cubes_y, cubes_z),
                        CubeDim::new_2d(config.tile / config.vector_width, config.block_rows),
                        output,
                        input,
                        src_offset,
                        batch_stride,
                        plan.dims[0],
                        plan.dims[1],
                        config.tile as usize,
                        block_rows,
                        padding,
                        vector_width,
                    );
                }
                return Ok(());
            }
        }
    }
    let src_strides = view_strides_i64(&plan.src_strides, op)?;
    let src_offset = view_offset_i64(plan.src_offset, op)?;
    unsafe {
        // SAFETY: `NativePermutationPlan` validated both allocation ranges,
        // destination non-overlap, and disjoint source/destination storage.
        structural::materialize_strided_kernel::launch_unchecked::<E, CubeclCudaRuntime>(
            backend.runtime().client(),
            cube_count_for_len(plan.len)?,
            cube_dim_1d(),
            output,
            input,
            comptime_sequence(&plan.dims),
            comptime_sequence(&src_strides),
            src_offset,
            plan.len,
            plan.dims.len(),
        );
    }
    Ok(())
}

fn scatter_update_len(meta: &ScatterLaunchMeta) -> crate::Result<usize> {
    let batch_len = checked_dim_product("scatter", "batch shape", &meta.batch_shape)?;
    let window_len =
        checked_dim_product("scatter", "window update shape", &meta.window_shape_updates)?;
    batch_len.checked_mul(window_len).ok_or_else(|| {
        crate::Error::invalid_argument(
            "scatter",
            "shape",
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
/// use tenferro_gpu::{cuda::CudaBackend, cuda::CudaDeviceError, cuda::CudaDeviceId};
///
/// let _ctor: fn(CudaDeviceId) -> Result<CudaBackend, CudaDeviceError> = CudaBackend::new;
/// ```
#[doc(hidden)]
struct CudaBackendSessionMarker;

#[derive(Clone)]
pub struct CudaBackend {
    inner: Arc<CudaBackendState>,
}

struct CudaBackendState {
    // CUDA library handles are dropped before `rt`; Rust drops fields in
    // declaration order, so cache-owned handles release while the CUDA primary
    // context is still retained by `CudaRuntime`.
    cutensor: OnceLock<ffi::cutensor::CutensorHandle>,
    extension_cache: CudaExtensionCache,
    rt: CudaRuntime,
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend")
            .field("runtime", &self.inner.rt)
            .field("cuda_extension_cache", &self.inner.extension_cache)
            .field("cutensor_initialized", &self.inner.cutensor.get().is_some())
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
const DEFAULT_CUDA_EXTENSION_CACHE_RETAINED_BYTES: usize = 64 * 1024 * 1024;

struct CudaExtensionCacheEntry {
    value: Box<dyn Any + Send>,
    retained_bytes: usize,
}

struct CudaExtensionCacheInner {
    max_entries: NonZeroUsize,
    max_retained_bytes: NonZeroUsize,
    entries: HashMap<TypeId, CudaExtensionCacheEntry>,
    order: VecDeque<TypeId>,
    retained_bytes: usize,
    stats: CacheStats,
}

impl CudaExtensionCacheInner {
    fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            max_entries,
            max_retained_bytes: NonZeroUsize::new(DEFAULT_CUDA_EXTENSION_CACHE_RETAINED_BYTES)
                .unwrap_or(NonZeroUsize::MIN),
            entries: HashMap::new(),
            order: VecDeque::new(),
            retained_bytes: 0,
            stats: CacheStats::empty(),
        }
    }

    fn evict_to_limit(&mut self) {
        while self.entries.len() > self.max_entries.get()
            || self.retained_bytes > self.max_retained_bytes.get()
        {
            let Some(type_id) = self.order.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&type_id) {
                self.retained_bytes = self.retained_bytes.saturating_sub(entry.retained_bytes);
                self.stats.evictions = self.stats.evictions.saturating_add(1);
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

    fn snapshot_stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            retained_bytes: self.retained_bytes,
            ..self.stats
        }
    }

    fn refresh_retained_bytes(&mut self) {
        self.retained_bytes = self
            .entries
            .values()
            .map(|entry| entry.retained_bytes)
            .sum();
    }
}

impl CudaExtensionCache {
    fn poisoned_lock_error() -> crate::Error {
        crate::Error::runtime_state("cuda_extension_cache", "extension cache lock poisoned")
    }

    fn lock_inner(&self) -> crate::Result<MutexGuard<'_, CudaExtensionCacheInner>> {
        self.inner.lock().map_err(|_| Self::poisoned_lock_error())
    }

    /// Create an empty extension cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaExtensionCache;
    ///
    /// let cache = CudaExtensionCache::new();
    /// assert!(cache.is_empty()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// The cache retains at most 16 extension states by default. Use
    /// [`Self::with_max_entries`] to choose a different bound. Later cache
    /// operations return [`crate::Error::RuntimeState`] if the cache mutex is
    /// poisoned.
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
    /// use tenferro_gpu::cuda::CudaExtensionCache;
    ///
    /// assert!(CudaExtensionCache::new().is_empty()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn is_empty(&self) -> crate::Result<bool> {
        Ok(self.lock_inner()?.entries.is_empty())
    }

    /// Remove every cached CUDA extension state value.
    ///
    /// This operation returns a runtime-state error if the cache mutex is
    /// poisoned.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn clear(&self) -> crate::Result<()> {
        let mut inner = self.lock_inner()?;
        inner.entries.clear();
        inner.order.clear();
        inner.retained_bytes = 0;
        let clears = inner.stats.clears.saturating_add(1);
        inner.stats = CacheStats {
            clears,
            ..CacheStats::empty()
        };
        Ok(())
    }

    /// Snapshot the number of retained entries and logical retained bytes.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn stats(&self) -> crate::Result<CacheStats> {
        let inner = self.lock_inner()?;
        Ok(inner.snapshot_stats())
    }

    /// Return the configured entry bound.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn max_entries(&self) -> crate::Result<NonZeroUsize> {
        Ok(self.lock_inner()?.max_entries)
    }

    /// Return the configured logical retained-byte bound.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn max_retained_bytes(&self) -> crate::Result<NonZeroUsize> {
        Ok(self.lock_inner()?.max_retained_bytes)
    }

    /// Replace the entry bound and evict oldest entries if needed.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned
    /// while changing the bound.
    pub fn set_max_entries(&self, max_entries: NonZeroUsize) -> crate::Result<()> {
        let mut inner = self.lock_inner()?;
        inner.max_entries = max_entries;
        inner.evict_to_limit();
        Ok(())
    }

    /// Configure the logical retained-byte bound and evict oldest entries if
    /// needed.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned
    /// while changing the bound.
    pub fn set_max_retained_bytes(&self, max_retained_bytes: NonZeroUsize) -> crate::Result<()> {
        let mut inner = self.lock_inner()?;
        inner.max_retained_bytes = max_retained_bytes;
        inner.evict_to_limit();
        Ok(())
    }

    /// Get or lazily initialize one cache entry keyed by `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaExtensionCache;
    ///
    /// let cache = CudaExtensionCache::new();
    /// let value = cache.get_or_try_init::<usize>(|| Ok(3)).unwrap();
    /// assert_eq!(*value, 3);
    /// ```
    /// # Errors
    ///
    /// Propagates the initializer's typed error, returns
    /// [`crate::Error::RuntimeState`] for a poisoned cache or a missing/wrongly
    /// typed entry, and preserves backend errors from initialization.
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
            inner.stats.misses = inner.stats.misses.saturating_add(1);
            inner.insert(type_id, init()?, std::mem::size_of::<T>());
        } else {
            inner.stats.hits = inner.stats.hits.saturating_add(1);
        }
        let value = inner
            .entries
            .get(&type_id)
            .and_then(|entry| entry.value.downcast_ref::<T>())
            .map(NonNull::from)
            .ok_or_else(|| {
                crate::Error::runtime_state(
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

    pub(crate) fn get_cloned<T>(&self) -> crate::Result<Option<T>>
    where
        T: Clone + 'static,
    {
        let inner = self.lock_inner()?;
        inner
            .entries
            .get(&TypeId::of::<T>())
            .map(|entry| {
                entry.value.downcast_ref::<T>().cloned().ok_or_else(|| {
                    crate::Error::runtime_state(
                        "cuda_extension_cache",
                        format!(
                            "stored entry for {} is missing or has the wrong type",
                            std::any::type_name::<T>()
                        ),
                    )
                })
            })
            .transpose()
    }

    /// Update the logical retained-byte estimate for an existing typed entry.
    ///
    /// This supports extension states whose own internal cache grows after the
    /// top-level entry is initialized. If another thread clears or evicts the
    /// typed entry before the update, the update is treated as a no-op.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub(crate) fn update_retained_bytes<T: 'static>(
        &self,
        retained_bytes: usize,
    ) -> crate::Result<()> {
        let type_id = TypeId::of::<T>();
        let mut inner = self.lock_inner()?;
        if let Some(entry) = inner.entries.get_mut(&type_id) {
            entry.retained_bytes = retained_bytes;
            inner.refresh_retained_bytes();
            inner.evict_to_limit();
        }
        Ok(())
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
    fn duplicate_typed<T>(&self, input: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
    {
        // Same-dtype casts are explicit copies. Use the native materialization
        // path so an identity copy preserves NaN payloads instead of routing
        // through cuTENSOR's alpha-scaled permutation operation.
        self.to_contiguous_view_typed(&input.as_view(), "cast")
    }

    fn duplicate_bool(
        &self,
        input: &TypedTensor<bool>,
        op: &'static str,
    ) -> crate::Result<TypedTensor<bool>> {
        launch_unary_bool_tensor(
            self.runtime(),
            input,
            input.shape(),
            op,
            |client, count, dim, out, input_arg| unsafe {
                structural::copy_bool_kernel::launch_unchecked::<CubeclCudaRuntime>(
                    client,
                    count,
                    dim,
                    out.into_array_arg(),
                    input_arg.into_array_arg(),
                );
            },
        )
    }

    /// Create a new CubeCL backend for the caller-selected CUDA device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaBackend, cuda::CudaDeviceError, cuda::CudaDeviceId};
    ///
    /// let _ctor: fn(CudaDeviceId) -> Result<CudaBackend, CudaDeviceError> = CudaBackend::new;
    /// ```
    /// # Errors
    ///
    /// Returns [`CudaDeviceError::Discovery`] when device discovery fails,
    /// [`CudaDeviceError::Unavailable`] when the selected device is not
    /// discovered, or [`CudaDeviceError::Initialization`] when CUDA runtime,
    /// context, or CubeCL client initialization fails.
    pub fn new(device_id: CudaDeviceId) -> Result<Self, CudaDeviceError> {
        Ok(Self {
            inner: Arc::new(CudaBackendState {
                cutensor: OnceLock::new(),
                extension_cache: CudaExtensionCache::new(),
                rt: CudaRuntime::new(device_id)?,
            }),
        })
    }

    /// Borrow the underlying CubeCL runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaBackend, cuda::CudaRuntime};
    ///
    /// let _runtime: fn(&CudaBackend) -> &CudaRuntime = CudaBackend::runtime;
    /// ```
    pub fn runtime(&self) -> &CudaRuntime {
        &self.inner.rt
    }

    /// Return the caller-selected CUDA device identity used by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaBackend, cuda::CudaDeviceId};
    ///
    /// let _device_id: fn(&CudaBackend) -> CudaDeviceId = CudaBackend::device_id;
    /// ```
    pub fn device_id(&self) -> CudaDeviceId {
        self.inner.rt.device_id()
    }

    /// Return the opaque identity of this exact executable backend instance.
    ///
    /// Clones of a backend return the same identity. Independently constructed
    /// backends return different identities even when they target the same
    /// CUDA device ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaBackend;
    ///
    /// let _identity = CudaBackend::runtime_identity;
    /// ```
    pub fn runtime_identity(&self) -> CudaRuntimeIdentity {
        self.inner.rt.runtime_identity()
    }

    fn cutensor_handle(&self) -> crate::Result<&ffi::cutensor::CutensorHandle> {
        if let Some(handle) = self.inner.cutensor.get() {
            return Ok(handle);
        }
        let handle = ffi::cutensor::CutensorHandle::load()?;
        let _ = self.inner.cutensor.set(handle);
        self.inner.cutensor.get().ok_or_else(|| {
            crate::Error::runtime_state(
                "cuda_cutensor",
                "cuTENSOR handle initialization completed without a stored handle",
            )
        })
    }

    #[doc(hidden)]
    pub fn cuda_extension_cache(&self) -> &CudaExtensionCache {
        &self.inner.extension_cache
    }

    /// Clear CUDA extension-owned backend state.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the extension cache mutex is
    /// poisoned.
    pub fn clear_cuda_extension_cache(&self) -> crate::Result<()> {
        self.inner.extension_cache.clear()
    }

    /// Return CUDA extension cache stats.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the extension cache mutex is
    /// poisoned.
    pub fn cuda_extension_cache_stats(&self) -> crate::Result<CacheStats> {
        self.inner.extension_cache.stats()
    }

    /// Return the CUDA extension cache entry bound.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the extension cache mutex is
    /// poisoned.
    pub fn cuda_extension_cache_max_entries(&self) -> crate::Result<NonZeroUsize> {
        self.inner.extension_cache.max_entries()
    }

    /// Return the CUDA extension cache logical retained-byte bound.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the extension cache mutex is
    /// poisoned.
    pub fn cuda_extension_cache_max_retained_bytes(&self) -> crate::Result<NonZeroUsize> {
        self.inner.extension_cache.max_retained_bytes()
    }

    /// Configure the CUDA extension cache entry bound.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the extension cache mutex is
    /// poisoned while changing the bound.
    pub fn set_cuda_extension_cache_max_entries(
        &self,
        max_entries: NonZeroUsize,
    ) -> crate::Result<()> {
        self.inner.extension_cache.set_max_entries(max_entries)
    }

    /// Configure the CUDA extension cache logical retained-byte bound.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the extension cache mutex is
    /// poisoned while changing the bound.
    pub fn set_cuda_extension_cache_max_retained_bytes(
        &self,
        max_retained_bytes: NonZeroUsize,
    ) -> crate::Result<()> {
        self.inner
            .extension_cache
            .set_max_retained_bytes(max_retained_bytes)
    }

    /// Return cuTENSOR contraction plan cache stats.
    ///
    /// The returned entry count is the number of retained cuTENSOR contraction
    /// plans inside the CUDA backend's extension cache entry. Logical retained
    /// bytes include cached cuTENSOR device workspace estimates.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn cutensor_plan_cache_stats(&self) -> crate::Result<CacheStats> {
        gemm::cutensor_plan_cache_stats(self)
    }

    /// Return the cuTENSOR contraction plan entry bound.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn cutensor_plan_cache_max_entries(&self) -> crate::Result<NonZeroUsize> {
        gemm::cutensor_plan_cache_max_entries(self)
    }

    /// Configure the cuTENSOR contraction plan entry bound.
    ///
    /// The cache is initialized if it does not already exist so a setting made
    /// before the first CUDA `dot_general` call is preserved.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn set_cutensor_plan_cache_max_entries(
        &self,
        max_entries: NonZeroUsize,
    ) -> crate::Result<()> {
        gemm::set_cutensor_plan_cache_max_entries(self, max_entries)
    }

    /// Return cuTENSOR structural permutation plan cache stats.
    ///
    /// The returned entry count is the number of retained cuTENSOR permutation
    /// plans inside the CUDA backend's extension cache entry. Logical retained
    /// bytes include cached descriptor and plan state.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn cutensor_permutation_plan_cache_stats(&self) -> crate::Result<CacheStats> {
        permutation::cutensor_permutation_plan_cache_stats(self)
    }

    /// Return the cuTENSOR structural permutation plan entry bound.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn cutensor_permutation_plan_cache_max_entries(&self) -> crate::Result<NonZeroUsize> {
        permutation::cutensor_permutation_plan_cache_max_entries(self)
    }

    /// Configure the cuTENSOR structural permutation plan entry bound.
    ///
    /// The cache is initialized if it does not already exist so a setting made
    /// before the first CUDA structural permutation call is preserved.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn set_cutensor_permutation_plan_cache_max_entries(
        &self,
        max_entries: NonZeroUsize,
    ) -> crate::Result<()> {
        permutation::set_cutensor_permutation_plan_cache_max_entries(self, max_entries)
    }

    fn transpose_typed<T>(
        &self,
        input: &TypedTensor<T>,
        perm: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
    {
        validate_permutation("transpose", perm, input.shape().len())?;
        let output_shape: Vec<usize> = perm.iter().map(|&axis| input.shape()[axis]).collect();
        ensure_resident_on_runtime(self.runtime(), input, "transpose")?;
        let input_strides =
            crate::native_permutation::compact_col_major_strides("transpose", input.shape())?;
        let plan = NativePermutationPlan::for_transpose(
            "transpose",
            input.shape(),
            &input_strides,
            perm,
            0,
            input.n_elements(),
            input.n_elements(),
            false,
        )?;
        let output = alloc_output::<T>(self.runtime(), &output_shape)?;
        let output_arg = typed_tensor_array_arg(&output, "transpose")?;
        let input_arg = typed_tensor_array_arg(input, "transpose")?;
        launch_native_materialization::<T>(self, output_arg, input_arg, &plan, "transpose")?;
        Ok(output)
    }

    fn transpose_bool(
        &self,
        input: &TypedTensor<bool>,
        perm: &[usize],
    ) -> crate::Result<TypedTensor<bool>> {
        validate_permutation("transpose", perm, input.shape().len())?;
        let output_shape: Vec<usize> = perm.iter().map(|&axis| input.shape()[axis]).collect();
        ensure_resident_on_runtime(self.runtime(), input, "transpose")?;
        let input_strides =
            crate::native_permutation::compact_col_major_strides("transpose", input.shape())?;
        let plan = NativePermutationPlan::for_transpose(
            "transpose",
            input.shape(),
            &input_strides,
            perm,
            0,
            input.n_elements(),
            input.n_elements(),
            false,
        )?;
        let output = alloc_bool_output(self.runtime(), &output_shape)?;
        let output_arg = bool_tensor_array_arg(&output, "transpose")?;
        let input_arg = bool_tensor_array_arg(input, "transpose")?;
        launch_native_materialization::<u8>(self, output_arg, input_arg, &plan, "transpose")?;
        Ok(output)
    }

    fn broadcast_typed<T>(
        &self,
        input: &TypedTensor<T>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
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

    fn broadcast_bool(
        &self,
        input: &TypedTensor<bool>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<TypedTensor<bool>> {
        validate_broadcast_in_dim(input.shape(), shape, dims)?;
        launch_unary_bool_tensor(
            self.runtime(),
            input,
            shape,
            "broadcast_in_dim",
            |client, count, dim, out, input_arg| unsafe {
                structural::broadcast_in_dim_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
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

    fn reverse_bool(
        &self,
        input: &TypedTensor<bool>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<bool>> {
        ensure_axes_unique("reverse", "axes", axes, input.shape().len())?;
        launch_unary_bool_tensor(
            self.runtime(),
            input,
            input.shape(),
            "reverse",
            |client, count, dim, out, input_arg| unsafe {
                structural::reverse_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + Clone + Send + Sync + 'static,
        R: TensorRank,
    {
        let len = checked_dim_product(op, "output shape", shape)?;
        let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("CubeCL output byte length overflow for shape {shape:?}"),
            )
        })?;
        let handle = self.runtime().client().empty(bytes);
        let shape = R::shape_from_vec(shape.to_vec().into())
            .map_err(|err| crate::Error::validation(op, err))?;
        TypedTensor::from_buffer_col_major(
            shape,
            StorageBuffer::Backend(Box::new(crate::CubeclBuffer::new(
                handle,
                bytes,
                self.runtime().device_ordinal(),
                self.runtime().allocation_domain_id(),
            ))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: self.runtime().device_ordinal(),
                }),
                cpu_affinity: None,
            },
        )
    }

    fn to_contiguous_view_typed<T, R>(
        &self,
        view: &TypedTensorView<'_, T, R>,
        op: &'static str,
    ) -> crate::Result<TypedTensor<T, R>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
        R: TensorRank,
    {
        ensure_view_resident_on_runtime(self.runtime(), view, op)?;
        let len = checked_dim_product(op, "output shape", view.shape())?;
        let source_allocation_len = view
            .backend_buffer()
            .map(|buffer| buffer.len())
            .ok_or_else(|| {
                crate::Error::runtime_state(op, "expected CUDA backend view, got host view")
            })?;
        let plan = NativePermutationPlan::for_contiguous_output(
            op,
            view.shape(),
            view.strides(),
            view.offset(),
            source_allocation_len,
            len,
            false,
        )?;
        let output = self.alloc_ranked_output::<T, R>(view.shape(), op)?;
        let output_arg = typed_tensor_array_arg(&output, op)?;
        let input_arg = typed_view_array_arg(view, op)?;
        launch_native_materialization::<T>(self, output_arg, input_arg, &plan, op)?;
        Ok(output)
    }

    fn to_contiguous_view_cutensor_or_cubecl<T, R>(
        &self,
        view: &TypedTensorView<'_, T, R>,
        op: &'static str,
    ) -> crate::Result<TypedTensor<T, R>>
    where
        T: permutation::CutensorPermutationScalar,
        R: TensorRank,
    {
        if view.strides().iter().any(|&stride| stride < 0) {
            // cuTENSOR 2.x rejects negative-stride tensor descriptors. This
            // keeps existing CUDA view coverage for a layout the vendor
            // permutation path cannot represent; it is not a missing-library
            // fallback for cuTENSOR-supported descriptors.
            return self.to_contiguous_view_typed(view, op);
        }
        permutation::to_contiguous_view(self, view, op)
    }

    fn copy_view_to_view_typed<T, R>(
        &self,
        src: &TypedTensorView<'_, T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
        op: &'static str,
    ) -> crate::Result<()>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
        R: TensorRank,
    {
        ensure_view_resident_on_runtime(self.runtime(), src, op)?;
        ensure_view_mut_resident_on_runtime(self.runtime(), dst, op)?;
        if src.shape() != dst.shape() {
            return Err(crate::Error::shape_mismatch(
                op,
                src.shape().to_vec(),
                dst.shape().to_vec(),
            ));
        }
        if src.offset() != 0 || !src.is_col_major_contiguous()? {
            return Err(crate::Error::invalid_argument(
                op,
                "source",
                "CUDA copy_into requires a compact source view covering its full allocation; arbitrary-stride source views are unsupported without explicit canonicalization",
            ));
        }
        let source_buffer = src.backend_buffer().ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                "CUDA backend expected a GPU source view; call upload_tensor() first",
            )
        })?;
        let destination_buffer = dst.backend_buffer().ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                "CUDA backend expected a GPU destination view; call upload_tensor() first",
            )
        })?;
        if std::ptr::eq(source_buffer, destination_buffer) {
            return Err(crate::Error::invalid_argument(
                op,
                "source/destination",
                "CUDA copy_into source and destination allocations must not alias",
            ));
        }
        let len = src.n_elements();
        if len == 0 {
            return Ok(());
        }
        let strides = view_strides_i64(dst.strides(), op)?;
        let base_offset = view_offset_i64(dst.offset(), op)?;
        let src_arg = typed_view_binding(src, op)?;
        let dst_arg = typed_view_mut_array_arg(dst, op)?;
        let rank = dst.shape().len();
        unsafe {
            // SAFETY: The source is an owned compact CubeCL tensor on this
            // runtime. Allocation identity validation above proves source and
            // destination do not alias. The destination view has validated
            // reachable offsets and no internal overlap, and the launch domain
            // covers each source element and destination logical coordinate
            // exactly once.
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

    fn copy_view_to_view_cutensor_or_cubecl<T, R>(
        &self,
        src: &TypedTensorView<'_, T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
        op: &'static str,
    ) -> crate::Result<()>
    where
        T: permutation::CutensorPermutationScalar,
        R: TensorRank,
    {
        if dst.strides().iter().any(|&stride| stride < 0) {
            return self.copy_view_to_view_typed(src, dst, op);
        }
        permutation::copy_view_into(self, src, dst, op)
    }

    fn convert_float_to_float<In, Out>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + TensorScalar + CubeFloat + Clone,
        Out: CubeElement + TensorScalar + CubeFloat + Clone,
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

    fn convert_numeric<In, Out>(&self, input: &TypedTensor<In>) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + TensorScalar + CubeNumeric + Clone,
        Out: CubeElement + TensorScalar + CubeNumeric + Clone,
    {
        self.launch_cast_unary(input, |client, count, dim, out, input| unsafe {
            structural::convert_numeric::launch_unchecked::<Out, In, CubeclCudaRuntime>(
                client, count, dim, out, input,
            );
        })
    }

    fn launch_cast_unary<In, Out>(
        &self,
        input: &TypedTensor<In>,
        launch: impl FnOnce(
            &ComputeClient<CubeclCudaRuntime>,
            CubeCount,
            CubeDim,
            ArrayArg<CubeclCudaRuntime>,
            ArrayArg<CubeclCudaRuntime>,
        ),
    ) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + TensorScalar + Clone,
        Out: CubeElement + TensorScalar + Clone,
    {
        ensure_resident_on_runtime(self.runtime(), input, "cast")?;
        let input_arg = typed_tensor_array_arg(input, "cast")?;
        let n = input.n_elements();
        let count = if n == 0 {
            None
        } else {
            Some(cube_count_for_len(n)?)
        };
        let output = alloc_output::<Out>(self.runtime(), input.shape())?;
        let Some(count) = count else {
            return Ok(output);
        };
        let output_arg = typed_tensor_array_arg(&output, "cast")?;
        launch(
            self.runtime().client(),
            count,
            cube_dim_1d(),
            output_arg,
            input_arg,
        );
        Ok(output)
    }

    fn convert_numeric_to_bool<In>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<bool>>
    where
        In: CubeElement + TensorScalar + CubeNumeric + Clone,
    {
        ensure_resident_on_runtime(self.runtime(), input, "cast")?;
        let input_arg = typed_tensor_array_arg(input, "cast")?;
        let n = input.n_elements();
        let count = if n == 0 {
            None
        } else {
            Some(cube_count_for_len(n)?)
        };
        let output = alloc_bool_output(self.runtime(), input.shape())?;
        let Some(count) = count else {
            return Ok(output);
        };
        let output_arg = bool_tensor_array_arg(&output, "cast")?;
        unsafe {
            structural::convert_numeric_to_bool::launch_unchecked::<In, CubeclCudaRuntime>(
                self.runtime().client(),
                count,
                cube_dim_1d(),
                output_arg,
                input_arg,
            );
        }
        Ok(output)
    }

    fn convert_bool_to_numeric<Out>(
        &self,
        input: &TypedTensor<bool>,
    ) -> crate::Result<TypedTensor<Out>>
    where
        Out: CubeElement + TensorScalar + CubeNumeric + Clone,
    {
        ensure_resident_on_runtime(self.runtime(), input, "cast")?;
        let input_arg = bool_tensor_array_arg(input, "cast")?;
        let n = input.n_elements();
        let count = if n == 0 {
            None
        } else {
            Some(cube_count_for_len(n)?)
        };
        let output = alloc_output::<Out>(self.runtime(), input.shape())?;
        let Some(count) = count else {
            return Ok(output);
        };
        let output_arg = typed_tensor_array_arg(&output, "cast")?;
        unsafe {
            structural::convert_bool_to_numeric::launch_unchecked::<Out, CubeclCudaRuntime>(
                self.runtime().client(),
                count,
                cube_dim_1d(),
                output_arg,
                input_arg,
            );
        }
        Ok(output)
    }

    fn convert_numeric_to_complex<In, OutComplex, OutFloat>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<OutComplex>>
    where
        In: CubeElement + TensorScalar + CubeNumeric + Clone,
        OutComplex: CubeElement + TensorScalar + Clone,
        OutFloat: CubeElement + CubeFloat + Clone,
    {
        self.convert_float_to_complex_raw::<In, OutComplex, OutFloat>(
            input,
            |client, out, input, count| {
                unsafe {
                    structural::convert_numeric_to_complex_raw::launch_unchecked::<
                        OutFloat,
                        In,
                        CubeclCudaRuntime,
                    >(client, count, cube_dim_1d(), out, input);
                }
                Ok(())
            },
        )
    }

    fn convert_bool_to_complex<OutComplex, OutFloat>(
        &self,
        input: &TypedTensor<bool>,
    ) -> crate::Result<TypedTensor<OutComplex>>
    where
        OutComplex: CubeElement + TensorScalar + Clone,
        OutFloat: CubeElement + CubeFloat + Clone,
    {
        ensure_resident_on_runtime(self.runtime(), input, "cast")?;
        let n = input.n_elements();
        let part_len = n.checked_mul(2).ok_or_else(|| {
            crate::Error::invalid_argument("cast", "shape", "complex output part length overflow")
        })?;
        let input_arg = bool_tensor_array_arg(input, "cast")?;
        let count = if n == 0 {
            None
        } else {
            Some(cube_count_for_len(n)?)
        };
        let output = alloc_output::<OutComplex>(self.runtime(), input.shape())?;
        let Some(count) = count else {
            return Ok(output);
        };
        let out = typed_tensor_array_arg_as::<OutComplex, OutFloat>(&output, part_len, "cast")?;
        unsafe {
            structural::convert_bool_to_complex_raw::launch_unchecked::<OutFloat, CubeclCudaRuntime>(
                self.runtime().client(),
                count,
                cube_dim_1d(),
                out,
                input_arg,
            );
        }
        Ok(output)
    }

    fn convert_complex_to_numeric<In, Out>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + TensorScalar + CubeComplex + Clone,
        Out: CubeElement + TensorScalar + CubeNumeric + Clone,
    {
        self.launch_cast_unary(input, |client, count, dim, out, input| unsafe {
            structural::convert_complex_to_numeric::launch_unchecked::<Out, In, CubeclCudaRuntime>(
                client, count, dim, out, input,
            );
        })
    }

    fn convert_complex_to_bool<In, F>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<bool>>
    where
        In: CubeElement + TensorScalar + CubeComplex<FloatElem = F> + Clone,
        F: CubeElement + TensorScalar + CubeFloat,
    {
        ensure_resident_on_runtime(self.runtime(), input, "cast")?;
        let part_len = input.n_elements().checked_mul(2).ok_or_else(|| {
            crate::Error::invalid_argument("cast", "shape", "complex input part length overflow")
        })?;
        let input_arg = typed_tensor_array_arg_as::<In, F>(input, part_len, "cast")?;
        let n = input.n_elements();
        let count = if n == 0 {
            None
        } else {
            Some(cube_count_for_len(n)?)
        };
        let output = alloc_bool_output(self.runtime(), input.shape())?;
        let Some(count) = count else {
            return Ok(output);
        };
        let output_arg = bool_tensor_array_arg(&output, "cast")?;
        unsafe {
            structural::convert_complex_raw_to_bool::launch_unchecked::<F, CubeclCudaRuntime>(
                self.runtime().client(),
                count,
                cube_dim_1d(),
                output_arg,
                input_arg,
            );
        }
        Ok(output)
    }

    fn convert_f32_to_c32(
        &self,
        input: &TypedTensor<f32>,
    ) -> crate::Result<TypedTensor<Complex32>> {
        self.convert_float_to_complex_raw::<f32, Complex32, f32>(
            input,
            |client, out, input, count| {
                unsafe {
                    // SAFETY: `convert_float_to_complex_raw` validated that
                    // `input` has `n` elements and `out` has `2 * n` scalar
                    // components. The kernel launches exactly `n` logical input
                    // positions and guards with `ABSOLUTE_POS < input.len()`.
                    structural::convert_f32_to_c32_raw::launch_unchecked::<CubeclCudaRuntime>(
                        client,
                        count,
                        cube_dim_1d(),
                        out,
                        input,
                    );
                }
                Ok(())
            },
        )
    }

    fn convert_f32_to_c64(
        &self,
        input: &TypedTensor<f32>,
    ) -> crate::Result<TypedTensor<Complex64>> {
        self.convert_float_to_complex_raw::<f32, Complex64, f64>(
            input,
            |client, out, input, count| {
                unsafe {
                    // SAFETY: `convert_float_to_complex_raw` validated that
                    // `input` has `n` elements and `out` has `2 * n` scalar
                    // components. The kernel launches exactly `n` logical input
                    // positions and guards with `ABSOLUTE_POS < input.len()`.
                    structural::convert_f32_to_c64_raw::launch_unchecked::<CubeclCudaRuntime>(
                        client,
                        count,
                        cube_dim_1d(),
                        out,
                        input,
                    );
                }
                Ok(())
            },
        )
    }

    fn convert_f64_to_c32(
        &self,
        input: &TypedTensor<f64>,
    ) -> crate::Result<TypedTensor<Complex32>> {
        self.convert_float_to_complex_raw::<f64, Complex32, f32>(
            input,
            |client, out, input, count| {
                unsafe {
                    // SAFETY: `convert_float_to_complex_raw` validated that
                    // `input` has `n` elements and `out` has `2 * n` scalar
                    // components. The kernel launches exactly `n` logical input
                    // positions and guards with `ABSOLUTE_POS < input.len()`.
                    structural::convert_f64_to_c32_raw::launch_unchecked::<CubeclCudaRuntime>(
                        client,
                        count,
                        cube_dim_1d(),
                        out,
                        input,
                    );
                }
                Ok(())
            },
        )
    }

    fn convert_f64_to_c64(
        &self,
        input: &TypedTensor<f64>,
    ) -> crate::Result<TypedTensor<Complex64>> {
        self.convert_float_to_complex_raw::<f64, Complex64, f64>(
            input,
            |client, out, input, count| {
                unsafe {
                    // SAFETY: `convert_float_to_complex_raw` validated that
                    // `input` has `n` elements and `out` has `2 * n` scalar
                    // components. The kernel launches exactly `n` logical input
                    // positions and guards with `ABSOLUTE_POS < input.len()`.
                    structural::convert_f64_to_c64_raw::launch_unchecked::<CubeclCudaRuntime>(
                        client,
                        count,
                        cube_dim_1d(),
                        out,
                        input,
                    );
                }
                Ok(())
            },
        )
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
            CubeCount,
        ) -> crate::Result<()>,
    ) -> crate::Result<TypedTensor<OutComplex>>
    where
        InFloat: CubeElement + TensorScalar + Clone,
        OutComplex: CubeElement + TensorScalar + Clone,
        OutFloat: CubeElement + Clone,
    {
        ensure_resident_on_runtime(self.runtime(), input, "convert")?;
        let input_arg = typed_tensor_array_arg(input, "convert")?;
        let n = input.n_elements();
        let output_part_len = n.checked_mul(2).ok_or_else(|| {
            crate::Error::invalid_argument(
                "convert",
                "shape",
                "complex output part length overflow",
            )
        })?;
        let count = if n == 0 {
            None
        } else {
            Some(cube_count_for_len(n)?)
        };
        let output = alloc_output::<OutComplex>(self.runtime(), input.shape())?;
        let Some(count) = count else {
            return Ok(output);
        };
        let output_parts =
            typed_tensor_array_arg_as::<OutComplex, OutFloat>(&output, output_part_len, "convert")?;
        // SAFETY: The checked raw-array helpers prove that `input_arg` covers
        // exactly the dense input shape and `output_parts` covers the complete
        // real/imaginary scalar representation of the output allocation.
        launch(self.runtime().client(), output_parts, input_arg, count)?;
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

    fn convert_complex_to_complex<In, Out, InFloat, OutFloat>(
        &self,
        input: &TypedTensor<In>,
    ) -> crate::Result<TypedTensor<Out>>
    where
        In: CubeElement + TensorScalar + CubeComplex + Clone,
        Out: CubeElement + TensorScalar + CubeComplex + Clone,
        InFloat: CubeElement + CubeFloat + Clone,
        OutFloat: CubeElement + CubeFloat + Clone,
    {
        ensure_resident_on_runtime(self.runtime(), input, "cast")?;
        let parts = input.n_elements().checked_mul(2).ok_or_else(|| {
            crate::Error::invalid_argument("cast", "shape", "complex component length overflow")
        })?;
        let input_arg = typed_tensor_array_arg_as::<In, InFloat>(input, parts, "cast")?;
        let count = if parts == 0 {
            None
        } else {
            Some(cube_count_for_len(parts)?)
        };
        let output = alloc_output::<Out>(self.runtime(), input.shape())?;
        let Some(count) = count else {
            return Ok(output);
        };
        let output_arg = typed_tensor_array_arg_as::<Out, OutFloat>(&output, parts, "cast")?;
        unsafe {
            structural::convert_complex_raw::launch_unchecked::<OutFloat, InFloat, CubeclCudaRuntime>(
                self.runtime().client(),
                count,
                cube_dim_1d(),
                output_arg,
                input_arg,
            );
        }
        Ok(output)
    }

    fn extract_diagonal_typed<T>(
        &self,
        input: &TypedTensor<T>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
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

    fn extract_diagonal_bool(
        &self,
        input: &TypedTensor<bool>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<TypedTensor<bool>> {
        let (output_shape, diag_output_axis) =
            extract_diagonal_shape(input.shape(), axis_a, axis_b)?;
        launch_unary_bool_tensor(
            self.runtime(),
            input,
            &output_shape,
            "extract_diagonal",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::extract_diagonal_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
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

    fn embed_diagonal_bool(
        &self,
        input: &TypedTensor<bool>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<TypedTensor<bool>> {
        let output_shape = embed_diagonal_shape(input.shape(), axis_a, axis_b)?;
        ensure_resident_on_runtime(self.runtime(), input, "embed_diagonal")?;
        typed_tensor_binding(input, "embed_diagonal")?;
        let output_len = checked_dim_product("embed_diagonal", "output shape", &output_shape)?;
        let output_count = cube_count_for_len(output_len)?;
        let input_count = cube_count_for_len(input.n_elements())?;
        let output = dispatch::alloc_bool_output(self.runtime(), &output_shape)?;
        launch_nullary_bool_into(
            self.runtime(),
            &output,
            "embed_diagonal",
            output_count,
            cube_dim_1d(),
            |client, count, dim, out| unsafe {
                structural::fill_zero_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
                    client, count, dim, out,
                );
            },
        )?;
        launch_bool_tensor_into(
            self.runtime(),
            &output,
            input,
            "embed_diagonal",
            input_count,
            cube_dim_1d(),
            |client, count, dim, out, input_arg| unsafe {
                diagonal::embed_diagonal_copy_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
    {
        if input.shape().len() < 2 {
            return Err(crate::Error::rank_mismatch("tril", 2, input.shape().len()));
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

    fn tril_bool(&self, input: &TypedTensor<bool>, k: i64) -> crate::Result<TypedTensor<bool>> {
        if input.shape().len() < 2 {
            return Err(crate::Error::rank_mismatch("tril", 2, input.shape().len()));
        }
        launch_unary_bool_tensor(
            self.runtime(),
            input,
            input.shape(),
            "tril",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::tril_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
    {
        if input.shape().len() < 2 {
            return Err(crate::Error::rank_mismatch("triu", 2, input.shape().len()));
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

    fn triu_bool(&self, input: &TypedTensor<bool>, k: i64) -> crate::Result<TypedTensor<bool>> {
        if input.shape().len() < 2 {
            return Err(crate::Error::rank_mismatch("triu", 2, input.shape().len()));
        }
        launch_unary_bool_tensor(
            self.runtime(),
            input,
            input.shape(),
            "triu",
            |client, count, dim, out, input_arg| unsafe {
                diagonal::triu_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + Clone,
    {
        let output_shape = reduction_keepdims_shape(input.shape(), axis);
        let input_binding = typed_tensor_binding(input, op)?;
        let output = alloc_output::<T>(self.runtime(), &output_shape)?;
        if output.n_elements() == 0 {
            return Ok(output);
        }

        let output_binding = typed_tensor_binding(&output, op)?;
        launch(self.runtime().client(), input_binding, output_binding)
            .map_err(|err| crate::Error::backend_source(op, err))?;
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
        T: CubeElement
            + CubePrimitive
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    {
        ensure_axes_unique(op, "axes", axes, input.shape().len())?;
        if axes.is_empty() {
            return self.to_contiguous_view_typed(&input.as_view(), op);
        }

        let final_shape = reduction_output_shape(input.shape(), axes);
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();

        let mut current = self.to_contiguous_view_typed(&input.as_view(), op)?;
        for axis in sorted_axes {
            current = launch_axis(self, &current, axis)?;
        }

        cubecl_reshape_metadata(current, final_shape, op)
    }

    fn reduce_sum_float_typed<
        F: CubeElement
            + CubePrimitive
            + CubeFloat
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
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

    fn reduce_sum_squares_float_typed<
        F: CubeElement
            + CubePrimitive
            + CubeFloat
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
        &self,
        input: &TypedTensor<F>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<F>> {
        let op = op_name(
            PrimitiveOpKind::ReduceSumSquares,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        ensure_axes_unique(op, "axes", axes, input.shape().len())?;
        let final_shape = reduction_output_shape(input.shape(), axes);
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        let (&first_axis, remaining_axes) = sorted_axes
            .split_first()
            .ok_or_else(|| crate::Error::invalid_argument(op, "axes", "axes must not be empty"))?;

        let mut current =
            self.launch_reduce_axis_typed(input, first_axis, op, |client, input, output| {
                cubecl_reduce::launch_sum_squares_float::<CubeclCudaRuntime, F>(
                    client,
                    input,
                    output,
                    first_axis,
                    ReduceStrategy::Auto,
                )
            })?;
        for &axis in remaining_axes {
            current =
                self.launch_reduce_axis_typed(&current, axis, op, |client, input, output| {
                    cubecl_reduce::launch_sum_float::<CubeclCudaRuntime, F>(
                        client,
                        input,
                        output,
                        axis,
                        ReduceStrategy::Auto,
                    )
                })?;
        }

        cubecl_reshape_metadata(current, final_shape, op)
    }

    fn reduce_sum_complex_typed<
        C: CubeElement
            + CubePrimitive
            + CubeComplex
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
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

    fn reduce_sum_int_typed<
        I: CubeElement
            + CubePrimitive
            + CubeInt
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
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

    fn reduce_prod_float_typed<
        F: CubeElement
            + CubePrimitive
            + CubeFloat
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
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

    fn reduce_prod_complex_typed<
        C: CubeElement
            + CubePrimitive
            + CubeComplex
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
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

    fn reduce_prod_int_typed<
        I: CubeElement
            + CubePrimitive
            + CubeInt
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
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

    fn reduce_max_float_typed<
        F: CubeElement
            + CubePrimitive
            + CubeFloat
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
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

    fn reduce_max_int_typed<
        I: CubeElement
            + CubePrimitive
            + CubeInt
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
        &self,
        input: &TypedTensor<I>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<I>> {
        let op = op_name(
            PrimitiveOpKind::ReduceMax,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_max_int::<CubeclCudaRuntime, I>(
                    client,
                    input,
                    output,
                    axis,
                    ReduceStrategy::Auto,
                )
            })
        })
    }

    fn reduce_min_float_typed<
        F: CubeElement
            + CubePrimitive
            + CubeFloat
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
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

    fn reduce_min_int_typed<
        I: CubeElement
            + CubePrimitive
            + CubeInt
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    >(
        &self,
        input: &TypedTensor<I>,
        axes: &[usize],
    ) -> crate::Result<TypedTensor<I>> {
        let op = op_name(
            PrimitiveOpKind::ReduceMin,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        self.reduce_axes_typed(input, axes, op, |backend, current, axis| {
            backend.launch_reduce_axis_typed(current, axis, op, |client, input, output| {
                cubecl_reduce::launch_min_int::<CubeclCudaRuntime, I>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
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

    fn slice_bool(
        &self,
        input: &TypedTensor<bool>,
        config: &SliceConfig,
    ) -> crate::Result<TypedTensor<bool>> {
        let output_shape = validate_slice(input.shape(), config)?;
        launch_unary_bool_tensor(
            self.runtime(),
            input,
            &output_shape,
            "slice",
            |client, count, dim, out, input_arg| unsafe {
                indexing::slice_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
        I: CubeElement + TensorScalar + CubePrimitive + CubeNumeric + Clone + CudaIndexValidation,
    {
        ensure_rank("dynamic_slice", input.shape().len(), slice_sizes.len())?;
        ensure_rank("dynamic_slice", 1, starts.shape().len())?;
        if starts.shape()[0] != input.shape().len() {
            return Err(crate::Error::rank_mismatch(
                "dynamic_slice",
                input.shape().len(),
                starts.shape()[0],
            ));
        }
        for (axis, (&window, &dim)) in slice_sizes.iter().zip(input.shape()).enumerate() {
            if window > dim {
                return Err(crate::Error::invalid_argument(
                    "dynamic_slice",
                    "slice_sizes",
                    format!("slice size exceeds dimension on axis {axis}"),
                ));
            }
        }
        let output_len = checked_dim_product("dynamic_slice", "output shape", slice_sizes)?;
        if output_len != 0 {
            cube_count_for_len(output_len)?;
        }
        ensure_resident_on_runtime(self.runtime(), input, "dynamic_slice")?;
        typed_tensor_binding(input, "dynamic_slice")?;
        ensure_resident_on_runtime(self.runtime(), starts, "dynamic_slice")?;
        typed_tensor_binding(starts, "dynamic_slice")?;
        I::validate(self, starts)?;
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

    fn dynamic_slice_bool<I>(
        &self,
        input: &TypedTensor<bool>,
        starts: &TypedTensor<I>,
        slice_sizes: &[usize],
    ) -> crate::Result<TypedTensor<bool>>
    where
        I: CubeElement + TensorScalar + CubePrimitive + CubeNumeric + Clone + CudaIndexValidation,
    {
        ensure_rank("dynamic_slice", input.shape().len(), slice_sizes.len())?;
        if starts.shape().len() != 1 {
            return Err(crate::Error::invalid_argument(
                "dynamic_slice",
                "starts",
                "starts must be a rank-1 tensor",
            ));
        }
        if starts.shape()[0] != input.shape().len() {
            return Err(crate::Error::invalid_argument(
                "dynamic_slice",
                "starts",
                format!(
                    "starts length {} must match input rank {}",
                    starts.shape()[0],
                    input.shape().len()
                ),
            ));
        }
        for (axis, (&window, &dim)) in slice_sizes.iter().zip(input.shape()).enumerate() {
            if window > dim {
                return Err(crate::Error::invalid_argument(
                    "dynamic_slice",
                    "slice_sizes",
                    format!("slice size exceeds dimension on axis {axis}"),
                ));
            }
        }
        let output_len = checked_dim_product("dynamic_slice", "output shape", slice_sizes)?;
        if output_len != 0 {
            cube_count_for_len(output_len)?;
        }
        ensure_resident_on_runtime(self.runtime(), input, "dynamic_slice")?;
        bool_tensor_array_arg(input, "dynamic_slice")?;
        ensure_resident_on_runtime(self.runtime(), starts, "dynamic_slice")?;
        typed_tensor_binding(starts, "dynamic_slice")?;
        I::validate(self, starts)?;
        launch_binary_bool_tensor(
            self.runtime(),
            input,
            starts,
            slice_sizes,
            "dynamic_slice",
            |client, count, dim, out, input_arg, starts_arg| unsafe {
                indexing::dynamic_slice_kernel::launch_unchecked::<u8, I, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
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

    fn pad_bool(
        &self,
        input: &TypedTensor<bool>,
        config: &PadConfig,
    ) -> crate::Result<TypedTensor<bool>> {
        let output_shape = pad_output_shape(input.shape(), config)?;
        launch_unary_bool_tensor(
            self.runtime(),
            input,
            &output_shape,
            "pad",
            |client, count, dim, out, input_arg| unsafe {
                indexing::pad_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
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

    fn concatenate_bool(
        &self,
        inputs: &[&TypedTensor<bool>],
        axis: usize,
    ) -> crate::Result<TypedTensor<bool>> {
        let output_shape = concatenate_output_shape(inputs, axis)?;
        for input in inputs {
            ensure_resident_on_runtime(self.runtime(), input, "concatenate")?;
            typed_tensor_binding(input, "concatenate")?;
        }
        checked_dim_product("concatenate", "output shape", &output_shape)?;
        let launch_counts = inputs
            .iter()
            .map(|input| cube_count_for_len(input.n_elements()))
            .collect::<crate::Result<Vec<_>>>()?;
        let output = dispatch::alloc_bool_output(self.runtime(), &output_shape)?;
        let mut offset = 0usize;
        for (input, launch_count) in inputs.iter().zip(launch_counts) {
            launch_bool_tensor_into(
                self.runtime(),
                &output,
                input,
                "concatenate",
                launch_count,
                cube_dim_1d(),
                |client, count, dim, out, input_arg| unsafe {
                    structural::concatenate_copy_kernel::launch_unchecked::<u8, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
        I: CubeElement + TensorScalar + CubePrimitive + CubeNumeric + Clone + CudaIndexValidation,
    {
        let meta = gather_launch_meta(operand.shape(), start_indices.shape(), config)?;
        let output_len = checked_dim_product("gather", "output shape", &meta.output_shape)?;
        if output_len != 0 {
            cube_count_for_len(output_len)?;
        }
        ensure_resident_on_runtime(self.runtime(), operand, "gather")?;
        typed_tensor_binding(operand, "gather")?;
        ensure_resident_on_runtime(self.runtime(), start_indices, "gather")?;
        typed_tensor_binding(start_indices, "gather")?;
        I::validate(self, start_indices)?;
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

    fn gather_bool<I>(
        &self,
        operand: &TypedTensor<bool>,
        start_indices: &TypedTensor<I>,
        config: &GatherConfig,
    ) -> crate::Result<TypedTensor<bool>>
    where
        I: CubeElement + TensorScalar + CubePrimitive + CubeNumeric + Clone + CudaIndexValidation,
    {
        let meta = gather_launch_meta(operand.shape(), start_indices.shape(), config)?;
        let output_len = checked_dim_product("gather", "output shape", &meta.output_shape)?;
        if output_len != 0 {
            cube_count_for_len(output_len)?;
        }
        ensure_resident_on_runtime(self.runtime(), operand, "gather")?;
        bool_tensor_array_arg(operand, "gather")?;
        ensure_resident_on_runtime(self.runtime(), start_indices, "gather")?;
        typed_tensor_binding(start_indices, "gather")?;
        I::validate(self, start_indices)?;
        launch_binary_bool_tensor(
            self.runtime(),
            operand,
            start_indices,
            &meta.output_shape,
            "gather",
            |client, count, dim, out, operand_arg, indices_arg| unsafe {
                indexing::gather_kernel::launch_unchecked::<u8, I, CubeclCudaRuntime>(
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
        T: CubeElement + TensorScalar + CubeFloat + Clone,
        I: CubeElement + TensorScalar + CubePrimitive + CubeNumeric + Clone + CudaIndexValidation,
    {
        let meta = scatter_launch_meta(
            operand.shape(),
            scatter_indices.shape(),
            updates.shape(),
            config,
        )?;
        let update_len = scatter_update_len(&meta)?;
        let output_len = checked_dim_product("scatter", "output shape", operand.shape())?;
        if output_len != 0 {
            cube_count_for_len(output_len)?;
        }
        if update_len != 0 {
            cube_count_for_len(update_len)?;
        }
        let client = self.runtime().client();
        ensure_resident_on_runtime(self.runtime(), operand, "scatter")?;
        typed_tensor_binding(operand, "scatter")?;
        ensure_resident_on_runtime(self.runtime(), scatter_indices, "scatter")?;
        typed_tensor_binding(scatter_indices, "scatter")?;
        ensure_resident_on_runtime(self.runtime(), updates, "scatter")?;
        typed_tensor_binding(updates, "scatter")?;
        ensure_atomic_add_supported::<T>(client, "scatter")?;
        I::validate(self, scatter_indices)?;
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

        if update_len == 0 {
            return Ok(output);
        }
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
        T: CubeElement + TensorScalar + CubeComplex + Clone,
        F: CubeElement + TensorScalar + CubeFloat + Clone,
        I: CubeElement + TensorScalar + CubePrimitive + CubeNumeric + Clone + CudaIndexValidation,
    {
        let meta = scatter_launch_meta(
            operand.shape(),
            scatter_indices.shape(),
            updates.shape(),
            config,
        )?;
        let update_len = scatter_update_len(&meta)?;
        let output_len = checked_dim_product("scatter", "output shape", operand.shape())?;
        let output_part_len = output_len.checked_mul(2).ok_or_else(|| {
            crate::Error::invalid_argument(
                "scatter",
                "shape",
                "complex output part length overflow",
            )
        })?;
        let update_part_len = updates.n_elements().checked_mul(2).ok_or_else(|| {
            crate::Error::invalid_argument(
                "scatter",
                "shape",
                "complex update part length overflow",
            )
        })?;
        if output_len != 0 {
            cube_count_for_len(output_len)?;
        }
        if update_len != 0 {
            cube_count_for_len(update_len)?;
        }
        let client = self.runtime().client();
        ensure_resident_on_runtime(self.runtime(), operand, "scatter")?;
        typed_tensor_binding(operand, "scatter")?;
        ensure_resident_on_runtime(self.runtime(), scatter_indices, "scatter")?;
        typed_tensor_binding(scatter_indices, "scatter")?;
        ensure_resident_on_runtime(self.runtime(), updates, "scatter")?;
        typed_tensor_binding(updates, "scatter")?;
        typed_tensor_array_arg_as::<T, F>(updates, update_part_len, "scatter")?;
        ensure_atomic_add_supported::<F>(client, "scatter")?;
        I::validate(self, scatter_indices)?;
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

        if update_len == 0 {
            return Ok(output);
        }
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

#[derive(Clone, Copy, Debug)]
enum CheckedIntegerDomain {
    DivisionByZero,
    NegativeExponent,
}

#[derive(Clone, Copy)]
enum CastIntegerTarget {
    I32,
    I64,
}

trait CudaCastFloat:
    CubeElement
    + TensorScalar
    + CubeFloat
    + CubePrimitive<WithScalar<bool> = bool, WithScalar<Self> = Self>
    + Clone
    + Send
    + Sync
    + Copy
    + fmt::Display
    + 'static
{
    fn bounds(target: CastIntegerTarget) -> (Self, Self, bool);
    fn read_flag(backend: &CudaBackend, flag: &TypedTensor<Self>) -> crate::Result<Self>;
    fn invalid_error(self, target: CastIntegerTarget) -> crate::Error;
    fn is_nonfinite(self) -> bool;
    fn cpu_real_display(self) -> String;
}

macro_rules! impl_cuda_cast_float {
    ($ty:ty, $variant:ident, $i32_max_inclusive:expr, $display:expr) => {
        impl CudaCastFloat for $ty {
            fn bounds(target: CastIntegerTarget) -> (Self, Self, bool) {
                match target {
                    CastIntegerTarget::I32 => (
                        i32::MIN as Self,
                        if $i32_max_inclusive {
                            i32::MAX as Self
                        } else {
                            2_147_483_648.0 as Self
                        },
                        $i32_max_inclusive,
                    ),
                    CastIntegerTarget::I64 => (
                        -9_223_372_036_854_775_808.0 as Self,
                        9_223_372_036_854_775_808.0 as Self,
                        false,
                    ),
                }
            }
            fn read_flag(backend: &CudaBackend, flag: &TypedTensor<Self>) -> crate::Result<Self> {
                let host = interop::download_typed_tensor(backend.runtime(), flag, "cast")?;
                host.as_slice()?.get(1).copied().ok_or_else(|| {
                    crate::Error::invalid_argument(
                        "cast",
                        "validation_flag",
                        "validation flag was malformed",
                    )
                })
            }
            fn invalid_error(self, target: CastIntegerTarget) -> crate::Error {
                let name = match target {
                    CastIntegerTarget::I32 => "i32",
                    CastIntegerTarget::I64 => "i64",
                };
                let message = if !self.is_finite() {
                    format!(
                        "real value must be finite when casting to {name}, got {}",
                        self.cpu_real_display()
                    )
                } else {
                    format!(
                        "real value {} is out of {name} range",
                        self.cpu_real_display()
                    )
                };
                crate::Error::invalid_argument("cast", "value", message)
            }
            fn is_nonfinite(self) -> bool {
                !self.is_finite()
            }
            fn cpu_real_display(self) -> String {
                ($display)(self)
            }
        }
    };
}
impl_cuda_cast_float!(f32, F32, false, |value: f32| format!("{}", value as f64));
impl_cuda_cast_float!(f64, F64, true, |value: f64| format!("{value}"));

fn validate_cuda_real_cast<S, F>(
    backend: &CudaBackend,
    input: &TypedTensor<S>,
    stride: usize,
    target: CastIntegerTarget,
) -> crate::Result<()>
where
    S: CubeElement + TensorScalar + Clone,
    F: CudaCastFloat,
{
    ensure_resident_on_runtime(backend.runtime(), input, "cast")?;
    let n = input.n_elements();
    let _validated_input = typed_tensor_array_arg(input, "cast")?;
    if n == 0 {
        return Ok(());
    }
    u32::try_from(n).map_err(|_| {
        crate::Error::invalid_argument(
            "cast",
            "shape",
            "validation domain exceeds u32::MAX elements",
        )
    })?;
    let count = cube_count_for_len(n)?;
    let input_parts = n.checked_mul(stride).ok_or_else(|| {
        crate::Error::invalid_argument("cast", "shape", "validation input length overflow")
    })?;
    let input_arg = typed_tensor_array_arg_as::<S, F>(input, input_parts, "cast")?;
    let flag = alloc_output::<F>(backend.runtime(), &[2])?;
    let flag_u32_len = std::mem::size_of::<F>()
        .checked_mul(2)
        .and_then(|x| x.checked_div(std::mem::size_of::<u32>()))
        .ok_or_else(|| {
            crate::Error::invalid_argument("cast", "shape", "validation flag size overflow")
        })?;
    let flag_atomic = typed_tensor_array_arg_as::<F, u32>(&flag, flag_u32_len, "cast")?;
    let flag_values = typed_tensor_array_arg(&flag, "cast")?;
    unsafe {
        indexing::init_float_index_validation_flag::launch_unchecked::<F, CubeclCudaRuntime>(
            backend.runtime().client(),
            CubeCount::Static(1, 1, 1),
            cube_dim_1d(),
            flag_atomic,
            flag_values,
        );
    }
    let flag_atomic = typed_tensor_array_arg_as::<F, u32>(&flag, flag_u32_len, "cast")?;
    let (min, max, inclusive) = F::bounds(target);
    unsafe {
        structural::validate_real_cast::launch_unchecked::<F, CubeclCudaRuntime>(
            backend.runtime().client(),
            count,
            cube_dim_1d(),
            input_arg,
            flag_atomic,
            min,
            max,
            stride,
            inclusive,
        );
    }
    let input_arg = typed_tensor_array_arg_as::<S, F>(input, input_parts, "cast")?;
    let flag_atomic = typed_tensor_array_arg_as::<F, u32>(&flag, flag_u32_len, "cast")?;
    let flag_values = typed_tensor_array_arg(&flag, "cast")?;
    unsafe {
        structural::extract_invalid_real_cast::launch_unchecked::<F, CubeclCudaRuntime>(
            backend.runtime().client(),
            CubeCount::Static(1, 1, 1),
            cube_dim_1d(),
            input_arg,
            flag_atomic,
            flag_values,
            stride,
        );
    }
    let value = F::read_flag(backend, &flag)?;
    let (min, max, inclusive) = F::bounds(target);
    if value.is_nonfinite() || value < min || if inclusive { value > max } else { value >= max } {
        return Err(value.invalid_error(target));
    }
    Ok(())
}

fn checked_integer_domain_error(
    domain: CheckedIntegerDomain,
    op: &'static str,
    dtype: crate::DType,
) -> crate::Error {
    match domain {
        CheckedIntegerDomain::DivisionByZero => error::division_by_zero(op, dtype),
        CheckedIntegerDomain::NegativeExponent => error::negative_integer_exponent(op, dtype),
    }
}

fn read_checked_integer_flag(
    backend: &CudaBackend,
    flag: &TypedTensor<i32>,
    op: &'static str,
) -> crate::Result<i32> {
    let host = interop::download_typed_tensor(backend.runtime(), flag, op)?;
    Ok(host.as_slice()?.first().copied().unwrap_or_default())
}

trait CudaFloatIndex:
    CubeElement
    + TensorScalar
    + CubePrimitive<WithScalar<bool> = bool, WithScalar<Self> = Self>
    + CubeFloat
    + Clone
    + Send
    + Sync
    + fmt::Display
    + Copy
    + 'static
{
    const MAX_EXACT_INTEGER: Self;
    fn is_invalid_index(self) -> bool;
    fn read_invalid_flag(backend: &CudaBackend, flag: &TypedTensor<Self>) -> crate::Result<Self>;
}

trait CudaIndexValidation: Sized {
    fn validate(backend: &CudaBackend, indices: &TypedTensor<Self>) -> crate::Result<()>;
}

impl CudaIndexValidation for f32 {
    fn validate(backend: &CudaBackend, indices: &TypedTensor<Self>) -> crate::Result<()> {
        validate_float_index_tensor(backend, indices)
    }
}

impl CudaIndexValidation for f64 {
    fn validate(backend: &CudaBackend, indices: &TypedTensor<Self>) -> crate::Result<()> {
        validate_float_index_tensor(backend, indices)
    }
}

impl CudaIndexValidation for i32 {
    fn validate(_backend: &CudaBackend, _indices: &TypedTensor<Self>) -> crate::Result<()> {
        Ok(())
    }
}

impl CudaIndexValidation for i64 {
    fn validate(_backend: &CudaBackend, _indices: &TypedTensor<Self>) -> crate::Result<()> {
        Ok(())
    }
}

impl CudaFloatIndex for f32 {
    const MAX_EXACT_INTEGER: Self = 16_777_216.0;

    fn is_invalid_index(self) -> bool {
        !self.is_finite() || self.fract() != 0.0 || self.abs() > 16_777_216.0
    }

    fn read_invalid_flag(backend: &CudaBackend, flag: &TypedTensor<Self>) -> crate::Result<Self> {
        let host = interop::download_typed_tensor(backend.runtime(), flag, "index_tensor")?;
        host.as_slice()?.get(1).copied().ok_or_else(|| {
            crate::Error::invalid_argument(
                "index_tensor",
                "validation_flag",
                "validation flag was malformed",
            )
        })
    }
}

impl CudaFloatIndex for f64 {
    const MAX_EXACT_INTEGER: Self = 9_007_199_254_740_992.0;

    fn is_invalid_index(self) -> bool {
        !self.is_finite() || self.fract() != 0.0 || self.abs() > 9_007_199_254_740_992.0
    }

    fn read_invalid_flag(backend: &CudaBackend, flag: &TypedTensor<Self>) -> crate::Result<Self> {
        let host = interop::download_typed_tensor(backend.runtime(), flag, "index_tensor")?;
        host.as_slice()?.get(1).copied().ok_or_else(|| {
            crate::Error::invalid_argument(
                "index_tensor",
                "validation_flag",
                "validation flag was malformed",
            )
        })
    }
}

fn validate_float_index_tensor<F>(
    backend: &CudaBackend,
    indices: &TypedTensor<F>,
) -> crate::Result<()>
where
    F: CudaFloatIndex,
{
    ensure_resident_on_runtime(backend.runtime(), indices, "index_tensor")?;
    let indices_arg = typed_tensor_binding(indices, "index_tensor")?;
    if indices.n_elements() == 0 {
        return Ok(());
    }
    u32::try_from(indices.n_elements()).map_err(|_| {
        crate::Error::invalid_argument(
            "index_tensor",
            "shape",
            "float index validation domain exceeds u32::MAX elements",
        )
    })?;
    let count = cube_count_for_len(indices.n_elements())?;
    let flag_u32_len = std::mem::size_of::<F>()
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_div(std::mem::size_of::<u32>()))
        .ok_or_else(|| {
            crate::Error::invalid_argument("index_tensor", "shape", "flag size overflow")
        })?;
    let flag = alloc_output::<F>(backend.runtime(), &[2])?;
    let flag_values = typed_tensor_array_arg(&flag, "index_tensor")?;
    let flag_atomic = typed_tensor_array_arg_as::<F, u32>(&flag, flag_u32_len, "index_tensor")?;
    unsafe {
        // SAFETY: the flag allocation has two `F` elements, and the checked
        // reinterpretation above proves the atomic-u32 view fits that buffer.
        indexing::init_float_index_validation_flag::launch_unchecked::<F, CubeclCudaRuntime>(
            backend.runtime().client(),
            CubeCount::Static(1, 1, 1),
            cube_dim_1d(),
            flag_atomic,
            flag_values,
        );
    }
    let flag_atomic = typed_tensor_array_arg_as::<F, u32>(&flag, flag_u32_len, "index_tensor")?;
    unsafe {
        // SAFETY: the input binding was validated before allocation, the
        // launch domain is the checked input length, and the scalar flag view
        // was bounds-checked above.
        indexing::validate_float_indices_kernel::launch_unchecked::<F, CubeclCudaRuntime>(
            backend.runtime().client(),
            count,
            cube_dim_1d(),
            indices_arg.into_tensor_arg(),
            flag_atomic,
            F::MAX_EXACT_INTEGER,
        );
    }
    let indices_arg = typed_tensor_binding(indices, "index_tensor")?;
    let flag_atomic = typed_tensor_array_arg_as::<F, u32>(&flag, flag_u32_len, "index_tensor")?;
    let flag_values = typed_tensor_array_arg(&flag, "index_tensor")?;
    unsafe {
        // SAFETY: one worker reads the atomically selected in-range index and
        // copies that single value into the second element of the same flag.
        indexing::extract_invalid_float_index_kernel::launch_unchecked::<F, CubeclCudaRuntime>(
            backend.runtime().client(),
            CubeCount::Static(1, 1, 1),
            cube_dim_1d(),
            indices_arg.into_tensor_arg(),
            flag_atomic,
            flag_values,
        );
    }
    let invalid = F::read_invalid_flag(backend, &flag)?;
    if invalid.is_invalid_index() {
        return Err(crate::Error::invalid_argument(
            "index_tensor",
            "index",
            format!("index value {invalid} is not an exactly representable i64"),
        ));
    }
    Ok(())
}

fn launch_checked_integer_binary<I>(
    backend: &CudaBackend,
    lhs: &TypedTensor<I>,
    rhs: &TypedTensor<I>,
    op: &'static str,
    dtype: crate::DType,
    domain: CheckedIntegerDomain,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<TypedTensor<I>>
where
    I: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
{
    dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
    ensure_resident_on_runtime(backend.runtime(), lhs, op)?;
    ensure_resident_on_runtime(backend.runtime(), rhs, op)?;

    let output = alloc_output::<I>(backend.runtime(), lhs.shape())?;
    if output.n_elements() == 0 {
        return Ok(output);
    }

    let flag = alloc_output::<i32>(backend.runtime(), &[1])?;
    launch_nullary_into(
        backend.runtime(),
        &flag,
        op,
        cube_count_for_len(flag.n_elements())?,
        cube_dim_1d(),
        |client, count, dim, out| unsafe {
            structural::fill_zero_kernel::launch_unchecked::<i32, CubeclCudaRuntime>(
                client, count, dim, out,
            );
        },
    )?;

    let output_arg = typed_tensor_array_arg(&output, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    let flag_arg = typed_tensor_array_arg(&flag, op)?;
    launch(
        backend.runtime().client(),
        cube_count_for_len(output.n_elements())?,
        cube_dim_1d(),
        output_arg,
        lhs_arg,
        rhs_arg,
        flag_arg,
    );

    if read_checked_integer_flag(backend, &flag, op)? != 0 {
        return Err(checked_integer_domain_error(domain, op, dtype));
    }
    Ok(output)
}

fn launch_scalar_binary<I>(
    backend: &CudaBackend,
    lhs: &TypedTensor<I>,
    rhs: &TypedTensor<I>,
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        bool,
    ),
) -> crate::Result<TypedTensor<I>>
where
    I: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
{
    if !(lhs.shape().is_empty() ^ rhs.shape().is_empty()) {
        return Err(crate::Error::shape_mismatch(
            op,
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }
    ensure_resident_on_runtime(backend.runtime(), lhs, op)?;
    ensure_resident_on_runtime(backend.runtime(), rhs, op)?;

    let lhs_scalar = lhs.shape().is_empty();
    let output_shape = if lhs_scalar { rhs.shape() } else { lhs.shape() };
    let output = alloc_output::<I>(backend.runtime(), output_shape)?;
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    launch(
        backend.runtime().client(),
        cube_count_for_len(output.n_elements())?,
        cube_dim_1d(),
        output_arg,
        lhs_arg,
        rhs_arg,
        lhs_scalar,
    );
    Ok(output)
}

fn launch_real_complex_scalar_binary<R, C>(
    backend: &CudaBackend,
    real: &TypedTensor<R>,
    complex: &TypedTensor<C>,
    op: &'static str,
    real_lhs: bool,
    mode: usize,
) -> crate::Result<TypedTensor<C>>
where
    R: TensorScalar + CubeFloat + CubeElement + CubePrimitive + Clone + Send + Sync + 'static,
    C: CubeComplex<FloatElem = R>
        + TensorScalar
        + CubeElement
        + CubePrimitive
        + Clone
        + Send
        + Sync
        + 'static,
{
    if !real.shape().is_empty() {
        return Err(crate::Error::shape_mismatch(
            op,
            if real_lhs {
                real.shape().to_vec()
            } else {
                complex.shape().to_vec()
            },
            if real_lhs {
                complex.shape().to_vec()
            } else {
                real.shape().to_vec()
            },
        ));
    }
    ensure_resident_on_runtime(backend.runtime(), real, op)?;
    ensure_resident_on_runtime(backend.runtime(), complex, op)?;
    let component_len = complex.n_elements().checked_mul(2).ok_or_else(|| {
        crate::Error::invalid_argument(op, "shape", "complex component length overflow")
    })?;
    let real_arg = typed_tensor_array_arg(real, op)?;
    // INVARIANT: `num_complex::Complex<T>` is `repr(C)` with interleaved `{ re, im }`
    // fields; the checked `2 * n_elements` length and binding validator prove this
    // real-component view covers exactly the resident complex allocation.
    let complex_arg = typed_tensor_array_arg_as::<C, R>(complex, component_len, op)?;

    let output = alloc_output::<C>(backend.runtime(), complex.shape())?;
    let output_arg = typed_tensor_array_arg_as::<C, R>(&output, component_len, op)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    unsafe {
        elementwise::scalar_real_complex_binary::launch_unchecked::<R, CubeclCudaRuntime>(
            backend.runtime().client(),
            cube_count_for_len(output.n_elements())?,
            cube_dim_1d(),
            output_arg,
            real_arg,
            complex_arg,
            real_lhs,
            mode,
        );
    }
    Ok(output)
}

fn promoted_real_complex_scalar_binary(
    backend: &CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    op: &'static str,
    mode: usize,
) -> Option<crate::Result<Tensor>> {
    match (lhs, rhs) {
        (Tensor::F32(real), Tensor::C32(complex)) if real.shape().is_empty() => Some(
            launch_real_complex_scalar_binary(backend, real, complex, op, true, mode)
                .map(Tensor::C32),
        ),
        (Tensor::C32(complex), Tensor::F32(real)) if real.shape().is_empty() => Some(
            launch_real_complex_scalar_binary(backend, real, complex, op, false, mode)
                .map(Tensor::C32),
        ),
        (Tensor::F64(real), Tensor::C64(complex)) if real.shape().is_empty() => Some(
            launch_real_complex_scalar_binary(backend, real, complex, op, true, mode)
                .map(Tensor::C64),
        ),
        (Tensor::C64(complex), Tensor::F64(real)) if real.shape().is_empty() => Some(
            launch_real_complex_scalar_binary(backend, real, complex, op, false, mode)
                .map(Tensor::C64),
        ),
        _ => None,
    }
}

fn launch_checked_integer_scalar_binary<I>(
    backend: &CudaBackend,
    lhs: &TypedTensor<I>,
    rhs: &TypedTensor<I>,
    op: &'static str,
    dtype: crate::DType,
    domain: CheckedIntegerDomain,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        bool,
    ),
) -> crate::Result<TypedTensor<I>>
where
    I: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
{
    if !(lhs.shape().is_empty() ^ rhs.shape().is_empty()) {
        return Err(crate::Error::shape_mismatch(
            op,
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }
    ensure_resident_on_runtime(backend.runtime(), lhs, op)?;
    ensure_resident_on_runtime(backend.runtime(), rhs, op)?;

    let lhs_scalar = lhs.shape().is_empty();
    let output_shape = if lhs_scalar { rhs.shape() } else { lhs.shape() };
    let output = alloc_output::<I>(backend.runtime(), output_shape)?;
    let output_arg = typed_tensor_array_arg(&output, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    let flag = alloc_output::<i32>(backend.runtime(), &[1])?;
    let flag_arg = typed_tensor_array_arg(&flag, op)?;
    launch_nullary_into(
        backend.runtime(),
        &flag,
        op,
        cube_count_for_len(flag.n_elements())?,
        cube_dim_1d(),
        |client, count, dim, out| unsafe {
            structural::fill_zero_kernel::launch_unchecked::<i32, CubeclCudaRuntime>(
                client, count, dim, out,
            );
        },
    )?;

    launch(
        backend.runtime().client(),
        cube_count_for_len(output.n_elements())?,
        cube_dim_1d(),
        output_arg,
        lhs_arg,
        rhs_arg,
        flag_arg,
        lhs_scalar,
    );
    if read_checked_integer_flag(backend, &flag, op)? != 0 {
        return Err(checked_integer_domain_error(domain, op, dtype));
    }
    Ok(output)
}

impl TensorElementwise for CudaBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        if let Some(result) =
            promoted_real_complex_scalar_binary(self, lhs, rhs, "add", elementwise::MIXED_ADD)
        {
            return result;
        }
        dispatch::dispatch_binary_float_complex_int!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Add,
            add_float,
            add_int,
            add_complex
        )
    }

    fn sub(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        if let Some(result) =
            promoted_real_complex_scalar_binary(self, lhs, rhs, "sub", elementwise::MIXED_SUB)
        {
            return result;
        }
        dispatch::dispatch_binary_float_complex_int!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Sub,
            sub_float,
            sub_int,
            sub_complex
        )
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        if let Some(result) =
            promoted_real_complex_scalar_binary(self, lhs, rhs, "mul", elementwise::MIXED_MUL)
        {
            return result;
        }
        dispatch::dispatch_binary_float_complex_int!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Mul,
            mul_float,
            mul_int,
            mul_complex
        )
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_complex_int!(
            self,
            input,
            PrimitiveOpKind::Neg,
            neg_float,
            neg_int,
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
                self.to_contiguous_view_typed(&tensor.as_view(), op)
                    .map(Tensor::F32)
            }
            Tensor::F64(tensor) => {
                ensure_resident_on_runtime(self.runtime(), tensor, op)?;
                self.to_contiguous_view_typed(&tensor.as_view(), op)
                    .map(Tensor::F64)
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
        let op = op_name(
            PrimitiveOpKind::Div,
            op_descriptor::GpuLaunchKind::BinaryFloatComplexInt,
        )?;
        if let Some(result) =
            promoted_real_complex_scalar_binary(self, lhs, rhs, op, elementwise::MIXED_DIV)
        {
            return result;
        }
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) if lhs.shape() != rhs.shape() => {
                launch_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                        elementwise::scalar_div_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                            client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::F32)
            }
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) if lhs.shape() != rhs.shape() => {
                launch_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                        elementwise::scalar_div_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                            client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::F64)
            }
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::I32(lhs), Tensor::I32(rhs)) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    crate::DType::I32,
                    CheckedIntegerDomain::DivisionByZero,
                    |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                        elementwise::scalar_div_int_checked::launch_unchecked::<
                            i32,
                            CubeclCudaRuntime,
                        >(
                            client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::I32)
            }
            (Tensor::I32(lhs), Tensor::I32(rhs)) => launch_checked_integer_binary(
                self,
                lhs,
                rhs,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::div_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::I32),
            (Tensor::I64(lhs), Tensor::I64(rhs)) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    crate::DType::I64,
                    CheckedIntegerDomain::DivisionByZero,
                    |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                        elementwise::scalar_div_int_checked::launch_unchecked::<
                            i64,
                            CubeclCudaRuntime,
                        >(
                            client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::I64)
            }
            (Tensor::I64(lhs), Tensor::I64(rhs)) => launch_checked_integer_binary(
                self,
                lhs,
                rhs,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::div_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::I64),
            (Tensor::C32(lhs), Tensor::C32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_complex::launch_unchecked::<Complex32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::C32),
            (Tensor::C64(lhs), Tensor::C64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_complex::launch_unchecked::<Complex64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::C64),
            _ => Err(dtype_mismatch(op, lhs, rhs)),
        }
    }

    fn rem(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Rem,
            op_descriptor::GpuLaunchKind::BinaryFloatInt,
        )?;
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) if lhs.shape() != rhs.shape() => {
                launch_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                        elementwise::scalar_rem_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                            client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::F32)
            }
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::rem_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) if lhs.shape() != rhs.shape() => {
                launch_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                        elementwise::scalar_rem_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                            client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::F64)
            }
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::rem_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::I32(lhs), Tensor::I32(rhs)) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    crate::DType::I32,
                    CheckedIntegerDomain::DivisionByZero,
                    |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                        elementwise::scalar_rem_int_checked::launch_unchecked::<
                            i32,
                            CubeclCudaRuntime,
                        >(
                            client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::I32)
            }
            (Tensor::I32(lhs), Tensor::I32(rhs)) => launch_checked_integer_binary(
                self,
                lhs,
                rhs,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::rem_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::I32),
            (Tensor::I64(lhs), Tensor::I64(rhs)) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    crate::DType::I64,
                    CheckedIntegerDomain::DivisionByZero,
                    |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                        elementwise::scalar_rem_int_checked::launch_unchecked::<
                            i64,
                            CubeclCudaRuntime,
                        >(
                            client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::I64)
            }
            (Tensor::I64(lhs), Tensor::I64(rhs)) => launch_checked_integer_binary(
                self,
                lhs,
                rhs,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::rem_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::I64),
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => {
                Err(unsupported_dtype(op, lhs.dtype()))
            }
            _ => Err(dtype_mismatch(op, lhs, rhs)),
        }
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        let descriptor = op_descriptor::require_gpu_descriptor(
            PrimitiveOpKind::Abs,
            op_descriptor::GpuLaunchKind::UnaryFloatInt,
        )?;
        let op = descriptor.name;
        dispatch::require_owned_capability(self, PrimitiveOpKind::Abs, input.dtype())?;
        match input {
            Tensor::F32(tensor) => {
                dispatch::launch_unary_elementwise_kernel!(self, tensor, op, abs_float, f32, F32)
            }
            Tensor::F64(tensor) => {
                dispatch::launch_unary_elementwise_kernel!(self, tensor, op, abs_float, f64, F64)
            }
            Tensor::I32(tensor) => {
                dispatch::launch_unary_elementwise_kernel!(self, tensor, op, abs_int, i32, I32)
            }
            Tensor::I64(tensor) => {
                dispatch::launch_unary_elementwise_kernel!(self, tensor, op, abs_int, i64, I64)
            }
            Tensor::C32(tensor) => dispatch::launch_unary(
                self.runtime(),
                tensor,
                tensor.shape(),
                op,
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::abs_complex32::launch_unchecked::<CubeclCudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F32),
            Tensor::C64(tensor) => dispatch::launch_unary(
                self.runtime(),
                tensor,
                tensor.shape(),
                op,
                |client, count, dim, out, input_arg| unsafe {
                    elementwise::abs_complex64::launch_unchecked::<CubeclCudaRuntime>(
                        client, count, dim, out, input_arg,
                    );
                },
            )
            .map(Tensor::F64),
            Tensor::Bool(_) => Err(unsupported_dtype(op, input.dtype())),
        }
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_int!(
            self,
            input,
            PrimitiveOpKind::Sign,
            sign_float,
            sign_int
        )
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_binary_float_int!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Maximum,
            maximum_float,
            maximum_int
        )
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_binary_float_int!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Minimum,
            minimum_float,
            minimum_int
        )
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Compare,
            op_descriptor::GpuLaunchKind::CompareFloatIntToBool,
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
            (Tensor::I32(lhs), Tensor::I32(rhs)) => launch_compare_bool(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::compare_int_bool::launch_unchecked::<i32, CubeclCudaRuntime>(
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
            (Tensor::I64(lhs), Tensor::I64(rhs)) => launch_compare_bool(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::compare_int_bool::launch_unchecked::<i64, CubeclCudaRuntime>(
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
            (Tensor::C32(_), Tensor::C32(_)) | (Tensor::C64(_), Tensor::C64(_)) => {
                Err(unsupported_dtype(op, lhs.dtype()))
            }
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
            op_descriptor::GpuLaunchKind::SelectBoolFloatInt,
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
            (Tensor::Bool(pred), Tensor::I32(on_true), Tensor::I32(on_false)) => {
                launch_select_bool(
                    self.runtime(),
                    pred,
                    on_true,
                    on_false,
                    pred.shape(),
                    op,
                    |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                        elementwise::select_bool_int::launch_unchecked::<i32, CubeclCudaRuntime>(
                            client, count, dim, out, pred_arg, true_arg, false_arg,
                        );
                    },
                )
                .map(Tensor::I32)
            }
            (Tensor::Bool(pred), Tensor::I64(on_true), Tensor::I64(on_false)) => {
                launch_select_bool(
                    self.runtime(),
                    pred,
                    on_true,
                    on_false,
                    pred.shape(),
                    op,
                    |client, count, dim, out, pred_arg, true_arg, false_arg| unsafe {
                        elementwise::select_bool_int::launch_unchecked::<i64, CubeclCudaRuntime>(
                            client, count, dim, out, pred_arg, true_arg, false_arg,
                        );
                    },
                )
                .map(Tensor::I64)
            }
            (Tensor::C32(_), Tensor::C32(_), Tensor::C32(_))
            | (Tensor::C64(_), Tensor::C64(_), Tensor::C64(_)) => {
                Err(unsupported_dtype(op, pred.dtype()))
            }
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
            | (Tensor::C64(_), Tensor::C64(_), Tensor::C64(_)) => {
                Err(unsupported_dtype(op, input.dtype()))
            }
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
        let op = op_name(
            PrimitiveOpKind::Pow,
            op_descriptor::GpuLaunchKind::BinaryFloatInt,
        )?;
        if lhs.dtype() != rhs.dtype() {
            return Err(dtype_mismatch(op, lhs, rhs));
        }
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) if lhs.shape() != rhs.shape() => {
                launch_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                        elementwise::scalar_pow_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                            client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::F32)
            }
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::pow_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F32),
            (Tensor::F64(lhs), Tensor::F64(rhs)) if lhs.shape() != rhs.shape() => {
                launch_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                        elementwise::scalar_pow_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                            client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::F64)
            }
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_binary(
                self.runtime(),
                lhs,
                rhs,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::pow_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::F64),
            (Tensor::I32(lhs), Tensor::I32(rhs)) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    crate::DType::I32,
                    CheckedIntegerDomain::NegativeExponent,
                    |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                        elementwise::scalar_pow_int_checked::launch_unchecked::<
                            i32,
                            CubeclCudaRuntime,
                        >(
                            client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::I32)
            }
            (Tensor::I32(lhs), Tensor::I32(rhs)) => launch_checked_integer_binary(
                self,
                lhs,
                rhs,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::NegativeExponent,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::pow_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::I32),
            (Tensor::I64(lhs), Tensor::I64(rhs)) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    lhs,
                    rhs,
                    op,
                    crate::DType::I64,
                    CheckedIntegerDomain::NegativeExponent,
                    |client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar| unsafe {
                        elementwise::scalar_pow_int_checked::launch_unchecked::<
                            i64,
                            CubeclCudaRuntime,
                        >(
                            client, count, dim, out, lhs_arg, rhs_arg, err_arg, lhs_scalar,
                        );
                    },
                )
                .map(Tensor::I64)
            }
            (Tensor::I64(lhs), Tensor::I64(rhs)) => launch_checked_integer_binary(
                self,
                lhs,
                rhs,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::NegativeExponent,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::pow_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::I64),
            (Tensor::C32(lhs), Tensor::C32(rhs)) => {
                dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
                Err(unsupported_dtype(op, crate::DType::C32))
            }
            (Tensor::C64(lhs), Tensor::C64(rhs)) => {
                dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
                Err(unsupported_dtype(op, crate::DType::C64))
            }
            _ => {
                dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
                Err(dtype_mismatch(op, lhs, rhs))
            }
        }
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Expm1, expm1_float)
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_only!(self, input, PrimitiveOpKind::Log1p, log1p_float)
    }
}

impl TensorStructural for CudaBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        macro_rules! materialize_cutensor {
            ($variant:ident, $view:expr) => {{
                let view = $view;
                self.to_contiguous_view_cutensor_or_cubecl(&view, "CudaBackend::to_contiguous_read")
                    .map(Tensor::$variant)
            }};
        }
        macro_rules! materialize_cubecl {
            ($variant:ident, $view:expr) => {{
                let view = $view;
                self.to_contiguous_view_typed(&view, "CudaBackend::to_contiguous_read")
                    .map(Tensor::$variant)
            }};
        }

        match input {
            TensorRead::Tensor(Tensor::F32(input)) => materialize_cutensor!(F32, input.as_view()),
            TensorRead::Tensor(Tensor::F64(input)) => materialize_cutensor!(F64, input.as_view()),
            TensorRead::Tensor(Tensor::I32(input)) => materialize_cubecl!(I32, input.as_view()),
            TensorRead::Tensor(Tensor::I64(input)) => materialize_cubecl!(I64, input.as_view()),
            TensorRead::Tensor(Tensor::Bool(_)) => Err(unsupported_dtype(
                "CudaBackend::to_contiguous_read",
                crate::DType::Bool,
            )),
            TensorRead::Tensor(Tensor::C32(input)) => materialize_cutensor!(C32, input.as_view()),
            TensorRead::Tensor(Tensor::C64(input)) => materialize_cutensor!(C64, input.as_view()),
            TensorRead::View(TensorView::F32(input)) => materialize_cutensor!(F32, input),
            TensorRead::View(TensorView::F64(input)) => materialize_cutensor!(F64, input),
            TensorRead::View(TensorView::I32(input)) => materialize_cubecl!(I32, input),
            TensorRead::View(TensorView::I64(input)) => materialize_cubecl!(I64, input),
            TensorRead::View(TensorView::Bool(_)) => Err(unsupported_dtype(
                "CudaBackend::to_contiguous_read",
                crate::DType::Bool,
            )),
            TensorRead::View(TensorView::C32(input)) => materialize_cutensor!(C32, input),
            TensorRead::View(TensorView::C64(input)) => materialize_cutensor!(C64, input),
        }
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> crate::Result<()> {
        let src_dtype = src.dtype();
        let dst_dtype = dst.dtype();
        macro_rules! copy_source_typed {
            ($variant:ident, $src:expr) => {{
                let src = $src;
                match dst {
                    TensorWrite::Tensor(Tensor::$variant(dst)) => {
                        let mut dst = dst.as_view_mut();
                        self.copy_view_to_view_typed(&src, &mut dst, "CudaBackend::copy_read_into")
                    }
                    TensorWrite::View(TensorViewMut::$variant(mut dst)) => {
                        self.copy_view_to_view_typed(&src, &mut dst, "CudaBackend::copy_read_into")
                    }
                    _ => Err(crate::Error::dtype_mismatch(
                        "CudaBackend::copy_read_into",
                        src_dtype,
                        dst_dtype,
                    )),
                }
            }};
        }
        macro_rules! copy_source_cutensor {
            ($variant:ident, $src:expr) => {{
                let src = $src;
                match dst {
                    TensorWrite::Tensor(Tensor::$variant(dst)) => {
                        let mut dst = dst.as_view_mut();
                        self.copy_view_to_view_cutensor_or_cubecl(
                            &src,
                            &mut dst,
                            "CudaBackend::copy_read_into",
                        )
                    }
                    TensorWrite::View(TensorViewMut::$variant(mut dst)) => self
                        .copy_view_to_view_cutensor_or_cubecl(
                            &src,
                            &mut dst,
                            "CudaBackend::copy_read_into",
                        ),
                    _ => Err(crate::Error::dtype_mismatch(
                        "CudaBackend::copy_read_into",
                        src_dtype,
                        dst_dtype,
                    )),
                }
            }};
        }
        macro_rules! reject_bool_source {
            () => {{
                match dst {
                    TensorWrite::Tensor(Tensor::Bool(_))
                    | TensorWrite::View(TensorViewMut::Bool(_)) => Err(unsupported_dtype(
                        "CudaBackend::copy_read_into",
                        crate::DType::Bool,
                    )),
                    _ => Err(crate::Error::dtype_mismatch(
                        "CudaBackend::copy_read_into",
                        src_dtype,
                        dst_dtype,
                    )),
                }
            }};
        }

        match src {
            TensorRead::Tensor(Tensor::F32(src)) => copy_source_cutensor!(F32, src.as_view()),
            TensorRead::Tensor(Tensor::F64(src)) => copy_source_cutensor!(F64, src.as_view()),
            TensorRead::Tensor(Tensor::I32(src)) => copy_source_typed!(I32, src.as_view()),
            TensorRead::Tensor(Tensor::I64(src)) => copy_source_typed!(I64, src.as_view()),
            TensorRead::Tensor(Tensor::Bool(_)) => reject_bool_source!(),
            TensorRead::Tensor(Tensor::C32(src)) => copy_source_cutensor!(C32, src.as_view()),
            TensorRead::Tensor(Tensor::C64(src)) => copy_source_cutensor!(C64, src.as_view()),
            TensorRead::View(TensorView::F32(src)) => copy_source_cutensor!(F32, src),
            TensorRead::View(TensorView::F64(src)) => copy_source_cutensor!(F64, src),
            TensorRead::View(TensorView::I32(src)) => copy_source_typed!(I32, src),
            TensorRead::View(TensorView::I64(src)) => copy_source_typed!(I64, src),
            TensorRead::View(TensorView::Bool(_)) => reject_bool_source!(),
            TensorRead::View(TensorView::C32(src)) => copy_source_cutensor!(C32, src),
            TensorRead::View(TensorView::C64(src)) => copy_source_cutensor!(C64, src),
        }
    }

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => permutation::transpose(self, t, perm).map(Tensor::F32),
            Tensor::F64(t) => permutation::transpose(self, t, perm).map(Tensor::F64),
            Tensor::I32(t) => self.transpose_typed(t, perm).map(Tensor::I32),
            Tensor::I64(t) => self.transpose_typed(t, perm).map(Tensor::I64),
            Tensor::Bool(t) => self.transpose_bool(t, perm).map(Tensor::Bool),
            Tensor::C32(t) => permutation::transpose(self, t, perm).map(Tensor::C32),
            Tensor::C64(t) => permutation::transpose(self, t, perm).map(Tensor::C64),
        }
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        let old_n = checked_dim_product("reshape", "input shape", input.shape())?;
        let new_n = checked_dim_product("reshape", "output shape", shape)?;
        if old_n != new_n {
            return Err(crate::Error::validation(
                "reshape",
                tenferro_tensor::ShapeMismatch::ReshapeElementCount {
                    from: old_n,
                    to: new_n,
                }
                .into(),
            ));
        }
        // An owned tensor cannot be returned by shallowly reusing a backend
        // buffer. Materialize one explicit same-placement copy first, then
        // change only its compact metadata.
        let contiguous = match input {
            Tensor::Bool(tensor) => self.duplicate_bool(tensor, "reshape").map(Tensor::Bool)?,
            _ => self.to_contiguous_read(TensorRead::from_tensor(input))?,
        };
        match contiguous {
            Tensor::F32(t) => {
                cubecl_reshape_metadata(t, shape.to_vec(), "reshape").map(Tensor::F32)
            }
            Tensor::F64(t) => {
                cubecl_reshape_metadata(t, shape.to_vec(), "reshape").map(Tensor::F64)
            }
            Tensor::I32(t) => {
                cubecl_reshape_metadata(t, shape.to_vec(), "reshape").map(Tensor::I32)
            }
            Tensor::I64(t) => {
                cubecl_reshape_metadata(t, shape.to_vec(), "reshape").map(Tensor::I64)
            }
            Tensor::Bool(t) => {
                cubecl_reshape_metadata(t, shape.to_vec(), "reshape").map(Tensor::Bool)
            }
            Tensor::C32(t) => {
                cubecl_reshape_metadata(t, shape.to_vec(), "reshape").map(Tensor::C32)
            }
            Tensor::C64(t) => {
                cubecl_reshape_metadata(t, shape.to_vec(), "reshape").map(Tensor::C64)
            }
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
            Tensor::Bool(t) => self.broadcast_bool(t, shape, dims).map(Tensor::Bool),
            Tensor::C32(t) => self.broadcast_typed(t, shape, dims).map(Tensor::C32),
            Tensor::C64(t) => self.broadcast_typed(t, shape, dims).map(Tensor::C64),
        }
    }

    fn cast(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        match (input, to) {
            (Tensor::F32(t), crate::DType::F32) => self.duplicate_typed(t).map(Tensor::F32),
            (Tensor::F64(t), crate::DType::F64) => self.duplicate_typed(t).map(Tensor::F64),
            (Tensor::I32(t), crate::DType::I32) => self.duplicate_typed(t).map(Tensor::I32),
            (Tensor::I64(t), crate::DType::I64) => self.duplicate_typed(t).map(Tensor::I64),
            (Tensor::Bool(t), crate::DType::Bool) => {
                self.duplicate_bool(t, "cast").map(Tensor::Bool)
            }
            (Tensor::C32(t), crate::DType::C32) => self.duplicate_typed(t).map(Tensor::C32),
            (Tensor::C64(t), crate::DType::C64) => self.duplicate_typed(t).map(Tensor::C64),
            (Tensor::F32(t), crate::DType::F64) => {
                self.convert_float_to_float::<f32, f64>(t).map(Tensor::F64)
            }
            (Tensor::F32(t), crate::DType::I32) => {
                validate_cuda_real_cast::<f32, f32>(self, t, 1, CastIntegerTarget::I32)?;
                self.convert_numeric::<f32, i32>(t).map(Tensor::I32)
            }
            (Tensor::F32(t), crate::DType::I64) => {
                validate_cuda_real_cast::<f32, f32>(self, t, 1, CastIntegerTarget::I64)?;
                self.convert_numeric::<f32, i64>(t).map(Tensor::I64)
            }
            (Tensor::F32(t), crate::DType::Bool) => {
                self.convert_numeric_to_bool(t).map(Tensor::Bool)
            }
            (Tensor::F32(t), crate::DType::C32) => self.convert_f32_to_c32(t).map(Tensor::C32),
            (Tensor::F32(t), crate::DType::C64) => self.convert_f32_to_c64(t).map(Tensor::C64),
            (Tensor::F64(t), crate::DType::F32) => {
                self.convert_float_to_float::<f64, f32>(t).map(Tensor::F32)
            }
            (Tensor::F64(t), crate::DType::I32) => {
                validate_cuda_real_cast::<f64, f64>(self, t, 1, CastIntegerTarget::I32)?;
                self.convert_numeric::<f64, i32>(t).map(Tensor::I32)
            }
            (Tensor::F64(t), crate::DType::I64) => {
                validate_cuda_real_cast::<f64, f64>(self, t, 1, CastIntegerTarget::I64)?;
                self.convert_numeric::<f64, i64>(t).map(Tensor::I64)
            }
            (Tensor::F64(t), crate::DType::Bool) => {
                self.convert_numeric_to_bool(t).map(Tensor::Bool)
            }
            (Tensor::F64(t), crate::DType::C32) => self.convert_f64_to_c32(t).map(Tensor::C32),
            (Tensor::F64(t), crate::DType::C64) => self.convert_f64_to_c64(t).map(Tensor::C64),
            (Tensor::I32(t), crate::DType::F32) => {
                self.convert_numeric::<i32, f32>(t).map(Tensor::F32)
            }
            (Tensor::I32(t), crate::DType::F64) => {
                self.convert_numeric::<i32, f64>(t).map(Tensor::F64)
            }
            (Tensor::I32(t), crate::DType::I64) => {
                self.convert_numeric::<i32, i64>(t).map(Tensor::I64)
            }
            (Tensor::I32(t), crate::DType::Bool) => {
                self.convert_numeric_to_bool(t).map(Tensor::Bool)
            }
            (Tensor::I32(t), crate::DType::C32) => self
                .convert_numeric_to_complex::<i32, Complex32, f32>(t)
                .map(Tensor::C32),
            (Tensor::I32(t), crate::DType::C64) => self
                .convert_numeric_to_complex::<i32, Complex64, f64>(t)
                .map(Tensor::C64),
            (Tensor::I64(t), crate::DType::F32) => {
                self.convert_numeric::<i64, f32>(t).map(Tensor::F32)
            }
            (Tensor::I64(t), crate::DType::F64) => {
                self.convert_numeric::<i64, f64>(t).map(Tensor::F64)
            }
            (Tensor::I64(t), crate::DType::I32) => {
                self.convert_numeric::<i64, i32>(t).map(Tensor::I32)
            }
            (Tensor::I64(t), crate::DType::Bool) => {
                self.convert_numeric_to_bool(t).map(Tensor::Bool)
            }
            (Tensor::I64(t), crate::DType::C32) => self
                .convert_numeric_to_complex::<i64, Complex32, f32>(t)
                .map(Tensor::C32),
            (Tensor::I64(t), crate::DType::C64) => self
                .convert_numeric_to_complex::<i64, Complex64, f64>(t)
                .map(Tensor::C64),
            (Tensor::Bool(t), crate::DType::F32) => {
                self.convert_bool_to_numeric::<f32>(t).map(Tensor::F32)
            }
            (Tensor::Bool(t), crate::DType::F64) => {
                self.convert_bool_to_numeric::<f64>(t).map(Tensor::F64)
            }
            (Tensor::Bool(t), crate::DType::I32) => {
                self.convert_bool_to_numeric::<i32>(t).map(Tensor::I32)
            }
            (Tensor::Bool(t), crate::DType::I64) => {
                self.convert_bool_to_numeric::<i64>(t).map(Tensor::I64)
            }
            (Tensor::Bool(t), crate::DType::C32) => self
                .convert_bool_to_complex::<Complex32, f32>(t)
                .map(Tensor::C32),
            (Tensor::Bool(t), crate::DType::C64) => self
                .convert_bool_to_complex::<Complex64, f64>(t)
                .map(Tensor::C64),
            (Tensor::C32(t), crate::DType::F32) => self.convert_c32_to_f32(t).map(Tensor::F32),
            (Tensor::C32(t), crate::DType::F64) => self.convert_c32_to_f64(t).map(Tensor::F64),
            (Tensor::C32(t), crate::DType::I32) => {
                validate_cuda_real_cast::<Complex32, f32>(self, t, 2, CastIntegerTarget::I32)?;
                self.convert_complex_to_numeric::<Complex32, i32>(t)
                    .map(Tensor::I32)
            }
            (Tensor::C32(t), crate::DType::I64) => {
                validate_cuda_real_cast::<Complex32, f32>(self, t, 2, CastIntegerTarget::I64)?;
                self.convert_complex_to_numeric::<Complex32, i64>(t)
                    .map(Tensor::I64)
            }
            (Tensor::C32(t), crate::DType::Bool) => self
                .convert_complex_to_bool::<Complex32, f32>(t)
                .map(Tensor::Bool),
            (Tensor::C32(t), crate::DType::C64) => self
                .convert_complex_to_complex::<Complex32, Complex64, f32, f64>(t)
                .map(Tensor::C64),
            (Tensor::C64(t), crate::DType::F32) => self.convert_c64_to_f32(t).map(Tensor::F32),
            (Tensor::C64(t), crate::DType::F64) => self.convert_c64_to_f64(t).map(Tensor::F64),
            (Tensor::C64(t), crate::DType::I32) => {
                validate_cuda_real_cast::<Complex64, f64>(self, t, 2, CastIntegerTarget::I32)?;
                self.convert_complex_to_numeric::<Complex64, i32>(t)
                    .map(Tensor::I32)
            }
            (Tensor::C64(t), crate::DType::I64) => {
                validate_cuda_real_cast::<Complex64, f64>(self, t, 2, CastIntegerTarget::I64)?;
                self.convert_complex_to_numeric::<Complex64, i64>(t)
                    .map(Tensor::I64)
            }
            (Tensor::C64(t), crate::DType::Bool) => self
                .convert_complex_to_bool::<Complex64, f64>(t)
                .map(Tensor::Bool),
            (Tensor::C64(t), crate::DType::C32) => self
                .convert_complex_to_complex::<Complex64, Complex32, f64, f32>(t)
                .map(Tensor::C32),
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
            Tensor::Bool(t) => self
                .extract_diagonal_bool(t, axis_a, axis_b)
                .map(Tensor::Bool),
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
            Tensor::Bool(t) => self
                .embed_diagonal_bool(t, axis_a, axis_b)
                .map(Tensor::Bool),
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
            Tensor::Bool(t) => self.tril_bool(t, k).map(Tensor::Bool),
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
            Tensor::Bool(t) => self.triu_bool(t, k).map(Tensor::Bool),
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

    fn reduce_sum_squares_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceSumSquares,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        let Some(input) = input.as_tensor() else {
            return Err(crate::Error::unsupported(
                op,
                "CUDA sum-of-squares requires a resident tensor",
            ));
        };
        if axes.is_empty() {
            return match input {
                Tensor::F32(_) | Tensor::F64(_) => self.mul(input, input),
                _ => Err(unsupported_dtype(op, input.dtype())),
            };
        }
        match input {
            Tensor::F32(t) => self
                .reduce_sum_squares_float_typed(t, axes)
                .map(Tensor::F32),
            Tensor::F64(t) => self
                .reduce_sum_squares_float_typed(t, axes)
                .map(Tensor::F64),
            _ => Err(unsupported_dtype(op, input.dtype())),
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
            Tensor::F32(t) => self.reduce_max_float_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_max_float_typed(t, axes).map(Tensor::F64),
            Tensor::I32(t) => self.reduce_max_int_typed(t, axes).map(Tensor::I32),
            Tensor::I64(t) => self.reduce_max_int_typed(t, axes).map(Tensor::I64),
            Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => {
                Err(unsupported_dtype(op, input.dtype()))
            }
        }
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceMin,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input {
            Tensor::F32(t) => self.reduce_min_float_typed(t, axes).map(Tensor::F32),
            Tensor::F64(t) => self.reduce_min_float_typed(t, axes).map(Tensor::F64),
            Tensor::I32(t) => self.reduce_min_int_typed(t, axes).map(Tensor::I32),
            Tensor::I64(t) => self.reduce_min_int_typed(t, axes).map(Tensor::I64),
            Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => {
                Err(unsupported_dtype(op, input.dtype()))
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

    // CUDA-native accumulation (tensor4all/tenferro-rs#1287): one cuTENSOR
    // contraction with C = D = out; no temporary result tensor, no host
    // transfer. Stage 2 accepts compact owned tensors and borrowed strided
    // views over device buffers on all three slots; host-backed views are an
    // explicit backend error.
    fn dot_general_read_into_accum(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        mut out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        tenferro_tensor::backend::validate_dot_general_accumulation(
            &lhs,
            &rhs,
            config,
            accumulation,
            &out,
            "dot_general",
        )?;
        gemm::dot_general_read_into_accum(self, &lhs, &rhs, config, accumulation, &mut out)
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
            (Tensor::Bool(operand), Tensor::F32(indices)) => {
                self.gather_bool(operand, indices, config).map(Tensor::Bool)
            }
            (Tensor::Bool(operand), Tensor::F64(indices)) => {
                self.gather_bool(operand, indices, config).map(Tensor::Bool)
            }
            (Tensor::Bool(operand), Tensor::I32(indices)) => {
                self.gather_bool(operand, indices, config).map(Tensor::Bool)
            }
            (Tensor::Bool(operand), Tensor::I64(indices)) => {
                self.gather_bool(operand, indices, config).map(Tensor::Bool)
            }
            (_, Tensor::Bool(_)) => Err(unsupported_dtype("gather", start_indices.dtype())),
            (_, Tensor::C32(_) | Tensor::C64(_)) => {
                Err(unsupported_dtype("gather", start_indices.dtype()))
            }
            (Tensor::I64(_), _) => Err(unsupported_dtype("gather", operand.dtype())),
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
            (_, Tensor::C32(_) | Tensor::C64(_), _) => {
                Err(unsupported_dtype("scatter", scatter_indices.dtype()))
            }
            (Tensor::Bool(_), _, _) => Err(unsupported_operation(
                "scatter",
                "Bool data tensors are not supported by additive scatter",
            )),
            (Tensor::I32(_), _, _) | (Tensor::I64(_), _, _) => {
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
            Tensor::Bool(t) => self.slice_bool(t, config).map(Tensor::Bool),
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
            (Tensor::Bool(input), Tensor::I32(starts)) => self
                .dynamic_slice_bool(input, starts, slice_sizes)
                .map(Tensor::Bool),
            (Tensor::Bool(input), Tensor::I64(starts)) => self
                .dynamic_slice_bool(input, starts, slice_sizes)
                .map(Tensor::Bool),
            (Tensor::Bool(input), Tensor::F32(starts)) => self
                .dynamic_slice_bool(input, starts, slice_sizes)
                .map(Tensor::Bool),
            (Tensor::Bool(input), Tensor::F64(starts)) => self
                .dynamic_slice_bool(input, starts, slice_sizes)
                .map(Tensor::Bool),
            (_, Tensor::Bool(_)) => Err(unsupported_dtype("dynamic_slice", starts.dtype())),
            (_, Tensor::C32(_) | Tensor::C64(_)) => {
                Err(unsupported_dtype("dynamic_slice", starts.dtype()))
            }
            (Tensor::I64(_), _) => Err(unsupported_dtype("dynamic_slice", input.dtype())),
        }
    }

    fn dynamic_update_slice(
        &mut self,
        _operand: &Tensor,
        _update: &Tensor,
        _starts: &Tensor,
    ) -> crate::Result<Tensor> {
        Err(unsupported_operation(
            "dynamic_update_slice",
            "not implemented for the CubeCL backend",
        ))
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        match input {
            Tensor::F32(t) => self.pad_typed(t, config).map(Tensor::F32),
            Tensor::F64(t) => self.pad_typed(t, config).map(Tensor::F64),
            Tensor::I32(t) => self.pad_typed(t, config).map(Tensor::I32),
            Tensor::I64(t) => self.pad_typed(t, config).map(Tensor::I64),
            Tensor::Bool(t) => self.pad_bool(t, config).map(Tensor::Bool),
            Tensor::C32(t) => self.pad_typed(t, config).map(Tensor::C32),
            Tensor::C64(t) => self.pad_typed(t, config).map(Tensor::C64),
        }
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        let first = inputs.first().copied().ok_or_else(|| {
            crate::Error::invalid_argument(
                "concatenate",
                "inputs",
                "concatenate requires at least one input",
            )
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
            Tensor::Bool(_) => {
                let typed: crate::Result<Vec<&TypedTensor<bool>>> = inputs
                    .iter()
                    .map(|tensor| match tensor {
                        Tensor::Bool(t) => Ok(t),
                        _ => Err(dtype_mismatch("concatenate", first, tensor)),
                    })
                    .collect();
                self.concatenate_bool(&typed?, axis).map(Tensor::Bool)
            }
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
            Tensor::Bool(t) => self.reverse_bool(t, axes).map(Tensor::Bool),
            Tensor::C32(t) => self.reverse_typed(t, axes).map(Tensor::C32),
            Tensor::C64(t) => self.reverse_typed(t, axes).map(Tensor::C64),
        }
    }
}

impl TensorDeviceTransfer for CudaBackend {
    fn download_to_host(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        let tensor = tensor.as_tensor().ok_or_else(|| {
            crate::Error::unsupported(
                "CudaBackend::download_to_host",
                "CUDA transfer currently requires an owned tensor; materialize a view explicitly first",
            )
        })?;
        download_tensor(self.runtime(), tensor)
    }

    fn upload_host_tensor(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        let tensor = tensor.as_tensor().ok_or_else(|| {
            crate::Error::unsupported(
                "CudaBackend::upload_host_tensor",
                "CUDA transfer currently requires an owned tensor; materialize a view explicitly first",
            )
        })?;
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

                fn copy_into(
                    &mut self,
                    src: &TypedTensorView<'_, $ty, R>,
                    dst: &mut TypedTensorViewMut<'_, $ty, R>,
                ) -> crate::Result<()> {
                    self.copy_view_to_view_typed(src, dst, "CudaBackend::copy_into")
                }
            }
        )*
    };
}

macro_rules! impl_cutensor_view_canonicalization {
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
                    self.to_contiguous_view_cutensor_or_cubecl(view, "CudaBackend::to_contiguous")
                }

                fn copy_into(
                    &mut self,
                    src: &TypedTensorView<'_, $ty, R>,
                    dst: &mut TypedTensorViewMut<'_, $ty, R>,
                ) -> crate::Result<()> {
                    self.copy_view_to_view_typed(src, dst, "CudaBackend::copy_into")
                }
            }
        )*
    };
}

impl_cutensor_view_canonicalization!(f32, f64, Complex32, Complex64);
impl_cubecl_view_canonicalization!(i32, i64);

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

    fn copy_into(
        &mut self,
        _src: &TypedTensorView<'_, bool, R>,
        _dst: &mut TypedTensorViewMut<'_, bool, R>,
    ) -> crate::Result<()> {
        Err(unsupported_dtype(
            "CudaBackend::copy_into",
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

    fn execute_broadcast_multiply(
        &mut self,
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<Tensor>> {
        let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (lhs, rhs) else {
            return Ok(None);
        };
        match (lhs, rhs) {
            (Tensor::F32(lhs), Tensor::F32(rhs)) => launch_broadcast_multiply_typed(
                self, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
            .map(Tensor::F32)
            .map(Some),
            (Tensor::F64(lhs), Tensor::F64(rhs)) => launch_broadcast_multiply_typed(
                self, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
            .map(Tensor::F64)
            .map(Some),
            (Tensor::I32(lhs), Tensor::I32(rhs)) => launch_broadcast_multiply_int_typed(
                self, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
            .map(Tensor::I32)
            .map(Some),
            (Tensor::I64(lhs), Tensor::I64(rhs)) => launch_broadcast_multiply_int_typed(
                self, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
            .map(Tensor::I64)
            .map(Some),
            (Tensor::C32(lhs), Tensor::C32(rhs)) => launch_broadcast_multiply_complex_typed(
                self, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
            .map(Tensor::C32)
            .map(Some),
            (Tensor::C64(lhs), Tensor::C64(rhs)) => launch_broadcast_multiply_complex_typed(
                self, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
            .map(Tensor::C64)
            .map(Some),
            (Tensor::Bool(_), Tensor::Bool(_)) => Ok(None),
            _ => Err(dtype_mismatch("broadcast_multiply", lhs, rhs)),
        }
    }
}

impl BackendSession for CudaBackend {
    fn session_type_id(&self) -> TypeId {
        TypeId::of::<CudaBackendSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

impl BackendCachedDot for CudaBackend {}

impl TensorBuffer for CudaBackend {}

impl TensorBackend for CudaBackend {}

fn validate_permutation(op: &'static str, perm: &[usize], rank: usize) -> crate::Result<()> {
    ensure_rank(op, rank, perm.len())?;
    ensure_axes_unique(op, "perm", perm, rank)
}

fn ensure_same_shape_for_broadcast_multiply(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> crate::Result<()> {
    if lhs_shape != rhs_shape {
        return Err(crate::Error::shape_mismatch(
            "broadcast_multiply",
            lhs_shape.to_vec(),
            rhs_shape.to_vec(),
        ));
    }
    Ok(())
}

fn launch_broadcast_multiply_typed<T>(
    backend: &CudaBackend,
    lhs: &TypedTensor<T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensor<T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + TensorScalar + CubePrimitive + CubeFloat + Clone,
{
    ensure_same_shape_for_broadcast_multiply(lhs_shape, rhs_shape)?;
    validate_broadcast_in_dim(lhs.shape(), lhs_shape, lhs_dims)?;
    validate_broadcast_in_dim(rhs.shape(), rhs_shape, rhs_dims)?;
    launch_binary_tensor(
        backend.runtime(),
        lhs,
        rhs,
        lhs_shape,
        "broadcast_multiply",
        |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
            elementwise::broadcast_multiply_float::launch_unchecked::<T, CubeclCudaRuntime>(
                client,
                count,
                dim,
                out.into_tensor_arg(),
                lhs_arg.into_tensor_arg(),
                rhs_arg.into_tensor_arg(),
                comptime_sequence(lhs_dims),
                comptime_sequence(rhs_dims),
                lhs_shape.len(),
            );
        },
    )
}

fn launch_broadcast_multiply_int_typed<T>(
    backend: &CudaBackend,
    lhs: &TypedTensor<T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensor<T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + TensorScalar + CubePrimitive + CubeInt + Clone,
{
    ensure_same_shape_for_broadcast_multiply(lhs_shape, rhs_shape)?;
    validate_broadcast_in_dim(lhs.shape(), lhs_shape, lhs_dims)?;
    validate_broadcast_in_dim(rhs.shape(), rhs_shape, rhs_dims)?;
    launch_binary_tensor(
        backend.runtime(),
        lhs,
        rhs,
        lhs_shape,
        "broadcast_multiply",
        |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
            elementwise::broadcast_multiply_int::launch_unchecked::<T, CubeclCudaRuntime>(
                client,
                count,
                dim,
                out.into_tensor_arg(),
                lhs_arg.into_tensor_arg(),
                rhs_arg.into_tensor_arg(),
                comptime_sequence(lhs_dims),
                comptime_sequence(rhs_dims),
                lhs_shape.len(),
            );
        },
    )
}

fn launch_broadcast_multiply_complex_typed<T>(
    backend: &CudaBackend,
    lhs: &TypedTensor<T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensor<T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + TensorScalar + CubePrimitive + CubeComplex + Clone,
{
    ensure_same_shape_for_broadcast_multiply(lhs_shape, rhs_shape)?;
    validate_broadcast_in_dim(lhs.shape(), lhs_shape, lhs_dims)?;
    validate_broadcast_in_dim(rhs.shape(), rhs_shape, rhs_dims)?;
    launch_binary_tensor(
        backend.runtime(),
        lhs,
        rhs,
        lhs_shape,
        "broadcast_multiply",
        |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
            elementwise::broadcast_multiply_complex::launch_unchecked::<T, CubeclCudaRuntime>(
                client,
                count,
                dim,
                out.into_tensor_arg(),
                lhs_arg.into_tensor_arg(),
                rhs_arg.into_tensor_arg(),
                comptime_sequence(lhs_dims),
                comptime_sequence(rhs_dims),
                lhs_shape.len(),
            );
        },
    )
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
            return Err(crate::Error::duplicate_axis(
                "broadcast_in_dim",
                dst_axis,
                "dims",
            ));
        }
        seen[dst_axis] = true;
        let src = input_shape[src_axis];
        let dst = shape[dst_axis];
        if src != dst && src != 1 {
            return Err(crate::Error::shape_mismatch(
                "broadcast_in_dim",
                input_shape.to_vec(),
                shape.to_vec(),
            ));
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
        return Err(crate::Error::duplicate_axis(
            "extract_diagonal",
            axis_a,
            "axes",
        ));
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
        return Err(crate::Error::axis_out_of_bounds(
            "embed_diagonal",
            axis_b,
            input_shape.len(),
        ));
    }
    let mut output_shape = input_shape.to_vec();
    output_shape.insert(axis_b, input_shape[axis_a]);
    Ok(output_shape)
}

fn reduction_output_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (!axes.contains(&axis)).then_some(dim))
        .collect()
}

fn reduction_keepdims_shape(input_shape: &[usize], axis: usize) -> Vec<usize> {
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = 1;
    output_shape
}

fn cubecl_reshape_metadata<T: crate::TensorScalar + Clone>(
    tensor: TypedTensor<T>,
    shape: Vec<usize>,
    op: &'static str,
) -> crate::Result<TypedTensor<T>> {
    let len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("shape product overflow for CubeCL reshape shape {shape:?}"),
            )
        })?;
    let tensor_len = tensor.n_elements();
    if len != tensor_len {
        return Err(crate::Error::validation(
            op,
            tenferro_tensor::ShapeMismatch::ReshapeElementCount {
                from: tensor_len,
                to: len,
            }
            .into(),
        ));
    }

    // `TypedTensor::into_parts` intentionally materializes host storage and
    // therefore cannot preserve a backend-owned root. Move the owner through
    // `TensorValue`/`AllocationGroup` instead so this metadata-only reshape
    // keeps the exact CubeCL allocation and performs no implicit download.
    let value = crate::TensorValue::from_tensor(
        <T as crate::TensorScalar>::typed_tensor_into_tensor(tensor),
    )
    .reshape_view(shape)?;
    let (group, slot, _, _) = value.try_into_group_parts().map_err(|_| {
        crate::Error::runtime_state(
            op,
            "failed to publish the reshaped tensor descriptor without copying",
        )
    })?;
    let tensor = group.into_tensor(slot).map_err(|(_, error)| {
        crate::Error::runtime_state(
            op,
            format!("failed to detach the reshaped tensor owner: {error}"),
        )
    })?;
    <T as crate::TensorScalar>::into_typed(tensor)
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
                return Err(crate::Error::invalid_argument(
                    "slice",
                    "bounds",
                    format!("start exceeds limit on axis {axis}"),
                ));
            }
            // INVARIANT: This boundary check intentionally mirrors CPU's
            // validator. CPU and GPU are independent backend leaves, and
            // sharing it via tenferro-tensor would require a new public
            // validation API.
            if limit > dim {
                return Err(crate::Error::invalid_argument(
                    "slice",
                    "configuration",
                    format!("limit {limit} on axis {axis} exceeds dimension size {dim}"),
                ));
            }
            if stride == 0 {
                return Err(crate::Error::invalid_argument(
                    "slice",
                    "strides",
                    format!("stride must be positive on axis {axis}"),
                ));
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
    for (axis, &input_dim_raw) in input_shape.iter().enumerate().take(rank) {
        if config.interior_padding[axis] < 0 {
            return Err(crate::Error::invalid_argument(
                "pad",
                "interior_padding",
                format!("interior padding must be non-negative on axis {axis}"),
            ));
        }
        let input_dim = i64::try_from(input_dim_raw).map_err(|_| {
            crate::Error::invalid_argument(
                "pad",
                "input_shape",
                format!("input dimension on axis {axis} must fit in i64"),
            )
        })?;
        let base = if input_dim == 0 {
            0
        } else {
            let spacing = config.interior_padding[axis]
                .checked_add(1)
                .ok_or_else(|| {
                    crate::Error::invalid_argument(
                        "pad",
                        "interior_padding",
                        format!("interior padding overflow on axis {axis}"),
                    )
                })?;
            input_dim
                .checked_sub(1)
                .and_then(|extent| extent.checked_mul(spacing))
                .and_then(|extent| extent.checked_add(1))
                .ok_or_else(|| {
                    crate::Error::invalid_argument(
                        "pad",
                        "interior_padding",
                        format!("padded interior extent overflow on axis {axis}"),
                    )
                })?
        };
        let dim = config.edge_padding_low[axis]
            .checked_add(config.edge_padding_high[axis])
            .and_then(|edge| edge.checked_add(base))
            .ok_or_else(|| {
                crate::Error::invalid_argument(
                    "pad",
                    "padding",
                    format!("output dimension overflow on axis {axis}"),
                )
            })?;
        out_shape.push(usize::try_from(dim).map_err(|_| {
            crate::Error::invalid_argument(
                "pad",
                "padding",
                format!("negative output dimension on axis {axis}"),
            )
        })?);
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
            return Err(crate::Error::invalid_argument(
                op,
                "slice_sizes",
                format!("slice_sizes[{axis}]={slice_size} exceeds operand dimension {dim_size}"),
            ));
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
        return Err(crate::Error::axis_out_of_bounds(
            "gather",
            config.index_vector_dim,
            start_indices_shape.len(),
        ));
    }
    let index_size = index_vector_size(start_indices_shape, config.index_vector_dim);
    if index_size != config.start_index_map.len() {
        return Err(crate::Error::invalid_argument(
            "gather",
            "start_index_map",
            "start_index_map length mismatch",
        ));
    }
    ensure_axes_unique(
        "gather",
        "collapsed_slice_dims",
        &config.collapsed_slice_dims,
        operand_shape.len(),
    )?;
    for &dim in &config.collapsed_slice_dims {
        if config.slice_sizes[dim] != 1 {
            return Err(crate::Error::invalid_argument(
                "gather",
                "collapsed_slice_dims",
                format!(
                    "collapsed slice dimension {dim} must have slice_size == 1, got {}",
                    config.slice_sizes[dim]
                ),
            ));
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
        return Err(crate::Error::invalid_argument(
            "gather",
            "offset_dims",
            "offset_dims length mismatch",
        ));
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
        return Err(crate::Error::axis_out_of_bounds(
            "scatter",
            config.index_vector_dim,
            scatter_indices_shape.len(),
        ));
    }
    let index_size = index_vector_size(scatter_indices_shape, config.index_vector_dim);
    if index_size != config.scatter_dims_to_operand_dims.len() {
        return Err(crate::Error::invalid_argument(
            "scatter",
            "scatter_dims_to_operand_dims",
            "scatter_dims_to_operand_dims length mismatch",
        ));
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
        return Err(crate::Error::invalid_argument(
            "scatter",
            "update_window_dims",
            "update_window_dims length mismatch",
        ));
    }
    let updates_batch_rank = updates_shape.len() - config.update_window_dims.len();
    if updates_batch_rank != batch_shape.len() {
        return Err(crate::Error::rank_mismatch(
            "scatter",
            batch_shape.len(),
            updates_batch_rank,
        ));
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
            return Err(crate::Error::shape_mismatch(
                "scatter",
                vec![expected],
                vec![actual],
            ));
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
                    crate::Error::invalid_argument(
                        "concatenate",
                        "shape",
                        "concatenate axis extent overflows usize",
                    )
                })?;
            } else if input.shape()[dim] != first.shape()[dim] {
                return Err(crate::Error::shape_mismatch(
                    "concatenate",
                    first.shape().to_vec(),
                    input.shape().to_vec(),
                ));
            }
        }
    }
    out_shape[axis] = axis_extent;
    Ok(out_shape)
}

#[cfg(test)]
mod tests;
