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
use std::sync::atomic::{AtomicU64, Ordering};
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
use tenferro_tensor::{
    ContractionScalar, DType, DotGeneralAccumulation, ElementwiseReadOp, TensorRead, TensorWrite,
};

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
    NativePermutationKind, NativePermutationPlan, NativeStridedCopyPlan, NativeTransposeTile,
};
use crate::{
    DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, StorageBuffer, Tensor, TensorRank,
    TensorScalar, TensorView, TensorViewCanonicalization, TensorViewMut, TypedTensor,
    TypedTensorView, TypedTensorViewMut,
};

/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
mod blas1;
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
mod plan_cache;
pub(crate) mod raw;
mod runtime;
mod runtime_adapter;
pub(crate) mod session_cubecl;
mod workspace_retirement;

pub use workspace_retirement::WorkspaceRetirementStats;

pub use gemm::CutensorWorkspaceStats;

use dispatch::{
    alloc_bool_output, alloc_output, bool_tensor_array_arg, comptime_sequence, cube_count_for_len,
    cube_dim_1d, dtype_mismatch, ensure_axes_unique, ensure_axis, ensure_rank,
    ensure_resident_on_runtime, ensure_view_mut_resident_on_runtime,
    ensure_view_resident_on_runtime, launch_binary, launch_binary_bool_tensor,
    launch_binary_tensor, launch_bool_tensor_into, launch_compare_bool, launch_nullary_bool_into,
    launch_nullary_into, launch_select_bool, launch_ternary, launch_unary,
    launch_unary_bool_tensor, launch_unary_tensor, launch_unary_tensor_into,
    ternary_dtype_mismatch, typed_tensor_array_arg, typed_tensor_array_arg_as,
    typed_tensor_binding, typed_tensor_mut_array_arg, typed_view_array_arg, typed_view_binding,
    typed_view_mut_array_arg,
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
            if let Some((cubes_x, cubes_y, cubes_z)) = config.dispatch_grid(
                op,
                plan.dims[0],
                plan.dims[1],
                plan.dims.get(2).copied().unwrap_or(1),
                65_535,
            )? {
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
    // Backend-level so the configured cap survives clearing or evicting the
    // extension-cache entry that owns the shared scratch pool itself.
    cutensor_workspace_max_retained_bytes: AtomicU64,
    // Backend-level for the same reason, and cumulative across cache clears:
    // it is a diagnostic for whether the cap is set below the workload's real
    // requirement, not a per-cache statistic.
    cutensor_workspace_temporary_uses: AtomicU64,
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

/// Default cap on retained shared cuTENSOR contraction scratch, in bytes.
///
/// This bounds only the scratch the backend keeps for reuse, not total device
/// memory: a contraction whose requirement exceeds the remaining cap still runs
/// in a temporary workspace. It is deliberately permissive, because there is no
/// single optimal cap across workloads and the cap never affects correctness;
/// callers that need to bound retained device memory configure a smaller value.
/// On a device with less free memory than this the cap simply never binds.
const DEFAULT_CUTENSOR_WORKSPACE_MAX_RETAINED_BYTES: u64 = 10 << 30;

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
            if inner.retained_bytes > inner.max_retained_bytes.get() {
                inner.entries.remove(&type_id);
                inner.order.retain(|candidate| *candidate != type_id);
                inner.stats.evictions = inner.stats.evictions.saturating_add(1);
                inner.refresh_retained_bytes();
            }
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
                cutensor_workspace_max_retained_bytes: AtomicU64::new(
                    DEFAULT_CUTENSOR_WORKSPACE_MAX_RETAINED_BYTES,
                ),
                cutensor_workspace_temporary_uses: AtomicU64::new(0),
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
        let handle =
            ffi::cutensor::CutensorHandle::load(self.inner.rt.device_info().compute_capability())?;
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
    /// bytes cover plan metadata, not the shared per-stream device scratch;
    /// [`CudaBackend::cutensor_workspace_stats`] reports that separately. The
    /// cache byte limit is therefore not a total device-memory limit.
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn cutensor_plan_cache_stats(&self) -> crate::Result<CacheStats> {
        gemm::cutensor_plan_cache_stats(self)
    }

    /// Return the retained shared cuTENSOR contraction scratch, in bytes.
    ///
    /// All cached cuTENSOR contraction plans share one lazily grown workspace
    /// per physical stream slot, so this is the sum over slots of the capacity
    /// each slot currently holds. It is bounded by
    /// [`CudaBackend::set_cutensor_workspace_max_retained_bytes`], reported
    /// separately from the extension-cache byte statistics, and released by
    /// clearing the extension cache or dropping the backend. This is retained
    /// scratch, not total device memory: a workspace in use by a queued
    /// contraction, a retiring allocation, and vendor-internal memory are all
    /// excluded.
    ///
    /// Read this to size a retention cap: it is the high-water demand of the
    /// workload shapes that have run so far.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // `gpu_available` never panics without a CUDA driver, so this
    /// // example also runs in CPU-only doctest environments.
    /// if gpu_available() {
    ///     let device = cuda_devices()?.remove(0);
    ///     let backend = CudaBackend::new(device.id())?;
    ///     let stats = backend.cutensor_workspace_stats()?;
    ///     println!("{:?}", (stats.retained_entries, stats.retained_bytes));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn cutensor_workspace_stats(&self) -> crate::Result<CutensorWorkspaceStats> {
        gemm::cutensor_workspace_stats(self)
    }

    /// Return the device bytes retained by the shared cuTENSOR contraction
    /// scratch. Equal to
    /// [`CudaBackend::cutensor_workspace_stats`]`().retained_bytes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // `gpu_available` never panics without a CUDA driver, so this
    /// // example also runs in CPU-only doctest environments.
    /// if gpu_available() {
    ///     let device = cuda_devices()?.remove(0);
    ///     let backend = CudaBackend::new(device.id())?;
    ///     println!("{} bytes retained", backend.cutensor_workspace_bytes()?);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the cache mutex is poisoned.
    pub fn cutensor_workspace_bytes(&self) -> crate::Result<u64> {
        Ok(self.cutensor_workspace_stats()?.retained_bytes)
    }

    /// Return the configured retention cap for shared cuTENSOR contraction
    /// scratch, in bytes.
    ///
    /// The default is 10 GiB. See
    /// [`CudaBackend::set_cutensor_workspace_max_retained_bytes`] for the
    /// contract; this value is not a device-memory reservation. The cap is
    /// plain backend state, so reading it cannot fail and never creates cache
    /// state.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // `gpu_available` never panics without a CUDA driver, so this
    /// // example also runs in CPU-only doctest environments.
    /// if gpu_available() {
    ///     let device = cuda_devices()?.remove(0);
    ///     let backend = CudaBackend::new(device.id())?;
    ///     println!("cap {} bytes", backend.cutensor_workspace_max_retained_bytes());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn cutensor_workspace_max_retained_bytes(&self) -> u64 {
        self.cutensor_workspace_limit()
    }

    /// Configure the retention cap for shared cuTENSOR contraction scratch.
    ///
    /// The cap bounds how much scratch the backend keeps for reuse, summed over
    /// physical stream slots. It never refuses a contraction: a requirement that
    /// does not fit the remaining cap runs in a temporary workspace that is
    /// released afterwards, and shrinking the cap drops retained buffers
    /// without evicting any cached plan. Other slots are never evicted to make
    /// room.
    ///
    /// `0` disables retention entirely; it is not "unlimited". The default
    /// (10 GiB) is finite but is not a practical memory protection, and neither
    /// the cap nor the reported statistics bound total device memory.
    ///
    /// Setting a cap below the steady-state working set makes matching
    /// contractions allocate and retire their scratch on every call, which can
    /// increase workspace-retirement stream barrier fallbacks. To choose a
    /// value, run the workload and read the
    /// [`CudaBackend::cutensor_workspace_bytes`] high-water: retaining every
    /// slot's rounded high-water needs a cap of at least the sum of
    /// `next_power_of_two(max(request, 1 MiB))` over the stream slots.
    ///
    /// The setting is stored on the backend, so it survives
    /// `CudaBackend::clear_cuda_extension_cache` and extension-cache eviction,
    /// and is shared by clones of this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // `gpu_available` never panics without a CUDA driver, so this
    /// // example also runs in CPU-only doctest environments.
    /// if gpu_available() {
    ///     let device = cuda_devices()?.remove(0);
    ///     let backend = CudaBackend::new(device.id())?;
    ///     backend.set_cutensor_workspace_max_retained_bytes(4 << 30)?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the plan-cache mutex is
    /// poisoned while releasing retained buffers.
    pub fn set_cutensor_workspace_max_retained_bytes(&self, bytes: u64) -> crate::Result<()> {
        self.inner
            .cutensor_workspace_max_retained_bytes
            .store(bytes, Ordering::Relaxed);
        gemm::set_cutensor_workspace_max_retained_bytes(self, bytes)
    }

    /// Current retention cap for shared cuTENSOR contraction scratch.
    fn cutensor_workspace_limit(&self) -> u64 {
        self.inner
            .cutensor_workspace_max_retained_bytes
            .load(Ordering::Relaxed)
    }

    /// Return how many contractions ran in a temporary shared-scratch
    /// workspace because their requirement did not fit the retention cap.
    ///
    /// This is the direct signal that the cap is binding. A nonzero value means
    /// the matching contractions allocated and retired their scratch on every
    /// call instead of reusing a retained buffer; the high-water from
    /// [`CudaBackend::cutensor_workspace_bytes`] then under-reports the real
    /// requirement. Raise
    /// [`CudaBackend::set_cutensor_workspace_max_retained_bytes`] until this
    /// stops increasing, or accept the churn deliberately.
    ///
    /// The count is cumulative for the backend, shared by clones, and is not
    /// reset by `CudaBackend::clear_cuda_extension_cache`; diff two reads to
    /// measure an interval. Reading it cannot fail and never creates cache
    /// state.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // `gpu_available` never panics without a CUDA driver, so this
    /// // example also runs in CPU-only doctest environments.
    /// if gpu_available() {
    ///     let device = cuda_devices()?.remove(0);
    ///     let backend = CudaBackend::new(device.id())?;
    ///     println!("{} temporary uses", backend.cutensor_workspace_temporary_uses());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn cutensor_workspace_temporary_uses(&self) -> u64 {
        self.inner
            .cutensor_workspace_temporary_uses
            .load(Ordering::Relaxed)
    }

    /// Record one cap-driven temporary workspace use.
    fn note_cutensor_temporary_workspace(&self) {
        self.inner
            .cutensor_workspace_temporary_uses
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Return deferred cuTENSOR workspace retirement counters.
    ///
    /// Retirements are deferred until the workspace's stream reaches the event
    /// recorded at retirement time. `in_flight` is the number of workspaces
    /// whose handle has not returned to the CubeCL pool yet.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if the retirement queue lock is
    /// poisoned.
    pub fn cutensor_workspace_retirement_stats(&self) -> crate::Result<WorkspaceRetirementStats> {
        gemm::cutensor_workspace_retirement_stats(self)
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
        if view.strides().iter().any(|&stride| stride <= 0) {
            // cuTENSOR 2.x rejects zero/negative-stride tensor descriptors. This
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
        let source_allocation_len = source_buffer.len();
        let destination_allocation_len = destination_buffer.len();
        if let Some(plan) = self.transpose_copy_plan(src, dst, op)? {
            let dst_arg = typed_view_mut_array_arg(dst, op)?;
            let src_arg = typed_view_array_arg(src, op)?;
            return launch_native_materialization::<T>(self, dst_arg, src_arg, &plan, op);
        }
        // Both operands keep their own strides and offsets: a region inside a
        // larger allocation is read and written in place instead of being
        // canonicalized into scratch first. Axis fusion collapses the affine
        // runs, so a compact sub-block still costs one flat pass.
        let plan = NativeStridedCopyPlan::new(
            op,
            src.shape(),
            src.strides(),
            src.offset(),
            source_allocation_len,
            dst.strides(),
            dst.offset(),
            destination_allocation_len,
            false,
        )?;
        // A fused plan that reduces to one matrix transpose is exactly what the
        // tiled transpose kernel implements, and that kernel is coalesced on
        // both operands, so take it before the flat kernels. A flat pass over a
        // multi-axis permutation reads one contiguous run per source coordinate
        // and scatters one element run per destination coordinate, which is why
        // the 1 GiB class of copies stays on the generic kernel (issue #1891).
        if dst.offset() == 0 && self.launch_tiled_transpose(&plan, src, dst, op)? {
            return Ok(());
        }
        if src.offset() == 0 && src.is_col_major_contiguous()? {
            let strides = view_strides_i64(dst.strides(), op)?;
            let base_offset = view_offset_i64(dst.offset(), op)?;
            let src_arg = typed_view_binding(src, op)?;
            let dst_arg = typed_view_mut_array_arg(dst, op)?;
            let rank = dst.shape().len();
            unsafe {
                // SAFETY: The source is a compact zero-offset CubeCL view on
                // this runtime. Allocation identity validation above proves
                // source and destination do not alias. The destination view
                // has validated reachable offsets and no internal overlap, and
                // the launch domain covers each source element and destination
                // logical coordinate exactly once.
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
            return Ok(());
        }
        let src_strides = view_strides_i64(&plan.src_strides, op)?;
        let dst_strides = view_strides_i64(&plan.dst_strides, op)?;
        let src_offset = view_offset_i64(plan.src_offset, op)?;
        let dst_offset = view_offset_i64(plan.dst_offset, op)?;
        let rank = plan.dims.len();
        let src_arg = typed_view_array_arg(src, op)?;
        let dst_arg = typed_view_mut_array_arg(dst, op)?;
        unsafe {
            // SAFETY: Both array bindings cover their whole root allocation,
            // and `NativeStridedCopyPlan` proved every logical coordinate maps
            // inside the source and destination allocation spans, that the
            // destination is injective, and (with the allocation identity
            // check above) that the two allocations are distinct. The launch
            // domain visits each logical coordinate exactly once.
            structural::strided_to_strided_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                self.runtime().client(),
                cube_count_for_len(plan.len)?,
                cube_dim_1d(),
                dst_arg,
                src_arg,
                comptime_sequence(&plan.dims),
                comptime_sequence(&src_strides),
                comptime_sequence(&dst_strides),
                src_offset,
                dst_offset,
                plan.len,
                rank,
            );
        }
        Ok(())
    }

    /// Launch the tiled transpose kernel for a fused plan that it implements.
    ///
    /// Returns `Ok(false)` when the layout is outside that kernel's contract or
    /// the tiled configuration is disabled, so every other copy keeps its
    /// current kernel.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] when the metadata exceeds the
    /// kernel's launch limits.
    fn launch_tiled_transpose<T, R>(
        &self,
        plan: &NativeStridedCopyPlan,
        src: &TypedTensorView<'_, T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
        op: &'static str,
    ) -> crate::Result<bool>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
        R: TensorRank,
    {
        let Some((dst_fast_extent, src_fast_extent)) = plan.tiled_transpose_matrix() else {
            return Ok(false);
        };
        let Some(config) = NativeTransposeTile::selected(op)? else {
            return Ok(false);
        };
        // A wide matrix needs a wider tile to stay inside the per-dimension
        // launch limit: an 1048576-element axis is 65536 blocks at the default
        // 16-wide tile, one past the limit. Widening the tile keeps the tiled
        // kernel available instead of falling back to a flat pass, and the
        // shared-memory budget bounds how far it can grow.
        const MAX_SHARED_BYTES: usize = 48 * 1024;
        let element_bytes = std::mem::size_of::<T>();
        let mut launched = None;
        for tile in [config.tile, 32] {
            let candidate = config.with_tile(tile);
            if candidate.shared_bytes(element_bytes) > MAX_SHARED_BYTES {
                continue;
            }
            if let Some(grid) =
                candidate.dispatch_grid(op, dst_fast_extent, src_fast_extent, 1, 65_535)?
            {
                launched = Some((candidate, grid));
                break;
            }
        }
        let Some((config, (cubes_x, cubes_y, cubes_z))) = launched else {
            return Ok(false);
        };
        let batch_stride = dst_fast_extent
            .checked_mul(src_fast_extent)
            .ok_or_else(|| {
                crate::Error::invalid_argument(
                    op,
                    "shape",
                    "tiled transpose matrix extent overflows usize",
                )
            })?;
        let src_offset = usize::try_from(plan.src_offset).map_err(|_| {
            crate::Error::invalid_argument(
                op,
                "offset",
                "tiled transpose requires a non-negative source offset",
            )
        })?;
        let dst_arg = typed_view_mut_array_arg(dst, op)?;
        let src_arg = typed_view_array_arg(src, op)?;
        unsafe {
            // SAFETY: `NativeStridedCopyPlan` proved every logical coordinate
            // maps inside both allocation spans, that the destination is
            // injective, and that the two allocations are distinct. The
            // orientation check proves the source is row-major and the
            // destination column-major over the same matrix, which is the
            // kernel's indexing contract; its bounds guards cover edge tiles
            // and every unit reaches the shared-memory barrier.
            structural::tiled_transpose_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                self.runtime().client(),
                CubeCount::Static(cubes_x, cubes_y, cubes_z),
                CubeDim::new_2d(config.tile / config.vector_width, config.block_rows),
                dst_arg,
                src_arg,
                src_offset,
                batch_stride,
                dst_fast_extent,
                src_fast_extent,
                config.tile as usize,
                config.block_rows as usize,
                config.padding as usize,
                config.vector_width as usize,
            );
        }
        Ok(true)
    }

    /// Build a tiled-transpose plan for a copy whose destination is a
    /// row-major-compact view at offset zero.
    ///
    /// Copying from a compact column-major source into a row-major compact
    /// destination of the same logical shape writes exactly the same physical
    /// bytes as materializing the transposed source into a compact
    /// column-major destination. Selecting that plan keeps the value-exact
    /// native path while replacing the uncoalesced access pattern of
    /// `contiguous_to_view_kernel` with the existing tiled transpose kernel.
    ///
    /// Returns `Ok(None)` whenever the layout is outside that narrow shape, so
    /// every other copy keeps its current kernel.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] when the transposed plan is rejected
    /// by bounds, stride-overflow, or destination-overlap validation; the
    /// remaining copy kernels would reject the same layouts.
    fn transpose_copy_plan<T, R>(
        &self,
        src: &TypedTensorView<'_, T, R>,
        dst: &TypedTensorViewMut<'_, T, R>,
        op: &'static str,
    ) -> crate::Result<Option<NativePermutationPlan>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone + Send + Sync + 'static,
        R: TensorRank,
    {
        let shape = dst.shape();
        if !(2..=3).contains(&shape.len()) || dst.offset() != 0 {
            return Ok(None);
        }
        // The transpose kernel writes a compact column-major region starting at
        // the bound allocation, so the destination view must be exactly that
        // address range: row-major compact at offset zero.
        let mut row_major_compact = vec![1isize; shape.len()];
        for axis in (0..shape.len() - 1).rev() {
            let extent = isize::try_from(shape[axis + 1]).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "shape",
                    "row-major stride extent exceeds the isize metadata limit",
                )
            })?;
            row_major_compact[axis] =
                row_major_compact[axis + 1]
                    .checked_mul(extent)
                    .ok_or_else(|| {
                        crate::Error::invalid_argument(
                            op,
                            "shape",
                            "row-major stride product overflow in the copy destination",
                        )
                    })?;
        }
        if dst.strides() != row_major_compact {
            return Ok(None);
        }
        let source_allocation_len = src
            .backend_buffer()
            .map(|buffer| buffer.len())
            .ok_or_else(|| crate::Error::runtime_state(op, "expected a CUDA source view"))?;
        let destination_allocation_len = dst
            .backend_buffer()
            .map(|buffer| buffer.len())
            .ok_or_else(|| crate::Error::runtime_state(op, "expected a CUDA destination view"))?;
        let permutation: Vec<usize> = (0..shape.len()).rev().collect();
        let plan = NativePermutationPlan::for_transpose(
            op,
            src.shape(),
            src.strides(),
            &permutation,
            src.offset(),
            source_allocation_len,
            destination_allocation_len,
            false,
        )?;
        if plan.kind != NativePermutationKind::TiledTranspose {
            return Ok(None);
        }
        Ok(Some(plan))
    }

    /// Copy through the cuTENSOR permutation executor when the layout supports
    /// it, otherwise through the exact native copy.
    ///
    /// Only real dtypes use this: for `F32`/`F64` the vendor `alpha = 1`
    /// multiply is exact, and cuTENSOR is markedly faster than the native
    /// kernel for a multi-axis permutation destination. Complex dtypes are
    /// routed to the native copy instead, because for them the same multiply
    /// turns a finite component into `NaN` (issue #1891).
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
        // cuTENSOR 2.x descriptors require positive strides on every operand,
        // so reversed and broadcast views stay on the native kernel. This is a
        // layout the vendor permutation path cannot represent, not a
        // missing-library fallback.
        if dst.strides().iter().any(|&stride| stride < 0)
            || src.strides().iter().any(|&stride| stride < 1)
        {
            return self.copy_view_to_view_typed(src, dst, op);
        }
        // Complex operands are exact through cuTENSOR only via the real view,
        // whose unit-stride run is the 16-byte real/imaginary pair; that caps
        // an exact multi-axis permutation at a third of the achievable
        // bandwidth (issue #1891). When the copy is one tiled 2D transpose the
        // native kernel is exact as well and coalesced on both sides, so prefer
        // it and keep the vendor plan for every other layout.
        if T::REAL_VIEW
            && dst.offset() == 0
            && NativeStridedCopyPlan::new(
                op,
                src.shape(),
                src.strides(),
                src.offset(),
                src.backend_buffer().map_or(0, |buffer| buffer.len()),
                dst.strides(),
                dst.offset(),
                dst.backend_buffer().map_or(0, |buffer| buffer.len()),
                false,
            )?
            .tiled_transpose_matrix()
            .is_some()
        {
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

        // The first reduction reads the caller-owned input directly. Subsequent
        // axes consume the fresh keepdims result from the preceding launch.
        let (first_axis, remaining_axes) = sorted_axes
            .split_first()
            .ok_or_else(|| crate::Error::invalid_argument(op, "axes", "axes must not be empty"))?;
        let mut current = launch_axis(self, input, *first_axis)?;
        for &axis in remaining_axes {
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

/// The typed tensor a tag names, or a typed error if the payload does not carry it.
///
/// Dispatch on [`Tensor::dtype`] and this accessor are the pair that lets a GPU operation be written
/// against the tag rather than against every `Tensor` variant. The error is reachable only if a tag
/// and its payload ever disagree, which the tag itself excludes, so it is a typed refusal rather than
/// a panic.
fn typed_or_unsupported<'a, T: tenferro_tensor::TensorScalar>(
    tensor: &'a Tensor,
    op: &'static str,
) -> crate::Result<&'a tenferro_tensor::TypedTensor<T>> {
    tensor.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(op, "the tensor does not carry the scalar its tag names")
    })
}

fn promoted_real_complex_scalar_binary(
    backend: &CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    op: &'static str,
    mode: usize,
) -> Option<crate::Result<Tensor>> {
    // Dispatch on the pair of tags and recover each typed tensor, which is what `as_typed` exists
    // for; the closure keeps the typed error inside the `Option` the caller expects.
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::C32) if lhs.shape().is_empty() => Some((|| {
            let real = typed_or_unsupported::<f32>(lhs, op)?;
            let complex = typed_or_unsupported::<Complex32>(rhs, op)?;
            launch_real_complex_scalar_binary(backend, real, complex, op, true, mode)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        })()),
        (DType::C32, DType::F32) if rhs.shape().is_empty() => Some((|| {
            let complex = typed_or_unsupported::<Complex32>(lhs, op)?;
            let real = typed_or_unsupported::<f32>(rhs, op)?;
            launch_real_complex_scalar_binary(backend, real, complex, op, false, mode)
                .map(Tensor::from_typed::<num_complex::Complex32>)
        })()),
        (DType::F64, DType::C64) if lhs.shape().is_empty() => Some((|| {
            let real = typed_or_unsupported::<f64>(lhs, op)?;
            let complex = typed_or_unsupported::<Complex64>(rhs, op)?;
            launch_real_complex_scalar_binary(backend, real, complex, op, true, mode)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        })()),
        (DType::C64, DType::F64) if rhs.shape().is_empty() => Some((|| {
            let complex = typed_or_unsupported::<Complex64>(lhs, op)?;
            let real = typed_or_unsupported::<f64>(rhs, op)?;
            launch_real_complex_scalar_binary(backend, real, complex, op, false, mode)
                .map(Tensor::from_typed::<num_complex::Complex64>)
        })()),
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

#[derive(Clone, Copy)]
enum UnaryReadOp {
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Expm1,
    Log1p,
}

/// Operand accepted by a CUDA `_read` entry point.
///
/// The traced runtime prepares operands as `TensorRead`, which is either an
/// owned tensor or a borrowed view over another tensor's storage. Operations
/// without a native view kernel retain the explicit materialization fallback.
enum CudaReadInput<'a> {
    /// The caller owns the tensor for the duration of the call.
    Borrowed(&'a Tensor),
    /// A borrowed view materialized into backend storage for the call.
    Materialized(Box<Tensor>),
}

impl CudaReadInput<'_> {
    fn as_tensor(&self) -> &Tensor {
        match self {
            Self::Borrowed(tensor) => tensor,
            Self::Materialized(tensor) => tensor,
        }
    }
}

fn launch_elementwise_binary_into<T>(
    backend: &CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    out: &mut Tensor,
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<()>
where
    T: CubeElement + TensorScalar + Clone,
{
    let lhs = lhs.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(op, "the GPU dispatch requires a preset scalar")
    })?;
    let rhs = rhs.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(op, "the GPU dispatch requires a preset scalar")
    })?;
    let out = out.as_typed_mut::<T>().ok_or_else(|| {
        crate::Error::unsupported(op, "the GPU dispatch requires a preset scalar")
    })?;
    ensure_resident_on_runtime(backend.runtime(), lhs, op)?;
    ensure_resident_on_runtime(backend.runtime(), rhs, op)?;
    let lhs_arg = typed_tensor_array_arg(lhs, op)?;
    let rhs_arg = typed_tensor_array_arg(rhs, op)?;
    let out_len = out.n_elements();
    ensure_resident_on_runtime(backend.runtime(), out, op)?;
    let out_arg = typed_tensor_mut_array_arg(out, op)?;
    if out_len == 0 {
        return Ok(());
    }
    launch(
        backend.runtime().client(),
        cube_count_for_len(out_len)?,
        cube_dim_1d(),
        out_arg,
        lhs_arg,
        rhs_arg,
    );
    Ok(())
}

fn launch_elementwise_unary_into<T>(
    backend: &CudaBackend,
    input: &Tensor,
    out: &mut Tensor,
    op: &'static str,
    launch: impl FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
) -> crate::Result<()>
where
    T: CubeElement + TensorScalar + Clone,
{
    let input = input.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(op, "the GPU dispatch requires a preset scalar")
    })?;
    let out = out.as_typed_mut::<T>().ok_or_else(|| {
        crate::Error::unsupported(op, "the GPU dispatch requires a preset scalar")
    })?;
    ensure_resident_on_runtime(backend.runtime(), input, op)?;
    let input_arg = typed_tensor_array_arg(input, op)?;
    let out_len = out.n_elements();
    ensure_resident_on_runtime(backend.runtime(), out, op)?;
    let out_arg = typed_tensor_mut_array_arg(out, op)?;
    if out_len == 0 {
        return Ok(());
    }
    launch(
        backend.runtime().client(),
        cube_count_for_len(out_len)?,
        cube_dim_1d(),
        out_arg,
        input_arg,
    );
    Ok(())
}

impl CudaBackend {
    /// Run the common same-shape elementwise cases directly into the caller's
    /// owned CUDA output. Views and broadcast/scalar cases retain the existing
    /// allocating fallback so their layout and ownership contracts are unchanged.
    fn elementwise_read_into_native(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: &mut TensorWrite<'_>,
    ) -> Option<crate::Result<()>> {
        if inputs.len() != op.arity() || inputs.iter().any(|input| input.as_tensor().is_none()) {
            return None;
        }
        let TensorWrite::Tensor(output) = out else {
            return None;
        };
        if inputs
            .iter()
            .any(|input| input.shape() != output.shape() || input.dtype() != output.dtype())
        {
            return None;
        }
        let first = inputs.first().and_then(|input| input.as_tensor())?;
        let second = inputs.get(1).and_then(|input| input.as_tensor());
        let dtype = output.dtype();

        macro_rules! binary {
            ($ty:ty, $kernel:ident) => {
                launch_elementwise_binary_into::<$ty>(
                    self,
                    first,
                    second?,
                    output,
                    op.label(),
                    // SAFETY: native dispatch checks matching owned compact shapes/dtypes;
                    // validate_read_into_destination checks overlap, and the launch helper
                    // validates runtime residency and prepares the exclusive output write.
                    |client, count, dim, out, lhs, rhs| unsafe {
                        elementwise::$kernel::launch_unchecked::<$ty, CubeclCudaRuntime>(
                            client, count, dim, out, lhs, rhs,
                        );
                    },
                )
            };
        }
        macro_rules! unary {
            ($ty:ty, $kernel:ident) => {
                launch_elementwise_unary_into::<$ty>(
                    self,
                    first,
                    output,
                    op.label(),
                    // SAFETY: native dispatch checks matching owned compact shapes/dtypes;
                    // validate_read_into_destination checks overlap, and the launch helper
                    // validates runtime residency and prepares the exclusive output write.
                    |client, count, dim, out, input| unsafe {
                        elementwise::$kernel::launch_unchecked::<$ty, CubeclCudaRuntime>(
                            client, count, dim, out, input,
                        );
                    },
                )
            };
        }

        let result = match op {
            ElementwiseReadOp::Add => match dtype {
                DType::F32 => binary!(f32, add_float),
                DType::F64 => binary!(f64, add_float),
                DType::I32 => binary!(i32, add_int),
                DType::I64 => binary!(i64, add_int),
                DType::C32 => binary!(Complex32, add_complex),
                DType::C64 => binary!(Complex64, add_complex),
                _ => return None,
            },
            ElementwiseReadOp::Subtract => match dtype {
                DType::F32 => binary!(f32, sub_float),
                DType::F64 => binary!(f64, sub_float),
                DType::I32 => binary!(i32, sub_int),
                DType::I64 => binary!(i64, sub_int),
                DType::C32 => binary!(Complex32, sub_complex),
                DType::C64 => binary!(Complex64, sub_complex),
                _ => return None,
            },
            ElementwiseReadOp::Multiply => match dtype {
                DType::F32 => binary!(f32, mul_float),
                DType::F64 => binary!(f64, mul_float),
                DType::I32 => binary!(i32, mul_int),
                DType::I64 => binary!(i64, mul_int),
                DType::C32 => binary!(Complex32, mul_complex),
                DType::C64 => binary!(Complex64, mul_complex),
                _ => return None,
            },
            ElementwiseReadOp::Negate => match dtype {
                DType::F32 => unary!(f32, neg_float),
                DType::F64 => unary!(f64, neg_float),
                DType::I32 => unary!(i32, neg_int),
                DType::I64 => unary!(i64, neg_int),
                DType::C32 => unary!(Complex32, neg_complex),
                DType::C64 => unary!(Complex64, neg_complex),
                _ => return None,
            },
            ElementwiseReadOp::Conj | ElementwiseReadOp::Divide => return None,
            _ => return None,
        };
        Some(result)
    }

    fn binary_read_native(
        &self,
        op: ElementwiseReadOp,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
    ) -> Option<crate::Result<Tensor>> {
        let lhs = lhs.tensor_view();
        let rhs = rhs.tensor_view();
        if lhs.dtype() != rhs.dtype() || lhs.shape() != rhs.shape() {
            return None;
        }
        let compact = |view: &TensorView| -> crate::Result<bool> {
            Ok(view.offset() == 0 && view.is_col_major_contiguous()?)
        };
        match (compact(&lhs), compact(&rhs)) {
            (Ok(true), Ok(true)) => {}
            (Ok(false), _) | (_, Ok(false)) => return None,
            (Err(error), _) | (_, Err(error)) => return Some(Err(error)),
        }

        macro_rules! binary {
            ($ty:ty, $lhs:expr, $rhs:expr, $kernel:ident) => {
                dispatch::launch_binary_views(
                    self.runtime(),
                    $lhs,
                    $rhs,
                    lhs.shape(),
                    op.label(),
                    // SAFETY: launch_binary_views validates equal shapes, zero-offset
                    // compact layouts and residency, and allocates an independent output.
                    |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                        elementwise::$kernel::launch_unchecked::<$ty, CubeclCudaRuntime>(
                            client, count, dim, out, lhs_arg, rhs_arg,
                        );
                    },
                )
                .map(Tensor::from_typed::<$ty>)
            };
        }
        macro_rules! dispatch_binary {
            ($variant:ident, $ty:ty, $kernel:ident) => {
                match (&lhs, &rhs) {
                    (TensorView::$variant(lhs), TensorView::$variant(rhs)) => {
                        Some(binary!($ty, lhs, rhs, $kernel))
                    }
                    _ => None,
                }
            };
        }

        match op {
            ElementwiseReadOp::Add => match lhs.dtype() {
                DType::F32 => dispatch_binary!(F32, f32, add_float),
                DType::F64 => dispatch_binary!(F64, f64, add_float),
                DType::I32 => dispatch_binary!(I32, i32, add_int),
                DType::I64 => dispatch_binary!(I64, i64, add_int),
                DType::C32 => dispatch_binary!(C32, Complex32, add_complex),
                DType::C64 => dispatch_binary!(C64, Complex64, add_complex),
                _ => None,
            },
            ElementwiseReadOp::Subtract => match lhs.dtype() {
                DType::F32 => dispatch_binary!(F32, f32, sub_float),
                DType::F64 => dispatch_binary!(F64, f64, sub_float),
                DType::I32 => dispatch_binary!(I32, i32, sub_int),
                DType::I64 => dispatch_binary!(I64, i64, sub_int),
                DType::C32 => dispatch_binary!(C32, Complex32, sub_complex),
                DType::C64 => dispatch_binary!(C64, Complex64, sub_complex),
                _ => None,
            },
            ElementwiseReadOp::Multiply => match lhs.dtype() {
                DType::F32 => dispatch_binary!(F32, f32, mul_float),
                DType::F64 => dispatch_binary!(F64, f64, mul_float),
                DType::I32 => dispatch_binary!(I32, i32, mul_int),
                DType::I64 => dispatch_binary!(I64, i64, mul_int),
                DType::C32 => dispatch_binary!(C32, Complex32, mul_complex),
                DType::C64 => dispatch_binary!(C64, Complex64, mul_complex),
                _ => None,
            },
            ElementwiseReadOp::Divide => match lhs.dtype() {
                DType::F32 => dispatch_binary!(F32, f32, div_float),
                DType::F64 => dispatch_binary!(F64, f64, div_float),
                DType::C32 => dispatch_binary!(C32, Complex32, div_complex),
                DType::C64 => dispatch_binary!(C64, Complex64, div_complex),
                _ => None,
            },
            _ => None,
        }
    }

    fn unary_read_native(
        &self,
        op: UnaryReadOp,
        input: TensorRead<'_>,
    ) -> Option<crate::Result<Tensor>> {
        let input = input.tensor_view();
        let compact = if input.offset() != 0 {
            false
        } else {
            match input.is_col_major_contiguous() {
                Ok(compact) => compact,
                Err(error) => return Some(Err(error)),
            }
        };
        if !compact {
            return None;
        }

        macro_rules! unary {
            ($variant:ident, $ty:ty, $kernel:ident) => {
                match &input {
                    TensorView::$variant(input) => Some(
                        dispatch::launch_unary_view(
                            self.runtime(),
                            input,
                            input.shape(),
                            match op {
                                UnaryReadOp::Neg => "neg",
                                UnaryReadOp::Exp => "exp",
                                UnaryReadOp::Log => "log",
                                UnaryReadOp::Sin => "sin",
                                UnaryReadOp::Cos => "cos",
                                UnaryReadOp::Tanh => "tanh",
                                UnaryReadOp::Sqrt => "sqrt",
                                UnaryReadOp::Rsqrt => "rsqrt",
                                UnaryReadOp::Expm1 => "expm1",
                                UnaryReadOp::Log1p => "log1p",
                            },
                            // SAFETY: launch_unary_view validates the shape, zero-offset
                            // compact layout and residency, and allocates a fresh output.
                            |client, count, dim, out, input_arg| unsafe {
                                elementwise::$kernel::launch_unchecked::<$ty, CubeclCudaRuntime>(
                                    client, count, dim, out, input_arg,
                                );
                            },
                        )
                        .map(Tensor::from_typed::<$ty>),
                    ),
                    _ => None,
                }
            };
        }

        match (input.dtype(), op) {
            (DType::F32, UnaryReadOp::Neg) => unary!(F32, f32, neg_float),
            (DType::F64, UnaryReadOp::Neg) => unary!(F64, f64, neg_float),
            (DType::I32, UnaryReadOp::Neg) => unary!(I32, i32, neg_int),
            (DType::I64, UnaryReadOp::Neg) => unary!(I64, i64, neg_int),
            (DType::C32, UnaryReadOp::Neg) => unary!(C32, Complex32, neg_complex),
            (DType::C64, UnaryReadOp::Neg) => unary!(C64, Complex64, neg_complex),
            (DType::F32, UnaryReadOp::Exp) => unary!(F32, f32, exp_float),
            (DType::F64, UnaryReadOp::Exp) => unary!(F64, f64, exp_float),
            (DType::F32, UnaryReadOp::Log) => unary!(F32, f32, log_float),
            (DType::F64, UnaryReadOp::Log) => unary!(F64, f64, log_float),
            (DType::F32, UnaryReadOp::Sin) => unary!(F32, f32, sin_float),
            (DType::F64, UnaryReadOp::Sin) => unary!(F64, f64, sin_float),
            (DType::F32, UnaryReadOp::Cos) => unary!(F32, f32, cos_float),
            (DType::F64, UnaryReadOp::Cos) => unary!(F64, f64, cos_float),
            (DType::F32, UnaryReadOp::Tanh) => unary!(F32, f32, tanh_float),
            (DType::F64, UnaryReadOp::Tanh) => unary!(F64, f64, tanh_float),
            (DType::F32, UnaryReadOp::Sqrt) => unary!(F32, f32, sqrt_float),
            (DType::F64, UnaryReadOp::Sqrt) => unary!(F64, f64, sqrt_float),
            (DType::F32, UnaryReadOp::Rsqrt) => unary!(F32, f32, rsqrt_float),
            (DType::F64, UnaryReadOp::Rsqrt) => unary!(F64, f64, rsqrt_float),
            (DType::F32, UnaryReadOp::Expm1) => unary!(F32, f32, expm1_float),
            (DType::F64, UnaryReadOp::Expm1) => unary!(F64, f64, expm1_float),
            (DType::F32, UnaryReadOp::Log1p) => unary!(F32, f32, log1p_float),
            (DType::F64, UnaryReadOp::Log1p) => unary!(F64, f64, log1p_float),
            _ => None,
        }
    }

    /// Accept a read operand the runtime prepared, materializing a view only
    /// when the operation has no native borrowed implementation.
    fn read_input<'a>(&mut self, input: TensorRead<'a>) -> crate::Result<CudaReadInput<'a>> {
        match input.as_tensor() {
            Some(tensor) => Ok(CudaReadInput::Borrowed(tensor)),
            None => Ok(CudaReadInput::Materialized(Box::new(
                self.to_contiguous_read(input)?,
            ))),
        }
    }
}

impl TensorElementwise for CudaBackend {
    // Borrowed-view entry points. Compact views use the same native kernels as
    // owned tensors; strided views retain the explicit materialization fallback.
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) =
            self.binary_read_native(ElementwiseReadOp::Add, lhs.clone(), rhs.clone())
        {
            return result;
        }
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        self.add(lhs.as_tensor(), rhs.as_tensor())
    }

    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) =
            self.binary_read_native(ElementwiseReadOp::Subtract, lhs.clone(), rhs.clone())
        {
            return result;
        }
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        let lhs = lhs.as_tensor();
        let rhs = rhs.as_tensor();
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

    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) =
            self.binary_read_native(ElementwiseReadOp::Multiply, lhs.clone(), rhs.clone())
        {
            return result;
        }
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        self.mul(lhs.as_tensor(), rhs.as_tensor())
    }

    fn neg_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Neg, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        self.neg(input.as_tensor())
    }

    fn conj_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.conj(input.as_tensor())
    }

    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) =
            self.binary_read_native(ElementwiseReadOp::Divide, lhs.clone(), rhs.clone())
        {
            return result;
        }
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        self.div(lhs.as_tensor(), rhs.as_tensor())
    }

    fn rem_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        self.rem(lhs.as_tensor(), rhs.as_tensor())
    }

    fn abs_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.abs(input.as_tensor())
    }

    fn sign_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.sign(input.as_tensor())
    }

    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        let lhs = lhs.as_tensor();
        let rhs = rhs.as_tensor();
        dispatch::dispatch_binary_float_int!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Maximum,
            maximum_float,
            maximum_int
        )
    }

    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        let lhs = lhs.as_tensor();
        let rhs = rhs.as_tensor();
        dispatch::dispatch_binary_float_int!(
            self,
            lhs,
            rhs,
            PrimitiveOpKind::Minimum,
            minimum_float,
            minimum_int
        )
    }

    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &CompareDir,
    ) -> crate::Result<Tensor> {
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        self.compare(lhs.as_tensor(), rhs.as_tensor(), dir)
    }

    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        let pred_owned = self.read_input(pred)?;
        let on_true_owned = self.read_input(on_true)?;
        let on_false_owned = self.read_input(on_false)?;
        let pred = pred_owned.as_tensor();
        let on_true = on_true_owned.as_tensor();
        let on_false = on_false_owned.as_tensor();
        let op = op_name(
            PrimitiveOpKind::Select,
            op_descriptor::GpuLaunchKind::SelectBoolFloatInt,
        )?;
        match (pred.dtype(), on_true.dtype(), on_false.dtype()) {
            (DType::Bool, DType::F32, DType::F32) => {
                let pred = typed_or_unsupported::<bool>(pred, op)?;
                let on_true = typed_or_unsupported::<f32>(on_true, op)?;
                let on_false = typed_or_unsupported::<f32>(on_false, op)?;
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
                .map(Tensor::from_typed::<f32>)
            }
            (DType::Bool, DType::F64, DType::F64) => {
                let pred = typed_or_unsupported::<bool>(pred, op)?;
                let on_true = typed_or_unsupported::<f64>(on_true, op)?;
                let on_false = typed_or_unsupported::<f64>(on_false, op)?;
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
                .map(Tensor::from_typed::<f64>)
            }
            (DType::Bool, DType::I32, DType::I32) => {
                let pred = typed_or_unsupported::<bool>(pred, op)?;
                let on_true = typed_or_unsupported::<i32>(on_true, op)?;
                let on_false = typed_or_unsupported::<i32>(on_false, op)?;
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
                .map(Tensor::from_typed::<i32>)
            }
            (DType::Bool, DType::I64, DType::I64) => {
                let pred = typed_or_unsupported::<bool>(pred, op)?;
                let on_true = typed_or_unsupported::<i64>(on_true, op)?;
                let on_false = typed_or_unsupported::<i64>(on_false, op)?;
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
                .map(Tensor::from_typed::<i64>)
            }
            (DType::C32, DType::C32, DType::C32) | (DType::C64, DType::C64, DType::C64) => {
                Err(unsupported_dtype(op, pred.dtype()))
            }
            _ => Err(ternary_dtype_mismatch(op, pred, on_true, on_false)),
        }
    }

    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        let lower = self.read_input(lower)?;
        let upper = self.read_input(upper)?;
        self.clamp(input.as_tensor(), lower.as_tensor(), upper.as_tensor())
    }

    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        mut out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        if inputs.len() != op.arity() {
            return Err(crate::Error::invalid_argument(
                op.label(),
                "inputs",
                format!("expected {} inputs, got {}", op.arity(), inputs.len()),
            ));
        }
        tenferro_tensor::backend::validate_read_into_destination(op.label(), inputs, &out)?;
        if let Some(result) = self.elementwise_read_into_native(op, inputs, &mut out) {
            return result;
        }
        tenferro_tensor::backend::elementwise_read_into_via_allocating_ops(self, op, inputs, out)
    }

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
        // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
        match input.dtype() {
            DType::F32 => {
                let tensor = typed_or_unsupported::<f32>(input, op)?;
                ensure_resident_on_runtime(self.runtime(), tensor, op)?;
                self.to_contiguous_view_typed(&tensor.as_view(), op)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let tensor = typed_or_unsupported::<f64>(input, op)?;
                ensure_resident_on_runtime(self.runtime(), tensor, op)?;
                self.to_contiguous_view_typed(&tensor.as_view(), op)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 | DType::I64 | DType::Bool => Err(unsupported_dtype(op, input.dtype())),
            DType::C32 => {
                let tensor = typed_or_unsupported::<Complex32>(input, op)?;
                launch_unary(
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
                .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let tensor = typed_or_unsupported::<Complex64>(input, op)?;
                launch_unary(
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
                .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "conj",
                "an externally defined payload is not supported by this GPU operation",
            )),
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
        match (lhs.dtype(), rhs.dtype()) {
            (DType::F32, DType::F32) if lhs.shape() != rhs.shape() => launch_scalar_binary(
                self,
                typed_or_unsupported::<f32>(lhs, op)?,
                typed_or_unsupported::<f32>(rhs, op)?,
                op,
                |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                    elementwise::scalar_div_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>),
            (DType::F32, DType::F32) => launch_binary(
                self.runtime(),
                typed_or_unsupported::<f32>(lhs, op)?,
                typed_or_unsupported::<f32>(rhs, op)?,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F64) if lhs.shape() != rhs.shape() => launch_scalar_binary(
                self,
                typed_or_unsupported::<f64>(lhs, op)?,
                typed_or_unsupported::<f64>(rhs, op)?,
                op,
                |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                    elementwise::scalar_div_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>),
            (DType::F64, DType::F64) => launch_binary(
                self.runtime(),
                typed_or_unsupported::<f64>(lhs, op)?,
                typed_or_unsupported::<f64>(rhs, op)?,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>),
            (DType::I32, DType::I32) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    typed_or_unsupported::<i32>(lhs, op)?,
                    typed_or_unsupported::<i32>(rhs, op)?,
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
                .map(Tensor::from_typed::<i32>)
            }
            (DType::I32, DType::I32) => launch_checked_integer_binary(
                self,
                typed_or_unsupported::<i32>(lhs, op)?,
                typed_or_unsupported::<i32>(rhs, op)?,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::div_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<i32>),
            (DType::I64, DType::I64) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    typed_or_unsupported::<i64>(lhs, op)?,
                    typed_or_unsupported::<i64>(rhs, op)?,
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
                .map(Tensor::from_typed::<i64>)
            }
            (DType::I64, DType::I64) => launch_checked_integer_binary(
                self,
                typed_or_unsupported::<i64>(lhs, op)?,
                typed_or_unsupported::<i64>(rhs, op)?,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::div_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<i64>),
            (DType::C32, DType::C32) => launch_binary(
                self.runtime(),
                typed_or_unsupported::<Complex32>(lhs, op)?,
                typed_or_unsupported::<Complex32>(rhs, op)?,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_complex::launch_unchecked::<Complex32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::C64) => launch_binary(
                self.runtime(),
                typed_or_unsupported::<Complex64>(lhs, op)?,
                typed_or_unsupported::<Complex64>(rhs, op)?,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::div_complex::launch_unchecked::<Complex64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
            _ => Err(dtype_mismatch(op, lhs, rhs)),
        }
    }

    fn rem(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Rem,
            op_descriptor::GpuLaunchKind::BinaryFloatInt,
        )?;
        match (lhs.dtype(), rhs.dtype()) {
            (DType::F32, DType::F32) if lhs.shape() != rhs.shape() => launch_scalar_binary(
                self,
                typed_or_unsupported::<f32>(lhs, op)?,
                typed_or_unsupported::<f32>(rhs, op)?,
                op,
                |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                    elementwise::scalar_rem_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>),
            (DType::F32, DType::F32) => launch_binary(
                self.runtime(),
                typed_or_unsupported::<f32>(lhs, op)?,
                typed_or_unsupported::<f32>(rhs, op)?,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::rem_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F64) if lhs.shape() != rhs.shape() => launch_scalar_binary(
                self,
                typed_or_unsupported::<f64>(lhs, op)?,
                typed_or_unsupported::<f64>(rhs, op)?,
                op,
                |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                    elementwise::scalar_rem_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>),
            (DType::F64, DType::F64) => launch_binary(
                self.runtime(),
                typed_or_unsupported::<f64>(lhs, op)?,
                typed_or_unsupported::<f64>(rhs, op)?,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::rem_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>),
            (DType::I32, DType::I32) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    typed_or_unsupported::<i32>(lhs, op)?,
                    typed_or_unsupported::<i32>(rhs, op)?,
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
                .map(Tensor::from_typed::<i32>)
            }
            (DType::I32, DType::I32) => launch_checked_integer_binary(
                self,
                typed_or_unsupported::<i32>(lhs, op)?,
                typed_or_unsupported::<i32>(rhs, op)?,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::rem_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<i32>),
            (DType::I64, DType::I64) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    typed_or_unsupported::<i64>(lhs, op)?,
                    typed_or_unsupported::<i64>(rhs, op)?,
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
                .map(Tensor::from_typed::<i64>)
            }
            (DType::I64, DType::I64) => launch_checked_integer_binary(
                self,
                typed_or_unsupported::<i64>(lhs, op)?,
                typed_or_unsupported::<i64>(rhs, op)?,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::DivisionByZero,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::rem_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<i64>),
            (DType::C32, DType::C32) | (DType::C64, DType::C64) => {
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
        // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
        match input.dtype() {
            DType::F32 => {
                let tensor = typed_or_unsupported::<f32>(input, op)?;
                dispatch::launch_unary_elementwise_kernel!(self, tensor, op, abs_float, f32, F32)
            }
            DType::F64 => {
                let tensor = typed_or_unsupported::<f64>(input, op)?;
                dispatch::launch_unary_elementwise_kernel!(self, tensor, op, abs_float, f64, F64)
            }
            DType::I32 => {
                let tensor = typed_or_unsupported::<i32>(input, op)?;
                dispatch::launch_unary_elementwise_kernel!(self, tensor, op, abs_int, i32, I32)
            }
            DType::I64 => {
                let tensor = typed_or_unsupported::<i64>(input, op)?;
                dispatch::launch_unary_elementwise_kernel!(self, tensor, op, abs_int, i64, I64)
            }
            DType::C32 => {
                let tensor = typed_or_unsupported::<Complex32>(input, op)?;
                dispatch::launch_unary(
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
                .map(Tensor::from_typed::<f32>)
            }
            DType::C64 => {
                let tensor = typed_or_unsupported::<Complex64>(input, op)?;
                dispatch::launch_unary(
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
                .map(Tensor::from_typed::<f64>)
            }
            DType::Bool => Err(unsupported_dtype(op, input.dtype())),
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "abs",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        dispatch::dispatch_unary_float_complex_int!(
            self,
            input,
            PrimitiveOpKind::Sign,
            sign_float,
            sign_int,
            sign_complex
        )
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Compare,
            op_descriptor::GpuLaunchKind::CompareFloatIntToBool,
        )?;
        match (lhs.dtype(), rhs.dtype()) {
            (DType::F32, DType::F32) => launch_compare_bool(
                self.runtime(),
                typed_or_unsupported::<f32>(lhs, op)?,
                typed_or_unsupported::<f32>(rhs, op)?,
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
            .map(Tensor::from_typed::<bool>),
            (DType::F64, DType::F64) => launch_compare_bool(
                self.runtime(),
                typed_or_unsupported::<f64>(lhs, op)?,
                typed_or_unsupported::<f64>(rhs, op)?,
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
            .map(Tensor::from_typed::<bool>),
            (DType::I32, DType::I32) => launch_compare_bool(
                self.runtime(),
                typed_or_unsupported::<i32>(lhs, op)?,
                typed_or_unsupported::<i32>(rhs, op)?,
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
            .map(Tensor::from_typed::<bool>),
            (DType::I64, DType::I64) => launch_compare_bool(
                self.runtime(),
                typed_or_unsupported::<i64>(lhs, op)?,
                typed_or_unsupported::<i64>(rhs, op)?,
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
            .map(Tensor::from_typed::<bool>),
            (DType::C32, DType::C32) | (DType::C64, DType::C64) => {
                Err(unsupported_dtype(op, lhs.dtype()))
            }
            _ => Err(dtype_mismatch(op, lhs, rhs)),
        }
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::Clamp,
            op_descriptor::GpuLaunchKind::ClampFloat,
        )?;
        // Dispatch on the tags and recover the typed tensors, which is what `as_typed` exists for.
        match (input.dtype(), lower.dtype(), upper.dtype()) {
            (DType::F32, DType::F32, DType::F32) => {
                let input = typed_or_unsupported::<f32>(input, op)?;
                let lower = typed_or_unsupported::<f32>(lower, op)?;
                let upper = typed_or_unsupported::<f32>(upper, op)?;
                launch_ternary(
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
                .map(Tensor::from_typed::<f32>)
            }
            (DType::F64, DType::F64, DType::F64) => {
                let input = typed_or_unsupported::<f64>(input, op)?;
                let lower = typed_or_unsupported::<f64>(lower, op)?;
                let upper = typed_or_unsupported::<f64>(upper, op)?;
                launch_ternary(
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
                .map(Tensor::from_typed::<f64>)
            }
            (DType::C32, DType::C32, DType::C32) | (DType::C64, DType::C64, DType::C64) => {
                Err(unsupported_dtype(op, input.dtype()))
            }
            _ => Err(ternary_dtype_mismatch(op, input, lower, upper)),
        }
    }
}

impl TensorAnalytic for CudaBackend {
    // Borrowed-view entry points. Compact views use the native elementwise
    // kernels; unsupported layouts retain the explicit materialization fallback.
    fn exp_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Exp, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Exp,
            exp_float
        )
    }

    fn log_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Log, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Log,
            log_float
        )
    }

    fn sin_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Sin, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Sin,
            sin_float
        )
    }

    fn cos_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Cos, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Cos,
            cos_float
        )
    }

    fn tanh_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Tanh, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Tanh,
            tanh_float
        )
    }

    fn sqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Sqrt, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Sqrt,
            sqrt_float
        )
    }

    fn rsqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Rsqrt, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Rsqrt,
            rsqrt_float
        )
    }

    fn pow_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        let lhs = self.read_input(lhs)?;
        let rhs = self.read_input(rhs)?;
        let lhs = lhs.as_tensor();
        let rhs = rhs.as_tensor();
        let op = op_name(
            PrimitiveOpKind::Pow,
            op_descriptor::GpuLaunchKind::BinaryFloatInt,
        )?;
        if lhs.dtype() != rhs.dtype() {
            return Err(dtype_mismatch(op, lhs, rhs));
        }
        match (lhs.dtype(), rhs.dtype()) {
            (DType::F32, DType::F32) if lhs.shape() != rhs.shape() => launch_scalar_binary(
                self,
                typed_or_unsupported::<f32>(lhs, op)?,
                typed_or_unsupported::<f32>(rhs, op)?,
                op,
                |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                    elementwise::scalar_pow_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>),
            (DType::F32, DType::F32) => launch_binary(
                self.runtime(),
                typed_or_unsupported::<f32>(lhs, op)?,
                typed_or_unsupported::<f32>(rhs, op)?,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::pow_float::launch_unchecked::<f32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F64) if lhs.shape() != rhs.shape() => launch_scalar_binary(
                self,
                typed_or_unsupported::<f64>(lhs, op)?,
                typed_or_unsupported::<f64>(rhs, op)?,
                op,
                |client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar| unsafe {
                    elementwise::scalar_pow_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, lhs_scalar,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>),
            (DType::F64, DType::F64) => launch_binary(
                self.runtime(),
                typed_or_unsupported::<f64>(lhs, op)?,
                typed_or_unsupported::<f64>(rhs, op)?,
                lhs.shape(),
                op,
                |client, count, dim, out, lhs_arg, rhs_arg| unsafe {
                    elementwise::pow_float::launch_unchecked::<f64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<f64>),
            (DType::I32, DType::I32) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    typed_or_unsupported::<i32>(lhs, op)?,
                    typed_or_unsupported::<i32>(rhs, op)?,
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
                .map(Tensor::from_typed::<i32>)
            }
            (DType::I32, DType::I32) => launch_checked_integer_binary(
                self,
                typed_or_unsupported::<i32>(lhs, op)?,
                typed_or_unsupported::<i32>(rhs, op)?,
                op,
                crate::DType::I32,
                CheckedIntegerDomain::NegativeExponent,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::pow_int_checked::launch_unchecked::<i32, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<i32>),
            (DType::I64, DType::I64) if lhs.shape() != rhs.shape() => {
                launch_checked_integer_scalar_binary(
                    self,
                    typed_or_unsupported::<i64>(lhs, op)?,
                    typed_or_unsupported::<i64>(rhs, op)?,
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
                .map(Tensor::from_typed::<i64>)
            }
            (DType::I64, DType::I64) => launch_checked_integer_binary(
                self,
                typed_or_unsupported::<i64>(lhs, op)?,
                typed_or_unsupported::<i64>(rhs, op)?,
                op,
                crate::DType::I64,
                CheckedIntegerDomain::NegativeExponent,
                |client, count, dim, out, lhs_arg, rhs_arg, err_arg| unsafe {
                    elementwise::pow_int_checked::launch_unchecked::<i64, CubeclCudaRuntime>(
                        client, count, dim, out, lhs_arg, rhs_arg, err_arg,
                    );
                },
            )
            .map(Tensor::from_typed::<i64>),
            (DType::C32, DType::C32) => {
                dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
                Err(unsupported_dtype(op, crate::DType::C32))
            }
            (DType::C64, DType::C64) => {
                dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
                Err(unsupported_dtype(op, crate::DType::C64))
            }
            _ => {
                dispatch::ensure_same_shape(op, lhs.shape(), rhs.shape())?;
                Err(dtype_mismatch(op, lhs, rhs))
            }
        }
    }

    fn expm1_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Expm1, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Expm1,
            expm1_float
        )
    }

    fn log1p_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        if let Some(result) = self.unary_read_native(UnaryReadOp::Log1p, input.clone()) {
            return result;
        }
        let input = self.read_input(input)?;
        dispatch::dispatch_unary_float_only!(
            self,
            input.as_tensor(),
            PrimitiveOpKind::Log1p,
            log1p_float
        )
    }
}

/// The typed tensor behind a contiguous-read adapter's tensor, or its refusal.
fn contiguous_read_typed<T: TensorScalar>(tensor: &Tensor) -> crate::Result<&TypedTensor<T>> {
    tensor.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(
            "CudaBackend::to_contiguous_read",
            "an externally defined payload is not supported by this GPU operation",
        )
    })
}

/// The typed tensor behind a copy read's tensor, or its refusal.
fn copy_read_typed<T: TensorScalar>(tensor: &Tensor) -> crate::Result<&TypedTensor<T>> {
    tensor.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(
            "copy_read_into",
            "an externally defined payload is not supported by this GPU operation",
        )
    })
}

impl TensorStructural for CudaBackend {
    // Borrowed-view entry points. The traced runtime prepares operands as
    // `TensorRead`; a view is materialized before the CUDA kernel runs.
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.transpose(input.as_tensor(), perm)
    }

    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.reshape(input.as_tensor(), shape)
    }

    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.broadcast_in_dim(input.as_tensor(), shape, dims)
    }

    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        macro_rules! materialize_cutensor {
            ($variant:ident, $view:expr) => {{
                let view = $view;
                self.to_contiguous_view_cutensor_or_cubecl(&view, "CudaBackend::to_contiguous_read")
                    .map(Tensor::from_typed::<preset_scalar!($variant)>)
            }};
        }
        macro_rules! materialize_cubecl {
            ($variant:ident, $view:expr) => {{
                let view = $view;
                self.to_contiguous_view_typed(&view, "CudaBackend::to_contiguous_read")
                    .map(Tensor::from_typed::<preset_scalar!($variant)>)
            }};
        }

        match input {
            TensorRead::Tensor(tensor) => match tensor.dtype() {
                DType::F32 => {
                    materialize_cutensor!(F32, contiguous_read_typed::<f32>(tensor)?.as_view())
                }
                DType::F64 => {
                    materialize_cutensor!(F64, contiguous_read_typed::<f64>(tensor)?.as_view())
                }
                DType::I32 => {
                    materialize_cubecl!(I32, contiguous_read_typed::<i32>(tensor)?.as_view())
                }
                DType::I64 => {
                    materialize_cubecl!(I64, contiguous_read_typed::<i64>(tensor)?.as_view())
                }
                DType::Bool => Err(unsupported_dtype(
                    "CudaBackend::to_contiguous_read",
                    crate::DType::Bool,
                )),
                DType::C32 => materialize_cutensor!(
                    C32,
                    contiguous_read_typed::<Complex32>(tensor)?.as_view()
                ),
                DType::C64 => materialize_cutensor!(
                    C64,
                    contiguous_read_typed::<Complex64>(tensor)?.as_view()
                ),
                // A caller-owned payload has no GPU implementation for this operation.
                DType::External(_) => Err(crate::Error::unsupported(
                    "CudaBackend::to_contiguous_read",
                    "an externally defined payload is not supported by this GPU operation",
                )),
            },
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
                    TensorWrite::Tensor(dst)
                        if dst.dtype()
                            == <preset_scalar!($variant) as tenferro_tensor::TensorScalar>::dtype() =>
                    {
                        let dst = dst
                            .as_typed_mut::<preset_scalar!($variant)>()
                            .expect("the dtype guard selects this arm");
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
        // Every numeric dtype uses the cuTENSOR permutation path, which is the
        // bandwidth-bound optimum for a multi-axis permutation destination.
        // Complex dtypes are planned through their real view with a real
        // `alpha = 1`, so the scaling multiply is exact and cannot turn
        // `(inf, finite)` into `(inf, NaN)` (issue #1891).
        macro_rules! copy_source_cutensor {
            ($variant:ident, $src:expr) => {{
                let src = $src;
                match dst {
                    TensorWrite::Tensor(dst)
                        if dst.dtype()
                            == <preset_scalar!($variant) as tenferro_tensor::TensorScalar>::dtype() =>
                    {
                        let dst = dst
                            .as_typed_mut::<preset_scalar!($variant)>()
                            .expect("the dtype guard selects this arm");
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
                    TensorWrite::Tensor(tensor) if tensor.dtype() == crate::DType::Bool => Err(
                        unsupported_dtype("CudaBackend::copy_read_into", crate::DType::Bool),
                    ),
                    TensorWrite::View(TensorViewMut::Bool(_)) => Err(unsupported_dtype(
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
            TensorRead::Tensor(tensor) => match tensor.dtype() {
                DType::F32 => copy_source_cutensor!(F32, copy_read_typed::<f32>(tensor)?.as_view()),
                DType::F64 => copy_source_cutensor!(F64, copy_read_typed::<f64>(tensor)?.as_view()),
                DType::I32 => copy_source_typed!(I32, copy_read_typed::<i32>(tensor)?.as_view()),
                DType::I64 => copy_source_typed!(I64, copy_read_typed::<i64>(tensor)?.as_view()),
                DType::Bool => reject_bool_source!(),
                DType::C32 => {
                    copy_source_cutensor!(C32, copy_read_typed::<Complex32>(tensor)?.as_view())
                }
                DType::C64 => {
                    copy_source_cutensor!(C64, copy_read_typed::<Complex64>(tensor)?.as_view())
                }
                // A caller-owned payload has no GPU implementation for this operation.
                DType::External(_) => Err(crate::Error::unsupported(
                    "copy_read_into",
                    "an externally defined payload is not supported by this GPU operation",
                )),
            },
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
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "transpose")?;
                permutation::transpose(self, t, perm).map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "transpose")?;
                permutation::transpose(self, t, perm).map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "transpose")?;
                self.transpose_typed(t, perm).map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "transpose")?;
                self.transpose_typed(t, perm).map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "transpose")?;
                self.transpose_bool(t, perm).map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "transpose")?;
                permutation::transpose(self, t, perm)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "transpose")?;
                permutation::transpose(self, t, perm)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "transpose",
                "an externally defined payload is not supported by this GPU operation",
            )),
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
        let contiguous = match input.dtype() {
            DType::Bool => self
                .duplicate_bool(
                    input.as_typed::<bool>().ok_or_else(|| {
                        crate::Error::unsupported(
                            "reshape",
                            "an externally defined payload is not supported by this GPU operation",
                        )
                    })?,
                    "reshape",
                )
                .map(Tensor::from_typed::<bool>)?,
            // Every other tag is materialized through the read path, which refuses the
            // externally defined payload itself.
            _ => self.to_contiguous_read(TensorRead::from_tensor(input))?,
        };
        match contiguous.dtype() {
            DType::F32 => {
                cubecl_reshape_metadata(contiguous.into_typed::<f32>()?, shape.to_vec(), "reshape")
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                cubecl_reshape_metadata(contiguous.into_typed::<f64>()?, shape.to_vec(), "reshape")
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                cubecl_reshape_metadata(contiguous.into_typed::<i32>()?, shape.to_vec(), "reshape")
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                cubecl_reshape_metadata(contiguous.into_typed::<i64>()?, shape.to_vec(), "reshape")
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                cubecl_reshape_metadata(contiguous.into_typed::<bool>()?, shape.to_vec(), "reshape")
                    .map(Tensor::from_typed::<bool>)
            }
            DType::C32 => cubecl_reshape_metadata(
                contiguous.into_typed::<Complex32>()?,
                shape.to_vec(),
                "reshape",
            )
            .map(Tensor::from_typed::<num_complex::Complex32>),
            DType::C64 => cubecl_reshape_metadata(
                contiguous.into_typed::<Complex64>()?,
                shape.to_vec(),
                "reshape",
            )
            .map(Tensor::from_typed::<num_complex::Complex64>),
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "reshape",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "broadcast_in_dim")?;
                self.broadcast_typed(t, shape, dims)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "broadcast_in_dim")?;
                self.broadcast_typed(t, shape, dims)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "broadcast_in_dim")?;
                self.broadcast_typed(t, shape, dims)
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "broadcast_in_dim")?;
                self.broadcast_typed(t, shape, dims)
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "broadcast_in_dim")?;
                self.broadcast_bool(t, shape, dims)
                    .map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "broadcast_in_dim")?;
                self.broadcast_typed(t, shape, dims)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "broadcast_in_dim")?;
                self.broadcast_typed(t, shape, dims)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "broadcast_in_dim",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn cast(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        match (input.dtype(), to) {
            // An externally defined destination has no CUDA conversion, so the
            // backend rejects it instead of guessing a representation.
            (_, crate::DType::External(_)) => Err(crate::Error::unsupported(
                "cast",
                "an externally defined scalar has no CUDA conversion",
            )),
            (DType::F32, crate::DType::F32) => self
                .duplicate_typed(typed_or_unsupported::<f32>(input, "cast")?)
                .map(Tensor::from_typed::<f32>),
            (DType::F64, crate::DType::F64) => self
                .duplicate_typed(typed_or_unsupported::<f64>(input, "cast")?)
                .map(Tensor::from_typed::<f64>),
            (DType::I32, crate::DType::I32) => self
                .duplicate_typed(typed_or_unsupported::<i32>(input, "cast")?)
                .map(Tensor::from_typed::<i32>),
            (DType::I64, crate::DType::I64) => self
                .duplicate_typed(typed_or_unsupported::<i64>(input, "cast")?)
                .map(Tensor::from_typed::<i64>),
            (DType::Bool, crate::DType::Bool) => self
                .duplicate_bool(typed_or_unsupported::<bool>(input, "cast")?, "cast")
                .map(Tensor::from_typed::<bool>),
            (DType::C32, crate::DType::C32) => self
                .duplicate_typed(typed_or_unsupported::<Complex32>(input, "cast")?)
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, crate::DType::C64) => self
                .duplicate_typed(typed_or_unsupported::<Complex64>(input, "cast")?)
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::F32, crate::DType::F64) => self
                .convert_float_to_float::<f32, f64>(typed_or_unsupported::<f32>(input, "cast")?)
                .map(Tensor::from_typed::<f64>),
            (DType::F32, crate::DType::I32) => {
                validate_cuda_real_cast::<f32, f32>(
                    self,
                    typed_or_unsupported::<f32>(input, "cast")?,
                    1,
                    CastIntegerTarget::I32,
                )?;
                self.convert_numeric::<f32, i32>(typed_or_unsupported::<f32>(input, "cast")?)
                    .map(Tensor::from_typed::<i32>)
            }
            (DType::F32, crate::DType::I64) => {
                validate_cuda_real_cast::<f32, f32>(
                    self,
                    typed_or_unsupported::<f32>(input, "cast")?,
                    1,
                    CastIntegerTarget::I64,
                )?;
                self.convert_numeric::<f32, i64>(typed_or_unsupported::<f32>(input, "cast")?)
                    .map(Tensor::from_typed::<i64>)
            }
            (DType::F32, crate::DType::Bool) => self
                .convert_numeric_to_bool(typed_or_unsupported::<f32>(input, "cast")?)
                .map(Tensor::from_typed::<bool>),
            (DType::F32, crate::DType::C32) => self
                .convert_f32_to_c32(typed_or_unsupported::<f32>(input, "cast")?)
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::F32, crate::DType::C64) => self
                .convert_f32_to_c64(typed_or_unsupported::<f32>(input, "cast")?)
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::F64, crate::DType::F32) => self
                .convert_float_to_float::<f64, f32>(typed_or_unsupported::<f64>(input, "cast")?)
                .map(Tensor::from_typed::<f32>),
            (DType::F64, crate::DType::I32) => {
                validate_cuda_real_cast::<f64, f64>(
                    self,
                    typed_or_unsupported::<f64>(input, "cast")?,
                    1,
                    CastIntegerTarget::I32,
                )?;
                self.convert_numeric::<f64, i32>(typed_or_unsupported::<f64>(input, "cast")?)
                    .map(Tensor::from_typed::<i32>)
            }
            (DType::F64, crate::DType::I64) => {
                validate_cuda_real_cast::<f64, f64>(
                    self,
                    typed_or_unsupported::<f64>(input, "cast")?,
                    1,
                    CastIntegerTarget::I64,
                )?;
                self.convert_numeric::<f64, i64>(typed_or_unsupported::<f64>(input, "cast")?)
                    .map(Tensor::from_typed::<i64>)
            }
            (DType::F64, crate::DType::Bool) => self
                .convert_numeric_to_bool(typed_or_unsupported::<f64>(input, "cast")?)
                .map(Tensor::from_typed::<bool>),
            (DType::F64, crate::DType::C32) => self
                .convert_f64_to_c32(typed_or_unsupported::<f64>(input, "cast")?)
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::F64, crate::DType::C64) => self
                .convert_f64_to_c64(typed_or_unsupported::<f64>(input, "cast")?)
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, crate::DType::F32) => self
                .convert_numeric::<i32, f32>(typed_or_unsupported::<i32>(input, "cast")?)
                .map(Tensor::from_typed::<f32>),
            (DType::I32, crate::DType::F64) => self
                .convert_numeric::<i32, f64>(typed_or_unsupported::<i32>(input, "cast")?)
                .map(Tensor::from_typed::<f64>),
            (DType::I32, crate::DType::I64) => self
                .convert_numeric::<i32, i64>(typed_or_unsupported::<i32>(input, "cast")?)
                .map(Tensor::from_typed::<i64>),
            (DType::I32, crate::DType::Bool) => self
                .convert_numeric_to_bool(typed_or_unsupported::<i32>(input, "cast")?)
                .map(Tensor::from_typed::<bool>),
            (DType::I32, crate::DType::C32) => self
                .convert_numeric_to_complex::<i32, Complex32, f32>(typed_or_unsupported::<i32>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::I32, crate::DType::C64) => self
                .convert_numeric_to_complex::<i32, Complex64, f64>(typed_or_unsupported::<i32>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I64, crate::DType::F32) => self
                .convert_numeric::<i64, f32>(typed_or_unsupported::<i64>(input, "cast")?)
                .map(Tensor::from_typed::<f32>),
            (DType::I64, crate::DType::F64) => self
                .convert_numeric::<i64, f64>(typed_or_unsupported::<i64>(input, "cast")?)
                .map(Tensor::from_typed::<f64>),
            (DType::I64, crate::DType::I32) => self
                .convert_numeric::<i64, i32>(typed_or_unsupported::<i64>(input, "cast")?)
                .map(Tensor::from_typed::<i32>),
            (DType::I64, crate::DType::Bool) => self
                .convert_numeric_to_bool(typed_or_unsupported::<i64>(input, "cast")?)
                .map(Tensor::from_typed::<bool>),
            (DType::I64, crate::DType::C32) => self
                .convert_numeric_to_complex::<i64, Complex32, f32>(typed_or_unsupported::<i64>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::I64, crate::DType::C64) => self
                .convert_numeric_to_complex::<i64, Complex64, f64>(typed_or_unsupported::<i64>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::Bool, crate::DType::F32) => self
                .convert_bool_to_numeric::<f32>(typed_or_unsupported::<bool>(input, "cast")?)
                .map(Tensor::from_typed::<f32>),
            (DType::Bool, crate::DType::F64) => self
                .convert_bool_to_numeric::<f64>(typed_or_unsupported::<bool>(input, "cast")?)
                .map(Tensor::from_typed::<f64>),
            (DType::Bool, crate::DType::I32) => self
                .convert_bool_to_numeric::<i32>(typed_or_unsupported::<bool>(input, "cast")?)
                .map(Tensor::from_typed::<i32>),
            (DType::Bool, crate::DType::I64) => self
                .convert_bool_to_numeric::<i64>(typed_or_unsupported::<bool>(input, "cast")?)
                .map(Tensor::from_typed::<i64>),
            (DType::Bool, crate::DType::C32) => self
                .convert_bool_to_complex::<Complex32, f32>(typed_or_unsupported::<bool>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::Bool, crate::DType::C64) => self
                .convert_bool_to_complex::<Complex64, f64>(typed_or_unsupported::<bool>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::C32, crate::DType::F32) => self
                .convert_c32_to_f32(typed_or_unsupported::<Complex32>(input, "cast")?)
                .map(Tensor::from_typed::<f32>),
            (DType::C32, crate::DType::F64) => self
                .convert_c32_to_f64(typed_or_unsupported::<Complex32>(input, "cast")?)
                .map(Tensor::from_typed::<f64>),
            (DType::C32, crate::DType::I32) => {
                validate_cuda_real_cast::<Complex32, f32>(
                    self,
                    typed_or_unsupported::<Complex32>(input, "cast")?,
                    2,
                    CastIntegerTarget::I32,
                )?;
                self.convert_complex_to_numeric::<Complex32, i32>(
                    typed_or_unsupported::<Complex32>(input, "cast")?,
                )
                .map(Tensor::from_typed::<i32>)
            }
            (DType::C32, crate::DType::I64) => {
                validate_cuda_real_cast::<Complex32, f32>(
                    self,
                    typed_or_unsupported::<Complex32>(input, "cast")?,
                    2,
                    CastIntegerTarget::I64,
                )?;
                self.convert_complex_to_numeric::<Complex32, i64>(
                    typed_or_unsupported::<Complex32>(input, "cast")?,
                )
                .map(Tensor::from_typed::<i64>)
            }
            (DType::C32, crate::DType::Bool) => self
                .convert_complex_to_bool::<Complex32, f32>(typed_or_unsupported::<Complex32>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<bool>),
            (DType::C32, crate::DType::C64) => self
                .convert_complex_to_complex::<Complex32, Complex64, f32, f64>(
                    typed_or_unsupported::<Complex32>(input, "cast")?,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::C64, crate::DType::F32) => self
                .convert_c64_to_f32(typed_or_unsupported::<Complex64>(input, "cast")?)
                .map(Tensor::from_typed::<f32>),
            (DType::C64, crate::DType::F64) => self
                .convert_c64_to_f64(typed_or_unsupported::<Complex64>(input, "cast")?)
                .map(Tensor::from_typed::<f64>),
            (DType::C64, crate::DType::I32) => {
                validate_cuda_real_cast::<Complex64, f64>(
                    self,
                    typed_or_unsupported::<Complex64>(input, "cast")?,
                    2,
                    CastIntegerTarget::I32,
                )?;
                self.convert_complex_to_numeric::<Complex64, i32>(
                    typed_or_unsupported::<Complex64>(input, "cast")?,
                )
                .map(Tensor::from_typed::<i32>)
            }
            (DType::C64, crate::DType::I64) => {
                validate_cuda_real_cast::<Complex64, f64>(
                    self,
                    typed_or_unsupported::<Complex64>(input, "cast")?,
                    2,
                    CastIntegerTarget::I64,
                )?;
                self.convert_complex_to_numeric::<Complex64, i64>(
                    typed_or_unsupported::<Complex64>(input, "cast")?,
                )
                .map(Tensor::from_typed::<i64>)
            }
            (DType::C64, crate::DType::Bool) => self
                .convert_complex_to_bool::<Complex64, f64>(typed_or_unsupported::<Complex64>(
                    input, "cast",
                )?)
                .map(Tensor::from_typed::<bool>),
            (DType::C64, crate::DType::C32) => self
                .convert_complex_to_complex::<Complex64, Complex32, f64, f32>(
                    typed_or_unsupported::<Complex64>(input, "cast")?,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            // A caller-owned payload has no GPU implementation for this operation.
            (DType::External(_), _) => Err(crate::Error::unsupported(
                "cast",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "extract_diagonal")?;
                self.extract_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "extract_diagonal")?;
                self.extract_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "extract_diagonal")?;
                self.extract_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "extract_diagonal")?;
                self.extract_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "extract_diagonal")?;
                self.extract_diagonal_bool(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "extract_diagonal")?;
                self.extract_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "extract_diagonal")?;
                self.extract_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "extract_diagonal",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "embed_diagonal")?;
                self.embed_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "embed_diagonal")?;
                self.embed_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "embed_diagonal")?;
                self.embed_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "embed_diagonal")?;
                self.embed_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "embed_diagonal")?;
                self.embed_diagonal_bool(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "embed_diagonal")?;
                self.embed_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "embed_diagonal")?;
                self.embed_diagonal_typed(t, axis_a, axis_b)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "embed_diagonal",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "tril")?;
                self.tril_typed(t, k).map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "tril")?;
                self.tril_typed(t, k).map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "tril")?;
                self.tril_typed(t, k).map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "tril")?;
                self.tril_typed(t, k).map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "tril")?;
                self.tril_bool(t, k).map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "tril")?;
                self.tril_typed(t, k)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "tril")?;
                self.tril_typed(t, k)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "tril",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "triu")?;
                self.triu_typed(t, k).map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "triu")?;
                self.triu_typed(t, k).map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "triu")?;
                self.triu_typed(t, k).map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "triu")?;
                self.triu_typed(t, k).map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "triu")?;
                self.triu_bool(t, k).map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "triu")?;
                self.triu_typed(t, k)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "triu")?;
                self.triu_typed(t, k)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "triu",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }
}

impl TensorReduction for CudaBackend {
    // Borrowed-view entry points. The traced runtime prepares operands as
    // `TensorRead`; a view is materialized before the CUDA kernel runs.
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.reduce_sum(input.as_tensor(), axes)
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.reduce_prod(input.as_tensor(), axes)
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.reduce_max(input.as_tensor(), axes)
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        let input = self.read_input(input)?;
        self.reduce_min(input.as_tensor(), axes)
    }

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceSum,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, op)?;
                self.reduce_sum_float_typed(t, axes)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, op)?;
                self.reduce_sum_float_typed(t, axes)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, op)?;
                self.reduce_sum_int_typed(t, axes)
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, op)?;
                self.reduce_sum_int_typed(t, axes)
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool => Err(unsupported_dtype(op, input.dtype())),
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, op)?;
                self.reduce_sum_complex_typed(t, axes)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, op)?;
                self.reduce_sum_complex_typed(t, axes)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "reduce_sum",
                "an externally defined payload is not supported by this GPU operation",
            )),
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
        // Dispatch on the tag and recover the typed tensor, which is what `as_typed` exists for.
        if axes.is_empty() {
            return match input.dtype() {
                DType::F32 | DType::F64 => self.mul(input, input),
                DType::I32 | DType::I64 | DType::Bool | DType::C32 | DType::C64 => {
                    Err(unsupported_dtype(op, input.dtype()))
                }
                DType::External(_) => Err(crate::Error::unsupported(
                    op,
                    "an externally defined payload is not supported by this GPU operation",
                )),
            };
        }
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, op)?;
                self.reduce_sum_squares_float_typed(t, axes)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, op)?;
                self.reduce_sum_squares_float_typed(t, axes)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 | DType::I64 | DType::Bool | DType::C32 | DType::C64 => {
                Err(unsupported_dtype(op, input.dtype()))
            }
            DType::External(_) => Err(crate::Error::unsupported(
                op,
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceProd,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, op)?;
                self.reduce_prod_float_typed(t, axes)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, op)?;
                self.reduce_prod_float_typed(t, axes)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, op)?;
                self.reduce_prod_int_typed(t, axes)
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, op)?;
                self.reduce_prod_int_typed(t, axes)
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool => Err(unsupported_dtype(op, input.dtype())),
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, op)?;
                self.reduce_prod_complex_typed(t, axes)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, op)?;
                self.reduce_prod_complex_typed(t, axes)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "reduce_prod",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceMax,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, op)?;
                self.reduce_max_float_typed(t, axes)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, op)?;
                self.reduce_max_float_typed(t, axes)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, op)?;
                self.reduce_max_int_typed(t, axes)
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, op)?;
                self.reduce_max_int_typed(t, axes)
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool | DType::C32 | DType::C64 => Err(unsupported_dtype(op, input.dtype())),
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "reduce_max",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        let op = op_name(
            PrimitiveOpKind::ReduceMin,
            op_descriptor::GpuLaunchKind::Reduction,
        )?;
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, op)?;
                self.reduce_min_float_typed(t, axes)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, op)?;
                self.reduce_min_float_typed(t, axes)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, op)?;
                self.reduce_min_int_typed(t, axes)
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, op)?;
                self.reduce_min_int_typed(t, axes)
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool | DType::C32 | DType::C64 => Err(unsupported_dtype(op, input.dtype())),
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "reduce_min",
                "an externally defined payload is not supported by this GPU operation",
            )),
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

    // Contract strided reads in place instead of materializing them through
    // `to_contiguous_read` (the `TensorDot` default behavior).
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        gemm::dot_general_read_allocating(self, lhs, rhs, config, false, false)
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
        match (operand.dtype(), start_indices.dtype()) {
            (DType::F32, DType::F32) => self
                .gather_typed(
                    typed_or_unsupported::<f32>(operand, "gather")?,
                    typed_or_unsupported::<f32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F32) => self
                .gather_typed(
                    typed_or_unsupported::<f64>(operand, "gather")?,
                    typed_or_unsupported::<f32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::F32) => self
                .gather_typed(
                    typed_or_unsupported::<Complex32>(operand, "gather")?,
                    typed_or_unsupported::<f32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::F32) => self
                .gather_typed(
                    typed_or_unsupported::<Complex64>(operand, "gather")?,
                    typed_or_unsupported::<f32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, DType::F32) => self
                .gather_typed(
                    typed_or_unsupported::<i32>(operand, "gather")?,
                    typed_or_unsupported::<f32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<i32>),
            (DType::F32, DType::F64) => self
                .gather_typed(
                    typed_or_unsupported::<f32>(operand, "gather")?,
                    typed_or_unsupported::<f64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F64) => self
                .gather_typed(
                    typed_or_unsupported::<f64>(operand, "gather")?,
                    typed_or_unsupported::<f64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::F64) => self
                .gather_typed(
                    typed_or_unsupported::<Complex32>(operand, "gather")?,
                    typed_or_unsupported::<f64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::F64) => self
                .gather_typed(
                    typed_or_unsupported::<Complex64>(operand, "gather")?,
                    typed_or_unsupported::<f64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, DType::F64) => self
                .gather_typed(
                    typed_or_unsupported::<i32>(operand, "gather")?,
                    typed_or_unsupported::<f64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<i32>),
            (DType::F32, DType::I32) => self
                .gather_typed(
                    typed_or_unsupported::<f32>(operand, "gather")?,
                    typed_or_unsupported::<i32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::I32) => self
                .gather_typed(
                    typed_or_unsupported::<f64>(operand, "gather")?,
                    typed_or_unsupported::<i32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::I32) => self
                .gather_typed(
                    typed_or_unsupported::<Complex32>(operand, "gather")?,
                    typed_or_unsupported::<i32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::I32) => self
                .gather_typed(
                    typed_or_unsupported::<Complex64>(operand, "gather")?,
                    typed_or_unsupported::<i32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, DType::I32) => self
                .gather_typed(
                    typed_or_unsupported::<i32>(operand, "gather")?,
                    typed_or_unsupported::<i32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<i32>),
            (DType::F32, DType::I64) => self
                .gather_typed(
                    typed_or_unsupported::<f32>(operand, "gather")?,
                    typed_or_unsupported::<i64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::I64) => self
                .gather_typed(
                    typed_or_unsupported::<f64>(operand, "gather")?,
                    typed_or_unsupported::<i64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::I64) => self
                .gather_typed(
                    typed_or_unsupported::<Complex32>(operand, "gather")?,
                    typed_or_unsupported::<i64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::I64) => self
                .gather_typed(
                    typed_or_unsupported::<Complex64>(operand, "gather")?,
                    typed_or_unsupported::<i64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, DType::I64) => self
                .gather_typed(
                    typed_or_unsupported::<i32>(operand, "gather")?,
                    typed_or_unsupported::<i64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<i32>),
            (DType::Bool, DType::F32) => self
                .gather_bool(
                    typed_or_unsupported::<bool>(operand, "gather")?,
                    typed_or_unsupported::<f32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<bool>),
            (DType::Bool, DType::F64) => self
                .gather_bool(
                    typed_or_unsupported::<bool>(operand, "gather")?,
                    typed_or_unsupported::<f64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<bool>),
            (DType::Bool, DType::I32) => self
                .gather_bool(
                    typed_or_unsupported::<bool>(operand, "gather")?,
                    typed_or_unsupported::<i32>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<bool>),
            (DType::Bool, DType::I64) => self
                .gather_bool(
                    typed_or_unsupported::<bool>(operand, "gather")?,
                    typed_or_unsupported::<i64>(start_indices, "gather")?,
                    config,
                )
                .map(Tensor::from_typed::<bool>),
            (_, DType::Bool) => Err(unsupported_dtype("gather", start_indices.dtype())),
            (_, DType::C32 | DType::C64) => Err(unsupported_dtype("gather", start_indices.dtype())),
            (DType::I64, _) => Err(unsupported_dtype("gather", operand.dtype())),
            // A caller-owned payload has no GPU implementation for this operation.
            (DType::External(_), _) | (_, DType::External(_)) => Err(crate::Error::unsupported(
                "gather",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        match (operand.dtype(), scatter_indices.dtype(), updates.dtype()) {
            (DType::F32, DType::F32, DType::F32) => self
                .scatter_float_typed(
                    typed_or_unsupported::<f32>(operand, "scatter")?,
                    typed_or_unsupported::<f32>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<f32>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F32, DType::F64) => self
                .scatter_float_typed(
                    typed_or_unsupported::<f64>(operand, "scatter")?,
                    typed_or_unsupported::<f32>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<f64>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::F32, DType::C32) => self
                .scatter_complex_typed::<_, f32, _>(
                    typed_or_unsupported::<Complex32>(operand, "scatter")?,
                    typed_or_unsupported::<f32>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<Complex32>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::F32, DType::C64) => self
                .scatter_complex_typed::<_, f64, _>(
                    typed_or_unsupported::<Complex64>(operand, "scatter")?,
                    typed_or_unsupported::<f32>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<Complex64>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::F32, DType::F64, DType::F32) => self
                .scatter_float_typed(
                    typed_or_unsupported::<f32>(operand, "scatter")?,
                    typed_or_unsupported::<f64>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<f32>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F64, DType::F64) => self
                .scatter_float_typed(
                    typed_or_unsupported::<f64>(operand, "scatter")?,
                    typed_or_unsupported::<f64>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<f64>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::F64, DType::C32) => self
                .scatter_complex_typed::<_, f32, _>(
                    typed_or_unsupported::<Complex32>(operand, "scatter")?,
                    typed_or_unsupported::<f64>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<Complex32>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::F64, DType::C64) => self
                .scatter_complex_typed::<_, f64, _>(
                    typed_or_unsupported::<Complex64>(operand, "scatter")?,
                    typed_or_unsupported::<f64>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<Complex64>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::F32, DType::I32, DType::F32) => self
                .scatter_float_typed(
                    typed_or_unsupported::<f32>(operand, "scatter")?,
                    typed_or_unsupported::<i32>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<f32>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::I32, DType::F64) => self
                .scatter_float_typed(
                    typed_or_unsupported::<f64>(operand, "scatter")?,
                    typed_or_unsupported::<i32>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<f64>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::I32, DType::C32) => self
                .scatter_complex_typed::<_, f32, _>(
                    typed_or_unsupported::<Complex32>(operand, "scatter")?,
                    typed_or_unsupported::<i32>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<Complex32>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::I32, DType::C64) => self
                .scatter_complex_typed::<_, f64, _>(
                    typed_or_unsupported::<Complex64>(operand, "scatter")?,
                    typed_or_unsupported::<i32>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<Complex64>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::F32, DType::I64, DType::F32) => self
                .scatter_float_typed(
                    typed_or_unsupported::<f32>(operand, "scatter")?,
                    typed_or_unsupported::<i64>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<f32>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::I64, DType::F64) => self
                .scatter_float_typed(
                    typed_or_unsupported::<f64>(operand, "scatter")?,
                    typed_or_unsupported::<i64>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<f64>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::I64, DType::C32) => self
                .scatter_complex_typed::<_, f32, _>(
                    typed_or_unsupported::<Complex32>(operand, "scatter")?,
                    typed_or_unsupported::<i64>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<Complex32>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::I64, DType::C64) => self
                .scatter_complex_typed::<_, f64, _>(
                    typed_or_unsupported::<Complex64>(operand, "scatter")?,
                    typed_or_unsupported::<i64>(scatter_indices, "scatter")?,
                    typed_or_unsupported::<Complex64>(updates, "scatter")?,
                    config,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (_, DType::Bool, _) => Err(unsupported_dtype("scatter", scatter_indices.dtype())),
            (_, DType::C32 | DType::C64, _) => {
                Err(unsupported_dtype("scatter", scatter_indices.dtype()))
            }
            (DType::Bool, _, _) => Err(unsupported_operation(
                "scatter",
                "Bool data tensors are not supported by additive scatter",
            )),
            (DType::I32, _, _) | (DType::I64, _, _) => {
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
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "slice")?;
                self.slice_typed(t, config).map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "slice")?;
                self.slice_typed(t, config).map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "slice")?;
                self.slice_typed(t, config).map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "slice")?;
                self.slice_typed(t, config).map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "slice")?;
                self.slice_bool(t, config).map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "slice")?;
                self.slice_typed(t, config)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "slice")?;
                self.slice_typed(t, config)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "slice",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        match (input.dtype(), starts.dtype()) {
            (DType::F32, DType::F32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<f32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<f64>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::F32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<Complex32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::F32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<Complex64>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, DType::F32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<i32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<i32>),
            (DType::F32, DType::F64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<f32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::F64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<f64>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::F64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<Complex32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::F64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<Complex64>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, DType::F64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<i32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<i32>),
            (DType::F32, DType::I32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<f32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::I32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<f64>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::I32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<Complex32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::I32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<Complex64>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, DType::I32) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<i32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<i32>),
            (DType::F32, DType::I64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<f32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<f32>),
            (DType::F64, DType::I64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<f64>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<f64>),
            (DType::C32, DType::I64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<Complex32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<num_complex::Complex32>),
            (DType::C64, DType::I64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<Complex64>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<num_complex::Complex64>),
            (DType::I32, DType::I64) => self
                .dynamic_slice_typed(
                    typed_or_unsupported::<i32>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<i32>),
            (DType::Bool, DType::I32) => self
                .dynamic_slice_bool(
                    typed_or_unsupported::<bool>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<bool>),
            (DType::Bool, DType::I64) => self
                .dynamic_slice_bool(
                    typed_or_unsupported::<bool>(input, "dynamic_slice")?,
                    typed_or_unsupported::<i64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<bool>),
            (DType::Bool, DType::F32) => self
                .dynamic_slice_bool(
                    typed_or_unsupported::<bool>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f32>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<bool>),
            (DType::Bool, DType::F64) => self
                .dynamic_slice_bool(
                    typed_or_unsupported::<bool>(input, "dynamic_slice")?,
                    typed_or_unsupported::<f64>(starts, "dynamic_slice")?,
                    slice_sizes,
                )
                .map(Tensor::from_typed::<bool>),
            (_, DType::Bool) => Err(unsupported_dtype("dynamic_slice", starts.dtype())),
            (_, DType::C32 | DType::C64) => Err(unsupported_dtype("dynamic_slice", starts.dtype())),
            (DType::I64, _) => Err(unsupported_dtype("dynamic_slice", input.dtype())),
            // A caller-owned payload has no GPU implementation for this operation.
            (DType::External(_), _) | (_, DType::External(_)) => Err(crate::Error::unsupported(
                "dynamic_slice",
                "an externally defined payload is not supported by this GPU operation",
            )),
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
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "pad")?;
                self.pad_typed(t, config).map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "pad")?;
                self.pad_typed(t, config).map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "pad")?;
                self.pad_typed(t, config).map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "pad")?;
                self.pad_typed(t, config).map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "pad")?;
                self.pad_bool(t, config).map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "pad")?;
                self.pad_typed(t, config)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "pad")?;
                self.pad_typed(t, config)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "pad",
                "an externally defined payload is not supported by this GPU operation",
            )),
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
        match first.dtype() {
            DType::F32 => {
                let typed: crate::Result<Vec<&TypedTensor<f32>>> = inputs
                    .iter()
                    .map(|tensor| {
                        tensor
                            .as_typed::<f32>()
                            .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis)
                    .map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let typed: crate::Result<Vec<&TypedTensor<f64>>> = inputs
                    .iter()
                    .map(|tensor| {
                        tensor
                            .as_typed::<f64>()
                            .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis)
                    .map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let typed: crate::Result<Vec<&TypedTensor<i32>>> = inputs
                    .iter()
                    .map(|tensor| {
                        tensor
                            .as_typed::<i32>()
                            .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis)
                    .map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let typed: crate::Result<Vec<&TypedTensor<i64>>> = inputs
                    .iter()
                    .map(|tensor| {
                        tensor
                            .as_typed::<i64>()
                            .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis)
                    .map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let typed: crate::Result<Vec<&TypedTensor<bool>>> = inputs
                    .iter()
                    .map(|tensor| {
                        tensor
                            .as_typed::<bool>()
                            .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                    })
                    .collect();
                self.concatenate_bool(&typed?, axis)
                    .map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let typed: crate::Result<Vec<&TypedTensor<Complex32>>> = inputs
                    .iter()
                    .map(|tensor| {
                        tensor
                            .as_typed::<Complex32>()
                            .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let typed: crate::Result<Vec<&TypedTensor<Complex64>>> = inputs
                    .iter()
                    .map(|tensor| {
                        tensor
                            .as_typed::<Complex64>()
                            .ok_or_else(|| dtype_mismatch("concatenate", first, tensor))
                    })
                    .collect();
                self.concatenate_typed(&typed?, axis)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "concatenate",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        match input.dtype() {
            DType::F32 => {
                let t = typed_or_unsupported::<f32>(input, "reverse")?;
                self.reverse_typed(t, axes).map(Tensor::from_typed::<f32>)
            }
            DType::F64 => {
                let t = typed_or_unsupported::<f64>(input, "reverse")?;
                self.reverse_typed(t, axes).map(Tensor::from_typed::<f64>)
            }
            DType::I32 => {
                let t = typed_or_unsupported::<i32>(input, "reverse")?;
                self.reverse_typed(t, axes).map(Tensor::from_typed::<i32>)
            }
            DType::I64 => {
                let t = typed_or_unsupported::<i64>(input, "reverse")?;
                self.reverse_typed(t, axes).map(Tensor::from_typed::<i64>)
            }
            DType::Bool => {
                let t = typed_or_unsupported::<bool>(input, "reverse")?;
                self.reverse_bool(t, axes).map(Tensor::from_typed::<bool>)
            }
            DType::C32 => {
                let t = typed_or_unsupported::<Complex32>(input, "reverse")?;
                self.reverse_typed(t, axes)
                    .map(Tensor::from_typed::<num_complex::Complex32>)
            }
            DType::C64 => {
                let t = typed_or_unsupported::<Complex64>(input, "reverse")?;
                self.reverse_typed(t, axes)
                    .map(Tensor::from_typed::<num_complex::Complex64>)
            }
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "reverse",
                "an externally defined payload is not supported by this GPU operation",
            )),
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

/// Operand of a fused CUDA kernel: an owned tensor or a compact borrowed view.
///
/// The eager einsum path prepares operands as borrowed views over already
/// allocated device storage, so fused entry points accept both forms, the way
/// the traced runtime already hands them.
// INVARIANT: the view variant carries the provider descriptor inline so a
// fused launch never allocates; the enum only lives for one kernel launch.
#[allow(clippy::large_enum_variant)]
enum CompactOperand<'a, T> {
    Tensor(&'a TypedTensor<T>),
    View(TypedTensorView<'a, T>),
}

impl<T: TensorScalar + Clone + 'static> CompactOperand<'_, T> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    fn ensure_resident(&self, rt: &CudaRuntime, op: &'static str) -> crate::Result<()> {
        match self {
            Self::Tensor(tensor) => dispatch::ensure_resident_on_runtime(rt, tensor, op),
            Self::View(view) => dispatch::ensure_view_resident_on_runtime(rt, view, op),
        }
    }

    fn binding(&self, op: &'static str) -> crate::Result<TensorBinding<CubeclCudaRuntime>> {
        match self {
            Self::Tensor(tensor) => dispatch::typed_tensor_binding(tensor, op),
            Self::View(view) => dispatch::typed_view_binding(view, op),
        }
    }
}

/// Dtype-erased borrowed view of the operands accepted by
/// [`CudaBackend::execute_broadcast_multiply`].
enum BroadcastMultiplyView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    C32(TypedTensorView<'a, Complex32>),
    C64(TypedTensorView<'a, Complex64>),
}

/// Accept a view only when the fused kernel can index it directly.
///
/// The kernel reads each operand in compact column-major order, which is the
/// same requirement [`dispatch::typed_view_binding`] enforces; other view
/// forms keep the caller's materializing fallback.
fn compact_view(view: TensorView<'_>) -> crate::Result<Option<BroadcastMultiplyView<'_>>> {
    fn usable<T: TensorScalar + Clone + 'static>(
        view: TypedTensorView<'_, T>,
    ) -> crate::Result<Option<TypedTensorView<'_, T>>> {
        Ok((view.offset() == 0 && view.is_col_major_contiguous()?).then_some(view))
    }

    Ok(match view {
        TensorView::F32(view) => usable(view)?.map(BroadcastMultiplyView::F32),
        TensorView::F64(view) => usable(view)?.map(BroadcastMultiplyView::F64),
        TensorView::I32(view) => usable(view)?.map(BroadcastMultiplyView::I32),
        TensorView::I64(view) => usable(view)?.map(BroadcastMultiplyView::I64),
        TensorView::C32(view) => usable(view)?.map(BroadcastMultiplyView::C32),
        TensorView::C64(view) => usable(view)?.map(BroadcastMultiplyView::C64),
        TensorView::Bool(_) => None,
    })
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
        match (lhs, rhs) {
            // Owned operands dispatch on the runtime tags, which is what `as_typed` exists for.
            (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) => {
                match (lhs.dtype(), rhs.dtype()) {
                    (DType::F32, DType::F32) => {
                        let lhs = typed_or_unsupported::<f32>(lhs, "broadcast_multiply")?;
                        let rhs = typed_or_unsupported::<f32>(rhs, "broadcast_multiply")?;
                        launch_broadcast_multiply_typed(
                            self,
                            &CompactOperand::Tensor(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::Tensor(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<f32>)
                        .map(Some)
                    }
                    (DType::F64, DType::F64) => {
                        let lhs = typed_or_unsupported::<f64>(lhs, "broadcast_multiply")?;
                        let rhs = typed_or_unsupported::<f64>(rhs, "broadcast_multiply")?;
                        launch_broadcast_multiply_typed(
                            self,
                            &CompactOperand::Tensor(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::Tensor(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<f64>)
                        .map(Some)
                    }
                    (DType::I32, DType::I32) => {
                        let lhs = typed_or_unsupported::<i32>(lhs, "broadcast_multiply")?;
                        let rhs = typed_or_unsupported::<i32>(rhs, "broadcast_multiply")?;
                        launch_broadcast_multiply_int_typed(
                            self,
                            &CompactOperand::Tensor(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::Tensor(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<i32>)
                        .map(Some)
                    }
                    (DType::I64, DType::I64) => {
                        let lhs = typed_or_unsupported::<i64>(lhs, "broadcast_multiply")?;
                        let rhs = typed_or_unsupported::<i64>(rhs, "broadcast_multiply")?;
                        launch_broadcast_multiply_int_typed(
                            self,
                            &CompactOperand::Tensor(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::Tensor(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<i64>)
                        .map(Some)
                    }
                    (DType::C32, DType::C32) => {
                        let lhs = typed_or_unsupported::<Complex32>(lhs, "broadcast_multiply")?;
                        let rhs = typed_or_unsupported::<Complex32>(rhs, "broadcast_multiply")?;
                        launch_broadcast_multiply_complex_typed(
                            self,
                            &CompactOperand::Tensor(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::Tensor(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<Complex32>)
                        .map(Some)
                    }
                    (DType::C64, DType::C64) => {
                        let lhs = typed_or_unsupported::<Complex64>(lhs, "broadcast_multiply")?;
                        let rhs = typed_or_unsupported::<Complex64>(rhs, "broadcast_multiply")?;
                        launch_broadcast_multiply_complex_typed(
                            self,
                            &CompactOperand::Tensor(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::Tensor(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<Complex64>)
                        .map(Some)
                    }
                    (DType::Bool, DType::Bool) => Ok(None),
                    _ => Err(dtype_mismatch("broadcast_multiply", lhs, rhs)),
                }
            }
            // The eager einsum path prepares operands as borrowed views over
            // already allocated device storage. A compact view is consumed
            // directly; other read forms keep the caller's fallback.
            (TensorRead::View(lhs), TensorRead::View(rhs)) => {
                let (Some(lhs), Some(rhs)) = (compact_view(lhs)?, compact_view(rhs)?) else {
                    return Ok(None);
                };
                match (lhs, rhs) {
                    (BroadcastMultiplyView::F32(lhs), BroadcastMultiplyView::F32(rhs)) => {
                        launch_broadcast_multiply_typed(
                            self,
                            &CompactOperand::View(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::View(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<f32>)
                        .map(Some)
                    }
                    (BroadcastMultiplyView::F64(lhs), BroadcastMultiplyView::F64(rhs)) => {
                        launch_broadcast_multiply_typed(
                            self,
                            &CompactOperand::View(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::View(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<f64>)
                        .map(Some)
                    }
                    (BroadcastMultiplyView::I32(lhs), BroadcastMultiplyView::I32(rhs)) => {
                        launch_broadcast_multiply_int_typed(
                            self,
                            &CompactOperand::View(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::View(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<i32>)
                        .map(Some)
                    }
                    (BroadcastMultiplyView::I64(lhs), BroadcastMultiplyView::I64(rhs)) => {
                        launch_broadcast_multiply_int_typed(
                            self,
                            &CompactOperand::View(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::View(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<i64>)
                        .map(Some)
                    }
                    (BroadcastMultiplyView::C32(lhs), BroadcastMultiplyView::C32(rhs)) => {
                        launch_broadcast_multiply_complex_typed(
                            self,
                            &CompactOperand::View(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::View(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<Complex32>)
                        .map(Some)
                    }
                    (BroadcastMultiplyView::C64(lhs), BroadcastMultiplyView::C64(rhs)) => {
                        launch_broadcast_multiply_complex_typed(
                            self,
                            &CompactOperand::View(lhs),
                            lhs_shape,
                            lhs_dims,
                            &CompactOperand::View(rhs),
                            rhs_shape,
                            rhs_dims,
                        )
                        .map(Tensor::from_typed::<Complex64>)
                        .map(Some)
                    }
                    // Mismatched dtypes keep the caller's fallback, which
                    // reports the mismatch on the owned path.
                    _ => Ok(None),
                }
            }
            // Mixed owned and borrowed operands keep the caller's fallback:
            // the traced runtime hands both operands in the same form.
            _ => Ok(None),
        }
    }
}

impl BackendSession for CudaBackend {
    fn vdot_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        blas1::vdot_read(self, lhs, rhs)
    }

    fn norm_squared_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        blas1::norm_squared_read(self, input)
    }

    fn axpby_read_into_accum(
        &mut self,
        alpha: ContractionScalar,
        x: TensorRead<'_>,
        beta: ContractionScalar,
        y: TensorWrite<'_>,
    ) -> crate::Result<()> {
        blas1::axpby_read_into_accum(self, alpha, x, beta, y)
    }

    fn session_type_id(&self) -> TypeId {
        TypeId::of::<CudaBackendSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

impl BackendCachedDot for CudaBackend {
    // Read-based cached dot paths keep strided operands on device; the plan
    // cache is per backend, so the runtime cache slot stays unused.
    fn dot_general_read_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        gemm::dot_general_read_allocating(self, lhs, rhs, config, false, false)
    }

    fn dot_general_with_conj_read_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        gemm::dot_general_read_allocating(self, lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

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
    lhs: &CompactOperand<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &CompactOperand<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + TensorScalar + CubePrimitive + CubeFloat + Clone,
{
    ensure_same_shape_for_broadcast_multiply(lhs_shape, rhs_shape)?;
    validate_broadcast_in_dim(lhs.shape(), lhs_shape, lhs_dims)?;
    validate_broadcast_in_dim(rhs.shape(), rhs_shape, rhs_dims)?;
    lhs.ensure_resident(backend.runtime(), "broadcast_multiply")?;
    rhs.ensure_resident(backend.runtime(), "broadcast_multiply")?;
    dispatch::launch_binary_bindings(
        backend.runtime(),
        lhs.binding("broadcast_multiply")?,
        rhs.binding("broadcast_multiply")?,
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
    lhs: &CompactOperand<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &CompactOperand<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + TensorScalar + CubePrimitive + CubeInt + Clone,
{
    ensure_same_shape_for_broadcast_multiply(lhs_shape, rhs_shape)?;
    validate_broadcast_in_dim(lhs.shape(), lhs_shape, lhs_dims)?;
    validate_broadcast_in_dim(rhs.shape(), rhs_shape, rhs_dims)?;
    lhs.ensure_resident(backend.runtime(), "broadcast_multiply")?;
    rhs.ensure_resident(backend.runtime(), "broadcast_multiply")?;
    dispatch::launch_binary_bindings(
        backend.runtime(),
        lhs.binding("broadcast_multiply")?,
        rhs.binding("broadcast_multiply")?,
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
    lhs: &CompactOperand<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &CompactOperand<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + TensorScalar + CubePrimitive + CubeComplex + Clone,
{
    ensure_same_shape_for_broadcast_multiply(lhs_shape, rhs_shape)?;
    validate_broadcast_in_dim(lhs.shape(), lhs_shape, lhs_dims)?;
    validate_broadcast_in_dim(rhs.shape(), rhs_shape, rhs_dims)?;
    lhs.ensure_resident(backend.runtime(), "broadcast_multiply")?;
    rhs.ensure_resident(backend.runtime(), "broadcast_multiply")?;
    dispatch::launch_binary_bindings(
        backend.runtime(),
        lhs.binding("broadcast_multiply")?,
        rhs.binding("broadcast_multiply")?,
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
