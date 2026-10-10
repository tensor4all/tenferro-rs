//! Core tensor types, views, backend traits, and backend-independent contracts.
//!
//! # Owned Tensors And Views
//!
//! [`TypedTensor<T>`](TypedTensor) and the dtype-erased [`Tensor`] enum are
//! owned tensor values. They are the right representation when a result is
//! materialized as compact column-major storage.
//!
//! [`TypedTensorView`] is a borrowed typed view over an existing tensor buffer.
//! It carries logical shape, arbitrary strides, and an offset, so metadata-only
//! layout changes such as transposes, slices, and broadcasts can be represented
//! without copying. Backend-aware code materializes and copies views through
//! [`TensorViewCanonicalization`], preserving placement and backend execution
//! policy.
//!
//! [`TensorRead`] is the dtype-erased borrowed input type used by eager kernels
//! and backend dispatch. It can borrow either an owned [`Tensor`] or a
//! [`TensorView`] with arbitrary strides. Prefer `TensorRead` for read-only
//! operation inputs so callers are not forced to materialize layout-only views.
//!
//! [`TensorValue`] is the owned lazy-value form. Use it when an API must store
//! a view result beyond the lifetime of a borrowed input, then expose a
//! short-lived `TensorRead` at kernel-dispatch time.
//!
//! Use [`Tensor::as_slice`] or [`TypedTensorView::as_slice`] only when compact
//! contiguous storage is part of the API contract. Use shape/stride-aware kernel
//! paths or `TensorRead` otherwise.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor::{Tensor, TypedTensor};
//!
//! let a = Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
//! assert_eq!(a.shape(), &[2]);
//! ```

/// Backend-independent rank/layout, dtype and scalar metadata re-exported from
/// `tenferro-tensor-core`. It holds metadata only: every tensor type, including
/// the default scalar set, is exported from this crate's root.
pub mod core {
    pub use tenferro_tensor_core::{
        col_major_strides, DType, DynRank, ErrorKind, IntoShapeVec, Rank, Result, ShapeMismatch,
        ShapeVec, SliceSpec, StrideVec, TensorLayout, TensorRank, TensorScalar, ValidationError,
        ValidationKind,
    };
}

// Re-exported so the exported dispatch macros can name them with `$crate` paths.
pub use num_complex::Complex;

/// The 32-bit complex scalar the exported dispatch macros name.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::Complex32;
///
/// assert_eq!(Complex32::new(1.0, 2.0).im, 2.0);
/// ```
pub type Complex32 = Complex<f32>;

/// The 64-bit complex scalar the exported dispatch macros name.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::Complex64;
///
/// assert_eq!(Complex64::new(1.0, 2.0).re, 1.0);
/// ```
pub type Complex64 = Complex<f64>;

pub use tenferro_tensor_core::{
    ErrorKind, IntoRankShape, IntoShapeVec, ShapeMismatch, ShapeVec, SliceSpec, StrideVec,
    ValidationError, ValidationKind,
};

mod default_scalars;
mod erased_host;
mod scalar_set;

pub mod backend;
pub mod cache;
pub mod capability;
pub mod config;
pub mod dispatch;
pub mod error;
mod native_session;
pub mod prelude;
mod session_entry;
pub mod types;
pub mod validate;

pub use backend::{
    has_active_backend_session, has_held_backend_session, with_session_entry_guard, ActivationOp,
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, ContractionScalar,
    DotGeneralAccumulation, ElementwiseReadOp, HeldSessionMarker, SessionCachedDot, TensorAnalytic,
    TensorBackend, TensorBackendOps, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
    TensorViewCanonicalization,
};
pub use cache::{CacheStats, RuntimeCacheControl};
pub use capability::{
    capability_output_dtype, BackendId, CapabilityAxis, CapabilityQuery, OperationCapability,
    SupportLevel, TensorBackendCapability,
};
pub use config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
pub use erased_host::ErasedHostTensor;
pub use error::{BoxError, Error, ReinterpretError, Result};
pub use native_session::NativeSessionRef;
pub use scalar_set::ScalarSet;
pub use session_entry::SessionEntryError;

pub use default_scalars::{DefaultScalars, DefaultScalarsRef, DefaultScalarsView};

pub use types::{
    col_major_strides, AllocationDomainId, AllocationId, BackendStorage, BackendStorageHandle,
    ColMajorView, ColMajorViewMut, CpuDomainId, DType, DeviceAccessError, DeviceAccessRequest,
    DeviceId, DeviceKind, DynRank, Dynamic, Gpu, GpuBackendKind, Host, HostAccessError,
    HostReadGuard, HostWriteGuard, MemoryKind, Placement, PreparedDeviceAccess, Rank,
    Representation, SharedTensorAllocationDomain, StorageBuffer, StridedSliceSpec, Tensor,
    TensorLayout, TensorRank, TensorRead, TensorScalar, TensorStorageRef, TensorStorageRefMut,
    TensorValue, TensorView, TensorViewMut, TensorWrite, TypedTensor, TypedTensorView,
    TypedTensorViewMut, TypedTensorViewMutSplit, TypedTensorWrite,
};

mod storage;

#[doc(hidden)]
pub use storage::{
    AccessError, AllocationKey, BackendAllocation, HostBufferRecycler, ProviderCapabilities,
    ProviderKind, ProviderReadMapping, ProviderWriteMapping, RootBoundSpan, RootResourceExtent,
    RootResourceId, SpanValidationError,
};
pub use storage::{AllocationGroup, DescriptorSlot, GroupError};

#[cfg(test)]
pub(crate) mod tests;
