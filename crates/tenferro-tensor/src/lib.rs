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
//! let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
//! assert_eq!(a.shape(), &[2]);
//! ```

/// Lightweight backend-independent host tensor data model.
///
/// Execution-capable tensors and backends in this crate remain separate from
/// the host-only core model during the crate-boundary split.
pub mod core {
    pub use tenferro_tensor_core::{
        col_major_strides, DType, DynRank, ErrorKind, HostTensor, HostTensorView, IntoShapeVec,
        Rank, Result, ShapeMismatch, ShapeVec, SliceSpec, StrideVec, Tensor, TensorLayout,
        TensorRank, TensorRef, TensorScalar, TensorView, ValidationError, ValidationKind,
    };
}

pub use tenferro_tensor_core::{
    ErrorKind, IntoShapeVec, ShapeMismatch, ShapeVec, SliceSpec, StrideVec, TensorRef,
    ValidationError, ValidationKind,
};

pub mod backend;
pub mod cache;
pub mod capability;
pub mod config;
pub mod dispatch;
pub mod error;
pub mod types;
pub mod validate;

pub use backend::{
    default_backend_session, BackendCachedDot, BackendRuntimeCache, BackendSession,
    BackendSessionHost, ContractionScalar, DotGeneralAccumulation, ElementwiseReadOp,
    SessionCachedDot, TensorAnalytic, TensorBackend, TensorBackendOps, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural, TensorViewCanonicalization,
};
pub use cache::{CacheStats, RuntimeCacheControl};
pub use capability::{
    capability_output_dtype, BackendId, CapabilityAxis, CapabilityQuery, OperationCapability,
    SupportLevel, TensorBackendCapability,
};
pub use config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
pub use error::{BoxError, Error, ReinterpretError, Result};
pub use types::{
    col_major_strides, AllocationDomainId, AllocationId, BackendStorage, BackendStorageHandle,
    CpuDomainId, DType, DeviceId, DeviceKind, DynRank, GpuBackendKind, HostAccessError,
    HostReadGuard, HostWriteGuard, MemoryKind, Placement, Rank, SharedTensorAllocationDomain,
    StorageBuffer, StridedSliceSpec, Tensor, TensorLayout, TensorRank, TensorRead, TensorScalar,
    TensorStorageRef, TensorStorageRefMut, TensorValue, TensorView, TensorViewMut, TensorWrite,
    TypedTensor, TypedTensorView, TypedTensorViewMut, TypedTensorViewMutSplit, TypedTensorWrite,
};

mod storage;

pub use storage::{AllocationGroup, DescriptorSlot, GroupError};

pub(crate) fn core_dtype(dtype: DType) -> tenferro_tensor_core::DType {
    match dtype {
        DType::F32 => tenferro_tensor_core::DType::F32,
        DType::F64 => tenferro_tensor_core::DType::F64,
        DType::I32 => tenferro_tensor_core::DType::I32,
        DType::I64 => tenferro_tensor_core::DType::I64,
        DType::Bool => tenferro_tensor_core::DType::Bool,
        DType::C32 => tenferro_tensor_core::DType::C32,
        DType::C64 => tenferro_tensor_core::DType::C64,
    }
}

#[cfg(test)]
mod tests;
