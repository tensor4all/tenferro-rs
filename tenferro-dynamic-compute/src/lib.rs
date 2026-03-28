//! Dynamic non-AD tensor compute facade for tenferro-rs.
//!
//! This crate is the public home for runtime-selected dtypes without autodiff.
//! It re-exports the shared dynamic tensor substrate plus runtime management.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_dynamic_compute::{ScalarType, Tensor};
//! use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
//!
//! let dense = DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor)?;
//! let value: Tensor = dense.into();
//!
//! assert_eq!(value.scalar_type(), ScalarType::F64);
//! # Ok::<(), tenferro_dynamic_compute::Error>(())
//! ```

pub mod snapshot {
    //! Dynamic primal snapshots shared across tenferro surface crates.

    pub use tenferro_internal_frontend_core::snapshot::*;
}

pub use tenferro_device::{ComputeDevice, LogicalMemorySpace};
pub use tenferro_internal_error::{Error, Result};
pub use tenferro_internal_frontend_core::{
    DynTensor as Tensor, ScalarType, ScalarValue, StructuredTensor,
};
pub use tenferro_internal_runtime::{
    set_default_runtime, with_default_runtime, with_runtime, DefaultRuntimeGuard, RuntimeContext,
};
pub use tenferro_tensor::MemoryOrder;
