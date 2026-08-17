//! Einsum and tensordot extension traits for concrete, eager, and traced values.

pub use crate::{
    ContractionTree, EinsumAxis, EinsumNotation, EinsumOptimize, EinsumSubscripts, TensorDotAxes,
    TensorEinsumExt, TensorEinsumIntoExt, TensorReadEinsumExt, TensorReadEinsumIntoExt,
    TensorTensordotExt, TraceContextEinsumExt, TracedTensorEinsumExt, TypedTensorEinsumExt,
    TypedTensorEinsumIntoExt, TypedTensorReadEinsumExt, TypedTensorReadEinsumIntoExt,
    TypedTensorTensordotExt,
};
pub use tenferro_runtime::{
    DType, Tensor, TensorBackend, TensorRead, TensorScalar, TensorView, TraceContext, TraceValue,
    TracedTensor, TypedTensor, TypedTensorView,
};
pub use tenferro_tensor::{DotGeneralAccumulation, TensorWrite, TypedTensorWrite};

#[cfg(feature = "autodiff")]
pub use crate::{EagerEinsumExt, EagerTensorEinsumExt};
#[cfg(feature = "autodiff")]
pub use tenferro_ad::{EagerRuntime, EagerTensor};
