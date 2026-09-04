//! Linear algebra extension traits and their method-resolution types.

pub use crate::{
    EighOptions, QrOptions, RankRevealingQrOptions, RankRevealingQrResult, SvdOptions,
};
pub use crate::{
    LinalgBackend, LinalgScalar, TensorLinalgExt, TensorReadLinalgExt, TracedTensorLinalgExt,
    TypedEig, TypedFullPivLu, TypedLu, TypedRankRevealingQrResult, TypedSvd, TypedTensorLinalgExt,
};

pub use tenferro_runtime::{Tensor, TensorRead, TracedTensor, TypedTensor};
pub use tenferro_tensor::BackendSessionHost;

#[cfg(feature = "autodiff")]
pub use crate::EagerTensorLinalgExt;
#[cfg(feature = "autodiff")]
pub use tenferro_ad::{EagerRuntime, EagerTensor};
