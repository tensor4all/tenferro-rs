//! The linear algebra grammar traits, options and result types (#2552).

pub use crate::{
    Eig, Eigh, FullPivLu, LinalgBackend, LinalgOps, LinalgScalar, Lu, Qr, Slogdet, Svd,
    TracedLinalgOps,
};
pub use crate::{
    EighOptions, QrOptions, RankRevealingQrOptions, RankRevealingQrResult, SvdOptions,
};

pub use tenferro_runtime::{Tensor, TensorRead, TracedTensor, TypedTensor};

#[cfg(feature = "autodiff")]
pub use crate::EagerLinalgOps;
#[cfg(feature = "autodiff")]
pub use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
