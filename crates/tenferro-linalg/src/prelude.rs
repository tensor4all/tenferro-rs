//! Linear algebra extension traits and their method-resolution types.

pub use crate::{
    CholeskyOptions, DetOptions, Diagonal, EigOptions, EighOptions, EigvalsOptions,
    EigvalshOptions, FullPivLuOptions, FullPivLuSolveOptions, HouseholderQrOptions, InvOptions,
    KeepDims, LstsqOptions, LuOptions, NormOptions, PinvOptions, QrOptions,
    RankRevealingQrOptions, RankRevealingQrResult, Side, SlogdetOptions, SolveOptions,
    SvdOptions, SvdShape, SvdvalsOptions, Transpose, TriangularSolveOptions, Uplo,
};
pub use crate::{
    LinalgBackend, LinalgScalar, TensorLinalgExt, TensorReadLinalgExt, TraceContextLinalgExt,
    TracedTensorLinalgExt, TypedEig, TypedFullPivLu, TypedLu, TypedRankRevealingQrResult, TypedSvd,
    TypedTensorLinalgExt,
};

pub use tenferro_runtime::{Tensor, TensorRead, TracedTensor, TypedTensor};

#[cfg(feature = "autodiff")]
pub use crate::{EagerSessionLinalgExt, EagerTensorLinalgExt};
#[cfg(feature = "autodiff")]
pub use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
