//! Linear algebra extension traits and their method-resolution types.

pub use crate::{
    CholeskyOptions, DetOptions, Diagonal, EigOptions, EighOptions, EigvalsOptions,
    EigvalshOptions, FullPivLuOptions, FullPivLuSolveOptions, HouseholderQrOptions, InvOptions,
    KeepDims, LstsqOptions, LuOptions, NormOptions, PinvOptions, QrOptions, RankRevealingQrOptions,
    RankRevealingQrResult, Side, SlogdetOptions, SolveOptions, SvdOptions, SvdShape,
    SvdvalsOptions, Transpose, TriangularSolveOptions, Uplo,
};
pub use crate::{Eig, Eigh, FullPivLu, Lu, Qr, Slogdet, Svd};
pub use crate::{LinalgBackend, LinalgInput, LinalgScalar, TensorLinalgExt, TracedTensorLinalgExt};

pub use tenferro_runtime::{Tensor, TensorRead, TracedTensor, TypedTensor};

#[cfg(feature = "autodiff")]
pub use crate::EagerSessionLinalgExt;
#[cfg(feature = "autodiff")]
pub use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
