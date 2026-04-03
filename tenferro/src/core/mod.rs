pub(crate) mod dynamic;

#[cfg(test)]
pub(crate) use dynamic::DynTensor;
#[cfg(test)]
#[doc(hidden)]
pub(crate) use dynamic::DynTensorTyped;
pub use dynamic::{
    CholeskyExResult, EigResult, EigenResult, EighResult, InvExResult, LstsqResult,
    LuFactorExResult, LuFactorResult, LuPivot, LuResult, QrResult, ScalarType, SlogdetResult,
    SolveExResult, SvdResult, Tensor, TensorScalarDowncast, TypedTensorRef,
};
pub use value::AdMode;
