//! Common eager and traced automatic-differentiation entry points.

pub use crate::{
    AdContext, AdContextBuilder, EagerNoGradGuard, EagerRuntime, EagerTensor, TracedTensorAdExt,
};
pub use tenferro_runtime::{Tensor, TracedTensor};
