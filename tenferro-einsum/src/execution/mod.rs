pub(crate) mod backend;
pub(crate) mod chain;
pub(crate) mod dispatch;
pub(crate) mod execute;
pub(crate) mod pool;
pub(crate) mod strict_binary;
pub(crate) mod unary;
pub(crate) mod util;

pub use backend::{BackendContext, EinsumBackend};
#[cfg(feature = "profile-dispatch")]
pub use dispatch::print_and_reset_profile;
