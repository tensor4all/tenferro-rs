pub mod contracts;
pub mod dispatch;

pub use tenferro_internal_runtime::{
    set_default_runtime, with_default_runtime, with_runtime, DefaultRuntimeGuard, RuntimeContext,
};
