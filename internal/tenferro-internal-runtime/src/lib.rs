//! Internal shared runtime holder for `tenferro` surface crates.
//!
//! # Examples
//!
//! ```
//! use tenferro_internal_runtime::{set_default_runtime, with_default_runtime, RuntimeContext};
//! use tenferro_prims::CpuContext;
//!
//! let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
//! let name = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
//! assert_eq!(name, "cpu");
//! ```

mod context;
pub mod contracts;
pub mod dispatch;

pub use context::{
    set_default_runtime, with_default_runtime, with_runtime, DefaultRuntimeGuard, RuntimeContext,
};

#[cfg(test)]
mod tests;
