mod context;
#[cfg(test)]
pub(crate) mod contracts;
#[cfg(test)]
pub(crate) mod dispatch;

use crate::Result;

/// Guard returned by [`set_default_runtime`].
///
/// When dropped, the previous runtime context is restored.
pub use tenferro_internal_runtime::DefaultRuntimeGuard;

/// Runtime execution context used by builder `.run()` entry points.
///
/// Current status:
///
/// - `Cpu`: supported by builder `.run()` paths.
/// - `Cuda`/`Rocm`: accepted as context values, but current builder execution
///   paths return [`crate::Error::UnsupportedRuntimeOp`].
///
/// # Examples
///
/// ```rust
/// use tenferro::{set_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// ```
pub use tenferro_internal_runtime::RuntimeContext;

/// Sets the default runtime context for builder `.run()`.
///
/// # Examples
///
/// ```rust
/// use tenferro::{set_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// ```
pub fn set_default_runtime(ctx: RuntimeContext) -> DefaultRuntimeGuard {
    tenferro_internal_runtime::set_default_runtime(ctx)
}

/// Runs `f` with an explicitly supplied runtime installed for the duration of
/// the closure.
///
/// Any previously configured default runtime is restored afterwards, even when
/// `f` returns an error.
///
/// # Examples
///
/// ```rust
/// use tenferro::{runtime, RuntimeContext, Tensor};
/// use tenferro_prims::CpuContext;
///
/// let out = runtime::with_runtime(RuntimeContext::Cpu(CpuContext::new(1)), || {
///     let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])?;
///     x.sum()
/// });
/// assert!(out.is_ok());
/// ```
pub fn with_runtime<R>(ctx: RuntimeContext, f: impl FnOnce() -> Result<R>) -> Result<R> {
    tenferro_internal_runtime::with_runtime(ctx, f)
}

/// Runs `f` with the default runtime context.
///
/// Returns [`crate::Error::RuntimeNotConfigured`] when runtime is not
/// configured.
///
/// # Examples
///
/// ```rust
/// use tenferro::{set_default_runtime, with_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let name = with_default_runtime(|rt| Ok(rt.name())).unwrap();
/// assert_eq!(name, "cpu");
/// ```
pub fn with_default_runtime<R>(f: impl FnOnce(&mut RuntimeContext) -> Result<R>) -> Result<R> {
    tenferro_internal_runtime::with_default_runtime(f)
}
