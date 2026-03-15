mod context;
pub(crate) mod contracts;
pub(crate) mod dispatch;

use crate::Result;
use tenferro_prims::{CpuContext, CudaContext, RocmContext};

pub use context::DefaultRuntimeGuard;

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
pub enum RuntimeContext {
    /// CPU runtime context.
    Cpu(CpuContext),
    /// CUDA runtime context.
    Cuda(CudaContext),
    /// ROCm runtime context.
    Rocm(RocmContext),
}

impl RuntimeContext {
    /// Returns the runtime name.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::RuntimeContext;
    /// use tenferro_prims::CpuContext;
    ///
    /// let rt = RuntimeContext::Cpu(CpuContext::new(1));
    /// assert_eq!(rt.name(), "cpu");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            Self::Cuda(_) => "cuda",
            Self::Rocm(_) => "rocm",
        }
    }
}

impl From<CpuContext> for RuntimeContext {
    fn from(value: CpuContext) -> Self {
        Self::Cpu(value)
    }
}

impl From<CudaContext> for RuntimeContext {
    fn from(value: CudaContext) -> Self {
        Self::Cuda(value)
    }
}

impl From<RocmContext> for RuntimeContext {
    fn from(value: RocmContext) -> Self {
        Self::Rocm(value)
    }
}

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
    context::set_runtime_context(ctx)
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
    context::with_runtime_context(f)
}

#[cfg(test)]
mod tests;
