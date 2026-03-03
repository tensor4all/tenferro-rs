use crate::context::{set_global_context, with_global_context, GlobalContextGuard};
use crate::{Error, Result};
use tenferro_prims::{CpuContext, CudaContext, RocmContext};

/// Runtime execution context used by builder `.run()` entry points.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{set_default_runtime, RuntimeContext};
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
    /// use tenferro_dyadtensor::RuntimeContext;
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
/// use tenferro_dyadtensor::{set_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// ```
pub fn set_default_runtime(ctx: RuntimeContext) -> GlobalContextGuard<RuntimeContext> {
    set_global_context(ctx)
}

/// Runs `f` with the default runtime context.
///
/// Returns [`Error::RuntimeNotConfigured`] when runtime is not configured.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{set_default_runtime, with_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let name = with_default_runtime(|rt| Ok(rt.name())).unwrap();
/// assert_eq!(name, "cpu");
/// ```
pub fn with_default_runtime<R>(f: impl FnOnce(&mut RuntimeContext) -> Result<R>) -> Result<R> {
    with_global_context::<RuntimeContext, _>(f).map_err(|err| match err {
        Error::MissingGlobalContext { .. } => Error::RuntimeNotConfigured,
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_roundtrip() {
        let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let runtime = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
        assert_eq!(runtime, "cpu");
    }
}
