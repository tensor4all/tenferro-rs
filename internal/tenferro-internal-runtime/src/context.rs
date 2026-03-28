use std::cell::RefCell;

use tenferro_internal_error::{Error, Result};
use tenferro_prims::{CpuContext, CudaContext, RocmContext};

thread_local! {
    static DEFAULT_RUNTIME: RefCell<Option<RuntimeContext>> = const { RefCell::new(None) };
}

/// Runtime execution context used by builder `.run()` entry points.
///
/// Current status:
///
/// - `Cpu`: supported by builder `.run()` paths.
/// - `Cuda`/`Rocm`: accepted as context values, but current builder execution
///   paths may reject runtime-specific operations elsewhere.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_runtime::{set_default_runtime, RuntimeContext};
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
    /// use tenferro_internal_runtime::RuntimeContext;
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

/// Guard returned by [`set_default_runtime`].
///
/// When dropped, the previous runtime context is restored.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_runtime::{set_default_runtime, with_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let name = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
/// assert_eq!(name, "cpu");
/// ```
pub struct DefaultRuntimeGuard {
    previous: Option<RuntimeContext>,
}

impl Drop for DefaultRuntimeGuard {
    fn drop(&mut self) {
        DEFAULT_RUNTIME.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

/// Sets the default runtime context for builder `.run()`.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_runtime::{set_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// ```
pub fn set_default_runtime(ctx: RuntimeContext) -> DefaultRuntimeGuard {
    let previous = DEFAULT_RUNTIME.with(|slot| slot.borrow_mut().replace(ctx));
    DefaultRuntimeGuard { previous }
}

/// Runs `f` with the default runtime context.
///
/// Returns [`Error::RuntimeNotConfigured`] when runtime is not configured.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_runtime::{set_default_runtime, with_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let name = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
/// assert_eq!(name, "cpu");
/// ```
pub fn with_default_runtime<R>(f: impl FnOnce(&mut RuntimeContext) -> Result<R>) -> Result<R> {
    DEFAULT_RUNTIME.with(|slot| {
        let mut slot = slot.borrow_mut();
        let ctx = slot.as_mut().ok_or(Error::RuntimeNotConfigured)?;
        f(ctx)
    })
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
/// use tenferro_internal_runtime::{with_default_runtime, with_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
///
/// let name = with_runtime(RuntimeContext::Cpu(CpuContext::new(1)), || {
///     with_default_runtime(|ctx| Ok(ctx.name()))
/// })
/// .unwrap();
/// assert_eq!(name, "cpu");
/// ```
pub fn with_runtime<R>(ctx: RuntimeContext, f: impl FnOnce() -> Result<R>) -> Result<R> {
    let _guard = set_default_runtime(ctx);
    f()
}
