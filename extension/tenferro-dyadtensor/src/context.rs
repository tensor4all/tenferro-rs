use std::cell::RefCell;

use crate::runtime::RuntimeContext;
use crate::{Error, Result};

thread_local! {
    static DEFAULT_RUNTIME: RefCell<Option<RuntimeContext>> = const { RefCell::new(None) };
}

/// Guard returned by [`crate::set_default_runtime`].
///
/// When dropped, the previous runtime context is restored.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{set_default_runtime, with_default_runtime, RuntimeContext};
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

pub(crate) fn set_runtime_context(ctx: RuntimeContext) -> DefaultRuntimeGuard {
    let previous = DEFAULT_RUNTIME.with(|slot| slot.borrow_mut().replace(ctx));
    DefaultRuntimeGuard { previous }
}

pub(crate) fn with_runtime_context<R>(
    f: impl FnOnce(&mut RuntimeContext) -> Result<R>,
) -> Result<R> {
    DEFAULT_RUNTIME.with(|slot| {
        let mut slot = slot.borrow_mut();
        let ctx = slot.as_mut().ok_or(Error::RuntimeNotConfigured)?;
        f(ctx)
    })
}

#[cfg(test)]
mod tests;
