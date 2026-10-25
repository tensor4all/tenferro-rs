//! Helpers shared by the crate's unit tests.

use crate::{CpuEngine, Pool};
use tenferro_tensor::Engine;

/// A CPU engine with `threads` workers: a serial engine for one thread, a
/// pooled engine otherwise.
pub(crate) fn cpu_engine(threads: usize) -> crate::Result<CpuEngine> {
    match threads {
        0 => Err(crate::Error::invalid_argument(
            "test_support::cpu_engine",
            "configuration",
            "thread count must be at least 1",
        )),
        1 => Ok(CpuEngine::serial()),
        threads => Pool::builder()
            .threads(threads)
            .build()
            .map(CpuEngine::new)
            .map_err(|source| crate::Error::backend_source("test_support::cpu_engine", source)),
    }
}

/// Something the tests can view as a type-erased engine.
pub(crate) trait TestEngine {
    fn as_engine(&mut self) -> &mut dyn Engine;
}

impl TestEngine for CpuEngine {
    fn as_engine(&mut self) -> &mut dyn Engine {
        self
    }
}

impl<T: TestEngine + ?Sized> TestEngine for &mut T {
    fn as_engine(&mut self) -> &mut dyn Engine {
        (**self).as_engine()
    }
}

/// Run `f` on `engine` as a type-erased engine.
pub(crate) fn on_engine<E: TestEngine + ?Sized, R>(
    engine: &mut E,
    f: impl FnOnce(&mut dyn Engine) -> R,
) -> crate::Result<R> {
    Ok(f(engine.as_engine()))
}

/// Run `f` on the engine itself (kept so older call sites read unchanged).
pub(crate) fn with_cpu_session<R>(
    engine: &mut CpuEngine,
    f: impl FnOnce(&mut CpuEngine) -> R,
) -> R {
    f(engine)
}

/// Run a closure on an engine viewed as `&mut dyn Engine`.
pub(crate) trait RunErased: TestEngine {
    fn run_erased<R>(&mut self, f: impl FnOnce(&mut dyn Engine) -> R) -> crate::Result<R> {
        Ok(f(self.as_engine()))
    }
}

impl<T: TestEngine + ?Sized> RunErased for T {}

#[cfg(test)]
mod tests;
