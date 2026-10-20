//! Run a closure on an engine through the backend-neutral `Engine` surface,
//! the way generic callers reach it.

#![allow(dead_code)]

pub trait EngineRun {
    fn run<R>(
        &mut self,
        f: impl FnOnce(&mut dyn tenferro_tensor::Engine) -> R,
    ) -> tenferro_tensor::Result<R>;
}

impl<E: tenferro_tensor::Engine> EngineRun for E {
    fn run<R>(
        &mut self,
        f: impl FnOnce(&mut dyn tenferro_tensor::Engine) -> R,
    ) -> tenferro_tensor::Result<R> {
        Ok(f(self))
    }
}
