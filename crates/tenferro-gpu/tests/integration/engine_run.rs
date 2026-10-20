//! Run a closure on an engine through the backend-neutral `Engine` surface,
//! the way generic callers reach it.

#![allow(dead_code)]

pub trait EngineRun {
    /// Run `f` on `self`.
    ///
    /// # Errors
    ///
    /// This helper is infallible: it returns [`Ok`] unconditionally and never
    /// constructs a [`tenferro_tensor::Error`], so its error side is
    /// effectively [`std::convert::Infallible`].
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
