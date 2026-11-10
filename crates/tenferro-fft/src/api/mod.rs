//! The #2552 call grammar for FFTs: `op` and `op_with` on three layers.
//!
//! [`registry`] declares every operation once; the traits it generates
//! ([`FftOps`], [`TracedFftOps`], `EagerFftOps`) forward to the hand-written
//! bodies in [`concrete`], [`traced`] and `eager`.

pub(crate) mod concrete;
#[cfg(feature = "autodiff")]
pub(crate) mod eager;
mod options;
mod registry;
mod repr;
pub(crate) mod traced;

#[cfg(feature = "autodiff")]
pub use registry::EagerFftOps;
pub use registry::{FftOps, TracedFftOps};

pub use options::{FftOptions, FftnOptions};
pub use repr::{FftRepr, FftScalar};

#[cfg(test)]
mod tests;
