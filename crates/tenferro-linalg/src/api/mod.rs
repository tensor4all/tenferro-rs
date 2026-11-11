//! The #2552 call grammar for linear algebra: `op`, `op_with`, `op_into`.
//!
//! [`registry`] declares every operation once; the traits it generates
//! ([`LinalgOps`], [`TracedLinalgOps`], [`EagerLinalgOps`]) forward to the
//! hand-written bodies in [`concrete`], [`traced`] and [`eager`].

pub(crate) mod concrete;
#[cfg(feature = "autodiff")]
pub(crate) mod eager;
mod options;
mod registry;
mod repr;
pub(crate) mod traced;

#[cfg(feature = "autodiff")]
pub use registry::EagerLinalgOps;
pub use registry::{LinalgOps, TracedLinalgOps};

pub use options::{
    CholeskyOptions, EigvalshOptions, NormOptions, NormOrd, PinvOptions, SvdvalsOptions,
    TriangularSolveOptions,
};
pub use repr::LinalgRepr;

#[cfg(test)]
mod tests;
