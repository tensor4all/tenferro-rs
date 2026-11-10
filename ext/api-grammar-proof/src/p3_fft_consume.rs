//! Prototype 3: consuming in-place FFT as a session-receiver operation.
//!
//! Adapter over `tenferro_fft::EagerTensorFftExt`, which takes the engine as an
//! argument. `EagerSession::engine()` is public, so the session-receiver form
//! needs no internal hook; see `findings/p3.md`.

use tenferro_ad::EagerSession;
use tenferro_ad::EagerTensor;
use tenferro_fft::{EagerFftInPlaceError, EagerTensorFftExt, FftNorm};
use tenferro_runtime::{Error, ErrorPhase};

use crate::policy::MaterializationPolicy;

/// Options of a consuming forward FFT.
///
/// There is no length field: the current in-place API fixes `n` to the input
/// extent (shape and allocation are preserved), so a length option would have
/// nothing to bind to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FftOptions {
    /// Transform axis; negative values count from the last axis.
    pub axis: isize,
    /// Normalization convention.
    pub norm: FftNorm,
}

impl Default for FftOptions {
    fn default() -> Self {
        Self {
            axis: -1,
            norm: FftNorm::Backward,
        }
    }
}

/// Options of a consuming inverse FFT (same shape as [`FftOptions`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IfftOptions {
    /// Transform axis; negative values count from the last axis.
    pub axis: isize,
    /// Normalization convention.
    pub norm: FftNorm,
}

impl Default for IfftOptions {
    fn default() -> Self {
        Self {
            axis: -1,
            norm: FftNorm::Backward,
        }
    }
}

/// Session-receiver consuming FFT operations.
pub trait SessionFftConsumeExt {
    /// Consume `input` and return its FFT computed in the same allocation.
    ///
    /// # Errors
    /// `Rejected` returns the unchanged owner; `Execution` has consumed it.
    fn fft_in_place(
        &mut self,
        input: EagerTensor,
        options: &FftOptions,
        policy: MaterializationPolicy,
    ) -> Result<EagerTensor, EagerFftInPlaceError>;

    /// Consume `input` and return its inverse FFT computed in the same allocation.
    ///
    /// # Errors
    /// Same contract as [`Self::fft_in_place`].
    fn ifft_in_place(
        &mut self,
        input: EagerTensor,
        options: &IfftOptions,
        policy: MaterializationPolicy,
    ) -> Result<EagerTensor, EagerFftInPlaceError>;
}

fn forbid_not_implemented(input: EagerTensor, op: &'static str) -> EagerFftInPlaceError {
    EagerFftInPlaceError::Rejected {
        input: Box::new(input),
        source: Error::runtime_state(
            op,
            ErrorPhase::Execution,
            "MaterializationPolicy::Forbid enforcement is not implemented",
        ),
    }
}

impl SessionFftConsumeExt for EagerSession<'_> {
    fn fft_in_place(
        &mut self,
        input: EagerTensor,
        options: &FftOptions,
        policy: MaterializationPolicy,
    ) -> Result<EagerTensor, EagerFftInPlaceError> {
        if policy == MaterializationPolicy::Forbid {
            return Err(forbid_not_implemented(input, "fft_in_place"));
        }
        input.fft_in_place(self.engine(), options.axis, options.norm)
    }

    fn ifft_in_place(
        &mut self,
        input: EagerTensor,
        options: &IfftOptions,
        policy: MaterializationPolicy,
    ) -> Result<EagerTensor, EagerFftInPlaceError> {
        if policy == MaterializationPolicy::Forbid {
            return Err(forbid_not_implemented(input, "ifft_in_place"));
        }
        input.ifft_in_place(self.engine(), options.axis, options.norm)
    }
}
