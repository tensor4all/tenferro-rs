use tenferro_tensor::DType;

/// FFT normalization convention.
///
/// `Backward` matches NumPy, JAX, and PyTorch defaults: the forward transform
/// is unscaled and the inverse transform is scaled by `1 / n`.
///
/// # Examples
///
/// ```
/// use tenferro_fft::FftNorm;
///
/// assert_eq!(FftNorm::default(), FftNorm::Backward);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FftNorm {
    /// Scale inverse transforms by `1 / n`.
    #[default]
    Backward,
    /// Scale forward transforms by `1 / n`.
    Forward,
    /// Scale both forward and inverse transforms by `1 / sqrt(n)`.
    Ortho,
}

impl FftNorm {
    #[cfg(feature = "autodiff")]
    pub(crate) fn c2c_adjoint(self) -> Self {
        match self {
            Self::Backward => Self::Forward,
            Self::Forward => Self::Backward,
            Self::Ortho => Self::Ortho,
        }
    }
}

/// One-dimensional FFT operation requested from an [`FftBackend`](crate::FftBackend).
///
/// # Examples
///
/// ```
/// use tenferro_fft::FftOperation;
///
/// assert_ne!(FftOperation::C2cForward, FftOperation::C2cInverse);
/// assert_ne!(FftOperation::R2cFull, FftOperation::R2cOnesided);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FftOperation {
    /// Forward complex-to-complex FFT.
    C2cForward,
    /// Inverse complex-to-complex FFT.
    C2cInverse,
    /// Real-to-complex FFT returning the full complex spectrum.
    R2cFull,
    /// Real-to-complex FFT returning the non-negative-frequency spectrum.
    R2cOnesided,
    /// Complex-to-real FFT consuming a one-sided Hermitian spectrum.
    C2r,
}

impl FftOperation {
    #[cfg(feature = "autodiff")]
    pub(crate) const fn is_c2c(self) -> bool {
        matches!(self, Self::C2cForward | Self::C2cInverse)
    }

    pub(crate) const fn is_forward(self) -> bool {
        matches!(self, Self::C2cForward | Self::R2cFull | Self::R2cOnesided)
    }

    pub(crate) const fn is_onesided(self) -> bool {
        matches!(self, Self::R2cOnesided)
    }

    /// Return the operation of the `index`-th declared axis of a `count`-axis request.
    ///
    /// `count == 1` returns `self`, so a one-axis request is exactly the
    /// historical one-dimensional transform. For the mixed real transforms the
    /// last declared axis carries the real transform while every other axis is
    /// complex, which is the NumPy `rfftn`/`irfftn` convention.
    pub(crate) const fn step_operation(self, index: usize, count: usize) -> Self {
        if index + 1 < count {
            match self {
                Self::R2cFull | Self::R2cOnesided => Self::C2cForward,
                Self::C2r => Self::C2cInverse,
                Self::C2cForward | Self::C2cInverse => self,
            }
        } else {
            self
        }
    }
}

/// Validated, backend-neutral description of one FFT request over one or more axes.
///
/// Backends receive this value only after the public FFT surface has validated
/// dtype, rank, axes, requested lengths, and inverse-spectrum shape. The layout
/// requirement is explicit so a backend can reject unsupported storage without
/// silently materializing or transferring the input.
///
/// A spec carries one or more axes. The n-dimensional executor splits a
/// multi-axis request into one single-axis spec per axis, so every spec a
/// backend actually executes names exactly one axis; [`Self::axes`] stays
/// useful for tests and diagnostics. Axis order is significant only for the
/// mixed real transforms, whose real transform is the last declared axis.
///
/// # Examples
///
/// ```
/// use tenferro_fft::{FftOperation, FftPlanSpec};
///
/// fn inspect(spec: &FftPlanSpec) {
///     let operation: FftOperation = spec.operation();
///     assert_eq!(operation, spec.operation());
///     assert!(spec.requires_compact_column_major());
///     assert!(!spec.axes().is_empty());
/// }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FftPlanSpec {
    operation: FftOperation,
    axes: Vec<usize>,
    lengths: Option<Vec<usize>>,
    norm: FftNorm,
    input_dtype: DType,
    input_shape: Vec<usize>,
    requires_compact_column_major: bool,
}

impl FftPlanSpec {
    /// Build a spec from already-validated axes and per-axis lengths.
    ///
    /// `axes` must be non-empty and `lengths`, when present, must match its
    /// length. Every public constructor validates that before reaching here.
    pub(crate) fn new(
        operation: FftOperation,
        axes: Vec<usize>,
        lengths: Option<Vec<usize>>,
        norm: FftNorm,
        input_dtype: DType,
        input_shape: Vec<usize>,
    ) -> Self {
        debug_assert!(!axes.is_empty(), "an FFT spec names at least one axis");
        debug_assert!(
            lengths
                .as_ref()
                .is_none_or(|lengths| lengths.len() == axes.len()),
            "one requested length per axis"
        );
        Self {
            operation,
            axes,
            lengths,
            norm,
            input_dtype,
            input_shape,
            requires_compact_column_major: true,
        }
    }

    /// Return the transform operation of the last declared axis.
    pub const fn operation(&self) -> FftOperation {
        self.operation
    }

    /// Return every validated axis of this request.
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    /// Return the caller-requested transform length of every axis, if supplied.
    pub fn lengths(&self) -> Option<&[usize]> {
        self.lengths.as_deref()
    }

    /// Return the non-negative, validated axis executed by a backend.
    ///
    /// Multi-axis requests are expanded into one single-axis spec per axis
    /// before a backend sees them, so this is the axis of the step to run now.
    pub fn normalized_axis(&self) -> usize {
        // INVARIANT: `new` requires a non-empty axis list.
        self.axes[0]
    }

    /// Return the caller-requested transform length of the executed step.
    pub fn requested_len(&self) -> Option<usize> {
        self.lengths.as_ref().map(|lengths| lengths[0])
    }

    /// Return the requested normalization convention.
    pub const fn norm(&self) -> FftNorm {
        self.norm
    }

    /// Return the validated input dtype.
    pub const fn input_dtype(&self) -> DType {
        self.input_dtype
    }

    /// Return the validated input shape.
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Return whether execution requires compact column-major storage.
    pub const fn requires_compact_column_major(&self) -> bool {
        self.requires_compact_column_major
    }
}
