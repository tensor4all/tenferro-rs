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
    pub(crate) const fn is_c2c(self) -> bool {
        matches!(self, Self::C2cForward | Self::C2cInverse)
    }

    pub(crate) const fn is_forward(self) -> bool {
        matches!(self, Self::C2cForward | Self::R2cFull | Self::R2cOnesided)
    }

    pub(crate) const fn is_onesided(self) -> bool {
        matches!(self, Self::R2cOnesided)
    }
}

/// Validated, backend-neutral description of one FFT request.
///
/// Backends receive this value only after the public FFT surface has validated
/// dtype, rank, axis, requested length, and inverse-spectrum shape. The layout
/// requirement is explicit so a backend can reject unsupported storage without
/// silently materializing or transferring the input.
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
/// }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FftPlanSpec {
    operation: FftOperation,
    normalized_axis: usize,
    requested_len: Option<usize>,
    norm: FftNorm,
    input_dtype: DType,
    input_shape: Vec<usize>,
    requires_compact_column_major: bool,
}

impl FftPlanSpec {
    pub(crate) fn new(
        operation: FftOperation,
        normalized_axis: usize,
        requested_len: Option<usize>,
        norm: FftNorm,
        input_dtype: DType,
        input_shape: Vec<usize>,
    ) -> Self {
        Self {
            operation,
            normalized_axis,
            requested_len,
            norm,
            input_dtype,
            input_shape,
            requires_compact_column_major: true,
        }
    }

    /// Return the transform operation.
    pub const fn operation(&self) -> FftOperation {
        self.operation
    }

    /// Return the non-negative, validated axis.
    pub const fn normalized_axis(&self) -> usize {
        self.normalized_axis
    }

    /// Return the caller-requested transform length, if one was supplied.
    pub const fn requested_len(&self) -> Option<usize> {
        self.requested_len
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
