//! Options of the FFT operations (the #2552 grammar).
//!
//! Both types are `#[non_exhaustive]`: build them with `Default::default()`
//! and the setters, whose names equal the field names.

use crate::FftNorm;

/// Options of the one-axis transforms `fft`, `ifft`, `rfft` and `irfft`.
///
/// - `n`: transform length along `axis`; the input is cropped or zero-padded
///   to it (for `irfft`, the length of the real output). `None` keeps the
///   input length (for `irfft`, `2 * (m - 1)`).
/// - `axis`: the transformed axis; negative values count from the end. The
///   default is `-1`, the last axis.
/// - `norm`: the normalization; the default is [`FftNorm::Backward`].
///
/// The in-place transforms (`fft_in_place`, `ifft_in_place`) cannot change
/// the length, so they reject `n: Some(_)` with a validation error.
///
/// # Examples
///
/// ```
/// use tenferro_fft::{FftNorm, FftOptions};
///
/// let opts = FftOptions::default();
/// assert_eq!((opts.n, opts.axis, opts.norm), (None, -1, FftNorm::Backward));
///
/// let opts = FftOptions::default().n(Some(8)).axis(0).norm(FftNorm::Ortho);
/// assert_eq!((opts.n, opts.axis, opts.norm), (Some(8), 0, FftNorm::Ortho));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FftOptions {
    /// Transform length along `axis`; `None` keeps the input length.
    pub n: Option<usize>,
    /// The transformed axis; negative values count from the end.
    pub axis: isize,
    /// The normalization.
    pub norm: FftNorm,
}

impl Default for FftOptions {
    fn default() -> Self {
        Self {
            n: None,
            axis: -1,
            norm: FftNorm::Backward,
        }
    }
}

impl FftOptions {
    /// Return options with the requested transform length.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::FftOptions;
    ///
    /// assert_eq!(FftOptions::default().n(Some(4)).n, Some(4));
    /// ```
    pub fn n(mut self, n: Option<usize>) -> Self {
        self.n = n;
        self
    }

    /// Return options with the requested axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::FftOptions;
    ///
    /// assert_eq!(FftOptions::default().axis(0).axis, 0);
    /// ```
    pub fn axis(mut self, axis: isize) -> Self {
        self.axis = axis;
        self
    }

    /// Return options with the requested normalization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::{FftNorm, FftOptions};
    ///
    /// assert_eq!(FftOptions::default().norm(FftNorm::Forward).norm, FftNorm::Forward);
    /// ```
    pub fn norm(mut self, norm: FftNorm) -> Self {
        self.norm = norm;
        self
    }
}

/// Options of the N-dimensional transforms `fftn`, `ifftn`, `rfftn` and
/// `irfftn`.
///
/// - `shape`: one transform length per transformed axis (for `irfftn`, the
///   real output lengths); `None` keeps the input lengths.
/// - `axes`: the transformed axes; negative values count from the end. Empty
///   (the default) follows NumPy: every axis when `shape` is `None`, otherwise
///   the last `shape.len()` axes. For the real transforms the last listed
///   axis is the real axis.
/// - `norm`: the normalization, applied once to the whole transform.
///
/// # Examples
///
/// ```
/// use tenferro_fft::{FftNorm, FftnOptions};
///
/// let opts = FftnOptions::default();
/// assert!(opts.shape.is_none() && opts.axes.is_empty());
/// assert_eq!(opts.norm, FftNorm::Backward);
///
/// let opts = FftnOptions::default().axes(vec![0, -1]).shape(Some(vec![4, 4]));
/// assert_eq!(opts.axes, vec![0, -1]);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FftnOptions {
    /// One transform length per transformed axis; `None` keeps the input lengths.
    pub shape: Option<Vec<usize>>,
    /// The transformed axes; empty selects them as NumPy does.
    pub axes: Vec<isize>,
    /// The normalization.
    pub norm: FftNorm,
}

impl FftnOptions {
    /// Return options with the requested transform lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::FftnOptions;
    ///
    /// assert_eq!(FftnOptions::default().shape(Some(vec![2, 2])).shape, Some(vec![2, 2]));
    /// ```
    pub fn shape(mut self, shape: Option<Vec<usize>>) -> Self {
        self.shape = shape;
        self
    }

    /// Return options with the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::FftnOptions;
    ///
    /// assert_eq!(FftnOptions::default().axes(vec![-1]).axes, vec![-1]);
    /// ```
    pub fn axes(mut self, axes: Vec<isize>) -> Self {
        self.axes = axes;
        self
    }

    /// Return options with the requested normalization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::{FftNorm, FftnOptions};
    ///
    /// assert_eq!(FftnOptions::default().norm(FftNorm::Ortho).norm, FftNorm::Ortho);
    /// ```
    pub fn norm(mut self, norm: FftNorm) -> Self {
        self.norm = norm;
        self
    }

    /// The axes to transform for an input of `rank`: `axes` when given,
    /// otherwise every axis (no `shape`) or the last `shape.len()` axes.
    /// A rank-0 input yields no axis, which the shared validator rejects.
    pub(crate) fn resolved_axes(&self, rank: usize) -> Vec<isize> {
        if !self.axes.is_empty() {
            return self.axes.clone();
        }
        let count = self.shape.as_ref().map_or(rank, |s| s.len().min(rank));
        (rank - count..rank)
            .map(|axis| isize::try_from(axis).unwrap_or(isize::MAX))
            .collect()
    }
}
