use tenferro_tensor::Error;

use crate::QrGauge;

/// Four fixed-shape outputs of a rank-revealing QR factorization.
///
/// `Q` and `R` use `T`; the column permutation and numerical rank use `M`.
/// For dynamic, eager, and traced tensors, `T` and `M` are the same tensor
/// type. Typed tensors use [`crate::TypedRankRevealingQrResult`], whose metadata
/// tensors have scalar type `i64`.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::RankRevealingQrResult;
///
/// let result = RankRevealingQrResult {
///     q: "q",
///     r: "r",
///     column_permutation: vec![1_i64, 0],
///     rank: vec![2_i64],
/// };
/// assert_eq!(result.column_permutation, [1, 0]);
/// assert_eq!(result.rank, [2]);
/// ```
#[derive(Clone, Debug)]
pub struct RankRevealingQrResult<T, M = T> {
    /// Thin orthonormal factor with shape `[m, min(m, n), batch...]`.
    pub q: T,
    /// Upper-trapezoidal factor with shape `[min(m, n), n, batch...]`.
    pub r: T,
    /// Zero-based original column at each factor column, shaped `[n, batch...]`.
    pub column_permutation: M,
    /// Leading-prefix numerical rank, shaped `[batch...]`.
    pub rank: M,
}

/// Options for column-pivoted rank-revealing QR.
///
/// Rank is the length of the leading diagonal prefix satisfying
/// `abs(R[i,i]) > max(atol, rtol * abs(R[0,0]))`. Both tolerances must be
/// finite and non-negative. The default uses zero tolerances so callers choose
/// an application-specific numerical threshold explicitly.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{QrGauge, RankRevealingQrOptions};
///
/// let options = RankRevealingQrOptions::default()
///     .gauge(QrGauge::PositiveDiagonal)
///     .rtol(1.0e-10)
///     .atol(1.0e-14);
/// assert_eq!(options.rtol, 1.0e-10);
/// assert_eq!(options.atol, 1.0e-14);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankRevealingQrOptions {
    /// QR sign or phase convention. The default is [`QrGauge::Raw`].
    pub gauge: QrGauge,
    /// Relative diagonal threshold. The default is `0.0`.
    pub rtol: f64,
    /// Absolute diagonal threshold. The default is `0.0`.
    pub atol: f64,
}

impl Default for RankRevealingQrOptions {
    fn default() -> Self {
        Self {
            gauge: QrGauge::Raw,
            rtol: 0.0,
            atol: 0.0,
        }
    }
}

impl RankRevealingQrOptions {
    /// Return options with the requested QR gauge.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{QrGauge, RankRevealingQrOptions};
    ///
    /// let options = RankRevealingQrOptions::default().gauge(QrGauge::PositiveDiagonal);
    /// assert_eq!(options.gauge, QrGauge::PositiveDiagonal);
    /// ```
    pub fn gauge(mut self, gauge: QrGauge) -> Self {
        self.gauge = gauge;
        self
    }

    /// Return options with the requested relative rank tolerance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::RankRevealingQrOptions;
    ///
    /// let options = RankRevealingQrOptions::default().rtol(1.0e-8);
    /// assert_eq!(options.rtol, 1.0e-8);
    /// ```
    pub fn rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Return options with the requested absolute rank tolerance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::RankRevealingQrOptions;
    ///
    /// let options = RankRevealingQrOptions::default().atol(1.0e-12);
    /// assert_eq!(options.atol, 1.0e-12);
    /// ```
    pub fn atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }
}

pub(crate) fn validate_rank_revealing_qr_options(
    op: &'static str,
    options: RankRevealingQrOptions,
) -> tenferro_tensor::Result<()> {
    for (name, value) in [("rtol", options.rtol), ("atol", options.atol)] {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid_argument(
                op,
                name,
                format!("must be finite and non-negative, got {value}"),
            ));
        }
    }
    Ok(())
}
