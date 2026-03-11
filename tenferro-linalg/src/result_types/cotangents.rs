use super::*;

/// Cotangent (adjoint) for SVD outputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::SvdCotangent;
///
/// let cotangent = SvdCotangent::<f64> {
///     u: None,
///     s: None,
///     vt: None,
/// };
/// assert!(cotangent.s.is_none());
/// ```
#[derive(Debug)]
pub struct SvdCotangent<T: Scalar> {
    /// Cotangent for `u`.
    pub u: Option<Tensor<T>>,
    /// Cotangent for `s`.
    pub s: Option<Tensor<T>>,
    /// Cotangent for `vt`.
    pub vt: Option<Tensor<T>>,
}

/// Cotangent (adjoint) for QR outputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::QrCotangent;
///
/// let cotangent = QrCotangent::<f64> { q: None, r: None };
/// assert!(cotangent.q.is_none());
/// ```
#[derive(Debug)]
pub struct QrCotangent<T: Scalar> {
    /// Cotangent for `q`.
    pub q: Option<Tensor<T>>,
    /// Cotangent for `r`.
    pub r: Option<Tensor<T>>,
}

/// Cotangent (adjoint) for LU outputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::LuCotangent;
///
/// let cotangent = LuCotangent::<f64> { l: None, u: None };
/// assert!(cotangent.l.is_none());
/// ```
#[derive(Debug)]
pub struct LuCotangent<T: Scalar> {
    /// Cotangent for `l`.
    pub l: Option<Tensor<T>>,
    /// Cotangent for `u`.
    pub u: Option<Tensor<T>>,
}

/// Cotangent (adjoint) for eigendecomposition outputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::EigenCotangent;
///
/// let cotangent = EigenCotangent::<f64> {
///     values: None,
///     vectors: None,
/// };
/// assert!(cotangent.values.is_none());
/// ```
#[derive(Debug)]
pub struct EigenCotangent<T: Scalar> {
    /// Cotangent for eigenvalues.
    pub values: Option<Tensor<T>>,
    /// Cotangent for eigenvectors.
    pub vectors: Option<Tensor<T>>,
}

/// Cotangent (adjoint) for general eigendecomposition outputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::EigCotangent;
///
/// let cotangent = EigCotangent::<f64> {
///     values: None,
///     vectors: None,
/// };
/// assert!(cotangent.vectors.is_none());
/// ```
#[derive(Debug)]
pub struct EigCotangent<R: LinalgScalar<Real = R> + num_traits::Float> {
    /// Cotangent for eigenvalues.
    pub values: Option<Tensor<num_complex::Complex<R>>>,
    /// Cotangent for eigenvectors.
    pub vectors: Option<Tensor<num_complex::Complex<R>>>,
}

/// Cotangent (adjoint) for slogdet outputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::SlogdetCotangent;
///
/// let cotangent = SlogdetCotangent::<f64> { logabsdet: None };
/// assert!(cotangent.logabsdet.is_none());
/// ```
#[derive(Debug)]
pub struct SlogdetCotangent<T: Scalar> {
    /// Cotangent for `logabsdet`.
    pub logabsdet: Option<Tensor<T>>,
}
