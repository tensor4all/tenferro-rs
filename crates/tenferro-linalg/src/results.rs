//! Structured results of linear-algebra decompositions (the #2552 grammar).

/// Thin or full singular value decomposition `A = U diag(S) Vh`.
///
/// Field order equals the former tuple order `(U, S, Vh)`.
///
/// - `u`: left singular vectors
/// - `s`: singular values
/// - `vh`: conjugate transpose of the right singular vectors
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Svd;
///
/// let r = Svd { u: 1, s: 2.0, vh: 3 };
/// let Svd { u, s, vh } = r;
/// assert_eq!((u, s, vh), (1, 2.0, 3));
/// ```
#[derive(Clone, Debug)]
pub struct Svd<V, R = V> {
    /// See the type-level documentation.
    pub u: V,
    /// See the type-level documentation.
    pub s: R,
    /// See the type-level documentation.
    pub vh: V,
}

/// QR decomposition `A = Q R`.
///
/// Field order equals the former tuple order `(Q, R)`.
///
/// - `q`: orthonormal factor
/// - `r`: upper-triangular factor
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Qr;
///
/// let r = Qr { q: 1, r: 2 };
/// let Qr { q, r } = r;
/// assert_eq!((q, r), (1, 2));
/// ```
#[derive(Clone, Debug)]
pub struct Qr<V> {
    /// See the type-level documentation.
    pub q: V,
    /// See the type-level documentation.
    pub r: V,
}

/// Partial-pivot LU decomposition `P A = L U`.
///
/// Field order equals the former tuple order `(P, L, U, parity)`.
///
/// - `p`: the permutation MATRIX tensor (not a pivot index vector)
/// - `l`: unit lower-triangular factor
/// - `u`: upper-triangular factor
/// - `parity`: permutation parity
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Lu;
///
/// let r = Lu { p: 1, l: 2, u: 3, parity: 4.0 };
/// let Lu { p, l, u, parity } = r;
/// assert_eq!((p, l, u, parity), (1, 2, 3, 4.0));
/// ```
#[derive(Clone, Debug)]
pub struct Lu<V, R = V> {
    /// See the type-level documentation.
    pub p: V,
    /// See the type-level documentation.
    pub l: V,
    /// See the type-level documentation.
    pub u: V,
    /// See the type-level documentation.
    pub parity: R,
}

/// Full-pivot LU decomposition `P A Q = L U`.
///
/// Field order equals the former tuple order `(P, L, U, Q, parity)`.
///
/// - `p`: row permutation MATRIX tensor (not a pivot index vector)
/// - `l`: unit lower-triangular factor
/// - `u`: upper-triangular factor
/// - `q`: column permutation matrix tensor
/// - `parity`: permutation parity
///
/// # Examples
///
/// ```
/// use tenferro_linalg::FullPivLu;
///
/// let r = FullPivLu { p: 1, l: 2, u: 3, q: 4, parity: 5.0 };
/// let FullPivLu { p, l, u, q, parity } = r;
/// assert_eq!((p, l, u, q, parity), (1, 2, 3, 4, 5.0));
/// ```
#[derive(Clone, Debug)]
pub struct FullPivLu<V, R = V> {
    /// See the type-level documentation.
    pub p: V,
    /// See the type-level documentation.
    pub l: V,
    /// See the type-level documentation.
    pub u: V,
    /// See the type-level documentation.
    pub q: V,
    /// See the type-level documentation.
    pub parity: R,
}

/// General eigendecomposition.
///
/// Field order equals the former tuple order `(values, vectors)`.
///
/// - `values`: eigenvalues
/// - `vectors`: eigenvectors
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Eig;
///
/// let r = Eig { values: 1, vectors: 2 };
/// let Eig { values, vectors } = r;
/// assert_eq!((values, vectors), (1, 2));
/// ```
#[derive(Clone, Debug)]
pub struct Eig<C> {
    /// See the type-level documentation.
    pub values: C,
    /// See the type-level documentation.
    pub vectors: C,
}

/// Hermitian/symmetric eigendecomposition.
///
/// Field order equals the former tuple order `(values, vectors)`.
///
/// - `values`: real eigenvalues
/// - `vectors`: eigenvectors
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Eigh;
///
/// let r = Eigh { values: 1.0, vectors: 2 };
/// let Eigh { values, vectors } = r;
/// assert_eq!((values, vectors), (1.0, 2));
/// ```
#[derive(Clone, Debug)]
pub struct Eigh<V, R = V> {
    /// See the type-level documentation.
    pub values: R,
    /// See the type-level documentation.
    pub vectors: V,
}

/// Sign and log-absolute-value of the determinant.
///
/// Field order equals the former tuple order `(sign, log|det|)`.
///
/// - `sign`: sign (or phase) of the determinant
/// - `log_abs_det`: `log|det|`
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Slogdet;
///
/// let r = Slogdet { sign: 1, log_abs_det: 2.0 };
/// let Slogdet { sign, log_abs_det } = r;
/// assert_eq!((sign, log_abs_det), (1, 2.0));
/// ```
#[derive(Clone, Debug)]
pub struct Slogdet<V, R = V> {
    /// See the type-level documentation.
    pub sign: V,
    /// See the type-level documentation.
    pub log_abs_det: R,
}
