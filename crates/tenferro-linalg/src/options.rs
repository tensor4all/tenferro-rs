//! Per-operation options and the enums that replace boolean flags (#2552 rule 3).
//!
//! Every linalg operation has an options type, including operations that have no
//! semantic option today: the short form `x.op(exec)` is `x.op_with(&Options::default(),
//! exec)`. Options are `#[non_exhaustive]`, so they are built from `Default` with the
//! `with_*` setters and gain fields without breaking callers. No option is a `bool`; each
//! two-way choice is an enum. Axes are `isize`, negative values counting from the end.
//!
//! The options of [`SvdOptions`], [`QrOptions`] and [`EighOptions`] live with the
//! extension op vocabulary in `extension.rs`, and [`RankRevealingQrOptions`] next to its
//! result type; this module holds the rest.
//!
//! [`SvdOptions`]: crate::SvdOptions
//! [`QrOptions`]: crate::QrOptions
//! [`EighOptions`]: crate::EighOptions
//! [`RankRevealingQrOptions`]: crate::RankRevealingQrOptions

/// Shape of the singular vectors returned by an SVD.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{SvdOptions, SvdShape};
///
/// assert_eq!(SvdOptions::default().shape, SvdShape::Thin);
/// let full = SvdOptions::default().with_shape(SvdShape::Full);
/// assert_eq!(full.shape, SvdShape::Full);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SvdShape {
    /// `U` is `[m, k]` and `Vᴴ` is `[k, n]` with `k = min(m, n)`.
    #[default]
    Thin,
    /// `U` is `[m, m]` and `Vᴴ` is `[n, n]` (full matrices).
    Full,
}

/// Which side of the unknown the triangular matrix multiplies.
///
/// `Left` solves `op(A) X = B`; `Right` solves `X op(A) = B`.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{Side, TriangularSolveOptions};
///
/// let options = TriangularSolveOptions::default().with_side(Side::Left);
/// assert_eq!(options.side, Side::Left);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    /// Solve `op(A) X = B`.
    Left,
    /// Solve `X op(A) = B`.
    #[default]
    Right,
}

/// Which triangle of the matrix holds the factor.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{TriangularSolveOptions, Uplo};
///
/// let options = TriangularSolveOptions::default().with_uplo(Uplo::Lower);
/// assert_eq!(options.uplo, Uplo::Lower);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Uplo {
    /// The lower triangle, diagonal included.
    Lower,
    /// The upper triangle, diagonal included.
    #[default]
    Upper,
}

/// Whether the triangular matrix is used as stored or transposed.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{Transpose, TriangularSolveOptions};
///
/// let options = TriangularSolveOptions::default().with_transpose(Transpose::Transpose);
/// assert_eq!(options.transpose, Transpose::Transpose);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Transpose {
    /// Use `A` as stored.
    #[default]
    NoTranspose,
    /// Use `Aᵀ` (the former `transpose_a = true`).
    Transpose,
}

/// Whether the diagonal of the triangular matrix is read or assumed to be one.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{Diagonal, TriangularSolveOptions};
///
/// let options = TriangularSolveOptions::default().with_diagonal(Diagonal::Unit);
/// assert_eq!(options.diagonal, Diagonal::Unit);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Diagonal {
    /// Read the stored diagonal.
    #[default]
    NonUnit,
    /// Assume a unit diagonal without reading it.
    Unit,
}

/// Whether a reduction keeps the reduced axes as length-one axes.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{KeepDims, NormOptions};
///
/// let options = NormOptions::default().with_keep_dims(KeepDims::Keep);
/// assert_eq!(options.keep_dims, KeepDims::Keep);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KeepDims {
    /// Remove the reduced axes.
    #[default]
    Drop,
    /// Keep each reduced axis with length one.
    Keep,
}

/// Options for `triangular_solve`.
///
/// The defaults are the all-`false` flags of the former four-`bool` signature
/// (`left_side`, `lower`, `transpose_a`, `unit_diagonal`), which are also the defaults of
/// `jax.lax.linalg.triangular_solve`.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{Diagonal, Side, Transpose, TriangularSolveOptions, Uplo};
///
/// let options = TriangularSolveOptions::default()
///     .with_side(Side::Left)
///     .with_uplo(Uplo::Lower)
///     .with_transpose(Transpose::NoTranspose)
///     .with_diagonal(Diagonal::NonUnit);
/// assert_eq!((options.side, options.uplo), (Side::Left, Uplo::Lower));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TriangularSolveOptions {
    /// Side of the unknown the matrix multiplies.
    pub side: Side,
    /// Triangle that holds the factor.
    pub uplo: Uplo,
    /// Whether the matrix is transposed.
    pub transpose: Transpose,
    /// Whether the diagonal is assumed to be one.
    pub diagonal: Diagonal,
}

impl TriangularSolveOptions {
    /// Return options with the requested side.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{Side, TriangularSolveOptions};
    ///
    /// assert_eq!(TriangularSolveOptions::default().with_side(Side::Left).side, Side::Left);
    /// ```
    pub fn with_side(mut self, side: Side) -> Self {
        self.side = side;
        self
    }

    /// Return options with the requested triangle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{TriangularSolveOptions, Uplo};
    ///
    /// assert_eq!(TriangularSolveOptions::default().with_uplo(Uplo::Lower).uplo, Uplo::Lower);
    /// ```
    pub fn with_uplo(mut self, uplo: Uplo) -> Self {
        self.uplo = uplo;
        self
    }

    /// Return options with the requested transposition.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{Transpose, TriangularSolveOptions};
    ///
    /// let options = TriangularSolveOptions::default().with_transpose(Transpose::Transpose);
    /// assert_eq!(options.transpose, Transpose::Transpose);
    /// ```
    pub fn with_transpose(mut self, transpose: Transpose) -> Self {
        self.transpose = transpose;
        self
    }

    /// Return options with the requested diagonal convention.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{Diagonal, TriangularSolveOptions};
    ///
    /// let options = TriangularSolveOptions::default().with_diagonal(Diagonal::Unit);
    /// assert_eq!(options.diagonal, Diagonal::Unit);
    /// ```
    pub fn with_diagonal(mut self, diagonal: Diagonal) -> Self {
        self.diagonal = diagonal;
        self
    }
}

/// Options for `norm`.
///
/// `ord: None` with `axes: None` is the Frobenius norm of the flattened input; the
/// meaning of each `ord` is unchanged from the former `norm(ord, dim, keepdim)` signature.
/// Axes may be negative and count from the last axis.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{KeepDims, NormOptions};
///
/// let options = NormOptions::default()
///     .with_ord(1.0)
///     .with_axes(vec![-1])
///     .with_keep_dims(KeepDims::Keep);
/// assert_eq!(options.ord, Some(1.0));
/// assert_eq!(options.axes.as_deref(), Some(&[-1_isize][..]));
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct NormOptions {
    /// Norm order; `None` selects the default for the reduced rank.
    pub ord: Option<f64>,
    /// Reduced axes; `None` reduces every axis.
    pub axes: Option<Vec<isize>>,
    /// Whether reduced axes are kept with length one.
    pub keep_dims: KeepDims,
}

impl NormOptions {
    /// Return options with the requested norm order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::NormOptions;
    ///
    /// assert_eq!(NormOptions::default().with_ord(f64::INFINITY).ord, Some(f64::INFINITY));
    /// ```
    pub fn with_ord(mut self, ord: f64) -> Self {
        self.ord = Some(ord);
        self
    }

    /// Return options that reduce the given axes; negative axes count from the end.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::NormOptions;
    ///
    /// let options = NormOptions::default().with_axes(vec![0, -1]);
    /// assert_eq!(options.axes, Some(vec![0, -1]));
    /// ```
    pub fn with_axes(mut self, axes: impl Into<Vec<isize>>) -> Self {
        self.axes = Some(axes.into());
        self
    }

    /// Return options with the requested reduced-axis convention.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{KeepDims, NormOptions};
    ///
    /// let options = NormOptions::default().with_keep_dims(KeepDims::Keep);
    /// assert_eq!(options.keep_dims, KeepDims::Keep);
    /// ```
    pub fn with_keep_dims(mut self, keep_dims: KeepDims) -> Self {
        self.keep_dims = keep_dims;
        self
    }
}

/// Options for `pinv`.
///
/// `rtol: None` uses the dtype-dependent default tolerance of the former `pinv`; a value
/// is the former `pinv_with_rtol(rtol)`.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::PinvOptions;
///
/// assert_eq!(PinvOptions::default().rtol, None);
/// assert_eq!(PinvOptions::default().with_rtol(1.0e-6).rtol, Some(1.0e-6));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct PinvOptions {
    /// Relative singular-value cutoff; `None` selects the default.
    pub rtol: Option<f64>,
}

impl PinvOptions {
    /// Return options with an explicit relative tolerance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::PinvOptions;
    ///
    /// assert_eq!(PinvOptions::default().with_rtol(0.5).rtol, Some(0.5));
    /// ```
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }
}

macro_rules! empty_options {
    ($($(#[$doc:meta])* $name:ident => $op:literal;)*) => {$(
        $(#[$doc])*
        ///
        /// The operation has no semantic option yet; the type exists so that every
        #[doc = concat!("operation has the same `", $op, "` / `", $op, "_with` shape (#2552).")]
        ///
        /// # Examples
        ///
        /// ```rust
        #[doc = concat!("let options = tenferro_linalg::", stringify!($name), "::default();")]
        #[doc = concat!("assert_eq!(options, tenferro_linalg::", stringify!($name), "::default());")]
        /// ```
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        #[non_exhaustive]
        pub struct $name {}
    )*};
}

empty_options! {
    /// Options for `svdvals`.
    SvdvalsOptions => "svdvals";
    /// Options for `householder_qr`.
    HouseholderQrOptions => "householder_qr";
    /// Options for `lu`.
    LuOptions => "lu";
    /// Options for `full_piv_lu`.
    FullPivLuOptions => "full_piv_lu";
    /// Options for `full_piv_lu_solve`.
    FullPivLuSolveOptions => "full_piv_lu_solve";
    /// Options for `solve`.
    SolveOptions => "solve";
    /// Options for `cholesky`.
    CholeskyOptions => "cholesky";
    /// Options for `eig`.
    EigOptions => "eig";
    /// Options for `eigvals`.
    EigvalsOptions => "eigvals";
    /// Options for `eigvalsh`.
    EigvalshOptions => "eigvalsh";
    /// Options for `slogdet`.
    SlogdetOptions => "slogdet";
    /// Options for `det`.
    DetOptions => "det";
    /// Options for `inv`.
    InvOptions => "inv";
    /// Options for `lstsq`.
    LstsqOptions => "lstsq";
}
