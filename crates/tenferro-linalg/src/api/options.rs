//! Options structs of the linear-algebra operations (the #2552 grammar).
//!
//! Every operation has one `XxxOptions`. All are `#[non_exhaustive]`: build
//! them with `Default::default()` and the setters, whose names equal the field
//! names. `SvdOptions`, `QrOptions`, `EighOptions` and
//! `RankRevealingQrOptions` keep their definitions in the modules that own
//! their gauge enums and are re-exported from the crate root next to these.

use tenferro_tensor::AxisVec;

use crate::extension::{EighDriver, SvdDriver};
use crate::flags::{Diag, KeepDims, MatOp, Side, Uplo};

/// Options for compact Householder QR. It has no settings yet; the type exists so every operation has the same `op` / `op_with` shape.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::HouseholderQrOptions;
///
/// let options = HouseholderQrOptions::default();
/// assert_eq!(options.clone(), HouseholderQrOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct HouseholderQrOptions {}

/// Options for partial-pivot LU. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::LuOptions;
///
/// let options = LuOptions::default();
/// assert_eq!(options.clone(), LuOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LuOptions {}

/// Options for complete-pivot LU. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::FullPivLuOptions;
///
/// let options = FullPivLuOptions::default();
/// assert_eq!(options.clone(), FullPivLuOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FullPivLuOptions {}

/// Options for a complete-pivot LU solve. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::FullPivLuSolveOptions;
///
/// let options = FullPivLuSolveOptions::default();
/// assert_eq!(options.clone(), FullPivLuSolveOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FullPivLuSolveOptions {}

/// Options for a linear solve `a x = b`. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::SolveOptions;
///
/// let options = SolveOptions::default();
/// assert_eq!(options.clone(), SolveOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SolveOptions {}

/// Options for a least-squares solve. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::LstsqOptions;
///
/// let options = LstsqOptions::default();
/// assert_eq!(options.clone(), LstsqOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LstsqOptions {}

/// Options for the general eigendecomposition. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::EigOptions;
///
/// let options = EigOptions::default();
/// assert_eq!(options.clone(), EigOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct EigOptions {}

/// Options for general eigenvalues. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::EigvalsOptions;
///
/// let options = EigvalsOptions::default();
/// assert_eq!(options.clone(), EigvalsOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct EigvalsOptions {}

/// Options for the sign and log-absolute determinant. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::SlogdetOptions;
///
/// let options = SlogdetOptions::default();
/// assert_eq!(options.clone(), SlogdetOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SlogdetOptions {}

/// Options for the determinant. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::DetOptions;
///
/// let options = DetOptions::default();
/// assert_eq!(options.clone(), DetOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DetOptions {}

/// Options for the matrix inverse. It has no settings yet.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::InvOptions;
///
/// let options = InvOptions::default();
/// assert_eq!(options.clone(), InvOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvOptions {}

/// Options for singular values only.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{SvdDriver, SvdvalsOptions};
///
/// let options = SvdvalsOptions::default().driver(SvdDriver::QrBased);
/// assert_eq!(options.driver, SvdDriver::QrBased);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct SvdvalsOptions {
    /// Algorithm family: a hint for providers that can choose a driver; the
    /// current CPU provider ignores it.
    pub driver: SvdDriver,
}

impl SvdvalsOptions {
    /// Return options with an explicit SVD driver.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{SvdDriver, SvdvalsOptions};
    ///
    /// let options = SvdvalsOptions::default().driver(SvdDriver::Jacobi);
    /// assert_eq!(options.driver, SvdDriver::Jacobi);
    /// ```
    pub fn driver(mut self, driver: SvdDriver) -> Self {
        self.driver = driver;
        self
    }
}

/// Options for Hermitian eigenvalues only.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{EighDriver, EigvalshOptions};
///
/// let options = EigvalshOptions::default().driver(EighDriver::Jacobi);
/// assert_eq!(options.driver, EighDriver::Jacobi);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct EigvalshOptions {
    /// Algorithm family: a hint for providers that can choose a driver; the
    /// current CPU provider ignores it.
    pub driver: EighDriver,
}

impl EigvalshOptions {
    /// Return options with an explicit eigensolver driver.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{EighDriver, EigvalshOptions};
    ///
    /// let options = EigvalshOptions::default().driver(EighDriver::DivideAndConquer);
    /// assert_eq!(options.driver, EighDriver::DivideAndConquer);
    /// ```
    pub fn driver(mut self, driver: EighDriver) -> Self {
        self.driver = driver;
        self
    }
}

/// Options for a triangular solve.
///
/// The defaults (`Left`, `Lower`, `None`, `NonUnit`) solve `a x = b` with the
/// lower triangle of `a`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{Diag, MatOp, Side, TriangularSolveOptions, Uplo};
///
/// let options = TriangularSolveOptions::default()
///     .side(Side::Right)
///     .uplo(Uplo::Upper)
///     .op(MatOp::Transpose)
///     .diag(Diag::Unit);
/// assert_eq!(options.side, Side::Right);
/// assert_eq!(options.uplo, Uplo::Upper);
/// assert_eq!(options.op, MatOp::Transpose);
/// assert_eq!(options.diag, Diag::Unit);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct TriangularSolveOptions {
    /// Which side the matrix acts from.
    pub side: Side,
    /// Which triangle of the matrix is referenced.
    pub uplo: Uplo,
    /// Transformation applied to the matrix. `MatOp::Adjoint` on a complex
    /// matrix returns `Unsupported`.
    pub op: MatOp,
    /// Whether the diagonal is implicitly one.
    pub diag: Diag,
}

impl TriangularSolveOptions {
    /// Return options with the requested side.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{Side, TriangularSolveOptions};
    ///
    /// assert_eq!(TriangularSolveOptions::default().side(Side::Right).side, Side::Right);
    /// ```
    pub fn side(mut self, side: Side) -> Self {
        self.side = side;
        self
    }

    /// Return options with the requested triangle.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{TriangularSolveOptions, Uplo};
    ///
    /// assert_eq!(TriangularSolveOptions::default().uplo(Uplo::Upper).uplo, Uplo::Upper);
    /// ```
    pub fn uplo(mut self, uplo: Uplo) -> Self {
        self.uplo = uplo;
        self
    }

    /// Return options with the requested matrix transformation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{MatOp, TriangularSolveOptions};
    ///
    /// assert_eq!(TriangularSolveOptions::default().op(MatOp::Transpose).op, MatOp::Transpose);
    /// ```
    pub fn op(mut self, op: MatOp) -> Self {
        self.op = op;
        self
    }

    /// Return options with the requested diagonal convention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{Diag, TriangularSolveOptions};
    ///
    /// assert_eq!(TriangularSolveOptions::default().diag(Diag::Unit).diag, Diag::Unit);
    /// ```
    pub fn diag(mut self, diag: Diag) -> Self {
        self.diag = diag;
        self
    }
}

/// Options for a Cholesky factorization.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{CholeskyOptions, Uplo};
///
/// let options = CholeskyOptions::default();
/// assert_eq!(options.uplo, Uplo::Lower);
/// assert_eq!(options.uplo(Uplo::Upper).uplo, Uplo::Upper);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct CholeskyOptions {
    /// Which triangular factor is returned. The implementation computes the
    /// lower factor only, so `Uplo::Upper` returns `Unsupported`.
    pub uplo: Uplo,
}

impl CholeskyOptions {
    /// Return options with the requested triangle.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{CholeskyOptions, Uplo};
    ///
    /// assert_eq!(CholeskyOptions::default().uplo(Uplo::Upper).uplo, Uplo::Upper);
    /// ```
    pub fn uplo(mut self, uplo: Uplo) -> Self {
        self.uplo = uplo;
        self
    }
}

/// Options for the Moore-Penrose pseudoinverse.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::PinvOptions;
///
/// assert_eq!(PinvOptions::default().rtol, None);
/// assert_eq!(PinvOptions::default().rtol(Some(1.0e-10)).rtol, Some(1.0e-10));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct PinvOptions {
    /// Relative singular-value cutoff. `None` uses `eps * max(m, n)` of the
    /// input dtype; `Some(rtol)` uses `rtol` (negative values act as zero).
    pub rtol: Option<f64>,
}

impl PinvOptions {
    /// Return options with an explicit relative cutoff.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::PinvOptions;
    ///
    /// assert_eq!(PinvOptions::default().rtol(Some(1.0e-8)).rtol, Some(1.0e-8));
    /// ```
    pub fn rtol(mut self, rtol: Option<f64>) -> Self {
        self.rtol = rtol;
        self
    }
}

/// The order of a norm.
///
/// Each variant maps one to one onto the former `Option<f64>` argument:
/// `Frobenius` is `None`, `Zero` is `Some(0.0)`, `Inf` and `NegInf` are
/// `Some(+inf)` and `Some(-inf)`, and `P(p)` is `Some(p)` for a finite nonzero
/// `p`. The meaning of `P(p)` depends on how many axes are reduced:
///
/// - one axis (a vector norm): the entrywise `p`-norm;
/// - two axes (a matrix norm): `P(1.0)` / `P(-1.0)` are the largest / smallest
///   absolute column sum, `P(2.0)` / `P(-2.0)` the largest / smallest singular
///   value, and any other `p` the entrywise `p`-norm of the matrix;
/// - more axes: the entrywise `p`-norm over all reduced axes.
///
/// `Zero` counts the nonzero entries; `Inf` / `NegInf` are the largest /
/// smallest absolute entry (vector) or absolute row sum (matrix).
/// `Frobenius` is the Euclidean norm of the reduced entries. `P(p)` with
/// `p == 0`, an infinite `p` or a NaN `p` is rejected as an invalid argument;
/// use `Zero`, `Inf` or `NegInf` instead.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::NormOrd;
///
/// assert_eq!(NormOrd::default(), NormOrd::Frobenius);
/// let spectral = NormOrd::P(2.0);
/// assert_eq!(spectral, NormOrd::P(2.0));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum NormOrd {
    /// The Euclidean (Frobenius) norm of the reduced entries.
    #[default]
    Frobenius,
    /// The number of nonzero entries.
    Zero,
    /// The largest absolute entry, or the largest absolute row sum.
    Inf,
    /// The smallest absolute entry, or the smallest absolute row sum.
    NegInf,
    /// The `p`-norm for a finite nonzero `p`.
    P(f64),
}

impl NormOrd {
    /// The former `Option<f64>` order, validating `P(p)`.
    pub(crate) fn to_ord(self) -> tenferro_tensor::Result<Option<f64>> {
        match self {
            Self::Frobenius => Ok(None),
            Self::Zero => Ok(Some(0.0)),
            Self::Inf => Ok(Some(f64::INFINITY)),
            Self::NegInf => Ok(Some(f64::NEG_INFINITY)),
            Self::P(p) if p.is_finite() && p != 0.0 => Ok(Some(p)),
            Self::P(p) => Err(tenferro_tensor::Error::invalid_argument(
                "norm",
                "ord",
                format!(
                    "P(p) needs a finite nonzero p, got {p}; use Zero, Inf or NegInf for those orders"
                ),
            )),
        }
    }
}

/// Options for a norm.
///
/// `axes` are signed (`-1` is the last axis) and normalized against the input
/// rank; `None` reduces every axis.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{KeepDims, NormOptions, NormOrd};
/// use tenferro_tensor::AxisVec;
///
/// let axes: AxisVec = [0, -1].into_iter().collect();
/// let options = NormOptions::default()
///     .ord(NormOrd::Inf)
///     .axes(Some(axes.clone()))
///     .keep_dims(KeepDims::Yes);
/// assert_eq!(options.ord, NormOrd::Inf);
/// assert_eq!(options.axes, Some(axes));
/// assert_eq!(options.keep_dims, KeepDims::Yes);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct NormOptions {
    /// The order of the norm.
    pub ord: NormOrd,
    /// Axes to reduce (signed); `None` reduces all axes.
    pub axes: Option<AxisVec>,
    /// Whether reduced axes stay as size-one axes.
    pub keep_dims: KeepDims,
}

impl NormOptions {
    /// Return options with the requested order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{NormOptions, NormOrd};
    ///
    /// assert_eq!(NormOptions::default().ord(NormOrd::Zero).ord, NormOrd::Zero);
    /// ```
    pub fn ord(mut self, ord: NormOrd) -> Self {
        self.ord = ord;
        self
    }

    /// Return options reducing the given signed axes (`None` reduces all).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::NormOptions;
    /// use tenferro_tensor::AxisVec;
    ///
    /// let axes: AxisVec = [1].into_iter().collect();
    /// assert_eq!(NormOptions::default().axes(Some(axes.clone())).axes, Some(axes));
    /// ```
    pub fn axes(mut self, axes: Option<AxisVec>) -> Self {
        self.axes = axes;
        self
    }

    /// Return options with the requested size-one-axes convention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::{KeepDims, NormOptions};
    ///
    /// assert_eq!(NormOptions::default().keep_dims(KeepDims::Yes).keep_dims, KeepDims::Yes);
    /// ```
    pub fn keep_dims(mut self, keep_dims: KeepDims) -> Self {
        self.keep_dims = keep_dims;
        self
    }

    /// The reduced axes as the former unsigned `dim` list for an input of
    /// `rank` axes; `None` when every axis is reduced.
    pub(crate) fn unsigned_axes(&self, rank: usize) -> tenferro_tensor::Result<Option<Vec<usize>>> {
        match &self.axes {
            None => Ok(None),
            Some(axes) => tenferro_tensor::normalize_axes(axes, rank)
                .map(|n| Some(n.to_vec()))
                .map_err(|e| tenferro_tensor::Error::validation("norm", e)),
        }
    }
}
