//! Control enums for linear-algebra operations (the #2552 grammar).
//!
//! These replace control `bool` arguments. They are linalg-only; `Conj` is
//! shared and lives in `tenferro-tensor`.

/// Which side the matrix operand acts from.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Side;
///
/// assert_eq!(Side::default(), Side::Left);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Side {
    /// The matrix acts from the left.
    #[default]
    Left,
    /// The matrix acts from the right.
    Right,
}

/// Which triangle of a matrix is referenced.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Uplo;
///
/// assert_eq!(Uplo::default(), Uplo::Lower);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Uplo {
    /// The lower triangle.
    #[default]
    Lower,
    /// The upper triangle.
    Upper,
}

/// Transformation applied to a matrix operand.
///
/// `Adjoint` is the conjugate transpose; an operation that cannot honour it for
/// complex input returns `Error::Unsupported` instead of treating it as
/// `Transpose`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::MatOp;
///
/// assert_eq!(MatOp::default(), MatOp::None);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MatOp {
    /// Use the matrix as it is.
    #[default]
    None,
    /// Use the transpose.
    Transpose,
    /// Use the conjugate transpose.
    Adjoint,
}

/// Whether a triangular matrix has an implicit unit diagonal.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::Diag;
///
/// assert_eq!(Diag::default(), Diag::NonUnit);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Diag {
    /// The diagonal is stored and used.
    #[default]
    NonUnit,
    /// The diagonal is implicitly one.
    Unit,
}

/// Whether a reduction keeps reduced axes as size-one axes.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::KeepDims;
///
/// assert_eq!(KeepDims::default(), KeepDims::No);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum KeepDims {
    /// Drop the reduced axes.
    #[default]
    No,
    /// Keep the reduced axes with size one.
    Yes,
}
