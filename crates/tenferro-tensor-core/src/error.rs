use crate::{DType, ShapeVec};

/// Coarse classification for shared tensor validation failures.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{ValidationError, ValidationKind};
///
/// let error = ValidationError::RankMismatch {
///     expected: 2,
///     actual: 1,
/// };
/// assert_eq!(error.kind(), ValidationKind::RankMismatch);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValidationKind {
    ShapeMismatch,
    RankMismatch,
    AxisOutOfBounds,
    DTypeMismatch,
    InvalidArgument,
}

/// Coarse classification shared by crate-local error types.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{ErrorKind, ValidationKind};
///
/// assert_eq!(
///     ErrorKind::Validation(ValidationKind::ShapeMismatch),
///     ErrorKind::Validation(ValidationKind::ShapeMismatch),
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    Validation(ValidationKind),
    Unsupported,
    NumericalFailure,
    BackendFailure,
    Io,
    RuntimeState,
    Internal,
}

/// Structured facts describing why two tensor shapes are incompatible.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{ShapeMismatch, ShapeVec};
///
/// let mismatch = ShapeMismatch::IncompatibleShapes {
///     lhs: ShapeVec::from_vec(vec![2, 3]),
///     rhs: ShapeVec::from_vec(vec![2, 4]),
/// };
/// assert!(mismatch.to_string().contains("incompatible shapes"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ShapeMismatch {
    #[error("incompatible shapes: lhs={lhs:?}, rhs={rhs:?}")]
    IncompatibleShapes { lhs: ShapeVec, rhs: ShapeVec },
    #[error("shape mismatch: expected={expected:?}, actual={actual:?}")]
    ExpectedActual {
        expected: ShapeVec,
        actual: ShapeVec,
    },
    #[error("reshape element-count mismatch: from {from} to {to}")]
    ReshapeElementCount { from: usize, to: usize },
    #[error(
        "contracted dimensions differ: lhs axis {lhs_axis} ({lhs_size}) vs rhs axis {rhs_axis} ({rhs_size})"
    )]
    ContractedDimensions {
        lhs_axis: usize,
        lhs_size: usize,
        rhs_axis: usize,
        rhs_size: usize,
    },
}

/// Structured validation failures owned by the tensor data model.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{ShapeMismatch, ShapeVec, ValidationError};
///
/// let error: ValidationError = ShapeMismatch::ExpectedActual {
///     expected: ShapeVec::from_vec(vec![2, 3]),
///     actual: ShapeVec::from_vec(vec![6]),
/// }
/// .into();
/// assert!(error.to_string().contains("shape mismatch"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ValidationError {
    #[error("{0}")]
    ShapeMismatch(#[source] Box<ShapeMismatch>),
    #[error("shape product {expected} does not match data length {actual}")]
    ShapeDataLengthMismatch { expected: usize, actual: usize },
    #[error("rank mismatch: expected {expected}, actual {actual}")]
    RankMismatch { expected: usize, actual: usize },
    #[error("axis {axis} out of bounds for rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
    #[error("duplicate {role} axis {axis}")]
    DuplicateAxis { axis: usize, role: &'static str },
    #[error("axis {axis} appears in both {first_role} and {second_role}")]
    AxisRoleConflict {
        axis: usize,
        first_role: &'static str,
        second_role: &'static str,
    },
    #[error("invalid permutation length: expected {expected}, actual {actual}")]
    InvalidPermutationLength { expected: usize, actual: usize },
    #[error("invalid slice step {step}; zero is invalid")]
    InvalidSliceStep { step: isize },
    #[error("invalid slice bounds: start={start}, end={end}, axis_len={axis_len}")]
    InvalidSliceBounds {
        start: isize,
        end: isize,
        axis_len: usize,
    },
    #[error("dtype mismatch: expected {expected:?}, actual {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("invalid argument {argument}: {message}")]
    InvalidArgument {
        argument: &'static str,
        message: String,
    },
    #[error("view is not slice-contiguous; materialize with to_contiguous before requesting a borrowed slice")]
    NonContiguousViewAsSlice,
    #[error("view metadata is out of borrowed-slice bounds")]
    ViewOutOfBounds,
    #[error("mutable tensor layout may overlap physical elements; materialize a contiguous owner before requesting mutable access")]
    OverlappingMutableLayout,
    #[error("integer overflow while validating tensor metadata")]
    IntegerOverflow,
}

impl From<ShapeMismatch> for ValidationError {
    fn from(error: ShapeMismatch) -> Self {
        Self::ShapeMismatch(Box::new(error))
    }
}

impl ValidationError {
    /// Return the stable coarse classification for this validation failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{ValidationError, ValidationKind};
    ///
    /// let error = ValidationError::AxisOutOfBounds { axis: 3, rank: 2 };
    /// assert_eq!(error.kind(), ValidationKind::AxisOutOfBounds);
    /// ```
    pub fn kind(&self) -> ValidationKind {
        match self {
            Self::ShapeMismatch(_) | Self::ShapeDataLengthMismatch { .. } => {
                ValidationKind::ShapeMismatch
            }
            Self::RankMismatch { .. } | Self::InvalidPermutationLength { .. } => {
                ValidationKind::RankMismatch
            }
            Self::AxisOutOfBounds { .. } => ValidationKind::AxisOutOfBounds,
            Self::DTypeMismatch { .. } => ValidationKind::DTypeMismatch,
            _ => ValidationKind::InvalidArgument,
        }
    }
}
