use thiserror::Error;

/// Operand metadata for PartialDiagonal planning.
///
/// `dims` and `axis_classes` are logical (uncompressed) axis metadata.
///
/// # Examples
///
/// ```ignore
/// use tenferro::OperandAxisClasses;
///
/// let operand = OperandAxisClasses::new(vec![3, 3], vec![0, 0]).unwrap();
/// assert_eq!(operand.dims.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandAxisClasses {
    /// Logical axis dimensions.
    pub dims: Vec<usize>,
    /// Axis class id per logical axis.
    pub axis_classes: Vec<usize>,
}

impl OperandAxisClasses {
    /// Construct operand metadata with length validation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::OperandAxisClasses;
    ///
    /// let x = OperandAxisClasses::new(vec![2, 2], vec![0, 0]).unwrap();
    /// assert_eq!(x.axis_classes, vec![0, 0]);
    /// ```
    pub fn new(dims: Vec<usize>, axis_classes: Vec<usize>) -> Result<Self, AxisClassPlanError> {
        if dims.len() != axis_classes.len() {
            return Err(AxisClassPlanError::InvalidOperand {
                operand: None,
                message: format!(
                    "dims length ({}) must match axis_classes length ({})",
                    dims.len(),
                    axis_classes.len()
                ),
            });
        }
        Ok(Self { dims, axis_classes })
    }
}

/// Per-operand metadata plan.
///
/// # Examples
///
/// ```ignore
/// use tenferro::OperandAxisClassPlan;
///
/// let plan = OperandAxisClassPlan {
///     class_roots: vec![0, 1],
///     duplicate_class_groups: vec![],
///     normalized_class_roots: vec![0, 1],
/// };
/// assert_eq!(plan.class_roots.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandAxisClassPlan {
    /// Global class root id per local class (local class order = first appearance).
    pub class_roots: Vec<usize>,
    /// Duplicate groups in local class coordinates.
    /// Each group with len >= 2 indicates payload axes that must be diagonalized.
    pub duplicate_class_groups: Vec<Vec<usize>>,
    /// Global class roots after per-operand duplicate elimination (first appearance order).
    pub normalized_class_roots: Vec<usize>,
}

/// Metadata plan for one PartialDiagonal einsum contraction.
///
/// # Examples
///
/// ```ignore
/// use tenferro::AxisClassMergePlan;
///
/// let plan = AxisClassMergePlan {
///     operand_plans: vec![],
///     operand_axis_roots: vec![],
///     output_class_roots: vec![],
///     output_axis_classes: vec![],
///     output_dims: vec![],
///     output_compressed_roots: vec![],
/// };
/// assert!(plan.output_dims.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisClassMergePlan {
    /// Per-operand normalization metadata.
    pub operand_plans: Vec<OperandAxisClassPlan>,
    /// Global class root id per logical axis for each operand.
    /// Outer index: operand, inner index: logical axis.
    pub operand_axis_roots: Vec<Vec<usize>>,
    /// Output global class roots in output logical-axis order.
    pub output_class_roots: Vec<usize>,
    /// Output axis class ids (canonicalized in output-order first appearance).
    pub output_axis_classes: Vec<usize>,
    /// Output logical-axis dimensions in output order.
    pub output_dims: Vec<usize>,
    /// Distinct output class roots in first-appearance order.
    pub output_compressed_roots: Vec<usize>,
}

/// Errors for metadata planning.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AxisClassPlanError {
    /// Number of operands does not match subscripts inputs.
    #[error("operand count mismatch: expected {expected}, found {found}")]
    InvalidOperandCount { expected: usize, found: usize },
    /// An operand has invalid metadata.
    #[error("invalid operand metadata: {message}")]
    InvalidOperand {
        /// Operand index when known.
        operand: Option<usize>,
        /// Human-readable details.
        message: String,
    },
    /// Subscripts are incompatible with an operand rank.
    #[error("invalid subscripts for operand {operand}: {message}")]
    InvalidSubscripts { operand: usize, message: String },
    /// Label dimension mismatch.
    #[error("label dimension mismatch for label {label}: expected {expected}, got {actual}")]
    LabelDimensionMismatch {
        /// Label id.
        label: u32,
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Merged class dimension mismatch.
    #[error("merged class dimension mismatch on root {root}: expected {expected}, got {actual}")]
    MergedClassDimensionMismatch {
        /// Canonical root id.
        root: usize,
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Output label does not appear in any input labels.
    #[error("output label {label} is not present in inputs")]
    MissingOutputLabel { label: u32 },
}
