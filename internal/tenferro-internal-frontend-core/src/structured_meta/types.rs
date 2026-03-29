use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandAxisClasses {
    pub dims: Vec<usize>,
    pub axis_classes: Vec<usize>,
}

impl OperandAxisClasses {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandAxisClassPlan {
    pub class_roots: Vec<usize>,
    pub duplicate_class_groups: Vec<Vec<usize>>,
    pub normalized_class_roots: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisClassMergePlan {
    pub operand_plans: Vec<OperandAxisClassPlan>,
    pub operand_axis_roots: Vec<Vec<usize>>,
    pub output_class_roots: Vec<usize>,
    pub output_axis_classes: Vec<usize>,
    pub output_dims: Vec<usize>,
    pub output_compressed_roots: Vec<usize>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AxisClassPlanError {
    #[error("operand count mismatch: expected {expected}, found {found}")]
    InvalidOperandCount { expected: usize, found: usize },
    #[error("invalid operand metadata: {message}")]
    InvalidOperand {
        operand: Option<usize>,
        message: String,
    },
    #[error("invalid subscripts for operand {operand}: {message}")]
    InvalidSubscripts { operand: usize, message: String },
    #[error("label dimension mismatch for label {label}: expected {expected}, got {actual}")]
    LabelDimensionMismatch {
        label: u32,
        expected: usize,
        actual: usize,
    },
    #[error("merged class dimension mismatch on root {root}: expected {expected}, got {actual}")]
    MergedClassDimensionMismatch {
        root: usize,
        expected: usize,
        actual: usize,
    },
    #[error("output label {label} is not present in inputs")]
    MissingOutputLabel { label: u32 },
}
