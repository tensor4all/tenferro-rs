use crate::{Error, Result, ValidationError};

const DOT_GENERAL_OP: &str = "dot_general";

fn invalid_dot_general_config(message: impl Into<String>) -> Error {
    Error::invalid_argument(DOT_GENERAL_OP, "dot_general_config", message)
}

/// DotGeneral dimension configuration.
///
/// Records only the dim-numbering roles (contracting / batch; free is derived).
/// Rank info travels with the enclosing `StdTensorOp::DotGeneral` variant at
/// the trace/StdTensorOp layer, and with `ExecInstruction::output_shapes` at
/// the exec layer. This separation makes it structurally impossible for
/// stored ranks to drift from actual tensor ranks (issue #664).
///
/// The output shape is `[lhs_free..., rhs_free..., batch...]` (col-major
/// batch-trailing convention). Batch dims have the largest stride so that
/// each batch slice occupies a contiguous block of memory.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::DotGeneralConfig;
///
/// let config = DotGeneralConfig {
///     lhs_contracting_dims: vec![1],
///     rhs_contracting_dims: vec![0],
///     lhs_batch_dims: vec![],
///     rhs_batch_dims: vec![],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig {
    pub lhs_contracting_dims: Vec<usize>,
    pub rhs_contracting_dims: Vec<usize>,
    pub lhs_batch_dims: Vec<usize>,
    pub rhs_batch_dims: Vec<usize>,
}

impl DotGeneralConfig {
    fn check_no_duplicates(dims: &[usize], label: &'static str) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for &d in dims {
            if !seen.insert(d) {
                return Err(Error::validation(
                    DOT_GENERAL_OP,
                    ValidationError::DuplicateAxis {
                        axis: d,
                        role: label,
                    },
                ));
            }
        }
        Ok(())
    }

    /// Validate that all dimension indices are within range for the given
    /// explicit ranks and that no axis appears in multiple roles.
    ///
    /// Call sites supply the actual operand ranks (from the tensor shapes they
    /// have in hand). The config itself carries only the dim-numbering roles.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DotGeneralConfig;
    ///
    /// let config = DotGeneralConfig {
    ///     lhs_contracting_dims: vec![1],
    ///     rhs_contracting_dims: vec![0],
    ///     lhs_batch_dims: vec![],
    ///     rhs_batch_dims: vec![],
    /// };
    /// config.validate_dims_with_ranks(2, 2).unwrap();
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with an axis, duplicate-axis,
    /// or configuration source when the dimension roles are invalid.
    pub fn validate_dims_with_ranks(&self, lhs_rank: usize, rhs_rank: usize) -> Result<()> {
        for &d in &self.lhs_contracting_dims {
            if d >= lhs_rank {
                return Err(Error::validation(
                    DOT_GENERAL_OP,
                    ValidationError::AxisOutOfBounds {
                        axis: d,
                        rank: lhs_rank,
                    },
                ));
            }
        }
        for &d in &self.rhs_contracting_dims {
            if d >= rhs_rank {
                return Err(Error::validation(
                    DOT_GENERAL_OP,
                    ValidationError::AxisOutOfBounds {
                        axis: d,
                        rank: rhs_rank,
                    },
                ));
            }
        }
        for &d in &self.lhs_batch_dims {
            if d >= lhs_rank {
                return Err(Error::validation(
                    DOT_GENERAL_OP,
                    ValidationError::AxisOutOfBounds {
                        axis: d,
                        rank: lhs_rank,
                    },
                ));
            }
        }
        for &d in &self.rhs_batch_dims {
            if d >= rhs_rank {
                return Err(Error::validation(
                    DOT_GENERAL_OP,
                    ValidationError::AxisOutOfBounds {
                        axis: d,
                        rank: rhs_rank,
                    },
                ));
            }
        }
        Self::check_no_duplicates(&self.lhs_contracting_dims, "lhs_contracting_dims")?;
        Self::check_no_duplicates(&self.rhs_contracting_dims, "rhs_contracting_dims")?;
        Self::check_no_duplicates(&self.lhs_batch_dims, "lhs_batch_dims")?;
        Self::check_no_duplicates(&self.rhs_batch_dims, "rhs_batch_dims")?;
        for &d in &self.lhs_contracting_dims {
            if self.lhs_batch_dims.contains(&d) {
                return Err(Error::validation(
                    DOT_GENERAL_OP,
                    ValidationError::AxisRoleConflict {
                        axis: d,
                        first_role: "lhs contracting",
                        second_role: "lhs batch",
                    },
                ));
            }
        }
        for &d in &self.rhs_contracting_dims {
            if self.rhs_batch_dims.contains(&d) {
                return Err(Error::validation(
                    DOT_GENERAL_OP,
                    ValidationError::AxisRoleConflict {
                        axis: d,
                        first_role: "rhs contracting",
                        second_role: "rhs batch",
                    },
                ));
            }
        }
        if self.lhs_contracting_dims.len() != self.rhs_contracting_dims.len() {
            return Err(invalid_dot_general_config(format!(
                "lhs/rhs contracting dim counts differ ({} vs {})",
                self.lhs_contracting_dims.len(),
                self.rhs_contracting_dims.len()
            )));
        }
        if self.lhs_batch_dims.len() != self.rhs_batch_dims.len() {
            return Err(invalid_dot_general_config(format!(
                "lhs/rhs batch dim counts differ ({} vs {})",
                self.lhs_batch_dims.len(),
                self.rhs_batch_dims.len()
            )));
        }
        Ok(())
    }
}

/// Comparison direction.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::CompareDir;
///
/// let dir = CompareDir::Eq;
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CompareDir {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// StableHLO gather dimension configuration.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::GatherConfig;
///
/// let config = GatherConfig {
///     offset_dims: vec![],
///     collapsed_slice_dims: vec![0],
///     start_index_map: vec![0],
///     index_vector_dim: 1,
///     slice_sizes: vec![1],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct GatherConfig {
    pub offset_dims: Vec<usize>,
    pub collapsed_slice_dims: Vec<usize>,
    pub start_index_map: Vec<usize>,
    pub index_vector_dim: usize,
    pub slice_sizes: Vec<usize>,
}

/// StableHLO scatter dimension configuration.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::ScatterConfig;
///
/// let config = ScatterConfig {
///     update_window_dims: vec![],
///     inserted_window_dims: vec![0],
///     scatter_dims_to_operand_dims: vec![0],
///     index_vector_dim: 1,
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ScatterConfig {
    pub update_window_dims: Vec<usize>,
    pub inserted_window_dims: Vec<usize>,
    pub scatter_dims_to_operand_dims: Vec<usize>,
    pub index_vector_dim: usize,
}

/// Slice configuration.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::SliceConfig;
///
/// let config = SliceConfig {
///     starts: vec![0],
///     limits: vec![2],
///     strides: vec![1],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SliceConfig {
    pub starts: Vec<usize>,
    pub limits: Vec<usize>,
    pub strides: Vec<usize>,
}

/// StableHLO pad configuration.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::PadConfig;
///
/// let config = PadConfig {
///     edge_padding_low: vec![1, 1],
///     edge_padding_high: vec![1, 1],
///     interior_padding: vec![0, 0],
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PadConfig {
    pub edge_padding_low: Vec<i64>,
    pub edge_padding_high: Vec<i64>,
    pub interior_padding: Vec<i64>,
}
