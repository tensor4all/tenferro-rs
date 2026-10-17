use tinyvec::TinyVec;

use crate::{Error, Result, ValidationError};

const DOT_GENERAL_OP: &str = "dot_general";

fn invalid_dot_general_config(message: impl Into<String>) -> Error {
    Error::invalid_argument(DOT_GENERAL_OP, "dot_general_config", message)
}

/// DotGeneral dimension configuration.
///
/// Each axis list stores up to four axes inline and spills to the heap for
/// larger contractions; this is not a tensor-rank limit.
/// Records only the dim-numbering roles (contracting / batch; free is derived).
/// Rank info travels with the enclosing `StdTensorOp::DotGeneral` variant at
/// the trace/StdTensorOp layer, and with `ExecInstruction::output_shapes` at
/// the exec layer. This separation makes it structurally impossible for
/// stored ranks to drift from actual tensor ranks (issue #664).
///
/// # Output layout
///
/// The output shape is `[lhs_free..., rhs_free..., batch...]` (col-major
/// batch-trailing convention): batch axes come **last**, unlike PyTorch's
/// batch-leading `bmm`. Batch dims have the largest stride so that each batch
/// slice occupies a contiguous block of memory. Free axes keep their input
/// order. For attention scores, `q[d, lq, b]` against `k[d, lk, b]` with
/// `d` contracted and `b` batched gives `[lq, lk, b]`; see the
/// "Contraction Output Layout" section of the tensor-operations guide for a
/// worked example. Transpose afterwards only where a consumer needs batch
/// first.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::DotGeneralConfig;
///
/// let config = DotGeneralConfig {
///     lhs_contracting_dims: [1].as_slice().iter().copied().collect(),
///     rhs_contracting_dims: [0].as_slice().iter().copied().collect(),
///     lhs_batch_dims: [].as_slice().iter().copied().collect(),
///     rhs_batch_dims: [].as_slice().iter().copied().collect(),
/// };
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig {
    pub lhs_contracting_dims: TinyVec<[usize; 4]>,
    pub rhs_contracting_dims: TinyVec<[usize; 4]>,
    pub lhs_batch_dims: TinyVec<[usize; 4]>,
    pub rhs_batch_dims: TinyVec<[usize; 4]>,
}

/// The degenerate algebra a `DotGeneral` actually is when it is not a
/// contraction, classified from the config and the operand shapes alone.
///
/// The semantic rewrite and the eager executor share this rule so a degenerate
/// form is rewritten once, in one place, instead of each backend implementing
/// it implicitly (invariant I4 of the #2112 umbrella).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegenerateDotKind {
    /// Every axis of both operands is a batch axis, in each operand's own
    /// order, and both operands carry the same dimensions: an elementwise
    /// product.
    Hadamard,
    /// No contracting and no batch dims, free dims on both sides: an outer
    /// product (the concatenated product).
    OuterProduct,
    /// No contracting and no batch dims, free dims on one side only: that
    /// operand, permuted into the output's order and scaled by the other.
    ScalarTranspose,
    /// Every dimension contracts: a product reduced to rank 0.
    FullContraction,
}

/// Classify a `DotGeneral` whose algebra is not a contraction.
///
/// Returns `None` when the form is a real contraction (or a shape this rule
/// does not model), so callers fall through to their general path.
#[must_use]
pub fn degenerate_dot_kind<T: PartialEq>(
    config: &DotGeneralConfig,
    lhs_shape: &[T],
    rhs_shape: &[T],
) -> Option<DegenerateDotKind> {
    if !config.lhs_contracting_dims.is_empty() || !config.rhs_contracting_dims.is_empty() {
        let all_contract = |rank: usize, dims: &[usize]| {
            dims.len() == rank
                && dims
                    .iter()
                    .enumerate()
                    .all(|(position, &axis)| axis == position)
        };
        return (config.lhs_batch_dims.is_empty()
            && config.rhs_batch_dims.is_empty()
            && all_contract(lhs_shape.len(), &config.lhs_contracting_dims)
            && all_contract(rhs_shape.len(), &config.rhs_contracting_dims))
        .then_some(DegenerateDotKind::FullContraction);
    }
    // Nothing contracts. Every axis of both operands being a batch axis means
    // nothing is free either and the product is elementwise - but only when both
    // batch lists put the axes in the operand's own order. Batch axis `k` pairs
    // `lhs_batch_dims[k]` with `rhs_batch_dims[k]`, so a permuted list pairs
    // different axes (`out[i, j] = a[i, j] * b[j, i]`) and a plain multiply would
    // be wrong. The general path handles that form; measured with `[2, 2]`
    // operands, batch `[0, 1]` against `[1, 0]`, which came back transposed.
    let identity_ordered = |dims: &[usize], rank: usize| {
        dims.len() == rank
            && dims
                .iter()
                .enumerate()
                .all(|(position, &axis)| axis == position)
    };
    if identity_ordered(&config.lhs_batch_dims, lhs_shape.len())
        && identity_ordered(&config.rhs_batch_dims, rhs_shape.len())
        && lhs_shape == rhs_shape
    {
        return Some(DegenerateDotKind::Hadamard);
    }
    if !config.lhs_batch_dims.is_empty() || !config.rhs_batch_dims.is_empty() {
        return None;
    }
    // No batch and no contracting dims means **every** operand axis is free, so
    // `dot_general_output_shape` makes the output `lhs ++ rhs` - an outer
    // product, whatever the two shapes are. Equal shapes do not make it a
    // Hadamard: that form needs every axis to be a *batch* axis, which the arm
    // above already covers. Reading equal shapes as a Hadamard here made both
    // consumers of this classifier produce `a.shape` where the graph declared
    // `lhs ++ rhs` (#2165).
    match (lhs_shape.is_empty(), rhs_shape.is_empty()) {
        (false, false) => Some(DegenerateDotKind::OuterProduct),
        (true, false) | (false, true) => Some(DegenerateDotKind::ScalarTranspose),
        (true, true) => None,
    }
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
    ///     lhs_contracting_dims: [1].as_slice().iter().copied().collect(),
    ///     rhs_contracting_dims: [0].as_slice().iter().copied().collect(),
    ///     lhs_batch_dims: [].as_slice().iter().copied().collect(),
    ///     rhs_batch_dims: [].as_slice().iter().copied().collect(),
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
