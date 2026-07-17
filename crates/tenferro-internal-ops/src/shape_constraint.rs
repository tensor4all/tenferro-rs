use tenferro_tensor::{DType, ErrorKind, ValidationKind};

use crate::SymDim;

/// A relation between two symbolic shape expressions.
///
/// # Examples
///
/// ```rust
/// use tenferro_ops::ShapeRelation;
///
/// assert_eq!(ShapeRelation::Equal, ShapeRelation::Equal);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapeRelation {
    /// The two symbolic expressions must evaluate to the same dimension.
    Equal,
}

/// A shape relation recorded by an extension during metadata inference.
///
/// The constraint stores expressions as provided. It does not attempt to
/// normalize or solve them.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionShapeConstraint {
    relation: ShapeRelation,
    lhs: SymDim,
    rhs: SymDim,
}

impl ExtensionShapeConstraint {
    fn equal(lhs: SymDim, rhs: SymDim) -> Self {
        Self {
            relation: ShapeRelation::Equal,
            lhs,
            rhs,
        }
    }

    /// Return the relation imposed by this constraint.
    #[doc(hidden)]
    pub fn relation(&self) -> ShapeRelation {
        self.relation
    }

    /// Return the left-hand symbolic expression.
    #[doc(hidden)]
    pub fn lhs(&self) -> &SymDim {
        &self.lhs
    }

    /// Return the right-hand symbolic expression.
    #[doc(hidden)]
    pub fn rhs(&self) -> &SymDim {
        &self.rhs
    }
}

/// Errors produced while extension shape requirements inspect input metadata.
///
/// # Examples
///
/// ```rust
/// use tenferro_ops::ExtensionShapeError;
///
/// let error = ExtensionShapeError::InputOutOfBounds {
///     family_id: "example.v1",
///     input: 2,
///     input_count: 1,
/// };
/// assert!(error.to_string().contains("example.v1"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExtensionShapeError {
    /// An input metadata index was outside the supplied input list.
    #[error(
        "extension family {family_id:?} input index {input} out of bounds for {input_count} inputs"
    )]
    InputOutOfBounds {
        /// Stable extension family identifier.
        family_id: &'static str,
        /// Requested input index.
        input: usize,
        /// Number of available inputs.
        input_count: usize,
    },
    /// An axis was outside the selected input's rank.
    #[error(
        "extension family {family_id:?} axis {axis} out of bounds for input {input} rank {rank}"
    )]
    AxisOutOfBounds {
        /// Stable extension family identifier.
        family_id: &'static str,
        /// Selected input index.
        input: usize,
        /// Requested axis index.
        axis: usize,
        /// Rank of the selected input.
        rank: usize,
    },
    /// Two inputs required to have the same shape had different ranks.
    #[error(
        "extension family {family_id:?} requires inputs {lhs_input} and {rhs_input} to have the same shape, but their ranks are {lhs_rank} and {rhs_rank}"
    )]
    RankMismatch {
        /// Stable extension family identifier.
        family_id: &'static str,
        /// Left-hand input index.
        lhs_input: usize,
        /// Left-hand input rank.
        lhs_rank: usize,
        /// Right-hand input index.
        rhs_input: usize,
        /// Right-hand input rank.
        rhs_rank: usize,
    },
}

impl From<ExtensionShapeError> for tenferro_tensor::Error {
    fn from(error: ExtensionShapeError) -> Self {
        let (family_id, kind) = match &error {
            ExtensionShapeError::InputOutOfBounds { family_id, .. } => (
                *family_id,
                ErrorKind::Validation(ValidationKind::InvalidArgument),
            ),
            ExtensionShapeError::AxisOutOfBounds { family_id, .. } => (
                *family_id,
                ErrorKind::Validation(ValidationKind::AxisOutOfBounds),
            ),
            ExtensionShapeError::RankMismatch { family_id, .. } => (
                *family_id,
                ErrorKind::Validation(ValidationKind::RankMismatch),
            ),
        };
        tenferro_tensor::Error::extension("extension", family_id, kind, error)
    }
}

/// Input metadata and equality requirements for one extension inference call.
///
/// The context records requirements declaratively. It does not prove, reject,
/// normalize, or solve them.
///
/// # Examples
///
/// ```rust
/// use tenferro_ops::ExtensionShapeContext;
///
/// fn infer(ctx: &mut ExtensionShapeContext<'_>) -> tenferro_tensor::Result<()> {
///     let lhs = ctx.input_axis(0, 0)?;
///     let rhs = ctx.input_axis(1, 0)?;
///     ctx.require_equal(lhs, 2 * rhs)?;
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct ExtensionShapeContext<'a> {
    family_id: &'static str,
    input_dtypes: &'a [DType],
    input_shapes: &'a [&'a [SymDim]],
    constraints: Vec<ExtensionShapeConstraint>,
}

impl<'a> ExtensionShapeContext<'a> {
    /// Construct a context for the internal extension inference driver.
    #[doc(hidden)]
    pub fn new_for_inference(
        family_id: &'static str,
        input_dtypes: &'a [DType],
        input_shapes: &'a [&'a [SymDim]],
    ) -> Self {
        Self {
            family_id,
            input_dtypes,
            input_shapes,
            constraints: Vec::new(),
        }
    }

    /// Return the dtype of one extension input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::ExtensionShapeContext;
    /// use tenferro_tensor::DType;
    ///
    /// fn infer(ctx: &ExtensionShapeContext<'_>) -> tenferro_tensor::Result<DType> {
    ///     Ok(ctx.input_dtype(0)?)
    /// }
    /// ```
    pub fn input_dtype(&self, input: usize) -> Result<DType, ExtensionShapeError> {
        self.input_dtypes
            .get(input)
            .copied()
            .ok_or(ExtensionShapeError::InputOutOfBounds {
                family_id: self.family_id,
                input,
                input_count: self.input_dtypes.len(),
            })
    }

    /// Return the symbolic shape of one extension input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::ExtensionShapeContext;
    ///
    /// fn infer(ctx: &ExtensionShapeContext<'_>) -> tenferro_tensor::Result<()> {
    ///     let _shape = ctx.input_shape(0)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn input_shape(&self, input: usize) -> Result<&[SymDim], ExtensionShapeError> {
        self.input_shapes
            .get(input)
            .copied()
            .ok_or(ExtensionShapeError::InputOutOfBounds {
                family_id: self.family_id,
                input,
                input_count: self.input_shapes.len(),
            })
    }

    /// Return one symbolic axis expression from an extension input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::ExtensionShapeContext;
    /// use tenferro_ops::SymDim;
    ///
    /// fn infer(ctx: &ExtensionShapeContext<'_>) -> tenferro_tensor::Result<SymDim> {
    ///     Ok(ctx.input_axis(0, 0)?)
    /// }
    /// ```
    pub fn input_axis(&self, input: usize, axis: usize) -> Result<SymDim, ExtensionShapeError> {
        let shape = self.input_shape(input)?;
        shape
            .get(axis)
            .cloned()
            .ok_or(ExtensionShapeError::AxisOutOfBounds {
                family_id: self.family_id,
                input,
                axis,
                rank: shape.len(),
            })
    }

    /// Record equality of two symbolic dimension expressions.
    ///
    /// This method records the expressions without trying to solve them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::ExtensionShapeContext;
    ///
    /// fn infer(ctx: &mut ExtensionShapeContext<'_>) -> tenferro_tensor::Result<()> {
    ///     let lhs = ctx.input_axis(0, 0)?;
    ///     let rhs = ctx.input_axis(1, 0)?;
    ///     ctx.require_equal(lhs, 2 * rhs)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn require_equal(&mut self, lhs: SymDim, rhs: SymDim) -> Result<(), ExtensionShapeError> {
        self.constraints
            .push(ExtensionShapeConstraint::equal(lhs, rhs));
        Ok(())
    }

    /// Record equality of two input axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::ExtensionShapeContext;
    ///
    /// fn infer(ctx: &mut ExtensionShapeContext<'_>) -> tenferro_tensor::Result<()> {
    ///     ctx.require_axes_equal((0, 0), (1, 0))?;
    ///     Ok(())
    /// }
    /// ```
    pub fn require_axes_equal(
        &mut self,
        lhs: (usize, usize),
        rhs: (usize, usize),
    ) -> Result<(), ExtensionShapeError> {
        let lhs = self.input_axis(lhs.0, lhs.1)?;
        let rhs = self.input_axis(rhs.0, rhs.1)?;
        self.require_equal(lhs, rhs)
    }

    /// Require two extension inputs to have the same rank and axis extents.
    ///
    /// A rank mismatch is returned before any equality is recorded.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::ExtensionShapeContext;
    ///
    /// fn infer(ctx: &mut ExtensionShapeContext<'_>) -> tenferro_tensor::Result<()> {
    ///     ctx.require_same_shape(0, 1)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn require_same_shape(
        &mut self,
        lhs_input: usize,
        rhs_input: usize,
    ) -> Result<(), ExtensionShapeError> {
        let lhs_shape = self.input_shape(lhs_input)?;
        let rhs_shape = self.input_shape(rhs_input)?;
        if lhs_shape.len() != rhs_shape.len() {
            return Err(ExtensionShapeError::RankMismatch {
                family_id: self.family_id,
                lhs_input,
                lhs_rank: lhs_shape.len(),
                rhs_input,
                rhs_rank: rhs_shape.len(),
            });
        }

        let equalities: Vec<_> = lhs_shape
            .iter()
            .cloned()
            .zip(rhs_shape.iter().cloned())
            .map(|(lhs, rhs)| ExtensionShapeConstraint::equal(lhs, rhs))
            .collect();
        self.constraints.extend(equalities);
        Ok(())
    }

    /// Borrow the constraints collected by this inference call.
    #[doc(hidden)]
    pub fn constraints(&self) -> &[ExtensionShapeConstraint] {
        &self.constraints
    }

    /// Consume the context and return its collected constraints.
    #[doc(hidden)]
    pub fn into_constraints(self) -> Vec<ExtensionShapeConstraint> {
        self.constraints
    }
}

#[cfg(test)]
mod tests;
