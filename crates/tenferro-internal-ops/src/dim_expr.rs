/// Arithmetic expression over tensor dimension sizes.
///
/// Evaluated at execution time from actual input tensor shapes.
/// `InputDim { input_idx, axis }` references the axis size of
/// the op's `input_idx`-th input tensor.
///
/// # Examples
///
/// ```rust
/// use tenferro_ops::dim_expr::DimExpr;
///
/// let expr = DimExpr::mul(
///     DimExpr::InputDim {
///         input_idx: 0,
///         axis: 0,
///     },
///     DimExpr::InputDim {
///         input_idx: 0,
///         axis: 1,
///     },
/// );
/// assert_eq!(expr.eval(&[&[3, 4]]), 12);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DimExpr {
    /// A concrete dimension size.
    Const(usize),
    /// Axis size of the op's `input_idx`-th input tensor.
    InputDim { input_idx: usize, axis: usize },
    /// Sum of two dimension expressions.
    Add(Box<DimExpr>, Box<DimExpr>),
    /// Difference of two dimension expressions.
    Sub(Box<DimExpr>, Box<DimExpr>),
    /// Product of two dimension expressions.
    Mul(Box<DimExpr>, Box<DimExpr>),
    /// Floor division of two dimension expressions.
    FloorDiv(Box<DimExpr>, Box<DimExpr>),
    /// Minimum of two dimension expressions.
    Min(Box<DimExpr>, Box<DimExpr>),
    /// Maximum of two dimension expressions.
    Max(Box<DimExpr>, Box<DimExpr>),
}

/// Error produced while evaluating a [`DimExpr`] against concrete shapes.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DimExprEvalError {
    /// `InputDim` referenced an input that was not provided.
    #[error(
        "DimExpr::InputDim input index {input_idx} out of bounds for {input_count} input shapes"
    )]
    InputOutOfBounds {
        input_idx: usize,
        input_count: usize,
    },
    /// `InputDim` referenced an axis that does not exist on the selected input.
    #[error("DimExpr::InputDim axis {axis} out of bounds for input {input_idx} rank {rank}")]
    AxisOutOfBounds {
        input_idx: usize,
        axis: usize,
        rank: usize,
    },
    /// Addition overflowed `usize`.
    #[error("DimExpr::Add overflow: {lhs} + {rhs}")]
    AddOverflow { lhs: usize, rhs: usize },
    /// Subtraction would underflow `usize`.
    #[error("DimExpr::Sub underflow: left operand {lhs} is smaller than {rhs}")]
    SubUnderflow { lhs: usize, rhs: usize },
    /// Multiplication overflowed `usize`.
    #[error("DimExpr::Mul overflow: {lhs} * {rhs}")]
    MulOverflow { lhs: usize, rhs: usize },
    /// Floor division divisor evaluated to zero.
    #[error("DimExpr::FloorDiv divide by zero: left operand {lhs}, divisor {rhs}")]
    FloorDivByZero { lhs: usize, rhs: usize },
}

impl DimExpr {
    /// Evaluate the expression using actual input tensor shapes.
    ///
    /// # Panics
    ///
    /// Panics if an `InputDim` node references an `input_idx` that is
    /// out of bounds for `input_shapes`, an `axis` that is out of bounds
    /// for the corresponding shape slice, a subtraction would underflow,
    /// or a floor-division divisor evaluates to zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::add(
    ///     DimExpr::InputDim {
    ///         input_idx: 0,
    ///         axis: 0,
    ///     },
    ///     DimExpr::Const(2),
    /// );
    /// assert_eq!(expr.eval(&[&[5, 7]]), 7);
    /// ```
    pub fn eval(&self, input_shapes: &[&[usize]]) -> usize {
        self.try_eval(input_shapes)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallibly evaluate the expression using actual input tensor shapes.
    ///
    /// This is the runtime-facing API. Use it when malformed symbolic shapes
    /// are possible user input; [`Self::eval`] is retained as the compatibility
    /// panic wrapper.
    pub fn try_eval(&self, input_shapes: &[&[usize]]) -> Result<usize, DimExprEvalError> {
        match self {
            Self::Const(v) => Ok(*v),
            Self::InputDim { input_idx, axis } => input_shapes
                .get(*input_idx)
                .ok_or(DimExprEvalError::InputOutOfBounds {
                    input_idx: *input_idx,
                    input_count: input_shapes.len(),
                })
                .and_then(|shape| {
                    shape
                        .get(*axis)
                        .copied()
                        .ok_or(DimExprEvalError::AxisOutOfBounds {
                            input_idx: *input_idx,
                            axis: *axis,
                            rank: shape.len(),
                        })
                }),
            Self::Add(a, b) => {
                let lhs = a.try_eval(input_shapes)?;
                let rhs = b.try_eval(input_shapes)?;
                lhs.checked_add(rhs)
                    .ok_or(DimExprEvalError::AddOverflow { lhs, rhs })
            }
            Self::Sub(a, b) => {
                let lhs = a.try_eval(input_shapes)?;
                let rhs = b.try_eval(input_shapes)?;
                lhs.checked_sub(rhs)
                    .ok_or(DimExprEvalError::SubUnderflow { lhs, rhs })
            }
            Self::Mul(a, b) => {
                let lhs = a.try_eval(input_shapes)?;
                let rhs = b.try_eval(input_shapes)?;
                lhs.checked_mul(rhs)
                    .ok_or(DimExprEvalError::MulOverflow { lhs, rhs })
            }
            Self::FloorDiv(a, b) => {
                let lhs = a.try_eval(input_shapes)?;
                let rhs = b.try_eval(input_shapes)?;
                if rhs == 0 {
                    return Err(DimExprEvalError::FloorDivByZero { lhs, rhs });
                }
                Ok(lhs / rhs)
            }
            Self::Min(a, b) => Ok(a.try_eval(input_shapes)?.min(b.try_eval(input_shapes)?)),
            Self::Max(a, b) => Ok(a.try_eval(input_shapes)?.max(b.try_eval(input_shapes)?)),
        }
    }

    /// Return the maximum referenced `input_idx`, or `None` if the expression
    /// contains only constants.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::add(
    ///     DimExpr::InputDim {
    ///         input_idx: 0,
    ///         axis: 0,
    ///     },
    ///     DimExpr::InputDim {
    ///         input_idx: 2,
    ///         axis: 1,
    ///     },
    /// );
    /// assert_eq!(expr.max_input_idx(), Some(2));
    /// ```
    pub fn max_input_idx(&self) -> Option<usize> {
        match self {
            Self::Const(_) => None,
            Self::InputDim { input_idx, .. } => Some(*input_idx),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::FloorDiv(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => match (a.max_input_idx(), b.max_input_idx()) {
                (Some(x), Some(y)) => Some(x.max(y)),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            },
        }
    }

    /// Remap `InputDim { input_idx: from, .. }` to `InputDim { input_idx: to, .. }`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::InputDim {
    ///     input_idx: 0,
    ///     axis: 1,
    /// };
    /// assert_eq!(expr.remap(0, 2), DimExpr::InputDim { input_idx: 2, axis: 1 });
    /// ```
    pub fn remap(&self, from: usize, to: usize) -> Self {
        match self {
            Self::Const(v) => Self::Const(*v),
            Self::InputDim { input_idx, axis } => Self::InputDim {
                input_idx: if *input_idx == from { to } else { *input_idx },
                axis: *axis,
            },
            Self::Add(a, b) => Self::add(a.remap(from, to), b.remap(from, to)),
            Self::Sub(a, b) => Self::sub(a.remap(from, to), b.remap(from, to)),
            Self::Mul(a, b) => Self::mul(a.remap(from, to), b.remap(from, to)),
            Self::FloorDiv(a, b) => Self::floor_div(a.remap(from, to), b.remap(from, to)),
            Self::Min(a, b) => Self::min(a.remap(from, to), b.remap(from, to)),
            Self::Max(a, b) => Self::max(a.remap(from, to), b.remap(from, to)),
        }
    }

    /// Construct a constant dimension expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// assert_eq!(DimExpr::constant(4), DimExpr::Const(4));
    /// ```
    pub fn constant(v: usize) -> Self {
        Self::Const(v)
    }

    /// Construct an addition node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::add(DimExpr::Const(2), DimExpr::Const(3));
    /// assert_eq!(expr.eval(&[]), 5);
    /// ```
    // Public constructor names mirror the DimExpr variants; operator traits are a separate API choice.
    #[allow(clippy::should_implement_trait)]
    pub fn add(a: Self, b: Self) -> Self {
        Self::Add(Box::new(a), Box::new(b))
    }

    /// Construct a subtraction node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::sub(DimExpr::Const(7), DimExpr::Const(2));
    /// assert_eq!(expr.eval(&[]), 5);
    /// ```
    // Public constructor names mirror the DimExpr variants; operator traits are a separate API choice.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(a: Self, b: Self) -> Self {
        Self::Sub(Box::new(a), Box::new(b))
    }

    /// Construct a multiplication node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::mul(DimExpr::Const(3), DimExpr::Const(4));
    /// assert_eq!(expr.eval(&[]), 12);
    /// ```
    // Public constructor names mirror the DimExpr variants; operator traits are a separate API choice.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(a: Self, b: Self) -> Self {
        Self::Mul(Box::new(a), Box::new(b))
    }

    /// Construct a floor-division node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::floor_div(DimExpr::Const(9), DimExpr::Const(2));
    /// assert_eq!(expr.eval(&[]), 4);
    /// ```
    pub fn floor_div(a: Self, b: Self) -> Self {
        Self::FloorDiv(Box::new(a), Box::new(b))
    }

    /// Construct a minimum node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::min(DimExpr::Const(3), DimExpr::Const(5));
    /// assert_eq!(expr.eval(&[]), 3);
    /// ```
    pub fn min(a: Self, b: Self) -> Self {
        Self::Min(Box::new(a), Box::new(b))
    }

    /// Construct a maximum node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let expr = DimExpr::max(DimExpr::Const(3), DimExpr::Const(5));
    /// assert_eq!(expr.eval(&[]), 5);
    /// ```
    pub fn max(a: Self, b: Self) -> Self {
        Self::Max(Box::new(a), Box::new(b))
    }

    /// Return `true` when this expression is a constant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// assert!(DimExpr::Const(3).is_const());
    /// ```
    pub fn is_const(&self) -> bool {
        matches!(self, Self::Const(_))
    }

    /// Convert a concrete shape to constant expressions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// assert_eq!(DimExpr::from_concrete(&[2, 3]), vec![DimExpr::Const(2), DimExpr::Const(3)]);
    /// ```
    pub fn from_concrete(shape: &[usize]) -> Vec<Self> {
        shape.iter().map(|&v| Self::Const(v)).collect()
    }

    /// Build `[InputDim(input_idx, 0), ..., InputDim(input_idx, rank - 1)]`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let shape = DimExpr::input_shape(1, 2);
    /// assert_eq!(
    ///     shape,
    ///     vec![
    ///         DimExpr::InputDim { input_idx: 1, axis: 0 },
    ///         DimExpr::InputDim { input_idx: 1, axis: 1 },
    ///     ]
    /// );
    /// ```
    pub fn input_shape(input_idx: usize, rank: usize) -> Vec<Self> {
        (0..rank)
            .map(|axis| Self::InputDim { input_idx, axis })
            .collect()
    }

    /// Evaluate a slice of expressions against actual input shapes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let exprs = vec![
    ///     DimExpr::InputDim { input_idx: 0, axis: 0 },
    ///     DimExpr::Const(4),
    /// ];
    /// assert_eq!(DimExpr::eval_all(&exprs, &[&[3, 5]]), vec![3, 4]);
    /// ```
    pub fn eval_all(exprs: &[Self], input_shapes: &[&[usize]]) -> Vec<usize> {
        exprs.iter().map(|e| e.eval(input_shapes)).collect()
    }

    /// Fallibly evaluate a slice of expressions against actual input shapes.
    pub fn try_eval_all(
        exprs: &[Self],
        input_shapes: &[&[usize]],
    ) -> Result<Vec<usize>, DimExprEvalError> {
        exprs.iter().map(|e| e.try_eval(input_shapes)).collect()
    }

    /// Remap all `InputDim` references in a slice of expressions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let exprs = vec![DimExpr::InputDim { input_idx: 0, axis: 0 }];
    /// assert_eq!(
    ///     DimExpr::remap_all(&exprs, 0, 1),
    ///     vec![DimExpr::InputDim { input_idx: 1, axis: 0 }]
    /// );
    /// ```
    pub fn remap_all(exprs: &[Self], from: usize, to: usize) -> Vec<Self> {
        exprs.iter().map(|e| e.remap(from, to)).collect()
    }

    /// Compute the maximum referenced `input_idx` across a slice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::dim_expr::DimExpr;
    ///
    /// let exprs = vec![
    ///     DimExpr::InputDim { input_idx: 0, axis: 0 },
    ///     DimExpr::InputDim { input_idx: 2, axis: 1 },
    /// ];
    /// assert_eq!(DimExpr::max_input_idx_all(&exprs), Some(2));
    /// ```
    pub fn max_input_idx_all(exprs: &[Self]) -> Option<usize> {
        exprs.iter().filter_map(Self::max_input_idx).max()
    }
}

impl From<usize> for DimExpr {
    fn from(v: usize) -> Self {
        Self::Const(v)
    }
}

impl From<&DimExpr> for DimExpr {
    fn from(value: &DimExpr) -> Self {
        value.clone()
    }
}
