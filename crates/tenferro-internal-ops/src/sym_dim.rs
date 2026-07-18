use std::ops::{Add, Div, Mul, Sub};

use thiserror::Error;

use crate::dim_expr::DimExpr;

/// A symbolic tensor dimension expression used to build shape-agnostic graphs.
///
/// `SymDim` values can be combined with basic arithmetic and later resolved
/// against op-local [`DimExpr`] expressions when shape metadata is propagated.
///
/// # Examples
///
/// ```rust
/// use tenferro_ops::sym_dim::SymDim;
///
/// let rows = SymDim::from(3usize);
/// let cols = SymDim::from(4usize);
/// let area = rows * cols;
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymDim(pub(crate) RawSymDim);

/// A symbolic dimension referenced a tensor that is not present in the
/// conversion map supplied by the graph builder.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("unknown symbolic tensor id {tensor_id}")]
pub struct SymDimConversionError {
    /// Identifier of the missing symbolic tensor.
    pub tensor_id: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RawSymDim {
    Const(usize),
    TensorAxis { tensor_id: u64, axis: usize },
    Add(Box<RawSymDim>, Box<RawSymDim>),
    Sub(Box<RawSymDim>, Box<RawSymDim>),
    Mul(Box<RawSymDim>, Box<RawSymDim>),
    FloorDiv(Box<RawSymDim>, Box<RawSymDim>),
    Min(Box<RawSymDim>, Box<RawSymDim>),
    Max(Box<RawSymDim>, Box<RawSymDim>),
}

impl SymDim {
    /// Return the smaller of two symbolic dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::sym_dim::SymDim;
    ///
    /// let clipped = SymDim::from(5usize).min(8usize);
    /// ```
    pub fn min(self, other: impl Into<SymDim>) -> Self {
        let other = other.into();
        binary(self, other, |lhs, rhs| {
            RawSymDim::Min(Box::new(lhs), Box::new(rhs))
        })
    }

    /// Return the larger of two symbolic dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::sym_dim::SymDim;
    ///
    /// let batch = SymDim::from(2usize).max(1usize);
    /// ```
    pub fn max(self, other: impl Into<SymDim>) -> Self {
        let other = other.into();
        binary(self, other, |lhs, rhs| {
            RawSymDim::Max(Box::new(lhs), Box::new(rhs))
        })
    }

    #[doc(hidden)]
    pub fn tensor_axis(tensor_id: u64, axis: usize) -> Self {
        Self(RawSymDim::TensorAxis { tensor_id, axis })
    }

    #[doc(hidden)]
    pub fn constant_value(&self) -> Option<usize> {
        raw_constant_value(&self.0)
    }

    #[doc(hidden)]
    pub fn from_dim_expr(expr: &DimExpr, input_shapes: &[&[SymDim]]) -> Self {
        match expr {
            DimExpr::Const(value) => Self::from(*value),
            DimExpr::InputDim { input_idx, axis } => input_shapes[*input_idx][*axis].clone(),
            DimExpr::Add(lhs, rhs) => {
                Self::from_dim_expr(lhs, input_shapes) + Self::from_dim_expr(rhs, input_shapes)
            }
            DimExpr::Sub(lhs, rhs) => {
                Self::from_dim_expr(lhs, input_shapes) - Self::from_dim_expr(rhs, input_shapes)
            }
            DimExpr::Mul(lhs, rhs) => {
                Self::from_dim_expr(lhs, input_shapes) * Self::from_dim_expr(rhs, input_shapes)
            }
            DimExpr::FloorDiv(lhs, rhs) => {
                Self::from_dim_expr(lhs, input_shapes) / Self::from_dim_expr(rhs, input_shapes)
            }
            DimExpr::Min(lhs, rhs) => {
                Self::from_dim_expr(lhs, input_shapes).min(Self::from_dim_expr(rhs, input_shapes))
            }
            DimExpr::Max(lhs, rhs) => {
                Self::from_dim_expr(lhs, input_shapes).max(Self::from_dim_expr(rhs, input_shapes))
            }
        }
    }

    #[doc(hidden)]
    pub fn to_dim_expr(
        &self,
        tensor_map: &[(u64, usize)],
    ) -> std::result::Result<DimExpr, SymDimConversionError> {
        raw_to_dim_expr(&self.0, tensor_map)
    }

    /// Collect the unique traced tensor IDs referenced by `TensorAxis`
    /// variants inside this expression, in traversal order (first
    /// occurrence wins).
    ///
    /// Used by traced composition wrappers that build multi-input ops
    /// from `SymDim`-valued target shapes — e.g.
    /// `TracedTensor::broadcast_in_dim_sym`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::sym_dim::SymDim;
    ///
    /// let lhs = SymDim::tensor_axis(7, 0);
    /// let rhs = SymDim::tensor_axis(9, 1);
    /// let sum = lhs + rhs;
    /// assert_eq!(sum.referenced_tensor_ids(), vec![7, 9]);
    /// ```
    pub fn referenced_tensor_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        collect_tensor_ids(&self.0, &mut ids);
        ids
    }
}

fn collect_tensor_ids(raw: &RawSymDim, ids: &mut Vec<u64>) {
    match raw {
        RawSymDim::Const(_) => {}
        RawSymDim::TensorAxis { tensor_id, .. } => {
            if !ids.contains(tensor_id) {
                ids.push(*tensor_id);
            }
        }
        RawSymDim::Add(lhs, rhs)
        | RawSymDim::Sub(lhs, rhs)
        | RawSymDim::Mul(lhs, rhs)
        | RawSymDim::FloorDiv(lhs, rhs)
        | RawSymDim::Min(lhs, rhs)
        | RawSymDim::Max(lhs, rhs) => {
            collect_tensor_ids(lhs, ids);
            collect_tensor_ids(rhs, ids);
        }
    }
}

fn raw_constant_value(raw: &RawSymDim) -> Option<usize> {
    match raw {
        RawSymDim::Const(value) => Some(*value),
        RawSymDim::TensorAxis { .. } => None,
        RawSymDim::Add(lhs, rhs) => raw_constant_value(lhs)?.checked_add(raw_constant_value(rhs)?),
        RawSymDim::Sub(lhs, rhs) => raw_constant_value(lhs)?.checked_sub(raw_constant_value(rhs)?),
        RawSymDim::Mul(lhs, rhs) => raw_constant_value(lhs)?.checked_mul(raw_constant_value(rhs)?),
        RawSymDim::FloorDiv(lhs, rhs) => {
            let rhs = raw_constant_value(rhs)?;
            raw_constant_value(lhs)?.checked_div(rhs)
        }
        RawSymDim::Min(lhs, rhs) => Some(raw_constant_value(lhs)?.min(raw_constant_value(rhs)?)),
        RawSymDim::Max(lhs, rhs) => Some(raw_constant_value(lhs)?.max(raw_constant_value(rhs)?)),
    }
}

impl From<usize> for SymDim {
    fn from(value: usize) -> Self {
        Self(RawSymDim::Const(value))
    }
}

fn raw_to_dim_expr(
    raw: &RawSymDim,
    tensor_map: &[(u64, usize)],
) -> std::result::Result<DimExpr, SymDimConversionError> {
    Ok(match raw {
        RawSymDim::Const(value) => DimExpr::Const(*value),
        RawSymDim::TensorAxis { tensor_id, axis } => {
            let input_idx = tensor_map
                .iter()
                .find_map(|(candidate_id, input_idx)| {
                    (candidate_id == tensor_id).then_some(*input_idx)
                })
                .ok_or(SymDimConversionError {
                    tensor_id: *tensor_id,
                })?;
            DimExpr::InputDim {
                input_idx,
                axis: *axis,
            }
        }
        RawSymDim::Add(lhs, rhs) => DimExpr::add(
            raw_to_dim_expr(lhs, tensor_map)?,
            raw_to_dim_expr(rhs, tensor_map)?,
        ),
        RawSymDim::Sub(lhs, rhs) => DimExpr::sub(
            raw_to_dim_expr(lhs, tensor_map)?,
            raw_to_dim_expr(rhs, tensor_map)?,
        ),
        RawSymDim::Mul(lhs, rhs) => DimExpr::mul(
            raw_to_dim_expr(lhs, tensor_map)?,
            raw_to_dim_expr(rhs, tensor_map)?,
        ),
        RawSymDim::FloorDiv(lhs, rhs) => DimExpr::floor_div(
            raw_to_dim_expr(lhs, tensor_map)?,
            raw_to_dim_expr(rhs, tensor_map)?,
        ),
        RawSymDim::Min(lhs, rhs) => DimExpr::min(
            raw_to_dim_expr(lhs, tensor_map)?,
            raw_to_dim_expr(rhs, tensor_map)?,
        ),
        RawSymDim::Max(lhs, rhs) => DimExpr::max(
            raw_to_dim_expr(lhs, tensor_map)?,
            raw_to_dim_expr(rhs, tensor_map)?,
        ),
    })
}

fn binary(lhs: SymDim, rhs: SymDim, f: impl FnOnce(RawSymDim, RawSymDim) -> RawSymDim) -> SymDim {
    SymDim(f(lhs.0, rhs.0))
}

impl Add for SymDim {
    type Output = SymDim;

    fn add(self, rhs: SymDim) -> Self::Output {
        binary(self, rhs, |lhs, rhs| {
            RawSymDim::Add(Box::new(lhs), Box::new(rhs))
        })
    }
}

impl Add<usize> for SymDim {
    type Output = SymDim;

    fn add(self, rhs: usize) -> Self::Output {
        self + SymDim::from(rhs)
    }
}

impl Add<SymDim> for usize {
    type Output = SymDim;

    fn add(self, rhs: SymDim) -> Self::Output {
        SymDim::from(self) + rhs
    }
}

impl Sub for SymDim {
    type Output = SymDim;

    fn sub(self, rhs: SymDim) -> Self::Output {
        binary(self, rhs, |lhs, rhs| {
            RawSymDim::Sub(Box::new(lhs), Box::new(rhs))
        })
    }
}

impl Sub<usize> for SymDim {
    type Output = SymDim;

    fn sub(self, rhs: usize) -> Self::Output {
        self - SymDim::from(rhs)
    }
}

impl Sub<SymDim> for usize {
    type Output = SymDim;

    fn sub(self, rhs: SymDim) -> Self::Output {
        SymDim::from(self) - rhs
    }
}

impl Mul for SymDim {
    type Output = SymDim;

    fn mul(self, rhs: SymDim) -> Self::Output {
        binary(self, rhs, |lhs, rhs| {
            RawSymDim::Mul(Box::new(lhs), Box::new(rhs))
        })
    }
}

impl Mul<usize> for SymDim {
    type Output = SymDim;

    fn mul(self, rhs: usize) -> Self::Output {
        self * SymDim::from(rhs)
    }
}

impl Mul<SymDim> for usize {
    type Output = SymDim;

    fn mul(self, rhs: SymDim) -> Self::Output {
        SymDim::from(self) * rhs
    }
}

impl Div for SymDim {
    type Output = SymDim;

    fn div(self, rhs: SymDim) -> Self::Output {
        binary(self, rhs, |lhs, rhs| {
            RawSymDim::FloorDiv(Box::new(lhs), Box::new(rhs))
        })
    }
}

impl Div<usize> for SymDim {
    type Output = SymDim;

    fn div(self, rhs: usize) -> Self::Output {
        self / SymDim::from(rhs)
    }
}

impl Div<SymDim> for usize {
    type Output = SymDim;

    fn div(self, rhs: SymDim) -> Self::Output {
        SymDim::from(self) / rhs
    }
}
