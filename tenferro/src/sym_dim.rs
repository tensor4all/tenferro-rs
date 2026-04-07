use std::ops::{Add, Div, Mul, Sub};

use tenferro_ops::dim_expr::DimExpr;

use crate::traced::TracedTensorId;

/// A symbolic tensor dimension expression used to build shape-agnostic graphs.
///
/// `SymDim` values are produced from traced tensor axes via
/// [`TracedTensor::sym_size`](crate::traced::TracedTensor::sym_size) and can be
/// combined with basic arithmetic before being passed to
/// [`TracedTensor::reshape_sym`](crate::traced::TracedTensor::reshape_sym).
///
/// # Examples
///
/// ```rust,ignore
/// let rows = x.sym_size(0);
/// let cols = x.sym_size(1);
/// let y = x.reshape_sym(&[rows * cols])?;
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymDim(pub(crate) RawSymDim);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RawSymDim {
    Const(usize),
    TensorAxis {
        tensor_id: TracedTensorId,
        axis: usize,
    },
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
    /// ```rust,ignore
    /// let clipped = x.sym_size(0).min(8usize);
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
    /// ```rust,ignore
    /// let batch = x.sym_size(0).max(1usize);
    /// ```
    pub fn max(self, other: impl Into<SymDim>) -> Self {
        let other = other.into();
        binary(self, other, |lhs, rhs| {
            RawSymDim::Max(Box::new(lhs), Box::new(rhs))
        })
    }

    pub(crate) fn tensor_axis(tensor_id: TracedTensorId, axis: usize) -> Self {
        Self(RawSymDim::TensorAxis { tensor_id, axis })
    }

    pub(crate) fn constant_value(&self) -> Option<usize> {
        match &self.0 {
            RawSymDim::Const(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn to_dim_expr(
        &self,
        tensor_map: &[(TracedTensorId, usize)],
    ) -> std::result::Result<DimExpr, String> {
        raw_to_dim_expr(&self.0, tensor_map)
    }
}

impl From<usize> for SymDim {
    fn from(value: usize) -> Self {
        Self(RawSymDim::Const(value))
    }
}

fn raw_to_dim_expr(
    raw: &RawSymDim,
    tensor_map: &[(TracedTensorId, usize)],
) -> std::result::Result<DimExpr, String> {
    Ok(match raw {
        RawSymDim::Const(value) => DimExpr::Const(*value),
        RawSymDim::TensorAxis { tensor_id, axis } => {
            let input_idx = tensor_map
                .iter()
                .find_map(|(candidate_id, input_idx)| {
                    (candidate_id == tensor_id).then_some(*input_idx)
                })
                .ok_or_else(|| format!("unknown symbolic tensor id {tensor_id}"))?;
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
