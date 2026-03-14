use chainrules::Tape;
use num_complex::{Complex32, Complex64};
use tenferro_tensor::MemoryOrder;

use super::super::ScalarType;
use super::layout::{
    contiguous_ad_tensor_typed, diag_embed_ad_tensor_typed, reshape_ad_tensor_typed,
    take_prefix_ad_tensor_typed,
};
use super::DynAdTensor;
use crate::{AdMode, AdTensor, DynTensor, Error, NodeId, Result, StructuredTensor};

fn reverse_tape_from_anchor(
    anchor: &DynAdTensor,
    op_name: &'static str,
) -> Result<Tape<DynTensor>> {
    let tape = match anchor {
        DynAdTensor::F32(value) => value.reverse_tape(),
        DynAdTensor::F64(value) => value.reverse_tape(),
        DynAdTensor::C32(value) => value.reverse_tape(),
        DynAdTensor::C64(value) => value.reverse_tape(),
    };
    tape.cloned().ok_or_else(|| Error::InvalidAdTensor {
        message: format!("{op_name} requires a reverse-mode DynAdTensor anchor"),
    })
}

impl DynAdTensor {
    /// Creates a primal tensor value from a dense or structured tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{DynAdTensor, StructuredTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// assert!(dense.is_dense());
    ///
    /// let diag = DynAdTensor::new_primal(
    ///     StructuredTensor::from_diagonal_vector(
    ///         Tensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    ///         2,
    ///     )
    ///     .unwrap(),
    /// );
    /// assert!(diag.is_diag());
    /// ```
    pub fn new_primal<T>(tensor: impl Into<StructuredTensor<T>>) -> Self
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        AdTensor::new_primal(tensor).into()
    }

    /// Creates a forward-mode tensor value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_forward(
    ///     Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    ///     Tensor::<f64>::from_slice(&[0.5], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .unwrap();
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn new_forward<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        Ok(AdTensor::new_forward(primal, tangent)?.into())
    }

    /// Creates a reverse-mode leaf on a fresh reverse graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_reverse_leaf(
    ///     Tensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .unwrap();
    /// assert_eq!(x.mode(), AdMode::Reverse);
    /// assert!(x.tape_id().is_some());
    /// ```
    pub fn new_reverse_leaf<T>(primal: impl Into<StructuredTensor<T>>) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tape = Tape::new();
        Ok(AdTensor::new_reverse_leaf(primal, &tape)?.into())
    }

    /// Creates a reverse-mode leaf with a tangent seed for HVP on a fresh graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_reverse_leaf_with_tangent(
    ///     Tensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    ///     Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .unwrap();
    /// assert_eq!(x.mode(), AdMode::Reverse);
    /// ```
    pub fn new_reverse_leaf_with_tangent<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tape = Tape::new();
        Ok(AdTensor::new_reverse_leaf_with_tangent(primal, tangent, &tape)?.into())
    }

    /// Creates another reverse-mode leaf on the same graph as `self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_reverse_leaf(
    ///     Tensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .unwrap();
    /// let y = x
    ///     .new_reverse_sibling(
    ///         Tensor::<f64>::from_slice(&[3.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    ///     )
    ///     .unwrap();
    /// assert!(x.shares_reverse_graph(&y));
    /// assert_eq!(y.mode(), AdMode::Reverse);
    /// ```
    pub fn new_reverse_sibling<T>(&self, primal: impl Into<StructuredTensor<T>>) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tape = reverse_tape_from_anchor(self, "DynAdTensor::new_reverse_sibling")?;
        Ok(AdTensor::new_reverse_leaf(primal, &tape)?.into())
    }

    /// Creates another reverse-mode leaf with a tangent seed on the same graph as `self`.
    pub fn new_reverse_sibling_with_tangent<T>(
        &self,
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tape = reverse_tape_from_anchor(self, "DynAdTensor::new_reverse_sibling_with_tangent")?;
        Ok(AdTensor::new_reverse_leaf_with_tangent(primal, tangent, &tape)?.into())
    }

    /// Returns runtime scalar type.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    /// Returns AD mode.
    pub fn mode(&self) -> AdMode {
        match self {
            Self::F32(v) => v.mode(),
            Self::F64(v) => v.mode(),
            Self::C32(v) => v.mode(),
            Self::C64(v) => v.mode(),
        }
    }

    /// Returns primal tensor dimensions.
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F32(v) => v.dims(),
            Self::F64(v) => v.dims(),
            Self::C32(v) => v.dims(),
            Self::C64(v) => v.dims(),
        }
    }

    /// Returns axis classes of the structured primal.
    pub fn axis_classes(&self) -> &[usize] {
        match self {
            Self::F32(v) => v.axis_classes(),
            Self::F64(v) => v.axis_classes(),
            Self::C32(v) => v.axis_classes(),
            Self::C64(v) => v.axis_classes(),
        }
    }

    /// Returns `true` when the structured primal is dense.
    pub fn is_dense(&self) -> bool {
        match self {
            Self::F32(v) => v.is_dense(),
            Self::F64(v) => v.is_dense(),
            Self::C32(v) => v.is_dense(),
            Self::C64(v) => v.is_dense(),
        }
    }

    /// Returns `true` when the structured primal is diagonal.
    pub fn is_diag(&self) -> bool {
        match self {
            Self::F32(v) => v.is_diag(),
            Self::F64(v) => v.is_diag(),
            Self::C32(v) => v.is_diag(),
            Self::C64(v) => v.is_diag(),
        }
    }

    /// Reshape a dense tensor while preserving AD mode.
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(reshape_ad_tensor_typed(v, new_dims)?)),
            Self::F64(v) => Ok(Self::F64(reshape_ad_tensor_typed(v, new_dims)?)),
            Self::C32(v) => Ok(Self::C32(reshape_ad_tensor_typed(v, new_dims)?)),
            Self::C64(v) => Ok(Self::C64(reshape_ad_tensor_typed(v, new_dims)?)),
        }
    }

    /// Take the first `len` entries along `axis` for a dense tensor.
    pub fn take_prefix(&self, axis: usize, len: usize) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(take_prefix_ad_tensor_typed(v, axis, len)?)),
            Self::F64(v) => Ok(Self::F64(take_prefix_ad_tensor_typed(v, axis, len)?)),
            Self::C32(v) => Ok(Self::C32(take_prefix_ad_tensor_typed(v, axis, len)?)),
            Self::C64(v) => Ok(Self::C64(take_prefix_ad_tensor_typed(v, axis, len)?)),
        }
    }

    /// Embed a rank-1 dense tensor as a structured diagonal tensor.
    pub fn diag_embed(&self, logical_rank: usize) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(diag_embed_ad_tensor_typed(v, logical_rank)?)),
            Self::F64(v) => Ok(Self::F64(diag_embed_ad_tensor_typed(v, logical_rank)?)),
            Self::C32(v) => Ok(Self::C32(diag_embed_ad_tensor_typed(v, logical_rank)?)),
            Self::C64(v) => Ok(Self::C64(diag_embed_ad_tensor_typed(v, logical_rank)?)),
        }
    }

    /// Returns a logically identical tensor with payloads made contiguous in `order`.
    pub fn contiguous(&self, order: MemoryOrder) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(contiguous_ad_tensor_typed(v, order)?)),
            Self::F64(v) => Ok(Self::F64(contiguous_ad_tensor_typed(v, order)?)),
            Self::C32(v) => Ok(Self::C32(contiguous_ad_tensor_typed(v, order)?)),
            Self::C64(v) => Ok(Self::C64(contiguous_ad_tensor_typed(v, order)?)),
        }
    }

    /// Returns primal tensor rank.
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns primal tensor element count.
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when primal tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the reverse-mode tape identifier when attached to a graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynAdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_reverse_leaf(
    ///     Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .unwrap();
    /// assert!(x.tape_id().is_some());
    /// ```
    pub fn tape_id(&self) -> Option<u64> {
        match self {
            Self::F32(v) => v.tape().map(|tape: &Tape<DynTensor>| tape.id() as u64),
            Self::F64(v) => v.tape().map(|tape: &Tape<DynTensor>| tape.id() as u64),
            Self::C32(v) => v.tape().map(|tape: &Tape<DynTensor>| tape.id() as u64),
            Self::C64(v) => v.tape().map(|tape: &Tape<DynTensor>| tape.id() as u64),
        }
    }

    /// Returns the reverse-mode node identifier when attached to a graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynAdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_reverse_leaf(
    ///     Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .unwrap();
    /// assert!(x.node_id().is_some());
    /// ```
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::F32(v) => v.node_id(),
            Self::F64(v) => v.node_id(),
            Self::C32(v) => v.node_id(),
            Self::C64(v) => v.node_id(),
        }
    }

    /// Returns `true` when both tensors participate in the same reverse graph.
    pub fn shares_reverse_graph(&self, other: &Self) -> bool {
        match (
            reverse_tape_from_anchor(self, "DynAdTensor::shares_reverse_graph"),
            reverse_tape_from_anchor(other, "DynAdTensor::shares_reverse_graph"),
        ) {
            (Ok(lhs), Ok(rhs)) => lhs.same_tape(&rhs),
            _ => false,
        }
    }

    /// Returns typed AD tensor ref when dtype is `f32`.
    pub fn as_f32(&self) -> Option<&AdTensor<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `f64`.
    pub fn as_f64(&self) -> Option<&AdTensor<f64>> {
        if let Self::F64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `Complex32`.
    pub fn as_c32(&self) -> Option<&AdTensor<Complex32>> {
        if let Self::C32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `Complex64`.
    pub fn as_c64(&self) -> Option<&AdTensor<Complex64>> {
        if let Self::C64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns true when scalar dtype is complex.
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::C32(_) | Self::C64(_))
    }

    /// Returns true when scalar dtype is real.
    pub fn is_real(&self) -> bool {
        matches!(self, Self::F32(_) | Self::F64(_))
    }
}
