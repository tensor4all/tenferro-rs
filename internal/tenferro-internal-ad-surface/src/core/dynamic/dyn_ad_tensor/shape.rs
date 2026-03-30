use tenferro_internal_ad_core::DynAdTensorRef;
use tenferro_tensor::MemoryOrder;

use super::layout::{
    contiguous_ad_tensor_typed, diag_embed_ad_tensor_typed, permute_ad_tensor_typed,
    reshape_ad_tensor_typed, take_prefix_ad_tensor_typed, view_ad_tensor_typed,
};
use super::Tensor;
use crate::Result;

macro_rules! match_dyn_ad_tensor_ref {
    ($tensor:expr, |$value:ident| $body:block) => {{
        match $tensor.as_dyn_ad_ref() {
            DynAdTensorRef::F32($value) => $body,
            DynAdTensorRef::F64($value) => $body,
            DynAdTensorRef::C32($value) => $body,
            DynAdTensorRef::C64($value) => $body,
        }
    }};
}

impl Tensor {
    /// Returns primal tensor dimensions.
    pub fn dims(&self) -> &[usize] {
        self.as_dyn_ad_ref().dims()
    }

    /// Returns axis classes of the structured primal.
    pub fn axis_classes(&self) -> &[usize] {
        self.as_dyn_ad_ref().axis_classes()
    }

    /// Returns `true` when the structured primal is dense.
    pub fn is_dense(&self) -> bool {
        self.as_dyn_ad_ref().is_dense()
    }

    /// Returns `true` when the structured primal is diagonal.
    pub fn is_diag(&self) -> bool {
        self.as_dyn_ad_ref().is_diag()
    }

    /// Return a zero-copy view of a dense tensor while preserving AD mode.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let y = x.view(&[4]).unwrap();
    /// assert_eq!(y.dims(), &[4]);
    /// ```
    pub fn view(&self, new_dims: &[usize]) -> Result<Self> {
        match_dyn_ad_tensor_ref!(self, |value| {
            Ok(Self::from(view_ad_tensor_typed(value, new_dims)?))
        })
    }

    /// Reshape a dense tensor while preserving AD mode.
    ///
    /// Returns a zero-copy view when the current layout is compatible and
    /// otherwise materializes a contiguous tensor first.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let y = x.reshape(&[4]).unwrap();
    /// assert_eq!(y.dims(), &[4]);
    /// ```
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Self> {
        match_dyn_ad_tensor_ref!(self, |value| {
            Ok(Self::from(reshape_ad_tensor_typed(value, new_dims)?))
        })
    }

    /// Permutes logical tensor axes while preserving AD mode.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let y = x.permute(&[1, 0]).unwrap();
    /// assert_eq!(y.dims(), &[2, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        match_dyn_ad_tensor_ref!(self, |value| {
            Ok(Self::from(permute_ad_tensor_typed(value, perm)?))
        })
    }

    /// Take the first `len` entries along `axis` for a dense tensor.
    pub fn take_prefix(&self, axis: usize, len: usize) -> Result<Self> {
        match_dyn_ad_tensor_ref!(self, |value| {
            Ok(Self::from(take_prefix_ad_tensor_typed(value, axis, len)?))
        })
    }

    /// Embed a rank-1 dense tensor as a structured diagonal tensor.
    pub fn diag_embed(&self, logical_rank: usize) -> Result<Self> {
        match_dyn_ad_tensor_ref!(self, |value| {
            Ok(Self::from(diag_embed_ad_tensor_typed(value, logical_rank)?))
        })
    }

    /// Builds a rank-2 diagonal tensor from a dense rank-1 vector payload.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let vector = Tensor::from_slice(&[2.0_f64, 3.0], &[2]).unwrap();
    /// let diag = Tensor::diag(&vector).unwrap();
    /// assert!(diag.is_diag());
    /// assert_eq!(diag.dims(), &[2, 2]);
    /// ```
    pub fn diag(vector: &Self) -> Result<Self> {
        vector.diag_embed(2)
    }

    /// Returns a logically identical tensor with payloads made contiguous in `order`.
    pub fn contiguous(&self, order: MemoryOrder) -> Result<Self> {
        match_dyn_ad_tensor_ref!(self, |value| {
            Ok(Self::from(contiguous_ad_tensor_typed(value, order)?))
        })
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
}
