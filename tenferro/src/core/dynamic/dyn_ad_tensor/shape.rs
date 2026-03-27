use tenferro_tensor::MemoryOrder;

use super::layout::{
    contiguous_ad_tensor_typed, diag_embed_ad_tensor_typed, permute_ad_tensor_typed,
    reshape_ad_tensor_typed, take_prefix_ad_tensor_typed, view_ad_tensor_typed,
};
use super::Tensor;
use crate::Result;

impl Tensor {
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

    /// Return a zero-copy view of a dense tensor while preserving AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let y = x.view(&[4]).unwrap();
    /// assert_eq!(y.dims(), &[4]);
    /// ```
    pub fn view(&self, new_dims: &[usize]) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(view_ad_tensor_typed(v, new_dims)?)),
            Self::F64(v) => Ok(Self::F64(view_ad_tensor_typed(v, new_dims)?)),
            Self::C32(v) => Ok(Self::C32(view_ad_tensor_typed(v, new_dims)?)),
            Self::C64(v) => Ok(Self::C64(view_ad_tensor_typed(v, new_dims)?)),
        }
    }

    /// Reshape a dense tensor while preserving AD mode.
    ///
    /// Returns a zero-copy view when the current layout is compatible and
    /// otherwise materializes a contiguous tensor first.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let y = x.reshape(&[4]).unwrap();
    /// assert_eq!(y.dims(), &[4]);
    /// ```
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(reshape_ad_tensor_typed(v, new_dims)?)),
            Self::F64(v) => Ok(Self::F64(reshape_ad_tensor_typed(v, new_dims)?)),
            Self::C32(v) => Ok(Self::C32(reshape_ad_tensor_typed(v, new_dims)?)),
            Self::C64(v) => Ok(Self::C64(reshape_ad_tensor_typed(v, new_dims)?)),
        }
    }

    /// Permutes logical tensor axes while preserving AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let y = x.permute(&[1, 0]).unwrap();
    /// assert_eq!(y.dims(), &[2, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(permute_ad_tensor_typed(v, perm)?)),
            Self::F64(v) => Ok(Self::F64(permute_ad_tensor_typed(v, perm)?)),
            Self::C32(v) => Ok(Self::C32(permute_ad_tensor_typed(v, perm)?)),
            Self::C64(v) => Ok(Self::C64(permute_ad_tensor_typed(v, perm)?)),
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

    /// Builds a rank-2 diagonal tensor from a dense rank-1 vector payload.
    ///
    /// # Examples
    ///
    /// ```rust
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
}
