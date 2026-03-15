use tenferro_tensor::MemoryOrder;

use super::layout::{
    contiguous_ad_tensor_typed, diag_embed_ad_tensor_typed, reshape_ad_tensor_typed,
    take_prefix_ad_tensor_typed,
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
}
