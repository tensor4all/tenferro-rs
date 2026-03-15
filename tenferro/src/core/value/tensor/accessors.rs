use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use super::{AdMode, AdTensor, StructuredTensor, TensorAdState};

impl<T: Scalar> AdTensor<T> {
    /// Returns AD mode.
    pub fn mode(&self) -> AdMode {
        match &self.0 {
            TensorAdState::Primal(_) => AdMode::Primal,
            TensorAdState::Forward { .. } => AdMode::Forward,
            TensorAdState::Reverse { .. } => AdMode::Reverse,
        }
    }

    /// Returns structured primal payload reference.
    pub fn structured_primal(&self) -> &StructuredTensor<T> {
        match &self.0 {
            TensorAdState::Primal(value) => value,
            TensorAdState::Forward { primal, .. } => primal,
            TensorAdState::Reverse { primal, .. } => primal,
        }
    }

    /// Returns compressed primal payload tensor reference.
    pub fn primal(&self) -> &Tensor<T> {
        self.structured_primal().payload()
    }

    /// Returns structured tangent reference when available.
    pub fn structured_tangent(&self) -> Option<&StructuredTensor<T>> {
        match &self.0 {
            TensorAdState::Primal(_) => None,
            TensorAdState::Forward { tangent, .. } => Some(tangent),
            TensorAdState::Reverse { tangent, .. } => tangent.as_ref(),
        }
    }

    /// Returns compressed tangent payload tensor reference when available.
    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.structured_tangent().map(StructuredTensor::payload)
    }

    /// Returns dimensions of the primal tensor.
    pub fn dims(&self) -> &[usize] {
        self.structured_primal().logical_dims()
    }

    /// Returns number of dimensions of the primal tensor.
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns total number of elements in the primal tensor.
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when primal tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns axis classes of the structured primal.
    pub fn axis_classes(&self) -> &[usize] {
        self.structured_primal().axis_classes()
    }

    /// Returns `true` when the structured primal is dense.
    pub fn is_dense(&self) -> bool {
        self.structured_primal().is_dense()
    }

    /// Returns `true` when the structured primal is diagonal.
    pub fn is_diag(&self) -> bool {
        self.structured_primal().is_diag()
    }
}
