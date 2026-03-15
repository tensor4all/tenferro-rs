use chainrules_core::Differentiable;
use tenferro_algebra::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::{accumulate_structured_tangent, StructuredTensor};

impl<T> Differentiable for StructuredTensor<T>
where
    T: Scalar,
{
    type Tangent = StructuredTensor<T>;

    fn zero_tangent(&self) -> Self::Tangent {
        let payload = Tensor::zeros(
            self.payload().dims(),
            self.payload().logical_memory_space(),
            MemoryOrder::ColumnMajor,
        );
        match StructuredTensor::new(
            self.logical_dims().to_vec(),
            self.axis_classes().to_vec(),
            payload,
        ) {
            Ok(value) => value,
            Err(err) => {
                unreachable!("StructuredTensor::zero_tangent should preserve a valid layout: {err}")
            }
        }
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        match accumulate_structured_tangent(a, b) {
            Ok(value) => value,
            Err(err) => unreachable!(
                "StructuredTensor::accumulate_tangent requires matching structured layouts: {err}"
            ),
        }
    }

    fn num_elements(&self) -> usize {
        self.logical_dims().iter().product()
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        let payload = Tensor::ones(
            self.payload().dims(),
            self.payload().logical_memory_space(),
            MemoryOrder::ColumnMajor,
        );
        match StructuredTensor::new(
            self.logical_dims().to_vec(),
            self.axis_classes().to_vec(),
            payload,
        ) {
            Ok(value) => value,
            Err(err) => unreachable!(
                "StructuredTensor::seed_cotangent should preserve a valid layout: {err}"
            ),
        }
    }
}

#[cfg(test)]
mod tests;
