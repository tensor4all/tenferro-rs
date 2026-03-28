use chainrules_core::Differentiable;
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use super::StructuredTensor;

impl<T> Differentiable for StructuredTensor<T>
where
    T: Scalar,
{
    type Tangent = StructuredTensor<T>;

    fn zero_tangent(&self) -> Self::Tangent {
        StructuredTensor(tenferro_tensor::StructuredTensor::from_validated_parts(
            self.logical_dims().to_vec(),
            self.axis_classes().to_vec(),
            self.payload().zero_tangent(),
        ))
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        assert_eq!(
            a.logical_dims(),
            b.logical_dims(),
            "StructuredTensor::accumulate_tangent requires matching logical dims"
        );
        assert_eq!(
            a.axis_classes(),
            b.axis_classes(),
            "StructuredTensor::accumulate_tangent requires matching axis classes"
        );
        let logical_dims = a.logical_dims().to_vec();
        let axis_classes = a.axis_classes().to_vec();
        let payload = Tensor::<T>::accumulate_tangent(a.0.into_payload(), b.payload());
        StructuredTensor(tenferro_tensor::StructuredTensor::from_validated_parts(
            logical_dims,
            axis_classes,
            payload,
        ))
    }

    fn num_elements(&self) -> usize {
        self.logical_dims().iter().product()
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        StructuredTensor(tenferro_tensor::StructuredTensor::from_validated_parts(
            self.logical_dims().to_vec(),
            self.axis_classes().to_vec(),
            self.payload().seed_cotangent(),
        ))
    }
}
