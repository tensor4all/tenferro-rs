use tenferro_internal_frontend_core::DynTensor;

/// Canonical reverse-mode value handle for tenferro's hard cut to tidu.
pub type DynValue = tidu::Value<DynTensor>;

pub fn new_dyn_value(primal: DynTensor) -> DynValue {
    tidu::Value::new(primal)
}

pub fn new_reverse_leaf(primal: DynTensor) -> DynValue {
    tidu::Value::new(primal).requires_grad_(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_internal_frontend_core::{DynTensor, ScalarType, StructuredTensor};
    use tenferro_tensor::{MemoryOrder, Tensor};

    fn dyn_tensor_from_slice(data: &[f64], dims: &[usize]) -> DynTensor {
        let tensor = Tensor::<f64>::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap();
        StructuredTensor::from(tensor).into()
    }

    #[test]
    fn new_dyn_value_stays_detached() {
        let primal = dyn_tensor_from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let value = new_dyn_value(primal);

        assert!(!value.requires_grad());
        assert_eq!(value.primal().scalar_type(), ScalarType::F64);
        assert_eq!(value.primal().dims(), &[2, 2]);
    }

    #[test]
    fn new_reverse_leaf_enables_grad_tracking() {
        let primal = dyn_tensor_from_slice(&[5.0, 6.0], &[2]);
        let value = new_reverse_leaf(primal);

        assert!(value.requires_grad());
        assert_eq!(value.primal().scalar_type(), ScalarType::F64);
        assert_eq!(value.primal().dims(), &[2]);
    }
}
