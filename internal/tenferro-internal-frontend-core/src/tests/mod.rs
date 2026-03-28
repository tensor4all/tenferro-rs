use chainrules_core::Differentiable;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use crate::{DynTensor, ScalarType, StructuredTensor};

fn dense_structured<T: tenferro_algebra::Scalar>(tensor: DenseTensor<T>) -> StructuredTensor<T> {
    StructuredTensor::from(tensor)
}

#[test]
fn scalar_type_variants_cover_all_supported_dynamic_dtypes() {
    let variants = [
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::C32,
        ScalarType::C64,
    ];
    assert_eq!(variants.len(), 4);
}

#[test]
fn dyn_tensor_scalar_type_roundtrip() {
    let tensor =
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let value: DynTensor = dense_structured(tensor).into();

    assert_eq!(value.scalar_type(), ScalarType::F64);
    assert_eq!(value.dims(), &[2]);
}

#[test]
fn dyn_tensor_differentiable_contract_preserves_layout_metadata() {
    let diag = DenseTensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let value: DynTensor = StructuredTensor::from(
        tenferro_tensor::StructuredTensor::from_diagonal_vector(diag, 2).unwrap(),
    )
    .into();

    let zero = value.zero_tangent();
    let seed = value.seed_cotangent();

    assert_eq!(zero.scalar_type(), ScalarType::F64);
    assert_eq!(seed.scalar_type(), ScalarType::F64);
    assert_eq!(zero.dims(), &[2, 2]);
    assert_eq!(seed.dims(), &[2, 2]);
}
