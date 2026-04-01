use chainrules_core::Differentiable;
use num_complex::{Complex32, Complex64};
use tenferro_device::{ComputeDevice, LogicalMemorySpace};
use tenferro_einsum::Subscripts;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use crate::{
    accumulate_tangent, first_duplicate_pair, plan_axis_classes_for_subscripts, reverse_subscripts,
    unique_ids_first_appearance, usize_vec_to_u32, AbsAsF64, AxisClassPlanError,
    OperandAxisClasses,
};
use crate::{DynTensor, ScalarType, StructuredTensor};

fn dense_structured<T: tenferro_algebra::Scalar>(tensor: DenseTensor<T>) -> StructuredTensor<T> {
    StructuredTensor::from(tensor)
}

#[test]
fn abs_as_f64_matches_real_and_complex_norms() {
    assert_eq!(1.25f32.abs_as_f64(), 1.25);
    assert_eq!((-2.5f64).abs_as_f64(), 2.5);
    assert_eq!(Complex32::new(-3.0, 4.0).abs_as_f64(), 5.0);
    assert_eq!(Complex64::new(5.0, 12.0).abs_as_f64(), 13.0);
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

#[test]
fn dyn_tensor_accumulate_tangent_preserves_dtype_and_layout() {
    let lhs: DynTensor = dense_structured(
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    )
    .into();
    let rhs: DynTensor = dense_structured(
        DenseTensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    )
    .into();
    let expected_dims = lhs.dims().to_vec();
    let expected_axis_classes = lhs.axis_classes().to_vec();

    let summed = DynTensor::accumulate_tangent(lhs, &rhs);

    assert_eq!(summed.scalar_type(), ScalarType::F64);
    assert_eq!(summed.dims(), expected_dims.as_slice());
    assert_eq!(summed.axis_classes(), expected_axis_classes.as_slice());
    assert_eq!(
        summed.payload_f64().unwrap().buffer().as_slice().unwrap(),
        &[4.0, 6.0]
    );
}

#[test]
fn structured_tensor_helpers_cover_layout_conjugation_and_placement() {
    let payload = DenseTensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let structured = StructuredTensor::from(
        tenferro_tensor::StructuredTensor::from_diagonal_vector(payload, 2).unwrap(),
    );

    assert_eq!(structured.memory_space(), LogicalMemorySpace::MainMemory);
    assert_eq!(structured.preferred_compute_device(), None);

    let moved = structured
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    moved.wait();
    assert!(moved.is_ready());
    assert_eq!(moved.logical_dims(), structured.logical_dims());
    assert_eq!(moved.axis_classes(), structured.axis_classes());

    let mut preferred = structured.clone();
    preferred.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));
    assert_eq!(
        preferred.preferred_compute_device(),
        Some(ComputeDevice::Cpu { device_id: 0 })
    );
    preferred.set_preferred_compute_device(None);
    assert_eq!(preferred.preferred_compute_device(), None);

    let relaid = structured
        .with_payload_like(
            DenseTensor::<Complex64>::from_slice(
                &[Complex64::new(7.0, 0.0), Complex64::new(8.0, 1.0)],
                &[2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(relaid.logical_dims(), structured.logical_dims());
    assert_eq!(relaid.axis_classes(), structured.axis_classes());

    let permuted = structured.permute_logical(&[1, 0]).unwrap();
    assert_eq!(permuted.logical_dims(), &[2, 2]);
    assert!(permuted.is_diag());

    let conjugated = structured.conj();
    assert_eq!(conjugated.logical_dims(), structured.logical_dims());
    assert_eq!(conjugated.axis_classes(), structured.axis_classes());
    assert_eq!(conjugated.is_diag(), structured.is_diag());

    let dense = structured.to_dense().unwrap();
    assert_eq!(dense.dims(), &[2, 2]);
    assert_eq!(dense.get(&[0, 0]), Some(&Complex64::new(1.0, 2.0)));
    assert_eq!(dense.get(&[1, 1]), Some(&Complex64::new(3.0, -4.0)));
    assert_eq!(dense.get(&[0, 1]), Some(&Complex64::new(0.0, 0.0)));
}

#[test]
fn structured_einsum_helpers_cover_reversal_and_duplicate_detection() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let reversed = reverse_subscripts(&subs, 0);
    assert_eq!(reversed.inputs, vec![vec![0, 2], vec![1, 2]]);
    assert_eq!(reversed.output, vec![0, 1]);

    assert_eq!(unique_ids_first_appearance(&[3, 1, 3, 2, 1]), vec![3, 1, 2]);
    assert_eq!(first_duplicate_pair(&[5, 2, 7, 2]), Some((1, 3)));
    assert_eq!(first_duplicate_pair(&[5, 2, 7, 8]), None);

    assert_eq!(usize_vec_to_u32(&[0, 17]).unwrap(), vec![0, 17]);
    assert!(usize_vec_to_u32(&[u32::MAX as usize + 1]).is_err());

    let lhs = dense_structured(
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );
    let rhs = dense_structured(
        DenseTensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );
    let summed = accumulate_tangent(lhs, &rhs).unwrap();
    assert_eq!(
        summed.to_dense().unwrap().buffer().as_slice().unwrap(),
        &[4.0, 6.0]
    );
}

#[test]
fn plan_axis_classes_for_subscripts_groups_and_validates_dimensions() {
    let operands = vec![OperandAxisClasses::new(vec![2, 2], vec![0, 1]).unwrap()];
    let subs = Subscripts::new(&[&[0, 0]], &[0]);

    let plan = plan_axis_classes_for_subscripts(&operands, &subs).unwrap();

    assert_eq!(plan.output_dims, vec![2]);
    assert_eq!(plan.output_axis_classes, vec![0]);
    assert_eq!(plan.output_compressed_roots, vec![0]);
    assert_eq!(
        plan.operand_plans[0].duplicate_class_groups,
        vec![vec![0, 1]]
    );
    assert_eq!(plan.operand_plans[0].normalized_class_roots, vec![0]);

    let err = plan_axis_classes_for_subscripts(&[], &subs).unwrap_err();
    assert!(matches!(
        err,
        AxisClassPlanError::InvalidOperandCount {
            expected: 1,
            found: 0
        }
    ));

    let mismatched = vec![
        OperandAxisClasses::new(vec![2, 2], vec![0, 1]).unwrap(),
        OperandAxisClasses::new(vec![3, 4], vec![0, 1]).unwrap(),
    ];
    let mismatch_subs = Subscripts::new(&[&[0, 0], &[0, 1]], &[1]);
    let err = plan_axis_classes_for_subscripts(&mismatched, &mismatch_subs).unwrap_err();
    assert!(matches!(
        err,
        AxisClassPlanError::LabelDimensionMismatch {
            label: 0,
            expected: 2,
            actual: 3
        }
    ));
}
