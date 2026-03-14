use tenferro_dyadtensor::{
    plan_axis_classes_for_subscripts, AdTensor, DynAdTensor, DynStructuredPrimal,
    OperandAxisClasses, StructuredTensor,
};
use tenferro_einsum::Subscripts;
use tenferro_tensor::{MemoryOrder, Tensor};

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn tensor2(values: &[f64], d0: usize, d1: usize) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[d0, d1], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn root_structured_tensor_supports_dense_and_diag_layouts() {
    let dense = StructuredTensor::from_dense(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    assert_eq!(dense.logical_dims(), &[2, 2]);
    assert_eq!(dense.axis_classes(), &[0, 1]);
    assert!(dense.is_dense());

    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    assert_eq!(diag.logical_dims(), &[2, 2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
    assert!(diag.is_diag());
}

#[test]
fn ad_tensor_wraps_structured_payload_and_reports_logical_dims() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let x = AdTensor::new_primal(diag);
    assert_eq!(x.dims(), &[2, 2]);
    assert!(x.structured_primal().is_diag());
    assert_eq!(x.primal().dims(), &[2]);
}

#[test]
fn dyn_ad_tensor_carries_diag_payload_without_dense_materialization() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let x: DynAdTensor = AdTensor::new_primal(diag).into();
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(x.axis_classes(), &[0, 0]);
    assert!(x.is_diag());
    assert_eq!(x.as_f64().unwrap().primal().dims(), &[2]);
}

#[test]
fn primal_snapshot_preserves_dense_and_structured_payloads() {
    let dense: DynAdTensor = AdTensor::new_primal(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2)).into();
    let diag: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    )
    .into();

    match dense.primal_snapshot().unwrap() {
        DynStructuredPrimal::F64(snapshot) => {
            assert!(snapshot.is_dense());
            assert_eq!(snapshot.logical_dims(), &[2, 2]);
        }
        other => panic!("expected F64 dense snapshot, got {:?}", other.scalar_type()),
    }

    match diag.primal_snapshot().unwrap() {
        DynStructuredPrimal::F64(snapshot) => {
            assert!(snapshot.is_diag());
            assert_eq!(snapshot.axis_classes(), &[0, 0]);
            assert_eq!(snapshot.payload().dims(), &[2]);
        }
        other => panic!(
            "expected F64 structured snapshot, got {:?}",
            other.scalar_type()
        ),
    }
}

#[test]
fn primal_snapshot_preserves_general_axis_classes() {
    let structured = StructuredTensor::new(
        vec![2, 2, 2],
        vec![0, 1, 1],
        tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2),
    )
    .unwrap();
    let x: DynAdTensor = AdTensor::new_primal(structured).into();

    match x.primal_snapshot().unwrap() {
        DynStructuredPrimal::F64(snapshot) => {
            assert_eq!(snapshot.logical_dims(), &[2, 2, 2]);
            assert_eq!(snapshot.axis_classes(), &[0, 1, 1]);
            assert_eq!(snapshot.payload().dims(), &[2, 2]);
        }
        other => panic!(
            "expected F64 structured snapshot, got {:?}",
            other.scalar_type()
        ),
    }
}

#[test]
fn root_metadata_planning_api_is_exposed_from_crate_root() {
    let operands = vec![
        OperandAxisClasses::new(vec![3, 3], vec![0, 0]).unwrap(),
        OperandAxisClasses::new(vec![3, 3], vec![0, 0]).unwrap(),
    ];
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let plan = plan_axis_classes_for_subscripts(&operands, &subs).unwrap();
    assert_eq!(plan.output_axis_classes, vec![0, 0]);
    assert_eq!(plan.output_dims, vec![3, 3]);
}
