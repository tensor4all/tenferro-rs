mod organization;

use super::*;
use tenferro_einsum::Subscripts;

fn operand(dims: &[usize], classes: &[usize]) -> OperandAxisClasses {
    OperandAxisClasses::new(dims.to_vec(), classes.to_vec()).unwrap()
}

#[test]
fn v1_dense_dense_metadata_baseline() {
    // A[i,j] * B[j,k] -> out[i,k]
    let ops = vec![operand(&[2, 3], &[0, 1]), operand(&[3, 4], &[0, 1])];
    let subs = Subscripts::new(&[&[0, 1][..], &[1, 2][..]], &[0, 2]);
    let plan = plan_axis_classes_for_subscripts(&ops, &subs).unwrap();
    assert_eq!(plan.output_dims, vec![2, 4]);
    assert_eq!(plan.output_axis_classes, vec![0, 1]);
    assert!(plan
        .operand_plans
        .iter()
        .all(|p| p.duplicate_class_groups.is_empty()));
}

#[test]
fn v2_diag_chain_merge_metadata() {
    let ops = vec![operand(&[3, 3], &[0, 0]), operand(&[3, 3], &[0, 0])];
    let subs = Subscripts::new(&[&[0, 1][..], &[1, 2][..]], &[0, 2]);
    let plan = plan_axis_classes_for_subscripts(&ops, &subs).unwrap();
    assert_eq!(plan.output_dims, vec![3, 3]);
    assert_eq!(plan.output_axis_classes, vec![0, 0]);
    assert_eq!(plan.output_compressed_roots.len(), 1);
}

#[test]
fn v3_diag_star_merge_metadata() {
    let ops = vec![
        operand(&[5, 5], &[0, 0]),
        operand(&[5, 5], &[0, 0]),
        operand(&[5, 5], &[0, 0]),
    ];
    let subs = Subscripts::new(&[&[0, 1][..], &[0, 2][..], &[0, 3][..]], &[1, 2, 3]);
    let plan = plan_axis_classes_for_subscripts(&ops, &subs).unwrap();
    assert_eq!(plan.output_dims, vec![5, 5, 5]);
    assert_eq!(plan.output_axis_classes, vec![0, 0, 0]);
    assert_eq!(plan.output_compressed_roots.len(), 1);
}

#[test]
fn v6_duplicate_root_requires_normalization() {
    // A: classes [0,1,2], B ties 0 and 1 through shared labels.
    let ops = vec![operand(&[2, 2, 5], &[0, 1, 2]), operand(&[2, 2], &[0, 0])];
    let subs = Subscripts::new(&[&[0, 1, 2][..], &[0, 1][..]], &[2]);
    let plan = plan_axis_classes_for_subscripts(&ops, &subs).unwrap();
    assert_eq!(plan.operand_plans[0].class_roots.len(), 3);
    assert_eq!(plan.operand_plans[0].normalized_class_roots.len(), 2);
    assert_eq!(
        plan.operand_plans[0].duplicate_class_groups,
        vec![vec![0, 1]]
    );
    assert_eq!(plan.output_dims, vec![5]);
    assert_eq!(plan.output_axis_classes, vec![0]);
}

#[test]
fn v8_full_contraction_scalar_output() {
    let ops = vec![operand(&[2, 3], &[0, 1]), operand(&[2, 3], &[0, 1])];
    let subs = Subscripts::new(&[&[0, 1][..], &[0, 1][..]], &[]);
    let plan = plan_axis_classes_for_subscripts(&ops, &subs).unwrap();
    assert!(plan.output_dims.is_empty());
    assert!(plan.output_axis_classes.is_empty());
    assert!(plan.output_compressed_roots.is_empty());
}

#[test]
fn v10_dimension_mismatch_is_error() {
    let ops = vec![operand(&[2, 2], &[0, 0]), operand(&[3, 3], &[0, 0])];
    let subs = Subscripts::new(&[&[0, 1][..], &[1, 2][..]], &[0, 2]);
    let err = plan_axis_classes_for_subscripts(&ops, &subs).unwrap_err();
    assert!(matches!(
        err,
        AxisClassPlanError::LabelDimensionMismatch { .. }
    ));
}
