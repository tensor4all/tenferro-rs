use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use super::*;
use tenferro_ops::ext_op::ExtensionOp;

#[test]
fn infer_output_meta_uses_output_labels_and_promotes_dtype() {
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
    let lhs_shape = [SymDim::from(2usize), SymDim::from(3usize)];
    let rhs_shape = [SymDim::from(3usize), SymDim::from(4usize)];

    let meta = op.infer_output_meta(
        &[DType::F32, DType::F64],
        &[lhs_shape.as_slice(), rhs_shape.as_slice()],
    );

    assert_eq!(meta[0].0, DType::F64);
    assert_eq!(meta[0].1, vec![SymDim::from(2usize), SymDim::from(4usize)]);
}

#[test]
fn payload_identity_ignores_static_tree_execution_hint() {
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let raw_subscripts = crate::Subscripts::from(&subscripts);
    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..]];
    let left_first =
        Arc::new(ContractionTree::from_pairs(&raw_subscripts, &shapes, &[(0, 1), (3, 2)]).unwrap());
    let right_first =
        Arc::new(ContractionTree::from_pairs(&raw_subscripts, &shapes, &[(1, 2), (0, 3)]).unwrap());

    let without_hint = EinsumExtensionOp::new(subscripts.clone());
    let hinted_left = EinsumExtensionOp::with_static_tree(subscripts.clone(), left_first);
    let hinted_right = EinsumExtensionOp::with_static_tree(subscripts, right_first);

    assert!(without_hint.payload_eq(&hinted_left));
    assert!(hinted_left.payload_eq(&hinted_right));
    assert_eq!(payload_hash(&without_hint), payload_hash(&hinted_left));
    assert_eq!(payload_hash(&hinted_left), payload_hash(&hinted_right));
}

#[test]
fn payload_identity_includes_output_shape_hint() {
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let without_hint = EinsumExtensionOp::new(subscripts.clone());
    let with_hint = EinsumExtensionOp::with_output_shape_hint(
        subscripts,
        vec![SymDim::from(2usize), SymDim::from(4usize)],
    );

    assert!(!without_hint.payload_eq(&with_hint));
    assert_ne!(payload_hash(&without_hint), payload_hash(&with_hint));
}

fn payload_hash(op: &EinsumExtensionOp) -> u64 {
    let mut hasher = DefaultHasher::new();
    op.payload_hash(&mut hasher);
    hasher.finish()
}
