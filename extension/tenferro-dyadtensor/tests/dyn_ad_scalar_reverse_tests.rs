use tenferro_dyadtensor::{AdValue, DynAdScalar, NodeId, TapeId};

#[test]
fn dyn_ad_scalar_try_mul_creates_new_reverse_output_node() {
    let lhs = DynAdScalar::from(AdValue::reverse(2.0_f64, NodeId(1), TapeId(11), None));
    let rhs = DynAdScalar::from(AdValue::reverse(3.0_f64, NodeId(2), TapeId(11), None));

    let out = lhs.try_mul(rhs).unwrap();
    assert_ne!(out.node_id(), Some(NodeId(1)));
    assert_ne!(out.node_id(), Some(NodeId(2)));
    assert_eq!(out.tape_id(), Some(TapeId(11)));
}

#[test]
fn dyn_ad_scalar_try_mul_rejects_mixed_reverse_tapes() {
    let lhs = DynAdScalar::from(AdValue::reverse(2.0_f64, NodeId(1), TapeId(7), None));
    let rhs = DynAdScalar::from(AdValue::reverse(3.0_f64, NodeId(2), TapeId(8), None));
    assert!(lhs.try_mul(rhs).is_err());
}
