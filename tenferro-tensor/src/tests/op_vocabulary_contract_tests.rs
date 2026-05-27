use std::collections::HashSet;

use tenferro_core_ops::{all_primitive_descriptors, OpCategory};

use crate::ElementwiseFusionOp;

#[test]
fn elementwise_fusion_ops_round_trip_through_catalog_kinds() {
    let fusion_kinds: HashSet<_> = ElementwiseFusionOp::iter()
        .map(|op| op.primitive_kind())
        .collect();

    for kind in &fusion_kinds {
        let descriptor = all_primitive_descriptors()
            .iter()
            .find(|descriptor| descriptor.kind == *kind)
            .unwrap_or_else(|| panic!("missing descriptor for fusion op {kind:?}"));
        assert!(
            matches!(
                descriptor.category,
                OpCategory::Elementwise | OpCategory::Analytic
            ),
            "fusion op {kind:?} should be cataloged as elementwise or analytic"
        );
        assert!(
            !descriptor.host_only,
            "fusion op {kind:?} should not be host-only"
        );
        assert_eq!(
            ElementwiseFusionOp::from_primitive_kind(*kind).map(|op| op.primitive_kind()),
            Some(*kind),
            "fusion op {kind:?} should round-trip through PrimitiveOpKind"
        );
    }

    assert_eq!(
        fusion_kinds.len(),
        ElementwiseFusionOp::iter().count(),
        "ElementwiseFusionOp::iter should list each variant exactly once"
    );
}
