use std::collections::HashSet;

use tenferro_core_ops::{all_primitive_descriptors, OpCategory};

use crate::ElementwiseFusionOp;

#[test]
fn tensor_view_public_surface_uses_canonical_names() {
    let types_rs = include_str!("../types.rs");
    let strided_view_rs = include_str!("../types/strided_view.rs");

    assert!(
        types_rs.contains("pub enum TensorView<'a>"),
        "dtype-erased read-only views should use TensorView"
    );
    assert!(
        types_rs.contains("pub enum TensorRead<'a>"),
        "dtype-erased tensor inputs should use TensorRead"
    );
    assert!(
        types_rs.contains("pub fn transpose_view"),
        "metadata-only axis permutations should use transpose_view"
    );

    for obsolete in [
        "pub enum StridedTensorView",
        "pub enum StridedTensorViewMut",
        "pub use strided_view::{StridedSliceSpec, StridedTensorView",
        "pub fn try_permute_axes",
        "pub fn new(shape:",
        "TypedTensorView::new",
    ] {
        assert!(
            !types_rs.contains(obsolete),
            "tenferro-tensor public surface should not expose obsolete `{obsolete}`"
        );
        assert!(
            !strided_view_rs.contains(obsolete),
            "tenferro-tensor strided view module should not expose obsolete `{obsolete}`"
        );
    }
}

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
