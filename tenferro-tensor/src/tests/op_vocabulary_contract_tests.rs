#[test]
fn elementwise_fusion_op_boilerplate_is_catalog_generated() {
    let source = include_str!("../backend.rs");

    assert!(
        source.contains("define_elementwise_fusion_op"),
        "ElementwiseFusionOp variants should be emitted from the shared primitive catalog"
    );
}
