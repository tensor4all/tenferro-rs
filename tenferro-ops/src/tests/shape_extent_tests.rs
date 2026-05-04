use crate::shape_extent::ShapeExtent;
use crate::SymDim;

#[test]
fn exact_extent_exposes_exact_dim() {
    let extent = ShapeExtent::exact(SymDim::from(4usize));
    assert_eq!(extent.as_exact(), Some(&SymDim::from(4usize)));
    assert!(extent.is_exact());
}

#[test]
fn upper_bound_is_not_exact() {
    let extent = ShapeExtent::upper_bound(SymDim::from(4usize));
    assert_eq!(extent.as_exact(), None);
    assert_eq!(extent.bound_expr(), Some(&SymDim::from(4usize)));
    assert!(!extent.is_exact());
}

#[test]
fn map_preserves_extent_kind() {
    let extent = ShapeExtent::upper_bound(SymDim::from(4usize)).map(|dim| dim + SymDim::from(1));
    assert!(matches!(extent, ShapeExtent::UpperBound(_)));
    assert_eq!(
        extent.bound_expr().and_then(SymDim::constant_value),
        Some(5)
    );
}
