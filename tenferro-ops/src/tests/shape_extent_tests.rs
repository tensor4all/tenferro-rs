use crate::shape_extent::{ShapeExtent, ShapeMeta};
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
fn shape_meta_reports_rank_and_exact_shape() {
    let meta = ShapeMeta::exact(vec![SymDim::from(2usize), SymDim::from(3usize)]);
    assert_eq!(meta.rank(), 2);
    assert_eq!(
        meta.exact_shape(),
        Some(vec![SymDim::from(2usize), SymDim::from(3usize)])
    );
}

#[test]
fn shape_meta_exact_shape_rejects_upper_bound() {
    let meta = ShapeMeta::new(vec![
        ShapeExtent::exact(SymDim::from(2usize)),
        ShapeExtent::upper_bound(SymDim::from(3usize)),
    ]);
    assert_eq!(meta.rank(), 2);
    assert_eq!(meta.exact_shape(), None);
}
