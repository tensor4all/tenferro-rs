use crate::axis::{normalize_axes, normalize_axis};
use crate::broadcast::{broadcast_input_plan, broadcast_shape, broadcast_shapes};
use crate::reduction::reduced_shape;

#[test]
fn broadcast_shape_accepts_scalar_rank_padding_and_singletons() {
    assert_eq!(broadcast_shape(&[], &[3, 4]).unwrap(), vec![3, 4]);
    assert_eq!(broadcast_shape(&[5], &[3, 5]).unwrap(), vec![3, 5]);
    assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
    assert_eq!(
        broadcast_shapes([&[3, 1][..], &[1, 4][..], &[3, 4][..]]).unwrap(),
        vec![3, 4]
    );
}

#[test]
fn broadcast_shape_rejects_incompatible_shapes() {
    let err = broadcast_shape(&[2, 3], &[3, 2]).unwrap_err();
    assert!(err.to_string().contains("broadcast"));
}

#[test]
fn broadcast_input_plan_drops_expanding_singletons() {
    let plan = broadcast_input_plan(&[3, 1], &[3, 4]).unwrap();
    assert_eq!(plan.source_shape, vec![3]);
    assert_eq!(plan.dims, vec![0]);

    let scalar = broadcast_input_plan(&[], &[3, 4]).unwrap();
    assert_eq!(scalar.source_shape, Vec::<usize>::new());
    assert_eq!(scalar.dims, Vec::<usize>::new());

    let vector = broadcast_input_plan(&[5], &[3, 5]).unwrap();
    assert_eq!(vector.source_shape, vec![5]);
    assert_eq!(vector.dims, vec![1]);
}

#[test]
fn normalize_axis_accepts_negative_axes_and_rejects_out_of_bounds() {
    assert_eq!(normalize_axis(0, 3).unwrap(), 0);
    assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
    assert_eq!(normalize_axis(-3, 3).unwrap(), 0);
    assert_eq!(normalize_axis(0, usize::MAX).unwrap(), 0);
    assert_eq!(normalize_axis(-1, usize::MAX).unwrap(), usize::MAX - 1);
    assert!(normalize_axis(3, 3).is_err());
    assert!(normalize_axis(-4, 3).is_err());
    assert!(normalize_axis(isize::MIN, 3).is_err());
}

#[test]
fn normalize_axes_rejects_duplicates_after_normalization() {
    assert_eq!(normalize_axes(&[0, -1], 3).unwrap(), vec![0, 2]);
    let err = normalize_axes(&[1, -2], 3).unwrap_err();
    assert!(err.to_string().contains("duplicate"));
}

#[test]
fn reduced_shape_supports_keepdims() {
    assert_eq!(reduced_shape(&[2, 3, 4], &[1], false).unwrap(), vec![2, 4]);
    assert_eq!(
        reduced_shape(&[2, 3, 4], &[1], true).unwrap(),
        vec![2, 1, 4]
    );
    assert_eq!(
        reduced_shape(&[2, 3], &[0, 1], false).unwrap(),
        Vec::<usize>::new()
    );
}
