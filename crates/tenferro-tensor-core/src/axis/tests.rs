use super::*;

#[test]
fn every_in_range_axis_maps_to_one_index() {
    for rank in 0..=4usize {
        let r = rank as isize;
        for axis in -r..r {
            let index = normalize_axis(axis, rank).unwrap();
            assert!(index < rank);
            let expected = if axis < 0 {
                (axis + r) as usize
            } else {
                axis as usize
            };
            assert_eq!(index, expected, "axis {axis} at rank {rank}");
        }
    }
}

#[test]
fn out_of_range_axes_keep_the_signed_value() {
    assert_eq!(
        normalize_axis(3, 3),
        Err(ValidationError::AxisOutOfBounds { axis: 3, rank: 3 })
    );
    assert_eq!(
        normalize_axis(-4, 3),
        Err(ValidationError::AxisOutOfBounds { axis: -4, rank: 3 })
    );
    assert_eq!(
        normalize_axis(0, 0),
        Err(ValidationError::AxisOutOfBounds { axis: 0, rank: 0 })
    );
    assert_eq!(
        normalize_axis(isize::MIN, usize::MAX),
        Ok(usize::MAX - isize::MIN.unsigned_abs())
    );
    assert_eq!(
        normalize_axis(isize::MIN, 3),
        Err(ValidationError::AxisOutOfBounds {
            axis: isize::MIN,
            rank: 3
        })
    );
}

#[test]
fn axis_lists_keep_order_and_reject_aliases() {
    assert_eq!(normalize_axes(&[], 2).unwrap().as_slice(), &[] as &[usize]);
    assert_eq!(normalize_axes(&[-1, 0], 2).unwrap().as_slice(), &[1, 0]);
    assert_eq!(
        normalize_axes(&[0, 0], 2),
        Err(ValidationError::DuplicateAxis {
            axis: 0,
            role: "axes"
        })
    );
    assert_eq!(
        normalize_axes(&[0, 5], 2),
        Err(ValidationError::AxisOutOfBounds { axis: 5, rank: 2 })
    );
}
