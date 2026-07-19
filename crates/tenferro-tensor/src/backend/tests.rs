use super::*;

#[test]
fn compact_host_accumulation_slice_selects_only_compact_host_views() {
    let mut compact_data = [0.0_f64; 4];
    let mut compact = TypedTensorViewMut::from_slice([2, 2], [1, 2], 0, &mut compact_data).unwrap();
    assert_eq!(
        compact_host_accumulation_slice(&mut compact, 4)
            .unwrap()
            .unwrap()
            .len(),
        4
    );

    let mut strided_data = [0.0_f64; 3];
    let mut strided = TypedTensorViewMut::from_slice([2], [2], 0, &mut strided_data).unwrap();
    assert!(compact_host_accumulation_slice(&mut strided, 2)
        .unwrap()
        .is_none());
}
