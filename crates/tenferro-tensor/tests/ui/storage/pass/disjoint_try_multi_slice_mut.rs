use tenferro_tensor::{StridedSliceSpec, TypedTensor};

fn main() {
    let mut tensor = TypedTensor::<i32>::from_vec_col_major(vec![4], vec![1, 2, 3, 4]).unwrap();
    let mut root = tensor.as_view_mut();
    let (mut first, mut second) = root
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(2), 1)],
            &[StridedSliceSpec::new(2, Some(4), 1)],
        )
        .unwrap()
        .unwrap();

    *first.get_mut(&[0]).unwrap() = 10;
    *second.get_mut(&[0]).unwrap() = 40;
    assert_eq!(first.get(&[0]), Some(&10));
    assert_eq!(second.get(&[0]), Some(&40));
}
