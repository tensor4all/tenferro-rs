use tenferro_tensor::TypedTensor;

fn main() {
    let mut tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let first = tensor.as_view_mut();
    let second = tensor.as_view_mut();
    assert_eq!(first.get(&[0]), second.get(&[0]));
}
