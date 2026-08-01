use tenferro_tensor::TypedTensor;

fn main() {
    let tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let first = tensor.as_view();
    let second = tensor.as_view();
    assert_eq!(first.get(&[1]), Some(&2));
    assert_eq!(second.get(&[0]), Some(&1));
}
