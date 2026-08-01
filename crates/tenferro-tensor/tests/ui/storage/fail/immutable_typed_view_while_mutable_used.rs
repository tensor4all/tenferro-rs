use tenferro_tensor::TypedTensor;

fn main() {
    let mut tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let mutable = tensor.as_view_mut();
    let immutable = tensor.as_view();
    assert_eq!(mutable.shape(), immutable.shape());
}
