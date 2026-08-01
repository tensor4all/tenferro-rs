use tenferro_tensor::TypedTensor;

fn main() {
    let mut tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    {
        let mut first = tensor.as_view_mut();
        *first.get_mut(&[0]).unwrap() = 10;
    }
    {
        let mut second = tensor.as_view_mut();
        *second.get_mut(&[1]).unwrap() = 20;
    }
    assert_eq!(tensor.host_data().unwrap(), &[10, 20]);
}
