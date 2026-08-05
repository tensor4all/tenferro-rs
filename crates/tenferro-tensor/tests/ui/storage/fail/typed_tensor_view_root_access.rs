use tenferro_tensor::TypedTensor;

fn main() {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    let view = tensor.as_view_mut();
    let _ = tensor.shape();
    drop(view);
}
