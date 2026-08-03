use tenferro_tensor::TypedTensor;

fn main() {
    let tensor = TypedTensor::<f64>::from_vec_col_major([1], vec![1.0]).unwrap();
    let _ = tensor.clone();
}
