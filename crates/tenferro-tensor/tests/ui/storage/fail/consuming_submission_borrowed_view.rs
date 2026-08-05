use tenferro_tensor::TypedTensor;

fn submit(owner: TypedTensor<f64>) {
    drop(owner);
}

fn main() {
    let owner = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let view = owner.as_view();
    submit(owner);
    let _ = view.shape();
}
