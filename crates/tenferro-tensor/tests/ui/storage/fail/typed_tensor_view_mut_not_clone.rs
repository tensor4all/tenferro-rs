use tenferro_tensor::TypedTensorViewMut;

fn require_clone<T: Clone>(_: T) {}

fn main() {
    let mut data = [1_i32, 2];
    let view = TypedTensorViewMut::from_col_major(&[2], &mut data).unwrap();
    require_clone(view);
}
