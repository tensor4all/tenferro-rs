use tenferro_tensor::TensorViewMut;

fn require_clone<T: Clone>(_: T) {}

fn main() {
    let mut data = [1_i32, 2];
    let view = TensorViewMut::i32(&[2], &mut data).unwrap();
    require_clone(view);
}
