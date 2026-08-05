use tenferro_tensor::{StorageBuffer, TypedTensor};

fn main() {
    let tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let mut replacement = StorageBuffer::Host(vec![3_i32, 4]);
    std::mem::swap(tensor.buffer(), &mut replacement);
}
