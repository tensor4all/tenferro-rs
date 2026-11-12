use tenferro_tensor::{StorageBuffer, TypedTensor};

fn main() {
    let tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let mut replacement = StorageBuffer::Host(vec![3_i32, 4]);
    // Check the borrow before calling std so diagnostics do not depend on rust-src.
    let buffer: &mut StorageBuffer<i32> = tensor.buffer();
    std::mem::swap(buffer, &mut replacement);
}
