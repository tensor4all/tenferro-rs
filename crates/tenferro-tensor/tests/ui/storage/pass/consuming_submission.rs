use tenferro_tensor::{Rank, TypedTensor};

// The detached runtime boundary consumes the owner. A borrowed view cannot
// cross this function because its lifetime is tied to the owner borrow.
fn submit(owner: TypedTensor<f64, Rank<2>>) {
    drop(owner);
}

fn main() {
    let owner = TypedTensor::<f64, Rank<2>>::from_vec_col_major(
        [2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    submit(owner);
}
