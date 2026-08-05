use tenferro_tensor::{Rank, TypedTensor, TypedTensorView};

fn requires_rank_three(_: TypedTensorView<'_, f64, Rank<3>>) {}

fn main() {
    let tensor =
        TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
    requires_rank_three(tensor.as_view());
}
