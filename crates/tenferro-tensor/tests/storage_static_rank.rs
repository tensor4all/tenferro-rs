use tenferro_tensor::{Rank, TypedTensor};

#[test]
fn owner_and_reborrows_preserve_static_rank() {
    let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
    let view = tensor.as_view();
    assert_eq!(view.rank(), 2);
    assert_eq!(view.shape(), &[2, 2]);

    let duplicate = tensor.duplicate().unwrap();
    assert_eq!(duplicate.rank(), 2);
    assert_eq!(duplicate.shape(), &[2, 2]);
}

