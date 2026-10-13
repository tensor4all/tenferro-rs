//! A compact tensor caches its column-major layout, and every path that builds
//! or reshapes one keeps that cache equal to the layout of its shape.

use tenferro_tensor::{DynRank, Host, Rank, TypedTensor};

fn expected(shape: &[usize]) -> Vec<isize> {
    tenferro_tensor_core::col_major_strides(shape)
        .unwrap()
        .to_vec()
}

#[test]
fn cached_strides_match_the_column_major_layout_for_every_rank() {
    // Rank 0 through 10: the last ranks exceed the inline capacity of the
    // stride vector, so the heap-backed path is covered too.
    for rank in 0..=10usize {
        let shape = vec![2usize; rank];
        let len = 2usize.pow(rank as u32);
        let tensor = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
            shape.clone(),
            vec![0.0; len],
        )
        .unwrap();

        assert_eq!(tensor.strides(), expected(&shape), "rank {rank}");
    }
}

#[test]
fn a_static_rank_owner_caches_the_same_layout() {
    let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 3], vec![1.0; 6]).unwrap();

    assert_eq!(tensor.strides(), expected(&[2, 3]));
}

#[test]
fn representation_conversions_carry_the_layout() {
    let host = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2, 3], vec![1.0; 6])
        .unwrap();
    let dynamic = host.into_dynamic();
    assert_eq!(dynamic.strides(), expected(&[2, 3]));

    let (_, layout, _) = dynamic.into_parts().unwrap();
    assert_eq!(layout.strides(), expected(&[2, 3]));
}

#[test]
fn a_rank_change_recomputes_the_layout() {
    let dynamic = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    let ranked = dynamic.try_into_rank::<2>().unwrap();

    assert_eq!(ranked.shape(), &[2, 3]);
    assert_eq!(ranked.strides(), expected(&[2, 3]));
}
