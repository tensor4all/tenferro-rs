use tenferro_df64_proof::Df64;
use tenferro_tensor::{Rank, TypedTensor, TypedTensorView};

#[test]
fn downstream_df64_is_a_generic_fixed_and_dynamic_rank_host_owner() {
    let data = (1..=6)
        .map(|n| Df64::from_f64(f64::from(n)))
        .collect::<Vec<_>>();
    let pointer = data.as_ptr();
    let mut fixed = TypedTensor::<Df64, Rank<2>>::from_vec_col_major([2, 3], data).unwrap();
    assert_eq!(fixed.host_data().unwrap().as_ptr(), pointer);
    assert_eq!(fixed.get(&[1, 0]).unwrap().hi, 2.0);
    *fixed.get_mut(&[0, 2]).unwrap() = Df64::from_f64(7.0);
    assert_eq!(fixed.as_view().get(&[0, 2]).unwrap().hi, 7.0);
    assert_eq!(fixed.duplicate().unwrap().get(&[0, 2]).unwrap().hi, 7.0);

    let dynamic =
        TypedTensor::<Df64>::from_vec_col_major(vec![2, 3], vec![Df64::from_f64(1.0); 6]).unwrap();
    assert_eq!(dynamic.get(&[1, 2]).unwrap(), &Df64::from_f64(1.0));
    assert!(dynamic.get(&[2, 0]).is_err());
    assert!(dynamic.get(&[0]).is_err());
}

#[test]
fn downstream_df64_explicit_row_import_and_strided_host_compaction() {
    let row_major = (1..=6)
        .map(|n| Df64::from_f64(f64::from(n)))
        .collect::<Vec<_>>();
    let imported = TypedTensor::<Df64, Rank<2>>::from_vec_row_major([2, 3], row_major).unwrap();
    assert_eq!(imported.get(&[1, 0]).unwrap().hi, 4.0);
    assert_eq!(imported.get(&[0, 2]).unwrap().hi, 3.0);
    assert_eq!(
        imported
            .host_data()
            .unwrap()
            .iter()
            .map(|value| value.hi)
            .collect::<Vec<_>>(),
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let transposed = imported.as_view().transpose_view([1, 0]).unwrap();
    let compact = transposed.to_col_major().unwrap();
    assert_eq!(compact.shape(), &[3, 2]);
    assert_eq!(
        compact
            .host_data()
            .unwrap()
            .iter()
            .map(|value| value.hi)
            .collect::<Vec<_>>(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let offset = TypedTensorView::<_, Rank<1>>::from_slice_ranked(
        [2],
        [-3],
        4,
        imported.host_data().unwrap(),
    )
    .unwrap();
    let copied = offset.to_col_major().unwrap();
    assert_eq!(
        copied
            .host_data()
            .unwrap()
            .iter()
            .map(|value| value.hi)
            .collect::<Vec<_>>(),
        [3.0, 4.0]
    );

    assert!(
        TypedTensor::<Df64, Rank<2>>::from_vec_row_major([2, 3], vec![Df64::zero(); 5]).is_err()
    );
    assert!(
        TypedTensor::<Df64, Rank<2>>::from_vec_row_major(vec![6], vec![Df64::zero(); 6]).is_err()
    );
    assert!(TypedTensor::<Df64>::from_vec_row_major([0, 2], Vec::new())
        .unwrap()
        .host_data()
        .unwrap()
        .is_empty());
    assert_eq!(
        TypedTensor::<Df64>::from_vec_row_major([], vec![Df64::from_f64(9.0)])
            .unwrap()
            .get(&[])
            .unwrap()
            .hi,
        9.0
    );
}
