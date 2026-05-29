use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
use tenferro_tensor::{Error, Tensor};

#[test]
fn maxplus_matmul_uses_shared_einsum_lowering() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 0.0, 1.0, 5.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 10.0, 0.0, 1.0]);

    let result =
        tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[11.0, 15.0, 10.0, 6.0]
    );
    assert_eq!(result.argmax[0].indices(), &[0, 1, 0, 1]);
}

#[test]
fn output_permutation_matches_subscripts() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 0.0, 1.0, 5.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 10.0, 0.0, 1.0]);

    let result =
        tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ki").unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[11.0, 10.0, 15.0, 6.0]
    );
    assert_eq!(result.argmax[0].indices(), &[0, 0, 1, 1]);
}

#[test]
fn rectangular_output_permutation_matches_subscripts() {
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 10.0, 0.0, 2.0, 6.0]);
    let b = Tensor::from_vec_col_major(
        vec![3, 4],
        vec![
            0.0, 1.0, 2.0, 5.0, 0.0, 1.0, -10.0, 20.0, 0.0, 3.0, 3.0, 3.0,
        ],
    );

    let result =
        tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ki").unwrap();

    assert_eq!(result.output.shape(), &[4, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[11.0, 10.0, 30.0, 13.0, 8.0, 9.0, 20.0, 9.0]
    );
    assert_eq!(result.argmax[0].indices(), &[1, 1, 1, 1, 2, 0, 1, 2]);
}

#[test]
fn minplus_matmul_uses_shared_einsum_lowering() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 4.0, 3.0, 2.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f32, 6.0, 7.0, 1.0]);

    let result =
        tropical_einsum_with_argmax(TropicalEinsumKind::MinPlus, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(
        result.output.as_slice::<f32>().unwrap(),
        &[6.0, 8.0, 4.0, 3.0]
    );
    assert_eq!(result.argmax[0].indices(), &[0, 1, 1, 1]);
}

#[test]
fn ties_keep_first_winner_through_einsum() {
    let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);

    let result =
        tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(result.output.as_slice::<f64>().unwrap(), &[3.0]);
    assert_eq!(result.argmax[0].indices(), &[0]);
}

#[test]
fn multi_contracted_modes_expose_fused_winner_coordinates() {
    let a = Tensor::from_vec_col_major(vec![1, 2, 2], vec![0.0_f64, 5.0, 3.0, 1.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2, 1], vec![0.0_f64, 0.0, 0.0, 0.0]);

    let result =
        tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ijk,jkl->il").unwrap();
    let step = &result.argmax[0];

    assert_eq!(result.output.as_slice::<f64>().unwrap(), &[5.0]);
    assert_eq!(step.indices(), &[1]);
    assert_eq!(step.contracted_subscripts(), &[b'j' as u32, b'k' as u32]);
    assert_eq!(step.contracted_shape(), &[2, 2]);
    assert_eq!(step.winner_coordinates(0).unwrap(), vec![1, 0]);
}

#[test]
fn empty_outputs_allow_zero_sized_contracted_modes() {
    let a = Tensor::from_vec_col_major(vec![0, 0], Vec::<f64>::new());
    let b = Tensor::from_vec_col_major(vec![0, 4], Vec::<f64>::new());

    let result =
        tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(result.output.shape(), &[0, 4]);
    assert!(result.output.as_slice::<f64>().unwrap().is_empty());
    assert!(result.argmax[0].indices().is_empty());
    assert_eq!(result.argmax[0].contracted_shape(), &[0]);
}

#[test]
fn unsupported_cases_return_invalid_config() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);

    let unsupported = [
        ("three inputs", vec![&a, &b, &a], "ij,jk,kl->il"),
        ("diagonal", vec![&a, &b], "ii,jk->ik"),
        ("pre-reduction", vec![&a, &b], "ij,jk->k"),
        ("batch modes", vec![&a, &b], "ij,ij->ij"),
        ("input permutation", vec![&a, &b], "ji,jk->ik"),
        ("outer product", vec![&a, &b], "ij,kl->ikjl"),
    ];

    for (case, inputs, notation) in unsupported {
        let err = tropical_einsum_with_argmax(TropicalEinsumKind::MaxPlus, &inputs, notation)
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::InvalidConfig {
                    op: "tropical_einsum_with_argmax",
                    ..
                }
            ),
            "{case} returned {err:?}"
        );
    }

    let int_tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4]);
    let err = tropical_einsum_with_argmax(
        TropicalEinsumKind::MaxPlus,
        &[&int_tensor, &int_tensor],
        "ij,jk->ik",
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "tropical_einsum_with_argmax",
            ..
        }
    ));

    let zero_lhs = Tensor::from_vec_col_major(vec![2, 0], Vec::<f64>::new());
    let zero_rhs = Tensor::from_vec_col_major(vec![0, 4], Vec::<f64>::new());
    let err = tropical_einsum_with_argmax(
        TropicalEinsumKind::MaxPlus,
        &[&zero_lhs, &zero_rhs],
        "ij,jk->ik",
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "tropical_einsum_with_argmax",
            ..
        }
    ));
}
