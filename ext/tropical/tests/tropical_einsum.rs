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
fn unsupported_cases_return_invalid_config() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);

    let unsupported = [
        ("three inputs", vec![&a, &b, &a], "ij,jk,kl->il"),
        ("diagonal", vec![&a, &b], "ii,jk->ik"),
        ("pre-reduction", vec![&a, &b], "ij,jk->k"),
        ("batch modes", vec![&a, &b], "ij,ij->ij"),
        ("input permutation", vec![&a, &b], "ji,jk->ik"),
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
}
