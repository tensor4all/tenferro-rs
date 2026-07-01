use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
use tenferro_tensor::{Error, Tensor};

#[test]
fn maxplus_matmul_uses_shared_einsum_lowering() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 0.0, 1.0, 5.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 10.0, 0.0, 1.0]).unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[11.0, 15.0, 10.0, 6.0]
    );
    assert_eq!(result.argmax[0].indices(), &[0, 1, 0, 1]);
}

#[test]
fn output_permutation_matches_subscripts() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 0.0, 1.0, 5.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 10.0, 0.0, 1.0]).unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ki").unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[11.0, 10.0, 15.0, 6.0]
    );
    assert_eq!(result.argmax[0].indices(), &[0, 0, 1, 1]);
}

#[test]
fn rectangular_output_permutation_matches_subscripts() {
    let a =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 10.0, 0.0, 2.0, 6.0]).unwrap();
    let b = Tensor::from_vec_col_major(
        vec![3, 4],
        vec![
            0.0, 1.0, 2.0, 5.0, 0.0, 1.0, -10.0, 20.0, 0.0, 3.0, 3.0, 3.0,
        ],
    )
    .unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ki").unwrap();

    assert_eq!(result.output.shape(), &[4, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[11.0, 10.0, 30.0, 13.0, 8.0, 9.0, 20.0, 9.0]
    );
    assert_eq!(result.argmax[0].indices(), &[1, 1, 1, 1, 2, 0, 1, 2]);
}

#[test]
fn minplus_matmul_uses_shared_einsum_lowering() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 4.0, 3.0, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f32, 6.0, 7.0, 1.0]).unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MinPlus, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(
        result.output.as_slice::<f32>().unwrap(),
        &[6.0, 8.0, 4.0, 3.0]
    );
    assert_eq!(result.argmax[0].indices(), &[0, 1, 1, 1]);
}

#[test]
fn ties_keep_first_winner_through_einsum() {
    let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]).unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(result.output.as_slice::<f64>().unwrap(), &[3.0]);
    assert_eq!(result.argmax[0].indices(), &[0]);
}

#[test]
fn batched_maxplus_matmul_supports_natural_batch_first_subscripts() {
    let a = Tensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0_f64, 2.0, 4.0, 0.0, 5.0, 1.0, 0.0, 7.0],
    )
    .unwrap();
    let b = Tensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![0.0_f64, 3.0, 2.0, 5.0, 10.0, 4.0, 1.0, 0.0],
    )
    .unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "bij,bjk->bik").unwrap();

    assert_eq!(result.output.shape(), &[2, 2, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[7.0, 6.0, 4.0, 12.0, 11.0, 6.0, 14.0, 7.0]
    );
    assert_eq!(result.argmax[0].indices(), &[1, 1, 0, 1, 0, 0, 0, 1]);
    assert_eq!(result.argmax[0].contracted_subscripts(), &[b'j' as u32]);
    assert_eq!(result.argmax[0].contracted_shape(), &[2]);
}

#[test]
fn target_order_batched_maxplus_matmul_applies_requested_output_permutation() {
    let a = Tensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0_f64, 4.0, 5.0, 0.0, 2.0, 0.0, 1.0, 7.0],
    )
    .unwrap();
    let b = Tensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![0.0_f64, 2.0, 10.0, 1.0, 3.0, 5.0, 4.0, 0.0],
    )
    .unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ijb,jkb->bik").unwrap();

    assert_eq!(result.output.shape(), &[2, 2, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[7.0, 6.0, 4.0, 12.0, 11.0, 6.0, 14.0, 7.0]
    );
    assert_eq!(result.argmax[0].indices(), &[1, 1, 0, 1, 0, 0, 0, 1]);
}

#[test]
fn fallback_handles_input_permutation_and_records_argmax() {
    let transposed_left =
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 5.0, 4.0, 0.0]).unwrap();
    let right = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 10.0, 1.0]).unwrap();

    let result = tropical_einsum_with_argmax(
        TropicalKind::MaxPlus,
        &[&transposed_left, &right],
        "ji,jk->ik",
    )
    .unwrap();

    assert_eq!(result.output.shape(), &[2, 2]);
    assert_eq!(
        result.output.as_slice::<f64>().unwrap(),
        &[7.0, 4.0, 11.0, 14.0]
    );
    assert_eq!(result.argmax[0].indices(), &[1, 0, 0, 0]);
}

#[test]
fn fallback_matches_reference_for_multi_contracted_permuted_labels() {
    let lhs = Tensor::from_vec_col_major(
        vec![2, 3, 2],
        vec![
            0.0_f64, 4.0, 1.0, 3.0, 2.0, 5.0, 6.0, 1.0, 7.0, 0.0, 8.0, 2.0,
        ],
    )
    .unwrap();
    let rhs = Tensor::from_vec_col_major(
        vec![2, 3, 2],
        vec![
            1.0_f64, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 1.0, 0.0, 2.0, 4.0, 6.0,
        ],
    )
    .unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&lhs, &rhs], "kji,ljk->il").unwrap();
    let expected =
        reference_tropical_einsum_f64(TropicalKind::MaxPlus, &lhs, b"kji", &rhs, b"ljk", b"il");

    assert_eq!(result.output.shape(), expected.shape);
    assert_eq!(result.output.as_slice::<f64>().unwrap(), expected.values);
    assert_eq!(result.argmax[0].indices(), expected.argmax);
    assert_eq!(
        result.argmax[0].contracted_subscripts(),
        &[b'k' as u32, b'j' as u32]
    );
    assert_eq!(result.argmax[0].contracted_shape(), &[2, 3]);
}

#[test]
fn fallback_matches_reference_for_minplus_output_permutation() {
    let lhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![4.0_f64, 1.0, 5.0, 0.0, 3.0, 2.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![2.0_f64, 4.0, 1.0, 5.0, 0.0, 3.0]).unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MinPlus, &[&lhs, &rhs], "ji,jk->ki").unwrap();
    let expected =
        reference_tropical_einsum_f64(TropicalKind::MinPlus, &lhs, b"ji", &rhs, b"jk", b"ki");

    assert_eq!(result.output.shape(), expected.shape);
    assert_eq!(result.output.as_slice::<f64>().unwrap(), expected.values);
    assert_eq!(result.argmax[0].indices(), expected.argmax);
}

#[test]
fn fallback_matches_reference_for_nan_and_all_nan_cells() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 2], vec![f64::NAN, 1.0, f64::NAN, f64::NAN]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 0.0, f64::NAN, 2.0]).unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&lhs, &rhs], "ji,jk->ik").unwrap();
    let expected =
        reference_tropical_einsum_f64(TropicalKind::MaxPlus, &lhs, b"ji", &rhs, b"jk", b"ik");

    assert_eq!(result.output.shape(), expected.shape);
    assert_eq!(result.output.as_slice::<f64>().unwrap(), expected.values);
    assert_eq!(result.argmax[0].indices(), expected.argmax);
}

#[test]
fn fallback_ties_keep_first_contracted_winner() {
    let transposed_left = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f32, 1.0]).unwrap();
    let right = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f32, 2.0]).unwrap();

    let result = tropical_einsum_with_argmax(
        TropicalKind::MaxPlus,
        &[&transposed_left, &right],
        "ji,jk->ik",
    )
    .unwrap();

    assert_eq!(result.output.as_slice::<f32>().unwrap(), &[3.0]);
    assert_eq!(result.argmax[0].indices(), &[0]);
}

#[test]
fn multi_contracted_modes_expose_fused_winner_coordinates() {
    let a = Tensor::from_vec_col_major(vec![1, 2, 2], vec![0.0_f64, 5.0, 3.0, 1.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2, 1], vec![0.0_f64, 0.0, 0.0, 0.0]).unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ijk,jkl->il").unwrap();
    let step = &result.argmax[0];

    assert_eq!(result.output.as_slice::<f64>().unwrap(), &[5.0]);
    assert_eq!(step.indices(), &[1]);
    assert_eq!(step.contracted_subscripts(), &[b'j' as u32, b'k' as u32]);
    assert_eq!(step.contracted_shape(), &[2, 2]);
    assert_eq!(step.winner_coordinates(0).unwrap(), vec![1, 0]);
}

#[test]
fn empty_outputs_allow_zero_sized_contracted_modes() {
    let a = Tensor::from_vec_col_major(vec![0, 0], Vec::<f64>::new()).unwrap();
    let b = Tensor::from_vec_col_major(vec![0, 4], Vec::<f64>::new()).unwrap();

    let result =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(result.output.shape(), &[0, 4]);
    assert!(result.output.as_slice::<f64>().unwrap().is_empty());
    assert!(result.argmax[0].indices().is_empty());
    assert_eq!(result.argmax[0].contracted_shape(), &[0]);
}

#[test]
fn unsupported_cases_return_invalid_config() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    let unsupported = [
        ("three inputs", vec![&a, &b, &a], "ij,jk,kl->il"),
        ("diagonal", vec![&a, &b], "ii,jk->ik"),
        ("pre-reduction", vec![&a, &b], "ij,jk->k"),
        ("outer product", vec![&a, &b], "ij,kl->ikjl"),
        ("repeated output", vec![&a, &b], "ij,jk->ii"),
    ];

    for (case, inputs, notation) in unsupported {
        let err =
            tropical_einsum_with_argmax(TropicalKind::MaxPlus, &inputs, notation).unwrap_err();
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

    let int_tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let err = tropical_einsum_with_argmax(
        TropicalKind::MaxPlus,
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

    let zero_lhs = Tensor::from_vec_col_major(vec![2, 0], Vec::<f64>::new()).unwrap();
    let zero_rhs = Tensor::from_vec_col_major(vec![0, 4], Vec::<f64>::new()).unwrap();
    let err =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&zero_lhs, &zero_rhs], "ij,jk->ik")
            .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "tropical_einsum_with_argmax",
            ..
        }
    ));
}

struct ReferenceResult {
    shape: Vec<usize>,
    values: Vec<f64>,
    argmax: Vec<u32>,
}

fn reference_tropical_einsum_f64(
    kind: TropicalKind,
    lhs: &Tensor,
    lhs_labels: &[u8],
    rhs: &Tensor,
    rhs_labels: &[u8],
    output_labels: &[u8],
) -> ReferenceResult {
    let lhs_data = lhs.as_slice::<f64>().unwrap();
    let rhs_data = rhs.as_slice::<f64>().unwrap();
    let contracted_labels: Vec<u8> = lhs_labels
        .iter()
        .copied()
        .filter(|label| rhs_labels.contains(label) && !output_labels.contains(label))
        .collect();
    let shape_for = |labels: &[u8]| {
        labels
            .iter()
            .map(|label| {
                lhs_labels
                    .iter()
                    .position(|candidate| candidate == label)
                    .map(|axis| lhs.shape()[axis])
                    .or_else(|| {
                        rhs_labels
                            .iter()
                            .position(|candidate| candidate == label)
                            .map(|axis| rhs.shape()[axis])
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let output_shape = shape_for(output_labels);
    let contracted_shape = shape_for(&contracted_labels);
    let output_len = product(&output_shape);
    let contracted_len = product(&contracted_shape);
    let lhs_strides = strides(lhs.shape());
    let rhs_strides = strides(rhs.shape());
    let mut values = Vec::with_capacity(output_len);
    let mut argmax = Vec::with_capacity(output_len);
    let mut output_index = vec![0usize; output_shape.len()];
    let mut contracted_index = vec![0usize; contracted_shape.len()];

    for _ in 0..output_len {
        let mut best = match kind {
            TropicalKind::MaxPlus => f64::NEG_INFINITY,
            TropicalKind::MinPlus => f64::INFINITY,
            _ => unreachable!("unknown tropical kind"),
        };
        let mut winner = 0_u32;
        let mut has_ordered_candidate = false;
        contracted_index.fill(0);
        for contracted_flat in 0..contracted_len {
            let lhs_offset = reference_offset(
                lhs_labels,
                output_labels,
                &contracted_labels,
                &lhs_strides,
                &output_index,
                &contracted_index,
            );
            let rhs_offset = reference_offset(
                rhs_labels,
                output_labels,
                &contracted_labels,
                &rhs_strides,
                &output_index,
                &contracted_index,
            );
            let candidate = lhs_data[lhs_offset] + rhs_data[rhs_offset];
            let better = match kind {
                TropicalKind::MaxPlus => !has_ordered_candidate || candidate > best,
                TropicalKind::MinPlus => !has_ordered_candidate || candidate < best,
                _ => unreachable!("unknown tropical kind"),
            };
            if !candidate.is_nan() && better {
                best = candidate;
                winner = contracted_flat as u32;
                has_ordered_candidate = true;
            }
            increment_index(&mut contracted_index, &contracted_shape);
        }
        values.push(best);
        argmax.push(winner);
        increment_index(&mut output_index, &output_shape);
    }

    ReferenceResult {
        shape: output_shape,
        values,
        argmax,
    }
}

fn reference_offset(
    input_labels: &[u8],
    output_labels: &[u8],
    contracted_labels: &[u8],
    strides: &[usize],
    output_index: &[usize],
    contracted_index: &[usize],
) -> usize {
    input_labels
        .iter()
        .zip(strides)
        .map(|(label, stride)| {
            let coordinate = output_labels
                .iter()
                .position(|candidate| candidate == label)
                .map(|axis| output_index[axis])
                .or_else(|| {
                    contracted_labels
                        .iter()
                        .position(|candidate| candidate == label)
                        .map(|axis| contracted_index[axis])
                })
                .unwrap();
            coordinate * stride
        })
        .sum()
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride *= extent;
    }
    strides
}

fn increment_index(index: &mut [usize], shape: &[usize]) {
    for (axis, extent) in index.iter_mut().zip(shape) {
        *axis += 1;
        if *axis < *extent {
            return;
        }
        *axis = 0;
    }
}
