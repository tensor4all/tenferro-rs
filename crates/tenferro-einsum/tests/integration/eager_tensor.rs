#![cfg(feature = "autodiff")]

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::EagerEinsumOps;
use tenferro_einsum::{EinsumAxis, EinsumNotation, EinsumSubscripts, TensorDotAxes};
use tenferro_runtime::{Error as RuntimeError, ErrorPhase, Tensor};
use tenferro_tensor::ValidationError;

// Test helpers: each call opens one eager session on the inputs' runtime and
// runs the borrowed-session contraction inside it.
fn einsum(inputs: &[&EagerTensor], subscripts: &str) -> tenferro_ad::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.einsum(inputs, subscripts))
}

#[allow(dead_code)]
fn einsum_notation(
    inputs: &[&EagerTensor],
    notation: &EinsumNotation,
) -> tenferro_ad::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.einsum(inputs, notation))
}

#[allow(dead_code)]
fn einsum_subscripts(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> tenferro_ad::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.einsum(inputs, subscripts))
}

#[allow(dead_code)]
fn tensordot(
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> tenferro_ad::Result<EagerTensor> {
    lhs.runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.tensordot(lhs, rhs, axes))
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn test_ctx() -> Arc<EagerRuntime> {
    unsafe {
        std::env::set_var("TENFERRO_PROFILE_EAGER_OP_AGG", "1");
        std::env::set_var("TENFERRO_PROFILE_EAGER_OP_PRINT_EVERY", "1");
    }
    EagerRuntime::new().unwrap()
}

#[test]
fn eager_tensor_einsum_ellipsis_broadcast_matches_programmatic_notation() {
    let ctx = test_ctx();
    let lhs = ctx
        .constant_from(Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12]).unwrap())
        .unwrap();
    let rhs = ctx
        .constant_from(Tensor::from_vec_col_major(vec![1, 3, 2], vec![1.0_f64; 6]).unwrap())
        .unwrap();
    let notation = EinsumNotation::new(
        &[
            &[
                EinsumAxis::Ellipsis,
                EinsumAxis::Label(0),
                EinsumAxis::Label(1),
            ],
            &[
                EinsumAxis::Ellipsis,
                EinsumAxis::Label(1),
                EinsumAxis::Label(2),
            ],
        ],
        &[
            EinsumAxis::Ellipsis,
            EinsumAxis::Label(0),
            EinsumAxis::Label(2),
        ],
    );

    let string_result = einsum(&[&lhs, &rhs], "...ij,...jk->...ik").unwrap();
    let programmatic_result = einsum_notation(&[&lhs, &rhs], &notation).unwrap();

    assert_eq!(string_result.shape(), &[2, 2, 2]);
    assert_eq!(programmatic_result.shape(), &[2, 2, 2]);
    assert_eq!(
        f64_data(
            &string_result
                .runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s
                    .duplicate_value(&string_result))
                .unwrap()
        ),
        &[3.0; 8]
    );
    assert_eq!(
        f64_data(
            &programmatic_result
                .runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s
                    .duplicate_value(&programmatic_result))
                .unwrap()
        ),
        &[3.0; 8]
    );
}

#[test]
fn eager_tensor_einsum_matmul_primal_matches_expected_values() {
    let ctx = test_ctx();
    let a = ctx
        .constant_from(
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();
    let b = ctx
        .constant_from(
            Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(
        f64_data(
            &c.runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&c))
                .unwrap()
        ),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_einsum_repeated_output_occurrences_keep_axis_order() {
    let ctx = test_ctx();
    let a = ctx
        .constant_from(
            Tensor::from_vec_col_major(vec![2, 3], (1..=6).map(f64::from).collect()).unwrap(),
        )
        .unwrap();

    let out = einsum(&[&a], "ij->iji").unwrap();
    let materialized = out
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&out))
        .unwrap();

    assert_eq!(materialized.shape(), &[2, 3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                let expected = if i == k {
                    *a.runtime()
                        .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&a))
                        .unwrap()
                        .get::<f64>(&[i, j])
                        .unwrap()
                } else {
                    0.0
                };
                assert_eq!(*materialized.get::<f64>(&[i, j, k]).unwrap(), expected);
            }
        }
    }
}

#[test]
fn eager_tensor_tensordot_count_contracts_last_lhs_with_first_rhs_axes() {
    let ctx = test_ctx();
    let lhs = ctx
        .constant_from(
            Tensor::from_vec_col_major(vec![2, 3, 4], (1..=24).map(f64::from).collect::<Vec<_>>())
                .unwrap(),
        )
        .unwrap();
    let rhs = ctx
        .constant_from(
            Tensor::from_vec_col_major(
                vec![3, 4, 2],
                (1..=24).map(|value| f64::from(value) * 0.5).collect(),
            )
            .unwrap(),
        )
        .unwrap();

    let out = tensordot(&lhs, &rhs, TensorDotAxes::Count(2)).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(
        f64_data(
            &out.runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&out))
                .unwrap()
        ),
        &[611.0, 650.0, 1475.0, 1586.0]
    );
}

#[test]
fn eager_tensor_tensordot_explicit_axes_accept_negative_indices() {
    let ctx = test_ctx();
    let lhs = ctx
        .constant_from(
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();
    let rhs = ctx
        .constant_from(
            Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();

    let out = tensordot(
        &lhs,
        &rhs,
        TensorDotAxes::Axes {
            lhs: &[-1],
            rhs: &[0],
        },
    )
    .unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(
        f64_data(
            &out.runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&out))
                .unwrap()
        ),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_tensordot_rejects_shape_mismatch() {
    let ctx = test_ctx();
    let lhs = ctx
        .constant_from(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap())
        .unwrap();
    let rhs = ctx
        .constant_from(Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]).unwrap())
        .unwrap();

    let err = match tensordot(&lhs, &rhs, TensorDotAxes::Count(1)) {
        Ok(_) => panic!("expected tensordot shape mismatch"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("contracted dimensions differ"));
}

#[test]
fn eager_tensor_tensordot_rejects_explicit_out_of_bounds_axis() {
    let ctx = test_ctx();
    let lhs = ctx
        .constant_from(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap())
        .unwrap();
    let rhs = ctx
        .constant_from(Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap())
        .unwrap();

    let err = match tensordot(
        &lhs,
        &rhs,
        TensorDotAxes::Axes {
            lhs: &[2],
            rhs: &[0],
        },
    ) {
        Ok(_) => panic!("expected explicit tensordot axis bounds error"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        RuntimeError::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
            ..
        }
    ));
}

#[test]
fn eager_tensor_einsum_integer_subscripts_match_string_path() {
    let ctx = test_ctx();
    let a = ctx
        .constant_from(
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();
    let b = ctx
        .constant_from(
            Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);

    let c = einsum_subscripts(&[&a, &b], &subscripts).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(
        f64_data(
            &c.runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&c))
                .unwrap()
        ),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_einsum_ellipsis_backward_matches_expected_values() {
    let ctx = test_ctx();
    let a = ctx
        .variable_from(Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12]).unwrap())
        .unwrap();
    let b = ctx
        .variable_from(Tensor::from_vec_col_major(vec![2, 3, 2], vec![1.0_f64; 12]).unwrap())
        .unwrap();

    let c = einsum(&[&a, &b], "...ij,...jk->...ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1, 2]))
        })
        .unwrap();
    let _ = loss
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&loss))
        .unwrap();

    assert_eq!(
        f64_data(
            &a.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[2.0; 12]
    );
    assert_eq!(
        f64_data(
            &b.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[2.0; 12]
    );
}

#[test]
fn eager_tensor_einsum_backward_populates_input_grads() {
    let ctx = test_ctx();
    let a = ctx
        .variable_from(
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();
    let b = ctx
        .variable_from(
            Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _cotangents = loss
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&loss))
        .unwrap();

    let grad_a = a.grad().unwrap().unwrap();
    let grad_b = b.grad().unwrap().unwrap();

    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[3, 2]);
    assert_eq!(
        f64_data(&grad_a.to_tensor(&mut CpuEngine::serial()).unwrap()),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(
        f64_data(&grad_b.to_tensor(&mut CpuEngine::serial()).unwrap()),
        &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]
    );
}

#[test]
fn eager_tensor_einsum_repeated_backward_accumulates_across_calls() {
    let ctx = test_ctx();
    let a = ctx
        .variable_from(
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();
    let b = ctx
        .variable_from(
            Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&loss))
        .unwrap();
    assert_eq!(
        f64_data(
            &a.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(
        f64_data(
            &b.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]
    );

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&loss))
        .unwrap();
    assert_eq!(
        f64_data(
            &a.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[10.0, 10.0, 14.0, 14.0, 18.0, 18.0]
    );
    assert_eq!(
        f64_data(
            &b.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[6.0, 14.0, 22.0, 6.0, 14.0, 22.0]
    );
}

#[test]
fn eager_tensor_einsum_context_clear_grads_resets_all_live_leaves() {
    let ctx = EagerRuntime::new().unwrap();
    let a = ctx
        .variable_from(
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();
    let b = ctx
        .variable_from(
            Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&loss))
        .unwrap();

    ctx.clear_grads().unwrap();

    assert!(a.grad().unwrap().is_none());
    assert!(b.grad().unwrap().is_none());

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&loss))
        .unwrap();

    assert_eq!(
        f64_data(
            &a.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(
        f64_data(
            &b.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]
    );
}

/// Binary, N-ary and borrowed-session einsum agree on the calling thread's
/// grad mode: under an outer `no_grad` none of them records, and without it all
/// of them do (the N-ary program now runs in one borrowed session).
#[test]
fn eager_einsum_paths_follow_the_calling_threads_no_grad() {
    let ctx = EagerRuntime::new().unwrap();
    let leaf = |shape: Vec<usize>| {
        let len = shape.iter().product();
        ctx.variable_from(Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).unwrap())
            .unwrap()
    };
    let (a, b, c) = (leaf(vec![2, 3]), leaf(vec![3, 4]), leaf(vec![4, 2]));

    {
        let _guard = ctx.no_grad();
        let binary = einsum(&[&a, &b], "ij,jk->ik").unwrap();
        let nary = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();
        let session = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| s.neg(&a))
            .unwrap();
        assert!(!binary.tracks_grad());
        assert!(!nary.tracks_grad());
        assert!(!session.tracks_grad());
    }

    let nary = einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();
    assert!(nary.tracks_grad());
    assert_eq!(
        f64_data(
            &nary
                .runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&nary))
                .unwrap()
        ),
        &[12.0; 4]
    );
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&nary, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&loss))
        .unwrap();
    // d(sum(a b c))/da[i, j] = sum_{k, l} b[j, k] c[k, l] = 8 for all-ones inputs.
    assert_eq!(
        f64_data(
            &a.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[8.0; 6]
    );
}

/// The reverse derivative of a singleton broadcast: `"ij,pqj->pqi"` with lhs
/// `[2, 1]` and rhs `[4, 2, 3]`, `out[p, q, i] = lhs[i, 0] * sum_j rhs[p, q, j]`.
/// `d sum(out) / d lhs[i, 0] = sum_{p,q,j} rhs[p, q, j]` sums the cotangent
/// over the broadcast `j` and keeps the primal shape `[2, 1]`;
/// `d sum(out) / d rhs[p, q, j] = sum_i lhs[i, 0]` is constant along `j`.
#[test]
fn eager_tensor_einsum_backward_sums_cotangent_over_a_singleton_broadcast_axis() {
    // The broadcast keeps the semantic einsum extension in the eager graph, so
    // its AD rules must be installed.
    let ad = tenferro_ad::AdContext::builder()
        .with_semantic_extension_rules(tenferro_einsum::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    let ctx = EagerRuntime::with_ad_context(&ad).unwrap();
    let a = ctx
        .variable_from(Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    let b = ctx
        .variable_from(
            Tensor::from_vec_col_major(vec![4, 2, 3], (1..=24).map(f64::from).collect()).unwrap(),
        )
        .unwrap();
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[2, 3, 1]], &[2, 3, 0]);

    let c = einsum_subscripts(&[&a, &b], &subscripts).unwrap();
    assert_eq!(c.shape(), &[4, 2, 2]);
    // Column-major rhs[p, q, j] = 1 + p + 4q + 8j, so sum_j rhs = 27 + 3(p + 4q).
    let row_sums: Vec<f64> = (0..8).map(|pq| 27.0 + 3.0 * f64::from(pq)).collect();
    let expected: Vec<f64> = [1.0, 2.0]
        .iter()
        .flat_map(|lhs| row_sums.iter().map(move |sum| lhs * sum))
        .collect();
    assert_eq!(
        f64_data(
            &c.runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&c))
                .unwrap()
        ),
        expected.as_slice()
    );
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1, 2]))
        })
        .unwrap();
    let _ = loss
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&loss))
        .unwrap();

    let grad_a = a
        .grad()
        .unwrap()
        .unwrap()
        .to_tensor(&mut CpuEngine::serial())
        .unwrap();
    assert_eq!(grad_a.shape(), &[2, 1]);
    // sum_{p,q,j} rhs = 1 + 2 + ... + 24 = 300.
    assert_eq!(f64_data(&grad_a), &[300.0, 300.0]);

    // d sum(out) / d rhs[p, q, j] = sum_i lhs[i, 0] = 3 for every element.
    let grad_b = b
        .grad()
        .unwrap()
        .unwrap()
        .to_tensor(&mut CpuEngine::serial())
        .unwrap();
    assert_eq!(grad_b.shape(), &[4, 2, 3]);
    assert_eq!(f64_data(&grad_b), &[3.0; 24]);
}

#[test]
fn eager_tensor_einsum_subscripts_broadcasts_singleton_contracting_extent() {
    let ctx = test_ctx();
    let a = ctx
        .constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64; 2]).unwrap())
        .unwrap();
    let b = ctx
        .constant_from(Tensor::from_vec_col_major(vec![4, 2, 3], vec![1.0_f64; 24]).unwrap())
        .unwrap();
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[2, 3, 1]], &[2, 3, 0]);

    let c = einsum_subscripts(&[&a, &b], &subscripts).unwrap();

    assert_eq!(c.shape(), &[4, 2, 2]);
    assert_eq!(
        f64_data(
            &c.runtime()
                .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&c))
                .unwrap()
        ),
        &[6.0; 16]
    );
}
