#![cfg(feature = "autodiff")]

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::EagerSessionEinsumExt;
use tenferro_einsum::{EinsumAxis, EinsumNotation, EinsumSubscripts, TensorDotAxes};
use tenferro_runtime::{Error as RuntimeError, ErrorPhase, Tensor};
use tenferro_tensor::ValidationError;

// Test helpers: each call opens one eager session on the inputs' runtime and
// runs the borrowed-session contraction inside it.
fn einsum(inputs: &[&EagerTensor], subscripts: &str) -> tenferro_einsum::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.einsum(inputs, subscripts))
}

#[allow(dead_code)]
fn einsum_notation(
    inputs: &[&EagerTensor],
    notation: &EinsumNotation,
) -> tenferro_einsum::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.einsum_notation(inputs, notation)
        })
}

#[allow(dead_code)]
fn einsum_subscripts(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> tenferro_einsum::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.einsum_subscripts(inputs, subscripts)
        })
}

#[allow(dead_code)]
fn tensordot(
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> tenferro_einsum::Result<EagerTensor> {
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
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1, 3, 2], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
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
        f64_data(&string_result.to_tensor(&mut CpuEngine::serial()).unwrap()),
        &[3.0; 8]
    );
    assert_eq!(
        f64_data(
            &programmatic_result
                .to_tensor(&mut CpuEngine::serial())
                .unwrap()
        ),
        &[3.0; 8]
    );
}

#[test]
fn eager_tensor_einsum_matmul_primal_matches_expected_values() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(
        f64_data(&c.to_tensor(&mut CpuEngine::serial()).unwrap()),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_einsum_repeated_output_occurrences_keep_axis_order() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], (1..=6).map(f64::from).collect()).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let out = einsum(&[&a], "ij->iji").unwrap();
    let materialized = out.to_tensor(&mut CpuEngine::serial()).unwrap();

    assert_eq!(materialized.shape(), &[2, 3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                let expected = if i == k {
                    *a.to_tensor(&mut CpuEngine::serial())
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
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3, 4], (1..=24).map(f64::from).collect::<Vec<_>>())
            .unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 4, 2],
            (1..=24).map(|value| f64::from(value) * 0.5).collect(),
        )
        .unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let out = tensordot(&lhs, &rhs, TensorDotAxes::Count(2)).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(
        f64_data(&out.to_tensor(&mut CpuEngine::serial()).unwrap()),
        &[611.0, 650.0, 1475.0, 1586.0]
    );
}

#[test]
fn eager_tensor_tensordot_explicit_axes_accept_negative_indices() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
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
        f64_data(&out.to_tensor(&mut CpuEngine::serial()).unwrap()),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_tensordot_rejects_shape_mismatch() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]).unwrap(),
        ctx.clone(),
    )
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
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap(),
        ctx.clone(),
    )
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
        tenferro_einsum::Error::Runtime(RuntimeError::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
            ..
        })
    ));
}

#[test]
fn eager_tensor_einsum_integer_subscripts_match_string_path() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);

    let c = einsum_subscripts(&[&a, &b], &subscripts).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(
        f64_data(&c.to_tensor(&mut CpuEngine::serial()).unwrap()),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn eager_tensor_einsum_ellipsis_backward_matches_expected_values() {
    let ctx = test_ctx();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3, 2], vec![1.0_f64; 12]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = einsum(&[&a, &b], "...ij,...jk->...ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1, 2]))
        })
        .unwrap();
    let _ = loss.backward(&mut CpuEngine::serial()).unwrap();

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
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _cotangents = loss.backward(&mut CpuEngine::serial()).unwrap();

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
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss.backward(&mut CpuEngine::serial()).unwrap();
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
    let _ = loss.backward(&mut CpuEngine::serial()).unwrap();
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
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss.backward(&mut CpuEngine::serial()).unwrap();

    ctx.clear_grads().unwrap();

    assert!(a.grad().unwrap().is_none());
    assert!(b.grad().unwrap().is_none());

    let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&c, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss.backward(&mut CpuEngine::serial()).unwrap();

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
        EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).unwrap(),
            ctx.clone(),
        )
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
        f64_data(&nary.to_tensor(&mut CpuEngine::serial()).unwrap()),
        &[12.0; 4]
    );
    let loss = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.reduce_sum(&nary, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss.backward(&mut CpuEngine::serial()).unwrap();
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
