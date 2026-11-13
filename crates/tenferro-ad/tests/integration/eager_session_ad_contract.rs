//! #2602: eager AD, value extraction and leaf construction are `EagerSession`
//! methods. These tests pin the contracts the move had to keep: runtime
//! membership of every operand and seed is checked before anything is
//! materialized or a gradient slot is written, inactive inputs, higher-order
//! derivatives, gradient accumulation, seeded backward, and the cross-runtime
//! copy that replaced `detach_into`.

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor, Error, Tensor};
use tenferro_cpu::CpuEngine;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn scalar(value: f64) -> Tensor {
    Tensor::from_vec_col_major(vec![], vec![value]).expect("scalar")
}

fn vector(values: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).expect("vector")
}

fn values(t: &EagerTensor) -> Vec<f64> {
    t.value()
        .expect("value")
        .as_slice::<f64>()
        .expect("f64")
        .to_vec()
}

fn grad_of(x: &EagerTensor) -> Option<Vec<f64>> {
    x.grad()
        .expect("gradient slot")
        .map(|g| g.as_slice::<f64>().expect("f64").to_vec())
}

fn is_context_mismatch<T: std::fmt::Debug>(result: Result<T, Error>) -> bool {
    matches!(result, Err(Error::ContextMismatch { .. }))
}

/// `sum(x * x)` recorded in `ctx`.
fn square_sum(ctx: &Arc<EagerRuntime>, x: &EagerTensor) -> EagerTensor {
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        let y = s.mul(x, x)?;
        s.reduce_sum(&y, None)
    })
    .expect("forward")
}

#[test]
fn every_entry_point_rejects_a_foreign_operand_before_writing_gradients() -> TestResult {
    let ctx_a = EagerRuntime::new()?;
    let ctx_b = EagerRuntime::new()?;
    let x_a = ctx_a.variable_from(vector(&[2.0, 3.0]))?;
    let x_b = ctx_b.variable_from(vector(&[5.0, 7.0]))?;
    let loss_a = square_sum(&ctx_a, &x_a);
    let seed_b = ctx_b.constant_from(scalar(1.0))?;
    let tangent_b = ctx_b.constant_from(vector(&[1.0, 1.0]))?;
    let mut engine = CpuEngine::serial();

    // A session of runtime B rejects runtime A's output, wrt and value.
    ctx_b.with_eager_session(&mut engine, |s| {
        assert!(is_context_mismatch(s.backward(&loss_a)));
        assert!(is_context_mismatch(s.backward_with(&loss_a, &seed_b)));
        assert!(is_context_mismatch(s.grad(&loss_a, &x_a)));
        assert!(is_context_mismatch(s.grad_optional(&loss_a, &x_b)));
        assert!(is_context_mismatch(s.vjp(&loss_a, &x_a, &seed_b)));
        assert!(is_context_mismatch(s.vjp_optional(&loss_a, &x_a, &seed_b)));
        assert!(is_context_mismatch(s.jvp(&loss_a, &x_a, &tangent_b)));
        assert!(is_context_mismatch(
            s.jvp_optional(&loss_a, &x_a, &tangent_b)
        ));
        assert!(is_context_mismatch(s.detach(&x_a)));
        assert!(is_context_mismatch(s.duplicate_value(&x_a)));
        Ok::<_, Error>(())
    })?;

    // A session of runtime A rejects a foreign wrt, seed or tangent.
    ctx_a.with_eager_session(&mut engine, |s| {
        assert!(is_context_mismatch(s.grad(&loss_a, &x_b)));
        assert!(is_context_mismatch(s.backward_with(&loss_a, &seed_b)));
        assert!(is_context_mismatch(s.vjp(&loss_a, &x_a, &seed_b)));
        assert!(is_context_mismatch(s.jvp(&loss_a, &x_a, &tangent_b)));
        Ok::<_, Error>(())
    })?;

    // Nothing was accumulated in either runtime.
    assert_eq!(grad_of(&x_a), None);
    assert_eq!(grad_of(&x_b), None);
    Ok(())
}

#[test]
fn runtime_membership_is_checked_before_the_scalar_seed_is_built() -> TestResult {
    // The removed `EagerRuntime::grad_optional` built and materialized the
    // all-ones seed before checking runtimes; the session checks first, so a
    // foreign non-scalar output reports the runtime, not its shape.
    let ctx_a = EagerRuntime::new()?;
    let ctx_b = EagerRuntime::new()?;
    let x_a = ctx_a.variable_from(vector(&[1.0, 2.0]))?;
    let non_scalar_a = ctx_a.with_eager_session(&mut CpuEngine::serial(), |s| s.mul(&x_a, &x_a))?;
    ctx_b.with_eager_session(&mut CpuEngine::serial(), |s| {
        assert!(is_context_mismatch(s.grad_optional(&non_scalar_a, &x_a)));
        assert!(is_context_mismatch(s.backward(&non_scalar_a)));
        Ok::<_, Error>(())
    })?;
    // In its own runtime the same output is a non-scalar error.
    let error = ctx_a
        .with_eager_session(&mut CpuEngine::serial(), |s| s.backward(&non_scalar_a))
        .expect_err("non-scalar backward");
    assert!(matches!(error, Error::NonScalarGrad { .. }), "{error:?}");
    Ok(())
}

#[test]
fn inactive_inputs_are_none_for_optional_forms_and_errors_otherwise() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let x = ctx.variable_from(scalar(3.0))?;
    let y = ctx.variable_from(scalar(4.0))?;
    let c = ctx.constant_from(scalar(5.0))?;
    let seed = ctx.constant_from(scalar(1.0))?;
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        let loss = s.mul(&y, &y)?;
        assert!(s.grad_optional(&loss, &x)?.is_none());
        assert!(s.grad_optional(&loss, &c)?.is_none());
        assert!(s.vjp_optional(&loss, &x, &seed)?.is_none());
        assert!(s.jvp_optional(&loss, &x, &seed)?.is_none());
        for error in [
            s.grad(&loss, &x).expect_err("inactive grad"),
            s.vjp(&loss, &x, &seed).expect_err("inactive vjp"),
            s.jvp(&loss, &x, &seed).expect_err("inactive jvp"),
        ] {
            assert!(error.to_string().contains("wrt"), "{error}");
        }
        // The active input is unaffected.
        assert_eq!(values(&s.grad(&loss, &y)?), [8.0]);
        Ok::<_, Error>(())
    })?;
    assert_eq!(grad_of(&x), None, "functional transforms touch no slot");
    assert_eq!(grad_of(&y), None, "functional transforms touch no slot");
    Ok(())
}

#[test]
fn functional_gradients_compose_to_higher_order() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let x = ctx.variable_from(scalar(3.0))?;
    let tangent = ctx.constant_from(scalar(1.0))?;
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        let x2 = s.mul(&x, &x)?;
        let y = s.mul(&x2, &x)?; // x^3
        let dy = s.grad(&y, &x)?; // 3x^2 = 27
        assert_eq!(values(&dy), [27.0]);
        let d2y = s.grad(&dy, &x)?; // 6x = 18
        assert_eq!(values(&d2y), [18.0]);
        let hvp = s.jvp(&dy, &x, &tangent)?; // forward-over-reverse: 6x = 18
        assert_eq!(values(&hvp), [18.0]);
        Ok::<_, Error>(())
    })?;
    Ok(())
}

#[test]
fn backward_accumulates_until_cleared() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let x = ctx.variable_from(vector(&[1.0, 2.0]))?;
    let mut engine = CpuEngine::serial();
    for _ in 0..2 {
        let loss = square_sum(&ctx, &x);
        let gradients = ctx.with_eager_session(&mut engine, |s| s.backward(&loss))?;
        assert_eq!(gradients.len(), 1);
    }
    assert_eq!(grad_of(&x), Some(vec![4.0, 8.0]));
    x.clear_grad()?;
    assert_eq!(grad_of(&x), None);
    // Forward and backward in one session.
    ctx.with_eager_session(&mut engine, |s| {
        let y = s.mul(&x, &x)?;
        let loss = s.reduce_sum(&y, None)?;
        s.backward(&loss)
    })?;
    assert_eq!(grad_of(&x), Some(vec![2.0, 4.0]));
    Ok(())
}

#[test]
fn seeded_backward_scales_by_the_seed_and_rejects_a_bad_seed_without_writing() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let x = ctx.variable_from(vector(&[2.0, 3.0]))?;
    let seed = ctx.constant_from(vector(&[1.0, 2.0]))?;
    let wrong_shape = ctx.constant_from(vector(&[1.0, 2.0, 3.0]))?;
    let mut engine = CpuEngine::serial();
    let y = ctx.with_eager_session(&mut engine, |s| s.mul(&x, &x))?;
    ctx.with_eager_session(&mut engine, |s| s.backward_with(&y, &seed))?;
    assert_eq!(grad_of(&x), Some(vec![4.0, 12.0]));
    let error = ctx
        .with_eager_session(&mut engine, |s| s.backward_with(&y, &wrong_shape))
        .expect_err("seed shape mismatch");
    assert!(!matches!(error, Error::ContextMismatch { .. }), "{error:?}");
    assert_eq!(
        grad_of(&x),
        Some(vec![4.0, 12.0]),
        "a rejected seed writes nothing"
    );
    Ok(())
}

#[test]
fn value_extraction_borrows_copies_or_consumes() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let x = ctx.variable_from(vector(&[1.0, -2.0]))?;
    let mut engine = CpuEngine::serial();
    let (copy, detached) = ctx.with_eager_session(&mut engine, |s| {
        Ok::<_, Error>((s.duplicate_value(&x)?, s.detach(&x)?))
    })?;
    // Borrowing: `value()` reads in place; copying: an independent tensor.
    assert_eq!(copy.as_slice::<f64>()?, values(&x).as_slice());
    assert_ne!(
        copy.as_slice::<f64>()?.as_ptr(),
        x.value()?.as_slice::<f64>()?.as_ptr()
    );
    // Detach: same runtime and value, no gradient flow.
    assert_eq!(detached.ctx_id(), x.ctx_id());
    assert!(!detached.tracks_grad());
    assert_eq!(values(&detached), values(&x));
    // Consuming: a unique untracked handle hands its allocation back.
    let unique = ctx.constant_from(vector(&[4.0]))?;
    let owned = unique.into_value().expect("unique handle");
    assert_eq!(owned.as_slice::<f64>()?, &[4.0]);
    Ok(())
}

#[test]
fn cross_runtime_copy_replaces_detach_into() -> TestResult {
    let ctx_a = EagerRuntime::new()?;
    let ctx_b = EagerRuntime::new()?;
    let x = ctx_a.variable_from(vector(&[1.0, 2.0]))?;
    let mut engine = CpuEngine::serial();
    let copy = ctx_a.with_eager_session(&mut engine, |s| s.duplicate_value(&x))?;
    let imported = ctx_b.with_eager_session(&mut engine, |s| s.constant_from(copy))?;
    assert_eq!(imported.ctx_id(), ctx_b.id());
    assert!(!imported.tracks_grad());
    assert_eq!(values(&imported), [1.0, 2.0]);
    // The imported leaf takes part in runtime B; the source stays in A.
    let z = ctx_b.with_eager_session(&mut engine, |s| s.add(&imported, &imported))?;
    assert_eq!(values(&z), [2.0, 4.0]);
    let loss = square_sum(&ctx_a, &x);
    ctx_a.with_eager_session(&mut engine, |s| s.backward(&loss))?;
    assert_eq!(grad_of(&x), Some(vec![2.0, 4.0]));
    Ok(())
}

#[test]
fn engine_free_and_session_leaf_constructors_agree_on_host_tensors() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let mut engine = CpuEngine::serial();
    let engine_free_constant = ctx.constant_from(vector(&[1.0, 2.0]))?;
    let engine_free_variable = ctx.variable_from(vector(&[1.0, 2.0]))?;
    let (constant, uploaded, variable) = ctx.with_eager_session(&mut engine, |s| {
        Ok::<_, Error>((
            s.constant_from(vector(&[1.0, 2.0]))?,
            s.constant_from_host(vector(&[1.0, 2.0]))?,
            s.variable_from(vector(&[1.0, 2.0]))?,
        ))
    })?;
    for leaf in [&engine_free_constant, &constant, &uploaded] {
        assert_eq!(values(leaf), [1.0, 2.0]);
        assert!(!leaf.tracks_grad());
    }
    for leaf in [&engine_free_variable, &variable] {
        assert_eq!(values(leaf), [1.0, 2.0]);
        assert!(leaf.tracks_grad());
    }
    Ok(())
}

#[test]
fn the_session_lends_one_engine_for_concrete_work() -> TestResult {
    use tenferro_tensor::TensorRead;

    let ctx = EagerRuntime::new()?;
    let host = vector(&[1.0, -2.0]);
    let copy = ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        s.engine()
            .to_contiguous_read(TensorRead::from_tensor(&host))
            .map_err(Error::from)
    })?;
    assert_eq!(copy.as_slice::<f64>()?, &[1.0, -2.0]);
    Ok(())
}
