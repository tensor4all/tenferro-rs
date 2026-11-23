//! A session's shared composite constants keep AD identities and ownership intact (#2701).
//!
//! An eager session reuses its rank-0 composite constants so a repeated composite transfers
//! nothing per call. That reuse must not merge two constants' differentiation identities while
//! gradients are recorded, and must not make a `full` value shared.

use tenferro_ad::{DType, EagerRuntime, Error};
use tenferro_cpu::CpuEngine;
use tenferro_runtime::Tensor;

type TestResult = Result<(), Error>;

#[test]
fn composite_constants_keep_their_own_identity_while_gradients_are_recorded() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(Vec::new(), vec![3.0_f64])?)?;
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.composite_constant(DType::F64, 2.0)?;
        let b = s.composite_constant(DType::F64, 2.0)?;
        let loss = s.mul(&x, &b)?;
        assert!(
            s.grad_optional(&loss, &a)?.is_none(),
            "a gradient with respect to one constant must not see another constant's use"
        );
        Ok(())
    })
}

#[test]
fn composite_constants_are_reused_without_gradient_recording() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let _no_grad = ctx.no_grad();
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.composite_constant(DType::F64, 2.0)?;
        let b = s.composite_constant(DType::F64, 2.0)?;
        // The second request returns the first value: it is shared, so not uniquely owned.
        assert!(b.into_value().is_err(), "the reused constant is shared");
        assert_eq!(s.duplicate_value(&a)?.as_slice::<f64>()?, &[2.0]);
        Ok(())
    })
}

#[test]
fn full_values_stay_uniquely_owned() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let _no_grad = ctx.no_grad();
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        // A composite constant of the same value is held by the session first, so a cached
        // `full` would come back shared.
        let _held = s.composite_constant(DType::F64, 2.0)?;
        let value = s.full(DType::F64, Vec::<usize>::new(), 2.0)?;
        let tensor = value
            .into_value()
            .unwrap_or_else(|_| panic!("a full value is uniquely owned inside the session"));
        assert_eq!(tensor.as_slice::<f64>()?, &[2.0]);
        Ok(())
    })
}

#[test]
fn composite_fills_are_constant_leaves() -> TestResult {
    let ctx = EagerRuntime::new()?;
    let x = ctx.variable_from(Tensor::from_vec_col_major(Vec::new(), vec![3.0_f64])?)?;
    let filled = {
        let _no_grad = ctx.no_grad();
        ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
            s.composite_filled(DType::F64, &[], 2.0)
        })?
    };
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        let loss = s.mul(&x, &filled)?;
        let grad = s
            .grad_optional(&loss, &filled)?
            .expect("a fill is a constant leaf");
        assert_eq!(s.duplicate_value(&grad)?.as_slice::<f64>()?, &[3.0]);
        Ok(())
    })
}
