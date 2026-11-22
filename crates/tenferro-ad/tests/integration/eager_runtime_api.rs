use std::sync::Arc;

use tenferro_ad::EagerRuntime;
use tenferro_runtime::Tensor;

#[test]
fn eager_runtime_constructors_return_typed_results() {
    let _: fn() -> tenferro_ad::Result<Arc<EagerRuntime>> = EagerRuntime::new;
    let _: fn(&tenferro_ad::AdContext) -> tenferro_ad::Result<Arc<EagerRuntime>> =
        EagerRuntime::with_ad_context;
}

#[test]
fn eager_runtime_replaces_eager_context_public_name() {
    let runtime = EagerRuntime::new().unwrap();
    let x = runtime
        .variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    let loss = runtime
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
            let squared = s.mul(&x, &x)?;
            s.reduce_sum(&squared, Some(&[0]))
        })
        .unwrap();

    loss.runtime()
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
            s.backward(&loss)
        })
        .unwrap();

    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 4.0]
    );
    runtime.clear_grads().unwrap();
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn eager_runtime_grad_accumulation_keeps_slot_locked_through_update() {
    let source = include_str!("../../src/eager.rs");

    assert!(
        !source.contains("let mut staged = Vec::new();"),
        "gradient accumulation must not stage slot updates after releasing the per-slot lock"
    );
}

#[test]
fn eager_filled_constants_hold_the_value_and_take_no_gradient() {
    use tenferro_ad::DType;

    let runtime = EagerRuntime::new().unwrap();
    let mut engine = tenferro_cpu::CpuEngine::serial();
    // The filled values are broadcast views of one element; `duplicate_value` compacts them.
    let (zeros, ones, scalar) = runtime
        .with_eager_session(&mut engine, |s| {
            let zeros = s.zeros(DType::Bool, [2])?;
            let ones = s.ones(DType::C64, [2, 2])?;
            let scalar = s.full(DType::I32, [0_usize; 0], -3.0)?;
            tenferro_ad::Result::Ok((
                s.duplicate_value(&zeros)?,
                s.duplicate_value(&ones)?,
                s.duplicate_value(&scalar)?,
            ))
        })
        .unwrap();
    assert_eq!(zeros.as_slice::<bool>().unwrap(), &[false, false]);
    assert_eq!(
        ones.as_slice::<tenferro_tensor::Complex64>().unwrap(),
        &[tenferro_tensor::Complex64::new(1.0, 0.0); 4]
    );
    assert_eq!(scalar.as_slice::<i32>().unwrap(), &[-3]);

    // `sum(x * c)` with a filled `c` differentiates to `c` and leaves `c` untracked.
    let x = runtime
        .variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap())
        .unwrap();
    let c = runtime
        .with_eager_session(&mut engine, |s| s.full(DType::F64, [3], 0.25))
        .unwrap();
    // One uploaded element broadcast on the backend, not a full-size host buffer.
    assert!(c.value().unwrap().as_slice::<f64>().is_err());
    runtime
        .with_eager_session(&mut engine, |s| {
            let product = s.mul(&x, &c)?;
            let loss = s.reduce_sum(&product, Some(&[0]))?;
            s.backward(&loss)
        })
        .unwrap();
    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[0.25; 3]
    );
    assert!(c.grad().unwrap().is_none());

    let error = runtime.with_eager_session(&mut engine, |s| s.full(DType::Bool, [2], 0.5));
    assert!(error.is_err());
}

#[test]
fn eager_full_shares_the_concrete_scalar_conversion_and_checks_the_shape() {
    use tenferro_ad::DType;

    fn compact(
        runtime: &Arc<EagerRuntime>,
        engine: &mut tenferro_cpu::CpuEngine,
        shape: &[usize],
        value: f64,
    ) -> tenferro_ad::Result<Tensor> {
        runtime.with_eager_session(engine, |s| {
            let filled = s.full(DType::I64, shape, value)?;
            s.duplicate_value(&filled)
        })
    }

    let runtime = EagerRuntime::new().unwrap();
    let mut engine = tenferro_cpu::CpuEngine::serial();

    // The same exact-integer conversion as `Tensor::full`: 2^53 is inside i64.
    assert_eq!(
        compact(&runtime, &mut engine, &[1], 2.0_f64.powi(53))
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[1_i64 << 53]
    );
    assert!(compact(&runtime, &mut engine, &[1], 2.0_f64.powi(63)).is_err());

    // An overflowing element count is a validation fault, not a wrapped backend
    // integer overflow, and it is rejected before the scalar is uploaded.
    let error = runtime
        .with_eager_session(&mut engine, |s| s.full(DType::F64, [usize::MAX, 2], 1.0))
        .unwrap_err();
    match error {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::Validation {
            source:
                tenferro_tensor::ValidationError::InvalidArgument {
                    argument: "shape", ..
                },
            ..
        }) => {}
        other => panic!("expected a typed shape InvalidArgument, got {other:?}"),
    }

    // Scalar and empty shapes stay valid.
    assert_eq!(
        compact(&runtime, &mut engine, &[0_usize; 0], 7.0)
            .unwrap()
            .shape(),
        &[0_usize; 0]
    );
    assert_eq!(
        compact(&runtime, &mut engine, &[2, 0], 7.0)
            .unwrap()
            .shape(),
        &[2, 0]
    );
}
