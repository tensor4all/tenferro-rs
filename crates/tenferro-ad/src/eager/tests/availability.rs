use super::*;

#[test]
fn semantic_ad_returns_gradients_without_an_enable_switch() {
    let ctx = EagerRuntime::new().unwrap();
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major([], vec![3.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([], vec![2.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let output = ctx
        .with_eager_session(&mut engine, |s| s.mul(&x, &x))
        .unwrap();
    let vjp = ctx
        .vjp_optional(&mut engine, &output, &x, &seed)
        .unwrap()
        .unwrap();
    let jvp = ctx
        .jvp_optional(&mut engine, &output, &x, &seed)
        .unwrap()
        .unwrap();
    assert_eq!(vjp.value().unwrap().as_slice::<f64>().unwrap(), &[12.0]);
    assert_eq!(jvp.value().unwrap().as_slice::<f64>().unwrap(), &[12.0]);
    let gradients = output.backward(&mut engine).unwrap();
    assert_eq!(gradients.len(), 1);
    assert_eq!(
        gradients.grad(&x.key).unwrap().as_slice::<f64>().unwrap(),
        &[6.0]
    );

    let zero = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([], vec![0.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let gradients = output.backward_with(&mut engine, &zero).unwrap();
    assert_eq!(gradients.len(), 1);
    assert_eq!(
        gradients.grad(&x.key).unwrap().as_slice::<f64>().unwrap(),
        &[0.0]
    );
}

#[test]
fn semantic_ad_ignores_retired_switch_in_fresh_processes() {
    for setting in [None, Some("0")] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            "eager::tests::availability::semantic_ad_returns_gradients_without_an_enable_switch",
            "--nocapture",
        ]);
        if let Some(value) = setting {
            command.env("TENFERRO_EAGER_SEMANTIC_VJP", value);
        } else {
            command.env_remove("TENFERRO_EAGER_SEMANTIC_VJP");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "setting {setting:?}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn backward_rejects_empty_gradients_but_storage_allows_empty_groups() {
    let ctx = EagerRuntime::new().unwrap();
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let output = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([], vec![3.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    assert!(matches!(
        output.backward(&mut engine),
        Err(Error::RuntimeState { .. })
    ));
    assert!(matches!(
        output.backward_with(&mut engine, &output),
        Err(Error::RuntimeState { .. })
    ));
    assert!(matches!(
        super::super::Gradients::from_tensors(HashMap::new()),
        Err(Error::RuntimeState { .. })
    ));
    let (_, bindings) = AllocationGroup::from_tensors(Vec::new()).unwrap();
    assert!(bindings.is_empty());
}

#[test]
fn backward_rejects_absent_derivative_before_accumulating_any_target() {
    let ctx = EagerRuntime::new().unwrap();
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let leaf = |value| {
        EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major([], vec![value]).unwrap(),
            Arc::clone(&ctx),
        )
        .unwrap()
    };
    let x = leaf(3.0_f64);
    let y = leaf(5.0_f64);
    let output = ctx
        .with_eager_session(&mut engine, |s| {
            let sign = s.sign(&x)?;
            s.add(&sign, &y)
        })
        .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([], vec![1.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    assert!(ctx
        .vjp_optional(&mut engine, &output, &x, &seed)
        .unwrap()
        .is_none());
    assert!(ctx
        .jvp_optional(&mut engine, &output, &x, &seed)
        .unwrap()
        .is_none());
    assert!(ctx.vjp(&mut engine, &output, &x, &seed).is_err());
    assert!(ctx.jvp(&mut engine, &output, &x, &seed).is_err());
    for error in [
        output.backward(&mut engine).unwrap_err(),
        output.backward_with(&mut engine, &seed).unwrap_err(),
    ] {
        assert!(matches!(error, Error::Validation { op: "backward", .. }));
        assert!(error.to_string().contains("EagerRuntime::vjp_optional"));
    }
    assert!(x.grad().unwrap().is_none());
    assert!(y.grad().unwrap().is_none());
}
