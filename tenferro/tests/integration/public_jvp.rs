use tenferro::{jvp, set_default_runtime, Error, JvpResult, RuntimeContext, Tensor};
use tenferro_prims::CpuContext;

fn approx_eq(lhs: &[f64], rhs: &[f64]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-12, "lhs={lhs}, rhs={rhs}");
    }
}

fn with_cpu_runtime() -> tenferro::DefaultRuntimeGuard {
    set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)))
}

#[test]
fn unary_jvp_over_exp_sum_reports_one_output_and_one_tangent() {
    let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2]).unwrap();
    let tangent = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();

    let result: JvpResult = jvp(
        |inputs| Ok(vec![inputs[0].exp().unwrap().sum().unwrap()]),
        &[x],
        &[Some(tangent)],
    )
    .unwrap();

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.output_tangents.len(), 1);

    approx_eq(
        &result.outputs[0].try_to_vec::<f64>().unwrap(),
        &[1.0 + std::f64::consts::E],
    );
    let tangent = result.output_tangents[0].as_ref().unwrap();
    approx_eq(
        &tangent.try_to_vec::<f64>().unwrap(),
        &[1.0 + 2.0 * std::f64::consts::E],
    );
}

#[test]
fn binary_jvp_over_add_sum_reports_both_primals_and_tangent() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    let y = Tensor::from_slice(&[3.0_f64, 4.0], &[2]).unwrap();
    let dx = Tensor::from_slice(&[10.0_f64, 20.0], &[2]).unwrap();
    let dy = Tensor::from_slice(&[100.0_f64, 200.0], &[2]).unwrap();

    let result: JvpResult = jvp(
        |inputs| Ok(vec![inputs[0].add(&inputs[1]).unwrap().sum().unwrap()]),
        &[x, y],
        &[Some(dx), Some(dy)],
    )
    .unwrap();

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.output_tangents.len(), 1);

    approx_eq(&result.outputs[0].try_to_vec::<f64>().unwrap(), &[10.0]);
    let tangent = result.output_tangents[0].as_ref().unwrap();
    approx_eq(&tangent.try_to_vec::<f64>().unwrap(), &[330.0]);
}

#[test]
fn none_tangent_input_produces_none_output_tangent() {
    let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2]).unwrap();

    let result: JvpResult = jvp(
        |inputs| Ok(vec![inputs[0].exp().unwrap().sum().unwrap()]),
        &[x],
        &[None],
    )
    .unwrap();

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.output_tangents.len(), 1);
    assert!(result.output_tangents[0].is_none());
    approx_eq(
        &result.outputs[0].try_to_vec::<f64>().unwrap(),
        &[1.0 + std::f64::consts::E],
    );
}

#[test]
fn multi_output_jvp_over_qr_reports_two_outputs_and_optional_tangents() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

    let result: JvpResult = jvp(
        |inputs| {
            let qr = inputs[0].qr().unwrap();
            Ok(vec![qr.q, qr.r])
        },
        &[a],
        &[None],
    )
    .unwrap();

    assert_eq!(result.outputs.len(), 2);
    assert_eq!(result.output_tangents.len(), 2);
    assert!(result.output_tangents[0].is_none());
    assert!(result.output_tangents[1].is_none());
    assert_eq!(result.outputs[0].dims(), &[2, 2]);
    assert_eq!(result.outputs[1].dims(), &[2, 2]);
    assert!(result.outputs[0].is_dense());
    assert!(result.outputs[1].is_dense());
}

#[test]
fn qr_without_installed_runtime_reports_runtime_missing() {
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

    let err = jvp(
        |inputs| inputs[0].qr().map(|qr| vec![qr.q, qr.r]),
        &[a],
        &[None],
    )
    .unwrap_err();

    assert!(matches!(err, Error::RuntimeNotConfigured));
}
