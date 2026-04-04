use tenferro_internal_ad_surface::{jvp, set_default_runtime, Error, RuntimeContext, Tensor};
use tenferro_prims::CpuContext;

fn approx_eq(lhs: &[f64], rhs: &[f64]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-12, "lhs={lhs}, rhs={rhs}");
    }
}

fn with_cpu_runtime() -> tenferro_internal_ad_surface::DefaultRuntimeGuard {
    set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)))
}

#[test]
fn jvp_propagates_add_exp_sum_tangents() {
    let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2]).unwrap();
    let tangent = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();

    let result = jvp(
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
    approx_eq(
        &result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[1.0 + 2.0 * std::f64::consts::E],
    );
}

#[test]
fn jvp_reports_runtime_missing_naturally_for_qr() {
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

    let err = jvp(
        |inputs| inputs[0].qr().map(|qr| vec![qr.q, qr.r]),
        &[a],
        &[None],
    )
    .unwrap_err();

    assert!(matches!(err, Error::RuntimeNotConfigured));
}

#[test]
fn jvp_handles_qr_with_runtime() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

    let result = jvp(
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
}

#[test]
fn jvp_try_to_vec_materializes_logical_dense_views() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 2.0], &[2, 2]).unwrap();
    let da = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap();

    let result = jvp(
        |inputs| inputs[0].matrix_exp().map(|out| vec![out]),
        &[a],
        &[Some(da)],
    )
    .unwrap();

    let tangent = result.output_tangents[0].as_ref().unwrap();
    approx_eq(
        &tangent.try_to_vec::<f64>().unwrap(),
        &[std::f64::consts::E, 0.0, 0.0, 0.0],
    );
}
