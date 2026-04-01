use num_complex::Complex64;
use tenferro::{jvp, set_default_runtime, Error, JvpResult, LuPivot, RuntimeContext, Tensor};
use tenferro_prims::CpuContext;

fn approx_eq(lhs: &[f64], rhs: &[f64]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-12, "lhs={lhs}, rhs={rhs}");
    }
}

fn complex_approx_eq(lhs: &[Complex64], rhs: &[Complex64]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        let diff = (*lhs - *rhs).norm();
        assert!(diff < 1.0e-12, "lhs={lhs:?}, rhs={rhs:?}, abs_diff={diff}");
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

#[test]
fn einsum_jvp_reports_primal_and_tangent() {
    let _runtime = with_cpu_runtime();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    let y = Tensor::from_slice(&[10.0_f64, 100.0], &[2]).unwrap();
    let dx = Tensor::from_slice(&[3.0_f64, 4.0], &[2]).unwrap();
    let dy = Tensor::from_slice(&[5.0_f64, 6.0], &[2]).unwrap();

    let result = jvp(
        |inputs| Tensor::einsum("i,i->", &[&inputs[0], &inputs[1]]).map(|out| vec![out]),
        &[x, y],
        &[Some(dx), Some(dy)],
    )
    .unwrap();

    approx_eq(&result.outputs[0].try_to_vec::<f64>().unwrap(), &[210.0]);
    approx_eq(
        &result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[447.0],
    );
}

#[test]
fn solve_jvp_reports_primal_and_tangent() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&[4.0_f64, 9.0], &[2]).unwrap();
    let db = Tensor::from_slice(&[1.0_f64, 1.0], &[2]).unwrap();

    let result = jvp(
        |inputs| inputs[0].solve(&inputs[1]).map(|out| vec![out]),
        &[a, b],
        &[None, Some(db)],
    )
    .unwrap();

    approx_eq(&result.outputs[0].try_to_vec::<f64>().unwrap(), &[2.0, 3.0]);
    approx_eq(
        &result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.5, 1.0 / 3.0],
    );
}

#[test]
fn det_and_norm_jvp_report_primal_and_tangent() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap();
    let da = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
    let x = Tensor::from_slice(&[3.0_f64, 4.0], &[2]).unwrap();
    let dx = Tensor::from_slice(&[0.0_f64, 1.0], &[2]).unwrap();

    let det_result = jvp(
        |inputs| inputs[0].det().map(|out| vec![out]),
        &[a],
        &[Some(da)],
    )
    .unwrap();
    approx_eq(&det_result.outputs[0].try_to_vec::<f64>().unwrap(), &[6.0]);
    approx_eq(
        &det_result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[3.0],
    );

    let norm_result = jvp(
        |inputs| inputs[0].norm(tenferro::NormKind::Fro).map(|out| vec![out]),
        &[x],
        &[Some(dx)],
    )
    .unwrap();
    approx_eq(&norm_result.outputs[0].try_to_vec::<f64>().unwrap(), &[5.0]);
    approx_eq(
        &norm_result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.8],
    );
}

#[test]
fn multi_output_jvp_over_svd_reports_three_outputs_and_optional_tangents() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[2.0_f64, 1.0, 0.0, 3.0], &[2, 2]).unwrap();

    let result = jvp(
        |inputs| {
            let svd = inputs[0].svd(None).unwrap();
            Ok(vec![svd.u, svd.s, svd.vt])
        },
        &[a],
        &[None],
    )
    .unwrap();

    assert_eq!(result.outputs.len(), 3);
    assert_eq!(result.output_tangents.len(), 3);
    assert!(result.output_tangents.iter().all(Option::is_none));
    assert_eq!(result.outputs[0].dims(), &[2, 2]);
    assert_eq!(result.outputs[1].dims(), &[2]);
    assert_eq!(result.outputs[2].dims(), &[2, 2]);
}

#[test]
fn complex_solve_jvp_is_supported() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap();
    let b =
        Tensor::from_slice(&[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)], &[2]).unwrap();
    let db = Tensor::from_slice(
        &[Complex64::new(0.5, -1.0), Complex64::new(2.0, 0.25)],
        &[2],
    )
    .unwrap();

    let result = jvp(
        |inputs| inputs[0].solve(&inputs[1]).map(|out| vec![out]),
        &[a, b],
        &[None, Some(db)],
    )
    .unwrap();

    assert_eq!(
        result.outputs[0].try_to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]
    );
    assert_eq!(
        result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        vec![Complex64::new(0.5, -1.0), Complex64::new(2.0, 0.25)]
    );
}

#[test]
fn solve_triangular_inv_slogdet_cholesky_pinv_and_matrix_exp_support_jvp() {
    let _runtime = with_cpu_runtime();

    let triangular = jvp(
        |inputs| {
            inputs[0]
                .solve_triangular(&inputs[1], true)
                .map(|out| vec![out])
        },
        &[
            Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap(),
            Tensor::from_slice(&[4.0_f64, 9.0], &[2]).unwrap(),
        ],
        &[
            None,
            Some(Tensor::from_slice(&[1.0_f64, 1.0], &[2]).unwrap()),
        ],
    )
    .unwrap();
    approx_eq(
        &triangular.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.5, 1.0 / 3.0],
    );

    let inv = jvp(
        |inputs| inputs[0].inv().map(|out| vec![out]),
        &[Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap()],
        &[Some(
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
        )],
    )
    .unwrap();
    approx_eq(
        &inv.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[-0.25, 0.0, 0.0, 0.0],
    );

    let slogdet = jvp(
        |inputs| {
            let out = inputs[0].slogdet().unwrap();
            Ok(vec![out.sign, out.logabsdet])
        },
        &[Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap()],
        &[Some(
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
        )],
    )
    .unwrap();
    approx_eq(
        &slogdet.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.0],
    );
    approx_eq(
        &slogdet.output_tangents[1]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.5],
    );

    let cholesky = jvp(
        |inputs| inputs[0].cholesky().map(|out| vec![out]),
        &[Tensor::from_slice(&[4.0_f64, 0.0, 0.0, 9.0], &[2, 2]).unwrap()],
        &[Some(
            Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
        )],
    )
    .unwrap();
    approx_eq(
        &cholesky.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.5, 0.0, 0.0, 0.0],
    );

    let pinv = jvp(
        |inputs| inputs[0].pinv(None).map(|out| vec![out]),
        &[Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap()],
        &[Some(
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
        )],
    )
    .unwrap();
    approx_eq(
        &pinv.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[-0.25, 0.0, 0.0, 0.0],
    );

    let matrix_exp = jvp(
        |inputs| inputs[0].matrix_exp().map(|out| vec![out]),
        &[Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 2.0], &[2, 2]).unwrap()],
        &[Some(
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
        )],
    )
    .unwrap();
    approx_eq(
        &matrix_exp.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[1.0_f64.exp(), 0.0, 0.0, 0.0],
    );
}

#[test]
fn complex_inv_cholesky_pinv_and_matrix_exp_support_jvp() {
    let _runtime = with_cpu_runtime();

    let z1 = Complex64::new(1.0, 1.0);
    let z2 = Complex64::new(2.0, -1.0);
    let a_values = [z1, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), z2];
    let da_values = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];

    let inv = jvp(
        |inputs| inputs[0].inv().map(|out| vec![out]),
        &[Tensor::from_slice(&a_values, &[2, 2]).unwrap()],
        &[Some(Tensor::from_slice(&da_values, &[2, 2]).unwrap())],
    )
    .unwrap();
    complex_approx_eq(
        &inv.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(0.0, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );

    let cholesky = jvp(
        |inputs| inputs[0].cholesky().map(|out| vec![out]),
        &[Tensor::from_slice(
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(9.0, 0.0),
            ],
            &[2, 2],
        )
        .unwrap()],
        &[Some(
            Tensor::from_slice(
                &[
                    Complex64::new(2.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                &[2, 2],
            )
            .unwrap(),
        )],
    )
    .unwrap();
    complex_approx_eq(
        &cholesky.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );

    let pinv = jvp(
        |inputs| inputs[0].pinv(None).map(|out| vec![out]),
        &[Tensor::from_slice(&a_values, &[2, 2]).unwrap()],
        &[Some(Tensor::from_slice(&da_values, &[2, 2]).unwrap())],
    )
    .unwrap();
    complex_approx_eq(
        &pinv.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(0.0, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );

    let matrix_exp = jvp(
        |inputs| inputs[0].matrix_exp().map(|out| vec![out]),
        &[Tensor::from_slice(&a_values, &[2, 2]).unwrap()],
        &[Some(Tensor::from_slice(&da_values, &[2, 2]).unwrap())],
    )
    .unwrap();
    complex_approx_eq(
        &matrix_exp.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            z1.exp(),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
}

#[test]
fn complex_det_and_slogdet_support_jvp() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
        ],
        &[2, 2],
    )
    .unwrap();
    let da = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let det_result = jvp(
        |inputs| inputs[0].det().map(|out| vec![out]),
        &[a],
        &[Some(da)],
    )
    .unwrap();
    complex_approx_eq(
        &det_result.outputs[0].try_to_vec::<Complex64>().unwrap(),
        &[Complex64::new(3.0, 1.0)],
    );
    complex_approx_eq(
        &det_result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[Complex64::new(2.0, -1.0)],
    );

    let a = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
        ],
        &[2, 2],
    )
    .unwrap();
    let da = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let slogdet_result = jvp(
        |inputs| {
            let result = inputs[0].slogdet()?;
            Ok(vec![result.sign, result.logabsdet])
        },
        &[a],
        &[Some(da)],
    )
    .unwrap();
    complex_approx_eq(
        &slogdet_result.outputs[0].try_to_vec::<Complex64>().unwrap(),
        &[Complex64::new(3.0 / 10.0_f64.sqrt(), 1.0 / 10.0_f64.sqrt())],
    );
    approx_eq(
        &slogdet_result.outputs[1].try_to_vec::<f64>().unwrap(),
        &[0.5 * 10.0_f64.ln()],
    );
    complex_approx_eq(
        &slogdet_result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[Complex64::new(
            0.5 / 10.0_f64.sqrt(),
            -1.5 / 10.0_f64.sqrt(),
        )],
    );
    approx_eq(
        &slogdet_result.output_tangents[1]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.5],
    );
}

#[test]
fn lstsq_lu_eig_and_eigh_support_jvp() {
    let _runtime = with_cpu_runtime();

    let lstsq = jvp(
        |inputs| {
            let out = inputs[0].lstsq(&inputs[1]).unwrap();
            Ok(vec![out.solution, out.residuals])
        },
        &[
            Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap(),
            Tensor::from_slice(&[4.0_f64, 9.0], &[2]).unwrap(),
        ],
        &[
            None,
            Some(Tensor::from_slice(&[1.0_f64, 1.0], &[2]).unwrap()),
        ],
    )
    .unwrap();
    approx_eq(
        &lstsq.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.5, 1.0 / 3.0],
    );
    assert_eq!(lstsq.output_tangents[1].as_ref().unwrap().dims(), &[0]);
    assert!(lstsq.output_tangents[1]
        .as_ref()
        .unwrap()
        .try_to_vec::<f64>()
        .unwrap()
        .is_empty());

    let lu = jvp(
        |inputs| {
            let out = inputs[0].lu(LuPivot::Partial).unwrap();
            Ok(vec![out.p, out.l, out.u])
        },
        &[Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap()],
        &[Some(
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
        )],
    )
    .unwrap();
    assert!(lu.output_tangents[0].is_none());
    approx_eq(
        &lu.output_tangents[1]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.0, 0.0, 0.0, 0.0],
    );
    approx_eq(
        &lu.output_tangents[2]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[1.0, 0.0, 0.0, 0.0],
    );

    let eig = jvp(
        |inputs| {
            let out = inputs[0].eig().unwrap();
            Ok(vec![out.values, out.vectors])
        },
        &[Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap()],
        &[Some(
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
        )],
    )
    .unwrap();
    assert_eq!(
        eig.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    assert_eq!(
        eig.output_tangents[1]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    );

    let eigh = jvp(
        |inputs| {
            let out = inputs[0].eigh().unwrap();
            Ok(vec![out.values, out.vectors])
        },
        &[Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap()],
        &[Some(
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 0.0], &[2, 2]).unwrap(),
        )],
    )
    .unwrap();
    approx_eq(
        &eigh.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[1.0, 0.0],
    );
    approx_eq(
        &eigh.output_tangents[1]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[0.0, 0.0, 0.0, 0.0],
    );
}

#[test]
fn complex_eigh_supports_public_jvp() {
    let _runtime = with_cpu_runtime();

    let result = jvp(
        |inputs| {
            let out = inputs[0].eigh().unwrap();
            Ok(vec![out.values, out.vectors])
        },
        &[Tensor::from_slice(
            &[
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            &[2, 2],
        )
        .unwrap()],
        &[Some(
            Tensor::from_slice(
                &[
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                &[2, 2],
            )
            .unwrap(),
        )],
    )
    .unwrap();

    approx_eq(
        &result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[1.0, 0.0],
    );
    assert_eq!(
        result.output_tangents[1]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    );
}

#[test]
fn complex_vector_and_matrix_norm_support_public_jvp() {
    let _runtime = with_cpu_runtime();

    let vector = Tensor::from_slice(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[3],
    )
    .unwrap();
    let dvector = Tensor::from_slice(
        &[
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[3],
    )
    .unwrap();

    let vector_result = jvp(
        |inputs| {
            inputs[0]
                .vector_norm(tenferro::VectorNormOrd::P(2.0), None, false)
                .map(|out| vec![out])
        },
        &[vector],
        &[Some(dvector)],
    )
    .unwrap();
    approx_eq(
        &vector_result.outputs[0].try_to_vec::<f64>().unwrap(),
        &[5.0],
    );
    approx_eq(
        &vector_result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[4.0 / 5.0],
    );

    let matrix = Tensor::from_slice(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap();
    let dmatrix = Tensor::from_slice(
        &[
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let matrix_result = jvp(
        |inputs| {
            inputs[0]
                .matrix_norm(tenferro::MatrixNormOrd::Fro, Some((0, 1)), false)
                .map(|out| vec![out])
        },
        &[matrix],
        &[Some(dmatrix)],
    )
    .unwrap();
    approx_eq(
        &matrix_result.outputs[0].try_to_vec::<f64>().unwrap(),
        &[5.0],
    );
    approx_eq(
        &matrix_result.output_tangents[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<f64>()
            .unwrap(),
        &[4.0 / 5.0],
    );
}
