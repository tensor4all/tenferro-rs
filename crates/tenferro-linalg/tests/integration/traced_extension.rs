use num_complex::{Complex32, Complex64};
use tenferro_linalg::{
    Diag, Eig, Eigh, FullPivLu, KeepDims, MatOp, NormOptions, NormOrd, PinvOptions, Qr, Svd,
    SvdShape, TriangularSolveOptions, Uplo,
};
use tenferro_linalg::{
    EighDriver, EighGauge, EighOptions, EigvalshOptions, LinalgOps, QrGauge, QrOptions,
    RankRevealingQrOptions, SvdDriver, SvdGauge, SvdOptions, SvdvalsOptions, TracedLinalgOps,
};
use tenferro_runtime::{
    DType, Error, GraphCompiler, Runtime, Tensor, TraceContext, TracedTensor, TypedTensor,
};
use tenferro_tensor::AxisVec;
use tenferro_tensor::Error as TensorError;

use super::support;

fn traced_with_dtype(trace: &TraceContext, dtype: DType, shape: Vec<usize>) -> TracedTensor {
    let n_elements = shape.iter().product();
    let tensor = match dtype {
        DType::F32 => Tensor::from_typed::<f32>(
            TypedTensor::from_vec_col_major(shape, vec![1.0_f32; n_elements]).unwrap(),
        ),
        DType::F64 => Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(shape, vec![1.0_f64; n_elements]).unwrap(),
        ),
        DType::I32 => Tensor::from_typed::<i32>(
            TypedTensor::from_vec_col_major(shape, vec![1_i32; n_elements]).unwrap(),
        ),
        DType::I64 => Tensor::from_typed::<i64>(
            TypedTensor::from_vec_col_major(shape, vec![1_i64; n_elements]).unwrap(),
        ),
        DType::Bool => Tensor::from_typed::<bool>(
            TypedTensor::from_vec_col_major(shape, vec![true; n_elements]).unwrap(),
        ),
        DType::C32 => Tensor::from_typed::<tenferro_tensor::Complex32>(
            TypedTensor::from_vec_col_major(shape, vec![Complex32::new(1.0, 0.5); n_elements])
                .unwrap(),
        ),
        DType::C64 => Tensor::from_typed::<tenferro_tensor::Complex64>(
            TypedTensor::from_vec_col_major(shape, vec![Complex64::new(1.0, 0.5); n_elements])
                .unwrap(),
        ),
        // Test fixtures cover the preset dtypes; an externally defined scalar has
        // no fixture and would change what these tests assert.
        DType::External(_) => unreachable!("test fixtures cover the preset dtypes"),
    };
    trace.from_tensor_concrete_shape(tensor).unwrap()
}

#[test]
fn rank_revealing_qr_traced_fixed_outputs_execute() {
    let trace = tenferro_runtime::TraceContext::new();
    let a = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(
                vec![3, 3],
                vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            .unwrap(),
        )
        .unwrap();
    let result = a
        .rank_revealing_qr_with(&RankRevealingQrOptions::default().rtol(1.0e-12))
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&[
            &result.q,
            &result.r,
            &result.column_permutation,
            &result.rank,
        ])
        .unwrap();
    let outputs = support::run_all(&program, &[]).unwrap();
    assert_eq!(outputs.len(), 4);
    assert_eq!(outputs[0].shape(), &[3, 3]);
    assert_eq!(outputs[1].shape(), &[3, 3]);
    assert_eq!(outputs[2].shape(), &[3]);
    assert_eq!(outputs[3].shape(), &[] as &[usize]);
    assert_eq!(outputs[3].as_slice::<i64>().unwrap(), &[2]);
}

#[test]
fn svd_executes_after_runtime_registration() {
    let trace = tenferro_runtime::TraceContext::new();
    let a = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap(),
        )
        .unwrap();
    let Svd { u, s, vh: vt } = a.svd().unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&u, &s, &vt]).unwrap();
    let outputs = support::run_all(&program, &[]).unwrap();

    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
}

#[test]
fn concrete_decomposition_options_execute_through_backend_defaults() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let (svd_outputs, eigh_outputs, qr_outputs) =
        support::with_cpu_linalg(&mut backend, |backend| {
            let Svd { u, s, vh } = a
                .svd_with(
                    &SvdOptions::default()
                        .gauge(SvdGauge::CanonicalPivot)
                        .derivative_eps(1.0e-10),
                    backend,
                )
                .unwrap();
            let Eigh { values, vectors } = a
                .eigh_with(
                    &EighOptions::default()
                        .gauge(EighGauge::CanonicalPivot)
                        .derivative_eps(1.0e-10),
                    backend,
                )
                .unwrap();
            let Qr { q, r } = a
                .qr_with(
                    &QrOptions::default().gauge(QrGauge::PositiveDiagonal),
                    backend,
                )
                .unwrap();
            ([u, s, vh], [values, vectors], [q, r])
        });

    assert_eq!(svd_outputs[0].shape(), &[2, 2]);
    assert_eq!(svd_outputs[1].shape(), &[2]);
    assert_eq!(svd_outputs[2].shape(), &[2, 2]);
    assert_eq!(eigh_outputs[0].shape(), &[2]);
    assert_eq!(eigh_outputs[1].shape(), &[2, 2]);
    assert_eq!(qr_outputs[0].shape(), &[2, 2]);
    assert_eq!(qr_outputs[1].shape(), &[2, 2]);
}

#[test]
fn traced_decomposition_options_execute_through_registered_runtime() {
    let trace = tenferro_runtime::TraceContext::new();
    let a = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap(),
        )
        .unwrap();
    let Svd { u, s, vh: vt } = a
        .svd_with(
            &SvdOptions::default()
                .gauge(SvdGauge::CanonicalPivot)
                .derivative_eps(1.0e-10),
        )
        .unwrap();
    let Eigh {
        values: eigh_values,
        vectors: eigh_vectors,
    } = a
        .eigh_with(
            &EighOptions::default()
                .gauge(EighGauge::CanonicalPivot)
                .derivative_eps(1.0e-10),
        )
        .unwrap();
    let Qr { q, r } = a
        .qr_with(&QrOptions::default().gauge(QrGauge::PositiveDiagonal))
        .unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&[&u, &s, &vt, &eigh_values, &eigh_vectors, &q, &r])
        .unwrap();
    let outputs = support::run_all(&program, &[]).unwrap();

    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
    assert_eq!(outputs[3].shape(), &[2]);
    assert_eq!(outputs[4].shape(), &[2, 2]);
    assert_eq!(outputs[5].shape(), &[2, 2]);
    assert_eq!(outputs[6].shape(), &[2, 2]);
}

#[test]
fn cpu_ignores_explicit_svd_driver_on_concrete_and_traced_paths() {
    let trace = TraceContext::new();
    for driver in [SvdDriver::QrBased, SvdDriver::Jacobi, SvdDriver::Polar] {
        check_cpu_ignores_svd_driver(&trace, driver);
    }
}

fn check_cpu_ignores_svd_driver(trace: &TraceContext, driver: SvdDriver) {
    // `SvdDriver` selects a cuSOLVER routine; the CPU provider has one SVD
    // kernel, so a forced driver must execute and match the default policy
    // bit-for-bit on the same host.
    let data = vec![3.0_f64, 1.0, 0.5, -2.0, 0.25, 1.5];
    let a = Tensor::from_vec_col_major(vec![3, 2], data.clone()).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let (default_outputs, forced_outputs, default_values, forced_values) =
        support::with_cpu_linalg(&mut backend, |backend| {
            let Svd { u, s, vh } = a.svd(backend).unwrap();
            let default_outputs = [u, s, vh];
            let Svd { u, s, vh } = a
                .svd_with(&SvdOptions::default().driver(driver), backend)
                .unwrap();
            let forced_outputs = [u, s, vh];
            let default_values = a.svdvals(backend).unwrap();
            let forced_values = a
                .svdvals_with(&SvdvalsOptions::default().driver(driver), backend)
                .unwrap();
            (
                default_outputs,
                forced_outputs,
                default_values,
                forced_values,
            )
        });
    for (default, forced) in default_outputs.iter().zip(&forced_outputs) {
        assert_eq!(
            default.as_slice::<f64>().unwrap(),
            forced.as_slice::<f64>().unwrap()
        );
    }
    // Ignoring the driver is a per-entry-point property, so the bit-for-bit
    // comparison stays within one entry point. The LAPACK provider reaches
    // `gesdd` with `jobz = b'S'` for the vectors path and `jobz = b'N'` for
    // the values-only path; those are different routines and their singular
    // values differ in the last ULPs, so the cross-path check is a tolerance.
    assert_eq!(
        forced_values.as_slice::<f64>().unwrap(),
        default_values.as_slice::<f64>().unwrap()
    );
    assert_spectra_close(
        forced_values.as_slice::<f64>().unwrap(),
        default_outputs[1].as_slice::<f64>().unwrap(),
    );

    let traced = trace
        .from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![3, 2], data).unwrap())
        .unwrap();
    let Svd { u, s, vh: vt } = traced
        .svd_with(&SvdOptions::default().driver(driver))
        .unwrap();
    // Only `S` is live here, so the op prunes to a values-only op that keeps
    // the forced driver; the CPU runtime must still execute it.
    let Svd { s: s_only, .. } = traced
        .svd_with(&SvdOptions::default().driver(driver).derivative_eps(2.0e-12))
        .unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&u, &s, &vt, &s_only]).unwrap();
    let outputs = support::run_all(&program, &[]).unwrap();
    for (traced, default) in outputs[..3].iter().zip(&default_outputs) {
        assert_eq!(
            traced.as_slice::<f64>().unwrap(),
            default.as_slice::<f64>().unwrap()
        );
    }
    // The pruned values-only op runs the same backend values path as
    // `svd_values`, so it matches that bit-for-bit; see the note above for why
    // the full-SVD `S` is only compared within tolerance.
    assert_eq!(
        outputs[3].as_slice::<f64>().unwrap(),
        default_values.as_slice::<f64>().unwrap()
    );
    assert_spectra_close(
        outputs[3].as_slice::<f64>().unwrap(),
        default_outputs[1].as_slice::<f64>().unwrap(),
    );
}

#[test]
fn cpu_ignores_explicit_eigh_driver_on_concrete_and_traced_paths() {
    let trace = tenferro_runtime::TraceContext::new();
    // `EighDriver` selects a cuSOLVER routine; the CPU provider has one eigh
    // kernel, so a forced driver must execute and match the default policy on
    // the same host. As with `SvdDriver`, that is a per-entry-point property:
    // the LAPACK provider reaches the vectors and values-only routines
    // through different jobz modes, which differ in the last ULPs.
    let data = vec![2.0_f64, 0.5, -0.25, 0.5, 3.0, 0.75, -0.25, 0.75, 4.0];
    let a = Tensor::from_vec_col_major(vec![3, 3], data.clone()).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let (default_outputs, forced_outputs, default_values, forced_values) =
        support::with_cpu_linalg(&mut backend, |backend| {
            let Eigh { values, vectors } = a.eigh(backend).unwrap();
            let default_outputs = [values, vectors];
            let Eigh { values, vectors } = a
                .eigh_with(&EighOptions::default().driver(EighDriver::Jacobi), backend)
                .unwrap();
            let forced_outputs = [values, vectors];
            let default_values = a.eigvalsh(backend).unwrap();
            let forced_values = a
                .eigvalsh_with(
                    &EigvalshOptions::default().driver(EighDriver::DivideAndConquer),
                    backend,
                )
                .unwrap();
            (
                default_outputs,
                forced_outputs,
                default_values,
                forced_values,
            )
        });
    for (default, forced) in default_outputs.iter().zip(&forced_outputs) {
        assert_eq!(
            default.as_slice::<f64>().unwrap(),
            forced.as_slice::<f64>().unwrap()
        );
    }
    assert_eq!(
        forced_values.as_slice::<f64>().unwrap(),
        default_values.as_slice::<f64>().unwrap()
    );
    assert_spectra_close(
        forced_values.as_slice::<f64>().unwrap(),
        default_outputs[0].as_slice::<f64>().unwrap(),
    );

    let traced = trace
        .from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![3, 3], data).unwrap())
        .unwrap();
    let Eigh {
        values: w,
        vectors: v,
    } = traced
        .eigh_with(&EighOptions::default().driver(EighDriver::Jacobi))
        .unwrap();
    // Only the eigenvalues are live here, so the op prunes to a values-only
    // op that keeps the forced driver; the CPU runtime must still execute it.
    let Eigh { values: w_only, .. } = traced
        .eigh_with(&EighOptions::default().driver(EighDriver::DivideAndConquer))
        .unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&w, &v, &w_only]).unwrap();
    let outputs = support::run_all(&program, &[]).unwrap();
    for (traced, default) in outputs[..2].iter().zip(&default_outputs) {
        assert_eq!(
            traced.as_slice::<f64>().unwrap(),
            default.as_slice::<f64>().unwrap()
        );
    }
    assert_eq!(
        outputs[2].as_slice::<f64>().unwrap(),
        default_values.as_slice::<f64>().unwrap()
    );
    assert_spectra_close(
        outputs[2].as_slice::<f64>().unwrap(),
        default_outputs[0].as_slice::<f64>().unwrap(),
    );
}

/// Assert two spectra (singular values or eigenvalues) agree to rounding.
///
/// Values-only and vectors LAPACK drivers are distinct routines, so their
/// results are equal only up to the last ULPs.
fn assert_spectra_close(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    let scale = right
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
        .max(1.0);
    let tolerance = 1.0e-12 * scale;
    for (left, right) in left.iter().zip(right) {
        assert!(
            (left - right).abs() <= tolerance,
            "spectra differ beyond rounding: {left} vs {right} (tolerance {tolerance})"
        );
    }
}

#[test]
fn complex_svd_runtime_singular_values_match_traced_real_dtype() {
    let trace = tenferro_runtime::TraceContext::new();
    let a = trace
        .from_tensor_concrete_shape(Tensor::from_typed::<tenferro_tensor::Complex64>(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex64::new(3.0, 0.5),
                    Complex64::new(0.2, -0.4),
                    Complex64::new(-0.1, 0.3),
                    Complex64::new(2.0, -0.2),
                ],
            )
            .unwrap(),
        ))
        .unwrap();
    let Svd { u: _u, s, vh: _vt } = a.svd().unwrap();
    assert_eq!(s.dtype(), DType::F64);

    let weights = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 2.0]).unwrap(),
        )
        .unwrap();
    let weighted = (&s * &weights).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&s, &weighted]).unwrap();
    let outputs = support::run_all(&program, &[]).unwrap();

    assert_eq!(outputs[0].dtype(), DType::F64);
    assert_eq!(outputs[1].dtype(), DType::F64);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2]);
}

#[test]
fn spectral_norm_compile_prunes_residual_svd_to_values_only_op() {
    let trace = tenferro_runtime::TraceContext::new();
    let a = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![3, 2], vec![3.0_f64, 0.1, 0.2, 0.3, 2.0, 0.4]).unwrap(),
        )
        .unwrap();
    let norm = a
        .norm_with(
            &NormOptions::default()
                .ord(NormOrd::P(2.0))
                .axes(Some([0, 1].into_iter().collect::<AxisVec>())),
        )
        .unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&norm).unwrap();
    let extension_ops = program
        .program()
        .operations()
        .filter_map(|operation| match operation.op() {
            tenferro_runtime::program::SemanticOpRef::Extension(op)
                if op.family_id() == tenferro_linalg::LINALG_EXTENSION_FAMILY_ID =>
            {
                Some(format!("{op:?}"))
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(
        extension_ops.iter().any(|op| op.contains("SvdVals")),
        "forward-only spectral norm should lower residual SVD to values-only op: {extension_ops:#?}"
    );
    assert!(
        !extension_ops.iter().any(|op| op.contains("Svd {")),
        "forward-only spectral norm should not execute full SVD after pruning: {extension_ops:#?}"
    );
}

#[test]
fn missing_runtime_reports_linalg_family() {
    let trace = TraceContext::new();
    let a = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
        )
        .unwrap();
    let y = a.cholesky().unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();
    let err = support::run_program(&runtime, &mut backend, &program, &[]).unwrap_err();

    assert!(err
        .to_string()
        .contains(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID));
}

#[test]
fn full_piv_lu_multi_output_slots_are_preserved() {
    let trace = TraceContext::new();
    let a = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]).unwrap(),
        )
        .unwrap();
    let FullPivLu { p, l, u, q, parity } = a.full_piv_lu().unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&p, &l, &u, &q, &parity]).unwrap();
    let outputs = support::run_all(&program, &[]).unwrap();

    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
    assert_eq!(outputs[3].shape(), &[2, 2]);
    assert_eq!(outputs[4].shape(), &[] as &[usize]);
}

#[test]
fn traced_metadata_matches_linalg_extension_shapes_and_dtypes() {
    let trace = TraceContext::new();
    let rectangular = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![3, 4, 2], vec![1.0_f64; 24]).unwrap(),
        )
        .unwrap();
    let square = trace
        .from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![3, 3, 2], vec![1.0_f64; 18]).unwrap(),
        )
        .unwrap();
    let complex_square = trace
        .from_tensor_concrete_shape(Tensor::from_typed::<tenferro_tensor::Complex64>(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![Complex64::new(1.0, 0.0); 4]).unwrap(),
        ))
        .unwrap();
    let ints = trace
        .from_tensor_concrete_shape(Tensor::from_typed::<i64>(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![1, 0, 0, 2]).unwrap(),
        ))
        .unwrap();

    let Svd { u, s, vh: vt } = rectangular.svd().unwrap();
    assert_eq!(u.concrete_shape().unwrap(), vec![3, 3, 2]);
    assert_eq!(s.concrete_shape().unwrap(), vec![3, 2]);
    assert_eq!(vt.concrete_shape().unwrap(), vec![3, 4, 2]);

    let Qr { q, r } = rectangular.qr().unwrap();
    assert_eq!(q.concrete_shape().unwrap(), vec![3, 3, 2]);
    assert_eq!(r.concrete_shape().unwrap(), vec![3, 4, 2]);

    let Eig { values, vectors } = ints.eig().unwrap();
    assert_eq!(values.dtype(), DType::C64);
    assert_eq!(vectors.dtype(), DType::C64);

    let Eigh {
        values: eigh_values,
        vectors: eigh_vectors,
    } = square.eigh().unwrap();
    assert_eq!(eigh_values.concrete_shape().unwrap(), vec![3, 2]);
    assert_eq!(eigh_vectors.concrete_shape().unwrap(), vec![3, 3, 2]);

    let Eigh {
        values: complex_eigh_values,
        vectors: complex_eigh_vectors,
    } = complex_square.eigh().unwrap();
    assert_eq!(complex_eigh_values.dtype(), DType::F64);
    assert_eq!(complex_eigh_vectors.dtype(), DType::C64);

    let FullPivLu { p, l, u, q, parity } = complex_square.full_piv_lu().unwrap();
    assert_eq!(p.dtype(), DType::C64);
    assert_eq!(l.dtype(), DType::C64);
    assert_eq!(u.dtype(), DType::C64);
    assert_eq!(q.dtype(), DType::C64);
    assert_eq!(parity.dtype(), DType::F64);
    assert_eq!(parity.rank(), 0);
}

#[test]
fn traced_metadata_promotes_linalg_dtypes_broadly() {
    let trace = TraceContext::new();
    let promotion_cases = [
        (DType::Bool, DType::I32, DType::I32),
        (DType::I32, DType::I64, DType::I64),
        (DType::I32, DType::F32, DType::F64),
        (DType::I64, DType::C32, DType::C64),
        (DType::F32, DType::F64, DType::F64),
        (DType::F32, DType::C32, DType::C32),
        (DType::F32, DType::C64, DType::C64),
        (DType::F64, DType::C32, DType::C64),
        (DType::C32, DType::C64, DType::C64),
    ];

    for (a_dtype, b_dtype, expected_dtype) in promotion_cases {
        let a = traced_with_dtype(&trace, a_dtype, vec![2, 2]);
        let b = traced_with_dtype(&trace, b_dtype, vec![2, 1]);
        let solved = a.solve(&b).unwrap();
        assert_eq!(solved.dtype(), expected_dtype);
        assert_eq!(solved.concrete_shape().unwrap(), vec![2, 1]);
    }

    let triangular = traced_with_dtype(&trace, DType::Bool, vec![2, 2])
        .triangular_solve_with(
            &traced_with_dtype(&trace, DType::F32, vec![2, 1]),
            &TriangularSolveOptions::default()
                .uplo(Uplo::Upper)
                .op(MatOp::Transpose)
                .diag(Diag::Unit),
        )
        .unwrap();
    assert_eq!(triangular.dtype(), DType::F32);
    assert_eq!(triangular.concrete_shape().unwrap(), vec![2, 1]);
}

#[test]
fn traced_metadata_covers_eig_output_dtype_rules() {
    let trace = TraceContext::new();
    for (input_dtype, expected_dtype) in [
        (DType::F64, DType::C64),
        (DType::C64, DType::C64),
        (DType::F32, DType::C32),
        (DType::C32, DType::C32),
        (DType::I32, DType::C64),
        (DType::I64, DType::C64),
        (DType::Bool, DType::C64),
    ] {
        let a = traced_with_dtype(&trace, input_dtype, vec![2, 2]);
        let Eig { values, vectors } = a.eig().unwrap();
        assert_eq!(values.dtype(), expected_dtype);
        assert_eq!(vectors.dtype(), expected_dtype);
        assert_eq!(values.concrete_shape().unwrap(), vec![2]);
        assert_eq!(vectors.concrete_shape().unwrap(), vec![2, 2]);
    }
}

#[test]
fn traced_norm_rejects_integer_and_bool_dtypes_before_scalar_rounding() {
    let trace = TraceContext::new();
    for dtype in [DType::I32, DType::I64, DType::Bool] {
        let tensor = traced_with_dtype(&trace, dtype, vec![3]);
        let err = match tensor.norm_with(
            &NormOptions::default()
                .ord(NormOrd::P(2.0))
                .axes(Some([0].into_iter().collect::<AxisVec>())),
        ) {
            Ok(_) => panic!("expected unsupported dtype error for {dtype:?}"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                Error::TensorRuntime(TensorError::Extension {
                    op: "norm",
                    family: tenferro_linalg::LINALG_EXTENSION_FAMILY_ID,
                    kind: tenferro_tensor::ErrorKind::Unsupported,
                    ..
                })
            ),
            "expected unsupported dtype error for {dtype:?}, got {err:?}"
        );

        let err = match tensor.pinv_with(&PinvOptions::default().rtol(Some(1.0e-12))) {
            Ok(_) => panic!("expected unsupported dtype error for {dtype:?}"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                Error::TensorRuntime(TensorError::Extension {
                    op: "pinv_with_rtol",
                    family: tenferro_linalg::LINALG_EXTENSION_FAMILY_ID,
                    kind: tenferro_tensor::ErrorKind::Unsupported,
                    ..
                })
            ),
            "expected unsupported dtype error for {dtype:?}, got {err:?}"
        );
    }
}

#[test]
fn traced_inv_rejects_rank_less_than_two_without_panicking() {
    let trace = TraceContext::new();
    let scalar = trace.from_vec_col_major(vec![], vec![2.0_f64]).unwrap();

    let err = scalar.inv().unwrap_err();

    assert!(matches!(
        err,
        Error::TensorRuntime(TensorError::Validation {
            op: "inv",
            source: tenferro_tensor::ValidationError::RankMismatch {
                expected: 2,
                actual: 0,
            },
        })
    ));
}

fn assert_linalg_rank_mismatch<T>(name: &str, result: tenferro_runtime::Result<T>, actual: usize) {
    let err = match result {
        Ok(_) => panic!("{name} should reject rank < 2 inputs"),
        Err(err) => err,
    };
    assert!(
        matches!(
            err,
            Error::TensorRuntime(TensorError::Validation {
                op: _,
                source: tenferro_tensor::ValidationError::RankMismatch {
                    expected: 2,
                    actual: got,
                },
            }) if got == actual
        ),
        "expected rank mismatch for {name}, got {err:?}"
    );
}

#[test]
fn traced_linalg_metadata_helpers_reject_rank_less_than_two_without_panicking() {
    let trace = TraceContext::new();
    let scalar = trace.from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let vector = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();

    assert_linalg_rank_mismatch("svd scalar", scalar.svd(), 0);
    assert_linalg_rank_mismatch("svd vector", vector.svd(), 1);
    assert_linalg_rank_mismatch("qr vector", vector.qr(), 1);
    assert_linalg_rank_mismatch("lu vector", vector.lu(), 1);
    assert_linalg_rank_mismatch("full_piv_lu vector", vector.full_piv_lu(), 1);
    assert_linalg_rank_mismatch("eigh vector", vector.eigh(), 1);
    assert_linalg_rank_mismatch("eig vector", vector.eig(), 1);
    assert_linalg_rank_mismatch("eigvalsh vector", vector.eigvalsh(), 1);
    assert_linalg_rank_mismatch("eigvals vector", vector.eigvals(), 1);
    assert_linalg_rank_mismatch("solve vector", vector.solve(&vector), 1);
    assert_linalg_rank_mismatch("slogdet vector", vector.slogdet(), 1);
    assert_linalg_rank_mismatch("det vector", vector.det(), 1);
    assert_linalg_rank_mismatch("pinv vector", vector.pinv(), 1);
}

#[test]
fn traced_linalg_helpers_reject_symbolic_shapes_without_panicking() {
    let trace = TraceContext::new();
    let matrix = trace.input_symbolic_shape(DType::F64, 2).unwrap();

    for (op, result) in [
        ("inv", matrix.inv()),
        ("pinv", matrix.pinv()),
        (
            "pinv_with_rtol",
            matrix.pinv_with(&PinvOptions::default().rtol(Some(1.0e-12))),
        ),
        (
            "norm",
            matrix.norm_with(
                &NormOptions::default()
                    .ord(NormOrd::P(2.0))
                    .axes(Some([0, 1].into_iter().collect::<AxisVec>()))
                    .keep_dims(KeepDims::Yes),
            ),
        ),
    ] {
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                Error::TensorRuntime(TensorError::Validation {
                    op: actual_op,
                    source: tenferro_tensor::ValidationError::InvalidArgument {
                        argument: "shape",
                        ..
                    },
                }) if actual_op == op
            ),
            "expected symbolic-shape error for {op}, got {err:?}"
        );
    }
}

#[test]
fn traced_norm_rejects_duplicate_axis_before_empty_shortcut() {
    let trace = TraceContext::new();
    let nonempty = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let empty = trace
        .from_tensor_concrete_shape(Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![0, 2], vec![]).unwrap(),
        ))
        .unwrap();

    for err in [
        nonempty
            .norm_with(&NormOptions::default().axes(Some([0, 0].into_iter().collect::<AxisVec>())))
            .unwrap_err(),
        empty
            .norm_with(&NormOptions::default().axes(Some([0, 0].into_iter().collect::<AxisVec>())))
            .unwrap_err(),
    ] {
        assert!(matches!(
            err,
            Error::TensorRuntime(TensorError::Validation {
                op: "norm",
                source: tenferro_tensor::ValidationError::DuplicateAxis {
                    axis: 0,
                    role: "axes"
                }
            })
        ));
    }
}

#[test]
fn traced_norm_rejects_out_of_range_axis_without_panicking() {
    let trace = TraceContext::new();
    let tensor = trace
        .from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])
        .unwrap();

    let err = tensor
        .norm_with(
            &NormOptions::default()
                .ord(NormOrd::P(2.0))
                .axes(Some([5].into_iter().collect::<AxisVec>())),
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::TensorRuntime(TensorError::Validation {
            op: "norm",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { axis: 5, rank: 1 },
        })
    ));
}

#[test]
fn traced_pinv_rejects_integer_and_bool_dtypes_before_scalar_rounding() {
    let trace = TraceContext::new();
    for dtype in [DType::I32, DType::I64, DType::Bool] {
        let tensor = traced_with_dtype(&trace, dtype, vec![2, 2]);
        let err = match tensor.pinv() {
            Ok(_) => panic!("expected unsupported dtype error for {dtype:?}"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                Error::TensorRuntime(TensorError::Extension {
                    op: "pinv",
                    family: tenferro_linalg::LINALG_EXTENSION_FAMILY_ID,
                    kind: tenferro_tensor::ErrorKind::Unsupported,
                    ..
                })
            ),
            "expected unsupported dtype error for {dtype:?}, got {err:?}"
        );
    }
}

fn assert_unsupported_dtype_at_construction(
    result: tenferro_runtime::Result<()>,
    op: &'static str,
    dtype: DType,
) {
    let err = match result {
        Ok(()) => panic!("expected unsupported dtype error for {op} {dtype:?}"),
        Err(err) => err,
    };
    assert!(
        matches!(
            err,
            Error::TensorRuntime(TensorError::Extension {
                op: found_op,
                family: tenferro_linalg::LINALG_EXTENSION_FAMILY_ID,
                kind: tenferro_tensor::ErrorKind::Unsupported,
                ..
            }) if found_op == op
        ),
        "expected unsupported dtype error for {op} {dtype:?}, got {err:?}"
    );
}

#[test]
fn traced_svd_eigh_qr_cholesky_reject_integer_and_bool_dtypes_at_construction() {
    let trace = TraceContext::new();
    for dtype in [DType::I32, DType::I64, DType::Bool] {
        let tensor = traced_with_dtype(&trace, dtype, vec![2, 2]);
        assert_unsupported_dtype_at_construction(tensor.svd().map(|_| ()), "svd", dtype);
        assert_unsupported_dtype_at_construction(
            tensor
                .svd_with(&SvdOptions::default().shape(SvdShape::Full))
                .map(|_| ()),
            "svd_full",
            dtype,
        );
        assert_unsupported_dtype_at_construction(tensor.eigh().map(|_| ()), "eigh", dtype);
        assert_unsupported_dtype_at_construction(tensor.eigvalsh().map(|_| ()), "eigvalsh", dtype);
        assert_unsupported_dtype_at_construction(tensor.qr().map(|_| ()), "qr", dtype);
        assert_unsupported_dtype_at_construction(tensor.cholesky().map(|_| ()), "cholesky", dtype);
    }
}
