//! The #2552 unified entry: `x.op(exec)`, `x.op_with(&options, exec)` and
//! `x.op_into(.., &options, exec, destination)` on every concrete input kind, and the
//! matching `op` / `op_with` forms on the traced and eager layers.

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{
    KeepDims, NormOptions, PinvOptions, Side, SolveOptions, SvdGauge, SvdOptions, SvdShape,
    TensorLinalgExt, TracedTensorLinalgExt, TriangularSolveOptions, Uplo,
};
use tenferro_runtime::{GraphCompiler, TraceContext};
use tenferro_tensor::{Tensor, TensorRead, TensorWrite, TypedTensor};

use super::support;

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(vec![rows, cols], data).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1.0e-10, "{actual:?} != {expected:?}");
    }
}

#[test]
fn concrete_svd_has_one_entry_for_owned_borrowed_and_typed_input() {
    let mut engine = CpuEngine::serial();
    let a = matrix(3, 2, vec![3.0, 0.0, 0.0, 0.0, 2.0, 0.0]);

    let (_, owned_s, _) = a.svd(&mut engine).unwrap();
    let read = TensorRead::from_tensor(&a);
    let (_, read_s, _) = read.svd(&mut engine).unwrap();
    assert_close(owned_s.as_slice::<f64>().unwrap(), &[3.0, 2.0]);
    assert_close(read_s.as_slice::<f64>().unwrap(), &[3.0, 2.0]);

    // Typed in, typed out: complex input gives complex factors and real singular values.
    let c = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(0.0, 4.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    let (u, s, vh): (
        TypedTensor<Complex64>,
        TypedTensor<f64>,
        TypedTensor<Complex64>,
    ) = c.as_view().svd(&mut engine).unwrap();
    assert_close(s.host_data().unwrap(), &[4.0, 1.0]);
    assert_eq!((u.shape(), vh.shape()), (&[2, 2][..], &[2, 2][..]));
}

#[test]
fn concrete_svd_shape_selects_full_matrices() {
    let mut engine = CpuEngine::serial();
    let a = matrix(3, 2, vec![3.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    let (u, s, vh) = a
        .svd_with(
            &SvdOptions::default().with_shape(SvdShape::Full),
            &mut engine,
        )
        .unwrap();
    assert_eq!(u.shape(), &[3, 3]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vh.shape(), &[2, 2]);

    let (thin_u, _, _) = a.svd_with(&SvdOptions::default(), &mut engine).unwrap();
    assert_eq!(thin_u.shape(), &[3, 2]);
}

#[test]
fn full_svd_with_a_gauge_is_an_explicit_error() {
    let mut engine = CpuEngine::serial();
    let a = matrix(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
    let options = SvdOptions::default()
        .with_shape(SvdShape::Full)
        .with_gauge(SvdGauge::CanonicalPivot);
    let error = a.svd_with(&options, &mut engine).unwrap_err();
    assert!(
        matches!(error, tenferro_tensor::Error::Unsupported { .. }),
        "{error:?}"
    );
}

#[test]
fn concrete_solve_into_writes_the_destination() {
    let mut engine = CpuEngine::serial();
    let a = matrix(2, 2, vec![2.0, 0.0, 0.0, 4.0]);
    let b = matrix(2, 1, vec![2.0, 8.0]);
    let mut x = matrix(2, 1, vec![0.0, 0.0]);
    a.solve_into(
        &b,
        &SolveOptions::default(),
        &mut engine,
        TensorWrite::from_tensor(&mut x),
    )
    .unwrap();
    assert_close(x.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    assert_close(
        a.solve(&b, &mut engine).unwrap().as_slice::<f64>().unwrap(),
        &[1.0, 2.0],
    );
}

#[test]
fn concrete_triangular_solve_takes_options_instead_of_bool_flags() {
    let mut engine = CpuEngine::serial();
    // Lower triangular [[2, 0], [1, 1]] stored column-major.
    let l = matrix(2, 2, vec![2.0, 1.0, 0.0, 1.0]);
    let b = matrix(2, 1, vec![2.0, 3.0]);
    let options = TriangularSolveOptions::default()
        .with_side(Side::Left)
        .with_uplo(Uplo::Lower);
    let x = l.triangular_solve_with(&b, &options, &mut engine).unwrap();
    assert_close(x.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn concrete_norm_accepts_negative_axes() {
    let mut engine = CpuEngine::serial();
    let a = matrix(2, 2, vec![3.0, 4.0, 0.0, 0.0]);
    let last = a
        .norm_with(&NormOptions::default().with_axes(vec![-2]), &mut engine)
        .unwrap();
    let first = a
        .norm_with(&NormOptions::default().with_axes(vec![0]), &mut engine)
        .unwrap();
    assert_close(last.as_slice::<f64>().unwrap(), &[5.0, 0.0]);
    assert_eq!(
        last.as_slice::<f64>().unwrap(),
        first.as_slice::<f64>().unwrap()
    );

    let kept = a
        .norm_with(
            &NormOptions::default()
                .with_axes(vec![0])
                .with_keep_dims(KeepDims::Keep),
            &mut engine,
        )
        .unwrap();
    assert_eq!(kept.shape(), &[1, 2]);

    let error = a
        .norm_with(&NormOptions::default().with_axes(vec![-3]), &mut engine)
        .unwrap_err();
    assert!(
        matches!(error, tenferro_tensor::Error::Validation { .. }),
        "{error:?}"
    );
}

#[test]
fn concrete_pinv_with_rtol_is_an_option() {
    let mut engine = CpuEngine::serial();
    let a = matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0e-3]);
    let default = a.pinv(&mut engine).unwrap();
    assert_close(default.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0e3]);
    let cut = a
        .pinv_with(&PinvOptions::default().with_rtol(1.0e-2), &mut engine)
        .unwrap();
    assert_close(cut.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn traced_svd_with_shape_full() {
    let trace = TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![3, 2], vec![3.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0])
        .unwrap();
    let (u, s, vh) = a
        .svd_with(&SvdOptions::default().with_shape(SvdShape::Full))
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&u, &s, &vh]).unwrap();
    let outputs = support::run_all(&program, &[]).unwrap();
    assert_eq!(outputs[0].shape(), &[3, 3]);
    assert_close(outputs[1].as_slice::<f64>().unwrap(), &[3.0, 2.0]);
    assert_eq!(outputs[2].shape(), &[2, 2]);

    let error = a
        .svd_with(
            &SvdOptions::default()
                .with_shape(SvdShape::Full)
                .with_gauge(SvdGauge::CanonicalPivot),
        )
        .unwrap_err();
    assert!(format!("{error}").contains("svd"), "{error}");
}

#[cfg(feature = "autodiff")]
#[test]
fn eager_session_entries_take_options_by_reference() {
    use tenferro_ad::{EagerRuntime, EagerTensor};
    use tenferro_linalg::EagerSessionLinalgExt;

    let runtime = EagerRuntime::new().unwrap();
    let a = EagerTensor::from_tensor_in(
        matrix(3, 2, vec![3.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
        runtime.clone(),
    )
    .unwrap();
    let (u, s, vh) = runtime
        .with_eager_session(&mut CpuEngine::serial(), |session| {
            session.svd_with(&a, &SvdOptions::default().with_shape(SvdShape::Full))
        })
        .unwrap();
    assert_eq!(u.shape(), &[3, 3]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vh.shape(), &[2, 2]);

    let l = EagerTensor::from_tensor_in(matrix(2, 2, vec![2.0, 1.0, 0.0, 1.0]), runtime.clone())
        .unwrap();
    let b = EagerTensor::from_tensor_in(matrix(2, 1, vec![2.0, 3.0]), runtime.clone()).unwrap();
    let x = runtime
        .with_eager_session(&mut CpuEngine::serial(), |session| {
            session.triangular_solve_with(
                &l,
                &b,
                &TriangularSolveOptions::default()
                    .with_side(Side::Left)
                    .with_uplo(Uplo::Lower),
            )
        })
        .unwrap();
    let x = x.to_tensor(&mut CpuEngine::serial()).unwrap();
    assert_close(x.as_slice::<f64>().unwrap(), &[1.0, 2.0]);

    let n = runtime
        .with_eager_session(&mut CpuEngine::serial(), |session| {
            session.norm_with(&l, &NormOptions::default().with_axes(vec![-1]))
        })
        .unwrap();
    assert_eq!(n.shape(), &[2]);
}
