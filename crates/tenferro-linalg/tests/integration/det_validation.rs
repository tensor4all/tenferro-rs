//! Square-only validation for the determinant contract and its sibling ops.
//!
//! The determinant family accepts only batched matrices with equal row and
//! column extents. Rectangular inputs must fail on every public surface before
//! an empty-input shortcut or a factorization can return a value. The audit
//! below covers the other square-only linalg operations so a future empty fast
//! path cannot silently reintroduce a rectangular value.

use tenferro_cpu::CpuEngine;
use tenferro_linalg::{
    TensorLinalgExt, TensorReadLinalgExt, TracedTensorLinalgExt, TypedTensorLinalgExt,
};
use tenferro_runtime::{DType, Error, GraphCompiler, TracedTensor};
use tenferro_tensor::{Tensor, TensorRead, TypedTensor, ValidationError};

#[cfg(feature = "autodiff")]
use std::sync::{Arc, OnceLock};
#[cfg(feature = "autodiff")]
use tenferro_ad::{EagerRuntime, EagerTensor};
#[cfg(feature = "autodiff")]
use tenferro_linalg::EagerSessionLinalgExt;

fn data_for(shape: &[usize]) -> Vec<f64> {
    vec![1.0; shape.iter().product()]
}

fn assert_tensor_shape_mismatch(label: &str, error: &tenferro_tensor::Error) {
    assert!(
        matches!(
            error,
            tenferro_tensor::Error::Validation {
                source: ValidationError::ShapeMismatch { .. },
                ..
            }
        ),
        "{label}: expected a shape mismatch, got {error:?}"
    );
}

fn assert_runtime_shape_mismatch(label: &str, error: &Error) {
    match error {
        Error::TensorRuntime(error) => assert_tensor_shape_mismatch(label, error),
        other => panic!("{label}: expected a tensor shape mismatch, got {other:?}"),
    }
}

fn assert_error<T, E>(label: &str, result: Result<T, E>) {
    assert!(result.is_err(), "{label}: expected an error, got a value");
}

fn assert_traced_error(label: &str, result: tenferro_runtime::Result<Vec<TracedTensor>>) {
    match result {
        Err(_) => {}
        Ok(outputs) => {
            let references: Vec<&TracedTensor> = outputs.iter().collect();
            let outcome = GraphCompiler::new()
                .compile_many(&references)
                .and_then(|program| super::support::run_all(&program, &[]));
            assert!(
                outcome.is_err(),
                "{label}: expected an error, but build, compile, and execution all returned a value"
            );
        }
    }
}

fn rhs_shape(shape: &[usize]) -> Vec<usize> {
    let mut rhs = vec![shape[0], 1];
    rhs.extend_from_slice(&shape[2..]);
    rhs
}

#[cfg(feature = "autodiff")]
fn eager_runtime() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::new().unwrap()).clone()
}

#[cfg(feature = "autodiff")]
fn eager(tensor: &Tensor) -> EagerTensor {
    EagerTensor::from_tensor_in(tensor.duplicate().unwrap(), eager_runtime()).unwrap()
}

const RECTANGULAR_DETERMINANT_SHAPES: [&[usize]; 5] =
    [&[2, 3], &[3, 2], &[0, 3], &[3, 0], &[2, 3, 0]];

#[test]
fn determinant_rejects_rectangular_matrices_on_every_surface() {
    let mut host = CpuEngine::serial();
    for shape in RECTANGULAR_DETERMINANT_SHAPES {
        let data = data_for(shape);
        let tensor = Tensor::from_vec_col_major(shape.to_vec(), data.clone()).unwrap();
        let typed = TypedTensor::<f64>::from_vec_col_major(shape.to_vec(), data).unwrap();

        assert_tensor_shape_mismatch("owned slogdet", &tensor.slogdet(&mut host).unwrap_err());
        assert_tensor_shape_mismatch("owned det", &tensor.det(&mut host).unwrap_err());
        assert_tensor_shape_mismatch(
            "read slogdet",
            &TensorRead::from_tensor(&tensor)
                .slogdet_read(&mut host)
                .unwrap_err(),
        );
        assert_tensor_shape_mismatch(
            "read det",
            &TensorRead::from_tensor(&tensor)
                .det_read(&mut host)
                .unwrap_err(),
        );
        assert_tensor_shape_mismatch("typed slogdet", &typed.slogdet(&mut host).unwrap_err());
        assert_tensor_shape_mismatch("typed det", &typed.det(&mut host).unwrap_err());

        #[cfg(feature = "autodiff")]
        {
            let runtime = eager_runtime();
            let eager = eager(&tensor);
            let slogdet = runtime
                .with_eager_session(&mut CpuEngine::serial(), |session| session.slogdet(&eager))
                .unwrap_err();
            assert_runtime_shape_mismatch("eager slogdet", &slogdet);
            let det = runtime
                .with_eager_session(&mut CpuEngine::serial(), |session| session.det(&eager))
                .unwrap_err();
            assert_runtime_shape_mismatch("eager det", &det);
        }

        let traced = TracedTensor::from_vec_col_major(shape.to_vec(), data_for(shape)).unwrap();
        let slogdet = traced.slogdet().unwrap_err();
        assert_runtime_shape_mismatch("traced slogdet", &slogdet);
        let det = traced.det().unwrap_err();
        assert_runtime_shape_mismatch("traced det", &det);
    }
}

#[test]
#[cfg(feature = "autodiff")]
fn empty_square_eager_determinant_rejects_unsupported_dtypes_like_the_concrete_surface() {
    let mut host = CpuEngine::serial();
    let runtime = eager_runtime();
    for tensor in [
        Tensor::from_vec_col_major(vec![0, 0], Vec::<i32>::new()).unwrap(),
        Tensor::from_vec_col_major(vec![0, 0], Vec::<bool>::new()).unwrap(),
    ] {
        let concrete = tensor.slogdet(&mut host).unwrap_err();
        assert_eq!(concrete.kind(), tenferro_tensor::ErrorKind::Unsupported);

        let eager = eager(&tensor);
        let eager_error = runtime
            .with_eager_session(&mut CpuEngine::serial(), |session| session.slogdet(&eager))
            .unwrap_err();
        assert_eq!(
            eager_error.kind(),
            concrete.kind(),
            "eager empty slogdet must reject {:?} like the concrete surface",
            tensor.dtype()
        );
    }
}

#[test]
fn empty_square_determinant_remains_one_and_zero() {
    let mut host = CpuEngine::serial();
    let tensor = Tensor::from_vec_col_major(vec![0, 0], Vec::<f64>::new()).unwrap();
    let typed = TypedTensor::<f64>::from_vec_col_major(vec![0, 0], Vec::<f64>::new()).unwrap();

    let (sign, logabsdet) = tensor.slogdet(&mut host).unwrap();
    assert_eq!(sign.as_slice::<f64>().unwrap(), &[1.0]);
    assert_eq!(logabsdet.as_slice::<f64>().unwrap(), &[0.0]);
    let det = tensor.det(&mut host).unwrap();
    assert_eq!(det.as_slice::<f64>().unwrap(), &[1.0]);

    let (sign, logabsdet) = { TensorRead::from_tensor(&tensor).slogdet_read(&mut host) }.unwrap();
    assert_eq!(sign.as_slice::<f64>().unwrap(), &[1.0]);
    assert_eq!(logabsdet.as_slice::<f64>().unwrap(), &[0.0]);
    let det = { TensorRead::from_tensor(&tensor).det_read(&mut host) }.unwrap();
    assert_eq!(det.as_slice::<f64>().unwrap(), &[1.0]);

    let (sign, logabsdet) = typed.slogdet(&mut host).unwrap();
    assert_eq!(sign.as_slice().unwrap(), &[1.0]);
    assert_eq!(logabsdet.as_slice().unwrap(), &[0.0]);
    let det = typed.det(&mut host).unwrap();
    assert_eq!(det.as_slice().unwrap(), &[1.0]);

    #[cfg(feature = "autodiff")]
    {
        let runtime = eager_runtime();
        let eager = eager(&tensor);
        let (sign, logabsdet) = runtime
            .with_eager_session(&mut CpuEngine::serial(), |session| session.slogdet(&eager))
            .unwrap();
        let sign = sign.to_tensor(&mut CpuEngine::serial()).unwrap();
        let logabsdet = logabsdet.to_tensor(&mut CpuEngine::serial()).unwrap();
        assert_eq!(sign.as_slice::<f64>().unwrap(), &[1.0]);
        assert_eq!(logabsdet.as_slice::<f64>().unwrap(), &[0.0]);
        let det = runtime
            .with_eager_session(&mut CpuEngine::serial(), |session| session.det(&eager))
            .unwrap()
            .to_tensor(&mut CpuEngine::serial())
            .unwrap();
        assert_eq!(det.as_slice::<f64>().unwrap(), &[1.0]);
    }

    let traced = TracedTensor::from_vec_col_major(vec![0, 0], Vec::<f64>::new()).unwrap();
    let (sign, logabsdet) = traced.slogdet().unwrap();
    let det = traced.det().unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&sign, &logabsdet, &det])
        .unwrap();
    let outputs = super::support::run_all(&program, &[]).unwrap();
    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[1.0]);
    assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[0.0]);
    assert_eq!(outputs[2].as_slice::<f64>().unwrap(), &[1.0]);
}

#[test]
fn symbolic_determinant_records_square_constraint_for_every_output() {
    let input = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let (sign, logabsdet) = input.slogdet().unwrap();
    let det = input.det().unwrap();

    for (label, output) in [("det", &det), ("sign", &sign), ("logabsdet", &logabsdet)] {
        for rectangular in [&[2, 3][..], &[3, 2][..]] {
            let error = GraphCompiler::new()
                .compile_with_input_specs(output, &[(&input, DType::F64, rectangular)])
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    Error::ShapeConstraintViolation {
                        family: "tenferro-linalg.linalg.v1",
                        ..
                    }
                ),
                "{label} with {rectangular:?}: expected a square-shape constraint violation, got {error:?}"
            );
        }
        GraphCompiler::new()
            .compile_with_input_specs(output, &[(&input, DType::F64, &[3, 3])])
            .unwrap_or_else(|error| panic!("{label}: square binding failed: {error:?}"));
    }
}

#[test]
fn square_only_ops_reject_rectangular_inputs_on_every_surface() {
    let mut host = CpuEngine::serial();
    const OPS: [&str; 10] = [
        "inv",
        "eigh",
        "eigvalsh",
        "eig",
        "eigvals",
        "cholesky",
        "solve",
        "triangular_solve",
        "full_piv_lu",
        "full_piv_lu_solve",
    ];
    const SHAPES: [&[usize]; 3] = [&[2, 3], &[0, 3], &[2, 3, 0]];

    for op in OPS {
        for shape in SHAPES {
            let rhs_shape = rhs_shape(shape);
            let a = Tensor::from_vec_col_major(shape.to_vec(), data_for(shape)).unwrap();
            let b = Tensor::from_vec_col_major(rhs_shape.clone(), data_for(&rhs_shape)).unwrap();
            let typed_a =
                TypedTensor::<f64>::from_vec_col_major(shape.to_vec(), data_for(shape)).unwrap();
            let typed_b =
                TypedTensor::<f64>::from_vec_col_major(rhs_shape.clone(), data_for(&rhs_shape))
                    .unwrap();
            let traced_a =
                TracedTensor::from_vec_col_major(shape.to_vec(), data_for(shape)).unwrap();
            let traced_b =
                TracedTensor::from_vec_col_major(rhs_shape.clone(), data_for(&rhs_shape)).unwrap();

            let label = |surface: &str| format!("{op} {surface} {shape:?}");

            #[cfg(feature = "autodiff")]
            let (eager_a, eager_b) = (eager(&a), eager(&b));

            match op {
                "inv" => {
                    assert_error(&label("owned"), a.inv(&mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a).inv_read(&mut host),
                    );
                    assert_error(&label("typed"), typed_a.inv(&mut host));
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.inv(&eager_a)
                        }),
                    );
                    assert_traced_error(&label("traced"), traced_a.inv().map(|x| vec![x]));
                }
                "eigh" => {
                    assert_error(&label("owned"), a.eigh(&mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a).eigh_read(&mut host),
                    );
                    assert_error(&label("typed"), typed_a.eigh(&mut host));
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.eigh(&eager_a)
                        }),
                    );
                    assert_traced_error(
                        &label("traced"),
                        traced_a
                            .eigh()
                            .map(|(values, vectors)| vec![values, vectors]),
                    );
                }
                "eigvalsh" => {
                    assert_error(&label("owned"), a.eigvalsh(&mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a).eigvalsh_read(&mut host),
                    );
                    assert_error(&label("typed"), typed_a.eigvalsh(&mut host));
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.eigvalsh(&eager_a)
                        }),
                    );
                    assert_traced_error(&label("traced"), traced_a.eigvalsh().map(|x| vec![x]));
                }
                "eig" => {
                    assert_error(&label("owned"), a.eig(&mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a).eig_read(&mut host),
                    );
                    assert_error(&label("typed"), typed_a.eig(&mut host));
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.eig(&eager_a)
                        }),
                    );
                    assert_traced_error(
                        &label("traced"),
                        traced_a
                            .eig()
                            .map(|(values, vectors)| vec![values, vectors]),
                    );
                }
                "eigvals" => {
                    assert_error(&label("owned"), a.eigvals(&mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a).eigvals_read(&mut host),
                    );
                    assert_error(&label("typed"), typed_a.eigvals(&mut host));
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.eigvals(&eager_a)
                        }),
                    );
                    assert_traced_error(&label("traced"), traced_a.eigvals().map(|x| vec![x]));
                }
                "cholesky" => {
                    assert_error(&label("owned"), a.cholesky(&mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a).cholesky_read(&mut host),
                    );
                    assert_error(&label("typed"), typed_a.cholesky(&mut host));
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.cholesky(&eager_a)
                        }),
                    );
                    assert_traced_error(&label("traced"), traced_a.cholesky().map(|x| vec![x]));
                }
                "solve" => {
                    assert_error(&label("owned"), a.solve(&b, &mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a)
                            .solve_read(TensorRead::from_tensor(&b), &mut host),
                    );
                    assert_error(&label("typed"), typed_a.solve(&typed_b, &mut host));
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.solve(&eager_a, &eager_b)
                        }),
                    );
                    assert_traced_error(
                        &label("traced"),
                        traced_a.solve(&traced_b).map(|x| vec![x]),
                    );
                }
                "triangular_solve" => {
                    assert_error(
                        &label("owned"),
                        a.triangular_solve(&b, true, false, false, false, &mut host),
                    );
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a).triangular_solve_read(
                            TensorRead::from_tensor(&b),
                            true,
                            false,
                            false,
                            false,
                            &mut host,
                        ),
                    );
                    assert_error(
                        &label("typed"),
                        typed_a.triangular_solve(&typed_b, true, false, false, false, &mut host),
                    );
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.triangular_solve(&eager_a, &eager_b, true, false, false, false)
                        }),
                    );
                    assert_traced_error(
                        &label("traced"),
                        traced_a
                            .triangular_solve(&traced_b, true, false, false, false)
                            .map(|x| vec![x]),
                    );
                }
                "full_piv_lu" => {
                    assert_error(&label("owned"), a.full_piv_lu(&mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a).full_piv_lu_read(&mut host),
                    );
                    assert_error(&label("typed"), typed_a.full_piv_lu(&mut host));
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.full_piv_lu(&eager_a)
                        }),
                    );
                    assert_traced_error(
                        &label("traced"),
                        traced_a
                            .full_piv_lu()
                            .map(|(p, l, u, q, parity)| vec![p, l, u, q, parity]),
                    );
                }
                "full_piv_lu_solve" => {
                    assert_error(&label("owned"), a.full_piv_lu_solve(&b, &mut host));
                    assert_error(
                        &label("read"),
                        TensorRead::from_tensor(&a)
                            .full_piv_lu_solve_read(TensorRead::from_tensor(&b), &mut host),
                    );
                    assert_error(
                        &label("typed"),
                        typed_a.full_piv_lu_solve(&typed_b, &mut host),
                    );
                    #[cfg(feature = "autodiff")]
                    assert_error(
                        &label("eager"),
                        eager_runtime().with_eager_session(&mut CpuEngine::serial(), |session| {
                            session.full_piv_lu_solve(&eager_a, &eager_b)
                        }),
                    );
                    assert_traced_error(
                        &label("traced"),
                        traced_a.full_piv_lu_solve(&traced_b).map(|x| vec![x]),
                    );
                }
                other => panic!("unregistered square-only audit op {other}"),
            }
        }
    }
}
