//! Prototype 5 tests: do family errors stay reachable through `source()` in all
//! three layers? Each test pins what the code does today. See findings/p5.md.

use std::any::TypeId;

use tenferro_api_grammar_proof::p5_error_causes::{chain_type_names, find_cause};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EagerSessionEinsumExt, TensorEinsumExt, TraceContextEinsumExt};
use tenferro_linalg::{EagerSessionLinalgExt, TensorLinalgExt, TracedTensorLinalgExt};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext, TracedTensor};
use tenferro_tensor::{ErrorKind, Tensor, TensorRead, TypedTensor, ValidationKind};

type LinalgError = tenferro_linalg::Error;
type EinsumError = tenferro_einsum::Error;
type TensorError = tenferro_tensor::Error;
type RuntimeError = tenferro_runtime::Error;

fn mat(rows_cols: [usize; 2], data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(rows_cols.to_vec(), data).unwrap())
}
/// [[1, 2], [2, 1]]: symmetric, indefinite, so Cholesky must fail.
fn indefinite() -> Tensor {
    mat([2, 2], vec![1.0, 2.0, 2.0, 1.0])
}
/// [[1, 2], [2, 4]]: singular.
fn singular() -> Tensor {
    mat([2, 2], vec![1.0, 2.0, 2.0, 4.0])
}
fn rhs() -> Tensor {
    mat([2, 1], vec![1.0, 1.0])
}

/// Compile succeeds or not, then run; returns `(compile_ok, run_result)`.
fn compile_and_run(out: &TracedTensor) -> (bool, tenferro_runtime::Result<Vec<Tensor>>) {
    let mut compiler = GraphCompiler::new();
    let program = match compiler.compile(out) {
        Ok(p) => p,
        Err(e) => return (false, Err(e)),
    };
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuEngine>(
                tenferro_cpu::runtime_engine_id().unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let runtime = builder.build().unwrap();
    let reads: Vec<TensorRead<'_>> = vec![];
    let run = (|| {
        let prepared = runtime.prepare(&program, &reads, &[engine.target()])?;
        let outputs = {
            let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
            runtime.run(&mut engines, &prepared, &reads)?
        };
        outputs
            .into_iter()
            .map(|o| Ok(o.into_tensor(&mut engine)?))
            .collect()
    })();
    (true, run)
}

// ---------------------------------------------------------------- linalg ---

fn assert_linalg_cause(
    err: &(dyn std::error::Error + 'static),
    outer_kind: ErrorKind,
    want: fn(&LinalgError) -> bool,
) {
    let cause = find_cause::<LinalgError>(err).expect("linalg cause must be reachable");
    assert!(want(cause), "unexpected cause {cause:?}");
    // Classification and cause agree for linalg (both NumericalFailure).
    assert_eq!(outer_kind, cause.kind());
    // The cause's message survives in the outer Display too.
    assert!(err.to_string().contains(&cause.to_string()));
}

#[test]
fn linalg_concrete_cholesky_keeps_typed_cause() {
    let mut eng = CpuEngine::serial();
    // Static type pins the concrete layer's outer error.
    let err: TensorError = indefinite().cholesky(&mut eng).unwrap_err();
    assert!(matches!(err, TensorError::Extension { .. }));
    assert_eq!(err.kind(), ErrorKind::NumericalFailure);
    assert_linalg_cause(&err, err.kind(), |c| {
        matches!(c, LinalgError::NonConvergence { op: "cholesky" })
    });
    // Chain: [tensor::Error::Extension, linalg::Error]
    assert_eq!(chain_type_names(&err).len(), 2);
    // Finding: a non-positive-definite input is reported as NonConvergence,
    // not as a positive-definiteness/Singular failure.
}

#[test]
fn linalg_concrete_solve_singular_keeps_typed_cause() {
    let mut eng = CpuEngine::serial();
    let err: TensorError = singular().solve(&rhs(), &mut eng).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NumericalFailure);
    assert_linalg_cause(&err, err.kind(), |c| {
        matches!(c, LinalgError::Singular { op: "solve" })
    });
}

#[test]
fn linalg_traced_failure_is_deferred_to_run_and_keeps_cause() {
    let trace = TraceContext::new();
    let a = trace.from_tensor_concrete_shape(indefinite()).unwrap();
    let f = a.cholesky().expect("build never sees numerical data");
    let (compiled, run) = compile_and_run(&f);
    assert!(compiled, "numerical failure must not surface at compile");
    let err: RuntimeError = run.unwrap_err();
    // Outer variant: runtime::Error::TensorRuntime(tensor::Error::Extension)
    let RuntimeError::TensorRuntime(inner) = &err else {
        panic!("unexpected outer variant {err:?}");
    };
    assert!(matches!(inner, TensorError::Extension { .. }));
    assert_eq!(err.kind(), ErrorKind::NumericalFailure);
    assert_linalg_cause(&err, err.kind(), |c| {
        matches!(c, LinalgError::NonConvergence { op: "cholesky" })
    });
    // `TensorRuntime` is #[error(transparent)]: its source() skips the wrapped
    // tensor::Error, so downcasting the chain to tensor::Error fails even
    // though one is wrapped inside. Only pattern matching reaches it.
    assert!(find_cause::<TensorError>(&err).is_none());
    assert_eq!(chain_type_names(&err).len(), 2);
}

#[test]
fn linalg_traced_solve_singular_keeps_cause() {
    let trace = TraceContext::new();
    let a = trace.from_tensor_concrete_shape(singular()).unwrap();
    let b = trace.from_tensor_concrete_shape(rhs()).unwrap();
    let x = a.solve(&b).unwrap();
    let err = compile_and_run(&x).1.unwrap_err();
    // Finding: same failure, but the cause names the lowered kernel
    // ("lu_factor_solve"), not the public op ("solve") the concrete layer reports.
    assert_linalg_cause(&err, err.kind(), |c| {
        matches!(
            c,
            LinalgError::Singular {
                op: "lu_factor_solve"
            }
        )
    });
}

#[test]
fn linalg_eager_failure_is_immediate_and_keeps_cause() {
    let ctx = tenferro_ad::EagerRuntime::new().unwrap();
    let err: tenferro_ad::Error = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            let x = s.constant_from(indefinite())?;
            s.cholesky(&x)
        })
        .map(|_| ())
        .unwrap_err();
    // tenferro_ad::Error is a re-export of tenferro_runtime::Error: one type,
    // not one type per layer.
    assert_eq!(
        TypeId::of::<tenferro_ad::Error>(),
        TypeId::of::<RuntimeError>()
    );
    assert!(matches!(err, RuntimeError::TensorRuntime(_)));
    assert_eq!(err.kind(), ErrorKind::NumericalFailure);
    assert_linalg_cause(&err, err.kind(), |c| {
        matches!(c, LinalgError::NonConvergence { op: "cholesky" })
    });
    assert!(find_cause::<TensorError>(&err).is_none());
}

// ---------------------------------------------------------------- einsum ---

const NOT_ONE_ARROW: &str = "ab,bc->ac->x";
const MISSING_OUTPUT_LABEL: &str = "ab,bc->zz";
const RANK_MISMATCH: &str = "ab,bcd->ac";

fn eager_einsum(spec: &str) -> Result<(), EinsumError> {
    let ctx = tenferro_ad::EagerRuntime::new().unwrap();
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        let x = s.constant_from(indefinite())?;
        s.einsum(&[&x, &x], spec).map(|_| ())
    })
}

fn traced_einsum(spec: &str) -> Result<(), EinsumError> {
    let mut trace = TraceContext::new();
    let x = trace.from_tensor_concrete_shape(indefinite()).unwrap();
    trace
        .einsum(&[x.trace_value(), x.trace_value()], spec)
        .map(|_| ())
}

fn concrete_einsum(spec: &str) -> Result<Tensor, EinsumError> {
    let a = indefinite();
    let mut eng = CpuEngine::serial();
    [&a, &a].einsum(spec, &mut eng)
}

#[test]
fn einsum_parse_error_is_einsum_error_in_all_three_layers() {
    // Outer error type is `tenferro_einsum::Error` in every layer (pinned by the
    // annotated signatures above), not tensor::Error / runtime::Error / ad::Error.
    for (layer, err) in [
        ("concrete", concrete_einsum(NOT_ONE_ARROW).unwrap_err()),
        ("traced", traced_einsum(NOT_ONE_ARROW).unwrap_err()),
        ("eager", eager_einsum(NOT_ONE_ARROW).unwrap_err()),
    ] {
        assert!(
            matches!(err, EinsumError::InvalidSubscripts { .. }),
            "{layer}: {err:?}"
        );
        assert_eq!(
            err.kind(),
            ErrorKind::Validation(ValidationKind::InvalidArgument)
        );
        // The cause IS the outer error: find_cause succeeds only by matching the
        // top of the chain; there is no deeper source.
        assert!(find_cause::<EinsumError>(&err).is_some());
        assert_eq!(chain_type_names(&err).len(), 1, "{layer}");
        assert!(find_cause::<TensorError>(&err).is_none());
    }
}

#[test]
fn einsum_missing_output_label_differs_between_layers() {
    // concrete and eager: parse-level InvalidSubscripts.
    for err in [
        concrete_einsum(MISSING_OUTPUT_LABEL).unwrap_err(),
        eager_einsum(MISSING_OUTPUT_LABEL).unwrap_err(),
    ] {
        assert!(
            matches!(err, EinsumError::InvalidSubscripts { .. }),
            "{err:?}"
        );
    }
    // traced: same bad spec is NOT rejected by the parser; it surfaces during
    // graph build as einsum::Error::Runtime(runtime::Error::Extension{..}) whose
    // cause chain ends in a ValidationError::InvalidArgument.
    let err = traced_einsum(MISSING_OUTPUT_LABEL).unwrap_err();
    let EinsumError::Runtime(rt) = &err else {
        panic!("{err:?}")
    };
    assert!(matches!(rt, RuntimeError::Extension { .. }), "{rt:?}");
    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    let v = find_cause::<tenferro_tensor::ValidationError>(&err)
        .expect("validation payload reachable through the runtime wrapper");
    assert!(matches!(
        v,
        tenferro_tensor::ValidationError::InvalidArgument { .. }
    ));
    // The wrapped tensor::Error is skipped by a transparent wrapper again.
    assert!(find_cause::<TensorError>(&err).is_none());
    assert!(chain_type_names(&err).len() >= 3);
}

#[test]
fn einsum_rank_mismatch_kind_and_cause_disagree_in_traced_layer() {
    let c = concrete_einsum(RANK_MISMATCH).unwrap_err();
    let t = traced_einsum(RANK_MISMATCH).unwrap_err();
    let e = eager_einsum(RANK_MISMATCH).unwrap_err();
    assert_eq!(
        c.kind(),
        ErrorKind::Validation(ValidationKind::RankMismatch)
    );
    assert_eq!(e.kind(), c.kind());
    // Traced: the cause is the same RankMismatch payload...
    let v = find_cause::<tenferro_tensor::ValidationError>(&t).unwrap();
    assert!(matches!(
        v,
        tenferro_tensor::ValidationError::RankMismatch { .. }
    ));
    assert_eq!(v.kind(), ValidationKind::RankMismatch);
    // ...but the layer's classification says InvalidArgument. kind and cause
    // diverge: exactly the hazard the proposal says tests must catch (rule 5).
    assert_eq!(
        t.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert_ne!(t.kind(), ErrorKind::Validation(v.kind()));
}

#[test]
fn einsum_valid_spec_succeeds_in_all_layers() {
    // Control: the harness itself works (2x2 @ 2x2).
    assert!(concrete_einsum("ab,bc->ac").is_ok());
    assert!(traced_einsum("ab,bc->ac").is_ok());
    assert!(eager_einsum("ab,bc->ac").is_ok());
}
