//! Linalg numerical failures keep their typed cause reachable through
//! `Error::source()` in the concrete, traced and eager layers (#2552 §5.5).

use std::error::Error as StdError;

use tenferro_cpu::CpuEngine;
use tenferro_linalg::{LinalgOps, TracedLinalgOps};
use tenferro_runtime::{GraphCompiler, TraceContext, TracedTensor};
use tenferro_tensor::{ErrorKind, Tensor, TypedTensor};

use crate::support;

type LinalgError = tenferro_linalg::Error;
type TensorError = tenferro_tensor::Error;
type RuntimeError = tenferro_runtime::Error;

fn mat(shape: [usize; 2], data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape.to_vec(), data).unwrap())
}

/// Symmetric indefinite, so Cholesky fails.
fn indefinite() -> Tensor {
    mat([2, 2], vec![1.0, 2.0, 2.0, 1.0])
}

/// Singular matrix.
fn singular() -> Tensor {
    mat([2, 2], vec![1.0, 2.0, 2.0, 4.0])
}

fn rhs() -> Tensor {
    mat([2, 1], vec![1.0, 1.0])
}

/// Walk `err` and its `source()` chain (starting at `err`) for a value of type `E`.
fn find_cause<'a, E: StdError + 'static>(err: &'a (dyn StdError + 'static)) -> Option<&'a E> {
    let mut cur: Option<&'a (dyn StdError + 'static)> = Some(err);
    while let Some(e) = cur {
        if let Some(found) = e.downcast_ref::<E>() {
            return Some(found);
        }
        cur = e.source();
    }
    None
}

fn run_traced(out: &TracedTensor) -> RuntimeError {
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile(out)
        .expect("a numerical failure must not surface at compile time");
    support::run_all(&program, &[]).expect_err("the traced run must fail")
}

fn assert_concrete(err: TensorError) {
    assert_eq!(err.kind(), ErrorKind::NumericalFailure);
    let cause = find_cause::<LinalgError>(&err).expect("linalg cause reachable");
    assert_eq!(err.kind(), cause.kind());
}

fn assert_runtime(err: RuntimeError) {
    assert_eq!(err.kind(), ErrorKind::NumericalFailure);
    let cause = find_cause::<LinalgError>(&err).expect("linalg cause reachable");
    assert_eq!(err.kind(), cause.kind());
    // `TensorRuntime` no longer hides the wrapped tensor error from `source()`.
    let tensor = find_cause::<TensorError>(&err).expect("tensor error reachable");
    assert!(matches!(tensor, TensorError::Extension { .. }));
    assert_eq!(tensor.kind(), cause.kind());
}

#[test]
fn concrete_cholesky_indefinite_keeps_linalg_cause() {
    let mut eng = CpuEngine::serial();
    let err: TensorError = indefinite().cholesky(&mut eng).unwrap_err();
    assert_concrete(err);
}

#[test]
fn concrete_solve_singular_keeps_linalg_cause() {
    let mut eng = CpuEngine::serial();
    let err: TensorError = singular().solve(&rhs(), &mut eng).unwrap_err();
    assert_concrete(err);
}

#[test]
fn traced_cholesky_indefinite_keeps_linalg_cause() {
    let trace = TraceContext::new();
    let a = trace.from_tensor_concrete_shape(indefinite()).unwrap();
    let f = a.cholesky().unwrap();
    assert_runtime(run_traced(&f));
}

#[test]
fn traced_solve_singular_keeps_linalg_cause() {
    let trace = TraceContext::new();
    let a = trace.from_tensor_concrete_shape(singular()).unwrap();
    let b = trace.from_tensor_concrete_shape(rhs()).unwrap();
    let x = a.solve(&b).unwrap();
    assert_runtime(run_traced(&x));
}

#[cfg(feature = "autodiff")]
mod eager {
    use super::*;
    use tenferro_linalg::EagerLinalgOps;

    #[test]
    fn eager_cholesky_indefinite_keeps_linalg_cause() {
        let ctx = tenferro_ad::EagerRuntime::new().unwrap();
        let err: tenferro_ad::Error = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                let x = s.constant_from(indefinite())?;
                s.cholesky(&x)
            })
            .map(|_| ())
            .unwrap_err();
        assert_runtime(err);
    }

    #[test]
    fn eager_solve_singular_keeps_linalg_cause() {
        let ctx = tenferro_ad::EagerRuntime::new().unwrap();
        let err: tenferro_ad::Error = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                let a = s.constant_from(singular())?;
                let b = s.constant_from(rhs())?;
                s.solve(&a, &b)
            })
            .map(|_| ())
            .unwrap_err();
        assert_runtime(err);
    }
}
