//! Recording linalg operations on traced tensors built from a `TraceContext`.
//!
//! The operations of `TracedLinalgOps` record graph nodes against the context
//! that owns their operands; these tests pin what a call records.

use tenferro_linalg::{NormOptions, Svd, TracedLinalgOps};
use tenferro_runtime::{TraceContext, TracedTensor};
use tenferro_tensor::AxisVec;

fn trace_with_matrix() -> (TraceContext, TracedTensor) {
    let trace = TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])
        .unwrap();
    (trace, a)
}

#[test]
fn traced_svd_records_exactly_one_decomposition() {
    let (trace, a) = trace_with_matrix();
    let Svd { u, s, vh } = a.svd().unwrap();
    let graph = trace
        .finish(&[u.trace_value(), s.trace_value(), vh.trace_value()])
        .unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}

#[test]
fn traced_solve_records_exactly_one_operation() {
    let trace = TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0])
        .unwrap();
    let b = trace
        .from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0])
        .unwrap();
    let x = a.solve(&b).unwrap();
    let graph = trace.finish(&[x.trace_value()]).unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}

#[test]
fn traced_solve_rejects_a_value_from_another_context() {
    let (_trace, a) = trace_with_matrix();
    let other = TraceContext::new();
    let b = other
        .from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])
        .unwrap();
    let error = a
        .solve(&b)
        .expect_err("a value from another context is a typed error");
    assert!(
        matches!(error, tenferro_runtime::Error::RuntimeStateSource { .. }),
        "got {error:?}"
    );
}

#[test]
fn traced_norm_with_no_axes_returns_the_same_value() {
    let (_trace, a) = trace_with_matrix();
    let same = a
        .norm_with(&NormOptions::default().axes(Some([].into_iter().collect::<AxisVec>())))
        .unwrap();
    assert_eq!(same.rank(), 2);
}

#[test]
fn traced_lu_records_on_a_symbolic_input() {
    let trace = TraceContext::new();
    let a = trace
        .input_symbolic_shape(tenferro_runtime::DType::F64, 2)
        .unwrap();
    let lu = a.lu().unwrap();
    let graph = trace
        .finish(&[
            lu.p.trace_value(),
            lu.l.trace_value(),
            lu.u.trace_value(),
            lu.parity.trace_value(),
        ])
        .unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}
