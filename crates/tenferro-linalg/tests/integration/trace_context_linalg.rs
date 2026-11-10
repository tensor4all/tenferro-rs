//! The `TracedTensorLinalgExt` entry point on tensors of one `TraceContext`.
//!
//! Each operation records its extension op on the context the tensor belongs
//! to; inputs from another context are a typed error.

use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_linalg::{Lu, NormOptions, Svd};
use tenferro_runtime::{TraceContext, TracedTensor};

fn trace_with_matrix() -> (TraceContext, TracedTensor) {
    let trace = TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])
        .unwrap();
    (trace, a)
}

#[test]
fn traced_tensor_records_exactly_one_decomposition() {
    let (trace, a) = trace_with_matrix();
    let Svd { u, s, vh: vt } = a.svd().unwrap();
    let graph = trace
        .finish(&[u.trace_value(), s.trace_value(), vt.trace_value()])
        .unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}

#[test]
fn traced_tensor_solves_against_a_second_value() {
    let trace = TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0])
        .unwrap();
    let b = trace
        .from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0])
        .unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.rank(), 2);
    let graph = trace.finish(&[x.trace_value()]).unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}

#[test]
fn traced_tensor_rejects_a_value_from_another_context() {
    let (_trace, a) = trace_with_matrix();
    let other = TraceContext::new();
    let b = other
        .from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])
        .unwrap();
    let error = a
        .solve(&b)
        .expect_err("a value from another context is a typed error");
    // `TracedTensor` operands from two contexts are refused while the graph is built.
    assert!(
        matches!(
            error,
            tenferro_runtime::Error::Validation {
                phase: tenferro_runtime::ErrorPhase::GraphBuild,
                ..
            }
        ),
        "got {error:?}"
    );
}

#[test]
fn norm_with_no_axes_returns_the_same_value() {
    let (_trace, a) = trace_with_matrix();
    let same = a
        .norm_with(&NormOptions::default().with_axes(Vec::<isize>::new()))
        .unwrap();
    assert_eq!(same.rank(), 2);
}

#[test]
fn traced_tensor_records_lu_on_a_symbolic_input() {
    let trace = TraceContext::new();
    let a = trace
        .input_symbolic_shape(tenferro_runtime::DType::F64, 2)
        .unwrap();
    let Lu { p, l, u, parity } = a.lu().unwrap();
    let graph = trace
        .finish(&[
            p.trace_value(),
            l.trace_value(),
            u.trace_value(),
            parity.trace_value(),
        ])
        .unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}
