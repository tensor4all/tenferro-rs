//! The `TraceContext` entry point of the linalg extension family.
//!
//! `TraceContextLinalgExt` records the same semantic operations as the
//! `TracedTensorLinalgExt` method-chaining surface, so the two paths must
//! produce the same program for the same inputs.

use tenferro_linalg::{Lu, Svd};
use tenferro_linalg::{TraceContextLinalgExt, TracedTensorLinalgExt};
use tenferro_runtime::{TraceContext, TracedTensor};

fn trace_with_matrix() -> (TraceContext, TracedTensor) {
    let trace = TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])
        .unwrap();
    (trace, a)
}

fn fingerprint(context: &TraceContext, outputs: &[TracedTensor]) -> Vec<u8> {
    let values: Vec<_> = outputs.iter().map(TracedTensor::trace_value).collect();
    context
        .finish(&values)
        .unwrap()
        .program()
        .semantic_fingerprint()
        .as_bytes()
        .to_vec()
}

#[test]
fn context_trait_records_exactly_one_decomposition() {
    let (trace, a) = trace_with_matrix();
    let (u, s, vt) = trace.svd(a.trace_value()).unwrap();
    let graph = trace
        .finish(&[u.trace_value(), s.trace_value(), vt.trace_value()])
        .unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}

#[test]
fn context_and_traced_tensor_agree_on_a_direct_decomposition() {
    let (trace, a) = trace_with_matrix();
    let (cu, cs, cvt) = trace.svd(a.trace_value()).unwrap();
    let context_path = fingerprint(&trace, &[cu, cs, cvt]);
    let Svd {
        u: tu,
        s: ts,
        vh: tvt,
    } = a.svd().unwrap();
    let traced_path = fingerprint(&trace, &[tu, ts, tvt]);
    assert_eq!(context_path, traced_path);
}

#[test]
fn context_and_traced_tensor_agree_on_a_composite() {
    // `pinv` composes `svd` with standard operations, so this covers the
    // composite bodies as well as the extension op itself.
    let (trace, a) = trace_with_matrix();
    let context_path = {
        let inverse = trace.pinv(a.trace_value()).unwrap();
        fingerprint(&trace, &[inverse])
    };
    let traced_path = {
        let inverse = a.pinv().unwrap();
        fingerprint(&trace, &[inverse])
    };
    assert_eq!(context_path, traced_path);
}

#[test]
fn context_trait_solves_against_a_second_value() {
    let trace = TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0])
        .unwrap();
    let b = trace
        .from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0])
        .unwrap();
    let x = trace.solve(a.trace_value(), b.trace_value()).unwrap();
    let traced = a.solve(&b).unwrap();
    assert_eq!(fingerprint(&trace, &[x]), fingerprint(&trace, &[traced]));
}

#[test]
fn context_trait_rejects_a_value_from_another_context() {
    let (trace, a) = trace_with_matrix();
    let other = TraceContext::new();
    let b = other
        .from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])
        .unwrap();
    let error = trace
        .solve(a.trace_value(), b.trace_value())
        .expect_err("a value from another context is a typed error");
    assert!(
        matches!(error, tenferro_runtime::Error::RuntimeStateSource { .. }),
        "got {error:?}"
    );
}

#[test]
fn context_norm_with_no_axes_returns_the_same_value() {
    let (trace, a) = trace_with_matrix();
    let same = trace.norm(a.trace_value(), None, Some(&[]), false).unwrap();
    assert_eq!(same.rank(), 2);
    // The identity case returns the same value, but a value handle cannot carry
    // the traced tensor's local attached-value cache.
    assert!(a.attached_value().is_some());
    assert!(same.attached_value().is_none());
}

#[test]
fn context_and_traced_tensor_agree_on_a_symbolic_input() {
    let trace = TraceContext::new();
    let a = trace
        .input_symbolic_shape(tenferro_runtime::DType::F64, 2)
        .unwrap();
    let context_path = {
        let (p, l, u, parity) = trace.lu(a.trace_value()).unwrap();
        fingerprint(&trace, &[p, l, u, parity])
    };
    let traced_path = {
        let Lu { p, l, u, parity } = a.lu().unwrap();
        fingerprint(&trace, &[p, l, u, parity])
    };
    assert_eq!(context_path, traced_path);
}
