//! A traced FFT records exactly one extension node in its `TraceContext`.
//!
//! The former `TraceContext` receiver of FFTs was removed (#2552 decision 4);
//! `TracedFftOps` on a `TracedTensor` is the only traced spelling.

use num_complex::Complex64;
use tenferro_fft::{FftnOptions, TracedFftOps};
use tenferro_runtime::{TraceContext, TracedTensor};

fn trace_with_spectrum() -> (TraceContext, TracedTensor) {
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
    (trace, x)
}

#[test]
fn a_traced_one_axis_transform_records_exactly_one_node() {
    let (trace, x) = trace_with_spectrum();
    let spectrum = x.fft().unwrap();
    let graph = trace.finish(&[spectrum.trace_value()]).unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}

#[test]
fn a_traced_n_dimensional_transform_records_exactly_one_node() {
    let (trace, x) = trace_with_spectrum();
    let spectrum = x
        .fftn_with(&FftnOptions::default().axes(vec![0, 1]))
        .unwrap();
    let graph = trace.finish(&[spectrum.trace_value()]).unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}
