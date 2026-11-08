//! The `TraceContext` entry point of the FFT extension family.
//!
//! `TraceContextFftExt` records the same semantic operations as the
//! `TracedTensorFftExt` method-chaining surface, so the two paths must produce
//! the same program for the same input.

use num_complex::Complex64;
use tenferro_fft::{FftNorm, TraceContextFftExt, TracedTensorFftExt};
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

fn fingerprint(context: &TraceContext, output: &TracedTensor) -> Vec<u8> {
    context
        .finish(&[output.trace_value()])
        .unwrap()
        .program()
        .semantic_fingerprint()
        .as_bytes()
        .to_vec()
}

#[test]
fn context_trait_records_exactly_one_transform() {
    let (trace, x) = trace_with_spectrum();
    let spectrum = trace
        .fft(x.trace_value(), None, -1, FftNorm::Backward)
        .unwrap();
    let graph = trace.finish(&[spectrum.trace_value()]).unwrap();
    assert_eq!(graph.program().operations().count(), 1);
}

#[test]
fn context_and_traced_tensor_agree_on_a_one_axis_transform() {
    let (trace, x) = trace_with_spectrum();
    let context_path = {
        let spectrum = trace
            .fft(x.trace_value(), None, -1, FftNorm::Backward)
            .unwrap();
        fingerprint(&trace, &spectrum)
    };
    let traced_path = {
        let spectrum = x.fft(None, -1, FftNorm::Backward).unwrap();
        fingerprint(&trace, &spectrum)
    };
    assert_eq!(context_path, traced_path);
}

#[test]
fn context_and_traced_tensor_agree_on_a_n_dimensional_transform() {
    let (trace, x) = trace_with_spectrum();
    let context_path = {
        let spectrum = trace
            .fftn(x.trace_value(), None, &[0, 1], FftNorm::Backward)
            .unwrap();
        fingerprint(&trace, &spectrum)
    };
    let traced_path = {
        let spectrum = x.fftn(None, &[0, 1], FftNorm::Backward).unwrap();
        fingerprint(&trace, &spectrum)
    };
    assert_eq!(context_path, traced_path);
}
