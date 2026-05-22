use num_complex::Complex64;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_fft::{fft, ifft, irfft, rfft, FftNorm};

fn run(output: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).unwrap();
    GraphExecutor::new(CpuBackend::new()).run(&program).unwrap()
}

fn assert_c64_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.re - e.re).abs() < 1e-10 && (a.im - e.im).abs() < 1e-10,
            "idx {idx}: actual={a:?}, expected={e:?}"
        );
    }
}

fn assert_f64_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < 1e-10, "idx {idx}: actual={a}, expected={e}");
    }
}

#[test]
fn fft_c64_matches_numpy_convention() {
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    );
    let y = fft(&x, None, -1, FftNorm::Backward);
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    );
}

#[test]
fn ifft_c64_applies_backward_normalization() {
    let spectrum = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    );
    let y = ifft(&spectrum, None, -1, FftNorm::Backward);
    let out = run(&y);

    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    );
}

#[test]
fn rfft_f64_returns_onesided_spectrum() {
    let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let y = rfft(&x, None, -1, FftNorm::Backward);
    let out = run(&y);

    assert_eq!(out.shape(), &[3]);
    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    );
}

#[test]
fn irfft_c64_reconstructs_real_signal() {
    let spectrum = TracedTensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    );
    let y = irfft(&spectrum, Some(4), -1, FftNorm::Backward);
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_f64_close(out.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn fft_c64_jvp_applies_fft_to_tangent() {
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    );
    let dx = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    );

    let y = fft(&x, None, -1, FftNorm::Backward);
    let dy = y.jvp(&x, &dx);
    let out = run(&dy);

    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 2.0),
        ],
    );
}
