use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftNorm, TensorFftExt, TensorReadFftExt};
use tenferro_tensor::{Tensor, TensorRead};

fn forward_scale(norm: FftNorm, n: usize) -> f64 {
    match norm {
        FftNorm::Backward => 1.0,
        FftNorm::Forward => 1.0 / n as f64,
        FftNorm::Ortho => 1.0 / (n as f64).sqrt(),
    }
}

// A direct DFT is deliberately independent of realfft and the CPU half-spectrum
// completion code. These fixtures are small and cover every output bin.
fn direct_dft(input: &[f64], shape: &[usize], axis: usize, norm: FftNorm) -> Vec<Complex64> {
    let n = shape[axis];
    let stride: usize = shape[..axis].iter().product();
    let block = stride * n;
    let scale = forward_scale(norm, n);
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
    for outer in 0..input.len() / block {
        for inner in 0..stride {
            let base = outer * block + inner;
            for k in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in 0..n {
                    let angle = -std::f64::consts::TAU * (j * k) as f64 / n as f64;
                    sum += Complex64::new(angle.cos(), angle.sin()) * input[base + j * stride];
                }
                output[base + k * stride] = sum * scale;
            }
        }
    }
    output
}

fn assert_close(actual: &[Complex64], expected: &[Complex64], tolerance: f64, context: &str) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite()),
        "{context}: nonfinite result"
    );
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm() / (1.0 + expected.norm()))
        .fold(0.0, f64::max);
    assert!(
        maximum <= tolerance,
        "{context}: maximum normalized error {maximum:e}"
    );
}

macro_rules! real_fft_regression {
    ($name:ident, $real:ty, $complex:ty, $tolerance:expr, $forward:expr) => {
        #[test]
        fn $name() {
            for threads in [1, 4] {
                let mut engine = if threads == 1 {
                    CpuEngine::serial()
                } else {
                    CpuEngine::new(
                        tenferro_cpu::Pool::builder()
                            .threads(threads).build().unwrap(),
                    )
                };
                assert_eq!(engine.num_threads(), threads);
                for n in [1, 2, 3, 7, 8, 9] {
                    for (shape, axis) in [(vec![n], 0), (vec![2, n, 3], 1), (vec![2, 3, n], 2)] {
                        let count: usize = shape.iter().product();
                        let values: Vec<$real> = (0..count)
                            .map(|index| (0.25 + index as f64 * 0.375 - (index % 3) as f64 * 0.125) as $real)
                            .collect();
                        let reference_input: Vec<f64> = values.iter().map(|&value| value as f64).collect();
                        let input = Tensor::from_vec_col_major(shape.clone(), values).unwrap();
                        for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
                            let reference = direct_dft(&reference_input, &shape, axis, norm);
                            let context = format!("threads={threads} n={n} shape={shape:?} axis={axis} norm={norm:?}");
                            if $forward {
                                for borrowed in [false, true] {
                                    let full = if borrowed {
                                        TensorRead::from_tensor(&input)
                                            .fft_read(None, axis as isize, norm, &mut engine)
                                    } else {
                                        input.fft(None, axis as isize, norm, &mut engine)
                                    }
                                    .unwrap();
                                    assert_eq!(full.shape(), shape.as_slice());
                                    let actual: Vec<_> = full.as_slice::<$complex>().unwrap().iter()
                                        .map(|value| Complex64::new(value.re as f64, value.im as f64))
                                        .collect();
                                    assert_close(&actual, &reference, $tolerance, &context);
                                }
                                continue;
                            }

                            let half = n / 2 + 1;
                            let stride: usize = shape[..axis].iter().product();
                            let mut spectrum = Vec::new();
                            for chunk in reference.chunks_exact(stride * n) {
                                for value in &chunk[..stride * half] {
                                    spectrum.push(<$complex>::new(value.re as $real, value.im as $real));
                                }
                            }
                            // DC is real for every length. Only even lengths
                            // have a Nyquist bin whose imaginary part is ignored.
                            for chunk in spectrum.chunks_exact_mut(stride * half) {
                                for value in &mut chunk[..stride] {
                                    value.im = 7.0;
                                }
                                if n.is_multiple_of(2) {
                                    for value in &mut chunk[stride * (half - 1)..] {
                                        value.im = 11.0;
                                    }
                                }
                            }
                            let mut spectrum_shape = shape.clone();
                            spectrum_shape[axis] = half;
                            let spectrum = Tensor::from_vec_col_major(spectrum_shape, spectrum).unwrap();
                            for borrowed in [false, true] {
                                let inverse = if borrowed {
                                    TensorRead::from_tensor(&spectrum)
                                        .irfft_read(Some(n), axis as isize, norm, &mut engine)
                                } else {
                                    spectrum.irfft(Some(n), axis as isize, norm, &mut engine)
                                }
                                .unwrap();
                                assert_eq!(inverse.shape(), shape.as_slice());
                                let actual: Vec<_> = inverse.as_slice::<$real>().unwrap().iter()
                                    .map(|&value| Complex64::new(value as f64, 0.0)).collect();
                                let expected: Vec<_> = reference_input.iter().map(|&value| Complex64::new(value, 0.0)).collect();
                                assert_close(&actual, &expected, $tolerance, &context);
                            }
                        }
                    }
                }
            }
        }
    };
}

real_fft_regression!(
    real_fft_matches_independent_dft_f32,
    f32,
    num_complex::Complex32,
    1.0e-5,
    true
);
real_fft_regression!(
    real_fft_matches_independent_dft_f64,
    f64,
    Complex64,
    1.0e-11,
    true
);
real_fft_regression!(
    real_irfft_matches_independent_dft_f32,
    f32,
    num_complex::Complex32,
    1.0e-5,
    false
);
real_fft_regression!(
    real_irfft_matches_independent_dft_f64,
    f64,
    Complex64,
    1.0e-11,
    false
);
