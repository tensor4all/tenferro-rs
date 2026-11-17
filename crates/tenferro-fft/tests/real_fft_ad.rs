//! AD of the real FFTs (#2644): `rfft`, `irfft`, `rfftn`, `irfftn` and `fft` of a
//! real input, checked against central finite differences on every real and
//! imaginary coordinate and against the same map written with the c2c rules.
//!
//! Cotangents follow `docs/guides/complex-ad.md`: the pairing is
//! `Re(conj(a) * b)`, so the VJP of a map `f` at cotangent `g` is the gradient
//! `dL/dRe(x) + i dL/dIm(x)` of `L(x) = Re(sum(conj(g) * f(x)))`.
#![cfg(feature = "autodiff")]

mod support;

use num_complex::Complex64;
use tenferro_ad::{AdContext, EagerRuntime};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftOps, FftNorm, FftOptions, FftnOptions, TracedFftOps};
use tenferro_runtime::{DType, GraphCompiler, Tensor, TraceContext, TracedTensor};
use tenferro_tensor::{PadConfig, SliceConfig};

const NORMS: [FftNorm; 3] = [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho];
/// Central-difference step. Every map checked here is linear or quadratic, so a
/// central difference has no truncation error and only rounding remains.
const STEP: f64 = 1.0e-4;
const FINITE_DIFFERENCE_TOLERANCE: f64 = 1.0e-8;
/// Two evaluations of the same linear map through different FFT rules.
const REFERENCE_TOLERANCE: f64 = 1.0e-10;

fn ad() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(tenferro_fft::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap()
}

fn run(output: &TracedTensor) -> Tensor {
    let program = GraphCompiler::new().compile(output).unwrap();
    support::run_one(&program, &[]).unwrap()
}

/// Values of an `F64` or `C64` tensor, real values widened to complex.
fn values(tensor: &Tensor) -> Vec<Complex64> {
    match tensor.dtype() {
        DType::F64 => tensor
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        DType::C64 => tensor.as_slice::<Complex64>().unwrap().to_vec(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// Deterministic, irregular test values; real when `complex` is false.
fn grid(len: usize, seed: usize, complex: bool) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let t = 0.37 * index as f64 + 1.1 * seed as f64;
            let imaginary = if complex { (1.7 * t).cos() } else { 0.0 };
            Complex64::new(t.sin() + 0.25, imaginary)
        })
        .collect()
}

fn traced(
    trace: &TraceContext,
    shape: &[usize],
    complex: bool,
    data: &[Complex64],
) -> TracedTensor {
    if complex {
        trace.from_vec_col_major(shape.to_vec(), data.to_vec())
    } else {
        let real: Vec<f64> = data.iter().map(|value| value.re).collect();
        trace.from_vec_col_major(shape.to_vec(), real)
    }
    .unwrap()
}

fn assert_close(actual: &[Complex64], expected: &[Complex64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() <= tolerance,
            "{what}: index {index}: actual={actual}, expected={expected}"
        );
    }
}

/// A real-linear map of one traced input.
struct LinearCase<'a> {
    label: String,
    shape: Vec<usize>,
    complex_input: bool,
    map: &'a dyn Fn(&TracedTensor) -> TracedTensor,
}

impl LinearCase<'_> {
    fn input_len(&self) -> usize {
        self.shape.iter().product()
    }

    fn evaluate(&self, data: &[Complex64]) -> Vec<Complex64> {
        let trace = TraceContext::new();
        let input = traced(&trace, &self.shape, self.complex_input, data);
        values(&run(&(self.map)(&input)))
    }
}

/// Primal output, JVP and VJP of one map at fixed input, tangent and cotangent.
struct Derivatives {
    output: Vec<Complex64>,
    cotangent: Vec<Complex64>,
    jvp: Vec<Complex64>,
    vjp: Vec<Complex64>,
}

fn derivatives(case: &LinearCase<'_>) -> Derivatives {
    let len = case.input_len();
    let trace = TraceContext::new();
    let x = traced(
        &trace,
        &case.shape,
        case.complex_input,
        &grid(len, 1, case.complex_input),
    );
    let dx = traced(
        &trace,
        &case.shape,
        case.complex_input,
        &grid(len, 2, case.complex_input),
    );
    let y = (case.map)(&x);
    let output = run(&y);
    let output_complex = output.dtype() == DType::C64;
    let cotangent = grid(output.shape().iter().product(), 3, output_complex);
    let g = traced(&trace, output.shape(), output_complex, &cotangent);

    let ad = ad();
    let jvp = run(&ad.jvp(&y, &x, &dx).unwrap());
    assert_eq!(jvp.dtype(), output.dtype(), "{}: jvp dtype", case.label);
    assert_eq!(jvp.shape(), output.shape(), "{}: jvp shape", case.label);
    let vjp = run(&ad.vjp(&y, &x, &g).unwrap());
    let input_dtype = if case.complex_input {
        DType::C64
    } else {
        DType::F64
    };
    assert_eq!(vjp.dtype(), input_dtype, "{}: vjp dtype", case.label);
    assert_eq!(vjp.shape(), &case.shape[..], "{}: vjp shape", case.label);
    Derivatives {
        output: values(&output),
        cotangent,
        jvp: values(&jvp),
        vjp: values(&vjp),
    }
}

/// The real coordinates of one input element: the real part, and the imaginary
/// part for a complex input.
fn coordinate_units(complex: bool) -> &'static [Complex64] {
    const UNITS: [Complex64; 2] = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
    if complex {
        &UNITS
    } else {
        &UNITS[..1]
    }
}

/// Gradient of a real scalar by central differences on every real coordinate.
fn finite_difference_gradient(
    base: &[Complex64],
    complex: bool,
    scalar: impl Fn(&[Complex64]) -> f64,
) -> Vec<Complex64> {
    let mut gradient = vec![Complex64::new(0.0, 0.0); base.len()];
    for index in 0..base.len() {
        for &unit in coordinate_units(complex) {
            let mut plus = base.to_vec();
            plus[index] += unit * STEP;
            let mut minus = base.to_vec();
            minus[index] -= unit * STEP;
            gradient[index] += unit * ((scalar(&plus) - scalar(&minus)) / (2.0 * STEP));
        }
    }
    gradient
}

fn check_finite_differences(case: &LinearCase<'_>) {
    let len = case.input_len();
    let base = grid(len, 1, case.complex_input);
    let tangent = grid(len, 2, case.complex_input);
    let derivatives = derivatives(case);

    let shifted = |sign: f64| {
        let data: Vec<_> = base
            .iter()
            .zip(&tangent)
            .map(|(base, tangent)| base + tangent * (sign * STEP))
            .collect();
        case.evaluate(&data)
    };
    let (plus, minus) = (shifted(1.0), shifted(-1.0));
    let jvp: Vec<_> = plus
        .iter()
        .zip(&minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * STEP))
        .collect();
    assert_close(
        &derivatives.jvp,
        &jvp,
        FINITE_DIFFERENCE_TOLERANCE,
        &format!("{}: jvp versus central differences", case.label),
    );

    let vjp = finite_difference_gradient(&base, case.complex_input, |data| {
        case.evaluate(data)
            .iter()
            .zip(&derivatives.cotangent)
            .map(|(output, cotangent)| (cotangent.conj() * output).re)
            .sum()
    });
    assert_close(
        &derivatives.vjp,
        &vjp,
        FINITE_DIFFERENCE_TOLERANCE,
        &format!("{}: vjp versus central differences", case.label),
    );
}

/// The map and its c2c composite agree on values, JVP and VJP.
fn check_c2c_reference(case: &LinearCase<'_>, reference: &dyn Fn(&TracedTensor) -> TracedTensor) {
    let reference_case = LinearCase {
        label: format!("{} (c2c reference)", case.label),
        shape: case.shape.clone(),
        complex_input: case.complex_input,
        map: reference,
    };
    let actual = derivatives(case);
    let expected = derivatives(&reference_case);
    for (actual, expected, what) in [
        (&actual.output, &expected.output, "value"),
        (&actual.jvp, &expected.jvp, "jvp"),
        (&actual.vjp, &expected.vjp, "vjp"),
    ] {
        assert_close(
            actual,
            expected,
            REFERENCE_TOLERANCE,
            &format!("{}: {what} versus the c2c composite", case.label),
        );
    }
}

fn check(case: LinearCase<'_>, reference: &dyn Fn(&TracedTensor) -> TracedTensor) {
    check_finite_differences(&case);
    check_c2c_reference(&case, reference);
}

/// Keep bins `0..len / 2 + 1` of `axis`: the one-sided part of a full spectrum.
fn one_sided(spectrum: &TracedTensor, axis: usize, len: usize) -> TracedTensor {
    let shape = spectrum.concrete_shape().unwrap();
    let mut limits = shape.clone();
    limits[axis] = len / 2 + 1;
    spectrum
        .slice(SliceConfig {
            starts: vec![0; shape.len()],
            limits,
            strides: vec![1; shape.len()],
        })
        .unwrap()
}

/// Truncate or zero-pad `axis` to `len` with core slice and pad operations.
///
/// The references resize explicitly instead of passing `n` to `ifft`, because
/// the c2c VJP of a zero-padding `ifft` publishes the padded extent for its
/// input cotangent, which an upstream elementwise VJP then rejects.
fn resized(tensor: &TracedTensor, axis: usize, len: usize) -> TracedTensor {
    let shape = tensor.concrete_shape().unwrap();
    let rank = shape.len();
    if len <= shape[axis] {
        let mut limits = shape.clone();
        limits[axis] = len;
        return tensor
            .slice(SliceConfig {
                starts: vec![0; rank],
                limits,
                strides: vec![1; rank],
            })
            .unwrap();
    }
    let mut edge_padding_high = vec![0; rank];
    edge_padding_high[axis] = (len - shape[axis]) as i64;
    tensor
        .pad(PadConfig {
            edge_padding_low: vec![0; rank],
            edge_padding_high,
            interior_padding: vec![0; rank],
        })
        .unwrap()
}

/// Multiply a one-sided spectrum by the Hermitian bin weights of an `irfft` of
/// length `len` along `axis`: 1 at DC and, for even `len`, Nyquist; 2 elsewhere.
/// `irfft(z) == Re(ifft(weights * z zero-padded to len))`, the c2c reference.
fn hermitian_weighted(spectrum: &TracedTensor, axis: usize, len: usize) -> TracedTensor {
    let shape = spectrum.concrete_shape().unwrap();
    let stride: usize = shape[..axis].iter().product();
    let weights: Vec<Complex64> = (0..shape.iter().product::<usize>())
        .map(|flat| {
            let bin = (flat / stride) % shape[axis];
            let self_conjugate = bin == 0 || (len.is_multiple_of(2) && bin == len / 2);
            Complex64::new(if self_conjugate { 1.0 } else { 2.0 }, 0.0)
        })
        .collect();
    let weights = spectrum
        .context()
        .from_vec_col_major(shape.clone(), weights)
        .unwrap();
    resized(&spectrum.mul(&weights).unwrap(), axis, len)
}

#[test]
fn rfft_derivatives_match_finite_differences_and_the_c2c_fft() {
    // (input extent, n): even and odd lengths, truncation, zero padding, and the
    // lengths one and two that have no mirrored bin.
    let lengths = [
        (4, None),
        (5, None),
        (6, Some(4)),
        (5, Some(3)),
        (4, Some(7)),
        (3, Some(6)),
        (1, None),
        (3, Some(2)),
    ];
    for norm in NORMS {
        for (extent, n) in lengths {
            let len = n.unwrap_or(extent);
            // A batch axis first, so the transform runs over strided lanes.
            let options = FftOptions::default().n(n).axis(1).norm(norm);
            let map = |x: &TracedTensor| x.rfft_with(&options).unwrap();
            let reference = |x: &TracedTensor| {
                let full = x.convert(DType::C64).unwrap().fft_with(&options).unwrap();
                one_sided(&full, 1, len)
            };
            let case = LinearCase {
                label: format!("rfft extent {extent} n {n:?} {norm:?}"),
                shape: vec![2, extent],
                complex_input: false,
                map: &map,
            };
            check(case, &reference);
        }
    }
}

#[test]
fn rfft_derivatives_along_the_leading_axis() {
    for norm in NORMS {
        for (extent, n) in [(5, None), (4, Some(6)), (6, Some(3))] {
            let len = n.unwrap_or(extent);
            let options = FftOptions::default().n(n).axis(0).norm(norm);
            let map = |x: &TracedTensor| x.rfft_with(&options).unwrap();
            let reference = |x: &TracedTensor| {
                let full = x.convert(DType::C64).unwrap().fft_with(&options).unwrap();
                one_sided(&full, 0, len)
            };
            let case = LinearCase {
                label: format!("rfft axis 0 extent {extent} n {n:?} {norm:?}"),
                shape: vec![extent, 3],
                complex_input: false,
                map: &map,
            };
            check(case, &reference);
        }
    }
}

#[test]
fn irfft_derivatives_match_finite_differences_and_the_c2c_ifft() {
    // (output length, n): `irfft` needs exactly `len / 2 + 1` spectrum bins, so
    // `None` covers the even default length and `Some` the odd ones.
    let lengths = [
        (4, None),
        (5, Some(5)),
        (6, None),
        (7, Some(7)),
        (1, Some(1)),
        (2, None),
    ];
    for norm in NORMS {
        for (len, n) in lengths {
            let options = FftOptions::default().n(n).axis(0).norm(norm);
            let map = |z: &TracedTensor| z.irfft_with(&options).unwrap();
            let reference = |z: &TracedTensor| {
                hermitian_weighted(z, 0, len)
                    .ifft_with(&options.n(None))
                    .unwrap()
                    .cast(DType::F64)
                    .unwrap()
            };
            let case = LinearCase {
                label: format!("irfft length {len} n {n:?} {norm:?}"),
                shape: vec![len / 2 + 1, 2],
                complex_input: true,
                map: &map,
            };
            check(case, &reference);
        }
    }
}

#[test]
fn irfft_derivatives_along_the_trailing_axis() {
    for norm in NORMS {
        for (len, n) in [(6, None), (5, Some(5))] {
            let options = FftOptions::default().n(n).axis(-1).norm(norm);
            let map = |z: &TracedTensor| z.irfft_with(&options).unwrap();
            let reference = |z: &TracedTensor| {
                hermitian_weighted(z, 1, len)
                    .ifft_with(&options.n(None))
                    .unwrap()
                    .cast(DType::F64)
                    .unwrap()
            };
            let case = LinearCase {
                label: format!("irfft axis -1 length {len} n {n:?} {norm:?}"),
                shape: vec![3, len / 2 + 1],
                complex_input: true,
                map: &map,
            };
            check(case, &reference);
        }
    }
}

/// Axes, shape option, and the transform lengths in axis order.
type NdCase = (Vec<isize>, Option<Vec<usize>>, [usize; 2]);

#[test]
fn rfftn_derivatives_match_finite_differences_and_the_c2c_fftn() {
    // The real axis is the last declared one; the shape option pads and truncates.
    let cases: [NdCase; 3] = [
        (vec![0, 1], None, [3, 4]),
        (vec![0, 1], Some(vec![2, 5]), [2, 5]),
        (vec![1, 0], Some(vec![3, 5]), [3, 5]),
    ];
    for norm in NORMS {
        for (axes, shape, lengths) in &cases {
            let options = FftnOptions::default()
                .axes(axes.clone())
                .shape(shape.clone())
                .norm(norm);
            let real_axis = axes[1] as usize;
            let map = |x: &TracedTensor| x.rfftn_with(&options).unwrap();
            let reference = |x: &TracedTensor| {
                let full = x.convert(DType::C64).unwrap().fftn_with(&options).unwrap();
                one_sided(&full, real_axis, lengths[1])
            };
            let case = LinearCase {
                label: format!("rfftn axes {axes:?} shape {shape:?} {norm:?}"),
                shape: vec![3, 4],
                complex_input: false,
                map: &map,
            };
            check(case, &reference);
        }
    }
}

#[test]
fn irfftn_derivatives_match_finite_differences_and_the_c2c_ifftn() {
    // A [3, 3] spectrum; the shape option pads or truncates the complex axis.
    let cases: [NdCase; 3] = [
        (vec![0, 1], None, [3, 4]),
        (vec![0, 1], Some(vec![4, 5]), [4, 5]),
        (vec![1, 0], Some(vec![2, 4]), [2, 4]),
    ];
    for norm in NORMS {
        for (axes, shape, lengths) in &cases {
            let options = FftnOptions::default()
                .axes(axes.clone())
                .shape(shape.clone())
                .norm(norm);
            let (complex_axis, real_axis) = (axes[0] as usize, axes[1] as usize);
            let map = |z: &TracedTensor| z.irfftn_with(&options).unwrap();
            let reference = |z: &TracedTensor| {
                let weighted = hermitian_weighted(z, real_axis, lengths[1]);
                resized(&weighted, complex_axis, lengths[0])
                    .ifftn_with(&options.clone().shape(None))
                    .unwrap()
                    .cast(DType::F64)
                    .unwrap()
            };
            let case = LinearCase {
                label: format!("irfftn axes {axes:?} shape {shape:?} {norm:?}"),
                shape: vec![3, 3],
                complex_input: true,
                map: &map,
            };
            check(case, &reference);
        }
    }
}

#[test]
fn fft_of_a_real_input_has_the_full_spectrum_derivatives() {
    for norm in NORMS {
        for (extent, n) in [(4, None), (5, Some(3)), (3, Some(6))] {
            let options = FftOptions::default().n(n).axis(1).norm(norm);
            let map = |x: &TracedTensor| x.fft_with(&options).unwrap();
            let reference =
                |x: &TracedTensor| x.convert(DType::C64).unwrap().fft_with(&options).unwrap();
            let case = LinearCase {
                label: format!("fft of real extent {extent} n {n:?} {norm:?}"),
                shape: vec![2, extent],
                complex_input: false,
                map: &map,
            };
            check(case, &reference);
        }
        let options = FftnOptions::default()
            .axes(vec![0, 1])
            .shape(Some(vec![2, 5]))
            .norm(norm);
        let map = |x: &TracedTensor| x.fftn_with(&options).unwrap();
        let reference =
            |x: &TracedTensor| x.convert(DType::C64).unwrap().fftn_with(&options).unwrap();
        let case = LinearCase {
            label: format!("fftn of real {norm:?}"),
            shape: vec![3, 4],
            complex_input: false,
            map: &map,
        };
        check(case, &reference);
    }
}

/// The Fourier-neural-operator spectral layer: `rfft`, multiply every mode by
/// a complex weight, `irfft` back to the signal length, and a squared loss.
fn spectral_layer_loss(x: &TracedTensor, weights: &TracedTensor, len: usize) -> TracedTensor {
    let signal_axis = FftOptions::default().axis(1);
    let spectrum = x.rfft_with(&signal_axis).unwrap();
    let y = spectrum
        .mul(weights)
        .unwrap()
        .irfft_with(&signal_axis.n(Some(len)))
        .unwrap();
    y.mul(&y).unwrap().reduce_sum(None).unwrap()
}

#[test]
fn fno_spectral_layer_gradients_match_finite_differences() {
    for len in [8, 7] {
        let modes = len / 2 + 1;
        let x_data = grid(2 * len, 4, false);
        let w_data = grid(2 * modes, 5, true);
        let loss_at = |x_data: &[Complex64], w_data: &[Complex64]| {
            let trace = TraceContext::new();
            let x = traced(&trace, &[2, len], false, x_data);
            let w = traced(&trace, &[2, modes], true, w_data);
            values(&run(&spectral_layer_loss(&x, &w, len)))[0].re
        };

        let trace = TraceContext::new();
        let x = traced(&trace, &[2, len], false, &x_data);
        let w = traced(&trace, &[2, modes], true, &w_data);
        let loss = spectral_layer_loss(&x, &w, len);
        let ad = ad();
        let x_grad = run(&ad.grad(&loss, &x).unwrap());
        assert_eq!(x_grad.dtype(), DType::F64);
        let w_grad = run(&ad.grad(&loss, &w).unwrap());

        let expected_x = finite_difference_gradient(&x_data, false, |x| loss_at(x, &w_data));
        assert_close(
            &values(&x_grad),
            &expected_x,
            FINITE_DIFFERENCE_TOLERANCE,
            &format!("spectral layer length {len}: d loss / d x"),
        );
        let expected_w = finite_difference_gradient(&w_data, true, |w| loss_at(&x_data, w));
        assert_close(
            &values(&w_grad),
            &expected_w,
            FINITE_DIFFERENCE_TOLERANCE,
            &format!("spectral layer length {len}: d loss / d weights"),
        );
    }
}

#[test]
fn rfft_vjp_of_f32_input_is_f32_and_matches_f64() {
    let data = [0.5_f32, -1.25, 2.0, 0.75, -0.5];
    let cotangent = grid(3, 6, true);
    let vjp = |real: DType| {
        let trace = TraceContext::new();
        let x = match real {
            DType::F32 => trace.from_vec_col_major(vec![5], data.to_vec()),
            _ => trace.from_vec_col_major(vec![5], data.map(f64::from).to_vec()),
        }
        .unwrap();
        let g = match real {
            DType::F32 => trace.from_vec_col_major(
                vec![3],
                cotangent
                    .iter()
                    .map(|value| num_complex::Complex32::new(value.re as f32, value.im as f32))
                    .collect(),
            ),
            _ => trace.from_vec_col_major(vec![3], cotangent.clone()),
        }
        .unwrap();
        let y = x.rfft().unwrap();
        run(&ad().vjp(&y, &x, &g).unwrap())
    };
    let single = vjp(DType::F32);
    assert_eq!(single.dtype(), DType::F32);
    let double = vjp(DType::F64);
    for (single, double) in single
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .zip(double.as_slice::<f64>().unwrap())
    {
        assert!(
            (f64::from(*single) - double).abs() < 1.0e-5,
            "{single} vs {double}"
        );
    }
}

#[test]
fn eager_rfft_and_irfft_vjps_match_the_traced_vjps() {
    let len = 5;
    let signal = grid(len, 7, false);
    let spectrum = grid(len / 2 + 1, 8, true);
    let signal_tensor = || {
        Tensor::from_vec_col_major(vec![len], signal.iter().map(|value| value.re).collect())
            .unwrap()
    };
    let spectrum_tensor = || Tensor::from_vec_col_major(vec![len / 2 + 1], spectrum.clone());
    let options = FftOptions::default().n(Some(len)).norm(FftNorm::Ortho);

    let ad = ad();
    let mut engine = CpuEngine::serial();
    let ctx = EagerRuntime::with_ad_context(&ad).unwrap();
    let x = ctx.variable_from(signal_tensor()).unwrap();
    let z = ctx.variable_from(spectrum_tensor().unwrap()).unwrap();
    let x_cotangent = ctx.constant_from(spectrum_tensor().unwrap()).unwrap();
    let z_cotangent = ctx.constant_from(signal_tensor()).unwrap();
    let (x_vjp, z_vjp) = ctx
        .with_eager_session(&mut engine, |s| {
            let y = s.rfft_with(&x, &options)?;
            let x_vjp = s.vjp(&y, &x, &x_cotangent)?;
            let w = s.irfft_with(&z, &options)?;
            let z_vjp = s.vjp(&w, &z, &z_cotangent)?;
            Ok::<_, tenferro_ad::Error>((x_vjp, z_vjp))
        })
        .unwrap();

    let trace = TraceContext::new();
    let tx = traced(&trace, &[len], false, &signal);
    let tz = traced(&trace, &[len / 2 + 1], true, &spectrum);
    let tx_cotangent = traced(&trace, &[len / 2 + 1], true, &spectrum);
    let tz_cotangent = traced(&trace, &[len], false, &signal);
    let traced_x_vjp = run(&ad
        .vjp(&tx.rfft_with(&options).unwrap(), &tx, &tx_cotangent)
        .unwrap());
    let traced_z_vjp = run(&ad
        .vjp(&tz.irfft_with(&options).unwrap(), &tz, &tz_cotangent)
        .unwrap());

    let eager_x_vjp: Vec<_> = x_vjp
        .value()
        .unwrap()
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();
    assert_close(
        &eager_x_vjp,
        &values(&traced_x_vjp),
        REFERENCE_TOLERANCE,
        "eager rfft vjp versus traced",
    );
    assert_close(
        z_vjp.value().unwrap().as_slice::<Complex64>().unwrap(),
        &values(&traced_z_vjp),
        REFERENCE_TOLERANCE,
        "eager irfft vjp versus traced",
    );
}
