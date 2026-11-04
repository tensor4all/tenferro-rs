//! n-dimensional FFT behavior on the public concrete and traced surfaces.
//!
//! `GOLDEN_*` values are a committed fixture, not regenerated at test time.
//! NumPy is not installed on the machine this was written on, so the reference
//! was a naive O(N^2) python DFT of the same input:
//!
//! ```text
//! N = (2, 3, 4); x[i,j,k] = (7i + 13j + 29k) % 11 - 5
//! X[k] = sum_n x[n] * exp(-2j*pi*sum_d k_d*n_d/N_d)
//! ```
//!
//! computed for the full spectrum (`GOLDEN_3D_FFTN`) and for the one-sided
//! spectrum with `k_2 <= N_2/2` (`GOLDEN_3D_RFFTN`).

mod support;

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftNorm, FftPlanCache, TracedTensorFftExt};
#[cfg(feature = "autodiff")]
use tenferro_runtime::TracedTensor;
use tenferro_runtime::{GraphCompiler, TraceContext};
use tenferro_tensor::{DotGeneralConfig, Tensor};

const GOLDEN_3D_SHAPE: [usize; 3] = [2, 3, 4];

const GOLDEN_3D_INPUT: [f64; 24] = [
    -5.0, 2.0, -3.0, 4.0, -1.0, -5.0, 2.0, -2.0, 4.0, 0.0, -5.0, 2.0, -2.0, 5.0, 0.0, -4.0, 2.0,
    -2.0, 5.0, 1.0, -4.0, 3.0, -2.0, 5.0,
];

const GOLDEN_3D_FFTN: [(f64, f64); 24] = [
    (0.0, 0.0),
    (-18.0, -1.102182119232618e-15),
    (9.000000000000002, -5.196152422706631),
    (2.6645352591003757e-15, 1.7763568394002505e-15),
    (8.999999999999995, 5.196152422706633),
    (8.881784197001252e-16, 5.329070518200751e-15),
    (-7.000000000000002, 7.0),
    (-10.999999999999998, -11.0),
    (-15.026279441628821, -4.026279441628835),
    (-4.026279441628839, 15.026279441628827),
    (4.026279441628817, 15.026279441628835),
    (15.026279441628825, -4.026279441628804),
    (-18.0, -3.306546357697853e-15),
    (0.0, 4.898587196589412e-15),
    (-1.7763568394002505e-15, 1.7763568394002505e-15),
    (-33.00000000000003, 19.052558883257618),
    (-2.220446049250313e-15, 9.769962616701378e-15),
    (-32.999999999999964, -19.05255888325769),
    (-6.999999999999995, -7.0),
    (-11.00000000000002, 11.0),
    (4.026279441628834, -15.02627944162882),
    (15.026279441628814, 4.026279441628858),
    (-15.026279441628837, 4.026279441628807),
    (-4.02627944162882, -15.026279441628821),
];

const GOLDEN_3D_RFFTN: [(f64, f64); 18] = [
    (0.0, 0.0),
    (-18.0, -1.102182119232618e-15),
    (9.000000000000002, -5.196152422706631),
    (2.6645352591003757e-15, 1.7763568394002505e-15),
    (8.999999999999995, 5.196152422706633),
    (8.881784197001252e-16, 5.329070518200751e-15),
    (-7.000000000000002, 7.0),
    (-10.999999999999998, -11.0),
    (-15.026279441628821, -4.026279441628835),
    (-4.026279441628839, 15.026279441628827),
    (4.026279441628817, 15.026279441628835),
    (15.026279441628825, -4.026279441628804),
    (-18.0, -3.306546357697853e-15),
    (0.0, 4.898587196589412e-15),
    (-1.7763568394002505e-15, 1.7763568394002505e-15),
    (-33.00000000000003, 19.052558883257618),
    (-2.220446049250313e-15, 9.769962616701378e-15),
    (-32.999999999999964, -19.05255888325769),
];

const TOLERANCE: f64 = 1.0e-10;

fn complex_pairs(values: &[(f64, f64)]) -> Vec<Complex64> {
    values
        .iter()
        .map(|(re, im)| Complex64::new(*re, *im))
        .collect()
}

/// Deterministic pseudo-random values, so the tests need no RNG dependency.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn complex_grid(shape: &[usize], seed: u64) -> Vec<Complex64> {
    let mut next = lcg(seed);
    let len = shape.iter().product();
    (0..len).map(|_| Complex64::new(next(), next())).collect()
}

fn real_grid(shape: &[usize], seed: u64) -> Vec<f64> {
    let mut next = lcg(seed);
    let len = shape.iter().product();
    (0..len).map(|_| next()).collect()
}

fn scaled(values: &[Complex64], factor: f64) -> Vec<Complex64> {
    values.iter().map(|value| value * factor).collect()
}

fn assert_c64_close(actual: &[Complex64], expected: &[Complex64], context: &str) {
    assert_c64_close_within(actual, expected, TOLERANCE, context);
}

fn assert_c64_close_within(
    actual: &[Complex64],
    expected: &[Complex64],
    tolerance: f64,
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() < tolerance,
            "{context}: index {index} actual {actual:?} expected {expected:?}"
        );
    }
}

#[test]
fn fftn_over_three_axes_matches_sequential_one_axis_fft() {
    let shape = [3usize, 4, 5];
    let values = complex_grid(&shape, 1);
    let input = Tensor::from_vec_col_major(shape.to_vec(), values.clone()).unwrap();
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let n_dimensional = executor
        .fftn(&input, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();

    let mut sequential = Tensor::from_vec_col_major(shape.to_vec(), values).unwrap();
    for axis in 0..3 {
        sequential = executor
            .fft(&sequential, None, axis, FftNorm::Backward, &mut engine)
            .unwrap();
    }

    assert_c64_close(
        n_dimensional.as_slice::<Complex64>().unwrap(),
        sequential.as_slice::<Complex64>().unwrap(),
        "fftn versus three fft calls",
    );
}

#[test]
fn fftn_real_input_matches_sequential_full_spectrum_fft() {
    let shape = [2usize, 3, 4];
    let values = real_grid(&shape, 2);
    let input = Tensor::from_vec_col_major(shape.to_vec(), values.clone()).unwrap();
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let n_dimensional = executor
        .fftn(&input, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();
    let mut sequential = Tensor::from_vec_col_major(shape.to_vec(), values).unwrap();
    for axis in 0..3 {
        sequential = executor
            .fft(&sequential, None, axis, FftNorm::Backward, &mut engine)
            .unwrap();
    }

    assert_eq!(n_dimensional.dtype(), sequential.dtype());
    assert_c64_close(
        n_dimensional.as_slice::<Complex64>().unwrap(),
        sequential.as_slice::<Complex64>().unwrap(),
        "real fftn versus three fft calls",
    );
}

#[test]
fn fftn_matches_the_committed_golden_fixture() {
    let input =
        Tensor::from_vec_col_major(GOLDEN_3D_SHAPE.to_vec(), GOLDEN_3D_INPUT.to_vec()).unwrap();
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let output = executor
        .fftn(&input, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();

    assert_c64_close(
        output.as_slice::<Complex64>().unwrap(),
        &complex_pairs(&GOLDEN_3D_FFTN),
        "fftn golden fixture",
    );
}

#[test]
fn rfftn_matches_the_golden_fixture_and_the_onesided_half_of_fftn() {
    let input =
        Tensor::from_vec_col_major(GOLDEN_3D_SHAPE.to_vec(), GOLDEN_3D_INPUT.to_vec()).unwrap();
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let onesided = executor
        .rfftn(&input, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();
    let full = executor
        .fftn(&input, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();

    let [n0, n1, n2] = GOLDEN_3D_SHAPE;
    assert_eq!(onesided.shape(), &[n0, n1, n2 / 2 + 1]);
    let onesided_values = onesided.as_slice::<Complex64>().unwrap();
    assert_c64_close(
        onesided_values,
        &complex_pairs(&GOLDEN_3D_RFFTN),
        "rfftn golden fixture",
    );

    // The Hermitian convention of rfft: the one-sided spectrum is the
    // non-negative half of the full spectrum, taken along the real axis.
    let full_values = full.as_slice::<Complex64>().unwrap();
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 / 2 + 1 {
                // Both arrays are column-major, and only the extent of the last
                // axis differs, so one flat offset addresses the same frequency.
                let onesided_index = i + n0 * j + n0 * n1 * k;
                let full_index = i + n0 * j + n0 * n1 * k;
                assert!(
                    (onesided_values[onesided_index] - full_values[full_index]).norm() < TOLERANCE,
                    "rfftn[{i},{j},{k}] is not the one-sided half of fftn"
                );
            }
        }
    }
}

#[test]
fn rfftn_irfftn_round_trip_restores_the_real_grid() {
    let shape = [3usize, 5, 6];
    let values = real_grid(&shape, 3);
    let input = Tensor::from_vec_col_major(shape.to_vec(), values.clone()).unwrap();
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let spectrum = executor
        .rfftn(&input, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();
    assert_eq!(spectrum.shape(), &[3, 5, 4]);

    let restored = executor
        .irfftn(&spectrum, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();
    assert_eq!(restored.shape(), shape.as_slice());
    let restored = restored.as_slice::<f64>().unwrap();
    for (index, (actual, expected)) in restored.iter().zip(&values).enumerate() {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "round trip: index {index} actual {actual} expected {expected}"
        );
    }

    // An explicit shape states the same output lengths as the input extents.
    let restored_with_shape = executor
        .irfftn(
            &spectrum,
            Some(&shape),
            &[0, 1, 2],
            FftNorm::Backward,
            &mut engine,
        )
        .unwrap();
    assert_eq!(
        restored_with_shape.as_slice::<f64>().unwrap(),
        restored,
        "explicit shape must not change the reconstructed grid"
    );
}

#[test]
fn fftn_with_one_axis_is_exactly_the_one_axis_fft() {
    let shape = [4usize, 3];
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let complex_input =
        Tensor::from_vec_col_major(shape.to_vec(), complex_grid(&shape, 4)).unwrap();
    let n_dimensional = executor
        .fftn(&complex_input, None, &[1], FftNorm::Ortho, &mut engine)
        .unwrap();
    let one_axis = executor
        .fft(&complex_input, None, 1, FftNorm::Ortho, &mut engine)
        .unwrap();
    assert_eq!(
        n_dimensional.as_slice::<Complex64>().unwrap(),
        one_axis.as_slice::<Complex64>().unwrap(),
        "C64: a one-element axes must be the one-axis transform, bit for bit"
    );

    let real_input = Tensor::from_vec_col_major(shape.to_vec(), real_grid(&shape, 5)).unwrap();
    let n_dimensional = executor
        .fftn(&real_input, None, &[-2], FftNorm::Backward, &mut engine)
        .unwrap();
    let one_axis = executor
        .fft(&real_input, None, -2, FftNorm::Backward, &mut engine)
        .unwrap();
    assert_eq!(
        n_dimensional.as_slice::<Complex64>().unwrap(),
        one_axis.as_slice::<Complex64>().unwrap(),
        "F64: a one-element axes must be the one-axis transform, bit for bit"
    );

    let real_input = Tensor::from_vec_col_major(shape.to_vec(), real_grid(&shape, 6)).unwrap();
    let n_dimensional = executor
        .rfftn(&real_input, None, &[0], FftNorm::Backward, &mut engine)
        .unwrap();
    let one_axis = executor
        .rfft(&real_input, None, 0, FftNorm::Backward, &mut engine)
        .unwrap();
    assert_eq!(
        n_dimensional.as_slice::<Complex64>().unwrap(),
        one_axis.as_slice::<Complex64>().unwrap(),
        "F64: a one-element axes must be the one-axis real transform, bit for bit"
    );
}

#[test]
fn fftn_normalization_is_applied_once_over_the_whole_transform() {
    let shape = [3usize, 4];
    let values = complex_grid(&shape, 7);
    let input = Tensor::from_vec_col_major(shape.to_vec(), values.clone()).unwrap();
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();
    let total = (shape[0] * shape[1]) as f64;

    // `Forward` leaves the inverse transform unscaled.
    let unscaled = executor
        .ifftn(&input, None, &[0, 1], FftNorm::Forward, &mut engine)
        .unwrap();
    let unscaled = unscaled.as_slice::<Complex64>().unwrap();

    // The one n-dimensional factor: 1 / prod(shape over axes).
    let once = executor
        .ifftn(&input, None, &[0, 1], FftNorm::Backward, &mut engine)
        .unwrap();
    let once = once.as_slice::<Complex64>().unwrap();
    assert_c64_close(
        once,
        &scaled(unscaled, 1.0 / total),
        "ifftn Backward scales once by 1 / prod(shape)",
    );

    // Composing the one-axis transforms states the same single factor.
    let mut sequential = Tensor::from_vec_col_major(shape.to_vec(), values).unwrap();
    for axis in 0..2 {
        sequential = executor
            .ifft(&sequential, None, axis, FftNorm::Backward, &mut engine)
            .unwrap();
    }
    assert_c64_close(
        once,
        sequential.as_slice::<Complex64>().unwrap(),
        "per-axis factors telescope to the single n-dimensional factor",
    );

    // The rejected reading: the whole transform's 1 / prod(shape) applied once
    // per axis, which is 1 / prod(shape)^2 for these two axes.
    let per_axis_reading = scaled(unscaled, 1.0 / (total * total));
    assert_c64_close_within(
        once,
        &scaled(&per_axis_reading, total),
        1.0e-9,
        "the per-axis reading of one whole factor differs by prod(shape)",
    );
}

#[test]
fn fftn_rejects_empty_repeated_and_mismatched_axes() {
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();
    let input = Tensor::from_vec_col_major([2usize, 3], complex_grid(&[2, 3], 9)).unwrap();

    for (axes, shape) in [
        (vec![], None),
        (vec![0, 0], None),
        (vec![0, 1], Some(vec![2])),
        (vec![2], None),
        (vec![-3], None),
    ] {
        let shape = shape.as_deref();
        let error = executor
            .fftn(&input, shape, &axes, FftNorm::Backward, &mut engine)
            .expect_err("invalid axes must be rejected");
        assert!(
            matches!(error, tenferro_tensor::Error::Validation { .. }),
            "axes {axes:?} shape {shape:?} produced {error:?}"
        );
    }
}

#[test]
fn fftn_reuses_one_plan_per_distinct_length_and_dtype() {
    let shape = [4usize, 4, 4];
    let input = Tensor::from_vec_col_major(shape.to_vec(), complex_grid(&shape, 10)).unwrap();
    let mut executor = FftExecutor::new(FftPlanCache::default());
    let mut engine = CpuEngine::serial();

    executor
        .fftn(&input, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();
    assert_eq!(
        executor.cache_stats().entries,
        1,
        "three axes of one length and dtype share a single cached plan"
    );

    let non_square = [2usize, 3, 4];
    let input =
        Tensor::from_vec_col_major(non_square.to_vec(), complex_grid(&non_square, 11)).unwrap();
    executor
        .fftn(&input, None, &[0, 1, 2], FftNorm::Backward, &mut engine)
        .unwrap();
    assert_eq!(
        executor.cache_stats().entries,
        3,
        "each distinct length keeps its own plan; the repeated length 4 is reused"
    );
}

#[test]
fn traced_fftn_family_matches_the_eager_executor() {
    let trace = TraceContext::new();
    let shape = [2usize, 3, 4];
    let complex_values = complex_grid(&shape, 12);
    let real_values = real_grid(&shape, 13);
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let complex_input = Tensor::from_vec_col_major(shape.to_vec(), complex_values.clone()).unwrap();
    let eager_full = executor
        .fftn(
            &complex_input,
            None,
            &[0, 1, 2],
            FftNorm::Backward,
            &mut engine,
        )
        .unwrap();

    let mut compiler = GraphCompiler::new();
    let traced = trace
        .from_vec_col_major(shape.to_vec(), complex_values)
        .unwrap()
        .fftn(None, &[0, 1, 2], FftNorm::Backward)
        .unwrap();
    let program = compiler.compile(&traced).unwrap();
    let traced_full = support::run_one(&program, &[]).unwrap();
    assert_c64_close(
        traced_full.as_slice::<Complex64>().unwrap(),
        eager_full.as_slice::<Complex64>().unwrap(),
        "traced fftn versus FftExecutor::fftn",
    );

    let eager_inverse = executor
        .ifftn(
            &eager_full,
            None,
            &[0, 1, 2],
            FftNorm::Backward,
            &mut engine,
        )
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let traced = trace
        .from_vec_col_major(
            eager_full.shape().to_vec(),
            eager_full.as_slice::<Complex64>().unwrap().to_vec(),
        )
        .unwrap()
        .ifftn(None, &[0, 1, 2], FftNorm::Backward)
        .unwrap();
    let program = compiler.compile(&traced).unwrap();
    let traced_inverse = support::run_one(&program, &[]).unwrap();
    assert_c64_close(
        traced_inverse.as_slice::<Complex64>().unwrap(),
        eager_inverse.as_slice::<Complex64>().unwrap(),
        "traced ifftn versus FftExecutor::ifftn",
    );

    let real_input = Tensor::from_vec_col_major(shape.to_vec(), real_values.clone()).unwrap();
    let eager_rfftn = executor
        .rfftn(
            &real_input,
            None,
            &[0, 1, 2],
            FftNorm::Backward,
            &mut engine,
        )
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let traced = trace
        .from_vec_col_major(shape.to_vec(), real_values)
        .unwrap()
        .rfftn(None, &[0, 1, 2], FftNorm::Backward)
        .unwrap();
    let program = compiler.compile(&traced).unwrap();
    let traced_rfftn = support::run_one(&program, &[]).unwrap();
    assert_c64_close(
        traced_rfftn.as_slice::<Complex64>().unwrap(),
        eager_rfftn.as_slice::<Complex64>().unwrap(),
        "traced rfftn versus FftExecutor::rfftn",
    );

    let eager_irfftn = executor
        .irfftn(
            &eager_rfftn,
            Some(&shape),
            &[0, 1, 2],
            FftNorm::Backward,
            &mut engine,
        )
        .unwrap();
    let mut compiler = GraphCompiler::new();
    let traced = trace
        .from_vec_col_major(
            eager_rfftn.shape().to_vec(),
            eager_rfftn.as_slice::<Complex64>().unwrap().to_vec(),
        )
        .unwrap()
        .irfftn(Some(&shape), &[0, 1, 2], FftNorm::Backward)
        .unwrap();
    let program = compiler.compile(&traced).unwrap();
    let traced_irfftn = support::run_one(&program, &[]).unwrap();
    let traced_irfftn = traced_irfftn.as_slice::<f64>().unwrap();
    for (index, (actual, expected)) in traced_irfftn
        .iter()
        .zip(eager_irfftn.as_slice::<f64>().unwrap())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "traced irfftn versus FftExecutor::irfftn: index {index}"
        );
    }
}

#[cfg(feature = "autodiff")]
fn fft_ad_context() -> tenferro_ad::AdContext {
    tenferro_ad::AdContext::builder()
        .with_semantic_extension_rules(tenferro_fft::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap()
}

#[cfg(feature = "autodiff")]
fn run_traced(output: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).unwrap();
    support::run_one(&program, &[]).unwrap()
}

/// Central finite difference of a complex-linear transform along `direction`.
#[cfg(feature = "autodiff")]
fn directional_finite_difference(
    shape: &[usize],
    transform: impl Fn(&TracedTensor) -> TracedTensor,
    base: &[Complex64],
    direction: &[Complex64],
    step: f64,
) -> Vec<Complex64> {
    let trace = TraceContext::new();
    let evaluate = |values: Vec<Complex64>| {
        let input = trace.from_vec_col_major(shape.to_vec(), values).unwrap();
        run_traced(&transform(&input))
            .as_slice::<Complex64>()
            .unwrap()
            .to_vec()
    };
    let plus = evaluate(
        base.iter()
            .zip(direction)
            .map(|(base, direction)| base + direction * step)
            .collect(),
    );
    let minus = evaluate(
        base.iter()
            .zip(direction)
            .map(|(base, direction)| base - direction * step)
            .collect(),
    );
    plus.iter()
        .zip(&minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * step))
        .collect()
}

#[cfg(feature = "autodiff")]
#[test]
fn fftn_jvp_matches_finite_differences_and_the_transform_of_the_tangent() {
    let trace = TraceContext::new();
    let shape = [2usize, 3, 4];
    let base = complex_grid(&shape, 14);
    let direction = complex_grid(&shape, 15);
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let x = trace
        .from_vec_col_major(shape.to_vec(), base.clone())
        .unwrap();
    let dx = trace
        .from_vec_col_major(shape.to_vec(), direction.clone())
        .unwrap();
    let y = x.fftn(None, &[0, 1, 2], FftNorm::Backward).unwrap();
    let dy = fft_ad_context().jvp(&y, &x, &dx).unwrap();
    let jvp = run_traced(&dy);
    let jvp = jvp.as_slice::<Complex64>().unwrap();

    // Closed form: an fft is complex-linear, so the tangent is the transform of
    // the tangent.
    let expected = executor
        .fftn(
            &Tensor::from_vec_col_major(shape.to_vec(), direction.clone()).unwrap(),
            None,
            &[0, 1, 2],
            FftNorm::Backward,
            &mut engine,
        )
        .unwrap();
    assert_c64_close(
        jvp,
        expected.as_slice::<Complex64>().unwrap(),
        "fftn jvp versus fftn of the tangent",
    );

    let finite = directional_finite_difference(
        &shape,
        |input| input.fftn(None, &[0, 1, 2], FftNorm::Backward).unwrap(),
        &base,
        &direction,
        1.0e-6,
    );
    // A central difference carries cancellation error, so compare it loosely.
    assert_c64_close_within(jvp, &finite, 1.0e-7, "fftn jvp versus central differences");
}

#[cfg(feature = "autodiff")]
#[test]
fn fftn_vjp_matches_the_closed_form_adjoint_in_three_dimensions() {
    let trace = TraceContext::new();
    let shape = [2usize, 3, 4];
    let base = complex_grid(&shape, 16);
    let cotangent = complex_grid(&shape, 17);
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let x = trace.from_vec_col_major(shape.to_vec(), base).unwrap();
    let g = trace
        .from_vec_col_major(shape.to_vec(), cotangent.clone())
        .unwrap();
    let y = x.fftn(None, &[0, 1, 2], FftNorm::Backward).unwrap();
    let dx = fft_ad_context().vjp(&y, &x, &g).unwrap();
    let vjp = run_traced(&dx);
    let vjp = vjp.as_slice::<Complex64>().unwrap();

    // Closed form: the unscaled forward F has adjoint F^H = P * F^-1, which is
    // exactly the inverse transform with `Forward` normalization left unscaled.
    let expected = executor
        .ifftn(
            &Tensor::from_vec_col_major(shape.to_vec(), cotangent).unwrap(),
            None,
            &[0, 1, 2],
            FftNorm::Forward,
            &mut engine,
        )
        .unwrap();
    assert_c64_close(
        vjp,
        expected.as_slice::<Complex64>().unwrap(),
        "fftn vjp versus the closed-form adjoint",
    );
}

#[cfg(feature = "autodiff")]
fn hermitian_inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * right)
        .sum()
}

/// Requested lengths that pad one axis, truncate another, and pad a third must
/// restore every resized axis in the transpose, not only the first one.
#[cfg(feature = "autodiff")]
#[test]
fn fftn_vjp_restores_every_resized_axis() {
    let trace = TraceContext::new();
    let shape = [2usize, 3, 4];
    let lengths = [3usize, 2, 6];
    let base = complex_grid(&shape, 22);
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let transformed = executor
        .fftn(
            &Tensor::from_vec_col_major(shape.to_vec(), base.clone()).unwrap(),
            Some(&lengths),
            &[0, 1, 2],
            FftNorm::Backward,
            &mut engine,
        )
        .unwrap();
    assert_eq!(transformed.shape(), &[3, 2, 6], "pad 2->3, 3->2, 4->6");
    let cotangent = complex_grid(transformed.shape(), 23);

    let x = trace
        .from_vec_col_major(shape.to_vec(), base.clone())
        .unwrap();
    let g = trace
        .from_vec_col_major(transformed.shape().to_vec(), cotangent.clone())
        .unwrap();
    let y = x
        .fftn(Some(&lengths), &[0, 1, 2], FftNorm::Backward)
        .unwrap();
    let dx = fft_ad_context().vjp(&y, &x, &g).unwrap();
    let vjp = run_traced(&dx);
    assert_eq!(
        vjp.shape(),
        shape.as_slice(),
        "the adjoint restores the input shape"
    );
    let vjp = vjp.as_slice::<Complex64>().unwrap();

    // Adjoint identity: <A x, g> == <x, A^H g>.
    let lhs = hermitian_inner(transformed.as_slice::<Complex64>().unwrap(), &cotangent);
    let rhs = hermitian_inner(&base, vjp);
    assert!(
        (lhs - rhs).norm() < 1.0e-9,
        "<A x, g> = {lhs:?} but <x, A^H g> = {rhs:?}"
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn fftn_with_one_axis_shares_the_ad_rule_of_the_one_axis_transform() {
    let trace = TraceContext::new();
    let shape = [4usize, 3];
    let base = complex_grid(&shape, 18);
    let direction = complex_grid(&shape, 19);
    let cotangent = complex_grid(&shape, 20);
    let mut executor = FftExecutor::default();
    let mut engine = CpuEngine::serial();

    let x = trace.from_vec_col_major(shape.to_vec(), base).unwrap();
    let dx = trace
        .from_vec_col_major(shape.to_vec(), direction.clone())
        .unwrap();
    let y = x.fftn(None, &[1], FftNorm::Backward).unwrap();
    let dy = fft_ad_context().jvp(&y, &x, &dx).unwrap();
    let jvp = run_traced(&dy);
    let expected = executor
        .fftn(
            &Tensor::from_vec_col_major(shape.to_vec(), direction).unwrap(),
            None,
            &[1],
            FftNorm::Backward,
            &mut engine,
        )
        .unwrap();
    assert_c64_close(
        jvp.as_slice::<Complex64>().unwrap(),
        expected.as_slice::<Complex64>().unwrap(),
        "one-axis fftn jvp",
    );

    let x = trace
        .from_vec_col_major(shape.to_vec(), complex_grid(&shape, 21))
        .unwrap();
    let g = trace
        .from_vec_col_major(shape.to_vec(), cotangent.clone())
        .unwrap();
    let y = x.fftn(None, &[1], FftNorm::Backward).unwrap();
    let dx = fft_ad_context().vjp(&y, &x, &g).unwrap();
    let vjp = run_traced(&dx);
    let expected = executor
        .ifftn(
            &Tensor::from_vec_col_major(shape.to_vec(), cotangent).unwrap(),
            None,
            &[1],
            FftNorm::Forward,
            &mut engine,
        )
        .unwrap();
    assert_c64_close(
        vjp.as_slice::<Complex64>().unwrap(),
        expected.as_slice::<Complex64>().unwrap(),
        "one-axis fftn vjp",
    );
}

/// A truncated operand keeps its bound through an extension op.
///
/// `DynamicTruncate` declares `[4]` and holds two elements. The FFT extension keeps
/// the axis length, so its output must not be published as an exact `[4]`: an outer
/// product with a literal `[3]` would otherwise broadcast to `[4, 3]` instead of
/// `[2, 3]`. Runs through `Runtime::prepare` with the production CPU registration, so
/// the placement-aware view folding is active.
#[test]
fn outer_product_of_a_truncated_fft_uses_the_truncated_length() {
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])
        .expect("complex input");
    let keep = trace
        .from_vec_col_major(vec![], vec![2.0_f64])
        .expect("truncation length");
    let truncated = x.dynamic_truncate(&keep, 0).expect("truncate");
    let transformed = truncated
        .fft(None, 0, FftNorm::Backward)
        .expect("traced fft");
    let other = trace
        .from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0])
        .expect("second operand");
    let outer = transformed
        .dot_general(
            &other,
            DotGeneralConfig {
                lhs_contracting_dims: Default::default(),
                rhs_contracting_dims: Default::default(),
                lhs_batch_dims: Default::default(),
                rhs_batch_dims: Default::default(),
            },
        )
        .expect("outer product");
    let program = GraphCompiler::new().compile(&outer).expect("compile");

    let x_value = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let keep_value = Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let other_value = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();
    let output = support::run_one(&program, &[&x_value, &keep_value, &other_value])
        .expect("execute prepared outer product");

    assert_eq!(output.shape(), &[2, 3]);
}
