use num_complex::{Complex32, Complex64};
use tenferro_gpu::cuda_next::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_tensor::Tensor;

use super::super::{
    canonical_metadata, canonicalize_input, output_shape_for, transform_mapping, FftNorm,
    FftOperation, FftPlanSpec,
};

fn assert_component(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "expected NaN, got {actual}");
    } else {
        assert_eq!(actual, expected);
    }
}

macro_rules! padding_regression {
    ($name:ident, $dtype:ty, $values:expr, $components:expr) => {
        #[test]
        #[ignore = "requires CUDA hardware"]
        fn $name() {
            assert!(gpu_available(), "requires CUDA hardware");
            let mut engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
            // Both a final axis and a transposed axis, with two independent
            // lanes. The canonical column-major shape is [2, 4].
            for (shape, axis) in [([2, 1], 1), ([1, 2], 0)] {
                let values: Vec<$dtype> = $values;
                let input = Tensor::from_vec_col_major(shape, values.clone()).unwrap();
                let input = upload_tensor(engine.runtime(), &input).unwrap();
                // Complex inputs use the C2C path, real inputs the R2C path.
                let operation = if matches!(
                    input.dtype(),
                    tenferro_tensor::DType::C32 | tenferro_tensor::DType::C64
                ) {
                    FftOperation::C2cForward
                } else {
                    FftOperation::R2cFull
                };
                let spec = FftPlanSpec::new(
                    operation,
                    axis,
                    Some(4),
                    FftNorm::Backward,
                    input.dtype(),
                    input.shape().to_vec(),
                );
                let (transform, direction, _) =
                    transform_mapping(operation, input.dtype()).unwrap();
                let canonical = canonical_metadata(
                    &input,
                    &spec,
                    output_shape_for(&input, &spec).unwrap(),
                    transform,
                    direction,
                )
                .unwrap();
                let padded = canonicalize_input(&mut engine, &input, &spec, &canonical)
                    .unwrap()
                    .unwrap();
                let padded = download_tensor(engine.runtime(), &padded).unwrap();
                assert_eq!(padded.shape(), &[2, 4]);
                let actual = padded.as_slice::<$dtype>().unwrap();
                let components = $components;
                for (&actual, expected) in actual[..2].iter().zip(values) {
                    let [actual_re, actual_im] = components(actual);
                    let [expected_re, expected_im] = components(expected);
                    assert_component(actual_re, expected_re);
                    assert_component(actual_im, expected_im);
                }
                // Inspect the input to cuFFT directly; transform results alone
                // cannot expose every poisoned zero when the input is NaN.
                for &value in &actual[2..] {
                    assert_eq!(components(value), [0.0, 0.0]);
                }
            }
        }
    };
}

padding_regression!(
    cuda_padding_is_zero_f32,
    f32,
    vec![f32::NAN, f32::INFINITY],
    |x: f32| [f64::from(x), 0.0]
);
padding_regression!(
    cuda_padding_is_zero_f64,
    f64,
    vec![f64::NAN, f64::INFINITY],
    |x: f64| [x, 0.0]
);
padding_regression!(
    cuda_padding_is_zero_c32,
    Complex32,
    vec![
        Complex32::new(f32::NAN, f32::INFINITY),
        Complex32::new(f32::NEG_INFINITY, f32::NAN)
    ],
    |x: Complex32| [f64::from(x.re), f64::from(x.im)]
);
padding_regression!(
    cuda_padding_is_zero_c64,
    Complex64,
    vec![
        Complex64::new(f64::NAN, f64::INFINITY),
        Complex64::new(f64::NEG_INFINITY, f64::NAN)
    ],
    |x: Complex64| [x.re, x.im]
);
