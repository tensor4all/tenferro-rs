//! Crate-internal n-dimensional FFT tests: borrowed strided views and the
//! single axis loop shared by the eager and extension execution paths.

use crate::FftnOptions;
use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::Engine;
use tenferro_tensor::{DType, Tensor, TensorRead, TensorView, TypedTensorView};

use crate::{FftExecutor, FftNorm, FftOperation, FftPlanCache};

fn view_of_complex<'a>(
    data: &'a [Complex64],
    shape: &[usize],
    strides: &[isize],
) -> TensorRead<'a> {
    TensorRead::from_view(TensorView::C64(
        TypedTensorView::from_slice(shape, strides, 0, data).unwrap(),
    ))
}

/// A transposed view of a compact `[4, 3]` buffer has row-major strides
/// `[4, 1]` for its `[3, 4]` shape, so it is not column-major contiguous.
#[test]
fn fftn_executes_on_a_transposed_non_contiguous_view() {
    let underlying: Vec<Complex64> = (0..12)
        .map(|index| Complex64::new(index as f64, (index as f64) * 0.5))
        .collect();
    let view = view_of_complex(&underlying, &[3, 4], &[4, 1]);
    match &view {
        TensorRead::View(view) => assert!(
            !view.is_col_major_contiguous().unwrap(),
            "the test view must be non-contiguous"
        ),
        TensorRead::Tensor(_) => panic!("expected a borrowed view"),
    }

    // The same logical elements in compact column-major order.
    let contiguous: Vec<Complex64> = (0..12)
        .map(|index| underlying[index % 3 * 4 + index / 3])
        .collect();
    let expected_input = Tensor::from_vec_col_major([3usize, 4], contiguous).unwrap();

    let mut engine = CpuEngine::serial();
    let mut executor = FftExecutor::default();
    let expected = executor
        .fftn_with(
            &expected_input,
            &FftnOptions::default().axes(vec![0, 1]),
            &mut engine,
        )
        .unwrap();

    let op = crate::validated_fft_op(
        "fftn",
        FftOperation::C2cForward,
        Some(DType::C64),
        2,
        Some(&[3, 4]),
        None,
        "shape",
        &[0, 1],
        FftNorm::Backward,
    )
    .unwrap();
    let mut plans = FftPlanCache::default();
    let mut outputs =
        crate::execute_fft_extension_reads_on_session(&op, &[view], &mut engine, plans.store_mut())
            .unwrap();

    let view_output = outputs.pop().unwrap();
    assert_eq!(view_output.shape(), expected.shape());
    for (index, (actual, expected)) in view_output
        .as_slice::<Complex64>()
        .unwrap()
        .iter()
        .zip(expected.as_slice::<Complex64>().unwrap())
        .enumerate()
    {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "index {index}: strided view {actual:?} versus contiguous copy {expected:?}"
        );
    }
}

/// The eager executor must not carry its own axis loop: it runs the same
/// extension execution entry point as traced execution.
#[test]
fn eager_fftn_and_the_extension_path_execute_the_same_axis_loop() {
    let values: Vec<Complex64> = (0..24)
        .map(|index| Complex64::new((index % 7) as f64, (index % 5) as f64))
        .collect();
    let input = Tensor::from_vec_col_major([2usize, 3, 4], values).unwrap();
    let mut engine = CpuEngine::serial();

    let mut executor = FftExecutor::default();
    let eager = executor
        .fftn_with(
            &input,
            &FftnOptions::default().axes(vec![2, 0]),
            &mut engine,
        )
        .unwrap();

    let op = crate::validated_fft_op(
        "fftn",
        FftOperation::C2cForward,
        Some(DType::C64),
        3,
        Some(&[2, 3, 4]),
        None,
        "shape",
        &[2, 0],
        FftNorm::Backward,
    )
    .unwrap();
    let mut plans = FftPlanCache::default();
    let extension = crate::execute_fft_extension_reads_on_session(
        &op,
        &[TensorRead::from_tensor(&input)],
        &mut engine as &mut dyn Engine,
        plans.store_mut(),
    )
    .unwrap()
    .pop()
    .unwrap();

    assert_eq!(
        eager.as_slice::<Complex64>().unwrap(),
        extension.as_slice::<Complex64>().unwrap(),
        "eager and extension execution must produce identical bits"
    );
}
