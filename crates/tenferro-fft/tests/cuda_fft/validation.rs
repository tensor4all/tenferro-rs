use std::error::Error as StdError;

use super::common::*;
use super::support;
use tenferro_fft::{FftExecutor, FftNorm, FFT_EXTENSION_FAMILY_ID};
use tenferro_gpu::cuda::gpu_available;
use tenferro_runtime::Tensor;
use tenferro_tensor::{Error, ErrorKind, ValidationError, ValidationKind};

fn assert_placement_error(result: tenferro_tensor::Result<Tensor>) {
    let error = result.expect_err("CUDA placement mismatch must return an error");
    assert_eq!(error.kind(), ErrorKind::RuntimeState);
    let Error::RuntimeStateSource {
        op: "cuda_fft",
        source,
    } = &error
    else {
        panic!("placement mismatch returned the wrong error variant: {error:?}");
    };
    let placement_source = source
        .source()
        .expect("placement wrapper must retain the backend error source");
    assert!(
        placement_source
            .downcast_ref::<tenferro_tensor::Error>()
            .is_some(),
        "placement wrapper must retain the typed CUDA residency error"
    );
}

fn assert_unsupported_dtype(result: tenferro_tensor::Result<Tensor>, op: &'static str) {
    let error = result.expect_err("unsupported CUDA dtype must return an error");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let Error::Extension {
        op: actual_op,
        family,
        kind: ErrorKind::Unsupported,
        source: _source,
    } = &error
    else {
        panic!("unsupported dtype returned the wrong error variant: {error:?}");
    };
    assert_eq!(*actual_op, op);
    assert_eq!(*family, FFT_EXTENSION_FAMILY_ID);
    let source = error
        .source()
        .expect("unsupported dtype must retain the typed FFT source");
    assert!(source.source().is_none());
}

fn assert_validation_error(
    result: tenferro_tensor::Result<Tensor>,
    op: &'static str,
    kind: ValidationKind,
    expected: impl FnOnce(&ValidationError),
) {
    let error = result.expect_err("invalid CUDA FFT input must return an error");
    assert!(matches!(&error, Error::Validation { op: actual_op, .. } if *actual_op == op));
    assert_eq!(error.kind(), ErrorKind::Validation(kind));
    let source = error
        .source()
        .expect("validation error must retain its source");
    let validation = source
        .downcast_ref::<ValidationError>()
        .expect("validation source must retain ValidationError");
    expected(validation);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_validation_rejects_host_foreign_integer_bool_and_invalid_lengths() {
    if !gpu_available() {
        return;
    }

    let mut cuda = support::cuda_backend();
    let mut executor = FftExecutor::default();
    let host = real_f64(&[4], 0.5);
    assert_placement_error(Operation::Fft.execute_executor(
        &mut executor,
        &mut cuda,
        &host,
        None,
        -1,
        FftNorm::Backward,
    ));
    let rejected_host_stats = executor.cache_stats();
    assert_eq!(rejected_host_stats.entries, 0);
    assert_eq!(rejected_host_stats.retained_bytes, 0);
    assert_eq!(rejected_host_stats.hits, 0);
    assert_eq!(rejected_host_stats.misses, 0);
    assert_eq!(rejected_host_stats.evictions, 0);
    assert_eq!(rejected_host_stats.clears, 0);

    let other = support::cuda_backend();
    let gpu_input = support::upload_cuda(other.runtime(), &host);
    assert_placement_error(Operation::Fft.execute_executor(
        &mut executor,
        &mut cuda,
        &gpu_input,
        None,
        -1,
        FftNorm::Backward,
    ));
    let rejected_foreign_stats = executor.cache_stats();
    assert_eq!(rejected_foreign_stats.entries, 0);
    assert_eq!(rejected_foreign_stats.retained_bytes, 0);
    assert_eq!(rejected_foreign_stats.hits, 0);
    assert_eq!(rejected_foreign_stats.misses, 0);
    assert_eq!(rejected_foreign_stats.evictions, 0);
    assert_eq!(rejected_foreign_stats.clears, 0);

    let integer = Tensor::from_vec_col_major(vec![4], vec![1_i32, 2, 3, 4]).unwrap();
    assert_unsupported_dtype(
        Operation::Fft.execute_cuda(&mut cuda, &integer, None, -1, FftNorm::Backward),
        "TensorFftExt::fft",
    );
    let boolean = Tensor::from_vec_col_major(vec![4], vec![true, false, true, false]).unwrap();
    assert_unsupported_dtype(
        Operation::Fft.execute_cuda(&mut cuda, &boolean, None, -1, FftNorm::Backward),
        "TensorFftExt::fft",
    );

    assert_validation_error(
        Operation::Fft.execute_cuda(&mut cuda, &host, None, 4, FftNorm::Backward),
        "TensorFftExt::fft",
        ValidationKind::AxisOutOfBounds,
        |source| {
            assert!(matches!(
                source,
                ValidationError::AxisOutOfBounds { axis: 4, rank: 1 }
            ));
        },
    );
    assert_validation_error(
        Operation::Fft.execute_cuda(&mut cuda, &host, Some(0), -1, FftNorm::Backward),
        "TensorFftExt::fft",
        ValidationKind::InvalidArgument,
        |source| {
            assert!(matches!(
                source,
                ValidationError::InvalidArgument { argument: "n", .. }
            ));
        },
    );

    let spectrum = complex_f64(&[3], 0.5);
    assert_validation_error(
        Operation::Irfft.execute_cuda(&mut cuda, &spectrum, Some(8), -1, FftNorm::Backward),
        "irfft",
        ValidationKind::InvalidArgument,
        |source| {
            assert!(matches!(
                source,
                ValidationError::InvalidArgument {
                    argument: "spectrum",
                    ..
                }
            ));
        },
    );
}
