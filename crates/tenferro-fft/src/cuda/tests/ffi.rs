use std::error::Error;
use std::ffi::{OsStr, OsString};

use super::super::error::{into_tensor_error, CudaFftError};
use super::super::ffi::{
    cufft_library_candidates, map_cufft_status, CufftLibrary, CUFFT_ALLOC_FAILED,
    CUFFT_EXEC_FAILED, CUFFT_INTERNAL_ERROR, CUFFT_INVALID_PLAN, CUFFT_INVALID_SIZE,
    CUFFT_INVALID_TYPE, CUFFT_INVALID_VALUE, CUFFT_SETUP_FAILED, CUFFT_SUCCESS,
    CUFFT_UNALIGNED_DATA,
};

#[test]
fn library_candidates_try_override_then_cuda11_cuda10_and_bare_sonames() {
    let candidates = cufft_library_candidates(Some(OsStr::new(
        "/opt/cuda-a/libcufft.so:/opt/cuda-b/libcufft.so",
    )));

    assert_eq!(
        candidates,
        vec![
            OsString::from("/opt/cuda-a/libcufft.so"),
            OsString::from("/opt/cuda-b/libcufft.so"),
            OsString::from("libcufft.so.11"),
            OsString::from("libcufft.so.10"),
            OsString::from("libcufft.so"),
        ]
    );
}

#[test]
fn library_candidates_use_defaults_when_override_is_absent_or_empty() {
    let expected = vec![
        OsString::from("libcufft.so.11"),
        OsString::from("libcufft.so.10"),
        OsString::from("libcufft.so"),
    ];
    assert_eq!(cufft_library_candidates(None), expected);
    assert_eq!(cufft_library_candidates(Some(OsStr::new(":"))), expected);
}

#[test]
fn candidate_only_load_failure_preserves_typed_source_without_process_environment() {
    let candidate = OsString::from("/definitely/missing/libcufft-for-unit-test.so");
    let error = match CufftLibrary::load_from_paths_for_tests(vec![candidate.clone()]) {
        Ok(_) => panic!("a missing candidate must not load"),
        Err(error) => error,
    };
    assert!(matches!(
        &error,
        CudaFftError::LibraryLoad { paths, .. } if paths == "/definitely/missing/libcufft-for-unit-test.so"
    ));
    assert!(error.source().is_some(), "loader source must be retained");
}

#[test]
fn cufft_success_status_maps_to_ok() {
    assert!(map_cufft_status("cufftCreate", CUFFT_SUCCESS).is_ok());
}

#[test]
fn representative_cufft_failures_preserve_function_and_status() {
    for status in [
        CUFFT_INVALID_PLAN,
        CUFFT_ALLOC_FAILED,
        CUFFT_INVALID_TYPE,
        CUFFT_INVALID_VALUE,
        CUFFT_INTERNAL_ERROR,
        CUFFT_EXEC_FAILED,
        CUFFT_SETUP_FAILED,
        CUFFT_INVALID_SIZE,
        CUFFT_UNALIGNED_DATA,
    ] {
        let error = map_cufft_status("cufftTest", status).expect_err("status should fail");
        assert!(matches!(
            error,
            CudaFftError::CufftStatus {
                function: "cufftTest",
                status: actual,
            } if actual == status
        ));
    }
}

#[test]
fn loader_failures_are_io_errors_with_a_typed_source_chain() {
    // SAFETY: the test opens only a deliberately missing path and never uses
    // a returned library handle.
    let missing_source = unsafe { libloading::Library::new("/definitely/missing/libcufft.so") }
        .expect_err("missing library path should produce a loader source");
    let library_error = CudaFftError::LibraryLoad {
        paths: "missing".to_owned(),
        attempts: "missing".to_owned(),
        source: missing_source,
    };
    let tensor_error = into_tensor_error("cuda_fft", library_error);
    assert!(matches!(
        &tensor_error,
        tenferro_tensor::Error::IoSource { op: "cuda_fft", .. }
    ));
    assert_eq!(tensor_error.kind(), tenferro_tensor::ErrorKind::Io);
    let source = tensor_error
        .source()
        .expect("I/O error must retain CudaFftError source");
    assert!(source.downcast_ref::<CudaFftError>().is_some());
}

#[test]
fn symbol_load_failures_are_io_errors_with_a_typed_source_chain() {
    // SAFETY: the test opens only a deliberately missing path and never uses
    // a returned library handle.
    let missing_source = unsafe { libloading::Library::new("/definitely/missing/libcufft.so") }
        .expect_err("missing library path should produce a loader source");
    let symbol_error = CudaFftError::SymbolLoad {
        name: "cufftCreate".to_owned(),
        source: missing_source,
    };
    let tensor_error = into_tensor_error("cuda_fft", symbol_error);
    assert!(matches!(
        &tensor_error,
        tenferro_tensor::Error::IoSource { op: "cuda_fft", .. }
    ));
    assert_eq!(tensor_error.kind(), tenferro_tensor::ErrorKind::Io);
    assert!(tensor_error.source().is_some());
}

#[test]
fn vendor_status_and_interop_failures_are_backend_errors_with_typed_sources() {
    for source in [
        CudaFftError::CufftStatus {
            function: "cufftExecC2C",
            status: 1,
        },
        CudaFftError::test_interop("cufft_execute"),
    ] {
        let tensor_error = into_tensor_error("cuda_fft", source);
        assert!(matches!(
            &tensor_error,
            tenferro_tensor::Error::BackendSource { op: "cuda_fft", .. }
        ));
        assert_eq!(
            tensor_error.kind(),
            tenferro_tensor::ErrorKind::BackendFailure
        );
        assert!(tensor_error.source().is_some());
    }
}
