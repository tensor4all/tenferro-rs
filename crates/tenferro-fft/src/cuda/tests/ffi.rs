use std::error::Error;
use std::ffi::{OsStr, OsString};

use super::super::error::CudaFftError;
use super::super::ffi::{
    cufft_library_candidates, map_cufft_status, CufftLibrary, CUFFT_ALLOC_FAILED,
    CUFFT_EXEC_FAILED, CUFFT_INTERNAL_ERROR, CUFFT_INVALID_PLAN, CUFFT_INVALID_SIZE,
    CUFFT_INVALID_TYPE, CUFFT_INVALID_VALUE, CUFFT_SETUP_FAILED, CUFFT_SUCCESS,
    CUFFT_UNALIGNED_DATA,
};

#[test]
fn library_candidates_try_override_then_cuda12_cuda11_and_bare_sonames() {
    let candidates = cufft_library_candidates(Some(OsStr::new(
        "/opt/cuda-a/libcufft.so:/opt/cuda-b/libcufft.so",
    )));

    assert_eq!(
        candidates,
        vec![
            OsString::from("/opt/cuda-a/libcufft.so"),
            OsString::from("/opt/cuda-b/libcufft.so"),
            OsString::from("libcufft.so.12"),
            OsString::from("libcufft.so.11"),
            OsString::from("libcufft.so"),
        ]
    );
}

#[test]
fn library_candidates_use_defaults_when_override_is_absent_or_empty() {
    let expected = vec![
        OsString::from("libcufft.so.12"),
        OsString::from("libcufft.so.11"),
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
