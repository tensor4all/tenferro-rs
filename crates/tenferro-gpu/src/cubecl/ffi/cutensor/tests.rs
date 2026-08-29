#[test]
fn cutensor_loader_missing_library_is_typed_io_error() {
    let err = match super::CutensorLibrary::load_from_paths(vec![
        "/definitely/not/libcutensor.so".to_string()
    ]) {
        Ok(_) => panic!("unexpectedly loaded cuTENSOR from a nonexistent path"),
        Err(err) => err,
    };

    match err {
        crate::Error::IoSource { op, source } => {
            assert_eq!(op, "cuda_cutensor");
            assert!(source
                .to_string()
                .contains("failed to load cuTENSOR library"));
        }
        other => panic!("expected typed cuTENSOR IoSource, got {other:?}"),
    }
}

#[test]
fn cutensor_volta_support_matches_vendor_version_boundary() {
    let volta = super::CudaComputeCapability { major: 7, minor: 0 };
    let turing = super::CudaComputeCapability { major: 7, minor: 5 };

    super::validate_device_support(20_100, volta).unwrap();
    super::validate_device_support(20_200, turing).unwrap();

    let error = super::validate_device_support(20_200, volta).unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
    assert!(error
        .to_string()
        .contains("cuTENSOR 2.2.0 is unsupported by tenferro on SM 7.0"));
}
