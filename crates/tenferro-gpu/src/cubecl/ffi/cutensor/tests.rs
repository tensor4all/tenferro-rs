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
