use crate::cubecl::CubeclRuntime;

#[test]
fn test_runtime_init() {
    let rt = CubeclRuntime::new(0);
    assert!(rt.is_ok(), "CubeCL runtime should init on device 0");
}

#[test]
fn test_raw_stream_extraction() {
    let rt = CubeclRuntime::new(0).unwrap();
    let stream_ptr = rt.raw_cuda_stream().unwrap();
    assert!(stream_ptr != 0, "Raw CUstream should be non-null");
}
