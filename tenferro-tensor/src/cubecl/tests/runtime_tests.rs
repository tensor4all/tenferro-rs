use crate::cubecl::memory::{device_ptr, download_tensor, upload_tensor};
use crate::cubecl::CubeclRuntime;
use crate::Tensor;

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

#[test]
fn test_upload_download_f64() {
    let rt = CubeclRuntime::new(0).unwrap();
    let host = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let gpu = upload_tensor(&rt, &host).unwrap();

    assert_eq!(gpu.dtype(), crate::DType::F64);

    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(back.shape(), host.shape());
    assert_eq!(
        back.as_slice::<f64>().unwrap(),
        host.as_slice::<f64>().unwrap()
    );
}

#[test]
fn test_upload_download_c64() {
    use num_complex::Complex64;

    let rt = CubeclRuntime::new(0).unwrap();
    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let host = Tensor::new(vec![2], data.clone());

    let gpu = upload_tensor(&rt, &host).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();

    assert_eq!(back.as_slice::<Complex64>().unwrap(), &data);
}

#[test]
fn test_pointer_bridge() {
    let rt = CubeclRuntime::new(0).unwrap();
    let host = Tensor::new(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]);

    let gpu = upload_tensor(&rt, &host).unwrap();
    let ptr = device_ptr(&rt, &gpu).unwrap();

    assert!(ptr != 0, "Device pointer should be non-null");
}
