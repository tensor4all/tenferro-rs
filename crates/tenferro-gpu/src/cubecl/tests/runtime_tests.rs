// Run with: cargo test -p tenferro-gpu --features cuda -- --ignored
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;

use crate::cubecl::interop::with_typed_device_ptr;
use crate::cubecl::memory::{download_tensor, upload_tensor};
use crate::cubecl::{gpu_available, CudaBackend, CudaDeviceId, CudaRuntime};
use crate::{Error, Tensor};
use tenferro_tensor::TensorElementwise;

#[cube(launch_unchecked)]
fn kernel_add_f64(output: &mut Array<f64>, a: &Array<f64>, b: &Array<f64>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

macro_rules! gpu_test {
    ($name:ident, $body:expr) => {
        #[test]
        #[ignore = "requires CUDA 12.8+ GPU"]
        fn $name() {
            if !gpu_available() {
                eprintln!("skipping {} — no CUDA device found", stringify!($name));
                return;
            }
            $body
        }
    };
}

#[test]
fn cube_count_for_len_rejects_u32_overflow() {
    let len = (u32::MAX as usize + 1) * super::super::dispatch::DEFAULT_CUBE_DIM_X as usize;
    let err = super::super::dispatch::cube_count_for_len(len).unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            op: "cube_count_for_len",
            source: tenferro_tensor::ValidationError::InvalidArgument {
                argument: "length",
                ..
            },
        }
    ));
}

gpu_test!(test_runtime_init, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0));
    assert!(rt.is_ok(), "CubeCL runtime should init on device 0");
});

gpu_test!(test_raw_stream_extraction, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let stream_ptr = rt.raw_cuda_stream().unwrap();
    assert!(stream_ptr != 0, "Raw CUstream should be non-null");
});

gpu_test!(test_upload_download_f64, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let gpu = upload_tensor(&rt, &host).unwrap();

    assert_eq!(gpu.dtype(), crate::DType::F64);

    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(back.shape(), host.shape());
    assert_eq!(
        back.as_slice::<f64>().unwrap(),
        host.as_slice::<f64>().unwrap()
    );
});

gpu_test!(test_upload_download_i64, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![2, 3], vec![1_i64, -2, 3, -4, 5, -6]).unwrap();
    let gpu = upload_tensor(&rt, &host).unwrap();

    assert_eq!(gpu.dtype(), crate::DType::I64);

    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(back.shape(), host.shape());
    assert_eq!(
        back.as_slice::<i64>().unwrap(),
        host.as_slice::<i64>().unwrap()
    );
});

gpu_test!(test_upload_download_i32, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![2, 3], vec![1_i32, -2, 3, -4, 5, -6]).unwrap();
    let gpu = upload_tensor(&rt, &host).unwrap();

    assert_eq!(gpu.dtype(), crate::DType::I32);

    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(back.shape(), host.shape());
    assert_eq!(
        back.as_slice::<i32>().unwrap(),
        host.as_slice::<i32>().unwrap()
    );
});

gpu_test!(test_upload_download_bool, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![2, 3], vec![true, false, true, true, false, false])
        .unwrap();
    let gpu = upload_tensor(&rt, &host).unwrap();

    assert_eq!(gpu.dtype(), crate::DType::Bool);

    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(back.shape(), host.shape());
    assert_eq!(
        back.as_slice::<bool>().unwrap(),
        host.as_slice::<bool>().unwrap()
    );
});

gpu_test!(test_download_empty_host_f64_rejects_before_fast_path, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap();

    let err = download_tensor(&rt, &host).unwrap_err();

    assert_download_rejects_host_tensor_before_empty_fast_path(err);
});

gpu_test!(test_download_empty_host_bool_rejects_before_fast_path, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![0], Vec::<bool>::new()).unwrap();

    let err = download_tensor(&rt, &host).unwrap_err();

    assert_download_rejects_host_tensor_before_empty_fast_path(err);
});

fn assert_download_rejects_host_tensor_before_empty_fast_path(err: Error) {
    assert!(matches!(err, Error::RuntimeState { .. }));
}

gpu_test!(test_upload_download_c64, {
    use num_complex::Complex64;

    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let host = Tensor::from_vec_col_major(vec![2], data.clone()).unwrap();

    let gpu = upload_tensor(&rt, &host).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();

    assert_eq!(back.as_slice::<Complex64>().unwrap(), &data);
});

gpu_test!(test_pointer_bridge, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    let gpu = upload_tensor(&rt, &host).unwrap();
    let Tensor::F64(gpu) = &gpu else {
        unreachable!("f64 upload should preserve dtype");
    };
    let ptr = with_typed_device_ptr(&rt, gpu, "test_pointer_bridge", |ptr| ptr).unwrap();

    assert!(!ptr.is_null(), "Device pointer should be non-null");
});

gpu_test!(test_backend_add_matches_cpu_reference, {
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let mut cpu = tenferro_cpu::CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let gpu_a = upload_tensor(backend.runtime(), &a).unwrap();
    let gpu_b = upload_tensor(backend.runtime(), &b).unwrap();
    let expected = cpu.add(&a, &b).unwrap();
    let actual_gpu = backend.add(&gpu_a, &gpu_b).unwrap();
    let actual = download_tensor(backend.runtime(), &actual_gpu).unwrap();
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(
        actual.as_slice::<f64>().unwrap(),
        expected.as_slice::<f64>().unwrap()
    );
});

gpu_test!(test_trivial_cube_kernel, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let client = rt.client();

    let a_data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let b_data = vec![10.0_f64, 20.0, 30.0, 40.0];
    let expected = vec![11.0_f64, 22.0, 33.0, 44.0];
    let n = a_data.len();

    let handle_a = client.create_from_slice(f64::as_bytes(&a_data));
    let handle_b = client.create_from_slice(f64::as_bytes(&b_data));
    let handle_out = client.empty(n * std::mem::size_of::<f64>());

    unsafe {
        kernel_add_f64::launch_unchecked::<CubeclCudaRuntime>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(n as u32),
            ArrayArg::from_raw_parts(handle_out.clone(), n),
            ArrayArg::from_raw_parts(handle_a, n),
            ArrayArg::from_raw_parts(handle_b, n),
        );
    }

    let result_bytes = client.read_one_unchecked(handle_out);
    let result = f64::from_bytes(&result_bytes);
    assert_eq!(result, &expected);
});

gpu_test!(test_full_round_trip_all_dtypes, {
    use num_complex::{Complex32, Complex64};

    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();

    let t = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<f64>().unwrap(),
        t.as_slice::<f64>().unwrap()
    );

    let t = Tensor::from_vec_col_major(vec![3], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<f32>().unwrap(),
        t.as_slice::<f32>().unwrap()
    );

    let t = Tensor::from_vec_col_major(vec![3], vec![1_i64, -2, 3]).unwrap();
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<i64>().unwrap(),
        t.as_slice::<i64>().unwrap()
    );

    let t = Tensor::from_vec_col_major(vec![3], vec![1_i32, -2, 3]).unwrap();
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<i32>().unwrap(),
        t.as_slice::<i32>().unwrap()
    );

    let t = Tensor::from_vec_col_major(vec![4], vec![true, false, false, true]).unwrap();
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<bool>().unwrap(),
        t.as_slice::<bool>().unwrap()
    );

    let t = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    )
    .unwrap();
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<Complex64>().unwrap(),
        t.as_slice::<Complex64>().unwrap()
    );

    let t = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
    )
    .unwrap();
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<Complex32>().unwrap(),
        t.as_slice::<Complex32>().unwrap()
    );
});

gpu_test!(test_pointer_and_stream_bridge, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let t = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let gpu = upload_tensor(&rt, &t).unwrap();

    let Tensor::F64(gpu_typed) = &gpu else {
        unreachable!("f64 upload should preserve dtype");
    };
    let ptr =
        with_typed_device_ptr(&rt, gpu_typed, "test_pointer_and_stream_bridge", |ptr| ptr).unwrap();
    assert!(!ptr.is_null());

    let stream = rt.raw_cuda_stream().unwrap();
    assert!(stream != 0);

    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(back.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
});
