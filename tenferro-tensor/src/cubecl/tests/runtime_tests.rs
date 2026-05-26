// Run with: cargo test -p tenferro-internal-tensor --features cuda -- --ignored
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;

use crate::cubecl::memory::{device_ptr, download_tensor, upload_tensor};
use crate::cubecl::{gpu_available, CubeclBackend, CubeclRuntime};
use crate::Tensor;
use crate::TensorBackend;

#[cube(launch_unchecked)]
fn kernel_add_f64(output: &mut Array<f64>, a: &Array<f64>, b: &Array<f64>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

macro_rules! gpu_test {
    ($name:ident, $body:expr) => {
        #[test]
        #[ignore = "requires CUDA 12+ GPU"]
        fn $name() {
            if !gpu_available() {
                eprintln!("skipping {} — no CUDA device found", stringify!($name));
                return;
            }
            $body
        }
    };
}

gpu_test!(test_runtime_init, {
    let rt = CubeclRuntime::new(0);
    assert!(rt.is_ok(), "CubeCL runtime should init on device 0");
});

gpu_test!(test_raw_stream_extraction, {
    let rt = CubeclRuntime::new(0).unwrap();
    let stream_ptr = rt.raw_cuda_stream().unwrap();
    assert!(stream_ptr != 0, "Raw CUstream should be non-null");
});

gpu_test!(test_upload_download_f64, {
    let rt = CubeclRuntime::new(0).unwrap();
    let host = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
    let rt = CubeclRuntime::new(0).unwrap();
    let host = Tensor::from_vec_col_major(vec![2, 3], vec![1_i64, -2, 3, -4, 5, -6]);
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
    let rt = CubeclRuntime::new(0).unwrap();
    let host = Tensor::from_vec_col_major(vec![2, 3], vec![1_i32, -2, 3, -4, 5, -6]);
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
    let rt = CubeclRuntime::new(0).unwrap();
    let host = Tensor::from_vec_col_major(vec![2, 3], vec![true, false, true, true, false, false]);
    let gpu = upload_tensor(&rt, &host).unwrap();

    assert_eq!(gpu.dtype(), crate::DType::Bool);

    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(back.shape(), host.shape());
    assert_eq!(
        back.as_slice::<bool>().unwrap(),
        host.as_slice::<bool>().unwrap()
    );
});

gpu_test!(test_upload_download_c64, {
    use num_complex::Complex64;

    let rt = CubeclRuntime::new(0).unwrap();
    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let host = Tensor::from_vec_col_major(vec![2], data.clone());

    let gpu = upload_tensor(&rt, &host).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();

    assert_eq!(back.as_slice::<Complex64>().unwrap(), &data);
});

gpu_test!(test_pointer_bridge, {
    let rt = CubeclRuntime::new(0).unwrap();
    let host = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]);

    let gpu = upload_tensor(&rt, &host).unwrap();
    let ptr = device_ptr(&rt, &gpu).unwrap();

    assert!(ptr != 0, "Device pointer should be non-null");
});

gpu_test!(test_backend_add_matches_cpu_reference, {
    let mut backend = CubeclBackend::new(0).unwrap();
    let mut cpu = crate::cpu::CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);
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
    let rt = CubeclRuntime::new(0).unwrap();
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

    let rt = CubeclRuntime::new(0).unwrap();

    let t = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<f64>().unwrap(),
        t.as_slice::<f64>().unwrap()
    );

    let t = Tensor::from_vec_col_major(vec![3], vec![1.0_f32, 2.0, 3.0]);
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<f32>().unwrap(),
        t.as_slice::<f32>().unwrap()
    );

    let t = Tensor::from_vec_col_major(vec![3], vec![1_i64, -2, 3]);
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<i64>().unwrap(),
        t.as_slice::<i64>().unwrap()
    );

    let t = Tensor::from_vec_col_major(vec![3], vec![1_i32, -2, 3]);
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<i32>().unwrap(),
        t.as_slice::<i32>().unwrap()
    );

    let t = Tensor::from_vec_col_major(vec![4], vec![true, false, false, true]);
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<bool>().unwrap(),
        t.as_slice::<bool>().unwrap()
    );

    let t = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    );
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<Complex64>().unwrap(),
        t.as_slice::<Complex64>().unwrap()
    );

    let t = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
    );
    let gpu = upload_tensor(&rt, &t).unwrap();
    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(
        back.as_slice::<Complex32>().unwrap(),
        t.as_slice::<Complex32>().unwrap()
    );
});

gpu_test!(test_pointer_and_stream_bridge, {
    let rt = CubeclRuntime::new(0).unwrap();
    let t = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let gpu = upload_tensor(&rt, &t).unwrap();

    let ptr = device_ptr(&rt, &gpu).unwrap();
    assert!(ptr != 0);

    let stream = rt.raw_cuda_stream().unwrap();
    assert!(stream != 0);

    let back = download_tensor(&rt, &gpu).unwrap();
    assert_eq!(back.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
});
