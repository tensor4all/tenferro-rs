// Run with: cargo test -p tenferro-gpu --features cuda -- --ignored
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;

use crate::cubecl::memory::{download_tensor, upload_tensor};
use crate::cubecl::{
    gpu_available, with_cuda_exec_session, CudaBackend, CudaDeviceId, CudaRuntime,
};
use crate::{Error, Tensor};
use tenferro_tensor::backend::BackendSessionHost;
use tenferro_tensor::TensorRead;

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
            assert!(gpu_available(), "CUDA test requires an available device");
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
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    let gpu = upload_tensor(backend.runtime(), &host).unwrap();
    let Some(gpu) = gpu.as_typed::<f64>() else {
        unreachable!("f64 upload should preserve dtype");
    };
    backend
        .with_backend_session(|session| {
            with_cuda_exec_session(session, |exec| {
                // SAFETY: the raw tensor ref validates exact runtime residency
                // and keeps the GPU buffer borrowed for the callback.
                exec.with_raw("test_pointer_bridge", |raw| unsafe {
                    let ptr = raw.tensor(gpu)?.raw_ptr();
                    assert!(!ptr.is_null(), "Device pointer should be non-null");
                    Ok(())
                })
            })
            .expect("CUDA backend session should be available")
        })
        .expect("raw session should run");
});

gpu_test!(test_backend_add_matches_cpu_reference, {
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let mut cpu = tenferro_cpu::CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let gpu_a = upload_tensor(backend.runtime(), &a).unwrap();
    let gpu_b = upload_tensor(backend.runtime(), &b).unwrap();
    let expected = cpu
        .with_backend_session(|__s| {
            __s.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        })
        .unwrap();
    let actual_gpu = backend
        .with_backend_session(|__s| {
            __s.add_read(
                TensorRead::from_tensor(&gpu_a),
                TensorRead::from_tensor(&gpu_b),
            )
        })
        .unwrap();
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

gpu_test!(
    test_zero_stride_view_materializes_without_cutensor_descriptor,
    {
        use tenferro_tensor::TensorValue;
        let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
        let host = Tensor::from_vec_col_major([2], vec![2.0_f64, 5.0]).unwrap();
        let input = upload_tensor(backend.runtime(), &host).unwrap();
        let view = TensorValue::from_tensor(input)
            .broadcast_in_dim_view([3, 2], [1])
            .unwrap();
        let output = backend
            .with_backend_session(|__s| __s.to_contiguous_read(view.tensor_read()))
            .unwrap();
        assert_eq!(
            backend
                .cutensor_permutation_plan_cache_stats()
                .unwrap()
                .entries,
            0
        );
        let actual = download_tensor(backend.runtime(), &output).unwrap();
        assert_eq!(actual.shape(), &[3, 2]);
        assert_eq!(
            actual.as_slice::<f64>().unwrap(),
            &[2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
        );
    }
);

gpu_test!(test_cached_scalar_read_observes_queued_writes, {
    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let output = upload_tensor(
        &rt,
        &Tensor::from_vec_col_major([1], vec![0.0_f64]).unwrap(),
    )
    .unwrap();
    let Some(typed) = output.as_typed::<f64>() else {
        unreachable!()
    };
    let handle = super::super::dispatch::cubecl_buffer(typed, "test")
        .unwrap()
        .handle()
        .clone();
    let a = rt.client().create_from_slice(f64::as_bytes(&[1.0]));
    let b = rt.client().create_from_slice(f64::as_bytes(&[2.0]));
    // Prime both pointer lookup and the raw stream before issuing more work.
    assert_eq!(
        download_tensor(&rt, &output)
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[0.0]
    );
    for i in 0..32 {
        let rhs = if i % 2 == 0 { &a } else { &b };
        // SAFETY: all handles contain one f64; output is fresh and is only read
        // after the runtime's stream-ordered download completes each iteration.
        unsafe {
            kernel_add_f64::launch_unchecked::<CubeclCudaRuntime>(
                rt.client(),
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(handle.clone(), 1),
                ArrayArg::from_raw_parts(a.clone(), 1),
                ArrayArg::from_raw_parts(rhs.clone(), 1),
            );
        }
        let actual = download_tensor(&rt, &output).unwrap();
        assert_eq!(
            actual.as_slice::<f64>().unwrap(),
            &[if i % 2 == 0 { 2.0 } else { 3.0 }]
        );
    }
});

gpu_test!(test_complex_sign_is_scale_safe, {
    use num_complex::{Complex32, Complex64};
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(
        [4],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(3e-200, 4e-200),
            Complex64::new(3e200, 4e200),
            Complex64::new(-3.0, 4.0),
        ],
    )
    .unwrap();
    let input = upload_tensor(backend.runtime(), &host).unwrap();
    let output = backend
        .with_backend_session(|__s| __s.sign_read(TensorRead::from_tensor(&input)))
        .unwrap();
    let actual = download_tensor(backend.runtime(), &output).unwrap();
    for (&value, expected) in actual.as_slice::<Complex64>().unwrap().iter().zip([
        Complex64::new(0.0, 0.0),
        Complex64::new(0.6, 0.8),
        Complex64::new(0.6, 0.8),
        Complex64::new(-0.6, 0.8),
    ]) {
        assert!(
            (value - expected).norm() < 1e-12,
            "{value:?} != {expected:?}"
        );
    }
    let host = Tensor::from_vec_col_major(
        [3],
        vec![
            Complex32::new(0.0, 0.0),
            Complex32::new(3e-30, 4e-30),
            Complex32::new(3e30, 4e30),
        ],
    )
    .unwrap();
    let input = upload_tensor(backend.runtime(), &host).unwrap();
    let output = backend
        .with_backend_session(|__s| __s.sign_read(TensorRead::from_tensor(&input)))
        .unwrap();
    let actual = download_tensor(backend.runtime(), &output).unwrap();
    for (&value, expected) in actual.as_slice::<Complex32>().unwrap().iter().zip([
        Complex32::new(0.0, 0.0),
        Complex32::new(0.6, 0.8),
        Complex32::new(0.6, 0.8),
    ]) {
        assert!(
            (value - expected).norm() < 1e-6,
            "{value:?} != {expected:?}"
        );
    }
});

#[test]
fn raw_synchronization_submits_cubecl_work_before_waiting() {
    // Guard the host-queue/driver boundary even when the queue happens to drain
    // early enough to hide stale reads in a hardware regression test.
    let source = include_str!("../runtime.rs");
    let body = source
        .split_once("    fn synchronize(&self) -> crate::Result<()> {")
        .unwrap()
        .1;
    let body = body.split_once("\n    }").unwrap().0;
    let flush = body
        .find("self.flush_cubecl(OP)?")
        .expect("must submit the host queue");
    let stream = body.find("self.raw_cuda_stream()?").unwrap();
    assert!(flush < stream);
}

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
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let t = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let gpu = upload_tensor(backend.runtime(), &t).unwrap();

    let Some(gpu_typed) = gpu.as_typed::<f64>() else {
        unreachable!("f64 upload should preserve dtype");
    };
    backend
        .with_backend_session(|session| {
            with_cuda_exec_session(session, |exec| {
                // SAFETY: the raw tensor ref validates exact runtime residency
                // and keeps the GPU buffer borrowed for the callback.
                exec.with_raw("test_pointer_and_stream_bridge", |raw| {
                    let stream = unsafe { raw.stream().raw_handle() };
                    assert!(stream != 0);
                    // SAFETY: the pointer is checked for non-null before the
                    // session-scoped borrow ends.
                    let ptr = unsafe { raw.tensor(gpu_typed)?.raw_ptr() };
                    assert!(!ptr.is_null(), "Device pointer should be non-null");
                    Ok(())
                })
            })
            .expect("CUDA backend session should be available")
        })
        .expect("raw session should run");

    let back = download_tensor(backend.runtime(), &gpu).unwrap();
    assert_eq!(back.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
});
