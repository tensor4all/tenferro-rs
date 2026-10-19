// Run with: cargo test -p tenferro-gpu --features cuda -- --ignored
use cubecl::prelude::*;

use crate::cubecl::memory::{download_tensor, upload_tensor};
use crate::cubecl::{gpu_available, with_cuda_exec_session, CudaDeviceId, CudaEngine, CudaRuntime};
use crate::{Error, Tensor};
use tenferro_tensor::backend::BackendSessionHost;
use tenferro_tensor::TensorRead;

#[cube(launch_unchecked)]
fn kernel_add_f64(output: &mut [f64], a: &[f64], b: &[f64]) {
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
    let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).unwrap();
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
        .unwrap()
        .expect("raw session should run");
});

gpu_test!(test_backend_add_matches_cpu_reference, {
    let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let mut cpu = tenferro_cpu::CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let gpu_a = upload_tensor(backend.runtime(), &a).unwrap();
    let gpu_b = upload_tensor(backend.runtime(), &b).unwrap();
    let expected = cpu
        .with_backend_session(|__s| {
            __s.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        })
        .unwrap()
        .unwrap();
    let actual_gpu = backend
        .with_backend_session(|__s| {
            __s.add_read(
                TensorRead::from_tensor(&gpu_a),
                TensorRead::from_tensor(&gpu_b),
            )
        })
        .unwrap()
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
        kernel_add_f64::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(n as u32),
            BufferArg::from_raw_parts(handle_out.clone(), n),
            BufferArg::from_raw_parts(handle_a, n),
            BufferArg::from_raw_parts(handle_b, n),
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
        let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).unwrap();
        let host = Tensor::from_vec_col_major([2], vec![2.0_f64, 5.0]).unwrap();
        let input = upload_tensor(backend.runtime(), &host).unwrap();
        let view = TensorValue::from_tensor(input)
            .broadcast_in_dim_view([3, 2], [1])
            .unwrap();
        let output = backend
            .with_backend_session(|__s| __s.to_contiguous_read(view.tensor_read()))
            .unwrap()
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
            kernel_add_f64::launch_unchecked(
                rt.client(),
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(handle.clone(), 1),
                BufferArg::from_raw_parts(a.clone(), 1),
                BufferArg::from_raw_parts(rhs.clone(), 1),
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
    let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).unwrap();
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
        .unwrap()
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
        .unwrap()
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
    let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).unwrap();
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
        .unwrap()
        .expect("raw session should run");

    let back = download_tensor(backend.runtime(), &gpu).unwrap();
    assert_eq!(back.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
});

// A dropped tensor returns its block to CubeCL's pool for the device, not to the
// driver, and the pool's reservations never shrink on their own. The explicit
// trim is what returns it (tenferro-rs #2081).
gpu_test!(test_trim_memory_returns_dropped_blocks_to_the_driver, {
    use cudarc::driver::result as cuda_result;

    let free_bytes = || -> u64 {
        cuda_result::init().expect("driver init");
        let (free, _total) = cuda_result::mem_get_info().expect("free memory");
        free as u64
    };

    let rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let gib = 1usize << 30;
    let elements = gib / 8;
    let tensor = Tensor::from_vec_col_major(vec![elements], vec![0.0f64; elements]).unwrap();

    let free_before = free_bytes();
    let gpu = upload_tensor(&rt, &tensor).unwrap();
    rt.synchronize().unwrap();
    let free_while_alive = free_bytes();
    assert!(
        free_before.saturating_sub(free_while_alive) >= (gib - (64 << 20)) as u64,
        "the 1 GiB upload should be visible to the driver"
    );

    drop(gpu);
    let free_after_drop = free_bytes();
    // Dropping alone must not be enough: the block stays reserved in the pool.
    assert!(
        free_after_drop.saturating_add(64 << 20) < free_before,
        "the dropped tensor was expected to stay reserved in the pool before the trim \
         (free_before {free_before}, free_after_drop {free_after_drop})"
    );
    rt.trim_memory();
    // `memory_cleanup` submits the release to the allocator's server thread and
    // returns; the stream synchronize lets that work finish before the read.
    rt.synchronize().unwrap();
    let free_after_trim = free_bytes();

    let recovered = free_after_trim.saturating_sub(free_after_drop);
    assert!(
        recovered >= (gib - (64 << 20)) as u64,
        "trim_memory returned {recovered} of the dropped 1 GiB (free {free_after_drop} -> \
         {free_after_trim}); the pool reservation did not reach the driver"
    );
    assert!(
        free_after_trim.saturating_add(64 << 20) >= free_before,
        "trim_memory should restore the driver free memory seen before the upload \
         ({free_before} before, {free_after_trim} after)"
    );
});

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn prepared_run_dispatches_cubecl_launches_before_recording_retirement() {
    use crate::cubecl::runtime::{blocking_flush_handoffs, reset_blocking_flush_handoffs};
    use tenferro_runtime::{DType, EngineId, GraphCompiler, Runtime, TracedTensor};

    assert!(gpu_available(), "CUDA test requires an available device");
    let backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("backend");
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            crate::cubecl::cuda_runtime_engine_registration(
                &backend,
                EngineId::new("tenferro-cuda.test.event-flush.v1").expect("engine id"),
            )
            .expect("cuda registration"),
        )
        .expect("register engine");
    let runtime = builder.build().expect("runtime");

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).expect("input");
    let y = (&x * &x).expect("square");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .expect("compile");
    let host = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0]).expect("host");
    let gpu_input = upload_tensor(backend.runtime(), &host).expect("upload");
    let prepared = runtime
        .prepare_compiled(&program, &[&gpu_input])
        .expect("prepare");

    // The CUDA event-domain run must dispatch queued CubeCL launches before it
    // records the completion event; otherwise `retire_cuda_run` could
    // synchronize a stream the launched kernels never reached (#2182 B3). This
    // counter is raised only by `CudaRuntime::flush_cubecl`, and the dispatch
    // helpers the run uses do not raise it.
    reset_blocking_flush_handoffs();
    let _output = runtime
        .run_prepared(&prepared, &[&gpu_input])
        .expect("prepared run");
    assert!(
        blocking_flush_handoffs() >= 1,
        "the CUDA event-domain run did not dispatch its queued CubeCL launch before recording the retirement event"
    );
}

fn prepared_matmul_config() -> tenferro_tensor::DotGeneralConfig {
    tenferro_tensor::DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn prepared_matmul_chain_input() -> Tensor {
    const SIZE: usize = 16;
    Tensor::from_vec_col_major(
        vec![SIZE, SIZE],
        (0..SIZE * SIZE)
            .map(|index| ((index % 17) as f64 + 1.0) * 0.01)
            .collect(),
    )
    .expect("host input")
}

fn cpu_matmul_chain_reference(matmuls: usize) -> Tensor {
    let config = prepared_matmul_config();
    let source = prepared_matmul_chain_input();
    let rhs = prepared_matmul_chain_input();
    let mut cpu = tenferro_cpu::CpuBackend::new();
    cpu.with_backend_session(|session| {
        let mut accumulator = source;
        for _ in 0..matmuls {
            accumulator = session.dot_general_read(
                TensorRead::from_tensor(&accumulator),
                TensorRead::from_tensor(&rhs),
                &config,
            )?;
        }
        Ok::<_, tenferro_tensor::Error>(accumulator)
    })
    .expect("cpu host session")
    .expect("cpu matmul chain")
}

fn assert_tensors_close(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    let actual = actual.as_slice::<f64>().expect("actual f64");
    let expected = expected.as_slice::<f64>().expect("expected f64");
    let max_difference = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_difference < 1e-8,
        "CUDA prepared chain differs from CPU by {max_difference}"
    );
}

/// Build a runtime that owns a prepared eight-node CUDA matmul chain.
fn prepared_matmul_chain(
    backend: &CudaEngine,
    engine: &str,
) -> (
    tenferro_runtime::Runtime,
    tenferro_runtime::PreparedCompiledGraph,
    Tensor,
) {
    use tenferro_runtime::{DType, EngineId, GraphCompiler, Runtime, TracedTensor};

    let mut builder = Runtime::builder();
    builder
        .register_engine(
            crate::cubecl::cuda_runtime_engine_registration(
                backend,
                EngineId::new(engine).expect("engine id"),
            )
            .expect("cuda registration"),
        )
        .expect("register engine");
    let runtime = builder.build().expect("runtime");

    const SIZE: usize = 16;
    const MATMULS: usize = 8;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).expect("input");
    let mut y = x.matmul(&x).expect("first matmul");
    for _ in 1..MATMULS {
        y = y.matmul(&x).expect("chained matmul");
    }
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[SIZE, SIZE])])
        .expect("compile");
    let input = upload_tensor(backend.runtime(), &prepared_matmul_chain_input()).expect("upload");
    let prepared = runtime
        .prepare_compiled(&program, &[&input])
        .expect("prepare");
    (runtime, prepared, input)
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn prepared_single_domain_run_coalesces_scheduler_events() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("backend");
    let counters = backend.runtime();
    let (runtime, prepared, input) =
        prepared_matmul_chain(&backend, "tenferro-cuda.test.coalesce.v1");

    counters.reset_scheduler_counters();
    let output = runtime
        .run_prepared(&prepared, &[&input])
        .expect("prepared run");
    let events = counters.scheduler_event_records();
    let fences = counters.scheduler_fences();
    let retirements = counters.scheduler_retirements();
    assert_eq!(
        events, 0,
        "a single-domain run must not record per-node scheduler events ({events})"
    );
    assert_eq!(
        fences, 1,
        "a single-domain run must issue exactly one scheduler fence per call ({fences})"
    );
    assert_eq!(
        retirements, 1,
        "the retirement barrier must run before the run returns ({retirements})"
    );

    let actual = download_tensor(backend.runtime(), &output[0]).expect("download");
    let expected = cpu_matmul_chain_reference(8);
    assert_tensors_close(&actual, &expected);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn prepared_in_held_single_domain_run_coalesces_scheduler_events() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("backend");
    let counters = backend.runtime().clone();
    let (runtime, prepared, input) =
        prepared_matmul_chain(&backend, "tenferro-cuda.test.coalesce-held.v1");

    let mut session = backend.open_session().expect("held session");
    counters.reset_scheduler_counters();
    let output = session
        .with_session(|view| runtime.run_prepared_in(view, &prepared, &[&input]))
        .expect("held callback")
        .expect("prepared run");
    let events = counters.scheduler_event_records();
    let fences = counters.scheduler_fences();
    let retirements = counters.scheduler_retirements();
    assert_eq!(
        events, 0,
        "a held single-domain run must not record per-node scheduler events ({events})"
    );
    assert_eq!(
        fences, 1,
        "a held single-domain run must issue exactly one scheduler fence per call ({fences})"
    );
    assert_eq!(
        retirements, 1,
        "the held single-domain retirement barrier must run before the run returns ({retirements})"
    );
    session.close().expect("close");

    let actual = download_tensor(backend.runtime(), &output[0]).expect("download");
    let expected = cpu_matmul_chain_reference(8);
    assert_tensors_close(&actual, &expected);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn prepared_deferred_batch_retires_once_and_matches_synchronous_runs() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("backend");
    let counters = backend.runtime().clone();
    let (runtime, prepared, input) =
        prepared_matmul_chain(&backend, "tenferro-cuda.test.deferred-batch.v1");

    let synchronous = runtime
        .run_prepared(&prepared, &[&input])
        .expect("synchronous prepared run");
    let expected = download_tensor(backend.runtime(), &synchronous[0]).expect("download");

    let inputs: [&Tensor; 1] = [&input];
    let mut session = backend.open_session().expect("held session");
    counters.reset_scheduler_counters();
    let runs = session
        .with_session(|view| {
            let mut deferred = runtime.begin_prepared_deferred(view, &prepared)?;
            for _ in 0..8 {
                deferred.run_prepared_deferred(&inputs)?;
            }
            deferred.finish_prepared()
        })
        .expect("held callback")
        .expect("deferred prepared runs");
    session.close().expect("close");

    assert_eq!(runs.len(), 8);
    for run in &runs {
        let actual = download_tensor(backend.runtime(), &run[0]).expect("download run output");
        assert_tensors_close(&actual, &expected);
    }
    assert_eq!(
        counters.scheduler_event_records(),
        0,
        "a coalesced deferred batch must not record per-node scheduler events"
    );
    assert_eq!(
        counters.scheduler_fences(),
        1,
        "one finish must issue one scheduler dispatch fence for the whole batch"
    );
    assert_eq!(
        counters.scheduler_retirements(),
        1,
        "one finish must take one retirement barrier for the whole batch"
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn prepared_deferred_interleaves_cubecl_and_vendor_nodes() {
    use tenferro_runtime::{DType, EngineId, GraphCompiler, Runtime, TracedTensor};

    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("backend");
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            crate::cubecl::cuda_runtime_engine_registration(
                &backend,
                EngineId::new("tenferro-cuda.test.deferred-interleaved.v1").expect("engine id"),
            )
            .expect("cuda registration"),
        )
        .expect("register engine");
    let runtime = builder.build().expect("runtime");

    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).expect("input");
    let square = (&x * &x).expect("CubeCL square");
    let product = square
        .matmul(&x)
        .expect("vendor matmul reads CubeCL output");
    let y = (&product + &square).expect("CubeCL add reads vendor output");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[16, 16])])
        .expect("compile");

    // Distinct inputs make a reversed or misassociated output package fail.
    let base = prepared_matmul_chain_input();
    let base_values = base.as_slice::<f64>().expect("host values");
    let host_inputs = (0..4)
        .map(|run| {
            let values = base_values
                .iter()
                .map(|value| value + run as f64 * 0.05)
                .collect::<Vec<_>>();
            Tensor::from_vec_col_major(vec![16, 16], values).expect("host input")
        })
        .collect::<Vec<_>>();
    let gpu_inputs = host_inputs
        .iter()
        .map(|host| upload_tensor(backend.runtime(), host).expect("upload input"))
        .collect::<Vec<_>>();
    let run_inputs = gpu_inputs
        .iter()
        .map(|input| [input])
        .collect::<Vec<[&Tensor; 1]>>();
    let prepared = runtime
        .prepare_compiled(&program, &[&gpu_inputs[0]])
        .expect("prepare");
    let expected = gpu_inputs
        .iter()
        .map(|input| {
            let synchronous = runtime
                .run_prepared(&prepared, &[input])
                .expect("synchronous prepared run");
            download_tensor(backend.runtime(), &synchronous[0]).expect("download output")
        })
        .collect::<Vec<_>>();
    assert_ne!(
        expected[0].as_slice::<f64>().expect("host values"),
        expected[1].as_slice::<f64>().expect("host values"),
        "the distinct inputs must produce distinct references"
    );

    let mut session = backend.open_session().expect("held session");
    let runs = session
        .with_session(|view| {
            let mut deferred = runtime.begin_prepared_deferred(view, &prepared)?;
            let mut indices = Vec::with_capacity(run_inputs.len());
            for input in &run_inputs {
                indices.push(deferred.run_prepared_deferred(input)?);
            }
            assert_eq!(indices, (0..run_inputs.len()).collect::<Vec<_>>());
            deferred.finish_prepared()
        })
        .expect("held callback")
        .expect("deferred prepared runs");
    session.close().expect("close");

    assert_eq!(runs.len(), expected.len());
    for (index, run) in runs.iter().enumerate() {
        let actual = download_tensor(backend.runtime(), &run[0]).expect("download run output");
        assert_tensors_close(&actual, &expected[index]);
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn prepared_deferred_guard_drop_retires_pending_work() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("backend");
    let counters = backend.runtime().clone();
    let (runtime, prepared, input) =
        prepared_matmul_chain(&backend, "tenferro-cuda.test.deferred-drop.v1");
    let expected_program_output = runtime
        .run_prepared(&prepared, &[&input])
        .expect("synchronous prepared run");

    let inputs: [&Tensor; 1] = [&input];
    let mut session = backend.open_session().expect("held session");
    counters.reset_scheduler_counters();
    session
        .with_session(|view| {
            let mut deferred = runtime.begin_prepared_deferred(view, &prepared)?;
            deferred.run_prepared_deferred(&inputs)?;
            Ok::<(), tenferro_runtime::Error>(())
        })
        .expect("held callback")
        .expect("deferred run");
    assert_eq!(
        counters.scheduler_retirements(),
        1,
        "dropping an unfinished guard must retire its pending work once"
    );

    // The session and prepared handle stay usable after the implicit finish.
    let output = session
        .with_session(|view| runtime.run_prepared_in(view, &prepared, &inputs))
        .expect("held callback")
        .expect("prepared run after drop");
    session.close().expect("close");
    let actual = download_tensor(backend.runtime(), &output[0]).expect("download");
    let expected =
        download_tensor(backend.runtime(), &expected_program_output[0]).expect("download expected");
    assert_tensors_close(&actual, &expected);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn prepared_identity_run_retires_collection_before_returning() {
    use tenferro_runtime::{DType, EngineId, GraphCompiler, Runtime, TracedTensor};

    assert!(gpu_available(), "CUDA test requires an available device");
    let backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("backend");
    let counters = backend.runtime();
    let engine = EngineId::new("tenferro-cuda.test.coalesce-identity.v1").expect("engine id");
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            crate::cubecl::cuda_runtime_engine_registration(&backend, engine)
                .expect("cuda registration"),
        )
        .expect("register engine");
    let runtime = builder.build().expect("runtime");

    // Output placeholder is the input: no instruction writes the output slot,
    // so collection copies the borrowed device input before the run retires.
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).expect("input");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&x, &[(&x, DType::F64, &[3])])
        .expect("compile");
    let host = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 7.0, 9.0]).expect("host");
    let input = upload_tensor(backend.runtime(), &host).expect("upload");
    let prepared = runtime
        .prepare_compiled(&program, &[&input])
        .expect("prepare");

    counters.reset_scheduler_counters();
    let output = runtime
        .run_prepared(&prepared, &[&input])
        .expect("identity run");
    assert_eq!(counters.scheduler_event_records(), 0);
    assert_eq!(counters.scheduler_fences(), 1);
    assert_eq!(
        counters.scheduler_retirements(),
        1,
        "identity output collection must be covered by the retirement barrier"
    );
    let actual = download_tensor(backend.runtime(), &output[0]).expect("download");
    assert_eq!(
        actual.as_slice::<f64>().expect("host values"),
        &[4.0, 7.0, 9.0]
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn prepared_coalesced_error_retires_before_returning() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("owner backend");
    let counters = backend.runtime();
    let (runtime, prepared, owner_input) =
        prepared_matmul_chain(&backend, "tenferro-cuda.test.coalesce-error.v1");

    // A tensor resident on a different CUDA runtime is rejected by ingress
    // after the run's coalesced event domains were created, so the run must
    // still retire before the error returns.
    let foreign_backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("foreign backend");
    let foreign_input = upload_tensor(foreign_backend.runtime(), &prepared_matmul_chain_input())
        .expect("foreign upload");

    counters.reset_scheduler_counters();
    let error = runtime
        .run_prepared(&prepared, &[&foreign_input])
        .expect_err("a foreign allocation domain must fail ingress");
    assert!(
        error.to_string().contains("residency")
            || error.to_string().contains("allocation")
            || error.to_string().contains("ingress"),
        "unexpected ingress error: {error}"
    );
    assert_eq!(
        counters.scheduler_event_records(),
        0,
        "no coalesced node may run before the ingress rejection"
    );
    assert_eq!(
        counters.scheduler_retirements(),
        1,
        "the coalesced error path must retire before returning"
    );

    // The runtime and prepared handle stay usable after the retired error.
    let recovered = runtime
        .run_prepared(&prepared, &[&owner_input])
        .expect("owner run after the retired error");
    let actual = download_tensor(backend.runtime(), &recovered[0]).expect("download");
    assert_tensors_close(&actual, &cpu_matmul_chain_reference(8));
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_event_domain_keeps_per_node_events_until_coalescing_is_enabled() {
    use crate::cubecl::event_domain::CudaEventDomainDriver;
    use tenferro_runtime::{EngineId, EventDomainDriver, Runtime};

    assert!(gpu_available(), "CUDA test requires an available device");
    let backend = CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("backend");
    let counters = backend.runtime();
    let engine = EngineId::new("tenferro-cuda.test.ineligible.v1").expect("engine id");
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            crate::cubecl::cuda_runtime_engine_registration(&backend, engine.clone())
                .expect("cuda registration"),
        )
        .expect("register engine");
    let runtime = builder.build().expect("runtime");
    let domain = runtime
        .snapshot()
        .expect("snapshot")
        .engine(&engine)
        .expect("engine snapshot")
        .event_domain_id();

    let driver = CudaEventDomainDriver::new(backend.runtime().clone());
    let mut launch = || Ok(());

    // Without the scheduler's eligibility proof the run keeps the per-node path:
    // one completion event and one dispatch fence per node, plus retirement.
    let mut per_node = driver.begin_run(domain).expect("run");
    counters.reset_scheduler_counters();
    per_node.enqueue(&[], &mut launch).expect("first enqueue");
    per_node.enqueue(&[], &mut launch).expect("second enqueue");
    assert_eq!(counters.scheduler_event_records(), 2);
    assert_eq!(counters.scheduler_fences(), 2);
    assert_eq!(counters.scheduler_retirements(), 0);
    per_node.drain().expect("drain");
    assert_eq!(counters.scheduler_fences(), 3);
    assert_eq!(counters.scheduler_retirements(), 1);

    // After the scheduler enables coalescing, the same two nodes add no events
    // and the single retirement dispatch is the only fence and retirement.
    let mut coalesced = driver.begin_run(domain).expect("run");
    coalesced.enable_single_domain_coalescing();
    counters.reset_scheduler_counters();
    coalesced.enqueue(&[], &mut launch).expect("first enqueue");
    coalesced.enqueue(&[], &mut launch).expect("second enqueue");
    assert_eq!(counters.scheduler_event_records(), 0);
    assert_eq!(counters.scheduler_fences(), 0);
    assert_eq!(counters.scheduler_retirements(), 0);
    coalesced.drain().expect("drain");
    assert_eq!(counters.scheduler_fences(), 1);
    assert_eq!(counters.scheduler_retirements(), 1);
}
