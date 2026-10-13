#![cfg(feature = "cuda")]

use crate::support;
use support::{cpu_runtime, RunTraced};
use tenferro_ad::{EagerRuntime, EagerTensor, TracedTensorAdExt};
use tenferro_gpu::{
    cuda::gpu_available, cuda::upload_tensor, cuda::CudaBackend, cuda::CudaDeviceId,
};
use tenferro_runtime::{
    DType, DotGeneralConfig, GatherConfig, Tensor, TensorRead, TracedTensor, TypedTensor,
};
use tenferro_tensor::StorageBuffer;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn assert_f64_tensor_close(actual: &Tensor, expected: &Tensor, rtol: f64, atol: f64) {
    match (actual.as_typed::<f64>(), expected.as_typed::<f64>()) {
        (Some(actual), Some(expected)) => {
            assert_eq!(actual.shape(), expected.shape());
            for (idx, (&actual, &expected)) in actual
                .host_data()
                .unwrap()
                .iter()
                .zip(expected.host_data().unwrap().iter())
                .enumerate()
            {
                let tol = atol + rtol * expected.abs();
                assert!(
                    (actual - expected).abs() <= tol,
                    "index {idx}: actual={actual}, expected={expected}, tol={tol}"
                );
            }
        }
        _ => panic!("expected f64 tensors, got actual={actual:?} expected={expected:?}"),
    }
}

fn assert_device_backed(tensor: &Tensor) {
    fn is_cubecl<T: 'static>(buffer: &StorageBuffer<T>) -> bool {
        matches!(buffer, StorageBuffer::Backend(buffer) if buffer.backend_family() == "cubecl")
    }

    fn assert_typed<T: tenferro_tensor::TensorScalar>(tensor: &Tensor) {
        assert!(is_cubecl(
            tensor
                .as_typed::<T>()
                .expect("the dtype guard selects this arm")
                .buffer()
        ));
    }

    match tensor.dtype() {
        tenferro_tensor::DType::F32 => assert_typed::<f32>(tensor),
        tenferro_tensor::DType::F64 => assert_typed::<f64>(tensor),
        tenferro_tensor::DType::I64 => assert_typed::<i64>(tensor),
        tenferro_tensor::DType::C32 => assert_typed::<num_complex::Complex32>(tensor),
        tenferro_tensor::DType::C64 => assert_typed::<num_complex::Complex64>(tensor),
        tenferro_tensor::DType::I32 => assert_typed::<i32>(tensor),
        tenferro_tensor::DType::Bool => assert_typed::<bool>(tensor),
        // A caller-owned payload has no device buffer to inspect.
        _ => panic!("a caller-owned payload is not a device tensor"),
    }
}

fn eval_cpu_tensor(runtime: &tenferro_runtime::Runtime, tensor: &TracedTensor) -> Tensor {
    tensor.run_with(runtime).unwrap()
}

fn matmul(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        },
    )
    .unwrap()
}

#[test]
#[ignore = "requires CUDA"]
fn test_gpu_eager_backward_smoke() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let upload_backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let x_gpu = upload_tensor(
        upload_backend.runtime(),
        &f64_tensor(vec![2], vec![2.0_f64, 3.0]),
    )
    .unwrap();
    let seed_gpu = upload_tensor(
        upload_backend.runtime(),
        &f64_tensor(vec![2], vec![1.0_f64, 1.0]),
    )
    .unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
    let x = EagerTensor::requires_grad_in(x_gpu, ctx.clone()).unwrap();
    let seed = EagerTensor::from_tensor_in(seed_gpu, ctx.clone()).unwrap();
    let y = ctx
        .with_eager_session(|session| session.mul(&x, &x))
        .unwrap();

    y.backward_with(&seed).unwrap();
    let grad = x.grad().unwrap().unwrap();
    let grad_tensor = grad.to_tensor().unwrap();
    let grad_host = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&grad_tensor))
        })
        .unwrap()
        .unwrap();

    assert_f64_tensor_close(
        &grad_host,
        &f64_tensor(vec![2], vec![4.0_f64, 6.0]),
        1.0e-10,
        1.0e-10,
    );
}

#[test]
#[ignore = "requires CUDA"]
fn test_gpu_borrowed_reduce_sum_preserves_residency_and_gradient() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let x_gpu = upload_tensor(
        backend.runtime(),
        &f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
    )
    .unwrap();
    let seed_gpu = upload_tensor(backend.runtime(), &f64_tensor(vec![2], vec![1.0, 1.0])).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(backend).unwrap();
    let x = EagerTensor::requires_grad_in(x_gpu, ctx.clone()).unwrap();
    let seed = EagerTensor::from_tensor_in(seed_gpu, ctx.clone()).unwrap();
    let sum = ctx
        .with_eager_session(|session| session.reduce_sum(&x, Some(&[0])))
        .unwrap();
    let result = sum.to_tensor().unwrap();
    assert_device_backed(&result);
    let host = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&result))
        })
        .unwrap()
        .unwrap();
    assert_f64_tensor_close(&host, &f64_tensor(vec![2], vec![3.0, 7.0]), 1e-10, 1e-10);

    sum.backward_with(&seed).unwrap();
    let grad = x.grad().unwrap().unwrap().to_tensor().unwrap();
    assert_device_backed(&grad);
    let host = ctx
        .with_execution_session(|session| session.download_to_host(TensorRead::from_tensor(&grad)))
        .unwrap()
        .unwrap();
    assert_f64_tensor_close(&host, &f64_tensor(vec![2, 2], vec![1.0; 4]), 1e-10, 1e-10);
}

#[test]
#[ignore = "requires CUDA"]
fn test_gpu_borrowed_concat_gather_and_index_select_stay_resident() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let x_gpu =
        upload_tensor(backend.runtime(), &f64_tensor(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let indices_gpu = upload_tensor(
        backend.runtime(),
        &Tensor::from_vec_col_major(vec![2, 1], vec![4_i64, 1]).unwrap(),
    )
    .unwrap();
    let ctx = EagerRuntime::with_cuda_backend(backend).unwrap();
    let x = EagerTensor::from_tensor_in(x_gpu, ctx.clone()).unwrap();
    let indices = EagerTensor::from_tensor_in(indices_gpu, ctx.clone()).unwrap();
    let (joined, selected, indexed) = ctx
        .with_eager_session(|session| {
            let joined = session.concatenate(&[&x, &x], 0)?;
            let selected = session.gather(
                &joined,
                &indices,
                GatherConfig {
                    offset_dims: vec![],
                    collapsed_slice_dims: vec![0],
                    start_index_map: vec![0],
                    index_vector_dim: 1,
                    slice_sizes: vec![1],
                },
            )?;
            let indexed = session.index_select(&joined, -1, &[4, 1])?;
            Ok::<_, tenferro_ad::Error>((joined, selected, indexed))
        })
        .unwrap();
    let joined = joined.to_tensor().unwrap();
    let selected = selected.to_tensor().unwrap();
    let indexed = indexed.to_tensor().unwrap();
    assert_device_backed(&joined);
    assert_device_backed(&selected);
    assert_device_backed(&indexed);
    for output in [&selected, &indexed] {
        let host = ctx
            .with_execution_session(|session| {
                session.download_to_host(TensorRead::from_tensor(output))
            })
            .unwrap()
            .unwrap();
        assert_f64_tensor_close(&host, &f64_tensor(vec![2], vec![2.0, 2.0]), 0.0, 0.0);
    }
}

#[test]
#[ignore = "requires CUDA"]
fn test_gpu_matmul_vjp() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let a_host = f64_tensor(
        vec![3, 4],
        vec![
            1.0, -2.0, 0.5, //
            3.0, 1.5, -0.25, //
            -1.0, 0.75, 2.5, //
            4.0, -3.0, 1.25,
        ],
    );
    let b_host = f64_tensor(
        vec![4, 5],
        vec![
            0.5, -1.5, 2.0, 0.25, //
            1.25, 0.75, -0.5, 3.0, //
            -2.0, 1.0, 0.5, -1.25, //
            0.75, -0.25, 1.5, 2.25, //
            -1.0, 0.5, 2.75, -0.75,
        ],
    );
    let cotangent_host = f64_tensor(
        vec![3, 5],
        vec![
            1.0, -0.5, 0.25, //
            2.0, 1.5, -1.0, //
            -0.75, 0.125, 0.5, //
            0.25, -1.5, 2.5, //
            -0.5, 1.0, 0.75,
        ],
    );

    let a_cpu = TracedTensor::from_tensor_concrete_shape(a_host.duplicate().unwrap()).unwrap();
    let b_cpu = TracedTensor::from_tensor_concrete_shape(b_host.duplicate().unwrap()).unwrap();
    let cotangent_cpu =
        TracedTensor::from_tensor_concrete_shape(cotangent_host.duplicate().unwrap()).unwrap();
    let cpu_engine = cpu_runtime();
    let y_cpu = matmul(&a_cpu, &b_cpu);
    let grad_a_cpu = y_cpu.vjp(&a_cpu, &cotangent_cpu).unwrap();
    let grad_b_cpu = y_cpu.vjp(&b_cpu, &cotangent_cpu).unwrap();
    let cpu_grad_a = eval_cpu_tensor(&cpu_engine, &grad_a_cpu);
    let cpu_grad_b = eval_cpu_tensor(&cpu_engine, &grad_b_cpu);

    let upload_backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let a_device = upload_tensor(upload_backend.runtime(), &a_host).unwrap();
    let b_device = upload_tensor(upload_backend.runtime(), &b_host).unwrap();
    let cotangent_device = upload_tensor(upload_backend.runtime(), &cotangent_host).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
    let a_gpu = EagerTensor::from_tensor_in(a_device, ctx.clone()).unwrap();
    let b_gpu = EagerTensor::from_tensor_in(b_device, ctx.clone()).unwrap();
    let cotangent_gpu = EagerTensor::from_tensor_in(cotangent_device, ctx.clone()).unwrap();
    let y_gpu = ctx
        .with_eager_session(|s| {
            // Def 1 (active-edge): functional VJP over untracked leaves requires
            // explicit capture on the session's calling thread.
            let _capture = ctx.capture_trace();
            s.dot_general(
                &a_gpu,
                &b_gpu,
                DotGeneralConfig {
                    lhs_contracting_dims: [1].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            )
        })
        .unwrap();
    let gpu_grad_a = ctx
        .vjp(&y_gpu, &a_gpu, &cotangent_gpu)
        .unwrap()
        .to_tensor()
        .unwrap();
    let gpu_grad_b = ctx
        .vjp(&y_gpu, &b_gpu, &cotangent_gpu)
        .unwrap()
        .to_tensor()
        .unwrap();
    assert_device_backed(&gpu_grad_a);
    assert_device_backed(&gpu_grad_b);
    let gpu_grad_a = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&gpu_grad_a))
        })
        .unwrap()
        .unwrap();
    let gpu_grad_b = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&gpu_grad_b))
        })
        .unwrap()
        .unwrap();

    assert_f64_tensor_close(&gpu_grad_a, &cpu_grad_a, 1.0e-10, 1.0e-10);
    assert_f64_tensor_close(&gpu_grad_b, &cpu_grad_b, 1.0e-10, 1.0e-10);
}

/// `scale_real` / `scale_complex` on CUDA`scale_real` / `scale_complex` on CUDA, on the eager and traced surfaces, with
/// CPU parity and traced/eager agreement (#2020).
///
/// Two independent defects, both backend-dependent:
///
/// - eager built the rank-0 factor with `constant_from`, which does not cross the
///   host/device boundary, so the first op needing it failed with
///   `CudaBackend::to_contiguous_read: expected CubeCL GPU tensor view, got host
///   tensor`;
/// - the traced surface emitted `Mul` with a rank-0 operand and no
///   `BroadcastInDim`, which the CUDA backend rejects
///   (`mul: incompatible shapes: lhs=[], rhs=[4, 3, 2]`) where the CPU backend
///   accepts a rank-0 operand, so the same graph ran on one backend and not the
///   other.
#[test]
#[ignore = "requires CUDA"]
fn test_gpu_scale_real_and_scale_complex_match_cpu() {
    use num_complex::Complex64;
    use tenferro_gpu::cuda::{cuda_runtime_engine_registration, download_tensor};
    use tenferro_runtime::{EngineId, GraphCompiler, Runtime};

    assert!(gpu_available(), "CUDA test requires an available device");

    let shape = vec![4usize, 3, 2];
    let len: usize = shape.iter().product();
    let real_values: Vec<f64> = (0..len).map(|i| i as f64 * 0.5 - 2.0).collect();
    let complex_values: Vec<Complex64> = (0..len)
        .map(|i| Complex64::new(i as f64 * 0.25, -(i as f64) * 0.5))
        .collect();
    let real = f64_tensor(shape.clone(), real_values.clone());
    let complex = Tensor::from_typed::<Complex64>(
        TypedTensor::from_vec_col_major(shape.clone(), complex_values.clone()).unwrap(),
    );

    let cuda_runtime = |backend: &CudaBackend| -> Runtime {
        let engine_id = EngineId::new("tenferro-ad.scale.cuda.v1").unwrap();
        let mut builder = Runtime::builder();
        builder
            .register_engine(cuda_runtime_engine_registration(backend, engine_id).unwrap())
            .unwrap();
        builder.build().unwrap()
    };

    let factor = |real_factor: Option<f64>, complex_factor: Option<Complex64>| {
        move |x: &TracedTensor| match (real_factor, complex_factor) {
            (Some(f), None) => x.scale_real(f),
            (None, Some(f)) => x.scale_complex(f),
            _ => unreachable!("exactly one factor"),
        }
    };

    // One graph, compiled for whichever runtime is given. The CUDA executor has
    // no ingress for a host-resident binding, so the caller uploads it first.
    let execute =
        |runtime: &Runtime,
         binding: &Tensor,
         apply: &dyn Fn(&TracedTensor) -> Result<TracedTensor, tenferro_runtime::Error>|
         -> Tensor {
            let placeholder =
                TracedTensor::input_concrete_shape(binding.dtype(), binding.shape()).unwrap();
            let scaled = apply(&placeholder).unwrap();
            // The binding is borrowed, not duplicated: a device-backed tensor cannot
            // be cloned through host data.
            let specs = [(&placeholder, binding.dtype(), binding.shape())];
            let program = GraphCompiler::new()
                .compile_with_input_specs(&scaled, &specs)
                .unwrap();
            runtime
                .run_compiled(&program, &[binding])
                .unwrap()
                .pop()
                .unwrap()
        };

    // --- CPU reference on the same graph
    let cpu = cpu_runtime();
    let cpu_real = execute(&cpu, &real, &factor(Some(2.5), None));
    let cpu_complex_real = execute(&cpu, &complex, &factor(Some(2.5), None));
    let cpu_complex_complex = execute(
        &cpu,
        &complex,
        &factor(None, Some(Complex64::new(0.0, 1.0))),
    );

    // The CPU values equal the obvious host arithmetic, so a wrong CPU result
    // cannot make a wrong CUDA result look correct.
    for (index, value) in cpu_real.as_slice::<f64>().unwrap().iter().enumerate() {
        assert_eq!(
            *value,
            real_values[index] * 2.5,
            "cpu scale_real real[{index}]"
        );
    }
    for (index, value) in cpu_complex_complex
        .as_slice::<Complex64>()
        .unwrap()
        .iter()
        .enumerate()
    {
        let source = complex_values[index];
        assert_eq!(
            *value,
            Complex64::new(-source.im, source.re),
            "cpu scale_complex complex[{index}]"
        );
    }

    // --- traced on CUDA
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let traced_runtime = cuda_runtime(&backend);
    let traced_case =
        |input: &Tensor,
         apply: &dyn Fn(&TracedTensor) -> Result<TracedTensor, tenferro_runtime::Error>|
         -> Tensor {
            let device = upload_tensor(backend.runtime(), input).unwrap();
            let out = execute(&traced_runtime, &device, apply);
            download_tensor(backend.runtime(), &out).unwrap()
        };
    assert_real_like_close(
        &traced_case(&real, &factor(Some(2.5), None)),
        &cpu_real,
        "scale_real/f64/traced",
    );
    assert_real_like_close(
        &traced_case(&complex, &factor(Some(2.5), None)),
        &cpu_complex_real,
        "scale_real/c64/traced",
    );
    assert_real_like_close(
        &traced_case(&complex, &factor(None, Some(Complex64::new(0.0, 1.0)))),
        &cpu_complex_complex,
        "scale_complex/c64/traced",
    );

    // --- eager on CUDA
    let eager_scale = |input: &Tensor,
                       apply: &dyn Fn(
        &mut tenferro_ad::EagerSession<'_>,
        &EagerTensor,
    ) -> tenferro_ad::Result<EagerTensor>|
     -> Tensor {
        let upload_backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
        let uploaded = upload_tensor(upload_backend.runtime(), input).unwrap();
        let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
        let x = EagerTensor::from_tensor_in(uploaded, ctx.clone()).unwrap();
        let scaled = ctx
            .with_eager_session(|session| apply(session, &x))
            .unwrap();
        // Materialize outside the execution session: `to_tensor` needs the
        // runtime's own session entry, so calling it inside `with_execution_session`
        // re-enters it.
        let value = scaled.to_tensor().unwrap();
        ctx.with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&value))
        })
        .unwrap()
        .unwrap()
    };
    assert_real_like_close(
        &eager_scale(&real, &|session, x| session.scale_real(x, 2.5)),
        &cpu_real,
        "scale_real/f64/eager",
    );
    assert_real_like_close(
        &eager_scale(&complex, &|session, x| session.scale_real(x, 2.5)),
        &cpu_complex_real,
        "scale_real/c64/eager",
    );
    assert_real_like_close(
        &eager_scale(&complex, &|session, x| {
            session.scale_complex(x, Complex64::new(0.0, 1.0))
        }),
        &cpu_complex_complex,
        "scale_complex/c64/eager",
    );

    // `scale_complex` still rejects a real input, on the device as on the host.
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let uploaded = upload_tensor(backend.runtime(), &real).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(backend).unwrap();
    let x = EagerTensor::from_tensor_in(uploaded, ctx.clone()).unwrap();
    assert!(ctx
        .with_eager_session(|session| session.scale_complex(&x, Complex64::new(0.0, 1.0)))
        .is_err());
}

/// Compare two host tensors of the same dtype (f64 or c64).
fn assert_real_like_close(actual: &Tensor, expected: &Tensor, what: &str) {
    use num_complex::Complex64;

    assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{what}: dtype");
    match expected.dtype() {
        DType::F64 => {
            let actual = actual.as_slice::<f64>().unwrap();
            let expected = expected.as_slice::<f64>().unwrap();
            for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-12 * e.abs().max(1.0),
                    "{what}[{index}]: actual {a}, expected {e}"
                );
            }
        }
        DType::C64 => {
            let actual = actual.as_slice::<Complex64>().unwrap();
            let expected = expected.as_slice::<Complex64>().unwrap();
            for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
                let error = ((a.re - e.re).powi(2) + (a.im - e.im).powi(2)).sqrt();
                let scale = (e.re.abs() + e.im.abs()).max(1.0);
                assert!(
                    error <= 1e-12 * scale,
                    "{what}[{index}]: actual {a:?}, expected {e:?}"
                );
            }
        }
        other => panic!("{what}: unexpected dtype {other:?}"),
    }
}

/// Host cost per eager op on a decode-sized tensor (#2079, #2112 item 7).
///
/// Prints the streaming arm (each result dropped, which is what a decode loop
/// does), the retained arm (every result held, so no allocation can be reused -
/// #2079's note is that the pool must not be "fixed" to satisfy that number) and
/// the raw-launch floor for the same shape. It asserts nothing: the acceptance is
/// a reading, taken with the same harness before and after a change. The official
/// arms (`eager_op_f32_add_decode_*` in `gpu/perf_issues`) live in
/// `tensor4all/tenferro-benchmark`; this reproduces the shape in-repo.
#[test]
#[ignore = "requires CUDA"]
fn eager_op_f64_decode_add_host_cost_benchmark() {
    use std::time::Instant;
    use tenferro_ad::EagerRuntime;

    assert!(gpu_available(), "CUDA test requires an available device");
    let shape = vec![1024usize, 8];
    let n: usize = shape.iter().product();
    let host_a = f64_tensor(shape.clone(), vec![1.0_f64; n]);
    let host_b = f64_tensor(shape.clone(), vec![2.0_f64; n]);

    let upload_backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let runtime = upload_backend.runtime().clone();
    let ga = upload_tensor(upload_backend.runtime(), &host_a).unwrap();
    let gb = upload_tensor(upload_backend.runtime(), &host_b).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
    let x = EagerTensor::from_tensor_in(ga, ctx.clone()).unwrap();
    let y = EagerTensor::from_tensor_in(gb, ctx.clone()).unwrap();
    let iters = 2000usize;

    let warm = |iters: usize, keep: bool| -> f64 {
        let started = Instant::now();
        ctx.with_eager_session(|s| -> tenferro_ad::Result<()> {
            let mut held = Vec::new();
            for _ in 0..iters {
                let out = s.add(&x, &y)?;
                if keep {
                    held.push(out);
                }
            }
            drop(held);
            Ok(())
        })
        .unwrap();
        runtime.synchronize().unwrap();
        started.elapsed().as_secs_f64() * 1e6 / iters as f64
    };

    let _ = warm(200, false);
    // The per-op cost is a few microseconds, so one 2000-iteration average still
    // carries a few percent of run-to-run noise: report several and let the
    // reader take the minimum, which is the least noisy estimate of host cost.
    let mut samples = Vec::new();
    for _ in 0..5 {
        samples.push(warm(iters, false));
    }
    let retained = warm(iters, true);
    let best = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "[2079] eager add 1024x8 f64: streaming {samples:?} (best {best:.2} us/op), retained {retained:.2} us/op"
    );
}
