//! Checked sub-spans and shareable compiled modules (issue #2086).
//!
//! The device tests require CUDA hardware and are ignored by default, matching
//! the regular CUDA test convention in this crate. The arithmetic test at the
//! end runs without a device.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::*;
use crate::cuda::raw;
use crate::cuda::{gpu_available, CudaEngine};

/// Kernel copying `n` f64 values, used to prove a sub-span points where the
/// caller asked and nowhere else.
const COPY_F64_KERNEL_SRC: &str = r#"
extern "C" __global__ void copy_f64(double* out, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = src[i];
    }
}
"#;

fn download_f64(runtime: &crate::cuda::CudaRuntime, tensor: &Tensor) -> Vec<f64> {
    let downloaded = crate::cuda::download_tensor(runtime, tensor).expect("download");
    let host = downloaded.into_typed::<f64>().expect("f64 download");
    let (_, flat) = host
        .into_vec_col_major()
        .map_err(|failure| failure.into_parts().1)
        .expect("flat f64 host tensor");
    flat
}

/// One arena allocation can expose its upper half as a checked read-only span
/// and as a checked mutable span, and both feed the existing `KernelArg`
/// constructors.
#[test]
#[ignore = "requires CUDA"]
fn raw_tensor_offsets_bind_arena_subspans() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = gpu_backend();
    let runtime = backend.runtime().clone();
    const N: usize = 8;
    const ELEMENT: usize = std::mem::size_of::<f64>();
    let half = N * ELEMENT;

    let host: Vec<f64> = (0..2 * N).map(|index| index as f64 + 0.25).collect();
    let arena = upload(&backend, &tensor_f64(vec![2 * N], host.clone()));
    let mut out_arena = upload(&backend, &tensor_f64(vec![2 * N], vec![0.0; 2 * N]));

    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_subspan", |raw| {
                let arena_ref = raw.tensor(arena.as_typed::<f64>().expect("f64 arena"))?;
                assert_eq!(arena_ref.byte_len(), 2 * half);
                let upper = arena_ref.offset(half, half)?;
                assert_eq!(upper.byte_len(), half);
                // SAFETY: observes the validated address values only.
                assert_eq!(
                    unsafe { upper.raw_ptr() } as usize,
                    unsafe { arena_ref.raw_ptr() } as usize + half,
                    "the read-only sub-span must point into the arena at the requested offset"
                );

                let out_base =
                    raw.tensor_mut(out_arena.as_typed_mut::<f64>().expect("f64 output arena"))?;
                let upper_out = out_base.offset(half, half)?;
                assert_eq!(upper_out.byte_len(), half);
                // SAFETY: observes the validated address values only.
                assert_eq!(
                    unsafe { upper_out.raw_ptr() } as usize,
                    unsafe { out_base.raw_ptr() } as usize + half,
                    "the mutable sub-span must point into the arena at the requested offset"
                );

                let module =
                    raw.compile_nvrtc(COPY_F64_KERNEL_SRC, &raw::NvrtcOptions::default())?;
                let function = module.function("copy_f64")?;
                let config = raw::LaunchConfig::flat(N as u32, N as u32, 0)?;
                // SAFETY: the kernel ABI is (double*, const double*, int);
                // `upper` reads exactly N values from arena[N..2N], `upper_out`
                // writes exactly N values at out_arena[N..2N], the two spans do
                // not alias, and both arenas and the module outlive the
                // synchronization below.
                unsafe {
                    raw.launch(
                        &function,
                        config,
                        &[
                            raw::KernelArg::tensor_mut(&upper_out),
                            raw::KernelArg::tensor(&upper),
                            raw::KernelArg::i32(N as i32),
                        ],
                    )?;
                }
                raw.synchronize()?;
                Ok(())
            })
            .unwrap();
    });

    let mut expected = vec![0.0; N];
    expected.extend_from_slice(&host[N..]);
    assert_eq!(
        download_f64(&runtime, &out_arena),
        expected,
        "the kernel must copy only the upper arena half; the lower half stays untouched"
    );
}

/// Out-of-range, overflowing, and misaligned sub-spans are typed validation
/// errors on both the read-only and the mutable path.
#[test]
#[ignore = "requires CUDA"]
fn raw_tensor_offsets_reject_out_of_range_and_overflow() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = gpu_backend();
    const N: usize = 4;
    let arena = upload(&backend, &tensor_f64(vec![N], vec![1.0; N]));
    let mut out = upload(&backend, &tensor_f64(vec![N], vec![0.0; N]));

    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_subspan_errors", |raw| {
                let arena_ref = raw.tensor(arena.as_typed::<f64>().expect("f64 arena"))?;
                let span = arena_ref.byte_len();

                let error = arena_ref.offset(span, 8).unwrap_err();
                assert_validation_kind(
                    &error,
                    "raw.tensor_ref.offset",
                    ValidationKind::InvalidArgument,
                );

                let error = arena_ref.offset(0, span + 8).unwrap_err();
                assert_validation_kind(
                    &error,
                    "raw.tensor_ref.offset",
                    ValidationKind::InvalidArgument,
                );

                let error = arena_ref.offset(usize::MAX - 7, 16).unwrap_err();
                assert_validation_kind(
                    &error,
                    "raw.tensor_ref.offset",
                    ValidationKind::InvalidArgument,
                );
                assert!(
                    format!("{error}").contains("overflows"),
                    "aligned offset+length overflow must be reported as such: {error}"
                );

                // A three-byte offset is not a whole f64.
                let error = arena_ref.offset(3, 8).unwrap_err();
                assert_validation_kind(
                    &error,
                    "raw.tensor_ref.offset",
                    ValidationKind::InvalidArgument,
                );

                let out_ref = raw.tensor_mut(out.as_typed_mut::<f64>().expect("f64 out"))?;
                let span = out_ref.byte_len();

                let error = out_ref.offset(span, 8).unwrap_err();
                assert_validation_kind(
                    &error,
                    "raw.tensor_mut.offset",
                    ValidationKind::InvalidArgument,
                );

                let error = out_ref.offset(usize::MAX - 7, 16).unwrap_err();
                assert_validation_kind(
                    &error,
                    "raw.tensor_mut.offset",
                    ValidationKind::InvalidArgument,
                );
                assert!(
                    format!("{error}").contains("overflows"),
                    "aligned offset+length overflow must be reported as such: {error}"
                );
                Ok(())
            })
            .unwrap();
    });
}

/// One compiled function is `Send + Sync`, so a second thread on the same
/// runtime reuses it instead of recompiling.
#[test]
#[ignore = "requires CUDA"]
fn raw_function_compiles_once_and_launches_from_another_thread() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = gpu_backend();
    let runtime = backend.runtime().clone();

    // Compile-time contract: a compiled module is a shareable handle.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<raw::Module>();
    assert_send_sync::<raw::Function>();

    let compiles = Arc::new(AtomicUsize::new(0));
    let compile_counter = Arc::clone(&compiles);
    let (module, function) = with_cuda_exec(&mut backend, move |session| {
        session
            .with_raw("test.raw_share_compile", move |raw| {
                compile_counter.fetch_add(1, Ordering::SeqCst);
                let module =
                    raw.compile_nvrtc(COPY_F64_KERNEL_SRC, &raw::NvrtcOptions::default())?;
                let function = module.function("copy_f64")?;
                Ok((module, function))
            })
            .unwrap()
    });
    assert_eq!(compiles.load(Ordering::SeqCst), 1, "one compile");

    const N: usize = 8;
    let input = upload(
        &backend,
        &tensor_f64(vec![N], (1..=N).map(|value| value as f64).collect()),
    );
    let output = upload(&backend, &tensor_f64(vec![N], vec![0.0; N]));

    // The clone shares the exact runtime state; the second thread enters a raw
    // session on that same runtime.
    let mut worker_backend = backend.clone();
    let output = std::thread::spawn(move || {
        with_cuda_exec(&mut worker_backend, move |session| {
            session
                .with_raw("test.raw_share_launch", move |raw| {
                    let input_ref = raw.tensor(input.as_typed::<f64>().expect("f64 input"))?;
                    let mut output = output;
                    let output_ref =
                        raw.tensor_mut(output.as_typed_mut::<f64>().expect("f64 output"))?;
                    // The module handle itself must be usable on this thread,
                    // not merely the function pre-derived on the first thread.
                    let looked_up = module
                        .function("copy_f64")
                        .expect("module lookup on the worker thread");
                    drop(looked_up);
                    let config = raw::LaunchConfig::flat(N as u32, N as u32, 0)?;
                    // SAFETY: the kernel ABI is (double*, const double*, int);
                    // both spans hold N f64 values, the output is uniquely
                    // borrowed and does not alias the input, and the function,
                    // input, output, and module owner outlive synchronization.
                    unsafe {
                        raw.launch(
                            &function,
                            config,
                            &[
                                raw::KernelArg::tensor_mut(&output_ref),
                                raw::KernelArg::tensor(&input_ref),
                                raw::KernelArg::i32(N as i32),
                            ],
                        )?;
                    }
                    raw.synchronize()?;
                    Ok(output)
                })
                .unwrap()
        })
    })
    .join()
    .expect("worker thread");

    assert_eq!(
        compiles.load(Ordering::SeqCst),
        1,
        "the second thread must reuse the compiled function, not compile again"
    );
    let expected: Vec<f64> = (1..=N).map(|value| value as f64).collect();
    assert_eq!(
        download_f64(&runtime, &output),
        expected,
        "the shared function must launch the same kernel on the second thread"
    );
}

/// A module cached in `Session::resource` is shared by a second thread on the
/// same runtime without a second compile.
#[test]
#[ignore = "requires CUDA"]
fn raw_module_caches_in_session_resource_across_threads() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = gpu_backend();
    let runtime = backend.runtime().clone();
    let compiles = Arc::new(AtomicUsize::new(0));
    let compile_counter = Arc::clone(&compiles);

    // `Session::resource` requires `T: Send`; the module handle now satisfies
    // it, so the documented per-runtime cache can hold the compiled module.
    let cached = with_cuda_exec(&mut backend, move |session| {
        session
            .with_raw("test.raw_module_cache_insert", move |raw| {
                compile_counter.fetch_add(1, Ordering::SeqCst);
                let module =
                    raw.compile_nvrtc(COPY_F64_KERNEL_SRC, &raw::NvrtcOptions::default())?;
                let cached = raw.resource(|| Ok(Arc::new(module)))?;
                Ok(Arc::clone(&cached))
            })
            .unwrap()
    });
    assert_eq!(compiles.load(Ordering::SeqCst), 1, "one compile");
    let stats = backend.cuda_extension_cache_stats().expect("cache stats");
    assert_eq!((stats.misses, stats.hits), (1, 0), "one cache miss");

    const N: usize = 8;
    let input = upload(
        &backend,
        &tensor_f64(vec![N], (1..=N).map(|value| value as f64).collect()),
    );
    let output = upload(&backend, &tensor_f64(vec![N], vec![0.0; N]));

    let mut worker_backend = backend.clone();
    let output = std::thread::spawn(move || {
        with_cuda_exec(&mut worker_backend, move |session| {
            session
                .with_raw("test.raw_module_cache_hit", move |raw| {
                    // The initializer only runs on a miss; returning an error
                    // proves the second thread did not recompile.
                    let cached_again = raw.resource(|| {
                        Err::<Arc<raw::Module>, _>(crate::Error::runtime_state(
                            "test.raw_module_cache_hit",
                            "cache miss on the second thread",
                        ))
                    })?;
                    assert!(
                        Arc::ptr_eq(&cached, &cached_again),
                        "the second thread must receive the cached module"
                    );
                    let function = cached_again.function("copy_f64")?;
                    let input_ref = raw.tensor(input.as_typed::<f64>().expect("f64 input"))?;
                    let mut output = output;
                    let output_ref =
                        raw.tensor_mut(output.as_typed_mut::<f64>().expect("f64 output"))?;
                    let config = raw::LaunchConfig::flat(N as u32, N as u32, 0)?;
                    // SAFETY: ABI and spans as in the direct sharing test; the
                    // cached module owner outlives the launch.
                    unsafe {
                        raw.launch(
                            &function,
                            config,
                            &[
                                raw::KernelArg::tensor_mut(&output_ref),
                                raw::KernelArg::tensor(&input_ref),
                                raw::KernelArg::i32(N as i32),
                            ],
                        )?;
                    }
                    raw.synchronize()?;
                    Ok(output)
                })
                .unwrap()
        })
    })
    .join()
    .expect("worker thread");

    let stats = backend.cuda_extension_cache_stats().expect("cache stats");
    assert_eq!(compiles.load(Ordering::SeqCst), 1, "still one compile");
    assert_eq!(stats.misses, 1, "no second compile on the worker thread");
    assert!(stats.hits >= 1, "the cached module must be a cache hit");
    let expected: Vec<f64> = (1..=N).map(|value| value as f64).collect();
    assert_eq!(download_f64(&runtime, &output), expected);
}

/// Dropping a module's last handle inside an active raw session must keep that
/// session's current context: the next module load and launch in the same
/// callback still depend on it (#2086).
#[test]
#[ignore = "requires CUDA"]
fn raw_dropping_a_module_keeps_the_live_session_context() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = gpu_backend();
    let runtime = backend.runtime().clone();
    const N: usize = 4;
    let input = upload(&backend, &tensor_f64(vec![N], vec![1.0, 2.0, 3.0, 4.0]));
    let mut output = upload(&backend, &tensor_f64(vec![N], vec![0.0; N]));

    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_drop_live_module", |raw| {
                let first =
                    raw.compile_nvrtc(COPY_F64_KERNEL_SRC, &raw::NvrtcOptions::default())?;
                let first_function = first.function("copy_f64")?;
                // Last handle of the first module: this unloads it while the
                // session is still live.
                drop(first_function);
                drop(first);
                assert!(
                    matches!(cudarc::driver::result::ctx::get_current(), Ok(Some(_))),
                    "the live raw session must still have its context current after the unload"
                );

                // Loading and launching a second module proves the context is
                // usable, not merely reported as current.
                let second =
                    raw.compile_nvrtc(COPY_F64_KERNEL_SRC, &raw::NvrtcOptions::default())?;
                let function = second.function("copy_f64")?;
                let input_ref = raw.tensor(input.as_typed::<f64>().expect("f64 input"))?;
                let output_ref =
                    raw.tensor_mut(output.as_typed_mut::<f64>().expect("f64 output"))?;
                let config = raw::LaunchConfig::flat(N as u32, N as u32, 0)?;
                // SAFETY: the kernel ABI is (double*, const double*, int); both
                // spans hold N f64 values, the output is uniquely borrowed and
                // does not alias the input, and the second module and both
                // tensors outlive the synchronization below.
                unsafe {
                    raw.launch(
                        &function,
                        config,
                        &[
                            raw::KernelArg::tensor_mut(&output_ref),
                            raw::KernelArg::tensor(&input_ref),
                            raw::KernelArg::i32(N as i32),
                        ],
                    )?;
                }
                raw.synchronize()?;
                Ok(())
            })
            .unwrap();
    });

    assert_eq!(
        download_f64(&runtime, &output),
        vec![1.0, 2.0, 3.0, 4.0],
        "a module loaded after another module's unload must launch in the same session"
    );
}

/// A `Function` outlives both its `Module` and the backend that loaded it, and
/// dropping the last handle on another thread restores the thread's previous
/// (here: absent) current context.
#[test]
#[ignore = "requires CUDA"]
fn raw_function_survives_its_module_and_backend_and_restores_no_context_on_drop() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let device = crate::cuda::cuda_devices()
        .expect("CUDA device discovery")
        .first()
        .expect("a CUDA device")
        .id();
    const N: usize = 4;

    let function = {
        let mut loader_backend = CudaEngine::new(device).expect("CUDA backend should initialize");
        let function = with_cuda_exec(&mut loader_backend, |session| {
            session
                .with_raw("test.raw_function_lifetime", |raw| {
                    let module =
                        raw.compile_nvrtc(COPY_F64_KERNEL_SRC, &raw::NvrtcOptions::default())?;
                    let function = module.function("copy_f64")?;
                    drop(module);
                    Ok(function)
                })
                .unwrap()
        });
        // The loading backend is dropped here. `function` must retain the
        // module image and its own runtime/context pin.
        function
    };

    // A fresh runtime on the same device uses the surviving function; both
    // runtimes retain the same per-device primary context.
    let mut backend = CudaEngine::new(device).expect("CUDA backend should initialize");
    let runtime = backend.runtime().clone();
    let input = upload(&backend, &tensor_f64(vec![N], vec![5.0, 6.0, 7.0, 8.0]));
    let mut output = upload(&backend, &tensor_f64(vec![N], vec![0.0; N]));
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_function_lifetime_launch", |raw| {
                let input_ref = raw.tensor(input.as_typed::<f64>().expect("f64 input"))?;
                let output_ref =
                    raw.tensor_mut(output.as_typed_mut::<f64>().expect("f64 output"))?;
                let config = raw::LaunchConfig::flat(N as u32, N as u32, 0)?;
                // SAFETY: ABI and spans as in the sharing test; the function
                // retained the module image after its `Module` was dropped.
                unsafe {
                    raw.launch(
                        &function,
                        config,
                        &[
                            raw::KernelArg::tensor_mut(&output_ref),
                            raw::KernelArg::tensor(&input_ref),
                            raw::KernelArg::i32(N as i32),
                        ],
                    )?;
                }
                raw.synchronize()?;
                Ok(())
            })
            .unwrap();
    });
    assert_eq!(
        download_f64(&runtime, &output),
        vec![5.0, 6.0, 7.0, 8.0],
        "a function must still launch after its module and loading backend are gone"
    );

    // Drop the tensors, backend, and runtime handles on this thread; only the
    // function still pins the module's context.
    drop(input);
    drop(output);
    drop(backend);
    drop(runtime);

    std::thread::spawn(move || {
        assert!(
            cudarc::driver::result::ctx::get_current()
                .expect("context query")
                .is_none(),
            "a fresh thread must start without a current CUDA context"
        );
        drop(function);
        assert!(
            cudarc::driver::result::ctx::get_current()
                .expect("context query")
                .is_none(),
            "dropping the last module/function handle must restore the \
             no-current-context state instead of leaving the primary context current"
        );
    })
    .join()
    .expect("drop thread");
}

/// Dropping the last handle on a thread that has a different current context
/// restores that context instead of leaving the module's primary context.
#[test]
#[ignore = "requires CUDA"]
fn raw_function_drop_restores_a_previous_non_null_context() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let device = crate::cuda::cuda_devices()
        .expect("CUDA device discovery")
        .first()
        .expect("a CUDA device")
        .id();
    let function = {
        let mut loader_backend = CudaEngine::new(device).expect("CUDA backend should initialize");
        with_cuda_exec(&mut loader_backend, |session| {
            session
                .with_raw("test.raw_function_nonnull_context", |raw| {
                    let module =
                        raw.compile_nvrtc(COPY_F64_KERNEL_SRC, &raw::NvrtcOptions::default())?;
                    module.function("copy_f64")
                })
                .unwrap()
        })
    };

    std::thread::spawn(move || {
        let ordinal = usize::try_from(device.ordinal()).expect("non-negative device ordinal");
        // A non-primary context is a valid, different previous context for
        // this thread; the module unload must restore it rather than leaving
        // the module's primary context current.
        let previous = cudarc::driver::safe::CudaContext::new_non_primary(ordinal, 0)
            .expect("non-primary CUDA context");
        let saved = previous.cu_ctx();
        assert_eq!(
            cudarc::driver::result::ctx::get_current().expect("context query"),
            Some(saved)
        );
        drop(function);
        assert_eq!(
            cudarc::driver::result::ctx::get_current().expect("context query"),
            Some(saved),
            "dropping the last module/function handle must restore the previous non-null context"
        );
    })
    .join()
    .expect("drop thread");
}

/// The address-overflow and alignment checks are arithmetic, not device work.
#[test]
fn raw_subspan_checks_address_overflow_and_alignment() {
    let base = usize::MAX as *mut std::ffi::c_void;
    // `bytes + len` is in range, but `base + bytes` overflows.
    let error =
        crate::cubecl::raw::checked_subspan(base, usize::MAX, 8, 8, 8, "test.subspan").unwrap_err();
    assert_validation_kind(&error, "test.subspan", ValidationKind::InvalidArgument);
    assert!(
        format!("{error}").contains("overflows"),
        "an aligned sub-span whose base address overflows must say so: {error}"
    );

    // A byte offset that is not a multiple of the element size is rejected.
    let error =
        crate::cubecl::raw::checked_subspan(std::ptr::null_mut(), 64, 3, 8, 8, "test.subspan")
            .unwrap_err();
    assert_validation_kind(&error, "test.subspan", ValidationKind::InvalidArgument);

    // A byte element type has no alignment restriction, and an empty sub-span
    // at the end of the parent is allowed.
    let address =
        crate::cubecl::raw::checked_subspan(std::ptr::null_mut(), 8, 8, 0, 1, "test.subspan")
            .expect("empty tail sub-span");
    assert_eq!(address as usize, 8);
}
