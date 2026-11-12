use num_complex::{Complex32, Complex64};
use std::error::Error as StdError;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tenferro_fft::FftOptions;
mod support;
#[cfg(feature = "autodiff")]
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuEngine;
#[cfg(feature = "autodiff")]
use tenferro_fft::EagerFftOps;
use tenferro_fft::{FftNorm, TracedFftOps};
use tenferro_runtime::{
    DType, Error as RuntimeError, ErrorPhase, ExecutionEngine, ExtensionCacheSelector,
    GraphCompiler, PrepareError, SymDim, Tensor, TracedTensor,
};
use tenferro_tensor::{
    BackendStorageHandle, DeviceId, DeviceKind, ErrorKind, GpuBackendKind, MemoryKind, Placement,
    StorageBuffer, TypedTensor, ValidationError,
};

fn run(output: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).unwrap();
    support::run_one(&program, &[]).unwrap()
}

#[cfg(feature = "autodiff")]
fn eager(input: Tensor) -> EagerTensor {
    (EagerRuntime::new().unwrap()).constant_from(input).unwrap()
}

fn assert_c64_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.re - e.re).abs() < 1e-10 && (a.im - e.im).abs() < 1e-10,
            "idx {idx}: actual={a:?}, expected={e:?}"
        );
    }
}

#[cfg(feature = "autodiff")]
fn assert_c64_close_tol(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.re - e.re).abs() <= tol && (a.im - e.im).abs() <= tol,
            "idx {idx}: actual={a:?}, expected={e:?}"
        );
    }
}

fn assert_f64_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < 1e-10, "idx {idx}: actual={a}, expected={e}");
    }
}

fn assert_c32_close(actual: &[Complex32], expected: &[Complex32]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.re - e.re).abs() < 1e-5 && (a.im - e.im).abs() < 1e-5,
            "idx {idx}: actual={a:?}, expected={e:?}"
        );
    }
}

fn assert_f32_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < 1e-5, "idx {idx}: actual={a}, expected={e}");
    }
}

#[cfg(feature = "autodiff")]
fn fft_ad_context() -> tenferro_ad::AdContext {
    tenferro_ad::AdContext::builder()
        .with_semantic_extension_rules(tenferro_fft::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap()
}

#[cfg(feature = "autodiff")]
#[test]
fn eager_fft_family_installs_and_executes_on_the_borrowed_session() {
    let mut engine = CpuEngine::serial();
    let ctx = EagerRuntime::new().unwrap();
    let (output, inverse, half, real_back) = ctx
        .with_eager_session(&mut engine, |session| {
            let input = session
                .constant_from(Tensor::from_vec_col_major([2], vec![1.0_f64, 2.0]).unwrap())?;
            let output = session.fft(&input)?;
            let inverse = session.ifft(&output)?;
            let half = session.rfft(&input)?;
            let real_back = session.irfft_with(&half, &FftOptions::default().n(Some(2)))?;
            Ok::<_, tenferro_ad::Error>((output, inverse, half, real_back))
        })
        .unwrap();
    assert_c64_close(
        output.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)],
    );
    assert_c64_close(
        inverse.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    );
    assert_c64_close(
        half.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)],
    );
    assert_f64_close(
        real_back.value().unwrap().as_slice::<f64>().unwrap(),
        &[1.0, 2.0],
    );
    let mut engine = CpuEngine::serial();
    let other = EagerRuntime::new().unwrap();
    let foreign = other
        .constant_from(Tensor::from_vec_col_major([2], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    let error = ctx
        .with_eager_session(&mut engine, |session| session.fft(&foreign))
        .unwrap_err();
    assert!(matches!(error, RuntimeError::ContextMismatch { .. }));
}

#[cfg(feature = "autodiff")]
#[test]
fn fft_semantic_ad_rules_register_all_roles() {
    let rules = tenferro_fft::semantic_ad_rules().unwrap();
    assert!(rules.lookup_linearize("tenferro-fft.fft.v1").is_some());
    assert!(rules
        .lookup_linear_transpose("tenferro-fft.fft.v1")
        .is_some());
    assert!(rules.lookup_primal_vjp("tenferro-fft.fft.v1").is_some());
}

#[cfg(feature = "autodiff")]
fn finite_diff_c64_directional(
    f: impl Fn(&[Complex64]) -> Vec<Complex64>,
    base: &[Complex64],
    tangent: &[Complex64],
    h: f64,
) -> Vec<Complex64> {
    assert_eq!(base.len(), tangent.len());
    let plus: Vec<_> = base
        .iter()
        .zip(tangent)
        .map(|(&value, &delta)| value + delta * h)
        .collect();
    let minus: Vec<_> = base
        .iter()
        .zip(tangent)
        .map(|(&value, &delta)| value - delta * h)
        .collect();
    let plus_out = f(&plus);
    let minus_out = f(&minus);
    plus_out
        .iter()
        .zip(minus_out.iter())
        .map(|(&plus, &minus)| (plus - minus) / (2.0 * h))
        .collect()
}

fn cuda_c64_tensor(shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_buffer_col_major(
            shape,
            StorageBuffer::Backend(Box::new(BackendStorageHandle::<Complex64>::new_with_len(
                7, len,
            ))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
                cpu_affinity: None,
            },
        )
        .unwrap(),
    )
}

#[test]
fn publishes_extension_family_id() {
    assert_eq!(tenferro_fft::FFT_EXTENSION_FAMILY_ID, "tenferro-fft.fft.v1");
}

#[test]
fn traced_tensor_fft_ext_exposes_fft() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let y = x.rfft().unwrap();

    assert_eq!(y.rank(), 1);
}

#[test]
fn registered_runtime_rejects_gpu_input_without_ingress() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace.input_concrete_shape(DType::C64, &[2]).unwrap();
    let y = x.fft().unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::C64, &[2])])
        .unwrap();
    let gpu_input = cuda_c64_tensor(vec![2]);
    let mut engine = CpuEngine::serial();
    let runtime = support::cpu_runtime_with_fft(&engine).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        support::run_on_engine(&runtime, &mut engine, &program, &[&gpu_input])
    }));
    assert!(
        result.is_ok(),
        "FFT GPU input should return an error, not panic"
    );
    let err = result
        .unwrap()
        .expect_err("FFT GPU input should be rejected without an ingress");
    let prepare_error = err
        .source()
        .and_then(StdError::source)
        .and_then(|source| source.downcast_ref::<PrepareError>())
        .expect("typed prepare error");
    assert!(matches!(
        prepare_error,
        PrepareError::NoInputIngress { input_index: 0, .. }
    ));
}

#[test]
fn fft_cpu_output_buffers_avoid_zero_fill_but_keep_lane_padding() {
    let source = include_str!("../src/cpu.rs");

    assert!(
        !source.contains("let mut output = vec![Complex::zero(); out_shape.iter().product()]"),
        "FFT CPU complex output buffers are fully overwritten and should not be zero-filled"
    );
    assert!(
        !source.contains("let mut output = vec![T::zero(); out_shape.iter().product()]"),
        "FFT CPU real output buffers are fully overwritten and should not be zero-filled"
    );
    let lanes = include_str!("../src/cpu/lanes.rs");
    assert!(source.contains("PooledUninitOutput::<O>::new"));
    assert!(
        lanes.contains("lane.resize(fft_len, Complex::zero())")
            && lanes.contains("// INVARIANT: only missing input positions are padding"),
        "FFT padding must explicitly initialize only missing input positions"
    );
    assert!(
        lanes.contains("buffers.acquire_with_capacity::<Complex<T>>(fft_len)")
            && lanes.contains("Complex::<T>::pool_release(buffers, lane)"),
        "the FFT lane must come from the session buffer pool instead of a zero-filled allocation"
    );
}

#[test]
fn fft_cpu_execution_reuses_cached_rustfft_plans() {
    let cache_source = include_str!("../src/cache.rs");
    assert!(cache_source.contains("trait FftPlanProvider"));
    for source in [
        include_str!("../src/cpu.rs"),
        include_str!("../src/cpu/lanes.rs"),
    ] {
        assert!(
            !source.contains("FftPlanner::<T>::new()"),
            "CPU kernels must reuse owned plans"
        );
    }
}

#[test]
fn fft_cpu_execution_uses_explicit_plan_provider() {
    let source = include_str!("../src/cpu/lanes.rs");
    assert!(source.contains("cached_fft_plan::<T, _>(plans"));
    assert!(source.contains("plan.process_with_scratch(lane.as_mut_slice(), &mut scratch)"));
    assert!(
        source.find("let mut scratch =").unwrap() < source.find("for _ in start..end").unwrap(),
        "scratch must be reused across each job's lanes"
    );
}

#[test]
fn fft_source_has_no_process_global_plan_cache() {
    let sources = [
        include_str!("../src/lib.rs"),
        include_str!("../src/cache.rs"),
        include_str!("../src/cpu.rs"),
        include_str!("../src/cpu/lanes.rs"),
    ];

    for source in sources {
        assert!(!source.contains("static F32_FFT_PLAN_CACHE"));
        assert!(!source.contains("static F64_FFT_PLAN_CACHE"));
        assert!(!source.contains("OnceLock<Mutex"));
    }
}

#[test]
fn graph_runs_keep_fft_plans_in_the_engine() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
    let y = x.fft().unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut engine = CpuEngine::serial();
    let runtime = support::cpu_runtime_with_fft(&engine).unwrap();

    support::run_on_engine(&runtime, &mut engine, &program, &[]).unwrap();
    support::run_on_engine(&runtime, &mut engine, &program, &[]).unwrap();
    // The runtime lends the engine's extension caches to each run and returns
    // them, so the plan stays with the engine across runs (#2206).
    let mut caches = engine.take_extension_caches();
    let stats = caches.stats(ExtensionCacheSelector::All);
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes > 0);
    assert!(
        stats.hits >= 1,
        "the second run must reuse the first run's plan"
    );
    caches.clear();
    assert_eq!(caches.stats(ExtensionCacheSelector::All).entries, 0);
    engine.restore_extension_caches(caches);
}

#[test]
fn fft_c64_matches_numpy_convention() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
    let y = x.fft().unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    );
}

#[test]
fn fft_with_longer_transform_uses_zero_padded_lanes() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        )
        .unwrap();
    let y = x.fft_with(&FftOptions::default().n(Some(4))).unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, -2.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(1.0, 2.0),
        ],
    );
}

#[test]
fn fft_c32_uses_host_runtime() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
            ],
        )
        .unwrap();
    let y = x.fft().unwrap();
    let out = run(&y);

    assert_c32_close(
        out.as_slice::<Complex32>().unwrap(),
        &[
            Complex32::new(10.0, 0.0),
            Complex32::new(-2.0, 2.0),
            Complex32::new(-2.0, 0.0),
            Complex32::new(-2.0, -2.0),
        ],
    );
}

#[test]
fn ifft_c64_applies_backward_normalization() {
    let trace = tenferro_runtime::TraceContext::new();
    let spectrum = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(-2.0, -2.0),
            ],
        )
        .unwrap();
    let y = spectrum.ifft().unwrap();
    let out = run(&y);

    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    );
}

#[test]
fn rfft_f64_returns_onesided_spectrum() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();
    let y = x.rfft().unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[3]);
    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    );
}

#[test]
fn rfft_f32_returns_onesided_spectrum() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![4], vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap();
    let y = x.rfft().unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[3]);
    assert_c32_close(
        out.as_slice::<Complex32>().unwrap(),
        &[
            Complex32::new(10.0, 0.0),
            Complex32::new(-2.0, 2.0),
            Complex32::new(-2.0, 0.0),
        ],
    );
}

#[test]
fn irfft_c64_reconstructs_real_signal() {
    let trace = tenferro_runtime::TraceContext::new();
    let spectrum = trace
        .from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ],
        )
        .unwrap();
    let y = spectrum
        .irfft_with(&FftOptions::default().n(Some(4)))
        .unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_f64_close(out.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn irfft_c32_reconstructs_real_signal() {
    let trace = tenferro_runtime::TraceContext::new();
    let spectrum = trace
        .from_vec_col_major(
            vec![3],
            vec![
                Complex32::new(10.0, 0.0),
                Complex32::new(-2.0, 2.0),
                Complex32::new(-2.0, 0.0),
            ],
        )
        .unwrap();
    let y = spectrum
        .irfft_with(&FftOptions::default().n(Some(4)))
        .unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_f32_close(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn folded_normalization_matches_the_reference_for_ifft_irfft_and_ortho() {
    let trace = tenferro_runtime::TraceContext::new();
    // The normalization is folded into the scatter instead of a separate lane
    // pass. A length-4 transform uses exactly representable factors (1/4 for
    // Backward, 1/2 for Ortho), so a missing, doubled, or reordered scale
    // changes the result instead of hiding inside a tolerance.
    let spectrum = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(-2.0, -2.0),
            ],
        )
        .unwrap();
    for (norm, expected) in [
        (FftNorm::Backward, [1.0, 2.0, 3.0, 4.0]),
        (FftNorm::Ortho, [2.0, 4.0, 6.0, 8.0]),
    ] {
        let out = run(&spectrum
            .ifft_with(&FftOptions::default().norm(norm))
            .unwrap());
        let actual = out.as_slice::<Complex64>().unwrap();
        for (value, expected) in actual.iter().zip(expected) {
            assert_eq!(value.re, expected, "ifft norm={norm:?}");
            assert_eq!(value.im, 0.0, "ifft norm={norm:?}");
        }
    }

    let onesided = trace
        .from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ],
        )
        .unwrap();
    for (norm, expected) in [
        (FftNorm::Backward, [1.0, 2.0, 3.0, 4.0]),
        (FftNorm::Ortho, [2.0, 4.0, 6.0, 8.0]),
    ] {
        let out = run(&onesided
            .irfft_with(&FftOptions::default().n(Some(4)).norm(norm))
            .unwrap());
        let actual = out.as_slice::<f64>().unwrap();
        for (value, expected) in actual.iter().zip(expected) {
            assert_eq!(*value, expected, "irfft norm={norm:?}");
        }
    }
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_fft_matches_traced_fft() {
    let trace = tenferro_runtime::TraceContext::new();
    let input = Tensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -0.25),
            Complex64::new(3.0, 0.75),
            Complex64::new(4.0, -1.0),
        ],
    )
    .unwrap();
    let traced = trace
        .from_tensor_concrete_shape(input.duplicate().unwrap())
        .unwrap()
        .fft_with(&FftOptions::default().n(Some(3)).norm(FftNorm::Ortho))
        .unwrap();
    let input = eager(input);
    let mut engine = CpuEngine::serial();
    let eager = input
        .runtime()
        .with_eager_session(&mut engine, |s| {
            s.fft_with(
                &input,
                &FftOptions::default().n(Some(3)).norm(FftNorm::Ortho),
            )
        })
        .unwrap();

    assert_c64_close(
        eager
            .runtime()
            .with_eager_session(&mut engine, |s| s.duplicate_value(&eager))
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        run(&traced).as_slice::<Complex64>().unwrap(),
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_ifft_matches_traced_ifft() {
    let trace = tenferro_runtime::TraceContext::new();
    let input = Tensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    )
    .unwrap();
    let traced = trace
        .from_tensor_concrete_shape(input.duplicate().unwrap())
        .unwrap()
        .ifft_with(&FftOptions::default().axis(0).norm(FftNorm::Forward))
        .unwrap();
    let input = eager(input);
    let mut engine = CpuEngine::serial();
    let eager = input
        .runtime()
        .with_eager_session(&mut engine, |s| {
            s.ifft_with(
                &input,
                &FftOptions::default().axis(0).norm(FftNorm::Forward),
            )
        })
        .unwrap();

    assert_c64_close(
        eager
            .runtime()
            .with_eager_session(&mut engine, |s| s.duplicate_value(&eager))
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        run(&traced).as_slice::<Complex64>().unwrap(),
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_rfft_matches_traced_rfft() {
    let trace = tenferro_runtime::TraceContext::new();
    let input = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let traced = trace
        .from_tensor_concrete_shape(input.duplicate().unwrap())
        .unwrap()
        .rfft()
        .unwrap();
    let input = eager(input);
    let mut engine = CpuEngine::serial();
    let eager = input
        .runtime()
        .with_eager_session(&mut engine, |s| s.rfft(&input))
        .unwrap();

    assert_c64_close(
        eager
            .runtime()
            .with_eager_session(&mut engine, |s| s.duplicate_value(&eager))
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        run(&traced).as_slice::<Complex64>().unwrap(),
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_irfft_matches_traced_irfft() {
    let trace = tenferro_runtime::TraceContext::new();
    let input = Tensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    )
    .unwrap();
    let traced = trace
        .from_tensor_concrete_shape(input.duplicate().unwrap())
        .unwrap()
        .irfft_with(&FftOptions::default().n(Some(4)))
        .unwrap();
    let input = eager(input);
    let mut engine = CpuEngine::serial();
    let eager = input
        .runtime()
        .with_eager_session(&mut engine, |s| {
            s.irfft_with(&input, &FftOptions::default().n(Some(4)))
        })
        .unwrap();

    assert_f64_close(
        eager.value().unwrap().as_slice::<f64>().unwrap(),
        run(&traced).as_slice::<f64>().unwrap(),
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_fft_reuses_c2c_vjp_rule() {
    let ad = fft_ad_context();
    let mut engine = CpuEngine::serial();
    let ctx = EagerRuntime::with_ad_context(&ad).unwrap();
    let x = ctx
        .variable_from(
            Tensor::from_vec_col_major(
                vec![4],
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(2.0, 0.0),
                    Complex64::new(3.0, 0.0),
                    Complex64::new(4.0, 0.0),
                ],
            )
            .unwrap(),
        )
        .unwrap();
    let cotangent = ctx
        .constant_from(
            Tensor::from_vec_col_major(
                vec![4],
                vec![
                    Complex64::new(1.0, 1.0),
                    Complex64::new(2.0, -1.0),
                    Complex64::new(-1.0, 0.5),
                    Complex64::new(0.25, -2.0),
                ],
            )
            .unwrap(),
        )
        .unwrap();

    let y = ctx
        .with_eager_session(&mut engine, |session| session.fft(&x))
        .unwrap();
    let dx = ctx
        .with_eager_session(&mut engine, |s| s.vjp(&y, &x, &cotangent))
        .unwrap();

    assert_c64_close(
        dx.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(2.25, -1.5),
            Complex64::new(1.0, 2.25),
            Complex64::new(-2.25, 4.5),
            Complex64::new(3.0, -1.25),
        ],
    );
}

#[test]
fn irfft_rejects_spectrum_shorter_than_expected_one_sided_length() {
    let trace = tenferro_runtime::TraceContext::new();
    let spectrum = trace
        .from_vec_col_major(
            vec![2],
            vec![Complex64::new(10.0, 0.0), Complex64::new(-2.0, 2.0)],
        )
        .unwrap();

    let err = match spectrum.irfft_with(&FftOptions::default().n(Some(4))) {
        Ok(_) => panic!("expected irfft to reject a short one-sided spectrum"),
        Err(err) => err,
    };
    let message = err.to_string();

    assert!(message.contains("one-sided"), "{message}");
    assert!(message.contains("expected 3"), "{message}");
    assert!(message.contains("got 2"), "{message}");
}

#[test]
fn irfft_rejects_spectrum_longer_than_expected_one_sided_length() {
    let trace = tenferro_runtime::TraceContext::new();
    let spectrum = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(99.0, 0.0),
            ],
        )
        .unwrap();

    let err = match spectrum.irfft_with(&FftOptions::default().n(Some(4))) {
        Ok(_) => panic!("expected irfft to reject a long one-sided spectrum"),
        Err(err) => err,
    };
    let message = err.to_string();

    assert!(message.contains("one-sided"), "{message}");
    assert!(message.contains("expected 3"), "{message}");
    assert!(message.contains("got 4"), "{message}");
}

#[test]
fn traced_fft_rejects_invalid_dtype_axis_and_length() {
    let trace = tenferro_runtime::TraceContext::new();
    let int_input = trace.from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
    let err = match int_input.fft() {
        Ok(_) => panic!("expected fft to reject integer input"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        RuntimeError::Extension {
            op: "fft",
            phase: ErrorPhase::GraphBuild,
            family: "tenferro-fft.fft.v1",
            kind: ErrorKind::Unsupported,
            ..
        }
    ));

    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let err = match x.rfft_with(&FftOptions::default().n(Some(0))) {
        Ok(_) => panic!("expected rfft to reject zero transform length"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        RuntimeError::Validation {
            op: "rfft",
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::InvalidArgument { argument: "n", .. },
        }
    ));

    let zero_axis = trace
        .from_tensor_concrete_shape(Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap(),
        ))
        .unwrap();
    let err = match zero_axis.rfft() {
        Ok(_) => panic!("expected rfft to reject zero-length input axis"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        RuntimeError::Validation {
            op: "rfft",
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::InvalidArgument { argument: "n", .. },
        }
    ));

    let err = match x.rfft_with(&FftOptions::default().axis(3)) {
        Ok(_) => panic!("expected rfft to reject out-of-bounds axis"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        RuntimeError::Validation {
            op: "rfft",
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::AxisOutOfBounds { axis: 3, rank: 1 },
        }
    ));
}

#[test]
#[cfg(feature = "autodiff")]
fn fft_c64_jvp_applies_fft_to_tangent() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
    let dx = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap();

    let y = x.fft().unwrap();
    let dy = fft_ad_context().jvp(&y, &x, &dx).unwrap();
    let out = run(&dy);

    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 2.0),
        ],
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn fft_c64_jvp_matches_finite_diff() {
    let trace = tenferro_runtime::TraceContext::new();
    let data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, -0.25),
        Complex64::new(3.0, 0.75),
        Complex64::new(4.0, -1.0),
    ];
    let tangent_data = vec![
        Complex64::new(0.2, -0.1),
        Complex64::new(0.0, 0.3),
        Complex64::new(-0.4, 0.2),
        Complex64::new(0.1, -0.5),
    ];
    let x = trace.from_vec_col_major(vec![4], data.clone()).unwrap();
    let dx = trace
        .from_vec_col_major(vec![4], tangent_data.clone())
        .unwrap();

    let y = x.fft().unwrap();
    let dy = fft_ad_context().jvp(&y, &x, &dx).unwrap();
    let out = run(&dy);

    let expected = finite_diff_c64_directional(
        |xs| {
            let x = trace.from_vec_col_major(vec![4], xs.to_vec()).unwrap();
            let y = x.fft().unwrap();
            run(&y).as_slice::<Complex64>().unwrap().to_vec()
        },
        &data,
        &tangent_data,
        1.0e-6,
    );
    assert_c64_close_tol(out.as_slice::<Complex64>().unwrap(), &expected, 1.0e-8);
}

#[test]
#[cfg(feature = "autodiff")]
fn fft_c64_vjp_uses_inverse_transform_with_adjoint_normalization() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
    let cotangent = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.25, -2.0),
            ],
        )
        .unwrap();

    let y = x.fft().unwrap();
    let dx = fft_ad_context().vjp(&y, &x, &cotangent).unwrap();
    let out = run(&dx);

    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(2.25, -1.5),
            Complex64::new(1.0, 2.25),
            Complex64::new(-2.25, 4.5),
            Complex64::new(3.0, -1.25),
        ],
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn fft_c64_vjp_longer_transform_slices_to_input_length() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        )
        .unwrap();
    let cotangent = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.25, -2.0),
            ],
        )
        .unwrap();

    let y = x.fft_with(&FftOptions::default().n(Some(4))).unwrap();
    let dx = fft_ad_context().vjp(&y, &x, &cotangent).unwrap();
    let out = run(&dx);

    assert_eq!(out.shape(), &[2]);
    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2.25, -1.5), Complex64::new(1.0, 2.25)],
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn fft_c64_vjp_shorter_transform_pads_to_input_length() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
    let cotangent = trace
        .from_vec_col_major(
            vec![2],
            vec![Complex64::new(3.0, 1.0), Complex64::new(-1.0, 2.0)],
        )
        .unwrap();

    let y = x.fft_with(&FftOptions::default().n(Some(2))).unwrap();
    let dx = fft_ad_context().vjp(&y, &x, &cotangent).unwrap();
    let out = run(&dx);

    assert_eq!(out.shape(), &[4]);
    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(2.0, 3.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn ifft_c64_vjp_uses_forward_transform_with_adjoint_normalization() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(-2.0, -2.0),
            ],
        )
        .unwrap();
    let cotangent = trace
        .from_vec_col_major(
            vec![4],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.25, -2.0),
            ],
        )
        .unwrap();

    let y = x.ifft().unwrap();
    let dx = fft_ad_context().vjp(&y, &x, &cotangent).unwrap();
    let out = run(&dx);

    assert_c64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(0.5625, -0.375),
            Complex64::new(0.75, -0.3125),
            Complex64::new(-0.5625, 1.125),
            Complex64::new(0.25, 0.5625),
        ],
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn rfft_vjp_unsupported_error_names_rfft_and_vjp() {
    let trace = tenferro_runtime::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();
    let cotangent = trace
        .from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, -1.0),
                Complex64::new(2.0, 0.0),
            ],
        )
        .unwrap();

    let y = x.rfft().unwrap();
    let err = match fft_ad_context().vjp_optional(&y, &x, &cotangent) {
        Ok(_) => panic!("rfft VJP should remain unsupported"),
        Err(err) => err,
    };
    let message = err.to_string();

    assert!(
        message.contains("unsupported vjp AD rule for tenferro-fft.rfft.v1"),
        "{message}"
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn irfft_jvp_unsupported_error_names_irfft_and_jvp() {
    let trace = tenferro_runtime::TraceContext::new();
    let spectrum = trace
        .from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ],
        )
        .unwrap();
    let tangent = trace
        .from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, -0.5),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

    let y = spectrum
        .irfft_with(&FftOptions::default().n(Some(4)))
        .unwrap();
    let err = match fft_ad_context().jvp_optional(&y, &spectrum, &tangent) {
        Ok(_) => panic!("irfft JVP should remain unsupported"),
        Err(err) => err,
    };
    let message = err.to_string();

    assert!(
        message.contains("unsupported jvp AD rule for tenferro-fft.irfft.v1"),
        "{message}"
    );
}

/// A symbolic `irfft` with `n = None` whose spectrum axis is 1 at prepare time
/// publishes a zero output extent, so preparation must refuse it exactly like the
/// concrete path does instead of deferring the failure to execution (#2434).
#[test]
fn irfft_length_one_spectrum_is_rejected_at_preparation() {
    let trace = tenferro_runtime::TraceContext::new();
    let symbolic = trace.input_symbolic_shape(DType::C64, 1).unwrap();
    let output = symbolic.irfft().unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output).unwrap();
    assert_eq!(program.input_count(), 1);

    let engine = CpuEngine::serial();
    let runtime = support::cpu_runtime_with_fft(&engine).unwrap();
    let spectrum = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]).unwrap();
    let reads = [tenferro_tensor::TensorRead::from_tensor(&spectrum)];
    let error = runtime
        .prepare(&program, &reads, &[engine.target()])
        .expect_err("a length-1 spectrum would publish a zero output extent");

    match &error {
        RuntimeError::Validation { phase, source, .. } => {
            assert_eq!(
                *phase,
                ErrorPhase::Compile,
                "preparation must reject the input, not execution: {error:?}"
            );
            assert!(
                matches!(
                    source,
                    ValidationError::InvalidArgument { argument, .. } if *argument == "output length"
                ),
                "the concrete path's typed validation must survive: {error:?}"
            );
        }
        other => panic!("expected typed validation at preparation, got {other:?}"),
    }
    assert!(error.to_string().contains("must be positive"), "{error}");
}

/// The preparation guard must not reject a symbolic `irfft` whose resolved
/// spectrum length does publish a positive output extent.
#[test]
fn symbolic_irfft_with_a_positive_resolved_extent_still_prepares() {
    let trace = tenferro_runtime::TraceContext::new();
    let symbolic = trace.input_symbolic_shape(DType::C64, 1).unwrap();
    let output = symbolic.irfft().unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output).unwrap();

    let mut engine = CpuEngine::serial();
    let runtime = support::cpu_runtime_with_fft(&engine).unwrap();
    let spectrum = Tensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    )
    .unwrap();
    let reads = [tenferro_tensor::TensorRead::from_tensor(&spectrum)];
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .expect("a length-3 spectrum publishes a positive output extent");
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .expect("run");
    let output = outputs
        .into_iter()
        .next()
        .unwrap()
        .into_tensor(&mut engine)
        .unwrap();
    assert_eq!(output.shape(), &[4]);
}

/// A zero on the batch axis is a legitimate empty transform, not an invalid extent.
///
/// `irfft` transforms the last axis only, so a symbolic `[0, 3]` spectrum infers
/// `[0, 4]`: the leading zero is an empty batch that the design allows
/// (`docs/design/fft-backend-execution.md`) and the CPU lanes return early for, while
/// the transformed axis carries the positive length 4. Preparation must accept it
/// (#2449 narrowed an over-broad guard here).
#[test]
fn symbolic_irfft_over_an_empty_batch_prepares_and_runs() {
    let trace = tenferro_runtime::TraceContext::new();
    let symbolic = trace.input_symbolic_shape(DType::C64, 2).unwrap();
    let output = symbolic.irfft().unwrap();
    assert_eq!(
        output.sym_shape().map(<[_]>::len),
        Some(2),
        "the transform keeps the batch rank"
    );
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output).unwrap();

    let mut engine = CpuEngine::serial();
    let runtime = support::cpu_runtime_with_fft(&engine).unwrap();
    let spectrum = Tensor::from_vec_col_major(vec![0, 3], Vec::<Complex64>::new()).unwrap();
    let reads = [tenferro_tensor::TensorRead::from_tensor(&spectrum)];
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .expect("an empty batch is a legitimate input");
    let outputs = runtime
        .run(
            &mut [&mut engine as &mut dyn ExecutionEngine],
            &prepared,
            &reads,
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    let tensor = outputs
        .into_iter()
        .next()
        .unwrap()
        .into_tensor(&mut engine)
        .unwrap();
    assert_eq!(tensor.shape(), &[0, 4]);
}

/// The concrete route takes the same decision for the same numbers.
#[test]
fn concrete_irfft_over_an_empty_batch_prepares_and_runs() {
    let trace = tenferro_runtime::TraceContext::new();
    let concrete = trace.input_concrete_shape(DType::C64, &[0, 3]).unwrap();
    let output = concrete.irfft().unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&output).unwrap();

    let mut engine = CpuEngine::serial();
    let runtime = support::cpu_runtime_with_fft(&engine).unwrap();
    let spectrum = Tensor::from_vec_col_major(vec![0, 3], Vec::<Complex64>::new()).unwrap();
    let reads = [tenferro_tensor::TensorRead::from_tensor(&spectrum)];
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .expect("a concrete empty batch must take the symbolic route's decision");
    let outputs = runtime
        .run(
            &mut [&mut engine as &mut dyn ExecutionEngine],
            &prepared,
            &reads,
        )
        .unwrap();
    let tensor = outputs
        .into_iter()
        .next()
        .unwrap()
        .into_tensor(&mut engine)
        .unwrap();
    assert_eq!(tensor.shape(), &[0, 4]);
}

/// Compile, prepare and run one traced `irfft` over `spectrum`, returning the shape.
fn irfft_prepared_shape(
    engine: &mut CpuEngine,
    output: &TracedTensor,
    spectrum: &Tensor,
) -> Vec<usize> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).unwrap();
    let runtime = support::cpu_runtime_with_fft(engine).unwrap();
    let reads = [tenferro_tensor::TensorRead::from_tensor(spectrum)];
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .expect("an empty batch is a legitimate input");
    let outputs = runtime
        .run(&mut [engine as &mut dyn ExecutionEngine], &prepared, &reads)
        .unwrap();
    outputs
        .into_iter()
        .next()
        .unwrap()
        .into_tensor(engine)
        .unwrap()
        .shape()
        .to_vec()
}

/// The symbolic and concrete routes must take the same decision for the same numbers.
///
/// `irfft(None, -1)` over a `[0, 3]` spectrum is a legitimate empty batch: the last
/// axis is transformed to length 4 and the leading zero is carried through, so the
/// op infers `[0, 4]`. The traced program only learns the extents at preparation,
/// while the concrete route knows them when the op is appended; preparation must not
/// apply an extent rule the extension does not have, or the two disagree on the same
/// numbers (#2449 rejected this input before the extension could accept it).
///
/// This pins a zero the op *carries* from an empty input axis. A zero the op derives
/// is rejected by the op's own inference before any runtime rule could fire, which
/// `irfft_length_one_spectrum_is_rejected_at_preparation` covers.
#[test]
fn symbolic_and_concrete_irfft_agree_on_an_empty_batch() {
    let spectrum = Tensor::from_vec_col_major(vec![0, 3], Vec::<Complex64>::new()).unwrap();
    let mut engine = CpuEngine::serial();

    let symbolic_trace = tenferro_runtime::TraceContext::new();
    let symbolic_input = symbolic_trace.input_symbolic_shape(DType::C64, 2).unwrap();
    let symbolic_output = symbolic_input.irfft().unwrap();

    let concrete_trace = tenferro_runtime::TraceContext::new();
    let concrete_input = concrete_trace
        .input_concrete_shape(DType::C64, &[0, 3])
        .unwrap();
    let concrete_output = concrete_input.irfft().unwrap();

    // The inference rule, read off the route that knows the extents at build time.
    assert_eq!(
        concrete_output.sym_shape(),
        Some([SymDim::from(0usize), SymDim::from(4usize)].as_slice()),
        "irfft over [0, 3] infers [0, 4]"
    );

    let symbolic_shape = irfft_prepared_shape(&mut engine, &symbolic_output, &spectrum);
    let concrete_shape = irfft_prepared_shape(&mut engine, &concrete_output, &spectrum);

    assert_eq!(
        symbolic_shape, concrete_shape,
        "the traced program and the concrete route must agree on [0, 3]"
    );
    assert_eq!(symbolic_shape, vec![0, 4]);
}
