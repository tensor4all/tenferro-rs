use num_complex::{Complex32, Complex64};
use std::error::Error as StdError;
use std::panic::{catch_unwind, AssertUnwindSafe};
mod support;
#[cfg(feature = "autodiff")]
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
#[cfg(feature = "autodiff")]
use tenferro_fft::EagerTensorFftExt;
use tenferro_fft::{FftNorm, TracedTensorFftExt};
use tenferro_runtime::{
    DType, Error as RuntimeError, ErrorPhase, GraphCompiler, PrepareError, Tensor, TracedTensor,
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
    EagerTensor::from_tensor_in(input, EagerRuntime::new().unwrap()).unwrap()
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

fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, after_start) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source section start {start}"));
    let (section, _) = after_start
        .split_once(end)
        .unwrap_or_else(|| panic!("missing source section end {end}"));
    section
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
    Tensor::C64(
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
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = x.rfft(None, -1, FftNorm::Backward).unwrap();

    assert_eq!(y.rank, 1);
}

#[test]
fn registered_runtime_rejects_gpu_input_without_ingress() {
    let x = TracedTensor::input_concrete_shape(DType::C64, &[2]).unwrap();
    let y = x.fft(None, -1, FftNorm::Backward).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::C64, &[2])])
        .unwrap();
    let gpu_input = cuda_c64_tensor(vec![2]);
    let backend = CpuBackend::new();
    let runtime = support::cpu_runtime_with_fft(&backend).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        runtime.run_compiled(&program, &[&gpu_input])
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
    assert!(
        source.contains("let mut lane = vec![Complex::zero(); fft_len]")
            && source.contains("// INVARIANT: zero-fill is transform padding semantics")
            && source.contains("lane.fill(Complex::zero())"),
        "FFT CPU scratch lanes must stay zero-filled for transform padding and carry an invariant marker"
    );
}

#[test]
fn fft_cpu_execution_reuses_cached_rustfft_plans() {
    let source = include_str!("../src/cpu.rs");
    let cache_source = include_str!("../src/cache.rs");
    let c2c = source_section(source, "fn execute_c2c<T>(", "fn execute_r2c<T>(");
    let r2c = source_section(source, "fn execute_r2c<T>(", "fn execute_c2r<T>(");
    let c2r = source_section(source, "fn execute_c2r<T>(", "fn scale_for<T>(");

    assert!(
        cache_source.contains("trait FftPlanProvider"),
        "FFT CPU execution should obtain plans from an explicit owner"
    );
    for (name, section) in [
        ("execute_c2c", c2c),
        ("execute_r2c", r2c),
        ("execute_c2r", c2r),
    ] {
        assert!(
            !section.contains("FftPlanner::<T>::new()"),
            "{name} must not rebuild a RustFFT planner per call"
        );
    }
}

#[test]
fn fft_cpu_execution_uses_explicit_plan_provider() {
    let source = include_str!("../src/cpu.rs");
    let c2c = source_section(source, "fn execute_c2c<T>(", "fn execute_r2c<T>(");
    let r2c = source_section(source, "fn execute_r2c<T>(", "fn execute_c2r<T>(");
    let c2r = source_section(source, "fn execute_c2r<T>(", "fn scale_for<T>(");

    for (name, section) in [
        ("execute_c2c", c2c),
        ("execute_r2c", r2c),
        ("execute_c2r", c2r),
    ] {
        let call_start = section
            .find("cached_fft_plan::<T, _>(plans")
            .unwrap_or_else(|| panic!("{name} must use its explicit plan provider"));
        let call_end = section[call_start..]
            .find(';')
            .map(|offset| call_start + offset + 1)
            .unwrap_or_else(|| panic!("{name} cached_fft_plan call must end in a statement"));
        let normalized_call = section[call_start..call_end]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        assert!(
            normalized_call.ends_with(");"),
            "unexpected plan call: {normalized_call}"
        );
    }
}

#[test]
fn fft_source_has_no_process_global_plan_cache() {
    let sources = [
        include_str!("../src/lib.rs"),
        include_str!("../src/cache.rs"),
        include_str!("../src/cpu.rs"),
    ];

    for source in sources {
        assert!(!source.contains("static F32_FFT_PLAN_CACHE"));
        assert!(!source.contains("static F64_FFT_PLAN_CACHE"));
        assert!(!source.contains("OnceLock<Mutex"));
    }
}

#[test]
fn graph_runtime_owns_fft_plan_cache() {
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let y = x.fft(None, -1, FftNorm::Backward).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let backend = CpuBackend::new();
    let runtime = support::cpu_runtime_with_fft(&backend).unwrap();

    runtime.run_compiled(&program, &[]).unwrap();
    runtime.run_compiled(&program, &[]).unwrap();
    let stats = runtime.cache_stats().unwrap().extensions;
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes > 0);

    runtime.clear_caches().unwrap();
    assert_eq!(runtime.cache_stats().unwrap().extensions.entries, 0);
}

#[test]
fn fft_c64_matches_numpy_convention() {
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let y = x.fft(None, -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();
    let y = x.fft(Some(4), -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let y = x.fft(None, -1, FftNorm::Backward).unwrap();
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
    let spectrum = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    )
    .unwrap();
    let y = spectrum.ifft(None, -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let y = x.rfft(None, -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let y = x.rfft(None, -1, FftNorm::Backward).unwrap();
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
    let spectrum = TracedTensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    )
    .unwrap();
    let y = spectrum.irfft(Some(4), -1, FftNorm::Backward).unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_f64_close(out.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn irfft_c32_reconstructs_real_signal() {
    let spectrum = TracedTensor::from_vec_col_major(
        vec![3],
        vec![
            Complex32::new(10.0, 0.0),
            Complex32::new(-2.0, 2.0),
            Complex32::new(-2.0, 0.0),
        ],
    )
    .unwrap();
    let y = spectrum.irfft(Some(4), -1, FftNorm::Backward).unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_f32_close(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_fft_matches_traced_fft() {
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
    let traced = TracedTensor::from_tensor_concrete_shape(input.clone())
        .unwrap()
        .fft(Some(3), -1, FftNorm::Ortho)
        .unwrap();
    let eager = eager(input).fft(Some(3), -1, FftNorm::Ortho).unwrap();

    assert_c64_close(
        eager
            .materialized()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        run(&traced).as_slice::<Complex64>().unwrap(),
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_ifft_matches_traced_ifft() {
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
    let traced = TracedTensor::from_tensor_concrete_shape(input.clone())
        .unwrap()
        .ifft(None, 0, FftNorm::Forward)
        .unwrap();
    let eager = eager(input).ifft(None, 0, FftNorm::Forward).unwrap();

    assert_c64_close(
        eager
            .materialized()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        run(&traced).as_slice::<Complex64>().unwrap(),
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_rfft_matches_traced_rfft() {
    let input = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let traced = TracedTensor::from_tensor_concrete_shape(input.clone())
        .unwrap()
        .rfft(None, -1, FftNorm::Backward)
        .unwrap();
    let eager = eager(input).rfft(None, -1, FftNorm::Backward).unwrap();

    assert_c64_close(
        eager
            .materialized()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        run(&traced).as_slice::<Complex64>().unwrap(),
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_irfft_matches_traced_irfft() {
    let input = Tensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    )
    .unwrap();
    let traced = TracedTensor::from_tensor_concrete_shape(input.clone())
        .unwrap()
        .irfft(Some(4), -1, FftNorm::Backward)
        .unwrap();
    let eager = eager(input).irfft(Some(4), -1, FftNorm::Backward).unwrap();

    assert_f64_close(
        eager.materialized().unwrap().as_slice::<f64>().unwrap(),
        run(&traced).as_slice::<f64>().unwrap(),
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn eager_fft_reuses_c2c_vjp_rule() {
    let ad = fft_ad_context();
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad).unwrap();
    let x = EagerTensor::requires_grad_in(
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
        ctx.clone(),
    )
    .unwrap();
    let cotangent = EagerTensor::from_tensor_in(
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
        ctx.clone(),
    )
    .unwrap();

    let y = x.fft(None, -1, FftNorm::Backward).unwrap();
    let dx = ctx.vjp(&y, &x, &cotangent).unwrap();

    assert_c64_close(
        dx.materialized().unwrap().as_slice::<Complex64>().unwrap(),
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
    let spectrum = TracedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(10.0, 0.0), Complex64::new(-2.0, 2.0)],
    )
    .unwrap();

    let err = match spectrum.irfft(Some(4), -1, FftNorm::Backward) {
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
    let spectrum = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(99.0, 0.0),
        ],
    )
    .unwrap();

    let err = match spectrum.irfft(Some(4), -1, FftNorm::Backward) {
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
    let int_input = TracedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
    let err = match int_input.fft(None, -1, FftNorm::Backward) {
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

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = match x.rfft(Some(0), -1, FftNorm::Backward) {
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

    let zero_axis = TracedTensor::from_tensor_concrete_shape(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap(),
    ))
    .unwrap();
    let err = match zero_axis.rfft(None, -1, FftNorm::Backward) {
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

    let err = match x.rfft(None, 3, FftNorm::Backward) {
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
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let dx = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();

    let y = x.fft(None, -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(vec![4], data.clone()).unwrap();
    let dx = TracedTensor::from_vec_col_major(vec![4], tangent_data.clone()).unwrap();

    let y = x.fft(None, -1, FftNorm::Backward).unwrap();
    let dy = fft_ad_context().jvp(&y, &x, &dx).unwrap();
    let out = run(&dy);

    let expected = finite_diff_c64_directional(
        |xs| {
            let x = TracedTensor::from_vec_col_major(vec![4], xs.to_vec()).unwrap();
            let y = x.fft(None, -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let cotangent = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.25, -2.0),
        ],
    )
    .unwrap();

    let y = x.fft(None, -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();
    let cotangent = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.25, -2.0),
        ],
    )
    .unwrap();

    let y = x.fft(Some(4), -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let cotangent = TracedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 1.0), Complex64::new(-1.0, 2.0)],
    )
    .unwrap();

    let y = x.fft(Some(2), -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    )
    .unwrap();
    let cotangent = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.25, -2.0),
        ],
    )
    .unwrap();

    let y = x.ifft(None, -1, FftNorm::Backward).unwrap();
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
    let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let cotangent = TracedTensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -1.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();

    let y = x.rfft(None, -1, FftNorm::Backward).unwrap();
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
    let spectrum = TracedTensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    )
    .unwrap();
    let tangent = TracedTensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();

    let y = spectrum.irfft(Some(4), -1, FftNorm::Backward).unwrap();
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

#[test]
fn webgpu_fft_adapter_keeps_client_ownership_and_raw_output_lifecycle_explicit() {
    let source = include_str!("../src/webgpu.rs");
    for launch in [
        "cfft_interleaved_launch(",
        "rfft_interleaved_launch_padded(",
        "irfft_interleaved_launch_padded(",
    ] {
        assert!(
            source.contains(launch),
            "missing explicit client launch {launch}"
        );
    }
    assert!(source.contains("webgpu_interop::client(backend)"));
    assert!(source.contains("output.into_raw_parts().handle"));
    assert!(source.contains("output.handle"));
    assert!(source.contains("webgpu_interop::finish_c32"));
    assert!(source.contains("webgpu_interop::finish_f32"));
    assert!(!source.contains("download_webgpu_tensor"));
    assert!(!source.contains("upload_webgpu_tensor"));
    assert!(source.contains("_cache: FftExecutionCache<'_>"));
    assert!(source.contains("Error::backend_source(op, error)"));
}
