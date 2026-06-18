use num_complex::{Complex32, Complex64};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
#[cfg(feature = "autodiff")]
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{
    traced_tensor::{fft, ifft, irfft, rfft},
    FftNorm,
};
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::{
    Buffer, BufferHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, TypedTensor,
};

fn run(output: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_fft::register_runtime)
        .unwrap();
    executor.run(&program).unwrap()
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
            Buffer::Backend(Arc::new(BufferHandle::<Complex64>::new_with_len(7, len))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
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
fn traced_tensor_namespace_exposes_fft() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = tenferro_fft::traced_tensor::rfft(&x, None, -1, FftNorm::Backward).unwrap();

    assert_eq!(y.rank, 1);
}

#[test]
fn registered_runtime_reports_gpu_input_as_unsupported() {
    let x = TracedTensor::input_concrete_shape(DType::C64, &[2]);
    let y = fft(&x, None, -1, FftNorm::Backward).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::C64, &[2])])
        .unwrap();
    let gpu_input = cuda_c64_tensor(vec![2]);
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_fft::register_runtime)
        .unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        executor.run_with_inputs(&program, &[(&x, &gpu_input)])
    }));
    assert!(
        result.is_ok(),
        "FFT GPU input should return an error, not panic"
    );
    let err = result
        .unwrap()
        .expect_err("FFT GPU input should be unsupported");
    let message = err.to_string();
    assert!(message.contains("unsupported"), "{message}");
    assert!(message.contains("download"), "{message}");
}

#[test]
fn fft_cpu_output_buffers_avoid_zero_fill_but_keep_lane_padding() {
    let source = include_str!("../src/lib.rs");

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
            && source.contains("lane.fill(Complex::zero())"),
        "FFT CPU scratch lanes must stay zero-filled for transform padding"
    );
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
    let y = fft(&x, None, -1, FftNorm::Backward).unwrap();
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
    let y = fft(&x, Some(4), -1, FftNorm::Backward).unwrap();
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
    let y = fft(&x, None, -1, FftNorm::Backward).unwrap();
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
    let y = ifft(&spectrum, None, -1, FftNorm::Backward).unwrap();
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
    let y = rfft(&x, None, -1, FftNorm::Backward).unwrap();
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
    let y = rfft(&x, None, -1, FftNorm::Backward).unwrap();
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
    let y = irfft(&spectrum, Some(4), -1, FftNorm::Backward).unwrap();
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
    let y = irfft(&spectrum, Some(4), -1, FftNorm::Backward).unwrap();
    let out = run(&y);

    assert_eq!(out.shape(), &[4]);
    assert_f32_close(out.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn traced_fft_rejects_invalid_dtype_axis_and_length() {
    let int_input = TracedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
    let err = match fft(&int_input, None, -1, FftNorm::Backward) {
        Ok(_) => panic!("expected fft to reject integer input"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("floating"), "{err}");

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = match rfft(&x, Some(0), -1, FftNorm::Backward) {
        Ok(_) => panic!("expected rfft to reject zero transform length"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("positive"), "{err}");

    let err = match rfft(&x, None, 3, FftNorm::Backward) {
        Ok(_) => panic!("expected rfft to reject out-of-bounds axis"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("out of bounds"), "{err}");
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

    let y = fft(&x, None, -1, FftNorm::Backward).unwrap();
    let dy = y.jvp(&x, &dx).unwrap();
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

    let y = fft(&x, None, -1, FftNorm::Backward).unwrap();
    let dy = y.jvp(&x, &dx).unwrap();
    let out = run(&dy);

    let expected = finite_diff_c64_directional(
        |xs| {
            let x = TracedTensor::from_vec_col_major(vec![4], xs.to_vec()).unwrap();
            let y = fft(&x, None, -1, FftNorm::Backward).unwrap();
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

    let y = fft(&x, None, -1, FftNorm::Backward).unwrap();
    let dx = y.vjp(&x, &cotangent).unwrap();
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

    let y = ifft(&x, None, -1, FftNorm::Backward).unwrap();
    let dx = y.vjp(&x, &cotangent).unwrap();
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

    let y = rfft(&x, None, -1, FftNorm::Backward).unwrap();
    let err = match y.vjp_optional(&x, &cotangent) {
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

    let y = irfft(&spectrum, Some(4), -1, FftNorm::Backward).unwrap();
    let err = match y.jvp_optional(&spectrum, &tangent) {
        Ok(_) => panic!("irfft JVP should remain unsupported"),
        Err(err) => err,
    };
    let message = err.to_string();

    assert!(
        message.contains("unsupported jvp AD rule for tenferro-fft.irfft.v1"),
        "{message}"
    );
}
