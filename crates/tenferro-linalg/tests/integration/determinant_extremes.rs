use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, TracedTensor};
use tenferro_tensor::Tensor;

fn evaluate(a: &TracedTensor, input: Tensor) -> Vec<Tensor> {
    let (sign, logabsdet) = a.slogdet().unwrap();
    let det = a.det().unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&det, &sign, &logabsdet])
        .unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let runtime = super::support::cpu_runtime_with_linalg(&backend).unwrap();
    let result = super::support::run_program(&runtime, &mut backend, &program, &[&input]).unwrap();
    #[cfg(feature = "autodiff")]
    {
        use tenferro_ad::{EagerRuntime, EagerTensor};
        use tenferro_linalg::EagerSessionLinalgExt;
        let runtime = EagerRuntime::new().unwrap();
        let eager = EagerTensor::from_tensor_in(input, runtime.clone()).unwrap();
        let (det, sign, log) = runtime
            .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
                let (sign, log) = session.slogdet(&eager)?;
                Ok::<_, tenferro_ad::Error>((session.det(&eager)?, sign, log))
            })
            .unwrap();
        for (actual, expected) in [det, sign, log].iter().zip(&result) {
            let actual = actual.to_tensor(&mut tenferro_cpu::CpuEngine::serial()).unwrap();
            assert_eq!(actual.shape(), expected.shape());
            if let Ok(values) = actual.as_slice::<f64>() {
                for (&a, &b) in values.iter().zip(expected.as_slice::<f64>().unwrap()) {
                    assert!(a == b || (b.is_finite() && (a - b).abs() <= 1e-12 * b.abs()));
                }
            } else {
                for (&a, &b) in actual
                    .as_slice::<Complex64>()
                    .unwrap()
                    .iter()
                    .zip(expected.as_slice::<Complex64>().unwrap())
                {
                    assert!((a - b).norm() <= 1e-12);
                }
            }
        }
    }
    result
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA"]
fn cuda_complex_determinant_retains_existing_support() {
    use tenferro_gpu::cuda::{cuda_runtime_engine_registration, download_tensor, upload_tensor, CudaBackend, CudaDeviceId};
    use tenferro_runtime::{EngineId, Runtime};
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, -1.0),
    ];
    let host = Tensor::from_vec_col_major([2, 2], data.clone()).unwrap();
    let input = upload_tensor(backend.runtime(), &host).unwrap();
    let traced = TracedTensor::from_vec_col_major([2, 2], data).unwrap();
    let program = GraphCompiler::new()
        .compile(&traced.det().unwrap())
        .unwrap();
    let engine = EngineId::new("tenferro.test.cuda.det.v1").unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(cuda_runtime_engine_registration(&backend, engine.clone()).unwrap())
        .unwrap();
    builder
        .install_extension_module(tenferro_linalg::extension_module::<CudaBackend>(engine).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();
    let result = runtime.run_compiled(&program, &[&input]).unwrap();
    let host = download_tensor(backend.runtime(), &result[0]).unwrap();
    assert!((host.as_slice::<Complex64>().unwrap()[0] - Complex64::new(3.0, 1.0)).norm() < 1e-12);
}

#[test]
fn determinant_extremes_preserve_value_and_sign() {
    // Each tuple is one diagonal matrix, expected determinant, and expected sign.
    for (diagonal, det, sign) in [
        (vec![2.0, 3.0], 6.0, 1.0),
        (vec![1e200, 1e200, 1e-200, 1e-200], 1.0, 1.0),
        ([vec![1e200; 8], vec![1e-200; 8]].concat(), 1.0, 1.0),
        (vec![-1e-200, 1e-200], 0.0, -1.0),
        (vec![1e200, 1e200], f64::INFINITY, 1.0),
        (vec![1.0, 0.0], 0.0, 0.0),
    ] {
        let n = diagonal.len();
        let mut data = vec![0.0; n * n];
        for (i, &value) in diagonal.iter().enumerate() {
            data[i + n * i] = value;
        }
        let input = Tensor::from_vec_col_major([n, n], data.clone()).unwrap();
        let a = TracedTensor::from_vec_col_major([n, n], data).unwrap();
        let result = evaluate(&a, input);
        let actual_det = result[0].as_slice::<f64>().unwrap()[0];
        let actual_sign = result[1].as_slice::<f64>().unwrap()[0];
        let logabsdet = result[2].as_slice::<f64>().unwrap()[0];
        assert_eq!(actual_sign, sign, "diagonal={diagonal:?}");
        assert!(
            actual_det == det || (det.is_finite() && (actual_det - det).abs() <= 1e-12 * det.abs()),
            "diagonal={diagonal:?}: {actual_det} != {det}"
        );
        assert_eq!(actual_det, actual_sign * logabsdet.exp());
    }
}

#[test]
fn batched_slogdet_tracks_pivot_parity_and_each_batch() {
    let data = vec![0.0, 1e-200, 1e-200, 0.0, 2.0, 0.0, 0.0, 3.0];
    let input = Tensor::from_vec_col_major([2, 2, 2], data.clone()).unwrap();
    let a = TracedTensor::from_vec_col_major([2, 2, 2], data).unwrap();
    let result = evaluate(&a, input);
    assert_eq!(result[1].as_slice::<f64>().unwrap(), &[-1.0, 1.0]);
    let det = result[0].as_slice::<f64>().unwrap();
    assert_eq!(det[0], 0.0);
    assert!((det[1] - 6.0).abs() < 1e-12);
    let log = result[2].as_slice::<f64>().unwrap();
    assert!((log[0] - 2.0 * 1e-200_f64.ln()).abs() < 1e-12);
    assert!((log[1] - 6.0_f64.ln()).abs() < 1e-12);
}

#[test]
fn complex_slogdet_preserves_tiny_phase_and_singular_zero() {
    for (first, expected_sign) in [
        (Complex64::new(0.0, 1e-200), Complex64::new(0.0, 1.0)),
        (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)),
    ] {
        let data = vec![
            Complex64::new(1e-200, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            first,
        ];
        let input = Tensor::from_vec_col_major([2, 2], data.clone()).unwrap();
        let a = TracedTensor::from_vec_col_major([2, 2], data).unwrap();
        let result = evaluate(&a, input);
        assert_eq!(result[1].as_slice::<Complex64>().unwrap(), &[expected_sign]);
        assert_eq!(
            result[0].as_slice::<Complex64>().unwrap(),
            &[Complex64::new(0.0, 0.0)]
        );
    }
}
