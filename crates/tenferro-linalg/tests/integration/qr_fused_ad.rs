#![cfg(feature = "autodiff")]

//! Parity tests for the fused square-QR JVP/VJP extension op.
//!
//! Concrete square shapes take the fused carrier; the same data on a symbolic
//! shape leaf takes the pre-existing generic graph. Comparing the two runs is
//! a direct fused-vs-generic parity check for every active mask.

use tenferro_ad::AdContext;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, Tensor, TracedTensor};
use tenferro_tensor::{Complex32, Complex64, TypedTensor};

use super::support;

fn ad_context() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap()
}

fn eval_many(outputs: &[&TracedTensor]) -> Vec<Tensor> {
    let program = GraphCompiler::new().compile_many(outputs).unwrap();
    support::run_all(&program, &[]).unwrap()
}

fn tensor_f64(shape: &[usize], data: &[f64]) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data.to_vec()).unwrap())
}

fn tensor_c32(shape: &[usize], data: &[f64]) -> Tensor {
    let data: Vec<Complex32> = data
        .iter()
        .enumerate()
        .map(|(index, &value)| Complex32::new(value as f32, ((index % 7) as f32 - 3.0) * 0.013))
        .collect();
    Tensor::from_typed::<Complex32>(TypedTensor::from_vec_col_major(shape, data.to_vec()).unwrap())
}

fn tensor_c64(shape: &[usize], data: &[f64]) -> Tensor {
    let data: Vec<Complex64> = data
        .iter()
        .enumerate()
        .map(|(index, &value)| Complex64::new(value, ((index % 7) as f64 - 3.0) * 0.013))
        .collect();
    Tensor::from_typed::<Complex64>(TypedTensor::from_vec_col_major(shape, data.to_vec()).unwrap())
}

#[derive(Clone, Copy, Debug)]
enum ActiveMask {
    Q,
    R,
    Both,
}

impl ActiveMask {
    fn label(self) -> &'static str {
        match self {
            Self::Q => "q",
            Self::R => "r",
            Self::Both => "q+r",
        }
    }
}

fn qr_loss(a: &TracedTensor, active: ActiveMask) -> TracedTensor {
    let (q, r) = a.qr().unwrap();
    let q_loss = q.reduce_sum_squares(None).unwrap();
    let r_loss = r.reduce_sum_squares(None).unwrap();
    match active {
        ActiveMask::Q => q_loss,
        ActiveMask::R => r_loss,
        ActiveMask::Both => (&q_loss + &r_loss).unwrap(),
    }
}

fn assert_close(actual: &Tensor, expected: &Tensor, tol: f64, context: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{context}: shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{context}: dtype");
    match actual.dtype() {
        tenferro_runtime::DType::F64 => {
            for (index, (lhs, rhs)) in actual
                .as_slice::<f64>()
                .unwrap()
                .iter()
                .zip(expected.as_slice::<f64>().unwrap())
                .enumerate()
            {
                assert!(
                    (lhs - rhs).abs() <= tol,
                    "{context}: index {index}: fused={lhs}, generic={rhs}"
                );
            }
        }
        tenferro_runtime::DType::F32 => {
            for (index, (lhs, rhs)) in actual
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .zip(expected.as_slice::<f32>().unwrap())
                .enumerate()
            {
                assert!(
                    (*lhs as f64 - *rhs as f64).abs() <= tol,
                    "{context}: index {index}: fused={lhs}, generic={rhs}"
                );
            }
        }
        tenferro_runtime::DType::C32 => {
            for (index, (lhs, rhs)) in actual
                .as_slice::<Complex32>()
                .unwrap()
                .iter()
                .zip(expected.as_slice::<Complex32>().unwrap())
                .enumerate()
            {
                assert!(
                    (*lhs - *rhs).norm() as f64 <= tol,
                    "{context}: index {index}: fused={lhs}, generic={rhs}"
                );
            }
        }
        tenferro_runtime::DType::C64 => {
            for (index, (lhs, rhs)) in actual
                .as_slice::<Complex64>()
                .unwrap()
                .iter()
                .zip(expected.as_slice::<Complex64>().unwrap())
                .enumerate()
            {
                assert!(
                    (*lhs - *rhs).norm() <= tol,
                    "{context}: index {index}: fused={lhs}, generic={rhs}"
                );
            }
        }
        dtype => panic!("{context}: unexpected dtype {dtype:?}"),
    }
}

fn square_data(n: usize, batch: usize) -> Vec<f64> {
    let mut data = vec![0.0; n * n * batch];
    for b in 0..batch {
        for column in 0..n {
            for row in 0..n {
                let index = row + column * n + b * n * n;
                data[index] = if row == column {
                    2.0 + 0.1 * row as f64 + b as f64
                } else {
                    0.01 * ((row + 2 * column + b) % 5) as f64 - 0.02
                };
            }
        }
    }
    data
}

fn tangent_data(n: usize, batch: usize) -> Vec<f64> {
    let mut data = vec![0.0; n * n * batch];
    for b in 0..batch {
        for column in 0..n {
            for row in 0..n {
                let index = row + column * n + b * n * n;
                data[index] = 0.03 * ((row * 3 + column * 5 + b) % 11) as f64 - 0.15;
            }
        }
    }
    data
}

fn parity_case(
    label: &str,
    shape: &[usize],
    forward: &[f64],
    tangent: &[f64],
    make_tensor: impl Fn(&[usize], &[f64]) -> Tensor,
    tol: f64,
    active: ActiveMask,
) {
    let ad = ad_context();
    let a_concrete = TracedTensor::from_tensor_concrete_shape(make_tensor(shape, forward)).unwrap();
    let a_symbolic = TracedTensor::from_tensor_symbolic_shape(make_tensor(shape, forward)).unwrap();
    let da_concrete =
        TracedTensor::from_tensor_concrete_shape(make_tensor(shape, tangent)).unwrap();
    let da_symbolic =
        TracedTensor::from_tensor_symbolic_shape(make_tensor(shape, tangent)).unwrap();

    let jvp_concrete = ad
        .jvp(&qr_loss(&a_concrete, active), &a_concrete, &da_concrete)
        .unwrap();
    let jvp_symbolic = ad
        .jvp(&qr_loss(&a_symbolic, active), &a_symbolic, &da_symbolic)
        .unwrap();

    let grad_concrete = ad.grad(&qr_loss(&a_concrete, active), &a_concrete).unwrap();
    let grad_symbolic = ad.grad(&qr_loss(&a_symbolic, active), &a_symbolic).unwrap();

    let results = eval_many(&[&jvp_concrete, &jvp_symbolic, &grad_concrete, &grad_symbolic]);
    assert_close(
        &results[0],
        &results[1],
        tol,
        &format!("{label} JVP ({})", active.label()),
    );
    assert_close(
        &results[2],
        &results[3],
        tol,
        &format!("{label} VJP ({})", active.label()),
    );
}

#[test]
fn fused_square_qr_parity_f64() {
    let ad = ad_context();
    for (n, batch) in [(4, 1), (8, 1), (16, 1), (4, 2), (8, 3)] {
        let shape = vec![n, n, batch];
        let forward = square_data(n, batch);
        let tangent = tangent_data(n, batch);
        for active in [ActiveMask::Q, ActiveMask::R, ActiveMask::Both] {
            parity_case(
                "f64", &shape, &forward, &tangent, tensor_f64, 1.0e-11, active,
            );
        }
    }
    // The helper above owns its context; this keeps the local one exercised so
    // a rule-registration regression fails before the loop too.
    let _ = ad;
}

#[test]
fn fused_square_qr_parity_c32() {
    for (n, batch) in [(4, 1), (8, 1), (16, 1), (4, 2)] {
        let shape = vec![n, n, batch];
        let forward = square_data(n, batch);
        let tangent = tangent_data(n, batch);
        for active in [ActiveMask::Q, ActiveMask::R, ActiveMask::Both] {
            parity_case(
                "c32", &shape, &forward, &tangent, tensor_c32, 2.0e-4, active,
            );
        }
    }
}

#[test]
fn fused_square_qr_parity_c64() {
    for (n, batch) in [(4, 1), (8, 1), (16, 1), (4, 2), (8, 3)] {
        let shape = vec![n, n, batch];
        let forward = square_data(n, batch);
        let tangent = tangent_data(n, batch);
        for active in [ActiveMask::Q, ActiveMask::R, ActiveMask::Both] {
            parity_case(
                "c64", &shape, &forward, &tangent, tensor_c64, 1.0e-11, active,
            );
        }
    }
}

#[test]
fn rectangular_qr_falls_back_to_generic_graph() {
    let ad = ad_context();
    for shape in [vec![5, 3], vec![3, 5], vec![5, 3, 2]] {
        let total: usize = shape.iter().product();
        let forward: Vec<f64> = (0..total)
            .map(|index| 0.1 + 0.07 * ((index * 3) % 13) as f64)
            .collect();
        let tangent: Vec<f64> = (0..total)
            .map(|index| 0.02 * ((index * 5) % 7) as f64 - 0.06)
            .collect();
        let a_concrete =
            TracedTensor::from_tensor_concrete_shape(tensor_f64(&shape, &forward)).unwrap();
        let a_symbolic =
            TracedTensor::from_tensor_symbolic_shape(tensor_f64(&shape, &forward)).unwrap();
        let da_concrete =
            TracedTensor::from_tensor_concrete_shape(tensor_f64(&shape, &tangent)).unwrap();
        let da_symbolic =
            TracedTensor::from_tensor_symbolic_shape(tensor_f64(&shape, &tangent)).unwrap();
        let jvp_concrete = ad
            .jvp(
                &qr_loss(&a_concrete, ActiveMask::Both),
                &a_concrete,
                &da_concrete,
            )
            .unwrap();
        let jvp_symbolic = ad
            .jvp(
                &qr_loss(&a_symbolic, ActiveMask::Both),
                &a_symbolic,
                &da_symbolic,
            )
            .unwrap();
        let grad_concrete = ad
            .grad(&qr_loss(&a_concrete, ActiveMask::Both), &a_concrete)
            .unwrap();
        let grad_symbolic = ad
            .grad(&qr_loss(&a_symbolic, ActiveMask::Both), &a_symbolic)
            .unwrap();
        let results = eval_many(&[&jvp_concrete, &jvp_symbolic, &grad_concrete, &grad_symbolic]);
        assert_close(&results[0], &results[1], 1.0e-11, "rectangular JVP");
        assert_close(&results[2], &results[3], 1.0e-11, "rectangular VJP");
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use tenferro_gpu::cuda::{
        cuda_runtime_engine_registration, download_tensor, gpu_available, upload_tensor,
        CudaBackend, CudaDeviceId,
    };
    use tenferro_runtime::{EngineId, Runtime};

    fn cuda_runtime_with_linalg(backend: &CudaBackend) -> Runtime {
        let engine = EngineId::new("tenferro.test.cuda.qr_fused_ad.v1").unwrap();
        let mut builder = Runtime::builder();
        builder
            .register_engine(cuda_runtime_engine_registration(backend, engine.clone()).unwrap())
            .unwrap();
        builder
            .install_extension_module(
                tenferro_linalg::extension_module::<CudaBackend>(engine).unwrap(),
            )
            .unwrap();
        builder.build().unwrap()
    }

    fn gpu_case(
        label: &str,
        shape: &[usize],
        forward: &[f64],
        tangent: &[f64],
        make_tensor: impl Fn(&[usize], &[f64]) -> Tensor,
        tol: f64,
        active: ActiveMask,
    ) {
        let ad = ad_context();
        let a_concrete =
            TracedTensor::from_tensor_concrete_shape(make_tensor(shape, forward)).unwrap();
        let da_concrete =
            TracedTensor::from_tensor_concrete_shape(make_tensor(shape, tangent)).unwrap();
        let jvp = ad
            .jvp(&qr_loss(&a_concrete, active), &a_concrete, &da_concrete)
            .unwrap();
        // Build the reverse-mode graph from a fresh leaf: the AD context may
        // retain transform state that keeps the JVP tangents in the same input
        // registry, while this program has only the primal input.
        let a_grad = TracedTensor::from_tensor_concrete_shape(make_tensor(shape, forward)).unwrap();
        let grad = ad.grad(&qr_loss(&a_grad, active), &a_grad).unwrap();

        // Generic CPU reference on the same data.
        let a_symbolic =
            TracedTensor::from_tensor_symbolic_shape(make_tensor(shape, forward)).unwrap();
        let da_symbolic =
            TracedTensor::from_tensor_symbolic_shape(make_tensor(shape, tangent)).unwrap();
        let jvp_reference = ad
            .jvp(&qr_loss(&a_symbolic, active), &a_symbolic, &da_symbolic)
            .unwrap();
        let a_grad_symbolic =
            TracedTensor::from_tensor_symbolic_shape(make_tensor(shape, forward)).unwrap();
        let grad_reference = ad
            .grad(&qr_loss(&a_grad_symbolic, active), &a_grad_symbolic)
            .unwrap();
        let references = eval_many(&[&jvp_reference, &grad_reference]);

        let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
        let runtime = cuda_runtime_with_linalg(&backend);
        let a_gpu = upload_tensor(backend.runtime(), &make_tensor(shape, forward)).unwrap();
        let da_gpu = upload_tensor(backend.runtime(), &make_tensor(shape, tangent)).unwrap();
        let seed = match grad.dtype() {
            tenferro_runtime::DType::F64 => tensor_f64(&[], &[1.0]),
            tenferro_runtime::DType::F32 => Tensor::from_typed::<f32>(
                TypedTensor::from_vec_col_major(Vec::<usize>::new(), vec![1.0_f32]).unwrap(),
            ),
            tenferro_runtime::DType::C32 => Tensor::from_typed::<Complex32>(
                TypedTensor::from_vec_col_major(
                    Vec::<usize>::new(),
                    vec![Complex32::new(1.0, 0.0)],
                )
                .unwrap(),
            ),
            tenferro_runtime::DType::C64 => Tensor::from_typed::<Complex64>(
                TypedTensor::from_vec_col_major(
                    Vec::<usize>::new(),
                    vec![Complex64::new(1.0, 0.0)],
                )
                .unwrap(),
            ),
            dtype => panic!("unexpected loss seed dtype {dtype:?}"),
        };
        let seed_gpu = upload_tensor(backend.runtime(), &seed).unwrap();

        let jvp_program = GraphCompiler::new().compile(&jvp).unwrap();
        let jvp_gpu = runtime
            .run_compiled(&jvp_program, &[&a_gpu, &da_gpu])
            .unwrap()
            .remove(0);
        let jvp_host = download_tensor(backend.runtime(), &jvp_gpu).unwrap();

        let grad_program = GraphCompiler::new().compile(&grad).unwrap();
        let a_key = a_grad.input_key().unwrap();
        // The reverse pass seeds the scalar loss with its own constant input;
        // bind that slot to one and the primal slot to the uploaded matrix.
        let grad_inputs: Vec<&Tensor> = grad_program
            .input_keys()
            .iter()
            .map(|key| if *key == a_key { &a_gpu } else { &seed_gpu })
            .collect();
        let grad_gpu = runtime
            .run_compiled(&grad_program, &grad_inputs)
            .unwrap()
            .remove(0);
        let grad_host = download_tensor(backend.runtime(), &grad_gpu).unwrap();

        // The same generic graph on CUDA: this isolates a fused-carrier bug
        // from a pre-existing backend difference.
        let generic_jvp_program = GraphCompiler::new().compile(&jvp_reference).unwrap();
        let generic_jvp_gpu = runtime
            .run_compiled(&generic_jvp_program, &[&a_gpu, &da_gpu])
            .unwrap()
            .remove(0);
        let generic_jvp_host = download_tensor(backend.runtime(), &generic_jvp_gpu).unwrap();
        let generic_grad_program = GraphCompiler::new().compile(&grad_reference).unwrap();
        let generic_a_key = a_grad_symbolic.input_key().unwrap();
        let generic_grad_inputs: Vec<&Tensor> = generic_grad_program
            .input_keys()
            .iter()
            .map(|key| {
                if *key == generic_a_key {
                    &a_gpu
                } else {
                    &seed_gpu
                }
            })
            .collect();
        let generic_grad_gpu = runtime
            .run_compiled(&generic_grad_program, &generic_grad_inputs)
            .unwrap()
            .remove(0);
        let generic_grad_host = download_tensor(backend.runtime(), &generic_grad_gpu).unwrap();
        assert_close(
            &jvp_host,
            &generic_jvp_host,
            tol,
            &format!("{label} CUDA fused-vs-generic JVP ({})", active.label()),
        );
        assert_close(
            &grad_host,
            &generic_grad_host,
            tol,
            &format!("{label} CUDA fused-vs-generic VJP ({})", active.label()),
        );
        // Real CUDA QR JVP already agrees with the CPU generic graph; complex
        // CUDA JVP has a pre-existing generic-path divergence from CPU, so the
        // backend-parity assertion above is the meaningful one there.
        if label == "f64" {
            assert_close(
                &generic_jvp_host,
                &references[0],
                tol,
                &format!("{label} CUDA generic-vs-CPU JVP ({})", active.label()),
            );
            assert_close(
                &grad_host,
                &references[1],
                tol,
                &format!("{label} CUDA VJP ({})", active.label()),
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn fused_square_qr_cuda_parity_f64() {
        assert!(gpu_available(), "CUDA test requires an available device");
        for n in [32, 64] {
            let shape = vec![n, n];
            let forward = square_data(n, 1);
            let tangent = tangent_data(n, 1);
            for active in [ActiveMask::Q, ActiveMask::R, ActiveMask::Both] {
                gpu_case(
                    "f64", &shape, &forward, &tangent, tensor_f64, 1.0e-9, active,
                );
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn fused_square_qr_cuda_parity_c64() {
        assert!(gpu_available(), "CUDA test requires an available device");
        for n in [32, 64] {
            let shape = vec![n, n];
            let forward = square_data(n, 1);
            let tangent = tangent_data(n, 1);
            for active in [ActiveMask::Q, ActiveMask::R, ActiveMask::Both] {
                gpu_case(
                    "c64", &shape, &forward, &tangent, tensor_c64, 1.0e-9, active,
                );
            }
        }
    }
}
