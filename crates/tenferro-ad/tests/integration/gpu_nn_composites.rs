#![cfg(feature = "cuda")]

//! CPU/CUDA parity of the composite operations and `erf` (#2010 PR-B1):
//! values and gradients on the eager, traced and concrete-session surfaces,
//! including the softmax edge-case policy.
//!
//! Run with: cargo test --features cuda -- --ignored

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_gpu::cuda::{
    cuda_runtime_engine_registration, download_tensor, gpu_available, upload_tensor, CudaBackend,
    CudaDeviceId,
};
use tenferro_runtime::{
    DType, EngineId, GraphCompiler, Runtime, Tensor, TensorRead, TensorSessionOpsExt, TracedTensor,
};
use tenferro_tensor::BackendSessionHost;

fn cuda_backend() -> CudaBackend {
    CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap()
}

fn download(ctx: &Arc<EagerRuntime>, tensor: &Tensor) -> Tensor {
    ctx.with_execution_session(|session| session.download_to_host(TensorRead::from_tensor(tensor)))
        .unwrap()
        .unwrap()
}

fn host_f64(tensor: &Tensor) -> Vec<f64> {
    match tensor.dtype() {
        DType::F64 => tensor.as_slice::<f64>().unwrap().to_vec(),
        DType::F32 => tensor
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// Elementwise parity: NaN/inf classes match exactly; finite values within
/// `rtol` (relative to `max(|cpu|, scale)`).
fn assert_parity(cuda: &[f64], cpu: &[f64], rtol: f64, scale: f64, what: &str) {
    assert_eq!(cuda.len(), cpu.len(), "{what}: length");
    for (index, (&g, &c)) in cuda.iter().zip(cpu).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "{what}[{index}]: cpu NaN, cuda {g}");
        } else if c.is_infinite() {
            assert_eq!(g, c, "{what}[{index}]");
        } else {
            let tol = rtol * c.abs().max(scale);
            assert!(
                (g - c).abs() <= tol,
                "{what}[{index}]: cuda {g}, cpu {c}, tol {tol}"
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Erf,
    Sigmoid,
    Silu,
    Softplus,
    Gelu,
    GeluTanh,
    ReduceMean,
    Softmax,
    LogSoftmax,
    MaskedSoftmax,
    MaskedLogSoftmax,
    LayerNorm,
    RmsNorm,
    TakeRows,
    TakeColumns,
}

const OPS: [Op; 15] = [
    Op::Erf,
    Op::Sigmoid,
    Op::Silu,
    Op::Softplus,
    Op::Gelu,
    Op::GeluTanh,
    Op::ReduceMean,
    Op::Softmax,
    Op::LogSoftmax,
    Op::MaskedSoftmax,
    Op::MaskedLogSoftmax,
    Op::LayerNorm,
    Op::RmsNorm,
    Op::TakeRows,
    Op::TakeColumns,
];

/// Shared inputs: x is [4, 3, 2] (rows, columns, batch).
struct Inputs {
    x: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor,
    rows: Tensor,
    columns: Tensor,
}

fn x_values() -> Vec<f64> {
    vec![
        -1000.0, -20.0, -1.5, -0.25, 0.0, 0.3, 1.0, 2.5, 20.0, 1000.0, -0.0, 0.75, //
        3.0, -3.0, 0.125, 7.5, -7.5, 1e-3, -1e-3, 4.0, -4.0, 0.5, 0.5, 0.5,
    ]
}

fn inputs(dtype: DType) -> Inputs {
    let x = x_values();
    let x = match dtype {
        DType::F64 => Tensor::from_vec_col_major(vec![4, 3, 2], x).unwrap(),
        _ => Tensor::from_vec_col_major(
            vec![4, 3, 2],
            x.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        )
        .unwrap(),
    };
    let real = |values: Vec<f64>| match dtype {
        DType::F64 => Tensor::from_vec_col_major(vec![values.len()], values).unwrap(),
        _ => Tensor::from_vec_col_major(
            vec![values.len()],
            values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        )
        .unwrap(),
    };
    // Column 1 of batch 0 is fully masked; the rest is partially masked.
    let mut mask = vec![true; 24];
    for row in 0..4 {
        mask[row + 4] = false;
    }
    mask[1] = false;
    mask[14] = false;
    Inputs {
        x,
        mask: Tensor::from_vec_col_major(vec![4, 3, 2], mask).unwrap(),
        weight: real(vec![1.5, -0.5, 2.0, 0.25]),
        bias: real(vec![0.1, 0.2, -0.3, 0.4]),
        rows: Tensor::from_vec_col_major(vec![3, 1, 2], vec![3_i64, 0, 0, 1, 1, 2]).unwrap(),
        columns: Tensor::from_vec_col_major(vec![1, 4, 2], vec![2_i32, 0, 1, 1, 0, 0, 2, 1])
            .unwrap(),
    }
}

fn eager_op(
    s: &mut EagerSession<'_>,
    op: Op,
    x: &EagerTensor,
    host: &Inputs,
) -> tenferro_ad::Result<EagerTensor> {
    match op {
        Op::Erf => s.erf(x),
        Op::Sigmoid => s.sigmoid(x),
        Op::Silu => s.silu(x),
        Op::Softplus => s.softplus(x),
        Op::Gelu => s.gelu(x),
        Op::GeluTanh => s.gelu_tanh(x),
        Op::ReduceMean => s.reduce_mean(x, Some(&[0, 2])),
        Op::Softmax => s.softmax(x, 0),
        Op::LogSoftmax => s.log_softmax(x, 1),
        Op::MaskedSoftmax | Op::MaskedLogSoftmax => {
            let mask = s.constant_from_host(host.mask.duplicate()?)?;
            if matches!(op, Op::MaskedSoftmax) {
                s.masked_softmax(x, &mask, 0)
            } else {
                s.masked_log_softmax(x, &mask, 0)
            }
        }
        Op::LayerNorm | Op::RmsNorm => {
            let w = s.constant_from_host(host.weight.duplicate()?)?;
            let b = s.constant_from_host(host.bias.duplicate()?)?;
            if matches!(op, Op::LayerNorm) {
                s.layer_norm(x, 0, Some(&w), Some(&b), 1e-5)
            } else {
                s.rms_norm(x, 0, Some(&w), Some(&b), 1e-5)
            }
        }
        Op::TakeRows => {
            let idx = s.constant_from_host(host.rows.duplicate()?)?;
            s.take_along_axis(x, &idx, 0)
        }
        Op::TakeColumns => {
            let idx = s.constant_from_host(host.columns.duplicate()?)?;
            s.take_along_axis(x, &idx, 1)
        }
    }
}

/// Run `op` eagerly on `ctx`; return the value and the gradient of
/// `sum(finite(y) * w)` with respect to x (non-finite outputs get weight 0).
fn eager_value_and_grad(ctx: &Arc<EagerRuntime>, op: Op, host: &Inputs) -> (Vec<f64>, Vec<f64>) {
    let x = ctx
        .with_eager_session(|s| s.constant_from_host(host.x.duplicate()?))
        .unwrap();
    let x = EagerTensor::requires_grad_in(x.to_tensor().unwrap(), ctx.clone()).unwrap();
    let y = ctx
        .with_eager_session(|s| eager_op(s, op, &x, host))
        .unwrap_or_else(|err| panic!("{op:?} {:?}: {err:?}", host.x.dtype()));
    let value = host_f64(&download(ctx, &y.to_tensor().unwrap()));
    let weights: Vec<f64> = value
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if v.is_finite() {
                0.5 + (i % 5) as f64 * 0.25
            } else {
                0.0
            }
        })
        .collect();
    let shape = y.shape().to_vec();
    let cotangent = match y.dtype() {
        DType::F64 => Tensor::from_vec_col_major(shape, weights).unwrap(),
        _ => {
            Tensor::from_vec_col_major(shape, weights.iter().map(|&v| v as f32).collect::<Vec<_>>())
                .unwrap()
        }
    };
    let cotangent = ctx
        .with_eager_session(|s| s.constant_from_host(cotangent))
        .unwrap();
    let grad = ctx.vjp(&y, &x, &cotangent).unwrap();
    let grad = host_f64(&download(ctx, &grad.to_tensor().unwrap()));
    (value, grad)
}

/// Absolute floor of the parity tolerance. `x/2 * (1 + erf(x/sqrt(2)))` and
/// similar forms cancel in the negative tail, so an F32 ulp difference of
/// `erf` near -1 is a relative difference of ~1e-5 in a small result; the
/// inputs are O(1), so F32 parity is checked against an O(1) scale.
fn scale(dtype: DType) -> f64 {
    if dtype == DType::F64 {
        1e-6
    } else {
        1.0
    }
}

fn tolerance(dtype: DType) -> f64 {
    if dtype == DType::F64 {
        1e-12
    } else {
        2e-5
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_gpu_composites_eager_values_and_gradients_match_cpu() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let cpu = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let gpu = EagerRuntime::with_cuda_backend(cuda_backend()).unwrap();
    for dtype in [DType::F64, DType::F32] {
        let host = inputs(dtype);
        for op in OPS {
            let (cpu_value, cpu_grad) = eager_value_and_grad(&cpu, op, &host);
            let (gpu_value, gpu_grad) = eager_value_and_grad(&gpu, op, &host);
            let tol = tolerance(dtype);
            assert_parity(
                &gpu_value,
                &cpu_value,
                tol,
                scale(dtype),
                &format!("{op:?} {dtype:?} value"),
            );
            assert_parity(
                &gpu_grad,
                &cpu_grad,
                tol,
                scale(dtype),
                &format!("{op:?} {dtype:?} grad"),
            );
            assert!(
                gpu_grad.iter().all(|g| g.is_finite()),
                "{op:?} {dtype:?}: {gpu_grad:?}"
            );
        }
        // The fully masked column (batch 0, column 1) is 0 / -inf with a zero gradient.
        for op in [Op::MaskedSoftmax, Op::MaskedLogSoftmax] {
            let (value, grad) = eager_value_and_grad(&gpu, op, &host);
            let fill = if matches!(op, Op::MaskedSoftmax) {
                0.0
            } else {
                f64::NEG_INFINITY
            };
            assert_eq!(&value[4..8], &[fill; 4], "{op:?} {dtype:?}");
            assert_eq!(&grad[4..8], &[0.0; 4], "{op:?} {dtype:?}");
        }
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_gpu_softmax_nan_and_infinity_policy_matches_cpu() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let cpu = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let gpu = EagerRuntime::with_cuda_backend(cuda_backend()).unwrap();
    // Columns: [NaN, 1, 2], [1, +inf, 3], [-inf, -inf, -inf], [0, -inf, 1].
    let x = Tensor::from_vec_col_major(
        vec![3, 4],
        vec![
            f64::NAN,
            1.0,
            2.0,
            1.0,
            f64::INFINITY,
            3.0,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            0.0,
            f64::NEG_INFINITY,
            1.0,
        ],
    )
    .unwrap();
    for log in [false, true] {
        let run = |ctx: &Arc<EagerRuntime>| {
            let y = ctx
                .with_eager_session(|s| {
                    let x = s.constant_from_host(x.duplicate()?)?;
                    if log {
                        s.log_softmax(&x, 0)
                    } else {
                        s.softmax(&x, 0)
                    }
                })
                .unwrap();
            host_f64(&download(ctx, &y.to_tensor().unwrap()))
        };
        let cpu_value = run(&cpu);
        let gpu_value = run(&gpu);
        assert_parity(&gpu_value, &cpu_value, 1e-13, 1e-6, &format!("log={log}"));
        assert!(cpu_value[..6].iter().all(|v| v.is_nan()));
        let fill = if log { f64::NEG_INFINITY } else { 0.0 };
        assert_eq!(&gpu_value[6..9], &[fill; 3]);
    }
}

fn cuda_runtime(backend: &CudaBackend) -> Runtime {
    let engine_id = EngineId::new("tenferro-ad.nn-composites.cuda.v1").unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(cuda_runtime_engine_registration(backend, engine_id).unwrap())
        .unwrap();
    builder.build().unwrap()
}

fn cpu_runtime() -> Runtime {
    crate::support::cpu_runtime()
}

/// Build `op` over placeholder inputs; returns the output and its placeholders.
fn traced_op(op: Op, host: &Inputs) -> (TracedTensor, Vec<(TracedTensor, Tensor)>) {
    let x = TracedTensor::input_concrete_shape(host.x.dtype(), host.x.shape()).unwrap();
    let mut bindings = vec![(x.clone(), host.x.duplicate().unwrap())];
    let mut placeholder = |tensor: &Tensor| {
        let p = TracedTensor::input_concrete_shape(tensor.dtype(), tensor.shape()).unwrap();
        bindings.push((p.clone(), tensor.duplicate().unwrap()));
        p
    };
    let y = match op {
        Op::Erf => x.erf(),
        Op::Sigmoid => x.sigmoid(),
        Op::Silu => x.silu(),
        Op::Softplus => x.softplus(),
        Op::Gelu => x.gelu(),
        Op::GeluTanh => x.gelu_tanh(),
        Op::ReduceMean => x.reduce_mean(Some(&[0, 2])),
        Op::Softmax => x.softmax(0),
        Op::LogSoftmax => x.log_softmax(1),
        Op::MaskedSoftmax => x.masked_softmax(&placeholder(&host.mask), 0),
        Op::MaskedLogSoftmax => x.masked_log_softmax(&placeholder(&host.mask), 0),
        Op::LayerNorm => {
            let w = placeholder(&host.weight);
            let b = placeholder(&host.bias);
            x.layer_norm(0, Some(&w), Some(&b), 1e-5)
        }
        Op::RmsNorm => {
            let w = placeholder(&host.weight);
            let b = placeholder(&host.bias);
            x.rms_norm(0, Some(&w), Some(&b), 1e-5)
        }
        Op::TakeRows => x.take_along_axis(&placeholder(&host.rows), 0),
        Op::TakeColumns => x.take_along_axis(&placeholder(&host.columns), 1),
    }
    .unwrap();
    (y, bindings)
}

fn run_traced(runtime: &Runtime, y: &TracedTensor, bindings: &[(TracedTensor, Tensor)]) -> Tensor {
    let specs: Vec<(&TracedTensor, DType, &[usize])> = bindings
        .iter()
        .map(|(p, t)| (p, t.dtype(), t.shape()))
        .collect();
    let program = GraphCompiler::new()
        .compile_with_input_specs(y, &specs)
        .unwrap();
    let inputs: Vec<&Tensor> = bindings.iter().map(|(_, t)| t).collect();
    let mut outputs = runtime.run_compiled(&program, &inputs).unwrap();
    outputs.pop().unwrap()
}

/// The traced surface on CUDA, including fused elementwise regions that
/// contain `erf` (CubeCL `Arithmetic::Erf`), agrees with the CPU runtime.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_gpu_composites_traced_values_match_cpu() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let backend = cuda_backend();
    let gpu = cuda_runtime(&backend);
    let cpu = cpu_runtime();
    for dtype in [DType::F64, DType::F32] {
        let host = inputs(dtype);
        for op in OPS {
            let (y, bindings) = traced_op(op, &host);
            let cpu_value = host_f64(&run_traced(&cpu, &y, &bindings));
            let device_bindings: Vec<(TracedTensor, Tensor)> = bindings
                .iter()
                .map(|(p, t)| (p.clone(), upload_tensor(backend.runtime(), t).unwrap()))
                .collect();
            let out = run_traced(&gpu, &y, &device_bindings);
            let gpu_value = host_f64(&download_tensor(backend.runtime(), &out).unwrap());
            assert_parity(
                &gpu_value,
                &cpu_value,
                tolerance(dtype),
                scale(dtype),
                &format!("traced {op:?} {dtype:?}"),
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_gpu_composites_concrete_session_matches_cpu() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = cuda_backend();
    let mut cpu = CpuBackend::new();
    let host = inputs(DType::F64);
    let device = |t: &Tensor| upload_tensor(gpu.runtime(), t).unwrap();
    let (x, mask, w, b, rows) = (
        device(&host.x),
        device(&host.mask),
        device(&host.weight),
        device(&host.bias),
        device(&host.rows),
    );
    type SessionFn<'a> = Box<
        dyn Fn(
                &Tensor,
                &Tensor,
                &Tensor,
                &Tensor,
                &Tensor,
                &mut dyn tenferro_tensor::Engine,
            ) -> tenferro_tensor::Result<Tensor>
            + Send
            + Sync
            + 'a,
    >;
    let cases: Vec<(&str, SessionFn<'_>)> = vec![
        ("erf", Box::new(|x, _, _, _, _, s| x.erf(s))),
        ("sigmoid", Box::new(|x, _, _, _, _, s| x.sigmoid(s))),
        ("softplus", Box::new(|x, _, _, _, _, s| x.softplus(s))),
        ("gelu", Box::new(|x, _, _, _, _, s| x.gelu(s))),
        (
            "reduce_mean",
            Box::new(|x, _, _, _, _, s| x.reduce_mean(None, s)),
        ),
        (
            "masked_log_softmax",
            Box::new(|x, m, _, _, _, s| x.masked_log_softmax(m, 0, s)),
        ),
        (
            "layer_norm",
            Box::new(|x, _, w, b, _, s| x.layer_norm(0, Some(w), Some(b), 1e-5, s)),
        ),
        (
            "rms_norm",
            Box::new(|x, _, w, _, _, s| x.rms_norm(0, Some(w), None, 1e-5, s)),
        ),
        (
            "take_along_axis",
            Box::new(|x, _, _, _, i, s| x.take_along_axis(i, 0, s)),
        ),
    ];
    for (name, f) in cases {
        let expected = cpu
            .with_backend_session(|s| {
                f(&host.x, &host.mask, &host.weight, &host.bias, &host.rows, s)
            })
            .unwrap()
            .unwrap();
        let out = gpu
            .with_backend_session(|s| f(&x, &mask, &w, &b, &rows, s))
            .unwrap()
            .unwrap();
        let out = download_tensor(gpu.runtime(), &out).unwrap();
        assert_parity(&host_f64(&out), &host_f64(&expected), 1e-12, 1e-6, name);
    }
}
