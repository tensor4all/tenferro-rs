//! End-to-end traced-AD check modeled on a physics-informed neural network for
//! the Korteweg–de Vries equation `u_t + 6 u u_x + u_xxx = 0`.
//!
//! One `TraceContext` holds the network parameters and the sample points. The
//! PDE residual comes from chained `jvp` calls, the scalar loss is
//! differentiated with `grad` with respect to every parameter, and each graph
//! is compiled with input specs and executed on the CPU engine. The full
//! training sample lives in tensor4all/tenferro-examples (`examples/kdv-pinn`).

use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{
    CompiledGraph, DType, DotGeneralConfig, GraphCompiler, Runtime, Tensor, TraceContext,
    TracedTensor,
};

use crate::support::{cpu_runtime, run_compiled_one};

const LAYER_SIZES: [usize; 4] = [2, 8, 8, 1];
const N_COL: usize = 256;
const N_IC: usize = 64;
const N_BC: usize = 64;

/// Exact single-soliton solution `u(x, t) = 2 sech^2(x - 4t)`.
fn soliton(x: f64, t: f64) -> f64 {
    2.0 / (x - 4.0 * t).cosh().powi(2)
}

/// Deterministic values in `[lo, hi)` from the additive recurrence with the
/// plastic-number step, so the test needs no random-number dependency.
fn spread(n: usize, offset: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n)
        .map(|k| lo + (hi - lo) * (((k + offset) as f64) * 0.754_877_666_246_692_7).fract())
        .collect()
}

fn column(data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(vec![data.len(), 1], data).unwrap()
}

struct Linear {
    weight: TracedTensor,
    bias: TracedTensor,
    out_features: usize,
}

impl Linear {
    fn new(trace: &TraceContext, in_features: usize, out_features: usize) -> Self {
        Self {
            weight: trace
                .input_concrete_shape(DType::F64, &[in_features, out_features])
                .unwrap(),
            bias: trace
                .input_concrete_shape(DType::F64, &[out_features])
                .unwrap(),
            out_features,
        }
    }

    fn forward(&self, x: &TracedTensor) -> TracedTensor {
        let y = x
            .dot_general(
                &self.weight,
                DotGeneralConfig {
                    lhs_contracting_dims: [1].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            )
            .unwrap();
        let bias = self.bias.reshape([1, self.out_features]).unwrap();
        y.add(&bias).unwrap()
    }
}

/// Fully connected network with `tanh` between layers. Every layer's
/// parameters are placeholders of the caller's trace, so the forward pass
/// combines them with the caller's inputs in a single graph.
struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(trace: &TraceContext, sizes: &[usize]) -> Self {
        let layers = sizes
            .windows(2)
            .map(|pair| Linear::new(trace, pair[0], pair[1]))
            .collect();
        Self { layers }
    }

    fn forward(&self, x: &TracedTensor) -> TracedTensor {
        let last = self.layers.len() - 1;
        let mut y = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            y = layer.forward(&y);
            if i < last {
                y = y.tanh().unwrap();
            }
        }
        y
    }

    fn parameters(&self) -> Vec<&TracedTensor> {
        self.layers
            .iter()
            .flat_map(|layer| [&layer.weight, &layer.bias])
            .collect()
    }

    /// Xavier-scaled deterministic weights and small nonzero biases.
    fn init_tensors(&self) -> Vec<Tensor> {
        let mut tensors = Vec::new();
        for (l, layer) in self.layers.iter().enumerate() {
            let shape = layer.weight.try_concrete_shape().unwrap();
            let scale = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
            let len = shape.iter().product();
            let weight = spread(len, 17 * l, -scale, scale);
            tensors.push(Tensor::from_vec_col_major(shape, weight).unwrap());
            let bias = spread(layer.out_features, 31 * l + 5, -0.1, 0.1);
            tensors.push(Tensor::from_vec_col_major(vec![layer.out_features], bias).unwrap());
        }
        tensors
    }
}

/// `u_t + 6 u u_x + u_xxx`, with every derivative taken by forward-mode AD.
fn kdv_residual(
    trace: &TraceContext,
    u: &TracedTensor,
    x: &TracedTensor,
    t: &TracedTensor,
) -> TracedTensor {
    let n = x.try_concrete_shape().unwrap()[0];
    let ones = trace
        .from_vec_col_major(vec![n, 1], vec![1.0_f64; n])
        .unwrap();
    let u_t = u.jvp(t, &ones).unwrap();
    let u_x = u.jvp(x, &ones).unwrap();
    let u_xx = u_x.jvp(x, &ones).unwrap();
    let u_xxx = u_xx.jvp(x, &ones).unwrap();
    let nonlinear = u.mul(&u_x).unwrap().scale_real(6.0).unwrap();
    u_t.add(&nonlinear).unwrap().add(&u_xxx).unwrap()
}

/// `sum(a^2) / n` for an `[n, 1]` tensor, as a scalar.
fn mean_square(a: &TracedTensor, n: usize) -> TracedTensor {
    a.mul(a)
        .unwrap()
        .reduce_sum(Some(&[0, 1]))
        .unwrap()
        .scale_real(1.0 / n as f64)
        .unwrap()
}

/// Stack two `[n, 1]` columns into an `[n, 2]` network input.
fn xt_input(x: &TracedTensor, t: &TracedTensor) -> TracedTensor {
    let n = x.try_concrete_shape().unwrap()[0];
    TracedTensor::stack(&[x, t], 1)
        .unwrap()
        .reshape([n, 2])
        .unwrap()
}

fn scalar(runtime: &Runtime, program: &CompiledGraph, inputs: &[&Tensor]) -> f64 {
    let out = run_compiled_one(runtime, program, inputs).unwrap();
    assert!(out.shape().is_empty(), "loss must be a scalar");
    out.as_slice::<f64>().unwrap()[0]
}

#[test]
fn kdv_residual_vanishes_on_exact_soliton() {
    let trace = TraceContext::new();
    let n = 5;
    let x = trace.input_concrete_shape(DType::F64, &[n, 1]).unwrap();
    let t = trace.input_concrete_shape(DType::F64, &[n, 1]).unwrap();
    // u = 2 sech^2(z) = 8 / (e^z + e^-z)^2 with z = x - 4t.
    let z = x.add(&t.scale_real(-4.0).unwrap()).unwrap();
    let cosh2 = z
        .exp()
        .unwrap()
        .add(&z.neg().unwrap().exp().unwrap())
        .unwrap();
    let eight = trace
        .from_vec_col_major(vec![n, 1], vec![8.0_f64; n])
        .unwrap();
    let u = eight.div(&cosh2.mul(&cosh2).unwrap()).unwrap();
    let residual = kdv_residual(&trace, &u, &x, &t);

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &residual,
            &[(&x, DType::F64, &[n, 1]), (&t, DType::F64, &[n, 1])],
        )
        .unwrap();
    let xs = vec![-2.0, -0.5, 0.0, 1.5, 3.0];
    let ts = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let out = run_compiled_one(&cpu_runtime(), &program, &[&column(xs), &column(ts)]).unwrap();
    assert_eq!(out.shape(), &[n, 1]);
    // Each term of the residual is O(1) on these points, so cancellation to
    // rounding level checks every derivative order.
    let max_abs = out
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .fold(0.0_f64, |acc, r| acc.max(r.abs()));
    assert!(
        max_abs < 1e-10,
        "residual of the exact soliton should vanish: max |r| = {max_abs:e}"
    );
}

#[test]
fn kdv_pinn_loss_gradients_run_and_match_finite_differences() {
    let trace = TraceContext::new();
    let net = Mlp::new(&trace, &LAYER_SIZES);

    let x_col = trace.input_concrete_shape(DType::F64, &[N_COL, 1]).unwrap();
    let t_col = trace.input_concrete_shape(DType::F64, &[N_COL, 1]).unwrap();
    let x_ic = trace.input_concrete_shape(DType::F64, &[N_IC, 1]).unwrap();
    let u_ic_true = trace.input_concrete_shape(DType::F64, &[N_IC, 1]).unwrap();
    let x_bc = trace.input_concrete_shape(DType::F64, &[N_BC, 1]).unwrap();
    let t_bc = trace.input_concrete_shape(DType::F64, &[N_BC, 1]).unwrap();
    let u_bc_true = trace.input_concrete_shape(DType::F64, &[N_BC, 1]).unwrap();

    let u_col = net.forward(&xt_input(&x_col, &t_col));
    let residual = kdv_residual(&trace, &u_col, &x_col, &t_col);

    let t_zero = trace
        .from_vec_col_major(vec![N_IC, 1], vec![0.0_f64; N_IC])
        .unwrap();
    let u_ic = net.forward(&xt_input(&x_ic, &t_zero));
    let u_bc = net.forward(&xt_input(&x_bc, &t_bc));

    let loss = mean_square(&residual, N_COL)
        .add(&mean_square(&u_ic.sub(&u_ic_true).unwrap(), N_IC))
        .unwrap()
        .add(&mean_square(&u_bc.sub(&u_bc_true).unwrap(), N_BC))
        .unwrap();
    let params = net.parameters();
    let grads: Vec<TracedTensor> = params.iter().map(|p| loss.grad(p).unwrap()).collect();

    let param_shapes: Vec<Vec<usize>> = params
        .iter()
        .map(|p| p.try_concrete_shape().unwrap())
        .collect();
    let col: &[usize] = &[N_COL, 1];
    let ic: &[usize] = &[N_IC, 1];
    let bc: &[usize] = &[N_BC, 1];
    let specs: Vec<(&TracedTensor, DType, &[usize])> = params
        .iter()
        .zip(&param_shapes)
        .map(|(p, shape)| (*p, DType::F64, shape.as_slice()))
        .chain([
            (&x_col, DType::F64, col),
            (&t_col, DType::F64, col),
            (&x_ic, DType::F64, ic),
            (&u_ic_true, DType::F64, ic),
            (&x_bc, DType::F64, bc),
            (&t_bc, DType::F64, bc),
            (&u_bc_true, DType::F64, bc),
        ])
        .collect();

    let mut compiler = GraphCompiler::new();
    let loss_program = compiler.compile_with_input_specs(&loss, &specs).unwrap();
    let grad_programs: Vec<CompiledGraph> = grads
        .iter()
        .map(|g| compiler.compile_with_input_specs(g, &specs).unwrap())
        .collect();

    let x_col_data = spread(N_COL, 0, -5.0, 5.0);
    let t_col_data = spread(N_COL, 1000, 0.0, 1.0);
    let x_ic_data = spread(N_IC, 2000, -5.0, 5.0);
    let u_ic_data: Vec<f64> = x_ic_data.iter().map(|&x| soliton(x, 0.0)).collect();
    let x_bc_data: Vec<f64> = (0..N_BC)
        .map(|k| if k % 2 == 0 { -5.0 } else { 5.0 })
        .collect();
    let t_bc_data = spread(N_BC, 3000, 0.0, 1.0);
    let u_bc_data: Vec<f64> = x_bc_data
        .iter()
        .zip(&t_bc_data)
        .map(|(&x, &t)| soliton(x, t))
        .collect();
    let data = [
        column(x_col_data),
        column(t_col_data),
        column(x_ic_data),
        column(u_ic_data),
        column(x_bc_data),
        column(t_bc_data),
        column(u_bc_data),
    ];

    let mut param_values = net.init_tensors();
    let runtime = cpu_runtime();
    let input_refs: Vec<&Tensor> = param_values.iter().chain(&data).collect();
    let loss_value = scalar(&runtime, &loss_program, &input_refs);
    assert!(
        loss_value.is_finite() && loss_value > 0.0,
        "loss = {loss_value}"
    );

    let grad_values: Vec<Tensor> = grad_programs
        .iter()
        .map(|program| run_compiled_one(&runtime, program, &input_refs).unwrap())
        .collect();

    // Central differences of the loss for the first and last entry of every
    // parameter tensor.
    let h = 1e-5;
    for (p, (grad, shape)) in grad_values.iter().zip(&param_shapes).enumerate() {
        assert_eq!(grad.shape(), shape.as_slice(), "gradient {p} shape");
        let grad = grad.as_slice::<f64>().unwrap();
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "gradient {p} not finite"
        );
        let mut checked = vec![0, grad.len() - 1];
        checked.dedup();
        for k in checked {
            let original = param_values[p].as_slice::<f64>().unwrap()[k];
            let mut loss_at = |value: f64| {
                let mut entries = param_values[p].as_slice::<f64>().unwrap().to_vec();
                entries[k] = value;
                param_values[p] = Tensor::from_vec_col_major(shape.clone(), entries).unwrap();
                let refs: Vec<&Tensor> = param_values.iter().chain(&data).collect();
                scalar(&runtime, &loss_program, &refs)
            };
            let fd = (loss_at(original + h) - loss_at(original - h)) / (2.0 * h);
            loss_at(original);
            let abs_err = (fd - grad[k]).abs();
            let scale = grad[k].abs().max(1.0);
            assert!(
                abs_err <= 1e-8 * scale,
                "parameter {p} entry {k}: grad = {:e}, central difference = {fd:e}, \
                 |error| = {abs_err:e}, |error| / max(|grad|, 1) = {:e}",
                grad[k],
                abs_err / scale
            );
        }
    }
}
