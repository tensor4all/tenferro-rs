//! Physics-informed neural network (PINN) example for the Korteweg–de Vries (KdV)
//! equation.
//!
//! This example trains a small multi-layer perceptron to approximate the KdV
//! single-soliton solution on the spatial domain `x ∈ [-5, 5]` and time domain
//! `t ∈ [0, 1]`. The loss combines the PDE residual in the interior, the initial
//! condition at `t = 0`, and periodic boundary data at `x = ±5`.
//!
//! Run the example with:
//!
//! ```bash
//! cargo run -p kdv_pinn --release
//! ```

mod loss;
mod network;
mod optimizer;
mod pde;
mod sampler;

use network::Mlp;
use optimizer::Sgd;
use sampler::Sampler;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::Tensor;

const N_IC: usize = 32;
const N_COL: usize = 256;
const N_BC: usize = 32;
const N_EVAL: usize = 100;
const LAMBDA_IC: f64 = 100.0;
const LAMBDA_BC: f64 = 10.0;
const LR: f64 = 0.001;
const EPOCHS: usize = 500;

fn main() {
    let net = Mlp::new(&[2, 16, 16, 1]);
    let mut rng = rand::thread_rng();
    let mut params = net.init_tensors(&mut rng);

    let x_ic = TracedTensor::input_concrete_shape(DType::F64, &[N_IC, 1]);
    let u_ic_true = TracedTensor::input_concrete_shape(DType::F64, &[N_IC, 1]);

    let t_zero = TracedTensor::from_vec_col_major(vec![N_IC, 1], vec![0.0_f64; N_IC]);
    let xt_ic = TracedTensor::stack(&[&x_ic, &t_zero], 1)
        .expect("stacking x_ic and t_zero must succeed")
        .reshape(&[N_IC, 2]);
    let u_ic = net.forward(&xt_ic);

    let x_col = TracedTensor::input_concrete_shape(DType::F64, &[N_COL, 1]);
    let t_col = TracedTensor::input_concrete_shape(DType::F64, &[N_COL, 1]);
    let xt_col = TracedTensor::stack(
        &[&x_col.reshape(&[N_COL, 1]), &t_col.reshape(&[N_COL, 1])],
        1,
    )
    .expect("stack x_col and t_col must succeed")
    .reshape(&[N_COL, 2]);
    let u_col = net.forward(&xt_col);

    let x_bc = TracedTensor::input_concrete_shape(DType::F64, &[N_BC, 1]);
    let t_bc = TracedTensor::input_concrete_shape(DType::F64, &[N_BC, 1]);
    let u_bc_true = TracedTensor::input_concrete_shape(DType::F64, &[N_BC, 1]);
    let xt_bc = TracedTensor::stack(&[&x_bc.reshape(&[N_BC, 1]), &t_bc.reshape(&[N_BC, 1])], 1)
        .expect("stack x_bc and t_bc must succeed")
        .reshape(&[N_BC, 2]);
    let u_bc = net.forward(&xt_bc);

    let residual = pde::kdv_residual(&u_col, &x_col, &t_col).expect("kdv_residual failed");
    let total_loss = loss::total_loss(
        &residual, &u_ic, &u_ic_true, &u_bc, &u_bc_true, N_COL, N_IC, N_BC, LAMBDA_IC, LAMBDA_BC,
    );

    let param_grads: Vec<TracedTensor> = net
        .parameters()
        .iter()
        .map(|p| total_loss.grad(p).expect("grad computation failed"))
        .collect();

    let mut compiler = GraphCompiler::new();
    let param_specs = net.input_specs();
    let col_spec: &[usize] = &[N_COL, 1];
    let ic_spec: &[usize] = &[N_IC, 1];
    let bc_spec: &[usize] = &[N_BC, 1];
    let specs: Vec<(&TracedTensor, DType, &[usize])> = param_specs
        .iter()
        .map(|(p, dtype, shape)| (*p, *dtype, shape.as_slice()))
        .chain([
            (&x_col, DType::F64, col_spec),
            (&t_col, DType::F64, col_spec),
            (&x_ic, DType::F64, ic_spec),
            (&u_ic_true, DType::F64, ic_spec),
            (&x_bc, DType::F64, bc_spec),
            (&t_bc, DType::F64, bc_spec),
            (&u_bc_true, DType::F64, bc_spec),
        ])
        .collect();
    let loss_program = compiler
        .compile_with_input_specs(&total_loss, &specs)
        .expect("compile total_loss failed");
    let grad_programs: Vec<_> = param_grads
        .iter()
        .map(|g| {
            compiler
                .compile_with_input_specs(g, &specs)
                .expect("compile grad failed")
        })
        .collect();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut opt = Sgd::new(LR);

    let mut final_loss = f64::INFINITY;
    for epoch in 0..EPOCHS {
        let (x_col_tensor, t_col_tensor) = sampler.collocation(N_COL, &mut rng);

        let (x_ic_tensor, u_ic_tensor) = sampler.initial(N_IC, &mut rng);
        let (x_bc_tensor, t_bc_tensor, u_bc_tensor) = sampler.boundary(N_BC, &mut rng);

        let mut bindings: Vec<(&TracedTensor, &Tensor)> = Vec::new();
        for (p, t) in net.parameters().iter().zip(params.iter()) {
            bindings.push((*p, t));
        }
        bindings.push((&x_col, &x_col_tensor));
        bindings.push((&t_col, &t_col_tensor));
        bindings.push((&x_ic, &x_ic_tensor));
        bindings.push((&u_ic_true, &u_ic_tensor));
        bindings.push((&x_bc, &x_bc_tensor));
        bindings.push((&t_bc, &t_bc_tensor));
        bindings.push((&u_bc_true, &u_bc_tensor));

        let loss_tensor = executor
            .run_with_inputs(&loss_program, &bindings)
            .expect("evaluate loss failed");
        final_loss = loss_tensor.as_slice::<f64>().expect("loss data")[0];

        let mut grads = Vec::new();
        for program in &grad_programs {
            grads.push(
                executor
                    .run_with_inputs(program, &bindings)
                    .expect("evaluate grad failed"),
            );
        }

        opt.step(&mut params, &grads);

        if epoch % 50 == 0 {
            println!("epoch {}: loss={:.6e}", epoch, final_loss);
        }
    }
    println!("final loss after {} epochs: {:.6e}", EPOCHS, final_loss);

    // Evaluation grid at t = 0.5.
    let x_eval = TracedTensor::input_concrete_shape(DType::F64, &[N_EVAL, 1]);
    let t_eval = TracedTensor::input_concrete_shape(DType::F64, &[N_EVAL, 1]);
    let xt_eval = TracedTensor::stack(
        &[&x_eval.reshape(&[N_EVAL, 1]), &t_eval.reshape(&[N_EVAL, 1])],
        1,
    )
    .expect("stack eval inputs must succeed")
    .reshape(&[N_EVAL, 2]);
    let u_eval = net.forward(&xt_eval);

    let mut eval_specs: Vec<(&TracedTensor, DType, &[usize])> = param_specs
        .iter()
        .map(|(p, dtype, shape)| (*p, *dtype, shape.as_slice()))
        .collect();
    eval_specs.push((&x_eval, DType::F64, &[N_EVAL, 1]));
    eval_specs.push((&t_eval, DType::F64, &[N_EVAL, 1]));
    let eval_program = compiler
        .compile_with_input_specs(&u_eval, &eval_specs)
        .expect("compile eval program failed");

    let mut x_eval_data = Vec::with_capacity(N_EVAL);
    let mut u_true_data = Vec::with_capacity(N_EVAL);
    for i in 0..N_EVAL {
        let x = -5.0 + 10.0 * (i as f64) / (N_EVAL as f64 - 1.0);
        let t = 0.5;
        x_eval_data.push(x);
        u_true_data.push(2.0 * (1.0 / ((x - 4.0 * t).cosh())).powi(2));
    }
    let x_eval_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], x_eval_data);
    let t_eval_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], vec![0.5_f64; N_EVAL]);
    let u_true_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], u_true_data);

    let mut eval_bindings: Vec<(&TracedTensor, &Tensor)> = net
        .parameters()
        .iter()
        .zip(params.iter())
        .map(|(p, t)| (*p, t))
        .collect();
    eval_bindings.push((&x_eval, &x_eval_tensor));
    eval_bindings.push((&t_eval, &t_eval_tensor));

    let u_pred = executor
        .run_with_inputs(&eval_program, &eval_bindings)
        .expect("evaluate eval program failed");
    let pred = u_pred.as_slice::<f64>().expect("predicted solution data");
    let truth = u_true_tensor.as_slice::<f64>().expect("true solution data");
    let l2_error = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        .sqrt()
        / truth.iter().map(|t| t.powi(2)).sum::<f64>().sqrt();
    println!("L2 relative error at t=0.5: {:.6e}", l2_error);
}
