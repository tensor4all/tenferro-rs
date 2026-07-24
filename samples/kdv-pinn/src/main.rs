//! Physics-informed neural network (PINN) example for the Korteweg–de Vries (KdV)
//! equation.
//!
//! This example trains a small multi-layer perceptron to approximate the KdV
//! single-soliton solution on the spatial domain `x ∈ [-5, 5]` and time domain
//! `t ∈ [0, 1]`. The loss combines the PDE residual in the interior, the initial
//! condition at `t = 0`, and exact-solution Dirichlet boundary data at `x = ±5`.
//!
//! Run the example with:
//!
//! ```bash
//! cargo run -p kdv_pinn --release
//! ```
//!
//! Optional outputs are controlled by command-line flags:
//!
//! - `--gif <path>` writes an animated comparison of the predicted and analytic
//!   solution over time.
//! - `--loss-png <path>` writes the training-loss curve (log-scale y-axis).
//!
//! ```bash
//! cargo run -p kdv_pinn --release -- --gif kdv_pinn.gif --loss-png loss.png
//! ```

mod loss;
mod network;
mod optimizer;
mod pde;
mod plot;
mod sampler;

use network::Mlp;
use optimizer::{step_decay_lr, Adam};
use sampler::Sampler;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::Tensor;

/// Return the value following the given command-line `flag`, if present.
///
/// For example, with arguments `--gif out.gif`, `arg_value("--gif")` returns
/// `Some("out.gif")`. Returns `None` when the flag is absent or has no value.
fn arg_value(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for window in args.windows(2) {
        if window[0] == flag {
            return Some(window[1].clone());
        }
    }
    None
}

const N_IC: usize = 128;
const N_COL: usize = 1024;
const N_BC: usize = 128;
const N_EVAL: usize = 100;
const LAMBDA_PDE: f64 = 1.0;
const LAMBDA_IC: f64 = 1.0;
const LAMBDA_BC: f64 = 1.0;
const LR: f64 = 0.001;
const EPOCHS: usize = 3000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let net = Mlp::new(&[2, 64, 64, 1])?;
    let mut rng = rand::thread_rng();
    let mut params = net.init_tensors(&mut rng);

    let x_ic = TracedTensor::input_concrete_shape(DType::F64, &[N_IC, 1])?;
    let u_ic_true = TracedTensor::input_concrete_shape(DType::F64, &[N_IC, 1])?;

    let t_zero = TracedTensor::from_vec_col_major(vec![N_IC, 1], vec![0.0_f64; N_IC])?;
    let xt_ic = TracedTensor::stack(&[&x_ic, &t_zero], 1)?.reshape(&[N_IC, 2])?;
    let u_ic = net.forward(&xt_ic)?;

    let x_col = TracedTensor::input_concrete_shape(DType::F64, &[N_COL, 1])?;
    let t_col = TracedTensor::input_concrete_shape(DType::F64, &[N_COL, 1])?;
    let xt_col = TracedTensor::stack(&[&x_col, &t_col], 1)?.reshape(&[N_COL, 2])?;
    let u_col = net.forward(&xt_col)?;

    let x_bc = TracedTensor::input_concrete_shape(DType::F64, &[N_BC, 1])?;
    let t_bc = TracedTensor::input_concrete_shape(DType::F64, &[N_BC, 1])?;
    let u_bc_true = TracedTensor::input_concrete_shape(DType::F64, &[N_BC, 1])?;
    let xt_bc = TracedTensor::stack(&[&x_bc, &t_bc], 1)?.reshape(&[N_BC, 2])?;
    let u_bc = net.forward(&xt_bc)?;

    let residual = pde::kdv_residual(&u_col, &x_col, &t_col)?;
    let total_loss = loss::total_loss(
        &residual, &u_ic, &u_ic_true, &u_bc, &u_bc_true, N_COL, N_IC, N_BC, LAMBDA_PDE, LAMBDA_IC,
        LAMBDA_BC,
    )?;

    let param_grads: Vec<TracedTensor> = net
        .parameters()
        .iter()
        .map(|p| total_loss.grad(p))
        .collect::<tenferro_runtime::Result<Vec<_>>>()?;

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
    let loss_program = compiler.compile_with_input_specs(&total_loss, &specs)?;
    let grad_programs: Vec<_> = param_grads
        .iter()
        .map(|g| compiler.compile_with_input_specs(g, &specs))
        .collect::<tenferro_runtime::Result<Vec<_>>>()?;

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut opt = Adam::new(LR);

    let mut final_loss = f64::INFINITY;
    let mut loss_history: Vec<f64> = Vec::with_capacity(EPOCHS);
    for epoch in 0..EPOCHS {
        let (x_col_tensor, t_col_tensor) = sampler.collocation(N_COL, &mut rng);
        let (x_ic_tensor, u_ic_tensor) = sampler.initial(N_IC, &mut rng);
        let (x_bc_tensor, t_bc_tensor, u_bc_tensor) = sampler.boundary(N_BC, &mut rng);

        let mut inputs: Vec<&Tensor> = Vec::with_capacity(params.len() + 7);
        inputs.extend(params.iter());
        inputs.extend([
            &x_col_tensor,
            &t_col_tensor,
            &x_ic_tensor,
            &u_ic_tensor,
            &x_bc_tensor,
            &t_bc_tensor,
            &u_bc_tensor,
        ]);

        let loss_tensor = executor.run_with_inputs(&loss_program, &inputs)?;
        final_loss = loss_tensor.as_slice::<f64>().expect("loss data")[0];
        loss_history.push(final_loss);

        let mut grads = Vec::new();
        for program in &grad_programs {
            grads.push(executor.run_with_inputs(program, &inputs)?);
        }

        opt.set_lr(step_decay_lr(epoch, EPOCHS, LR));
        opt.step(&mut params, &grads);

        if epoch % 50 == 0 {
            println!("epoch {}: loss={:.6e}", epoch, final_loss);
        }
    }
    println!("final loss after {} epochs: {:.6e}", EPOCHS, final_loss);

    if let Some(loss_png_path) = arg_value("--loss-png") {
        plot::write_loss_png(&loss_png_path, &loss_history).expect("write loss png failed");
        println!("saved loss curve to {}", loss_png_path);
    }

    // Evaluation grid at t = 0.5.
    let x_eval = TracedTensor::input_concrete_shape(DType::F64, &[N_EVAL, 1])?;
    let t_eval = TracedTensor::input_concrete_shape(DType::F64, &[N_EVAL, 1])?;
    let xt_eval = TracedTensor::stack(&[&x_eval, &t_eval], 1)?.reshape(&[N_EVAL, 2])?;
    let u_eval = net.forward(&xt_eval)?;

    let mut eval_specs: Vec<(&TracedTensor, DType, &[usize])> = param_specs
        .iter()
        .map(|(p, dtype, shape)| (*p, *dtype, shape.as_slice()))
        .collect();
    eval_specs.push((&x_eval, DType::F64, &[N_EVAL, 1]));
    eval_specs.push((&t_eval, DType::F64, &[N_EVAL, 1]));
    let eval_program = compiler.compile_with_input_specs(&u_eval, &eval_specs)?;

    let mut x_eval_data = Vec::with_capacity(N_EVAL);
    let mut u_true_data = Vec::with_capacity(N_EVAL);
    for i in 0..N_EVAL {
        let x = -5.0 + 10.0 * (i as f64) / (N_EVAL as f64 - 1.0);
        let t = 0.5;
        x_eval_data.push(x);
        u_true_data.push(2.0 * (1.0 / ((x - 4.0 * t).cosh())).powi(2));
    }
    let x_eval_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], x_eval_data.clone())?;
    let t_eval_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], vec![0.5_f64; N_EVAL])?;
    let u_true_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], u_true_data)?;

    let mut eval_inputs: Vec<&Tensor> = Vec::with_capacity(params.len() + 2);
    eval_inputs.extend(params.iter());
    eval_inputs.push(&x_eval_tensor);
    eval_inputs.push(&t_eval_tensor);

    let u_pred = executor.run_with_inputs(&eval_program, &eval_inputs)?;
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

    if let Some(gif_path) = arg_value("--gif") {
        const N_FRAMES: usize = 30;
        let mut frames: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::with_capacity(N_FRAMES);
        for frame in 0..N_FRAMES {
            let t = frame as f64 / (N_FRAMES as f64 - 1.0);
            let t_eval_tensor = Tensor::from_vec_col_major(vec![N_EVAL, 1], vec![t; N_EVAL])?;
            let mut frame_inputs: Vec<&Tensor> = Vec::with_capacity(params.len() + 2);
            frame_inputs.extend(params.iter());
            frame_inputs.push(&x_eval_tensor);
            frame_inputs.push(&t_eval_tensor);

            let u_pred = executor.run_with_inputs(&eval_program, &frame_inputs)?;
            let pred = u_pred
                .as_slice::<f64>()
                .expect("predicted frame data")
                .to_vec();

            let analytic: Vec<f64> = x_eval_data
                .iter()
                .map(|&x| 2.0 * (1.0 / ((x - 4.0 * t).cosh())).powi(2))
                .collect();

            frames.push((t, analytic, pred));
            if frame % 10 == 0 {
                println!("rendered frame {} / {}", frame, N_FRAMES - 1);
            }
        }
        plot::write_comparison_gif(&gif_path, &x_eval_data, &frames)
            .expect("write comparison gif failed");
        println!("saved animation to {}", gif_path);
    }
    Ok(())
}
