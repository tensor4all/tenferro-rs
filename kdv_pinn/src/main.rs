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
const LR: f64 = 0.01;
const EPOCHS: usize = 500;

fn main() {
    let net = Mlp::new(&[2, 16, 16, 1]);
    let mut rng = rand::thread_rng();
    let mut params = net.init_tensors(&mut rng);

    let x_ic = TracedTensor::input_concrete_shape(DType::F64, &[N_IC, 1]);
    let u_ic_true = TracedTensor::input_concrete_shape(DType::F64, &[N_IC, 1]);

    let t_zero = TracedTensor::from_vec_col_major(vec![N_IC, 1], vec![0.0_f64; N_IC]);
    let xt_ic = TracedTensor::stack(&[&x_ic, &t_zero], 1)
        .unwrap()
        .reshape(&[N_IC, 2]);
    let u_ic = net.forward(&xt_ic);
    let loss = loss::mean_square(&u_ic, &u_ic_true, N_IC);

    let param_grads: Vec<TracedTensor> = net
        .parameters()
        .iter()
        .map(|p| loss.grad(p).expect("grad computation failed"))
        .collect();

    let mut compiler = GraphCompiler::new();
    let param_specs = net.input_specs();
    let ic_x_spec: &[usize] = &[N_IC, 1];
    let ic_u_spec: &[usize] = &[N_IC, 1];
    let specs: Vec<(&TracedTensor, DType, &[usize])> = param_specs
        .iter()
        .map(|(p, dtype, shape)| (*p, *dtype, shape.as_slice()))
        .chain([
            (&x_ic, DType::F64, ic_x_spec),
            (&u_ic_true, DType::F64, ic_u_spec),
        ])
        .collect();
    let loss_program = compiler.compile_with_input_specs(&loss, &specs).unwrap();
    let grad_programs: Vec<_> = param_grads
        .iter()
        .map(|g| compiler.compile_with_input_specs(g, &specs).unwrap())
        .collect();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let sampler = Sampler::new(-5.0, 5.0, 0.0, 1.0);
    let mut opt = Sgd::new(LR);

    for epoch in 0..EPOCHS {
        let (x_ic_tensor, u_ic_tensor) = sampler.initial(N_IC, &mut rng);
        let mut bindings: Vec<(&TracedTensor, &Tensor)> = Vec::new();
        for (p, t) in net.parameters().iter().zip(params.iter()) {
            bindings.push((*p, t));
        }
        bindings.push((&x_ic, &x_ic_tensor));
        bindings.push((&u_ic_true, &u_ic_tensor));

        let loss_tensor = executor.run_with_inputs(&loss_program, &bindings).unwrap();
        let loss_value = loss_tensor.as_slice::<f64>().unwrap()[0];

        let mut grads = Vec::new();
        for program in &grad_programs {
            grads.push(executor.run_with_inputs(program, &bindings).unwrap());
        }

        opt.step(&mut params, &grads);

        if epoch % 50 == 0 {
            println!("epoch {}: loss={:.6e}", epoch, loss_value);
        }
    }
}
