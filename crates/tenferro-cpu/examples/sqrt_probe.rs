//! Throwaway 1T probe for tenferro-rs#2284 (not committed).
use std::time::Instant;
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{Tensor, TensorAnalytic, TensorRead};

const N: usize = 33_554_432;

fn main() {
    let mut engine = CpuEngine::serial();
    let x = Tensor::from_vec_col_major(vec![N], (0..N).map(|i| (i % 1000) as f64 + 0.5).collect()).unwrap();
    let x2 = Tensor::from_vec_col_major(vec![N], (0..N).map(|i| (i % 1000) as f64 + 0.5).collect()).unwrap();
    let traffic = |ms: f64| 2.0 * N as f64 * 8.0 / 1e9 / (ms / 1e3);
    let measure = |engine: &mut CpuEngine, name: &str, f: &mut dyn FnMut(&mut CpuEngine) -> Tensor| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let y = f(engine);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(y.as_slice::<f64>().unwrap()[0]);
        }
        println!("{name:8}: {best:8.2} ms  ({:.1} GB/s)", traffic(best));
    };
    measure(&mut engine, "sqrt", &mut |e| e.sqrt_read(TensorRead::from_tensor(&x)).unwrap());
    measure(&mut engine, "rsqrt", &mut |e| e.rsqrt_read(TensorRead::from_tensor(&x)).unwrap());
    measure(&mut engine, "exp", &mut |e| e.exp_read(TensorRead::from_tensor(&x)).unwrap());
    measure(&mut engine, "neg", &mut |e| e.neg_read(TensorRead::from_tensor(&x)).unwrap());
    measure(&mut engine, "mul", &mut |e| e.mul_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x2)).unwrap());
}
