//! A generic helper over `&mut dyn Engine`.
//!
//! This is the shape scheduler and extension code use: the helper receives an
//! engine and must not create one.

use tenferro_cpu::CpuEngine;
use tenferro_tensor::{Engine, Tensor, TensorRead};

/// Two chained additions on the caller's engine.
fn add_twice(engine: &mut dyn Engine, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
    let first = engine.add_read(TensorRead::from_tensor(a), TensorRead::from_tensor(b))?;
    engine.add_read(TensorRead::from_tensor(&first), TensorRead::from_tensor(b))
}

#[test]
fn session_dyn_erased() {
    let mut engine = CpuEngine::serial();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let out = add_twice(&mut engine, &a, &b).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[7.0, 10.0]);
}
