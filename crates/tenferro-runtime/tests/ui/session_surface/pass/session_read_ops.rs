//! The engine read surface: `_read` operations on one engine.
//!
//! Every tensor operation is spelled on the engine the caller holds.

use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    DotGeneralConfig, Tensor, TensorDot, TensorElementwise, TensorRead, TensorReduction,
};

const NONE: &[usize] = &[];

fn dot_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [0].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: NONE.iter().copied().collect(),
        rhs_batch_dims: NONE.iter().copied().collect(),
    }
}

#[test]
fn session_read_ops() {
    let mut engine = CpuEngine::serial();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let sum = engine
        .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .unwrap();
    let total = engine
        .reduce_sum_read(TensorRead::from_tensor(&sum), &[0])
        .unwrap();
    let dot = engine
        .dot_general_read(
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
            &dot_config(),
        )
        .unwrap();

    assert_eq!(total.as_slice::<f64>().unwrap(), &[10.0]);
    assert_eq!(dot.as_slice::<f64>().unwrap(), &[11.0]);
}
