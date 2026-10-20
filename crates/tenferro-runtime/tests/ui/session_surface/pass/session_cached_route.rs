//! The cache-aware route used by the scheduler.
//!
//! The engine owns its plan cache and its `_cached` contraction entry point;
//! the operation is spelled directly on the engine, which is the route the
//! scheduler uses.

use tenferro_cpu::CpuEngine;
use tenferro_tensor::{DotGeneralConfig, SessionCachedDot, Tensor, TensorRead};

const NONE: &[usize] = &[];

#[test]
fn session_cached_route() {
    let mut engine = CpuEngine::serial();
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: NONE.iter().copied().collect(),
        rhs_batch_dims: NONE.iter().copied().collect(),
    };

    let out = engine
        .dot_general_read_cached(
            None,
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
        )
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}
