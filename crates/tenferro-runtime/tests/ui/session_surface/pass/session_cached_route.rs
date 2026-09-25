//! The cache-aware session route used by the scheduler.
//!
//! `with_backend_session_cached` plus a session `_cached` contraction is how a
//! runtime passes its prepared-plan cache to an execution region. It survives
//! the unification of issue #1926, and after the one-shot deletion it is the
//! only cached route: the owner-level `BackendCachedDot` spelling disappears.

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendRuntimeCache, BackendSessionHost, DotGeneralConfig, Tensor, TensorRead,
};

const NONE: &[usize] = &[];

fn main() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: NONE.into(),
        rhs_batch_dims: NONE.into(),
    };

    let out = backend
        .with_backend_session_cached(&mut cache, |session| {
            session.dot_general_read_cached(
                None,
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
        })
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}
