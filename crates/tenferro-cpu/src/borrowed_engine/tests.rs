//! Unit tests of the lane carrier's one unsafe bridge. They avoid BLAS, faer and threads so
//! they also run under Miri.

use cpueinsum::Exec;
use tenferro_tensor::backend::Engine;
use tenferro_tensor::{ErrorKind, Tensor, TensorElementwise, TensorRead};

use super::{with_borrowed_cpu_engine, BorrowedCpuEngine};
use crate::buffer_pool::BufferPool;
use crate::CpuEngine;

#[test]
fn the_bridge_lends_the_bundle_and_pool_of_a_borrowed_engine() {
    let mut exec = Exec::serial();
    let mut pool = BufferPool::new();
    let mut carrier = BorrowedCpuEngine::new(&mut exec, &mut pool);
    let erased: &mut dyn Engine = &mut carrier;
    let seen = with_borrowed_cpu_engine(erased, |exec, pool| (exec.budget(), pool.stats()));
    assert_eq!(seen.map(|(budget, _)| budget), Some(1));
    // A second visit reaches the same carrier again: the cast does not consume it.
    let again = with_borrowed_cpu_engine(erased, |_, pool| pool.stats());
    assert_eq!(again, seen.map(|(_, stats)| stats));
    assert!(format!("{carrier:?}").contains("BorrowedCpuEngine"));
}

#[test]
fn the_bridge_answers_none_for_an_engine_that_is_not_a_lane() {
    let mut engine = CpuEngine::serial();
    let erased: &mut dyn Engine = &mut engine;
    assert!(with_borrowed_cpu_engine(erased, |_, _| ()).is_none());
}

#[test]
fn a_lane_carrier_refuses_operations_other_than_the_factorizations() {
    let mut exec = Exec::serial();
    let mut pool = BufferPool::new();
    let mut carrier = BorrowedCpuEngine::new(&mut exec, &mut pool);
    let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let refused = carrier
        .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
        .unwrap_err();
    assert_eq!(refused.kind(), ErrorKind::Unsupported);
}
