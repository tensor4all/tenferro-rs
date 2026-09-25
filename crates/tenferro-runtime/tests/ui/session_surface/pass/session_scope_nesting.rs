//! An execution scope wrapping one session entry.
//!
//! Both `with_execution_scope` and `with_backend_session` survive the
//! unification of issue #1926: the scope holds the resource permit and the
//! entry reuses it. Nesting a session inside a scope is the amortizing shape a
//! caller uses when it cannot hold one session for the whole workload.

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};

fn main() {
    let owner = CpuBackend::with_threads(1).unwrap();
    let mut operations = owner.clone();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let out = owner
        .with_execution_scope(|| {
            operations.with_backend_session(|session| {
                session.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
            })
        })
        .unwrap()
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}
