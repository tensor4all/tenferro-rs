//! A generic helper over `&mut dyn BackendSession`.
//!
//! This is the shape scheduler and extension code use: the helper receives a
//! session and must not create one. It survives the unification, and after the
//! one-shot deletion it is the only way to write such a helper.

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{BackendSession, BackendSessionHost, Tensor, TensorRead};

/// Two chained additions inside the caller's session.
fn add_twice(
    session: &mut dyn BackendSession,
    a: &Tensor,
    b: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    let first = session.add_read(TensorRead::from_tensor(a), TensorRead::from_tensor(b))?;
    session.add_read(TensorRead::from_tensor(&first), TensorRead::from_tensor(b))
}

fn main() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let out = backend
        .with_backend_session(|session| add_twice(session, &a, &b))
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[7.0, 10.0]);
}
