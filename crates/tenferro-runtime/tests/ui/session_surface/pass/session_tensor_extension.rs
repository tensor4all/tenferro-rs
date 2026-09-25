//! The `Tensor`-side session extension, which is the only broadcasting
//! spelling.
//!
//! `TensorElementwise::add` requires equal shapes; NumPy-style broadcasting
//! lives in the session extension surface. This fixture pins that asymmetry so
//! the unification cannot delete the only spelling that can express a
//! broadcast operand set.

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

fn main() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let column = Tensor::from_vec_col_major(vec![1], vec![0.5_f64]).unwrap();
    let row = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let out = backend
        .with_backend_session(|session| {
            let broadcast = column.add(&row, session)?;
            assert_eq!(broadcast.shape(), &[2]);
            broadcast.reduce_sum(&[0], session)
        })
        .unwrap();

    // `column` broadcasts to [1.5, 2.5], so the total is 4.0.
    assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0]);
}
