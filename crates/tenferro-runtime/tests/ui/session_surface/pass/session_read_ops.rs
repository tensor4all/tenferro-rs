//! The borrowed-session read surface: `_read` operations on one session entry.
//!
//! This is the spelling that survives the unification of issue #1926. It must
//! compile before and after the one-shot idiom is deleted.

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor, TensorRead};

const NONE: &[usize] = &[];

fn dot_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [0].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: NONE.into(),
        rhs_batch_dims: NONE.into(),
    }
}

fn main() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let (total, dot) = backend
        .with_backend_session(|session| {
            let sum = session.add_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
            )?;
            let total = session.reduce_sum_read(TensorRead::from_tensor(&sum), &[0])?;
            let dot = session.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &dot_config(),
            )?;
            Ok::<_, tenferro_tensor::Error>((total, dot))
        })
        .unwrap();

    assert_eq!(total.as_slice::<f64>().unwrap(), &[10.0]);
    assert_eq!(dot.as_slice::<f64>().unwrap(), &[11.0]);
}
