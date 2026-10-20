//! The `Tensor`-side engine extension, which is the broadcasting spelling.
//!
//! `TensorElementwise::add` requires equal shapes; NumPy-style broadcasting
//! lives in the tensor engine extension surface. This fixture pins that
//! asymmetry so the extension cannot be dropped accidentally.

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

#[test]
fn session_tensor_extension() {
    let mut engine = CpuEngine::serial();
    let column = Tensor::from_vec_col_major(vec![1], vec![0.5_f64]).unwrap();
    let row = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let broadcast = column.add(&row, &mut engine).unwrap();
    assert_eq!(broadcast.shape(), &[2]);
    let out = broadcast.reduce_sum(Some(&[0]), &mut engine).unwrap();

    // `column` broadcasts to [1.5, 2.5], so the total is 4.0.
    assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0]);
}
