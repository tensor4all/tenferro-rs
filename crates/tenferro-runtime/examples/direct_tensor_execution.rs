use tenferro_cpu::CpuEngine;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = CpuEngine::serial();

    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    let c = a.matmul(&b, &mut engine)?;
    assert_eq!(c.shape(), &[2, 2]);

    Ok(())
}
