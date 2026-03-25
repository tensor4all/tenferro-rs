use std::path::PathBuf;
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::matrix_batch_1_norm_tensor;

fn repo_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(path)).unwrap()
}

#[test]
fn matrix_exp_helpers_are_explicitly_batchwise_and_mask_based() {
    let matrix_exp = repo_file("src/ad_helpers/matrix_exp.rs");

    assert!(
        matrix_exp.contains("matrix_batch_1_norm_tensor"),
        "matrix_exp helper module should define a batchwise 1-norm helper"
    );
    assert!(
        matrix_exp.contains("blend_tensor_by_real_mask_same_shape"),
        "matrix_exp helper module should define a reusable real-mask blend helper"
    );
    assert!(
        matrix_exp.contains("sum_keep_axes"),
        "matrix_exp batchwise 1-norm helper should reduce over rows/cols rather than return one global scalar"
    );
}

fn tensor_data(tensor: &Tensor<f64>) -> Vec<f64> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

#[test]
fn matrix_batch_1_norm_tensor_returns_one_value_per_batch() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[1.0_f64, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 1.0],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let norms = matrix_batch_1_norm_tensor(&mut ctx, &a).unwrap();

    assert_eq!(norms.logical_memory_space(), LogicalMemorySpace::MainMemory);
    assert_eq!(norms.dims(), &[2]);
    assert_eq!(tensor_data(&norms), vec![2.0, 4.0]);
}
