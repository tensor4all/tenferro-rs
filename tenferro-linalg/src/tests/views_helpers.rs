use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ad_helpers::{matrix_to_rhs, matrix_transpose, rhs_to_matrix};

#[test]
fn matrix_transpose_rejects_1d() {
    let t = Tensor::<f64>::ones(
        &[3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert!(matrix_transpose(&t).is_err());
}

#[test]
fn rhs_to_matrix_rejects_invalid_rank() {
    let t = Tensor::<f64>::ones(
        &[3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert!(rhs_to_matrix(&t, 0).is_err());
    assert!(rhs_to_matrix(&t, 3).is_err());
}

#[test]
fn matrix_to_rhs_rejects_invalid_rank() {
    let t = Tensor::<f64>::ones(
        &[3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert!(matrix_to_rhs(t.clone(), 0).is_err());
    assert!(matrix_to_rhs(t, 3).is_err());
}
