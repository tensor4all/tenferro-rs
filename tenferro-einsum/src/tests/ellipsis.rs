use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::einsum;

fn make_context() -> CpuContext {
    CpuContext::new(1)
}

#[test]
fn test_ellipsis_batched_matrix_multiply() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 4, 5], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 3, 5]);
}

#[test]
fn test_ellipsis_single_batch_dim() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[3, 2, 3], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[3, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[3, 2, 4]);
}

#[test]
fn test_ellipsis_no_batch_dims() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 4]);
}

#[test]
fn test_ellipsis_multiple_batch_dims() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 2, 3], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 3, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 3, 2, 4]);
}

#[test]
fn test_ellipsis_output_no_ellipsis() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 4, 5], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->ik", &[&a, &b], None).unwrap();

    assert_eq!(result.dims(), &[3, 5]);
}

#[test]
fn test_ellipsis_with_values() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3], col).unwrap();

    let b_data: Vec<f64> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let b = Tensor::<f64>::from_slice(&b_data, &[3, 2], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 2]);
}

#[test]
fn test_ellipsis_inconsistent_batch_dims_error() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 3, 3, 5], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None);

    assert!(result.is_err());
}

#[test]
fn test_ellipsis_insufficient_dims_error() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None);

    assert!(result.is_err());
}

#[test]
fn test_ellipsis_mixed_with_explicit_labels() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4, 5], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 3, 5, 6], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 3, 4, 6]);
}

#[test]
fn test_ellipsis_elementwise_multiply() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...,...->...", &[&a, &b], None).unwrap();

    assert_eq!(result.dims(), &[2, 3, 4]);
}

#[test]
fn test_ellipsis_sum_over_batch() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij->...", &[&a], None).unwrap();

    assert_eq!(result.dims(), &[2]);
}

#[test]
fn test_ellipsis_outer_product() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...i,...j->...ij", &[&a, &b], None).unwrap();

    assert_eq!(result.dims(), &[2, 3, 4]);
}
