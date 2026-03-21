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

#[test]
fn test_ellipsis_with_f32() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f32>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f32>::zeros(&[2, 4, 5], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f32>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 3, 5]);
}

#[test]
fn test_ellipsis_trace() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[3, 4, 4], LogicalMemorySpace::MainMemory, col);

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ii->...", &[&a], None).unwrap();

    assert_eq!(result.dims(), &[3]);
}

#[test]
fn test_ellipsis_diagonal_extraction() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4, 4], LogicalMemorySpace::MainMemory, col);

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ii->...i", &[&a], None).unwrap();

    assert_eq!(result.dims(), &[2, 3, 4]);
}

#[test]
fn test_ellipsis_large_batch_dims() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4, 5, 6], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 3, 4, 6, 7], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 3, 4, 5, 7]);
}

#[test]
fn test_ellipsis_unary_operations() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij->...ji", &[&a], None).unwrap();

    assert_eq!(result.dims(), &[2, 4, 3]);
}

#[test]
fn test_ellipsis_sum_all() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...->", &[&a], None).unwrap();

    assert_eq!(result.dims(), &[]);
}

#[test]
fn test_ellipsis_ellipsis_only_input() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...->...", &[&a], None).unwrap();

    assert_eq!(result.dims(), &[2, 3, 4]);
}

#[test]
fn test_ellipsis_double_ellipsis_error() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...i...j->...ij", &[&a], None);

    assert!(result.is_err());
}

#[test]
fn test_ellipsis_invalid_dot_error() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, col);

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, ".ij->ij", &[&a], None);

    assert!(result.is_err());
}

#[test]
fn test_ellipsis_matrix_vector_product() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);
    let v = Tensor::<f64>::zeros(&[2, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...j->...i", &[&a, &v], None).unwrap();

    assert_eq!(result.dims(), &[2, 3]);
}

#[test]
fn test_ellipsis_contraction_with_batch_broadcast() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[4, 5], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,jk->...ik", &[&a, &b], None).unwrap();

    assert_eq!(result.dims(), &[2, 3, 5]);
}

#[test]
fn test_ellipsis_with_actual_values() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 2, 3], col).unwrap();

    let b_data: Vec<f64> = (0..18).map(|i| i as f64).collect();
    let b = Tensor::<f64>::from_slice(&b_data, &[2, 3, 3], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 2, 3]);
}

#[test]
fn test_ellipsis_issue_529_example() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a3d = Tensor::<f64>::zeros(&[2, 2, 2], LogicalMemorySpace::MainMemory, col);
    let b3d = Tensor::<f64>::zeros(&[2, 2, 2], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a3d, &b3d], None);

    assert!(
        result.is_ok(),
        "Ellipsis notation should be supported as per issue #529"
    );
    assert_eq!(result.unwrap().dims(), &[2, 2, 2]);
}
