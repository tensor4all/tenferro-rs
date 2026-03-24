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

/// Tests ellipsis notation with f32 precision and mathematical verification.
/// Ensures the ellipsis feature works correctly with single-precision floats.
#[test]
fn test_ellipsis_with_f32() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let a = Tensor::<f32>::from_slice(&a_data, &[2, 2, 3], col).unwrap();

    let b_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let b = Tensor::<f32>::from_slice(&b_data, &[2, 3, 2], col).unwrap();

    let result =
        einsum::<Standard<f32>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 2, 2]);

    let result_data = result.buffer().as_slice().unwrap();
    assert!(
        result_data.iter().all(|&v| v.is_finite()),
        "All f32 result values should be finite"
    );
    assert!(
        result_data.iter().any(|&v| v != 0.0),
        "f32 result should contain non-zero values"
    );
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

/// Tests the exact example from issue #529: batched matrix multiplication with ellipsis notation.
/// This verifies that the ellipsis (...) notation works correctly for batch dimensions,
/// as required for NumPy/PyTorch/JAX compatibility.
///
/// # Mathematical Verification (Column-Major Storage)
///
/// Input data [1,2,3,4,5,6,7,8] with shape [2,2,2] in column-major:
/// - Batch 0: A[0] = [[1,5],[3,7]], B[0] = [[1,5],[3,7]]
/// - Batch 1: A[1] = [[2,6],[4,8]], B[1] = [[2,6],[4,8]]
///
/// Matrix multiplication C[b] = A[b] @ B[b]:
/// - C[0] = [[16,40],[24,64]]
/// - C[1] = [[28,60],[40,88]]
///
/// Result stored in column-major: [16,28,24,40,40,60,64,88]
#[test]
fn test_ellipsis_issue_529_example() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (0..8).map(|i| (i + 1) as f64).collect();
    let b_data: Vec<f64> = (0..8).map(|i| (i + 1) as f64).collect();

    let a3d = Tensor::<f64>::from_slice(&a_data, &[2, 2, 2], col).unwrap();
    let b3d = Tensor::<f64>::from_slice(&b_data, &[2, 2, 2], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a3d, &b3d], None);

    assert!(
        result.is_ok(),
        "Ellipsis notation should be supported as per issue #529"
    );

    let result_tensor = result.unwrap();
    assert_eq!(result_tensor.dims(), &[2, 2, 2]);

    let result_data = result_tensor
        .buffer()
        .as_slice()
        .expect("CPU tensor should have slice access");

    assert!(
        result_data.iter().any(|&v| v != 0.0),
        "Result should contain non-zero values"
    );

    assert!(
        result_data.iter().all(|&v| v.is_finite()),
        "All result values should be finite"
    );

    let expected: Vec<f64> = vec![16.0, 28.0, 24.0, 40.0, 40.0, 60.0, 64.0, 88.0];
    for (i, (&got, &exp)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "Result[{}] = {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Verifies that ellipsis notation produces identical results to explicit batch indexing.
/// This confirms that the ellipsis expansion is semantically correct.
#[test]
fn test_ellipsis_equivalence_to_explicit() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
    let b_data: Vec<f64> = (0..12).map(|i| (i + 13) as f64).collect();

    let a = Tensor::<f64>::from_slice(&a_data, &[2, 2, 3], col).unwrap();
    let b = Tensor::<f64>::from_slice(&b_data, &[2, 3, 2], col).unwrap();

    let result_ellipsis =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    let result_explicit =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&a, &b], None).unwrap();

    assert_eq!(result_ellipsis.dims(), result_explicit.dims());

    let ellipsis_data = result_ellipsis.buffer().as_slice().unwrap();
    let explicit_data = result_explicit.buffer().as_slice().unwrap();

    for (i, (&e, &x)) in ellipsis_data.iter().zip(explicit_data.iter()).enumerate() {
        assert!(
            (e - x).abs() < 1e-14,
            "Ellipsis result[{}] = {} differs from explicit result {}",
            i,
            e,
            x
        );
    }
}

/// Tests ellipsis notation with row-major (C-order) memory layout.
/// Verifies that the ellipsis feature works correctly regardless of memory order.
#[test]
fn test_ellipsis_row_major_order() {
    let mut ctx = make_context();
    let row = MemoryOrder::RowMajor;

    let a_data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
    let b_data: Vec<f64> = (0..12).map(|i| (i + 13) as f64).collect();

    let a = Tensor::<f64>::from_slice(&a_data, &[2, 2, 3], row).unwrap();
    let b = Tensor::<f64>::from_slice(&b_data, &[2, 3, 2], row).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 2, 2]);

    let result_data = result.buffer().as_slice().unwrap();
    assert!(
        result_data.iter().all(|&v| v.is_finite()),
        "All result values should be finite in row-major order"
    );
}

/// Tests ellipsis notation with 4-dimensional batch dimensions.
/// Verifies that the ellipsis correctly expands to multiple batch dimensions.
#[test]
fn test_ellipsis_four_batch_dims() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 4, 5, 6], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 3, 4, 6, 7], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 3, 4, 5, 7]);
}

/// Tests ellipsis notation with scalar contraction (trace) across batch dimensions.
#[test]
fn test_ellipsis_batched_trace() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (0..18).map(|i| (i + 1) as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3, 3], col).unwrap();

    let result = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ii->...", &[&a], None).unwrap();

    assert_eq!(result.dims(), &[2]);

    let result_data = result.buffer().as_slice().unwrap();
    assert!(
        result_data.iter().all(|&v| v.is_finite()),
        "All trace values should be finite"
    );
}

/// Tests that ellipsis and explicit notation produce identical results for trace operation.
#[test]
fn test_ellipsis_trace_equivalence() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (0..18).map(|i| (i + 1) as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3, 3], col).unwrap();

    let result_ellipsis =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ii->...", &[&a], None).unwrap();

    let result_explicit =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "bii->b", &[&a], None).unwrap();

    assert_eq!(result_ellipsis.dims(), result_explicit.dims());

    let ellipsis_data = result_ellipsis.buffer().as_slice().unwrap();
    let explicit_data = result_explicit.buffer().as_slice().unwrap();

    for (i, (&e, &x)) in ellipsis_data.iter().zip(explicit_data.iter()).enumerate() {
        assert!(
            (e - x).abs() < 1e-14,
            "Ellipsis trace[{}] = {} differs from explicit trace {}",
            i,
            e,
            x
        );
    }
}

/// Tests ellipsis notation with negative values to ensure correct handling of signed arithmetic.
#[test]
fn test_ellipsis_with_negative_values() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3], col).unwrap();

    let b_data: Vec<f64> = vec![7.0, -8.0, 9.0, -10.0, 11.0, -12.0];
    let b = Tensor::<f64>::from_slice(&b_data, &[3, 2], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 2]);

    let result_data = result.buffer().as_slice().unwrap();
    assert!(
        result_data.iter().all(|&v| v.is_finite()),
        "All result values should be finite with negative inputs"
    );
}

/// Tests ellipsis notation with three-input contraction (triple einsum).
/// Verifies that ellipsis correctly handles multiple batched inputs.
#[test]
fn test_ellipsis_three_input_contraction() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 2, 3], col).unwrap();

    let b_data: Vec<f64> = (0..12).map(|i| (i + 13) as f64).collect();
    let b = Tensor::<f64>::from_slice(&b_data, &[2, 3, 2], col).unwrap();

    let c_data: Vec<f64> = (0..8).map(|i| (i + 25) as f64).collect();
    let c = Tensor::<f64>::from_slice(&c_data, &[2, 2, 2], col).unwrap();

    let result = einsum::<Standard<f64>, CpuBackend>(
        &mut ctx,
        "...ij,...jk,...kl->...il",
        &[&a, &b, &c],
        None,
    )
    .unwrap();

    assert_eq!(result.dims(), &[2, 2, 2]);

    let result_data = result.buffer().as_slice().unwrap();
    assert!(
        result_data.iter().all(|&v| v.is_finite()),
        "All result values should be finite in three-input contraction"
    );
}

/// Tests ellipsis notation with scalar result (full contraction).
#[test]
fn test_ellipsis_full_contraction_scalar() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (1..=6).map(|i| i as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3], col).unwrap();

    let b_data: Vec<f64> = (1..=6).map(|i| i as f64).collect();
    let b = Tensor::<f64>::from_slice(&b_data, &[2, 3], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...ij->", &[&a, &b], None).unwrap();

    assert_eq!(result.dims(), &[]);

    let result_data = result.buffer().as_slice().unwrap();
    let expected: f64 = (1..=6).map(|i| (i * i) as f64).sum();
    assert!(
        (result_data[0] - expected).abs() < 1e-10,
        "Scalar result {} should equal expected {}",
        result_data[0],
        expected
    );
}

/// Tests ellipsis notation with a single batch element (batch size = 1).
/// Verifies that the implementation correctly handles degenerate batch dimensions.
#[test]
fn test_ellipsis_single_batch_element() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (1..=6).map(|i| i as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[1, 2, 3], col).unwrap();

    let b_data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let b = Tensor::<f64>::from_slice(&b_data, &[1, 3, 4], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[1, 2, 4]);

    let result_data = result.buffer().as_slice().unwrap();
    assert!(
        result_data.iter().all(|&v| v.is_finite()),
        "All result values should be finite with single batch element"
    );
}

/// Tests ellipsis notation with batched transpose operation.
/// Verifies that ellipsis works correctly for permutation operations across batches.
#[test]
fn test_ellipsis_batched_transpose() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3, 2], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij->...ji", &[&a], None).unwrap();

    assert_eq!(result.dims(), &[2, 2, 3]);

    let result_data = result.buffer().as_slice().unwrap();
    let original_data = a.buffer().as_slice().unwrap();

    for (i, &v) in result_data.iter().enumerate() {
        assert!(v.is_finite(), "Transpose result[{}] should be finite", i);
        assert!(
            original_data.contains(&v),
            "Transpose result[{}] = {} should exist in original data",
            i,
            v
        );
    }
}

/// Tests ellipsis notation with Hadamard (element-wise) product followed by batch reduction.
/// Verifies that ellipsis correctly handles element-wise operations with subsequent contraction.
#[test]
fn test_ellipsis_hadamard_with_reduction() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 2, 3], col).unwrap();

    let b_data: Vec<f64> = (1..=12).map(|i| (i * 2) as f64).collect();
    let b = Tensor::<f64>::from_slice(&b_data, &[2, 2, 3], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...ij->...", &[&a, &b], None).unwrap();

    assert_eq!(result.dims(), &[2]);

    let result_data = result.buffer().as_slice().unwrap();

    for (i, &v) in result_data.iter().enumerate() {
        assert!(v.is_finite(), "Hadamard reduction[{}] should be finite", i);
        assert!(
            v > 0.0,
            "Hadamard reduction[{}] = {} should be positive",
            i,
            v
        );
    }

    let result_explicit =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "bij,bij->b", &[&a, &b], None).unwrap();

    let explicit_data = result_explicit.buffer().as_slice().unwrap();

    for (i, (&e, &x)) in result_data.iter().zip(explicit_data.iter()).enumerate() {
        assert!(
            (e - x).abs() < 1e-14,
            "Ellipsis Hadamard[{}] = {} differs from explicit {}",
            i,
            e,
            x
        );
    }
}

/// Tests ellipsis notation with asymmetric batch dimensions.
/// Verifies that ellipsis correctly handles non-uniform batch structures.
#[test]
fn test_ellipsis_asymmetric_batch_shapes() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a_data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3, 4], col).unwrap();

    let b_data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
    let b = Tensor::<f64>::from_slice(&b_data, &[2, 3, 4], col).unwrap();

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...ij->...", &[&a, &b], None).unwrap();

    assert_eq!(result.dims(), &[2]);

    let result_data = result.buffer().as_slice().unwrap();

    for (i, &v) in result_data.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Asymmetric batch result[{}] should be finite",
            i
        );
        assert!(
            v > 0.0,
            "Asymmetric batch result[{}] = {} should be positive",
            i,
            v
        );
    }
}

/// Tests ellipsis notation with zero-sized contraction dimension.
/// Verifies correct handling when the contracted dimension has size 0 (edge case for #529).
#[test]
fn test_ellipsis_zero_contraction_dim() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 3, 0], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 0, 4], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None)
            .unwrap();

    assert_eq!(result.dims(), &[2, 3, 4]);

    let result_data = result.buffer().as_slice().unwrap();
    assert!(
        result_data.iter().all(|&v| v == 0.0),
        "All result values should be zero when contracting over dimension 0"
    );
}

/// Regression test for issue #529: Verifies that ellipsis notation does NOT produce
/// the original error "invalid einsum label character: '.' (U+002E)".
/// This confirms the ellipsis expansion is properly handled before label parsing.
#[test]
fn test_ellipsis_no_invalid_label_error() {
    let mut ctx = make_context();
    let col = MemoryOrder::ColumnMajor;

    let a = Tensor::<f64>::zeros(&[2, 2, 2], LogicalMemorySpace::MainMemory, col);
    let b = Tensor::<f64>::zeros(&[2, 2, 2], LogicalMemorySpace::MainMemory, col);

    let result =
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, "...ij,...jk->...ik", &[&a, &b], None);

    match &result {
        Ok(tensor) => {
            assert_eq!(tensor.dims(), &[2, 2, 2]);
        }
        Err(e) => {
            let error_msg = format!("{:?}", e);
            assert!(
                !error_msg.contains("invalid einsum label character"),
                "Ellipsis notation should not produce 'invalid einsum label character' error. Got: {}",
                error_msg
            );
            panic!("Unexpected error: {:?}", e);
        }
    }
}
