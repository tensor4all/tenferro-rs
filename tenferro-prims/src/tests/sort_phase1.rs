use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{CpuBackend, CpuContext, SortPrimsDescriptor, TensorSortPrims};

const CPU: LogicalMemorySpace = LogicalMemorySpace::MainMemory;
const COL: MemoryOrder = MemoryOrder::ColumnMajor;

fn f64_tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, COL).unwrap()
}

fn val_data(t: &Tensor<f64>) -> Vec<f64> {
    let c = t.contiguous(COL);
    c.buffer().as_slice().unwrap().to_vec()
}

fn idx_data(t: &Tensor<i64>) -> Vec<i64> {
    let c = t.contiguous(COL);
    c.buffer().as_slice().unwrap().to_vec()
}

// ============================================================================
// Sort ascending 1D
// ============================================================================

#[test]
fn sort_ascending_1d() {
    let input = f64_tensor(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);

    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Sort {
        axis: 0,
        descending: false,
        stable: true,
    };
    let plan =
        <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[5]]).unwrap();

    let mut values = Tensor::<f64>::zeros(&[5], CPU, COL).unwrap();
    let mut indices = Tensor::<i64>::zeros(&[5], CPU, COL).unwrap();

    <CpuBackend as TensorSortPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &input,
        &mut values,
        &mut indices,
    )
    .unwrap();

    assert_eq!(val_data(&values), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    // Stable sort: first 1.0 (index 1) comes before second 1.0 (index 3)
    assert_eq!(idx_data(&indices), vec![1, 3, 0, 2, 4]);
}

// ============================================================================
// Sort descending 1D
// ============================================================================

#[test]
fn sort_descending_1d() {
    let input = f64_tensor(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);

    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Sort {
        axis: 0,
        descending: true,
        stable: true,
    };
    let plan =
        <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[5]]).unwrap();

    let mut values = Tensor::<f64>::zeros(&[5], CPU, COL).unwrap();
    let mut indices = Tensor::<i64>::zeros(&[5], CPU, COL).unwrap();

    <CpuBackend as TensorSortPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &input,
        &mut values,
        &mut indices,
    )
    .unwrap();

    assert_eq!(val_data(&values), vec![5.0, 4.0, 3.0, 1.0, 1.0]);
    assert_eq!(idx_data(&indices), vec![4, 2, 0, 1, 3]);
}

// ============================================================================
// Sort along axis 0 of a matrix
// ============================================================================

#[test]
fn sort_along_axis0_matrix() {
    // Matrix [3, 2] column-major: [3, 1, 2, 6, 4, 5]
    // Logical matrix (rows x cols):
    //   [[3, 6],
    //    [1, 4],
    //    [2, 5]]
    // Sort along axis 0 (sort each column independently, ascending):
    //   col0: [3, 1, 2] -> [1, 2, 3] (indices: [1, 2, 0])
    //   col1: [6, 4, 5] -> [4, 5, 6] (indices: [1, 2, 0])
    // col-major values: [1, 2, 3, 4, 5, 6]
    // col-major indices: [1, 2, 0, 1, 2, 0]
    let input = f64_tensor(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[3, 2]);

    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Sort {
        axis: 0,
        descending: false,
        stable: true,
    };
    let plan =
        <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[3, 2]]).unwrap();

    let mut values = Tensor::<f64>::zeros(&[3, 2], CPU, COL).unwrap();
    let mut indices = Tensor::<i64>::zeros(&[3, 2], CPU, COL).unwrap();

    <CpuBackend as TensorSortPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &input,
        &mut values,
        &mut indices,
    )
    .unwrap();

    assert_eq!(val_data(&values), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(idx_data(&indices), vec![1, 2, 0, 1, 2, 0]);
}

// ============================================================================
// Argsort returns correct permutation
// ============================================================================

#[test]
fn argsort_returns_correct_permutation() {
    let input = f64_tensor(&[30.0, 10.0, 20.0], &[3]);

    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Argsort {
        axis: 0,
        descending: false,
        stable: true,
    };
    let plan =
        <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[3]]).unwrap();

    // values_out should remain unmodified for argsort
    let mut values = Tensor::<f64>::zeros(&[3], CPU, COL).unwrap();
    let mut indices = Tensor::<i64>::zeros(&[3], CPU, COL).unwrap();

    <CpuBackend as TensorSortPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &input,
        &mut values,
        &mut indices,
    )
    .unwrap();

    // Sorted order: 10.0 (idx 1), 20.0 (idx 2), 30.0 (idx 0)
    assert_eq!(idx_data(&indices), vec![1, 2, 0]);
    // values_out should remain zeros (argsort does not write values)
    assert_eq!(val_data(&values), vec![0.0, 0.0, 0.0]);
}

// ============================================================================
// Topk with k=2 from 5 elements (largest)
// ============================================================================

#[test]
fn topk_largest_k2() {
    let input = f64_tensor(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);

    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Topk {
        axis: 0,
        k: 2,
        largest: true,
        sorted: true,
    };
    let plan =
        <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[5]]).unwrap();

    let mut values = Tensor::<f64>::zeros(&[2], CPU, COL).unwrap();
    let mut indices = Tensor::<i64>::zeros(&[2], CPU, COL).unwrap();

    <CpuBackend as TensorSortPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &input,
        &mut values,
        &mut indices,
    )
    .unwrap();

    assert_eq!(val_data(&values), vec![5.0, 4.0]);
    assert_eq!(idx_data(&indices), vec![4, 2]);
    assert_eq!(values.dims(), &[2]);
    assert_eq!(indices.dims(), &[2]);
}

// ============================================================================
// Topk smallest (largest=false)
// ============================================================================

#[test]
fn topk_smallest_k2() {
    let input = f64_tensor(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);

    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Topk {
        axis: 0,
        k: 2,
        largest: false,
        sorted: true,
    };
    let plan =
        <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[5]]).unwrap();

    let mut values = Tensor::<f64>::zeros(&[2], CPU, COL).unwrap();
    let mut indices = Tensor::<i64>::zeros(&[2], CPU, COL).unwrap();

    <CpuBackend as TensorSortPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &input,
        &mut values,
        &mut indices,
    )
    .unwrap();

    // Smallest 2: 1.0 (idx 1), 1.0 (idx 3)
    assert_eq!(val_data(&values), vec![1.0, 1.0]);
    assert_eq!(idx_data(&indices), vec![1, 3]);
}

// ============================================================================
// has_sort_support returns true for all variants
// ============================================================================

#[test]
fn cpu_backend_supports_all_sort_ops() {
    assert!(
        <CpuBackend as TensorSortPrims<Standard<f64>>>::has_sort_support(
            &SortPrimsDescriptor::Sort {
                axis: 0,
                descending: false,
                stable: true,
            }
        )
    );
    assert!(
        <CpuBackend as TensorSortPrims<Standard<f64>>>::has_sort_support(
            &SortPrimsDescriptor::Argsort {
                axis: 0,
                descending: false,
                stable: true,
            }
        )
    );
    assert!(
        <CpuBackend as TensorSortPrims<Standard<f64>>>::has_sort_support(
            &SortPrimsDescriptor::Topk {
                axis: 0,
                k: 3,
                largest: true,
                sorted: true,
            }
        )
    );
}

// ============================================================================
// Plan validation
// ============================================================================

#[test]
fn plan_rejects_wrong_number_of_shapes() {
    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Sort {
        axis: 0,
        descending: false,
        stable: true,
    };
    let result = <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3], &[3]], // 2 shapes, expect 1
    );
    assert!(result.is_err());
}

#[test]
fn sort_axis_out_of_bounds_returns_error() {
    let input = f64_tensor(&[3.0, 1.0, 2.0], &[3]);

    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Sort {
        axis: 1, // only axis 0 valid for 1D
        descending: false,
        stable: true,
    };
    let plan =
        <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[3]]).unwrap();

    let mut values = Tensor::<f64>::zeros(&[3], CPU, COL).unwrap();
    let mut indices = Tensor::<i64>::zeros(&[3], CPU, COL).unwrap();

    let result = <CpuBackend as TensorSortPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &input,
        &mut values,
        &mut indices,
    );
    assert!(result.is_err());
}

#[test]
fn topk_k_exceeds_axis_size_returns_error() {
    let input = f64_tensor(&[1.0, 2.0, 3.0], &[3]);

    let mut ctx = CpuContext::new(1);
    let desc = SortPrimsDescriptor::Topk {
        axis: 0,
        k: 5, // axis size is 3
        largest: true,
        sorted: true,
    };
    let plan =
        <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[3]]).unwrap();

    let mut values = Tensor::<f64>::zeros(&[5], CPU, COL).unwrap();
    let mut indices = Tensor::<i64>::zeros(&[5], CPU, COL).unwrap();

    let result = <CpuBackend as TensorSortPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &input,
        &mut values,
        &mut indices,
    );
    assert!(result.is_err());
}
