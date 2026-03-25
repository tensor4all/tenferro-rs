use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    CpuBackend, CpuContext, IndexingPrimsDescriptor, ScatterReduction, TensorIndexingPrims,
};

const CPU: LogicalMemorySpace = LogicalMemorySpace::MainMemory;
const COL: MemoryOrder = MemoryOrder::ColumnMajor;

fn f64_tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, COL).unwrap()
}

fn i64_tensor(data: &[i64], dims: &[usize]) -> Tensor<i64> {
    Tensor::from_slice(data, dims, COL).unwrap()
}

fn out_data(t: &Tensor<f64>) -> Vec<f64> {
    let c = t.contiguous(COL);
    c.buffer().as_slice().unwrap().to_vec()
}

// ============================================================================
// IndexSelect
// ============================================================================

#[test]
fn index_select_rows_from_matrix() {
    // source: [3, 2] column-major:
    //   col0: [1.0, 2.0, 3.0], col1: [4.0, 5.0, 6.0]
    // logical matrix:
    //   [[1, 4],
    //    [2, 5],
    //    [3, 6]]
    // Select rows 0 and 2 along axis 0 -> [2, 2]
    let source = f64_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let indices = i64_tensor(&[0, 2], &[2]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::IndexSelect { axis: 0 };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3, 2], &[2], &[2, 2]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[2, 2], CPU, COL).unwrap();
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    )
    .unwrap();

    // Expected col-major: [1.0, 3.0, 4.0, 6.0]
    // Row 0: [1, 4], Row 2: [3, 6]
    assert_eq!(out_data(&output), vec![1.0, 3.0, 4.0, 6.0]);
    assert_eq!(output.dims(), &[2, 2]);
}

#[test]
fn index_select_columns_from_matrix() {
    // source: [3, 2] column-major: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    // Select column 1 along axis 1 -> [3, 1]
    let source = f64_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let indices = i64_tensor(&[1], &[1]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::IndexSelect { axis: 1 };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3, 2], &[1], &[3, 1]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[3, 1], CPU, COL).unwrap();
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    )
    .unwrap();

    // Column 1 data: [4.0, 5.0, 6.0]
    assert_eq!(out_data(&output), vec![4.0, 5.0, 6.0]);
    assert_eq!(output.dims(), &[3, 1]);
}

#[test]
fn index_select_single_element() {
    let source = f64_tensor(&[10.0, 20.0, 30.0], &[3]);
    let indices = i64_tensor(&[1], &[1]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::IndexSelect { axis: 0 };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3], &[1], &[1]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[1], CPU, COL).unwrap();
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    )
    .unwrap();

    assert_eq!(out_data(&output), vec![20.0]);
}

#[test]
fn index_select_empty_indices() {
    let source = f64_tensor(&[1.0, 2.0, 3.0], &[3]);
    let indices = i64_tensor(&[], &[0]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::IndexSelect { axis: 0 };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3], &[0], &[0]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[0], CPU, COL).unwrap();
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    )
    .unwrap();

    assert_eq!(output.dims(), &[0]);
}

// ============================================================================
// Gather
// ============================================================================

#[test]
fn gather_along_axis0() {
    // source [3, 2] col-major: [1, 2, 3, 4, 5, 6]
    // Logical:
    //   [[1, 4],
    //    [2, 5],
    //    [3, 6]]
    // indices [2, 2] col-major: [0, 2, 1, 0]
    // Logical indices:
    //   [[0, 1],
    //    [2, 0]]
    // Gather along axis 0:
    //   out[i, j] = src[idx[i, j], j]
    //   out[0,0] = src[0,0] = 1
    //   out[1,0] = src[2,0] = 3
    //   out[0,1] = src[1,1] = 5
    //   out[1,1] = src[0,1] = 4
    // col-major out: [1, 3, 5, 4]
    let source = f64_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let indices = i64_tensor(&[0, 2, 1, 0], &[2, 2]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::Gather { axis: 0 };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3, 2], &[2, 2], &[2, 2]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[2, 2], CPU, COL).unwrap();
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    )
    .unwrap();

    assert_eq!(out_data(&output), vec![1.0, 3.0, 5.0, 4.0]);
}

#[test]
fn gather_along_axis1() {
    // source [2, 3] col-major: [1, 2, 3, 4, 5, 6]
    // Logical:
    //   [[1, 3, 5],
    //    [2, 4, 6]]
    // indices [2, 2] col-major: [2, 0, 1, 1]
    // Logical indices:
    //   [[2, 1],
    //    [0, 1]]
    // Gather along axis 1:
    //   out[i, j] = src[i, idx[i,j]]
    //   out[0,0] = src[0,2] = 5
    //   out[1,0] = src[1,0] = 2
    //   out[0,1] = src[0,1] = 3
    //   out[1,1] = src[1,1] = 4
    // col-major out: [5, 2, 3, 4]
    let source = f64_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let indices = i64_tensor(&[2, 0, 1, 1], &[2, 2]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::Gather { axis: 1 };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 2], &[2, 2]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[2, 2], CPU, COL).unwrap();
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    )
    .unwrap();

    assert_eq!(out_data(&output), vec![5.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// Scatter
// ============================================================================

#[test]
fn scatter_along_axis0() {
    // source [2, 2] col-major: [10, 20, 30, 40]
    // indices [2, 2] col-major: [0, 2, 1, 0]
    // output [3, 2] initially zeros
    //
    // Scatter along axis 0:
    //   src[0,0]=10 -> out[0,0] = 10
    //   src[1,0]=20 -> out[2,0] = 20
    //   src[0,1]=30 -> out[1,1] = 30
    //   src[1,1]=40 -> out[0,1] = 40
    // col-major out: [10, 0, 20, 40, 30, 0]
    let source = f64_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let indices = i64_tensor(&[0, 2, 1, 0], &[2, 2]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::Scatter {
        axis: 0,
        reduction: ScatterReduction::None,
    };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 2], &[2, 2], &[3, 2]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[3, 2], CPU, COL).unwrap();
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    )
    .unwrap();

    assert_eq!(out_data(&output), vec![10.0, 0.0, 20.0, 40.0, 30.0, 0.0]);
}

#[test]
fn scatter_add_accumulates() {
    // source [3] col-major: [1.0, 2.0, 3.0]
    // indices [3]: [0, 0, 1]
    // output [2] initially [10.0, 20.0]
    //
    // ScatterAdd along axis 0:
    //   out[0] += 1.0 -> 11.0
    //   out[0] += 2.0 -> 13.0
    //   out[1] += 3.0 -> 23.0
    let source = f64_tensor(&[1.0, 2.0, 3.0], &[3]);
    let indices = i64_tensor(&[0, 0, 1], &[3]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::Scatter {
        axis: 0,
        reduction: ScatterReduction::Add,
    };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3], &[3], &[2]],
    )
    .unwrap();

    let mut output = f64_tensor(&[10.0, 20.0], &[2]);
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    )
    .unwrap();

    assert_eq!(out_data(&output), vec![13.0, 23.0]);
}

// ============================================================================
// IndexPut
// ============================================================================

#[test]
fn index_put_overwrites() {
    // values [2]: [100.0, 200.0]
    // indices [2]: [1, 3]
    // output [4] initially [0, 0, 0, 0]
    let values = f64_tensor(&[100.0, 200.0], &[2]);
    let indices = i64_tensor(&[1, 3], &[2]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::IndexPut { accumulate: false };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2], &[2], &[4]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[4], CPU, COL).unwrap();
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&values],
        &indices,
        &mut output,
    )
    .unwrap();

    assert_eq!(out_data(&output), vec![0.0, 100.0, 0.0, 200.0]);
}

#[test]
fn index_put_accumulates() {
    // values [2]: [5.0, 7.0]
    // indices [2]: [0, 0]
    // output [2] initially [10.0, 20.0]
    let values = f64_tensor(&[5.0, 7.0], &[2]);
    let indices = i64_tensor(&[0, 0], &[2]);

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::IndexPut { accumulate: true };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2], &[2], &[2]],
    )
    .unwrap();

    let mut output = f64_tensor(&[10.0, 20.0], &[2]);
    <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&values],
        &indices,
        &mut output,
    )
    .unwrap();

    // out[0] = 10 + 5 + 7 = 22, out[1] = 20
    assert_eq!(out_data(&output), vec![22.0, 20.0]);
}

// ============================================================================
// has_indexing_support
// ============================================================================

#[test]
fn cpu_backend_supports_all_indexing_ops() {
    assert!(
        <CpuBackend as TensorIndexingPrims<Standard<f64>>>::has_indexing_support(
            IndexingPrimsDescriptor::IndexSelect { axis: 0 }
        )
    );
    assert!(
        <CpuBackend as TensorIndexingPrims<Standard<f64>>>::has_indexing_support(
            IndexingPrimsDescriptor::Gather { axis: 0 }
        )
    );
    assert!(
        <CpuBackend as TensorIndexingPrims<Standard<f64>>>::has_indexing_support(
            IndexingPrimsDescriptor::Scatter {
                axis: 0,
                reduction: ScatterReduction::None,
            }
        )
    );
    assert!(
        <CpuBackend as TensorIndexingPrims<Standard<f64>>>::has_indexing_support(
            IndexingPrimsDescriptor::Scatter {
                axis: 0,
                reduction: ScatterReduction::Add,
            }
        )
    );
    assert!(
        <CpuBackend as TensorIndexingPrims<Standard<f64>>>::has_indexing_support(
            IndexingPrimsDescriptor::IndexPut { accumulate: false }
        )
    );
    assert!(
        <CpuBackend as TensorIndexingPrims<Standard<f64>>>::has_indexing_support(
            IndexingPrimsDescriptor::IndexPut { accumulate: true }
        )
    );
}

// ============================================================================
// Edge cases: out-of-bounds
// ============================================================================

#[test]
fn index_select_out_of_bounds_returns_error() {
    let source = f64_tensor(&[1.0, 2.0, 3.0], &[3]);
    let indices = i64_tensor(&[5], &[1]); // out of bounds

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::IndexSelect { axis: 0 };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3], &[1], &[1]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[1], CPU, COL).unwrap();
    let result = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    );
    assert!(result.is_err());
}

#[test]
fn gather_out_of_bounds_returns_error() {
    let source = f64_tensor(&[1.0, 2.0], &[2]);
    let indices = i64_tensor(&[3], &[1]); // out of bounds

    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::Gather { axis: 0 };
    let plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2], &[1], &[1]],
    )
    .unwrap();

    let mut output = Tensor::<f64>::zeros(&[1], CPU, COL).unwrap();
    let result = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &[&source],
        &indices,
        &mut output,
    );
    assert!(result.is_err());
}

// ============================================================================
// Plan validation
// ============================================================================

#[test]
fn plan_rejects_wrong_number_of_shapes() {
    let mut ctx = CpuContext::new(1);
    let desc = IndexingPrimsDescriptor::IndexSelect { axis: 0 };
    let result = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[3], &[1]], // only 2, need 3
    );
    assert!(result.is_err());
}
