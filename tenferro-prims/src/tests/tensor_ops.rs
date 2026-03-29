use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::tensor_ops;
use crate::CpuContext;

fn make_ctx() -> CpuContext {
    CpuContext::new(1)
}

fn mem() -> LogicalMemorySpace {
    LogicalMemorySpace::MainMemory
}

fn col() -> MemoryOrder {
    MemoryOrder::ColumnMajor
}

// ---------------------------------------------------------------------------
// triu
// ---------------------------------------------------------------------------

#[test]
fn triu_3x3_default_diagonal() {
    let mut ctx = make_ctx();
    // 3x3 matrix of ones
    let a = Tensor::<f64>::ones(&[3, 3], mem(), col()).unwrap();
    let result = tensor_ops::triu(&mut ctx, &a, 0).unwrap();
    assert_eq!(result.dims(), &[3, 3]);
    let data = result.buffer().as_slice().unwrap();
    // Column-major: col0=[1,0,0], col1=[1,1,0], col2=[1,1,1]
    assert_eq!(data, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn triu_3x3_diagonal_plus1() {
    let mut ctx = make_ctx();
    let a = Tensor::<f64>::ones(&[3, 3], mem(), col()).unwrap();
    let result = tensor_ops::triu(&mut ctx, &a, 1).unwrap();
    let data = result.buffer().as_slice().unwrap();
    // triu with diagonal=1: strictly above main diagonal
    // col-major: col0=[0,0,0], col1=[1,0,0], col2=[1,1,0]
    assert_eq!(data, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn triu_3x3_diagonal_minus1() {
    let mut ctx = make_ctx();
    let a = Tensor::<f64>::ones(&[3, 3], mem(), col()).unwrap();
    let result = tensor_ops::triu(&mut ctx, &a, -1).unwrap();
    let data = result.buffer().as_slice().unwrap();
    // triu with diagonal=-1: main diagonal + one below
    // col-major: col0=[1,1,0], col1=[1,1,1], col2=[1,1,1]
    assert_eq!(data, &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// tril
// ---------------------------------------------------------------------------

#[test]
fn tril_3x3_default_diagonal() {
    let mut ctx = make_ctx();
    let a = Tensor::<f64>::ones(&[3, 3], mem(), col()).unwrap();
    let result = tensor_ops::tril(&mut ctx, &a, 0).unwrap();
    assert_eq!(result.dims(), &[3, 3]);
    let data = result.buffer().as_slice().unwrap();
    // col-major: col0=[1,1,1], col1=[0,1,1], col2=[0,0,1]
    assert_eq!(data, &[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn tril_3x3_diagonal_plus1() {
    let mut ctx = make_ctx();
    let a = Tensor::<f64>::ones(&[3, 3], mem(), col()).unwrap();
    let result = tensor_ops::tril(&mut ctx, &a, 1).unwrap();
    let data = result.buffer().as_slice().unwrap();
    // tril with diagonal=1: main diagonal + one above
    // col-major: col0=[1,1,1], col1=[1,1,1], col2=[0,1,1]
    assert_eq!(data, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// triu/tril on batched matrix
// ---------------------------------------------------------------------------

#[test]
fn triu_batched_3x3x2() {
    let mut ctx = make_ctx();
    // 3x3 batch of 2, all ones
    let a = Tensor::<f64>::ones(&[3, 3, 2], mem(), col()).unwrap();
    let result = tensor_ops::triu(&mut ctx, &a, 0).unwrap();
    assert_eq!(result.dims(), &[3, 3, 2]);
    let data = result.buffer().as_slice().unwrap();
    // Each batch slice should be the same triu pattern
    // Batch 0: col0=[1,0,0], col1=[1,1,0], col2=[1,1,1] (9 elements)
    // Batch 1: same
    let expected_batch = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    assert_eq!(&data[..9], &expected_batch);
    assert_eq!(&data[9..18], &expected_batch);
}

#[test]
fn tril_batched_3x3x2() {
    let mut ctx = make_ctx();
    let a = Tensor::<f64>::ones(&[3, 3, 2], mem(), col()).unwrap();
    let result = tensor_ops::tril(&mut ctx, &a, 0).unwrap();
    assert_eq!(result.dims(), &[3, 3, 2]);
    let data = result.buffer().as_slice().unwrap();
    let expected_batch = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    assert_eq!(&data[..9], &expected_batch);
    assert_eq!(&data[9..18], &expected_batch);
}

// ---------------------------------------------------------------------------
// cat
// ---------------------------------------------------------------------------

#[test]
fn cat_along_axis_0() {
    let mut ctx = make_ctx();
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], col()).unwrap();
    let b = Tensor::from_slice(&[5.0_f64, 6.0, 7.0, 8.0], &[2, 2], col()).unwrap();
    let result = tensor_ops::cat(&mut ctx, &[&a, &b], 0).unwrap();
    assert_eq!(result.dims(), &[4, 2]);
}

#[test]
fn cat_along_axis_1() {
    let mut ctx = make_ctx();
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], col()).unwrap();
    let b = Tensor::from_slice(&[5.0_f64, 6.0, 7.0, 8.0], &[2, 2], col()).unwrap();
    let result = tensor_ops::cat(&mut ctx, &[&a, &b], 1).unwrap();
    assert_eq!(result.dims(), &[2, 4]);
}

// ---------------------------------------------------------------------------
// stack
// ---------------------------------------------------------------------------

#[test]
fn stack_along_new_axis_0() {
    let mut ctx = make_ctx();
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], col()).unwrap();
    let b = Tensor::from_slice(&[4.0_f64, 5.0, 6.0], &[3], col()).unwrap();
    let result = tensor_ops::stack(&mut ctx, &[&a, &b], 0).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
}

#[test]
fn stack_along_new_axis_1() {
    let mut ctx = make_ctx();
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], col()).unwrap();
    let b = Tensor::from_slice(&[4.0_f64, 5.0, 6.0], &[3], col()).unwrap();
    let result = tensor_ops::stack(&mut ctx, &[&a, &b], 1).unwrap();
    assert_eq!(result.dims(), &[3, 2]);
}

// ---------------------------------------------------------------------------
// error cases
// ---------------------------------------------------------------------------

#[test]
fn triu_rejects_rank_1() {
    let mut ctx = make_ctx();
    let a = Tensor::<f64>::ones(&[5], mem(), col()).unwrap();
    let err = tensor_ops::triu(&mut ctx, &a, 0).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn tril_rejects_rank_1() {
    let mut ctx = make_ctx();
    let a = Tensor::<f64>::ones(&[5], mem(), col()).unwrap();
    let err = tensor_ops::tril(&mut ctx, &a, 0).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

// ---------------------------------------------------------------------------
// cat data values
// ---------------------------------------------------------------------------

#[test]
fn cat_axis_0_data_values() {
    let mut ctx = make_ctx();
    // col-major 2x2: data stored as [col0_row0, col0_row1, col1_row0, col1_row1]
    // a = [[1,3],[2,4]] stored as [1,2,3,4]
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], col()).unwrap();
    // b = [[5,7],[6,8]] stored as [5,6,7,8]
    let b = Tensor::from_slice(&[5.0_f64, 6.0, 7.0, 8.0], &[2, 2], col()).unwrap();
    let result = tensor_ops::cat(&mut ctx, &[&a, &b], 0).unwrap();
    assert_eq!(result.dims(), &[4, 2]);
    let data = result.buffer().as_slice().unwrap();
    // Result 4x2 col-major: col0=[1,2,5,6], col1=[3,4,7,8]
    assert_eq!(data, &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn stack_axis_0_data_values() {
    let mut ctx = make_ctx();
    let a = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], col()).unwrap();
    let b = Tensor::from_slice(&[4.0_f64, 5.0, 6.0], &[3], col()).unwrap();
    let result = tensor_ops::stack(&mut ctx, &[&a, &b], 0).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let data = result.buffer().as_slice().unwrap();
    // 2x3 col-major: col0=[1,4], col1=[2,5], col2=[3,6]
    assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
