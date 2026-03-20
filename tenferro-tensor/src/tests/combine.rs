use super::*;
use tenferro_device::LogicalMemorySpace;

const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;
const COL: MemoryOrder = MemoryOrder::ColumnMajor;

fn make_tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, COL).unwrap()
}

#[test]
fn stack_basic_2d() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = make_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]);

    let stacked = Tensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(stacked.dims(), &[2, 2, 3]);

    let data = stacked.buffer().as_slice().unwrap();
    assert_eq!(stacked.strides(), &[1, 2, 4]);
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 7.0);
}

#[test]
fn stack_at_end_dim() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let stacked = Tensor::stack(&[&a, &b], 2).unwrap();
    assert_eq!(stacked.dims(), &[2, 2, 2]);
}

#[test]
fn stack_negative_dim() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let stacked = Tensor::stack(&[&a, &b], -1).unwrap();
    assert_eq!(stacked.dims(), &[2, 2, 2]);

    let stacked2 = Tensor::stack(&[&a, &b], -3).unwrap();
    assert_eq!(stacked2.dims(), &[2, 2, 2]);
}

#[test]
fn stack_single_tensor() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let stacked = Tensor::stack(&[&a], 0).unwrap();
    assert_eq!(stacked.dims(), &[1, 2, 2]);
}

#[test]
fn stack_empty_tensors_error() {
    let result: Result<Tensor<f64>, _> = Tensor::stack(&[], 0);
    assert!(result.is_err());
}

#[test]
fn stack_shape_mismatch_error() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let result = Tensor::stack(&[&a, &b], 0);
    assert!(matches!(
        result,
        Err(tenferro_device::Error::ShapeMismatch { .. })
    ));
}

#[test]
fn stack_dim_out_of_range() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let result = Tensor::stack(&[&a, &b], 3);
    assert!(result.is_err());
}

#[test]
fn stack_negative_dim_out_of_range() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let result = Tensor::stack(&[&a, &b], -4);
    assert!(result.is_err());
}

#[test]
fn stack_rank0_tensor() {
    let a = Tensor::<f64>::zeros(&[], MEM, COL);
    let b = Tensor::<f64>::zeros(&[], MEM, COL);

    let stacked = Tensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(stacked.dims(), &[2]);
}

#[test]
fn stack_empty_dim_tensor() {
    let a = Tensor::<f64>::zeros(&[0, 3], MEM, COL);
    let b = Tensor::<f64>::zeros(&[0, 3], MEM, COL);

    let stacked = Tensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(stacked.dims(), &[2, 0, 3]);
}

#[test]
fn cat_basic_2d_dim0() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = make_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]);

    let concatenated = Tensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(concatenated.dims(), &[4, 3]);

    let data = concatenated.buffer().as_slice().unwrap();
    assert_eq!(concatenated.strides(), &[1, 4]);
    assert_eq!(data[0], 1.0);
    assert_eq!(data[2], 7.0);
}

#[test]
fn cat_basic_2d_dim1() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let concatenated = Tensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(concatenated.dims(), &[2, 4]);
}

#[test]
fn cat_negative_dim() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let concatenated = Tensor::cat(&[&a, &b], -1).unwrap();
    assert_eq!(concatenated.dims(), &[2, 4]);

    let concatenated2 = Tensor::cat(&[&a, &b], -2).unwrap();
    assert_eq!(concatenated2.dims(), &[4, 2]);
}

#[test]
fn cat_single_tensor() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let concatenated = Tensor::cat(&[&a], 0).unwrap();
    assert_eq!(concatenated.dims(), &[2, 2]);
}

#[test]
fn cat_empty_tensors_error() {
    let result: Result<Tensor<f64>, _> = Tensor::cat(&[], 0);
    assert!(result.is_err());
}

#[test]
fn cat_rank0_error() {
    let a = Tensor::<f64>::zeros(&[], MEM, COL);
    let b = Tensor::<f64>::zeros(&[], MEM, COL);

    let result = Tensor::cat(&[&a, &b], 0);
    assert!(result.is_err());
}

#[test]
fn cat_rank_mismatch_error() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let result = Tensor::cat(&[&a, &b], 0);
    assert!(result.is_err());
}

#[test]
fn cat_dim_out_of_range() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let result = Tensor::cat(&[&a, &b], 2);
    assert!(result.is_err());
}

#[test]
fn cat_negative_dim_out_of_range() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let result = Tensor::cat(&[&a, &b], -3);
    assert!(result.is_err());
}

#[test]
fn cat_non_concat_dim_mismatch_error() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let result = Tensor::cat(&[&a, &b], 0);
    assert!(matches!(
        result,
        Err(tenferro_device::Error::ShapeMismatch { .. })
    ));
}

#[test]
fn cat_empty_dim_tensor() {
    let a = Tensor::<f64>::zeros(&[0, 3], MEM, COL);
    let b = Tensor::<f64>::zeros(&[2, 3], MEM, COL);

    let concatenated = Tensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(concatenated.dims(), &[2, 3]);
}

#[test]
fn cat_3d_tensors() {
    let a = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);
    let b = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);

    let concatenated = Tensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(concatenated.dims(), &[2, 6, 4]);
}

#[test]
fn stack_three_tensors() {
    let a = make_tensor(&[1.0, 2.0], &[2]);
    let b = make_tensor(&[3.0, 4.0], &[2]);
    let c = make_tensor(&[5.0, 6.0], &[2]);

    let stacked = Tensor::stack(&[&a, &b, &c], 0).unwrap();
    assert_eq!(stacked.dims(), &[3, 2]);
}

#[test]
fn cat_three_tensors() {
    let a = make_tensor(&[1.0, 2.0], &[2]);
    let b = make_tensor(&[3.0, 4.0], &[2]);
    let c = make_tensor(&[5.0, 6.0], &[2]);

    let concatenated = Tensor::cat(&[&a, &b, &c], 0).unwrap();
    assert_eq!(concatenated.dims(), &[6]);
}
