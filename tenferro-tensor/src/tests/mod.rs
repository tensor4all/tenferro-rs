use super::*;
use tenferro_device::LogicalMemorySpace;

mod buffer;

#[cfg(feature = "cuda")]
mod cuda;
mod organization;

#[test]
fn tensor_debug_is_summary_style() {
    let tensor = Tensor::<f32>::zeros(
        &[2, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );

    let dbg = format!("{:?}", tensor);
    assert!(dbg.contains("Tensor"));
    assert!(dbg.contains("f32"));
    assert!(dbg.contains("[2, 3]"));
    assert!(dbg.contains("logical_memory_space"));
    assert!(dbg.contains("is_contiguous"));
}

#[test]
fn eye_creates_identity_matrix_col_major() {
    let id = Tensor::<f64>::eye(3, LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    assert_eq!(id.dims(), &[3, 3]);

    let data = id.buffer().as_slice().unwrap();
    assert_eq!(data[0], 1.0);
    assert_eq!(data[4], 1.0);
    assert_eq!(data[8], 1.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    assert_eq!(data[3], 0.0);
}

#[test]
fn eye_creates_identity_matrix_row_major() {
    let id = Tensor::<f64>::eye(3, LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    assert_eq!(id.dims(), &[3, 3]);

    let data = id.buffer().as_slice().unwrap();
    assert_eq!(data[0], 1.0);
    assert_eq!(data[4], 1.0);
    assert_eq!(data[8], 1.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    assert_eq!(data[3], 0.0);
}

#[test]
fn tril_extracts_lower_triangular() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let a = Tensor::<f64>::from_slice(&data, &[3, 3], MemoryOrder::ColumnMajor).unwrap();
    let lower = a.tril(0);

    let expected = [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0];
    assert_eq!(lower.buffer().as_slice().unwrap(), expected);
}

#[test]
fn triu_extracts_upper_triangular() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let a = Tensor::<f64>::from_slice(&data, &[3, 3], MemoryOrder::ColumnMajor).unwrap();
    let upper = a.triu(0);

    let expected = [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0];
    assert_eq!(upper.buffer().as_slice().unwrap(), expected);
}

#[test]
fn tril_with_diagonal_offset() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let a = Tensor::<f64>::from_slice(&data, &[3, 3], MemoryOrder::ColumnMajor).unwrap();
    let lower = a.tril(1);

    let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0];
    assert_eq!(lower.buffer().as_slice().unwrap(), expected);
}

#[test]
fn triu_with_diagonal_offset() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let a = Tensor::<f64>::from_slice(&data, &[3, 3], MemoryOrder::ColumnMajor).unwrap();
    let upper = a.triu(-1);

    let expected = [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_eq!(upper.buffer().as_slice().unwrap(), expected);
}

#[test]
fn narrow_returns_subrange() {
    let t = Tensor::<f64>::zeros(
        &[2, 10],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let sub = t.narrow(1, 2, 3).unwrap();
    assert_eq!(sub.dims(), &[2, 3]);
}

#[test]
fn narrow_rejects_out_of_bounds() {
    let t = Tensor::<f64>::zeros(
        &[2, 10],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    assert!(t.narrow(1, 8, 5).is_err());
    assert!(t.narrow(1, 0, 15).is_err());
}

#[test]
fn select_returns_single_slice() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 4], MemoryOrder::ColumnMajor).unwrap();
    let slice = t.select(1, 1).unwrap();
    assert_eq!(slice.dims(), &[2]);
    let slice_data = slice.contiguous(MemoryOrder::ColumnMajor);
    assert_eq!(slice_data.buffer().as_slice().unwrap(), &[3.0, 4.0]);
}
