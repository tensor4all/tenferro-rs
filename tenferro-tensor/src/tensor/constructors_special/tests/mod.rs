use super::*;
use tenferro_device::LogicalMemorySpace;

const CPU: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

#[test]
fn eye_covers_diagonal_fill_for_both_memory_orders() {
    let col = Tensor::<f64>::eye(3, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(col.dims(), &[3, 3]);
    assert_eq!(col.strides(), &[1, 3]);
    assert_eq!(
        col.buffer().as_slice().unwrap(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );

    let row = Tensor::<f64>::eye(3, CPU, MemoryOrder::RowMajor).unwrap();
    assert_eq!(row.dims(), &[3, 3]);
    assert_eq!(row.strides(), &[3, 1]);
    assert_eq!(
        row.buffer().as_slice().unwrap(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn arange_covers_positive_negative_and_empty_ranges() {
    let ascending = Tensor::<f64>::arange(0.0, 5.0, 1.5, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(ascending.dims(), &[4]);
    assert_eq!(
        ascending.buffer().as_slice().unwrap(),
        &[0.0, 1.5, 3.0, 4.5]
    );

    let descending = Tensor::<f64>::arange(5.0, -1.0, -2.0, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(descending.dims(), &[3]);
    assert_eq!(descending.buffer().as_slice().unwrap(), &[5.0, 3.0, 1.0]);

    let empty = Tensor::<f64>::arange(-1.0, 5.0, -1.0, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(empty.dims(), &[0]);
    assert!(empty.buffer().as_slice().unwrap().is_empty());
}

#[test]
fn linspace_covers_zero_one_many_and_end_override_paths() {
    let empty = Tensor::<f64>::linspace(0.0, 1.0, 0, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(empty.dims(), &[0]);
    assert!(empty.buffer().as_slice().unwrap().is_empty());

    let singleton = Tensor::<f64>::linspace(4.0, 10.0, 1, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(singleton.dims(), &[1]);
    assert_eq!(singleton.buffer().as_slice().unwrap(), &[4.0]);

    let many = Tensor::<f64>::linspace(0.0, 1.0, 3, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(many.dims(), &[3]);
    assert_eq!(many.buffer().as_slice().unwrap(), &[0.0, 0.5, 1.0]);
}
