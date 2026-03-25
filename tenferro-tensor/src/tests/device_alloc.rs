use tenferro_device::LogicalMemorySpace;

use crate::{DataBuffer, MemoryOrder, Tensor};

#[test]
fn zeros_on_device_cpu_returns_zero_filled_buffer() {
    let buf = DataBuffer::<f64>::zeros_on_device(5, LogicalMemorySpace::MainMemory).unwrap();
    assert_eq!(buf.len(), 5);
    assert!(buf.is_owned());
    assert_eq!(buf.as_slice().unwrap(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn zeros_on_device_cpu_empty() {
    let buf = DataBuffer::<f64>::zeros_on_device(0, LogicalMemorySpace::MainMemory).unwrap();
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

#[test]
fn allocate_uninit_on_device_cpu_returns_correct_length() {
    let buf =
        DataBuffer::<f64>::allocate_uninit_on_device(7, LogicalMemorySpace::MainMemory).unwrap();
    assert_eq!(buf.len(), 7);
    assert!(buf.is_owned());
    // Content is unspecified, but the buffer should be accessible
    assert!(buf.as_slice().is_some());
}

#[test]
fn allocate_uninit_on_device_cpu_empty() {
    let buf =
        DataBuffer::<f64>::allocate_uninit_on_device(0, LogicalMemorySpace::MainMemory).unwrap();
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

#[test]
fn tensor_zeros_main_memory_still_works() {
    let t = Tensor::<f64>::zeros(
        &[2, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.len(), 6);
    let data = t.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn tensor_zeros_row_major_main_memory() {
    let t = Tensor::<f64>::zeros(
        &[3, 4],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::RowMajor,
    )
    .unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.len(), 12);
    let data = t.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn tensor_empty_main_memory_returns_zeros() {
    let t = Tensor::<f64>::empty(
        &[4, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert_eq!(t.dims(), &[4, 2]);
    let data = t.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn tensor_empty_strided_main_memory_returns_zeros() {
    let t =
        Tensor::<f64>::empty_strided(&[2, 3], &[1, 2], 0, LogicalMemorySpace::MainMemory).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.strides(), &[1, 2]);
    let data = t.buffer().as_slice().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn zeros_on_device_complex_cpu() {
    use num_complex::Complex64;
    let buf = DataBuffer::<Complex64>::zeros_on_device(3, LogicalMemorySpace::MainMemory).unwrap();
    assert_eq!(buf.len(), 3);
    let slice = buf.as_slice().unwrap();
    for &v in slice {
        assert_eq!(v, Complex64::new(0.0, 0.0));
    }
}

#[test]
fn tensor_zeros_scalar_shape() {
    let t = Tensor::<f32>::zeros(
        &[],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert_eq!(t.dims(), &[] as &[usize]);
    assert_eq!(t.len(), 1);
    assert_eq!(t.buffer().as_slice().unwrap(), &[0.0_f32]);
}
