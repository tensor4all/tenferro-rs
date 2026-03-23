use super::*;
use tenferro_device::{Error, LogicalMemorySpace};

const CPU: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

#[cfg(feature = "cuda")]
const GPU0: LogicalMemorySpace = LogicalMemorySpace::GpuMemory { device_id: 0 };

fn assert_dense_f64_sequence(
    tensor: &Tensor<f64>,
    dims: &[usize],
    space: LogicalMemorySpace,
    expected: &[f64],
) {
    let host = tensor
        .to_memory_space_async(CPU)
        .expect("constructor result should be movable back to main memory for comparison");
    assert_eq!(tensor.dims(), dims);
    assert_eq!(tensor.logical_memory_space(), space);
    assert_eq!(host.buffer().as_slice().unwrap(), expected);
}

#[test]
fn cpu_empty_reports_shape_layout_and_device_semantics() {
    let got = Tensor::<f64>::empty(&[2, 3], CPU, MemoryOrder::ColumnMajor).unwrap();

    assert_eq!(got.dims(), &[2, 3]);
    assert_eq!(got.strides(), &[1, 2]);
    assert_eq!(got.logical_memory_space(), CPU);
    assert!(got.is_contiguous());
}

#[test]
fn cpu_empty_strided_validates_layout_and_reports_invalid_layouts() {
    let got = Tensor::<f64>::empty_strided(&[2, 2], &[1, 2], 0, CPU).unwrap();
    assert_eq!(got.dims(), &[2, 2]);
    assert_eq!(got.strides(), &[1, 2]);
    assert_eq!(got.logical_memory_space(), CPU);

    let err = Tensor::<f64>::empty_strided(&[2, 2], &[1, 2], -1, CPU).unwrap_err();
    assert!(
        matches!(err, Error::StrideError(ref msg) if msg.contains("layout") || msg.contains("buffer positions")),
        "expected empty_strided layout validation error, got {err:?}"
    );
}

#[test]
fn cpu_full_and_like_constructors_fill_expected_values() {
    let base = Tensor::<f64>::zeros(&[2, 3], CPU, MemoryOrder::ColumnMajor).unwrap();
    let row_major_base = Tensor::<f64>::zeros(&[2, 3], CPU, MemoryOrder::RowMajor).unwrap();

    let full = Tensor::<f64>::full(&[2, 3], 7.5, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_dense_f64_sequence(&full, &[2, 3], CPU, &[7.5; 6]);

    let empty_like = Tensor::<f64>::empty_like(&base).unwrap();
    assert_eq!(empty_like.dims(), base.dims());
    assert_eq!(
        empty_like.logical_memory_space(),
        base.logical_memory_space()
    );

    let zeros_like = Tensor::<f64>::zeros_like(&base).unwrap();
    assert_dense_f64_sequence(&zeros_like, &[2, 3], CPU, &[0.0; 6]);

    let ones_like = Tensor::<f64>::ones_like(&base).unwrap();
    assert_dense_f64_sequence(&ones_like, &[2, 3], CPU, &[1.0; 6]);

    let full_like = Tensor::<f64>::full_like(&base, 3.25).unwrap();
    assert_dense_f64_sequence(&full_like, &[2, 3], CPU, &[3.25; 6]);

    let row_major_empty_like = Tensor::<f64>::empty_like(&row_major_base).unwrap();
    assert_eq!(row_major_empty_like.strides(), row_major_base.strides());
    assert!(row_major_empty_like.is_row_major_contiguous());

    let row_major_zeros_like = Tensor::<f64>::zeros_like(&row_major_base).unwrap();
    assert_eq!(row_major_zeros_like.strides(), row_major_base.strides());
    assert!(row_major_zeros_like.is_row_major_contiguous());

    let row_major_ones_like = Tensor::<f64>::ones_like(&row_major_base).unwrap();
    assert_eq!(row_major_ones_like.strides(), row_major_base.strides());
    assert!(row_major_ones_like.is_row_major_contiguous());

    let row_major_full_like = Tensor::<f64>::full_like(&row_major_base, 3.25).unwrap();
    assert_eq!(row_major_full_like.strides(), row_major_base.strides());
    assert!(row_major_full_like.is_row_major_contiguous());
}

#[test]
fn cpu_arange_and_linspace_construct_expected_sequences() {
    let arange = Tensor::<f64>::arange(0.0, 5.0, 1.0, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_dense_f64_sequence(&arange, &[5], CPU, &[0.0, 1.0, 2.0, 3.0, 4.0]);

    let linspace = Tensor::<f64>::linspace(0.0, 1.0, 5, CPU, MemoryOrder::ColumnMajor).unwrap();
    assert_dense_f64_sequence(&linspace, &[5], CPU, &[0.0, 0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn cpu_arange_and_linspace_reject_invalid_inputs() {
    let err = Tensor::<f64>::arange(0.0, 5.0, 0.0, CPU, MemoryOrder::ColumnMajor).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("step")),
        "expected arange step validation error, got {err:?}"
    );

    let err = Tensor::<f64>::linspace(0.0, 1.0, -1, CPU, MemoryOrder::ColumnMajor).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("steps")),
        "expected linspace steps validation error, got {err:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_reports_shape_layout_and_device_semantics() {
    let got = Tensor::<f64>::empty(&[2, 3], GPU0, MemoryOrder::ColumnMajor).unwrap();

    assert_eq!(got.dims(), &[2, 3]);
    assert_eq!(got.strides(), &[1, 2]);
    assert_eq!(got.logical_memory_space(), GPU0);
    assert!(got.is_contiguous());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_strided_validates_layout_and_reports_invalid_layouts() {
    let got = Tensor::<f64>::empty_strided(&[2, 2], &[1, 2], 0, GPU0).unwrap();
    assert_eq!(got.dims(), &[2, 2]);
    assert_eq!(got.strides(), &[1, 2]);
    assert_eq!(got.logical_memory_space(), GPU0);

    let err = Tensor::<f64>::empty_strided(&[2, 2], &[1, 2], -1, GPU0).unwrap_err();
    assert!(
        matches!(err, Error::StrideError(ref msg) if msg.contains("layout") || msg.contains("buffer positions")),
        "expected empty_strided layout validation error, got {err:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_full_and_like_constructors_fill_expected_values() {
    let base = Tensor::<f64>::zeros(&[2, 3], GPU0, MemoryOrder::ColumnMajor).unwrap();
    let row_major_base = Tensor::<f64>::zeros(&[2, 3], GPU0, MemoryOrder::RowMajor).unwrap();

    let full = Tensor::<f64>::full(&[2, 3], 7.5, GPU0, MemoryOrder::ColumnMajor).unwrap();
    assert_dense_f64_sequence(&full, &[2, 3], GPU0, &[7.5; 6]);

    let empty_like = Tensor::<f64>::empty_like(&base).unwrap();
    assert_eq!(empty_like.dims(), base.dims());
    assert_eq!(
        empty_like.logical_memory_space(),
        base.logical_memory_space()
    );

    let zeros_like = Tensor::<f64>::zeros_like(&base).unwrap();
    assert_dense_f64_sequence(&zeros_like, &[2, 3], GPU0, &[0.0; 6]);

    let ones_like = Tensor::<f64>::ones_like(&base).unwrap();
    assert_dense_f64_sequence(&ones_like, &[2, 3], GPU0, &[1.0; 6]);

    let full_like = Tensor::<f64>::full_like(&base, 3.25).unwrap();
    assert_dense_f64_sequence(&full_like, &[2, 3], GPU0, &[3.25; 6]);

    let row_major_empty_like = Tensor::<f64>::empty_like(&row_major_base).unwrap();
    assert_eq!(row_major_empty_like.strides(), row_major_base.strides());
    assert!(row_major_empty_like.is_row_major_contiguous());

    let row_major_zeros_like = Tensor::<f64>::zeros_like(&row_major_base).unwrap();
    assert_eq!(row_major_zeros_like.strides(), row_major_base.strides());
    assert!(row_major_zeros_like.is_row_major_contiguous());

    let row_major_ones_like = Tensor::<f64>::ones_like(&row_major_base).unwrap();
    assert_eq!(row_major_ones_like.strides(), row_major_base.strides());
    assert!(row_major_ones_like.is_row_major_contiguous());

    let row_major_full_like = Tensor::<f64>::full_like(&row_major_base, 3.25).unwrap();
    assert_eq!(row_major_full_like.strides(), row_major_base.strides());
    assert!(row_major_full_like.is_row_major_contiguous());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_arange_and_linspace_construct_expected_sequences() {
    let arange = Tensor::<f64>::arange(0.0, 5.0, 1.0, GPU0, MemoryOrder::ColumnMajor).unwrap();
    assert_dense_f64_sequence(&arange, &[5], GPU0, &[0.0, 1.0, 2.0, 3.0, 4.0]);

    let linspace = Tensor::<f64>::linspace(0.0, 1.0, 5, GPU0, MemoryOrder::ColumnMajor).unwrap();
    assert_dense_f64_sequence(&linspace, &[5], GPU0, &[0.0, 0.25, 0.5, 0.75, 1.0]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_arange_and_linspace_reject_invalid_inputs() {
    let err = Tensor::<f64>::arange(0.0, 5.0, 0.0, GPU0, MemoryOrder::ColumnMajor).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("step")),
        "expected arange step validation error, got {err:?}"
    );

    let err = Tensor::<f64>::linspace(0.0, 1.0, -1, GPU0, MemoryOrder::ColumnMajor).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("steps")),
        "expected linspace steps validation error, got {err:?}"
    );
}
