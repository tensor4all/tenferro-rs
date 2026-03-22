use super::*;
use num_complex::Complex64;
#[cfg(feature = "cuda")]
use tenferro_device::LogicalMemorySpace;
use tenferro_device::{ComputeDevice, Error};

fn col_tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn complex_col_tensor(data: &[Complex64], dims: &[usize]) -> Tensor<Complex64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn stack_materializes_along_leading_and_trailing_axes() {
    let a = col_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = col_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);

    let leading = Tensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(leading.dims(), &[2, 2, 2]);
    assert_eq!(
        leading.buffer().as_slice().unwrap(),
        &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
    );

    let trailing = Tensor::stack(&[&a, &b], -1).unwrap();
    assert_eq!(trailing.dims(), &[2, 2, 2]);
    assert_eq!(
        trailing.buffer().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
    );
}

#[test]
fn stack_preserves_empty_shapes_when_any_extent_is_zero() {
    let a = col_tensor(&[], &[0, 2]);
    let b = col_tensor(&[], &[0, 2]);

    let stacked = Tensor::stack(&[&a, &b], 1).unwrap();
    assert_eq!(stacked.dims(), &[0, 2, 2]);
    assert!(stacked.buffer().as_slice().unwrap().is_empty());
}

#[test]
fn stack_validates_inputs_and_dimension_range() {
    let a = col_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let wrong_shape = col_tensor(&[1.0, 2.0], &[2, 1]);

    let empty_err = Tensor::<f64>::stack(&[], 0).unwrap_err();
    assert!(
        matches!(empty_err, Error::InvalidArgument(ref msg) if msg.contains("at least one tensor"))
    );

    let shape_err = Tensor::stack(&[&a, &wrong_shape], 0).unwrap_err();
    assert!(
        matches!(shape_err, Error::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {shape_err:?}"
    );

    let dim_err = Tensor::stack(&[&a], 3).unwrap_err();
    assert!(
        matches!(dim_err, Error::InvalidArgument(ref msg) if msg.contains("out of range")),
        "expected out-of-range error, got {dim_err:?}"
    );
}

#[test]
fn cat_materializes_along_existing_axes_and_supports_negative_dims() {
    let a = col_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b_cols = col_tensor(&[10.0, 20.0], &[2, 1]);
    let b_rows = col_tensor(&[10.0, 20.0], &[1, 2]);

    let cat_cols = Tensor::cat(&[&a, &b_cols], 1).unwrap();
    assert_eq!(cat_cols.dims(), &[2, 3]);
    assert_eq!(
        cat_cols.buffer().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0]
    );

    let cat_rows = Tensor::cat(&[&a, &b_rows], -2).unwrap();
    assert_eq!(cat_rows.dims(), &[3, 2]);
    assert_eq!(
        cat_rows.buffer().as_slice().unwrap(),
        &[1.0, 2.0, 10.0, 3.0, 4.0, 20.0]
    );
}

#[test]
fn cat_validates_rank_shape_and_dimension_range() {
    let matrix = col_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let wrong_rank = col_tensor(&[1.0, 2.0], &[2]);
    let wrong_shape = col_tensor(&[1.0, 2.0, 3.0], &[3, 1]);
    let scalar = col_tensor(&[1.0], &[]);

    let empty_err = Tensor::<f64>::cat(&[], 0).unwrap_err();
    assert!(
        matches!(empty_err, Error::InvalidArgument(ref msg) if msg.contains("at least one tensor"))
    );

    let scalar_err = Tensor::cat(&[&scalar], 0).unwrap_err();
    assert!(
        matches!(scalar_err, Error::InvalidArgument(ref msg) if msg.contains("rank-0 tensors"))
    );

    let rank_err = Tensor::cat(&[&matrix, &wrong_rank], 0).unwrap_err();
    assert!(
        matches!(rank_err, Error::InvalidArgument(ref msg) if msg.contains("expected rank")),
        "expected rank mismatch, got {rank_err:?}"
    );

    let shape_err = Tensor::cat(&[&matrix, &wrong_shape], 1).unwrap_err();
    assert!(
        matches!(shape_err, Error::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {shape_err:?}"
    );

    let dim_err = Tensor::cat(&[&matrix], 2).unwrap_err();
    assert!(
        matches!(dim_err, Error::InvalidArgument(ref msg) if msg.contains("out of range")),
        "expected out-of-range error, got {dim_err:?}"
    );
}

#[test]
fn cat_preserves_uniform_conjugation_flag_and_clears_preferred_device_hint() {
    let mut a = complex_col_tensor(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
        ],
        &[2, 2],
    )
    .conj();
    a.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 7 }));
    let b = complex_col_tensor(
        &[
            Complex64::new(5.0, -1.0),
            Complex64::new(6.0, -2.0),
            Complex64::new(7.0, -3.0),
            Complex64::new(8.0, -4.0),
        ],
        &[2, 2],
    )
    .conj();

    let got = Tensor::cat(&[&a, &b], 1).unwrap();
    assert!(got.is_conjugated());
    assert_eq!(got.preferred_compute_device(), None);
    assert_eq!(
        got.buffer().as_slice().unwrap(),
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
            Complex64::new(5.0, -1.0),
            Complex64::new(6.0, -2.0),
            Complex64::new(7.0, -3.0),
            Complex64::new(8.0, -4.0),
        ]
    );
}

#[test]
fn cat_rejects_mixed_conjugation_flags() {
    let a = complex_col_tensor(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
        ],
        &[2, 2],
    )
    .conj();
    let b = complex_col_tensor(
        &[
            Complex64::new(5.0, -1.0),
            Complex64::new(6.0, -2.0),
            Complex64::new(7.0, -3.0),
            Complex64::new(8.0, -4.0),
        ],
        &[2, 2],
    );

    let err = Tensor::cat(&[&a, &b], 1).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("conjugation")),
        "expected mixed-conjugation rejection, got {err:?}"
    );
}

#[test]
fn stack_preserves_uniform_conjugation_flag_and_clears_preferred_device_hint() {
    let mut a = complex_col_tensor(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
        ],
        &[2, 2],
    )
    .conj();
    a.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 7 }));
    let b = complex_col_tensor(
        &[
            Complex64::new(5.0, -1.0),
            Complex64::new(6.0, -2.0),
            Complex64::new(7.0, -3.0),
            Complex64::new(8.0, -4.0),
        ],
        &[2, 2],
    )
    .conj();

    let got = Tensor::stack(&[&a, &b], 0).unwrap();
    assert!(got.is_conjugated());
    assert_eq!(got.preferred_compute_device(), None);
    assert_eq!(got.dims(), &[2, 2, 2]);
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    fn cuda_device_zero_is_available() -> bool {
        std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
    }

    #[test]
    fn gpu_cat_materializes_along_nonzero_axis_when_cuda_is_available() {
        if !cuda_device_zero_is_available() {
            return;
        }

        let a = col_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = col_tensor(&[10.0, 20.0], &[2, 1]);
        let expected = Tensor::cat(&[&a, &b], 1).unwrap();

        let gpu_a = a
            .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let gpu_b = b
            .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let got = Tensor::cat(&[&gpu_a, &gpu_b], 1)
            .expect("GPU cat should accept GPU tensors on the requested axis");
        assert_eq!(
            got.logical_memory_space(),
            LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        let got = got
            .to_memory_space_async(LogicalMemorySpace::MainMemory)
            .unwrap();

        assert_eq!(got.dims(), expected.dims());
        assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
    }

    #[test]
    fn gpu_stack_materializes_on_device_when_cuda_is_available() {
        if !cuda_device_zero_is_available() {
            return;
        }

        let a = col_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = col_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let expected = Tensor::stack(&[&a, &b], 0).unwrap();

        let gpu_a = a
            .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let gpu_b = b
            .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let got = Tensor::stack(&[&gpu_a, &gpu_b], 0)
            .expect("GPU stack should accept GPU tensors on the requested axis");
        assert_eq!(
            got.logical_memory_space(),
            LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        let got = got
            .to_memory_space_async(LogicalMemorySpace::MainMemory)
            .unwrap();

        assert_eq!(got.dims(), expected.dims());
        assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
    }
}
