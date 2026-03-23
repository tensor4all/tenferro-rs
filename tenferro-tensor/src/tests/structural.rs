use num_complex::Complex64;
use tenferro_device::{ComputeDevice, Error, LogicalMemorySpace, OpKind};

use super::*;

fn col_tensor<T: tenferro_algebra::Scalar>(data: &[T], dims: &[usize]) -> Tensor<T> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn zero_trailing_by_counts_cpu_zero_fills_after_keep_count_real_payload() {
    let payload = col_tensor(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[2, 2, 2]);
    let keep_counts = col_tensor(&[1.0, 2.0], &[2]);

    let got = payload.zero_trailing_by_counts(&keep_counts, 1, 2).unwrap();

    assert_eq!(got.dims(), payload.dims());
    assert_eq!(got.strides(), &[1, 2, 4]);
    assert_eq!(got.logical_memory_space(), LogicalMemorySpace::MainMemory);
    assert_eq!(
        got.buffer().as_slice().unwrap(),
        &[1.0, 2.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0]
    );
}

#[test]
fn zero_trailing_by_counts_cpu_zero_fills_complex_payload() {
    let payload = col_tensor(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
            Complex64::new(5.0, 5.0),
            Complex64::new(6.0, 6.0),
            Complex64::new(7.0, 7.0),
            Complex64::new(8.0, 8.0),
            Complex64::new(9.0, 9.0),
            Complex64::new(10.0, 10.0),
            Complex64::new(11.0, 11.0),
            Complex64::new(12.0, 12.0),
        ],
        &[3, 2, 2],
    );
    let keep_counts = col_tensor(&[2.0, 1.0], &[2]);

    let got = payload.zero_trailing_by_counts(&keep_counts, 0, 2).unwrap();

    assert_eq!(got.dims(), payload.dims());
    assert_eq!(
        got.buffer().as_slice().unwrap(),
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 4.0),
            Complex64::new(5.0, 5.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(7.0, 7.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(10.0, 10.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    );
}

#[test]
fn zero_trailing_by_counts_cpu_rejects_invalid_keep_counts() {
    let payload = Tensor::<f64>::ones(
        &[2, 3, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let shape_mismatch = col_tensor(&[1.0, 2.0, 3.0], &[3]);
    let err = payload
        .zero_trailing_by_counts(&shape_mismatch, 1, 2)
        .unwrap_err();
    assert!(
        matches!(err, Error::ShapeMismatch { ref expected, ref got } if *expected == vec![2] && *got == vec![3]),
        "expected keep-count shape mismatch, got {err:?}"
    );

    let negative = col_tensor(&[-1.0, 2.0], &[2]);
    let err = payload
        .zero_trailing_by_counts(&negative, 1, 2)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("keep_counts") && msg.contains("non-negative")),
        "expected negative-count error, got {err:?}"
    );

    let non_integer = col_tensor(&[1.5, 2.0], &[2]);
    let err = payload
        .zero_trailing_by_counts(&non_integer, 1, 2)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("integer-valued")),
        "expected non-integer-count error, got {err:?}"
    );

    let too_large = col_tensor(&[4.0, 2.0], &[2]);
    let err = payload
        .zero_trailing_by_counts(&too_large, 1, 2)
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("exceeds axis length")),
        "expected too-large-count error, got {err:?}"
    );
}

#[test]
fn merge_strict_lower_and_upper_cpu_packs_rectangular_sources() {
    let lower = col_tensor(&[10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
    let upper = col_tensor(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]);

    let got = Tensor::merge_strict_lower_and_upper(&lower, &upper).unwrap();

    assert_eq!(got.dims(), &[3, 2]);
    assert_eq!(
        got.buffer().as_slice().unwrap(),
        &[1.0, 20.0, 30.0, 3.0, 4.0, 60.0]
    );
}

#[test]
fn merge_strict_lower_and_upper_cpu_packs_complex_sources() {
    let lower = col_tensor(
        &[
            Complex64::new(10.0, 1.0),
            Complex64::new(20.0, 2.0),
            Complex64::new(30.0, 3.0),
            Complex64::new(40.0, 4.0),
        ],
        &[2, 2],
    );
    let upper = col_tensor(
        &[
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(4.0, -4.0),
        ],
        &[2, 2],
    );

    let got = Tensor::merge_strict_lower_and_upper(&lower, &upper).unwrap();

    assert_eq!(
        got.buffer().as_slice().unwrap(),
        &[
            Complex64::new(1.0, -1.0),
            Complex64::new(20.0, 2.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(4.0, -4.0),
        ]
    );
}

#[test]
fn merge_strict_lower_and_upper_rejects_invalid_shapes() {
    let lower = col_tensor(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]);
    let bad_upper = col_tensor(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2, 1]);
    let err = Tensor::merge_strict_lower_and_upper(&lower, &bad_upper).unwrap_err();
    assert!(matches!(err, Error::RankMismatch { .. }));

    let bad_lower = col_tensor(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]);
    let bad_upper = col_tensor(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let err = Tensor::merge_strict_lower_and_upper(&bad_lower, &bad_upper).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("min(m, n)")),
        "expected LU-shape contract error, got {err:?}"
    );
}

#[test]
fn merge_strict_lower_and_upper_rejects_conjugated_inputs() {
    let lower = col_tensor(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
        ],
        &[2, 2],
    )
    .conj();
    let upper = col_tensor(
        &[
            Complex64::new(5.0, -1.0),
            Complex64::new(6.0, -2.0),
            Complex64::new(7.0, -3.0),
            Complex64::new(8.0, -4.0),
        ],
        &[2, 2],
    )
    .conj();

    let err = Tensor::merge_strict_lower_and_upper(&lower, &upper).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("conjugated")),
        "expected conjugation contract error, got {err:?}"
    );
}

#[test]
fn merge_strict_lower_and_upper_clears_stale_main_memory_device_hint() {
    let mut lower = col_tensor(&[10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
    let mut upper = col_tensor(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]);
    lower.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 1 }));
    upper.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 1 }));

    let got = Tensor::merge_strict_lower_and_upper(&lower, &upper).unwrap();

    assert_eq!(got.preferred_compute_device(), None);
    assert_eq!(
        got.effective_compute_devices(OpKind::BatchedGemm).unwrap(),
        vec![ComputeDevice::Cpu { device_id: 0 }]
    );
}
