use super::*;
use num_complex::{Complex32, Complex64};
use tenferro_device::{Error, LogicalMemorySpace};

#[cfg(feature = "cuda")]
fn cuda_device_zero_is_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

#[cfg(not(feature = "cuda"))]
fn cuda_device_zero_is_available() -> bool {
    false
}

macro_rules! representation_suite {
    (
        $module:ident,
        $complex:ty,
        $real:ty,
        $complex_values:expr,
        $real_values:expr,
        $real_expected_strides:expr,
        $complex_expected_strides:expr,
        $complex_invalid_layout_values:expr
    ) => {
        mod $module {
            use super::*;

            fn complex_tensor() -> Tensor<$complex> {
                Tensor::<$complex>::from_slice(
                    &$complex_values,
                    &[2, 2],
                    MemoryOrder::ColumnMajor,
                )
                .unwrap()
            }

            fn real_tensor() -> Tensor<$real> {
                Tensor::<$real>::from_slice(
                    &$real_values,
                    &[2, 2, 2],
                    MemoryOrder::RowMajor,
                )
                .unwrap()
            }

            #[test]
            fn view_as_real_preserves_shape_stride_and_values_on_cpu() {
                let base = complex_tensor();
                let expected = Tensor::<$real>::from_slice(
                    &$real_values,
                    &[2, 2, 2],
                    MemoryOrder::RowMajor,
                )
                .unwrap();

                let view = base.view_as_real().unwrap();
                assert_eq!(view.dims(), &[2, 2, 2]);
                assert_eq!(view.strides(), $real_expected_strides);
                assert_eq!(view.offset(), 0);
                assert_eq!(view.logical_memory_space(), base.logical_memory_space());
                assert_eq!(view.buffer().as_slice(), expected.buffer().as_slice());
                assert_eq!(
                    view.buffer().as_ptr().unwrap() as usize,
                    base.buffer().as_ptr().unwrap() as usize
                );
            }

            #[test]
            fn view_as_complex_roundtrips_shape_stride_and_values_on_cpu() {
                let base = real_tensor();
                let expected = Tensor::<$complex>::from_slice(
                    &$complex_values,
                    &[2, 2],
                    MemoryOrder::ColumnMajor,
                )
                .unwrap();

                let view = base.view_as_complex().unwrap();
                assert_eq!(view.dims(), &[2, 2]);
                assert_eq!(view.strides(), $complex_expected_strides);
                assert_eq!(view.offset(), 0);
                assert_eq!(view.logical_memory_space(), base.logical_memory_space());
                assert_eq!(view.buffer().as_slice(), expected.buffer().as_slice());
                assert_eq!(
                    view.buffer().as_ptr().unwrap() as usize,
                    base.buffer().as_ptr().unwrap() as usize
                );

                let roundtrip = view.view_as_real().unwrap();
                assert_eq!(roundtrip.dims(), base.dims());
                assert_eq!(roundtrip.strides(), base.strides());
                assert_eq!(roundtrip.buffer().as_slice(), base.buffer().as_slice());
            }

            #[test]
            fn view_as_complex_rejects_nonconforming_layout() {
                let invalid = Tensor::<$real>::from_slice(
                    &$complex_invalid_layout_values,
                    &[2, 2],
                    MemoryOrder::ColumnMajor,
                )
                .unwrap();

                let err = invalid.view_as_complex().unwrap_err();
                assert!(
                    matches!(err, Error::InvalidArgument(ref msg) if msg.contains("stride 1") || msg.contains("last stride") || msg.contains("last dimension")),
                    "expected layout contract error, got {err:?}"
                );
            }

            #[test]
            fn view_as_complex_allows_odd_leading_stride_for_singleton_dims() {
                let singleton = Tensor::<$real>::from_vec(
                    vec![1.0 as $real, 11.0 as $real, 99.0 as $real],
                    &[1, 2],
                    &[1, 1],
                    0,
                )
                .unwrap();

                let view = singleton.view_as_complex().unwrap();
                assert_eq!(view.dims(), &[1]);
                assert_eq!(view.strides(), &[0]);
                assert_eq!(
                    view.buffer().as_slice().unwrap()[0],
                    <$complex>::new(1.0 as $real, 11.0 as $real)
                );
            }

            #[test]
            fn view_as_complex_does_not_require_even_backing_storage_len() {
                let subview = Tensor::<$real>::from_vec(
                    vec![1.0 as $real, 11.0 as $real, 99.0 as $real],
                    &[1, 2],
                    &[2, 1],
                    0,
                )
                .unwrap();

                let view = subview.view_as_complex().unwrap();
                assert_eq!(view.dims(), &[1]);
                assert_eq!(view.strides(), &[1]);
                assert_eq!(
                    view.buffer().as_slice().unwrap()[0],
                    <$complex>::new(1.0 as $real, 11.0 as $real)
                );
            }

            #[test]
            fn view_as_real_matches_cuda_parity_when_available() {
                if !cuda_device_zero_is_available() {
                    return;
                }

                let base = complex_tensor();
                let expected = base.view_as_real().unwrap();
                let got = base
                    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
                    .unwrap()
                    .view_as_real()
                    .unwrap()
                    .to_memory_space_async(LogicalMemorySpace::MainMemory)
                    .unwrap();

                assert_eq!(got.dims(), expected.dims());
                assert_eq!(got.strides(), expected.strides());
                assert_eq!(got.offset(), expected.offset());
                assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
            }

            #[test]
            fn view_as_complex_matches_cuda_parity_when_available() {
                if !cuda_device_zero_is_available() {
                    return;
                }

                let base = real_tensor();
                let expected = base.view_as_complex().unwrap();
                let got = base
                    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
                    .unwrap()
                    .view_as_complex()
                    .unwrap()
                    .to_memory_space_async(LogicalMemorySpace::MainMemory)
                    .unwrap();

                assert_eq!(got.dims(), expected.dims());
                assert_eq!(got.strides(), expected.strides());
                assert_eq!(got.offset(), expected.offset());
                assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
            }
        }
    };
}

representation_suite!(
    complex64,
    Complex64,
    f64,
    [
        Complex64::new(1.0, 11.0),
        Complex64::new(2.0, 22.0),
        Complex64::new(3.0, 33.0),
        Complex64::new(4.0, 44.0),
    ],
    [1.0_f64, 11.0, 2.0, 22.0, 3.0, 33.0, 4.0, 44.0],
    &[2, 4, 1],
    &[2, 1],
    [1.0_f64, 11.0, 2.0, 22.0]
);

representation_suite!(
    complex32,
    Complex32,
    f32,
    [
        Complex32::new(1.0, 11.0),
        Complex32::new(2.0, 22.0),
        Complex32::new(3.0, 33.0),
        Complex32::new(4.0, 44.0),
    ],
    [1.0_f32, 11.0, 2.0, 22.0, 3.0, 33.0, 4.0, 44.0],
    &[2, 4, 1],
    &[2, 1],
    [1.0_f32, 11.0, 2.0, 22.0]
);

macro_rules! component_view_suite {
    ($module:ident, $complex:ty, $real:ty, $values:expr, $real_values:expr, $imag_values:expr) => {
        mod $module {
            use super::*;

            fn complex_tensor() -> Tensor<$complex> {
                Tensor::<$complex>::from_slice(&$values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
            }

            #[test]
            fn real_returns_zero_copy_component_view() {
                let base = complex_tensor();
                let real = base.real().unwrap();

                assert_eq!(real.dims(), &[2, 2]);
                assert_eq!(real.strides(), &[2, 4]);
                assert_eq!(real.offset(), 0);
                assert_eq!(real.buffer().as_ptr().unwrap() as usize, base.buffer().as_ptr().unwrap() as usize);
                assert_eq!(
                    real.contiguous(MemoryOrder::ColumnMajor)
                        .buffer()
                        .as_slice()
                        .unwrap(),
                    &$real_values
                );
            }

            #[test]
            fn imag_returns_zero_copy_component_view() {
                let base = complex_tensor();
                let imag = base.imag().unwrap();

                assert_eq!(imag.dims(), &[2, 2]);
                assert_eq!(imag.strides(), &[2, 4]);
                assert_eq!(imag.offset(), 1);
                assert_eq!(imag.buffer().as_ptr().unwrap() as usize, base.buffer().as_ptr().unwrap() as usize);
                assert_eq!(
                    imag.contiguous(MemoryOrder::ColumnMajor)
                        .buffer()
                        .as_slice()
                        .unwrap(),
                    &$imag_values
                );
            }

            #[test]
            fn real_and_imag_reject_conjugated_inputs() {
                let base = complex_tensor();

                let real_err = base.conj().real().unwrap_err();
                assert!(
                    matches!(real_err, Error::InvalidArgument(ref msg) if msg.contains("resolved complex tensor")),
                    "expected resolved-complex error from real(), got {real_err:?}"
                );

                let imag_err = base.conj().imag().unwrap_err();
                assert!(
                    matches!(imag_err, Error::InvalidArgument(ref msg) if msg.contains("resolved complex tensor")),
                    "expected resolved-complex error from imag(), got {imag_err:?}"
                );
            }
        }
    };
}

component_view_suite!(
    complex64_components,
    Complex64,
    f64,
    [
        Complex64::new(1.0, 11.0),
        Complex64::new(2.0, 22.0),
        Complex64::new(3.0, 33.0),
        Complex64::new(4.0, 44.0),
    ],
    [1.0_f64, 2.0, 3.0, 4.0],
    [11.0_f64, 22.0, 33.0, 44.0]
);

component_view_suite!(
    complex32_components,
    Complex32,
    f32,
    [
        Complex32::new(1.0, 11.0),
        Complex32::new(2.0, 22.0),
        Complex32::new(3.0, 33.0),
        Complex32::new(4.0, 44.0),
    ],
    [1.0_f32, 2.0, 3.0, 4.0],
    [11.0_f32, 22.0, 33.0, 44.0]
);

#[test]
fn view_as_real_rejects_conjugated_complex_tensors() {
    let complex64 = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 2.0)],
        &[1],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = complex64.conj().view_as_real().unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("resolved complex tensor"))
    );

    let complex32 = Tensor::<Complex32>::from_slice(
        &[Complex32::new(1.0, 2.0)],
        &[1],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = complex32.conj().view_as_real().unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("resolved complex tensor"))
    );
}

#[test]
fn view_as_complex_rejects_scalar_wrong_last_dim_wrong_last_stride_and_odd_offset() {
    let scalar = Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap();
    let err = scalar.view_as_complex().unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("at least one dimension"))
    );

    let wrong_last_dim =
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let err = wrong_last_dim.view_as_complex().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(ref msg) if msg.contains("last dimension")));

    let wrong_last_stride =
        Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &[1, 2], 0).unwrap();
    let err = wrong_last_stride.view_as_complex().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(ref msg) if msg.contains("last stride")));

    let odd_offset = Tensor::<f64>::from_vec(vec![0.0, 1.0, 11.0], &[1, 2], &[2, 1], 1).unwrap();
    let err = odd_offset.view_as_complex().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(ref msg) if msg.contains("even element offset")));
}

#[test]
fn view_as_complex_rejects_same_error_cases_for_f32() {
    let scalar = Tensor::<f32>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap();
    let err = scalar.view_as_complex().unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("at least one dimension"))
    );

    let wrong_last_dim =
        Tensor::<f32>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let err = wrong_last_dim.view_as_complex().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(ref msg) if msg.contains("last dimension")));

    let wrong_last_stride =
        Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &[1, 2], 0).unwrap();
    let err = wrong_last_stride.view_as_complex().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(ref msg) if msg.contains("last stride")));

    let odd_offset = Tensor::<f32>::from_vec(vec![0.0, 1.0, 11.0], &[1, 2], &[2, 1], 1).unwrap();
    let err = odd_offset.view_as_complex().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(ref msg) if msg.contains("even element offset")));
}
