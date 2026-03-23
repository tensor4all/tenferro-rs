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
                    matches!(err, Error::InvalidArgument(ref msg) if msg.contains("stride 1") || msg.contains("last dimension")),
                    "expected layout contract error, got {err:?}"
                );
            }

            #[test]
            fn view_as_real_matches_cuda_parity_when_available() {
                if !cuda_device_zero_is_available() {
                    return;
                }

                let base = complex_tensor();
                let expected = base.view_as_real();
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
