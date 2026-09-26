use super::super::blas1::blas1_len;
use cudarc::cublas::sys as cublas;
use num_complex::{Complex32, Complex64};
use tenferro_tensor::BackendSessionHost;

#[test]
fn blas1_length_stays_within_the_portable_cublas_interface() {
    assert_eq!(blas1_len(i32::MAX as usize, "test").unwrap(), i32::MAX);
    let error = blas1_len(i32::MAX as usize + 1, "test").unwrap_err();
    assert!(error.to_string().contains("exceeds i32::MAX"));
}

/// Issue #1832: an arbitrary-stride `x` runs the native strided-source pass in
/// place. The counter pins the data movement: a regression that canonicalizes
/// `x` back into scratch would serve the call through cuBLAS and leave it at
/// zero, even though the numerics would still match.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_axpby_reads_an_offset_strided_source_in_place() {
    use num_complex::Complex64;
    use tenferro_tensor::{ContractionScalar, TensorRead, TensorView, TensorWrite};

    use super::super::blas1::{
        reset_strided_source_passes_for_test, strided_source_passes_for_test,
    };
    use super::{cpu_backend, download, gpu_backend, tensor_c64, upload};

    let mut gpu = gpu_backend();
    let mut cpu = cpu_backend();

    // 24-element source allocation; the operand is a 2x3 region at offset 5
    // with axis strides (1, 4), so neither the offset nor the strides are
    // expressible as a compact span.
    let source_values: Vec<Complex64> = (0..24)
        .map(|value| Complex64::new(f64::from(value), 0.5 - f64::from(value)))
        .collect();
    let host_source = tensor_c64(vec![24], source_values.clone());
    let host_destination = tensor_c64(
        vec![2, 3],
        (0..6)
            .map(|value| Complex64::new(2.0 - f64::from(value), 0.25 * f64::from(value)))
            .collect(),
    );
    let alpha = ContractionScalar::C64(Complex64::new(0.75, -1.25));
    let beta = ContractionScalar::C64(Complex64::new(-0.5, 2.0));

    let mut expected = host_destination.duplicate().unwrap();
    {
        let mut host_region = Vec::with_capacity(6);
        for column in 0..3usize {
            for row in 0..2usize {
                host_region.push(source_values[5 + row + 4 * column]);
            }
        }
        let host_x = tensor_c64(vec![2, 3], host_region);
        cpu.with_backend_session(|__s| {
            __s.axpby_read_into_accum(
                alpha,
                TensorRead::from_tensor(&host_x),
                beta,
                TensorWrite::from_tensor(&mut expected),
            )
        })
        .unwrap();
    }

    let gpu_source = upload(&gpu, &host_source);
    let mut gpu_destination = upload(&gpu, &host_destination);
    let Some(source) = gpu_source.as_typed::<Complex64>() else {
        panic!("expected a C64 source");
    };
    let source_region = source
        .backend_region_view(vec![2, 3], vec![1, 4], 5)
        .unwrap();

    reset_strided_source_passes_for_test();
    gpu.with_backend_session(|__s| {
        __s.axpby_read_into_accum(
            alpha,
            TensorRead::from_view(TensorView::C64(source_region)),
            beta,
            TensorWrite::from_tensor(&mut gpu_destination),
        )
    })
    .unwrap();
    assert_eq!(
        strided_source_passes_for_test(),
        1,
        "a strided source must run the native in-place pass"
    );

    let actual = download(&gpu, &gpu_destination);
    let actual = actual.as_slice::<Complex64>().unwrap();
    let expected = expected.as_slice::<Complex64>().unwrap();
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).norm() <= 1.0e-12,
            "element {index}: {actual} != {expected}"
        );
    }
}

/// A compact operand must keep using the cuBLAS vector entry: the native
/// strided pass exists only for layouts the vendor path cannot address.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_axpby_keeps_compact_operands_on_cublas() {
    use num_complex::Complex64;
    use tenferro_tensor::{ContractionScalar, TensorRead, TensorWrite};

    use super::super::blas1::{
        reset_strided_source_passes_for_test, strided_source_passes_for_test,
    };
    use super::{gpu_backend, tensor_c64, upload};

    let mut gpu = gpu_backend();
    let host_x = tensor_c64(vec![4], vec![Complex64::new(1.0, -1.0); 4]);
    let host_y = tensor_c64(vec![4], vec![Complex64::new(0.5, 0.25); 4]);
    let gpu_x = upload(&gpu, &host_x);
    let mut gpu_y = upload(&gpu, &host_y);

    reset_strided_source_passes_for_test();
    gpu.with_backend_session(|__s| {
        __s.axpby_read_into_accum(
            ContractionScalar::C64(Complex64::new(1.0, 0.0)),
            TensorRead::from_tensor(&gpu_x),
            ContractionScalar::C64(Complex64::new(1.0, 0.0)),
            TensorWrite::from_tensor(&mut gpu_y),
        )
    })
    .unwrap();

    assert_eq!(strided_source_passes_for_test(), 0);
}

/// The reason `geam_accum` takes its coefficients by value (issue #1870).
///
/// cuBLAS reads a host-mode coefficient as its FFI type, and CUDA declares
/// `cuDoubleComplex` as `__align__(16)` `double2`. `num_complex` is only
/// 8-aligned, so a `&Complex64 -> *const cuDoubleComplex` cast is a
/// misaligned pointer whenever the stack places the value at 8 mod 16.
/// cuBLAS 12.9 faults on it; 12.1 and 12.6 happen to tolerate it. If this
/// assertion ever fails because the bindings changed, the copy in
/// `geam_accum` became unnecessary -- that is a decision to record, not a
/// line to delete silently.
#[test]
fn cublas_complex_scalars_are_more_strictly_aligned_than_num_complex() {
    assert!(
        align_of::<cublas::cuDoubleComplex>() > align_of::<Complex64>(),
        "cuDoubleComplex {} vs Complex64 {}",
        align_of::<cublas::cuDoubleComplex>(),
        align_of::<Complex64>()
    );
    assert!(
        align_of::<cublas::cuComplex>() > align_of::<Complex32>(),
        "cuComplex {} vs Complex32 {}",
        align_of::<cublas::cuComplex>(),
        align_of::<Complex32>()
    );
    // The copy is only sound because the sizes match; `geam_accum` also
    // asserts this at compile time for each instantiated pair.
    assert_eq!(size_of::<cublas::cuDoubleComplex>(), size_of::<Complex64>());
    assert_eq!(size_of::<cublas::cuComplex>(), size_of::<Complex32>());
}
