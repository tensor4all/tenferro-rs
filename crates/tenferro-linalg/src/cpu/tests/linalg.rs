use super::*;

use crate::TensorReadLinalgExt;
use std::sync::Arc;
use tenferro_tensor::{BackendStorageHandle, MemoryKind, Placement, StorageBuffer};

fn assert_c32_close(actual: Complex32, expected: Complex32) {
    assert!(
        (actual - expected).norm() < 1.0e-5,
        "expected {expected:?}, got {actual:?}"
    );
}

fn compiled_linalg_provider_kinds() -> Vec<tenferro_cpu::CpuBackendKind> {
    vec![
        #[cfg(feature = "cpu-faer")]
        tenferro_cpu::CpuBackendKind::Faer,
        #[cfg(feature = "cpu-blas")]
        tenferro_cpu::CpuBackendKind::Blas,
    ]
}

#[test]
fn solve_read_accepts_owned_and_strided_inputs_for_compiled_providers() {
    let a = Tensor::from_vec_col_major([2, 2], vec![3.0_f64, 1.0, 0.0, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major([2, 1], vec![7.0_f64, 4.0]).unwrap();
    let a_storage = vec![-1.0, 3.0, -1.0, 1.0, -1.0, -1.0, 0.0, -1.0, 2.0];
    let b_storage = vec![-1.0, 7.0, -1.0, 4.0];

    for kind in compiled_linalg_provider_kinds() {
        let mut backend = CpuBackend::with_kind(kind).unwrap();
        with_cpu_linalg(&mut backend, |backend| {
            let owned = backend
                .solve_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
                .unwrap();
            let strided = backend
                .solve_read(
                    TensorRead::from_view(TensorView::F64(
                        TypedTensorView::from_slice(vec![2, 2], vec![2, 5], 1, &a_storage).unwrap(),
                    )),
                    TensorRead::from_view(TensorView::F64(
                        TypedTensorView::from_slice(vec![2, 1], vec![2, 5], 1, &b_storage).unwrap(),
                    )),
                )
                .unwrap();

            for row in 0..2 {
                assert_f64_close(get_f64(&owned, &[row, 0]), get_f64(&strided, &[row, 0]));
            }

            let vector_b = Tensor::from_vec_col_major([2], vec![7.0_f64, 4.0]).unwrap();
            let vector_x = backend
                .solve_read(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&vector_b),
                )
                .unwrap();
            assert_eq!(vector_x.shape(), &[2]);
        });
    }
}

#[test]
fn solve_read_into_writes_owned_and_padded_column_major_outputs() {
    let a = Tensor::from_vec_col_major([2, 2], vec![3.0_f64, 1.0, 0.0, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major([2, 1], vec![7.0_f64, 4.0]).unwrap();

    for kind in compiled_linalg_provider_kinds() {
        let mut backend = CpuBackend::with_kind(kind).unwrap();
        with_cpu_linalg(&mut backend, |backend| {
            let mut owned = Tensor::from_vec_col_major([2, 1], vec![-9.0_f64; 2]).unwrap();
            backend
                .solve_read_into(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&b),
                    tenferro_tensor::TensorWrite::from_tensor(&mut owned),
                )
                .unwrap();
            assert_f64_close(get_f64(&owned, &[0, 0]), 7.0 / 3.0);
            assert_f64_close(get_f64(&owned, &[1, 0]), 5.0 / 6.0);

            let mut storage = vec![-17.0_f64; 8];
            let view =
                tenferro_tensor::TypedTensorViewMut::from_slice([2, 1], [1, 4], 1, &mut storage)
                    .unwrap();
            backend
                .solve_read_into(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&b),
                    tenferro_tensor::TensorWrite::from_view(tenferro_tensor::TensorViewMut::F64(
                        view,
                    )),
                )
                .unwrap();
            assert_f64_close(storage[1], 7.0 / 3.0);
            assert_f64_close(storage[2], 5.0 / 6.0);
            assert_eq!(storage[0], -17.0);
            assert_eq!(storage[5], -17.0);
        });
    }
}

#[test]
fn solve_read_into_rejects_bad_destination_before_mutation() {
    let a = Tensor::from_vec_col_major([2, 2], vec![3.0_f64, 1.0, 0.0, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major([2, 1], vec![7.0_f64, 4.0]).unwrap();

    for kind in compiled_linalg_provider_kinds() {
        let mut backend = CpuBackend::with_kind(kind).unwrap();
        with_cpu_linalg(&mut backend, |backend| {
            let mut out = Tensor::from_vec_col_major([3, 1], vec![-23.0_f64; 3]).unwrap();
            let error = backend
                .solve_read_into(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&b),
                    tenferro_tensor::TensorWrite::from_tensor(&mut out),
                )
                .unwrap_err();
            assert!(matches!(error, tenferro_tensor::Error::Validation { .. }));
            match out {
                Tensor::F64(out) => assert_eq!(out.host_data().unwrap(), &[-23.0, -23.0, -23.0]),
                _ => unreachable!("test output is f64"),
            }
        });
    }
}

#[test]
fn solve_read_into_covers_public_extension_complex_rhs_and_singular_atomicity() {
    let a = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let b = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 4.0, 4.0, 8.0]).unwrap();
    let mut out = Tensor::from_vec_col_major([2, 2], vec![-31.0_f64; 4]).unwrap();
    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        TensorRead::from_tensor(&a)
            .solve_read_into(
                TensorRead::from_tensor(&b),
                tenferro_tensor::TensorWrite::from_tensor(&mut out),
                backend,
            )
            .unwrap();
        assert_f64_close(get_f64(&out, &[0, 0]), 1.0);
        assert_f64_close(get_f64(&out, &[1, 0]), 1.0);
        assert_f64_close(get_f64(&out, &[0, 1]), 2.0);
        assert_f64_close(get_f64(&out, &[1, 1]), 2.0);

        let a = Tensor::from_vec_col_major(
            [2, 2],
            vec![
                Complex32::new(2.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(4.0, 0.0),
            ],
        )
        .unwrap();
        let b = Tensor::from_vec_col_major(
            [2, 2],
            vec![
                Complex32::new(2.0, 2.0),
                Complex32::new(4.0, 4.0),
                Complex32::new(4.0, -2.0),
                Complex32::new(8.0, -4.0),
            ],
        )
        .unwrap();
        let mut storage = vec![Complex32::new(-19.0, 0.0); 8];
        let view = tenferro_tensor::TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut storage)
            .unwrap();
        backend
            .solve_read_into(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                tenferro_tensor::TensorWrite::from_view(tenferro_tensor::TensorViewMut::C32(view)),
            )
            .unwrap();
        assert_c32_close(storage[1], Complex32::new(1.0, 1.0));
        assert_c32_close(storage[2], Complex32::new(1.0, 1.0));
        assert_c32_close(storage[4], Complex32::new(2.0, -1.0));
        assert_c32_close(storage[5], Complex32::new(2.0, -1.0));
        assert_eq!(storage[0], Complex32::new(-19.0, 0.0));
        assert_eq!(storage[3], Complex32::new(-19.0, 0.0));

        let singular = Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();
        let rhs = Tensor::from_vec_col_major([2, 1], vec![3.0_f64, 6.0]).unwrap();
        let mut sentinel = Tensor::from_vec_col_major([2, 1], vec![-37.0_f64; 2]).unwrap();
        let error = backend
            .solve_read_into(
                TensorRead::from_tensor(&singular),
                TensorRead::from_tensor(&rhs),
                tenferro_tensor::TensorWrite::from_tensor(&mut sentinel),
            )
            .unwrap_err();
        assert_eq!(error.kind(), tenferro_tensor::ErrorKind::NumericalFailure);
        assert_eq!(get_f64(&sentinel, &[0, 0]), -37.0);
        assert_eq!(get_f64(&sentinel, &[1, 0]), -37.0);
    });
}

#[test]
fn output_from_rhs_view_covers_vector_matrix_and_rank_validation() {
    let vector = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 5.0]).unwrap();
    let mut matrix_storage = vec![-1.0; 8];
    matrix_storage[1] = 7.0;
    matrix_storage[2] = 11.0;
    matrix_storage[5] = 13.0;
    matrix_storage[6] = 17.0;
    let matrix = TypedTensorView::from_slice(vec![2, 2], vec![1, 4], 1, &matrix_storage).unwrap();
    let rank3_storage = [19.0];
    let rank3 =
        TypedTensorView::from_slice(vec![1, 1, 1], vec![1, 1, 1], 0, &rank3_storage).unwrap();
    let mut backend = CpuBackend::new();

    with_cpu_linalg(&mut backend, |backend| {
        backend
            .with_linalg_pool(|_, buffers| {
                let vector_output = crate::cpu::linalg::output_from_rhs_view(
                    buffers,
                    &vector.as_view(),
                    "test_output_from_rhs_view",
                )?;
                assert_eq!(vector_output.host_data()?, &[3.0, 5.0]);

                let matrix_output = crate::cpu::linalg::output_from_rhs_view(
                    buffers,
                    &matrix,
                    "test_output_from_rhs_view",
                )?;
                assert_eq!(matrix_output.host_data()?, &[7.0, 11.0, 13.0, 17.0]);

                let error = crate::cpu::linalg::output_from_rhs_view(
                    buffers,
                    &rank3,
                    "test_output_from_rhs_view",
                )
                .unwrap_err();
                assert!(matches!(
                    error,
                    tenferro_tensor::Error::Validation {
                        source: tenferro_tensor::ValidationError::RankMismatch {
                            expected: 2,
                            actual: 3,
                        },
                        ..
                    }
                ));

                let backend_vector = TypedTensor::<f64>::from_buffer_col_major(
                    vec![2],
                    StorageBuffer::Backend(Arc::new(BackendStorageHandle::<f64>::new_with_len(
                        120, 2,
                    ))),
                    Placement {
                        memory_kind: MemoryKind::Device,
                        device: None,
                        cpu_affinity: None,
                    },
                )?;
                let backend_vector_view =
                    backend_vector.backend_region_view(vec![2], vec![1], 0)?;
                let error = crate::cpu::linalg::output_from_rhs_view(
                    buffers,
                    &backend_vector_view,
                    "test_output_from_rhs_view",
                )
                .unwrap_err();
                assert!(matches!(
                    error,
                    tenferro_tensor::Error::RuntimeState {
                        op: "test_output_from_rhs_view",
                        ..
                    }
                ));

                let backend_matrix = TypedTensor::<f64>::from_buffer_col_major(
                    vec![2, 2],
                    StorageBuffer::Backend(Arc::new(BackendStorageHandle::<f64>::new_with_len(
                        121, 4,
                    ))),
                    Placement {
                        memory_kind: MemoryKind::Device,
                        device: None,
                        cpu_affinity: None,
                    },
                )?;
                let backend_matrix_view =
                    backend_matrix.backend_region_view(vec![2, 2], vec![1, 2], 0)?;
                let error = crate::cpu::linalg::output_from_rhs_view(
                    buffers,
                    &backend_matrix_view,
                    "test_output_from_rhs_view",
                )
                .unwrap_err();
                assert!(matches!(
                    error,
                    tenferro_tensor::Error::RuntimeState {
                        op: "test_output_from_rhs_view",
                        ..
                    }
                ));
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn solve_accepts_tiny_nonzero_real_and_complex_pivots_for_compiled_providers() {
    // What: ordinary and prepared solve treat scaling as units, not as an implicit rank cutoff.
    for kind in compiled_linalg_provider_kinds() {
        let mut backend = CpuBackend::with_kind(kind).unwrap();
        with_cpu_linalg(&mut backend, |backend| {
            let scale = 2.0_f32.powi(-80);
            let a = Tensor::from_vec_col_major([2, 2], vec![scale, 0.0, 0.0, 2.0 * scale]).unwrap();
            let b = Tensor::from_vec_col_major([2, 1], vec![3.0 * scale, 10.0 * scale]).unwrap();
            let direct = backend.solve(&a, &b).unwrap();
            assert!((get_f32(&direct, &[0, 0]) - 3.0).abs() < 1.0e-5);
            assert!((get_f32(&direct, &[1, 0]) - 5.0).abs() < 1.0e-5);
            let read = backend
                .solve_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
                .unwrap();
            assert!((get_f32(&read, &[0, 0]) - 3.0).abs() < 1.0e-5);
            let factors = backend.lu_factor(&a).unwrap();
            let prepared = backend
                .lu_solve_prepared(&a, &factors[0], &factors[1], &b, false, false)
                .unwrap();
            assert!((get_f32(&prepared, &[1, 0]) - 5.0).abs() < 1.0e-5);

            let scale = 2.0_f64.powi(-600);
            let a = Tensor::from_vec_col_major([2, 2], vec![scale, 0.0, 0.0, 2.0 * scale]).unwrap();
            let b = Tensor::from_vec_col_major([2, 1], vec![3.0 * scale, 10.0 * scale]).unwrap();
            let direct = backend.solve(&a, &b).unwrap();
            assert_f64_close(get_f64(&direct, &[0, 0]), 3.0);
            assert_f64_close(get_f64(&direct, &[1, 0]), 5.0);
            let factors = backend.lu_factor(&a).unwrap();
            let prepared = backend
                .lu_solve_prepared(&a, &factors[0], &factors[1], &b, false, false)
                .unwrap();
            assert_f64_close(get_f64(&prepared, &[1, 0]), 5.0);

            let scale = 2.0_f32.powi(-80);
            let zero = Complex32::new(0.0, 0.0);
            let a = Tensor::from_vec_col_major(
                [2, 2],
                vec![
                    Complex32::new(scale, 0.0),
                    zero,
                    zero,
                    Complex32::new(2.0 * scale, 0.0),
                ],
            )
            .unwrap();
            let b = Tensor::from_vec_col_major(
                [2, 1],
                vec![
                    Complex32::new(3.0 * scale, 0.0),
                    Complex32::new(10.0 * scale, 0.0),
                ],
            )
            .unwrap();
            let direct = backend.solve(&a, &b).unwrap();
            assert!((get_c32(&direct, &[0, 0]) - Complex32::new(3.0, 0.0)).norm() < 1.0e-5);
            let factors = backend.lu_factor(&a).unwrap();
            let prepared = backend
                .lu_solve_prepared(&a, &factors[0], &factors[1], &b, false, false)
                .unwrap();
            assert!((get_c32(&prepared, &[1, 0]) - Complex32::new(5.0, 0.0)).norm() < 1.0e-5);

            let scale = 2.0_f64.powi(-600);
            let zero = Complex64::new(0.0, 0.0);
            let a = Tensor::from_vec_col_major(
                [2, 2],
                vec![
                    Complex64::new(0.0, scale),
                    zero,
                    zero,
                    Complex64::new(0.0, 2.0 * scale),
                ],
            )
            .unwrap();
            let b = Tensor::from_vec_col_major(
                [2, 1],
                vec![
                    Complex64::new(0.0, 3.0 * scale),
                    Complex64::new(0.0, 10.0 * scale),
                ],
            )
            .unwrap();
            let direct = backend.solve(&a, &b).unwrap();
            assert_c64_close(get_c64(&direct, &[0, 0]), Complex64::new(3.0, 0.0));
            let factors = backend.lu_factor(&a).unwrap();
            let prepared = backend
                .lu_solve_prepared(&a, &factors[0], &factors[1], &b, false, false)
                .unwrap();
            assert_c64_close(get_c64(&prepared, &[1, 0]), Complex64::new(5.0, 0.0));
        });
    }
}

#[test]
fn triangular_solve_read_accepts_owned_and_strided_inputs_for_compiled_providers() {
    let a = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 1.0, 3.0]).unwrap();
    let b = Tensor::from_vec_col_major([2, 1], vec![5.0_f64, 6.0]).unwrap();
    let a_storage = vec![-1.0, 2.0, -1.0, 0.0, -1.0, -1.0, 1.0, -1.0, 3.0];
    let b_storage = vec![-1.0, 5.0, -1.0, 6.0];

    for kind in compiled_linalg_provider_kinds() {
        let mut backend = CpuBackend::with_kind(kind).unwrap();
        with_cpu_linalg(&mut backend, |backend| {
            let owned = backend
                .triangular_solve_read(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&b),
                    true,
                    false,
                    false,
                    false,
                )
                .unwrap();
            let strided = backend
                .triangular_solve_read(
                    TensorRead::from_view(TensorView::F64(
                        TypedTensorView::from_slice(vec![2, 2], vec![2, 5], 1, &a_storage).unwrap(),
                    )),
                    TensorRead::from_view(TensorView::F64(
                        TypedTensorView::from_slice(vec![2, 1], vec![2, 5], 1, &b_storage).unwrap(),
                    )),
                    true,
                    false,
                    false,
                    false,
                )
                .unwrap();

            for row in 0..2 {
                assert_f64_close(get_f64(&owned, &[row, 0]), get_f64(&strided, &[row, 0]));
            }
        });
    }
}

#[test]
fn svd_read_accepts_an_owned_tensor_read() {
    let input = Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();

    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.svd_read(TensorRead::from_tensor(&input))
    })
    .unwrap();

    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[1].shape(), &[2]);
}

#[test]
fn lapack_batched_value_factor_paths_reuse_pooled_batch_input() {
    let lu_source = include_str!("../linalg/lapack_linalg/lu.rs");
    let lu_factor = source_from(lu_source, "pub(crate) fn lu_factor");
    assert!(lu_factor.contains("tensor_from_pooled_slice_with_template("));
    assert!(lu_factor.contains("refill_tensor_from_slice("));
    assert!(!lu_factor.contains("input.host_data()?[range].to_vec()"));

    let eigh_source = include_str!("../linalg/lapack_linalg/eigh.rs");
    let eigh_values = source_from(eigh_source, "pub(crate) fn eigh_values");
    assert!(eigh_values.contains("batched_multi_convert(\"eigh_values\""));
    assert!(!eigh_values.contains("input.host_data()?[range].to_vec()"));

    let svd_source = include_str!("../linalg/lapack_linalg/svd.rs");
    let svd_values = source_from(svd_source, "pub(crate) fn svd_values");
    assert!(svd_values.contains("batched_multi_convert(\"svd_values\""));
    assert!(!svd_values.contains("input.host_data()?[range].to_vec()"));
}

#[test]
fn svd_canonicalizes_transposed_host_view_before_lapack() {
    let data = vec![1.0, -2.0, 3.0, 0.5, -1.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.svd_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();

    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);

    let u = matrix_f64_from_tensor(&outputs[0], 3, 2);
    let s = (0..2)
        .map(|i| get_f64(&outputs[1], &[i]))
        .collect::<Vec<_>>();
    let vt = matrix_f64_from_tensor(&outputs[2], 2, 2);
    let recon = matmul_f64(&matmul_f64(&u, &diag_f64(&s), 3, 2, 2), &vt, 3, 2, 2);
    let expected = transpose_f64(&data, 2, 3);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

fn source_from<'a>(source: &'a str, start: &str) -> &'a str {
    let start_idx = source
        .find(start)
        .unwrap_or_else(|| panic!("missing source start marker: {start}"));
    &source[start_idx..]
}

#[test]
fn qr_read_canonicalizes_transposed_host_view_before_lapack() {
    let data = vec![1.0, -2.0, 3.0, 0.5, -1.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.qr_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let q = matrix_f64_from_tensor(&outputs[0], 3, 2);
    let r = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let recon = matmul_f64(&q, &r, 3, 2, 2);
    let expected = transpose_f64(&data, 2, 3);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn qr_read_canonicalizes_transposed_complex_host_view_before_lapack() {
    let data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, -0.25),
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.0, 0.75),
        Complex64::new(4.0, 1.5),
    ];
    let a = TypedTensor::<Complex64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.qr_read(TensorRead::from_view(TensorView::C64(view)))
    })
    .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let q = matrix_c64_from_tensor(&outputs[0], 3, 2);
    let r = matrix_c64_from_tensor(&outputs[1], 2, 2);
    let recon = matmul_c64(&q, &r, 3, 2, 2);
    let expected = transpose_c64(&data, 2, 3);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn eigh_read_canonicalizes_transposed_host_view_before_lapack() {
    let data = vec![4.0, 1.0, 1.0, 3.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.eigh_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let values = match &outputs[0] {
        Tensor::F64(inner) => inner.host_data().unwrap().to_vec(),
        _ => panic!("expected f64 eigenvalues"),
    };
    let vectors = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let recon = matmul_f64(
        &matmul_f64(&vectors, &diag_f64(&values), 2, 2, 2),
        &transpose_f64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_f64(&data, 2, 2);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn eigh_read_canonicalizes_transposed_complex_host_view_before_lapack() {
    let data = vec![
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, -0.5),
        Complex64::new(1.0, 0.5),
        Complex64::new(3.0, 0.0),
    ];
    let a = TypedTensor::<Complex64>::from_vec_col_major(vec![2, 2], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.eigh_read(TensorRead::from_view(TensorView::C64(view)))
    })
    .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].dtype(), DType::F64);
    assert_eq!(outputs[1].dtype(), DType::C64);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let values = vector_f64_from_tensor(&outputs[0], 2);
    let vectors = matrix_c64_from_tensor(&outputs[1], 2, 2);
    let recon = matmul_c64(
        &matmul_c64(&vectors, &diag_c64_from_real(&values), 2, 2, 2),
        &conjugate_transpose_c64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_c64(&data, 2, 2);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn qr_read_accepts_all_supported_linalg_view_dtypes() {
    let f32_input =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0, -2.0, 0.5, 4.0]).unwrap();
    let f64_input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, -2.0, 0.5, 4.0]).unwrap();
    let c32_input = TypedTensor::<Complex32>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(-2.0, 1.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(4.0, 1.5),
        ],
    )
    .unwrap();
    let c64_input = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(4.0, 1.5),
        ],
    )
    .unwrap();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        for (view, dtype) in [
            (
                TensorRead::from_view(TensorView::F32(
                    f32_input.as_view().transpose_view([1, 0]).unwrap(),
                )),
                DType::F32,
            ),
            (
                TensorRead::from_view(TensorView::F64(
                    f64_input.as_view().transpose_view([1, 0]).unwrap(),
                )),
                DType::F64,
            ),
            (
                TensorRead::from_view(TensorView::C32(
                    c32_input.as_view().transpose_view([1, 0]).unwrap(),
                )),
                DType::C32,
            ),
            (
                TensorRead::from_view(TensorView::C64(
                    c64_input.as_view().transpose_view([1, 0]).unwrap(),
                )),
                DType::C64,
            ),
        ] {
            let outputs = backend.qr_read(view).unwrap();
            assert_eq!(outputs.len(), 2);
            assert_eq!(outputs[0].dtype(), dtype);
            assert_eq!(outputs[1].dtype(), dtype);
            assert_eq!(outputs[0].shape(), &[2, 2]);
            assert_eq!(outputs[1].shape(), &[2, 2]);
        }
    });
}

#[test]
fn eigh_read_accepts_all_supported_linalg_view_dtypes() {
    let f32_input =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let f64_input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let c32_input = TypedTensor::<Complex32>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, -0.5),
            Complex32::new(1.0, 0.5),
            Complex32::new(3.0, 0.0),
        ],
    )
    .unwrap();
    let c64_input = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(1.0, 0.5),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        for (view, value_dtype, vector_dtype) in [
            (
                TensorRead::from_view(TensorView::F32(
                    f32_input.as_view().transpose_view([1, 0]).unwrap(),
                )),
                DType::F32,
                DType::F32,
            ),
            (
                TensorRead::from_view(TensorView::F64(
                    f64_input.as_view().transpose_view([1, 0]).unwrap(),
                )),
                DType::F64,
                DType::F64,
            ),
            (
                TensorRead::from_view(TensorView::C32(
                    c32_input.as_view().transpose_view([1, 0]).unwrap(),
                )),
                DType::F32,
                DType::C32,
            ),
            (
                TensorRead::from_view(TensorView::C64(
                    c64_input.as_view().transpose_view([1, 0]).unwrap(),
                )),
                DType::F64,
                DType::C64,
            ),
        ] {
            let outputs = backend.eigh_read(view).unwrap();
            assert_eq!(outputs.len(), 2);
            assert_eq!(outputs[0].dtype(), value_dtype);
            assert_eq!(outputs[1].dtype(), vector_dtype);
            assert_eq!(outputs[0].shape(), &[2]);
            assert_eq!(outputs[1].shape(), &[2, 2]);
        }
    });
}

#[test]
fn linalg_read_rejects_non_float_view_dtypes() {
    fn assert_unsupported_dtype(
        result: tenferro_tensor::Result<Vec<Tensor>>,
        expected_op: &'static str,
    ) {
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            tenferro_tensor::Error::Extension {
                op,
                family: crate::LINALG_EXTENSION_FAMILY_ID,
                kind: tenferro_tensor::ErrorKind::Unsupported,
                ..
            } if op == expected_op
        ));
        assert!(std::error::Error::source(&err)
            .and_then(|source| source.downcast_ref::<crate::Error>())
            .is_some_and(|source| matches!(
                source,
                crate::Error::UnsupportedDType { op, .. } if *op == expected_op
            )));
    }

    let i32_input = TypedTensor::<i32>::from_vec_col_major(vec![2, 2], vec![1, 0, 0, 1]).unwrap();
    let i64_input =
        TypedTensor::<i64>::from_vec_col_major(vec![2, 2], vec![1_i64, 0, 0, 1]).unwrap();
    let bool_input =
        TypedTensor::<bool>::from_vec_col_major(vec![2, 2], vec![true, false, false, true])
            .unwrap();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        assert_unsupported_dtype(
            backend.svd_read(TensorRead::from_view(TensorView::I32(i32_input.as_view()))),
            "svd",
        );
        assert_unsupported_dtype(
            backend.svd_read(TensorRead::from_view(TensorView::I64(i64_input.as_view()))),
            "svd",
        );
        assert_unsupported_dtype(
            backend.svd_read(TensorRead::from_view(TensorView::Bool(
                bool_input.as_view(),
            ))),
            "svd",
        );
        assert_unsupported_dtype(
            backend.qr_read(TensorRead::from_view(TensorView::I32(i32_input.as_view()))),
            "qr",
        );
        assert_unsupported_dtype(
            backend.qr_read(TensorRead::from_view(TensorView::I64(i64_input.as_view()))),
            "qr",
        );
        assert_unsupported_dtype(
            backend.qr_read(TensorRead::from_view(TensorView::Bool(
                bool_input.as_view(),
            ))),
            "qr",
        );
        assert_unsupported_dtype(
            backend.eigh_read(TensorRead::from_view(TensorView::I32(i32_input.as_view()))),
            "eigh",
        );
        assert_unsupported_dtype(
            backend.eigh_read(TensorRead::from_view(TensorView::I64(i64_input.as_view()))),
            "eigh",
        );
        assert_unsupported_dtype(
            backend.eigh_read(TensorRead::from_view(TensorView::Bool(
                bool_input.as_view(),
            ))),
            "eigh",
        );

        fn assert_unsupported_dtype_single(
            result: tenferro_tensor::Result<Tensor>,
            expected_op: &'static str,
        ) {
            let err = result.unwrap_err();
            assert!(matches!(
                err,
                tenferro_tensor::Error::Extension {
                    op,
                    family: crate::LINALG_EXTENSION_FAMILY_ID,
                    kind: tenferro_tensor::ErrorKind::Unsupported,
                    ..
                } if op == expected_op
            ));
            assert!(std::error::Error::source(&err)
                .and_then(|source| source.downcast_ref::<crate::Error>())
                .is_some_and(|source| matches!(
                    source,
                    crate::Error::UnsupportedDType { op, .. } if *op == expected_op
                )));
        }

        assert_unsupported_dtype_single(
            backend.cholesky_read(TensorRead::from_view(TensorView::I32(i32_input.as_view()))),
            "cholesky",
        );
        assert_unsupported_dtype_single(
            backend.cholesky_read(TensorRead::from_view(TensorView::I64(i64_input.as_view()))),
            "cholesky",
        );
        assert_unsupported_dtype_single(
            backend.cholesky_read(TensorRead::from_view(TensorView::Bool(
                bool_input.as_view(),
            ))),
            "cholesky",
        );
        assert_unsupported_dtype(
            backend.lu_read(TensorRead::from_view(TensorView::I32(i32_input.as_view()))),
            "lu",
        );
        assert_unsupported_dtype(
            backend.lu_read(TensorRead::from_view(TensorView::I64(i64_input.as_view()))),
            "lu",
        );
        assert_unsupported_dtype(
            backend.lu_read(TensorRead::from_view(TensorView::Bool(
                bool_input.as_view(),
            ))),
            "lu",
        );
        assert_unsupported_dtype(
            backend.full_piv_lu_read(TensorRead::from_view(TensorView::I32(i32_input.as_view()))),
            "full_piv_lu",
        );
        assert_unsupported_dtype(
            backend.full_piv_lu_read(TensorRead::from_view(TensorView::I64(i64_input.as_view()))),
            "full_piv_lu",
        );
        assert_unsupported_dtype(
            backend.full_piv_lu_read(TensorRead::from_view(TensorView::Bool(
                bool_input.as_view(),
            ))),
            "full_piv_lu",
        );
        assert_unsupported_dtype(
            backend.eig_read(TensorRead::from_view(TensorView::I32(i32_input.as_view()))),
            "eig",
        );
        assert_unsupported_dtype(
            backend.eig_read(TensorRead::from_view(TensorView::I64(i64_input.as_view()))),
            "eig",
        );
        assert_unsupported_dtype(
            backend.eig_read(TensorRead::from_view(TensorView::Bool(
                bool_input.as_view(),
            ))),
            "eig",
        );
    });
}

#[test]
fn cholesky_read_canonicalizes_transposed_host_view_before_factorization() {
    // A = L * L^T where L = [[2, 0], [1, 3]] => A = [[4, 2], [2, 10]]
    // column-major storage: [4, 2, 2, 10]
    let data = vec![4.0_f64, 2.0, 2.0, 10.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], data.clone()).unwrap();
    // Transposed view of a symmetric matrix is still the same matrix, so cholesky should succeed.
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();
    let output = with_cpu_linalg(&mut backend, |backend| {
        backend.cholesky_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    // Verify L * L^T ≈ A
    let l = matrix_f64_from_tensor(&output, 2, 2);
    let recon = matmul_f64(&l, &transpose_f64(&l, 2, 2), 2, 2, 2);
    for (actual, expected) in recon.iter().zip(data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn lu_read_canonicalizes_transposed_host_view_before_factorization() {
    // col-major [1, 2, 3, 4] => matrix [[1, 3], [2, 4]]; the transposed view is
    // [[1, 2], [3, 4]] (col-major [1, 3, 2, 4]), which is what the fast path must factor.
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], data).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.lu_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();
    assert_eq!(outputs.len(), 4);
    // P has shape [2, 2], L has shape [2, 2], U has shape [2, 2], parity is scalar []
    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
    assert_eq!(outputs[3].shape(), &[] as &[usize]);

    // Reconstruct P * A_view == L * U (the canonical LU convention).
    let a_view = vec![1.0_f64, 3.0, 2.0, 4.0];
    let p = matrix_f64_from_tensor(&outputs[0], 2, 2);
    let l = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let u = matrix_f64_from_tensor(&outputs[2], 2, 2);
    let pa = matmul_f64(&p, &a_view, 2, 2, 2);
    let lu = matmul_f64(&l, &u, 2, 2, 2);
    for (actual, expected) in lu.iter().zip(pa.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn full_piv_lu_read_canonicalizes_transposed_host_view_before_factorization() {
    // col-major [1, 2, 3, 4] => matrix [[1, 3], [2, 4]]; transposed view is col-major [1, 3, 2, 4].
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], data).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.full_piv_lu_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();
    assert_eq!(outputs.len(), 5);
    // P_row, L, U, P_col, parity
    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
    assert_eq!(outputs[3].shape(), &[2, 2]);
    assert_eq!(outputs[4].shape(), &[] as &[usize]);

    // Reconstruct P * A_view * Q^T == L * U (the complete-pivot convention).
    let a_view = vec![1.0_f64, 3.0, 2.0, 4.0];
    let p = matrix_f64_from_tensor(&outputs[0], 2, 2);
    let l = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let u = matrix_f64_from_tensor(&outputs[2], 2, 2);
    let q = matrix_f64_from_tensor(&outputs[3], 2, 2);
    let pa = matmul_f64(&p, &a_view, 2, 2, 2);
    let paqt = matmul_f64(&pa, &transpose_f64(&q, 2, 2), 2, 2, 2);
    let lu = matmul_f64(&l, &u, 2, 2, 2);
    for (actual, expected) in lu.iter().zip(paqt.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn eig_read_returns_correct_outputs_for_diagonal_matrix() {
    // diagonal matrix [[2, 0], [0, 3]] col-major: [2, 0, 0, 3]
    let data = vec![2.0_f64, 0.0, 0.0, 3.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], data).unwrap();
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| {
        backend.eig_read(TensorRead::from_view(TensorView::F64(a.as_view())))
    })
    .unwrap();
    assert_eq!(outputs.len(), 2);
    // eig on real returns complex outputs
    assert!(matches!(outputs[0], Tensor::C64(_)));
    assert!(matches!(outputs[1], Tensor::C64(_)));
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
}

macro_rules! real_rotation_eig_residual_test {
    ($name:ident, $real:ty, $real_variant:ident, $get_complex:ident, $tol:expr) => {
        #[test]
        fn $name() {
            let input = Tensor::$real_variant(
                TypedTensor::<$real>::from_vec_col_major(
                    vec![2, 2],
                    vec![0.0 as $real, 1.0 as $real, -1.0 as $real, 0.0 as $real],
                )
                .unwrap(),
            );

            for kind in compiled_linalg_provider_kinds() {
                let mut backend = CpuBackend::with_threads_and_kind(1, kind).unwrap();
                let outputs = with_cpu_linalg(&mut backend, |backend| backend.eig(&input)).unwrap();
                let mut residual_squared = 0.0_f64;
                let mut vector_squared = 0.0_f64;
                let mut diagonal_squared = 0.0_f64;
                for column in 0..2 {
                    let v0 = $get_complex(&outputs[1], &[0, column]);
                    let v1 = $get_complex(&outputs[1], &[1, column]);
                    let lambda = $get_complex(&outputs[0], &[column]);
                    residual_squared +=
                        ((-v1 - v0 * lambda).norm_sqr() + (v0 - v1 * lambda).norm_sqr()) as f64;
                    vector_squared += (v0.norm_sqr() + v1.norm_sqr()) as f64;
                    diagonal_squared += lambda.norm_sqr() as f64;
                }
                let vector_norm = vector_squared.sqrt();
                let relative_residual = residual_squared.sqrt()
                    / (2.0_f64.sqrt() * vector_norm + vector_norm * diagonal_squared.sqrt());

                assert!(
                    relative_residual.is_finite() && relative_residual <= $tol,
                    "{kind:?} relative AV-VD residual {relative_residual:e} exceeds {}",
                    $tol
                );
            }
        }
    };
}

real_rotation_eig_residual_test!(
    real_f32_rotation_eig_has_column_major_eigenvectors,
    f32,
    F32,
    get_c32,
    1.0e-6
);
real_rotation_eig_residual_test!(
    real_f64_rotation_eig_has_column_major_eigenvectors,
    f64,
    F64,
    get_c64,
    1.0e-14
);

#[test]
fn cholesky_read_accepts_all_supported_linalg_view_dtypes() {
    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        let spd_f32 =
            TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![4.0_f32, 2.0, 2.0, 3.0])
                .unwrap();
        let spd_f64 =
            TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])
                .unwrap();
        let spd_c32 = TypedTensor::<Complex32>::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex32::new(4.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
            ],
        )
        .unwrap();
        let spd_c64 = TypedTensor::<Complex64>::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        )
        .unwrap();

        let out = backend
            .cholesky_read(TensorRead::from_view(TensorView::F32(spd_f32.as_view())))
            .unwrap();
        assert!(matches!(out, Tensor::F32(_)));
        assert_eq!(out.shape(), &[2, 2]);

        let out = backend
            .cholesky_read(TensorRead::from_view(TensorView::F64(spd_f64.as_view())))
            .unwrap();
        assert!(matches!(out, Tensor::F64(_)));
        assert_eq!(out.shape(), &[2, 2]);

        let out = backend
            .cholesky_read(TensorRead::from_view(TensorView::C32(spd_c32.as_view())))
            .unwrap();
        assert!(matches!(out, Tensor::C32(_)));
        assert_eq!(out.shape(), &[2, 2]);

        let out = backend
            .cholesky_read(TensorRead::from_view(TensorView::C64(spd_c64.as_view())))
            .unwrap();
        assert!(matches!(out, Tensor::C64(_)));
        assert_eq!(out.shape(), &[2, 2]);
    });
}

#[test]
fn lu_read_accepts_all_supported_linalg_view_dtypes() {
    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        let mat_f32 =
            TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0])
                .unwrap();
        let mat_f64 =
            TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])
                .unwrap();
        let mat_c32 = TypedTensor::<Complex32>::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
            ],
        )
        .unwrap();
        let mat_c64 = TypedTensor::<Complex64>::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();

        let outs = backend
            .lu_read(TensorRead::from_view(TensorView::F32(mat_f32.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 4);
        assert!(matches!(outs[0], Tensor::F32(_)));

        let outs = backend
            .lu_read(TensorRead::from_view(TensorView::F64(mat_f64.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 4);
        assert!(matches!(outs[0], Tensor::F64(_)));

        let outs = backend
            .lu_read(TensorRead::from_view(TensorView::C32(mat_c32.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 4);
        assert!(matches!(outs[0], Tensor::C32(_)));

        let outs = backend
            .lu_read(TensorRead::from_view(TensorView::C64(mat_c64.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 4);
        assert!(matches!(outs[0], Tensor::C64(_)));
    });
}

#[test]
fn full_piv_lu_read_accepts_all_supported_linalg_view_dtypes() {
    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        let mat_f32 =
            TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0])
                .unwrap();
        let mat_f64 =
            TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])
                .unwrap();
        let mat_c32 = TypedTensor::<Complex32>::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, 0.0),
                Complex32::new(4.0, 0.0),
            ],
        )
        .unwrap();
        let mat_c64 = TypedTensor::<Complex64>::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();

        let outs = backend
            .full_piv_lu_read(TensorRead::from_view(TensorView::F32(mat_f32.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 5);
        assert!(matches!(outs[0], Tensor::F32(_)));

        let outs = backend
            .full_piv_lu_read(TensorRead::from_view(TensorView::F64(mat_f64.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 5);
        assert!(matches!(outs[0], Tensor::F64(_)));

        let outs = backend
            .full_piv_lu_read(TensorRead::from_view(TensorView::C32(mat_c32.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 5);
        assert!(matches!(outs[0], Tensor::C32(_)));

        let outs = backend
            .full_piv_lu_read(TensorRead::from_view(TensorView::C64(mat_c64.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 5);
        assert!(matches!(outs[0], Tensor::C64(_)));
    });
}

#[test]
fn eig_read_accepts_all_supported_linalg_view_dtypes() {
    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        // Use diagonal matrices so eig is well-conditioned
        let mat_f32 =
            TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![2.0_f32, 0.0, 0.0, 3.0])
                .unwrap();
        let mat_f64 =
            TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0])
                .unwrap();
        let mat_c32 = TypedTensor::<Complex32>::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex32::new(2.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(3.0, 0.0),
            ],
        )
        .unwrap();
        let mat_c64 = TypedTensor::<Complex64>::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        )
        .unwrap();

        // f32 -> C32 outputs
        let outs = backend
            .eig_read(TensorRead::from_view(TensorView::F32(mat_f32.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 2);
        assert!(matches!(outs[0], Tensor::C32(_)));
        assert!(matches!(outs[1], Tensor::C32(_)));

        // f64 -> C64 outputs
        let outs = backend
            .eig_read(TensorRead::from_view(TensorView::F64(mat_f64.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 2);
        assert!(matches!(outs[0], Tensor::C64(_)));
        assert!(matches!(outs[1], Tensor::C64(_)));

        // C32 -> C32 outputs
        let outs = backend
            .eig_read(TensorRead::from_view(TensorView::C32(mat_c32.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 2);
        assert!(matches!(outs[0], Tensor::C32(_)));
        assert!(matches!(outs[1], Tensor::C32(_)));

        // C64 -> C64 outputs
        let outs = backend
            .eig_read(TensorRead::from_view(TensorView::C64(mat_c64.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 2);
        assert!(matches!(outs[0], Tensor::C64(_)));
        assert!(matches!(outs[1], Tensor::C64(_)));
    });
}

#[test]
fn test_batched_cholesky() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);

    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![3, 3, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| backend.cholesky(&input)).unwrap();

    assert_eq!(out.shape(), &[3, 3, 2]);
    for batch_idx in 0..2 {
        let l = batch_matrix_f64_from_tensor(&out, 3, 3, batch_idx);
        let recon = matmul_f64(&l, &transpose_f64(&l, 3, 3), 3, 3, 3);
        let expected = batch_matrix_f64_from_tensor(&input, 3, 3, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_batched_svd() {
    let a0 = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 1.5, 2.0, 0.0, 1.0, -0.5];
    let a1 = vec![
        2.0, -1.0, 0.5, 3.0, -0.25, 1.5, -2.0, 0.75, 1.0, 2.5, -1.0, 4.0,
    ];
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![4, 3, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| backend.svd(&input)).unwrap();

    assert_eq!(out.len(), 3);
    assert_eq!(out[0].shape(), &[4, 3, 2]);
    assert_eq!(out[1].shape(), &[3, 2]);
    assert_eq!(out[2].shape(), &[3, 3, 2]);

    for batch_idx in 0..2 {
        let u = batch_matrix_f64_from_tensor(&out[0], 4, 3, batch_idx);
        let s = batch_vector_f64_from_tensor(&out[1], 3, batch_idx);
        let vt = batch_matrix_f64_from_tensor(&out[2], 3, 3, batch_idx);
        let recon = matmul_f64(&matmul_f64(&u, &diag_f64(&s), 4, 3, 3), &vt, 4, 3, 3);
        let expected = batch_matrix_f64_from_tensor(&input, 4, 3, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-9);
        }
    }
}

#[test]
fn test_batched_qr() {
    let a0 = [1.0, 2.0, 3.0, 4.0, 0.5, -1.0];
    let a1 = [2.0, -1.0, 0.5, 3.0, -0.25, 1.5];
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![3, 2, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| backend.qr(&input)).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[3, 2, 2]);
    assert_eq!(out[1].shape(), &[2, 2, 2]);

    for batch_idx in 0..2 {
        let q = batch_matrix_f64_from_tensor(&out[0], 3, 2, batch_idx);
        let r = batch_matrix_f64_from_tensor(&out[1], 2, 2, batch_idx);
        let recon = matmul_f64(&q, &r, 3, 2, 2);
        let expected = batch_matrix_f64_from_tensor(&input, 3, 2, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-9);
        }
    }
}

#[test]
fn test_batched_solve() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);
    let a = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![3, 3, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
    let b = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![3, 1, 2], vec![1.0, 2.0, 3.0, -1.0, 4.0, 0.5])
            .unwrap(),
    );

    let mut backend = CpuBackend::new();
    let x = with_cpu_linalg(&mut backend, |backend| backend.solve(&a, &b)).unwrap();

    assert_eq!(x.shape(), &[3, 1, 2]);
    for batch_idx in 0..2 {
        let a_batch = batch_matrix_f64_from_tensor(&a, 3, 3, batch_idx);
        let x_batch = batch_matrix_f64_from_tensor(&x, 3, 1, batch_idx);
        let recon = matmul_f64(&a_batch, &x_batch, 3, 3, 1);
        let expected = batch_matrix_f64_from_tensor(&b, 3, 1, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_triangular_solve_lower() {
    let l_data = vec![2.0, 1.0, -0.5, 0.0, 3.0, 1.25, 0.0, 0.0, 1.5];
    let b_data = vec![1.0, -2.0, 0.5];
    let l = Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 3], l_data.clone()).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 1], b_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let x = with_cpu_linalg(&mut backend, |backend| {
        backend.triangular_solve(&l, &b, true, true, false, false)
    })
    .unwrap();

    assert_eq!(x.shape(), &[3, 1]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().unwrap(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&l_data, x_data, 3, 3, 1);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_right_side_unit_transpose() {
    let a_data = vec![1.0, 2.0, 0.0, 1.0];
    let b_data = vec![7.0, 5.0];
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 2], b_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let x = with_cpu_linalg(&mut backend, |backend| {
        backend.triangular_solve(&a, &b, false, true, true, true)
    })
    .unwrap();

    assert_eq!(x.shape(), &[1, 2]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().unwrap().to_vec(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&x_data, &transpose_f64(&a_data, 2, 2), 1, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_covers_all_real_branch_combinations() {
    let expected_x = vec![1.0, -2.0, 0.5, 3.0];

    for &left_side in &[true, false] {
        for &lower in &[true, false] {
            for &transpose_a in &[false, true] {
                for &unit_diagonal in &[false, true] {
                    let diagonal = if unit_diagonal {
                        (1.0, 1.0)
                    } else {
                        (2.0, 1.5)
                    };
                    let a_data = if lower {
                        vec![diagonal.0, -0.75, 0.0, diagonal.1]
                    } else {
                        vec![diagonal.0, 0.0, 0.5, diagonal.1]
                    };
                    let op_a = if transpose_a {
                        transpose_f64(&a_data, 2, 2)
                    } else {
                        a_data.clone()
                    };
                    let b_data = if left_side {
                        matmul_f64(&op_a, &expected_x, 2, 2, 2)
                    } else {
                        matmul_f64(&expected_x, &op_a, 2, 2, 2)
                    };

                    let a =
                        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data).unwrap());
                    let b =
                        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], b_data).unwrap());
                    let mut backend = CpuBackend::new();
                    let x = with_cpu_linalg(&mut backend, |backend| {
                        backend.triangular_solve(
                            &a,
                            &b,
                            left_side,
                            lower,
                            transpose_a,
                            unit_diagonal,
                        )
                    })
                    .unwrap();

                    let x_data = match &x {
                        Tensor::F64(inner) => inner.host_data().unwrap(),
                        _ => panic!("expected f64 tensor"),
                    };
                    for (actual, expected) in x_data.iter().zip(expected_x.iter()) {
                        assert_f64_close_tol(*actual, *expected, 1.0e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn test_batched_complex_solve() {
    let l0 = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let l1 = vec![
        Complex64::new(1.25, 0.0),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let a0 = matmul_c64(&l0, &conjugate_transpose_c64(&l0, 2, 2), 2, 2, 2);
    let a1 = matmul_c64(&l1, &conjugate_transpose_c64(&l1, 2, 2), 2, 2, 2);
    let a = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
    let b = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 1, 2],
            vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(0.5, 2.0),
                Complex64::new(-2.0, 0.25),
                Complex64::new(1.5, -0.75),
            ],
        )
        .unwrap(),
    );

    let mut backend = CpuBackend::new();
    let x = with_cpu_linalg(&mut backend, |backend| backend.solve(&a, &b)).unwrap();

    assert_eq!(x.shape(), &[2, 1, 2]);
    for batch_idx in 0..2 {
        let a_batch = batch_matrix_c64_from_tensor(&a, 2, 2, batch_idx);
        let x_batch = batch_matrix_c64_from_tensor(&x, 2, 1, batch_idx);
        let recon = matmul_c64(&a_batch, &x_batch, 2, 2, 1);
        let expected = batch_matrix_c64_from_tensor(&b, 2, 1, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_real_solve_non_batched() {
    let a_data = vec![3.0, 1.0, 1.0, 2.0];
    let b_data = vec![5.0, 1.0, -2.0, 4.0];
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], b_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let x = with_cpu_linalg(&mut backend, |backend| backend.solve(&a, &b)).unwrap();

    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().unwrap(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&a_data, x_data, 2, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_real_lu_returns_permutation_factors_and_parity() {
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]).unwrap());
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| backend.lu(&input)).unwrap();

    assert_eq!(outputs.len(), 4);
    let p = matrix_f64_from_tensor(&outputs[0], 2, 2);
    let l = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let u = matrix_f64_from_tensor(&outputs[2], 2, 2);
    let parity = get_f64(&outputs[3], &[]);

    let pa = matmul_f64(&p, &matrix_f64_from_tensor(&input, 2, 2), 2, 2, 2);
    let lu = matmul_f64(&l, &u, 2, 2, 2);
    for (actual, expected) in pa.iter().zip(lu.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
    assert_f64_close(parity, -1.0);
}

#[test]
fn test_real_eig_returns_complex_outputs() {
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]).unwrap());
    let mut backend = CpuBackend::new();
    let outputs = with_cpu_linalg(&mut backend, |backend| backend.eig(&input)).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let mut values = vector_c64_from_tensor(&outputs[0], 2);
    values.sort_by(|lhs, rhs| {
        lhs.re
            .partial_cmp(&rhs.re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assert_c64_close(values[0], Complex64::new(1.0, 0.0));
    assert_c64_close(values[1], Complex64::new(3.0, 0.0));
}

#[test]
fn test_batched_complex_eigh() {
    let l0 = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let l1 = vec![
        Complex64::new(1.25, 0.0),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let a0 = matmul_c64(&l0, &conjugate_transpose_c64(&l0, 2, 2), 2, 2, 2);
    let a1 = matmul_c64(&l1, &conjugate_transpose_c64(&l1, 2, 2), 2, 2, 2);
    let input = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );

    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| backend.eigh(&input)).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].dtype(), DType::F64);
    assert_eq!(out[1].dtype(), DType::C64);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_eq!(out[1].shape(), &[2, 2, 2]);

    for batch_idx in 0..2 {
        let values = batch_vector_f64_from_tensor(&out[0], 2, batch_idx);
        let vectors = batch_matrix_c64_from_tensor(&out[1], 2, 2, batch_idx);
        let recon = matmul_c64(
            &matmul_c64(&vectors, &diag_c64_from_real(&values), 2, 2, 2),
            &conjugate_transpose_c64(&vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = batch_matrix_c64_from_tensor(&input, 2, 2, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_real_eigh() {
    let a_data = vec![4.0, 1.0, 1.0, 3.0];
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| backend.eigh(&input)).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let values = match &out[0] {
        Tensor::F64(inner) => inner.host_data().unwrap().to_vec(),
        _ => panic!("expected f64 tensor"),
    };
    let vectors = matrix_f64_from_tensor(&out[1], 2, 2);
    let recon = matmul_f64(
        &matmul_f64(&vectors, &diag_f64(&values), 2, 2, 2),
        &transpose_f64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    for (actual, expected) in recon.iter().zip(a_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_real_cholesky_returns_error_for_non_positive_definite_input() {
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 2.0, 1.0]).unwrap());
    let mut backend = CpuBackend::new();
    let err = with_cpu_linalg(&mut backend, |backend| backend.cholesky(&input)).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::Extension {
            op: "cholesky",
            kind: tenferro_tensor::ErrorKind::NumericalFailure,
            ..
        }
    ));
}

#[test]
fn test_real_solve_returns_error_for_singular_matrix() {
    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 1.0]).unwrap());
    let mut backend = CpuBackend::new();
    let err = with_cpu_linalg(&mut backend, |backend| backend.solve(&a, &b)).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::Extension {
            op: "solve",
            kind: tenferro_tensor::ErrorKind::NumericalFailure,
            ..
        }
    ));
    assert!(std::error::Error::source(&err)
        .and_then(|source| source.downcast_ref::<crate::Error>())
        .is_some_and(|source| matches!(source, crate::Error::Singular { op: "solve" })));
}

#[test]
fn test_complex_cholesky() {
    let l = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let a = matmul_c64(&l, &conjugate_transpose_c64(&l, 2, 2), 2, 2, 2);
    let input = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| backend.cholesky(&input)).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    let l_out = matrix_c64_from_tensor(&out, 2, 2);
    let recon = matmul_c64(&l_out, &conjugate_transpose_c64(&l_out, 2, 2), 2, 2, 2);
    for (actual, expected) in recon.iter().zip(a.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_complex_cholesky_returns_error_for_non_positive_definite_input() {
    let input = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let err = with_cpu_linalg(&mut backend, |backend| backend.cholesky(&input)).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::Extension {
            op: "cholesky",
            kind: tenferro_tensor::ErrorKind::NumericalFailure,
            ..
        }
    ));
}

#[test]
fn test_complex_qr() {
    let input_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -0.5),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(-0.25, 1.5),
        Complex64::new(3.0, 0.75),
    ];
    let input =
        Tensor::C64(TypedTensor::from_vec_col_major(vec![3, 2], input_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| backend.qr(&input)).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let q = matrix_c64_from_tensor(&out[0], 3, 2);
    let r = matrix_c64_from_tensor(&out[1], 2, 2);
    let recon = matmul_c64(&q, &r, 3, 2, 2);
    for (actual, expected) in recon.iter().zip(input_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn test_complex_svd() {
    let input_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -0.5),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(-0.25, 1.5),
        Complex64::new(3.0, 0.75),
    ];
    let input =
        Tensor::C64(TypedTensor::from_vec_col_major(vec![3, 2], input_data.clone()).unwrap());
    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| backend.svd(&input)).unwrap();

    assert_eq!(out.len(), 3);
    assert_eq!(out[0].dtype(), DType::C64);
    assert_eq!(out[1].dtype(), DType::F64);
    assert_eq!(out[2].dtype(), DType::C64);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2]);
    assert_eq!(out[2].shape(), &[2, 2]);

    let u = matrix_c64_from_tensor(&out[0], 3, 2);
    let s = vector_f64_from_tensor(&out[1], 2);
    let vt = matrix_c64_from_tensor(&out[2], 2, 2);
    let recon = matmul_c64(
        &matmul_c64(&u, &diag_c64_from_real(&s), 3, 2, 2),
        &vt,
        3,
        2,
        2,
    );
    for (actual, expected) in recon.iter().zip(input_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn test_complex_triangular_solve_right_side_unit_transpose() {
    let a_data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let b_data = vec![Complex64::new(2.0, 1.0), Complex64::new(-1.0, 0.5)];
    let a = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap());
    let b = Tensor::C64(TypedTensor::from_vec_col_major(vec![1, 2], b_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let x = with_cpu_linalg(&mut backend, |backend| {
        backend.triangular_solve(&a, &b, false, true, true, true)
    })
    .unwrap();

    assert_eq!(x.shape(), &[1, 2]);
    let x_data = match &x {
        Tensor::C64(inner) => inner.host_data().unwrap().to_vec(),
        _ => panic!("expected c64 tensor"),
    };
    let recon = matmul_c64(&x_data, &transpose_c64(&a_data, 2, 2), 1, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_covers_all_complex_branch_combinations() {
    let expected_x = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, -0.5),
        Complex64::new(3.0, -1.0),
    ];

    for &left_side in &[true, false] {
        for &lower in &[true, false] {
            for &transpose_a in &[false, true] {
                for &unit_diagonal in &[false, true] {
                    let diagonal = if unit_diagonal {
                        (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0))
                    } else {
                        (Complex64::new(2.0, 0.0), Complex64::new(1.5, 0.0))
                    };
                    let a_data = if lower {
                        vec![
                            diagonal.0,
                            Complex64::new(-0.75, 0.25),
                            Complex64::new(0.0, 0.0),
                            diagonal.1,
                        ]
                    } else {
                        vec![
                            diagonal.0,
                            Complex64::new(0.0, 0.0),
                            Complex64::new(0.5, -0.25),
                            diagonal.1,
                        ]
                    };
                    let op_a = if transpose_a {
                        transpose_c64(&a_data, 2, 2)
                    } else {
                        a_data.clone()
                    };
                    let b_data = if left_side {
                        matmul_c64(&op_a, &expected_x, 2, 2, 2)
                    } else {
                        matmul_c64(&expected_x, &op_a, 2, 2, 2)
                    };

                    let a =
                        Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a_data).unwrap());
                    let b =
                        Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], b_data).unwrap());
                    let mut backend = CpuBackend::new();
                    let x = with_cpu_linalg(&mut backend, |backend| {
                        backend.triangular_solve(
                            &a,
                            &b,
                            left_side,
                            lower,
                            transpose_a,
                            unit_diagonal,
                        )
                    })
                    .unwrap();

                    let x_data = match &x {
                        Tensor::C64(inner) => inner.host_data().unwrap(),
                        _ => panic!("expected c64 tensor"),
                    };
                    for (actual, expected) in x_data.iter().zip(expected_x.iter()) {
                        assert_c64_close_tol(*actual, *expected, 1.0e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn test_complex_solve_returns_error_for_singular_matrix() {
    let a = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap(),
    );
    let b = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 1],
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let err = with_cpu_linalg(&mut backend, |backend| backend.solve(&a, &b)).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::Extension {
            op: "solve",
            kind: tenferro_tensor::ErrorKind::NumericalFailure,
            ..
        }
    ));
}

#[test]
fn svd_read_faer_strided_view_matches_contiguous() {
    // 2x3 matrix stored col-major, then transposed to give a 3x2 strided view.
    let data = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 4.0]; // 2x3 col-major
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap(); // 3x2 strided
    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| {
        backend.svd_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();
    // For 3x2 input (m=3, n=2), thin SVD gives U:[3,2], S:[2], Vt:[2,2].
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2]);
    assert_eq!(out[2].shape(), &[2, 2]);

    let u = matrix_f64_from_tensor(&out[0], 3, 2);
    let s = (0..2).map(|i| get_f64(&out[1], &[i])).collect::<Vec<_>>();
    let vt = matrix_f64_from_tensor(&out[2], 2, 2);
    let recon = matmul_f64(&matmul_f64(&u, &diag_f64(&s), 3, 2, 2), &vt, 3, 2, 2);
    let expected = transpose_f64(&data, 2, 3); // A^T is 3x2 col-major
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn svd_read_faer_strided_c64_view() {
    let data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, -0.25),
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.0, 0.75),
        Complex64::new(4.0, 1.5),
    ];
    let a = TypedTensor::<Complex64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap(); // 3x2 strided
    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| {
        backend.svd_read(TensorRead::from_view(TensorView::C64(view)))
    })
    .unwrap();
    assert_eq!(out[0].shape(), &[3, 2]); // U (thin, complex)
    assert_eq!(out[1].shape(), &[2]); // S (real singular values)
    assert_eq!(out[2].shape(), &[2, 2]); // Vt (thin, complex)

    // Singular values are returned as a real tensor, mirroring the materialized path.
    let s_vals = match &out[1] {
        Tensor::F64(t) => t.host_data().unwrap().to_vec(),
        Tensor::C64(t) => t.host_data().unwrap().iter().map(|c| c.re).collect(),
        _ => panic!("unexpected type for singular values"),
    };
    assert!(s_vals.iter().all(|&v| v.is_finite() && v >= 0.0));
    assert!(s_vals[0] >= s_vals[1]); // singular values descending
}

#[test]
fn qr_read_faer_strided_view_matches_contiguous() {
    let data = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap(); // 3x2 strided
    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| {
        backend.qr_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let q = matrix_f64_from_tensor(&out[0], 3, 2);
    let r = matrix_f64_from_tensor(&out[1], 2, 2);
    let recon = matmul_f64(&q, &r, 3, 2, 2);
    let expected = transpose_f64(&data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn eigh_read_faer_strided_view_matches_contiguous() {
    // Symmetric 2x2 stored col-major, then transposed to a strided view (still symmetric).
    let data = vec![4.0_f64, 1.0, 1.0, 3.0]; // 2x2 symmetric
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap(); // 2x2 strided (still symmetric)
    let mut backend = CpuBackend::new();
    let out = with_cpu_linalg(&mut backend, |backend| {
        backend.eigh_read(TensorRead::from_view(TensorView::F64(view)))
    })
    .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[2]); // eigenvalues
    assert_eq!(out[1].shape(), &[2, 2]); // eigenvectors

    let eigenvalues = (0..2).map(|i| get_f64(&out[0], &[i])).collect::<Vec<_>>();
    assert!(eigenvalues[0].is_finite());
    assert!(eigenvalues[1].is_finite());

    // Reconstruct A = V diag(lambda) V^T and compare against the (symmetric) input view.
    let vectors = matrix_f64_from_tensor(&out[1], 2, 2);
    let recon = matmul_f64(
        &matmul_f64(&vectors, &diag_f64(&eigenvalues), 2, 2, 2),
        &transpose_f64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_f64(&data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

// ---------------------------------------------------------------------------
// Fallback-path tests: rank-3 views force `faer_strided_ok` to return false
// (rank != 2), so each `*_read` method takes the `to_contiguous` fallback
// branch for every supported dtype (F32, F64, C32, C64).
// ---------------------------------------------------------------------------

/// Shared column-major data for a batch of two 2x2 matrices.
/// Layout: [2, 2, 2] with the batch dimension last, column-major for the
/// matrix core. Each matrix is independently well-conditioned.
///
/// batch 0: [[1, 2], [3, 4]]   (col-major: [1, 3, 2, 4])
/// batch 1: [[2, 0], [0, 3]]   (col-major: [2, 0, 0, 3])
fn rank3_batched_2x2_f32() -> TypedTensor<f32> {
    TypedTensor::<f32>::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0_f32, 3.0, 2.0, 4.0, 2.0, 0.0, 0.0, 3.0],
    )
    .unwrap()
}

fn rank3_batched_2x2_f64() -> TypedTensor<f64> {
    TypedTensor::<f64>::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0_f64, 3.0, 2.0, 4.0, 2.0, 0.0, 0.0, 3.0],
    )
    .unwrap()
}

fn rank3_batched_2x2_c32() -> TypedTensor<Complex32> {
    // batch 0: [[1+0i, 2+0i], [3+0i, 4+0i]]  col-major: [(1,0),(3,0),(2,0),(4,0)]
    // batch 1: [[2+0i, 0+0i], [0+0i, 3+0i]]  col-major: [(2,0),(0,0),(0,0),(3,0)]
    TypedTensor::<Complex32>::from_vec_col_major(
        vec![2, 2, 2],
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.0, 0.0),
        ],
    )
    .unwrap()
}

fn rank3_batched_2x2_c64() -> TypedTensor<Complex64> {
    TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2, 2],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap()
}

/// SPD 2x2 matrices packed into a rank-3 [2, 2, 2] tensor for cholesky tests.
/// batch 0: A = L*L^T where L = [[2, 0], [1, 3]]  => A = [[4, 2], [2, 10]]
/// batch 1: A = L*L^T where L = [[1, 0], [0, 2]]  => A = [[1, 0], [0, 4]]
fn rank3_batched_spd_2x2_f32() -> TypedTensor<f32> {
    // col-major batch 0: [4, 2, 2, 10]
    // col-major batch 1: [1, 0, 0, 4]
    TypedTensor::<f32>::from_vec_col_major(
        vec![2, 2, 2],
        vec![4.0_f32, 2.0, 2.0, 10.0, 1.0, 0.0, 0.0, 4.0],
    )
    .unwrap()
}

fn rank3_batched_spd_2x2_f64() -> TypedTensor<f64> {
    TypedTensor::<f64>::from_vec_col_major(
        vec![2, 2, 2],
        vec![4.0_f64, 2.0, 2.0, 10.0, 1.0, 0.0, 0.0, 4.0],
    )
    .unwrap()
}

fn rank3_batched_spd_2x2_c32() -> TypedTensor<Complex32> {
    TypedTensor::<Complex32>::from_vec_col_major(
        vec![2, 2, 2],
        vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(10.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(4.0, 0.0),
        ],
    )
    .unwrap()
}

fn rank3_batched_spd_2x2_c64() -> TypedTensor<Complex64> {
    TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2, 2],
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(10.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap()
}

/// `svd_read` fallback: rank-3 view forces `to_contiguous` path for all dtypes.
#[test]
fn svd_read_to_contiguous_fallback_rank3_all_dtypes() {
    let f32_t = rank3_batched_2x2_f32();
    let f64_t = rank3_batched_2x2_f64();
    let c32_t = rank3_batched_2x2_c32();
    let c64_t = rank3_batched_2x2_c64();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        // F32: view is rank-3 so faer_strided_ok returns false; fallback is taken.
        let outs = backend
            .svd_read(TensorRead::from_view(TensorView::F32(f32_t.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].dtype(), DType::F32);
        assert_eq!(outs[0].shape(), &[2, 2, 2]);
        assert_eq!(outs[1].shape(), &[2, 2]);
        assert_eq!(outs[2].shape(), &[2, 2, 2]);

        // F64
        let outs = backend
            .svd_read(TensorRead::from_view(TensorView::F64(f64_t.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].dtype(), DType::F64);
        assert_eq!(outs[0].shape(), &[2, 2, 2]);
        assert_eq!(outs[1].shape(), &[2, 2]);
        assert_eq!(outs[2].shape(), &[2, 2, 2]);

        // C32: SVD of complex inputs returns U (C32), S (F32), Vt (C32).
        let outs = backend
            .svd_read(TensorRead::from_view(TensorView::C32(c32_t.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].dtype(), DType::C32);
        assert_eq!(outs[0].shape(), &[2, 2, 2]);
        assert_eq!(outs[1].shape(), &[2, 2]);
        assert_eq!(outs[2].shape(), &[2, 2, 2]);

        // C64: SVD of complex inputs returns U (C64), S (F64), Vt (C64).
        let outs = backend
            .svd_read(TensorRead::from_view(TensorView::C64(c64_t.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].dtype(), DType::C64);
        assert_eq!(outs[0].shape(), &[2, 2, 2]);
        assert_eq!(outs[1].shape(), &[2, 2]);
        assert_eq!(outs[2].shape(), &[2, 2, 2]);
    });
}

/// `qr_read` fallback: rank-3 view forces `to_contiguous` path for all dtypes.
#[test]
fn qr_read_to_contiguous_fallback_rank3_all_dtypes() {
    let f32_t = rank3_batched_2x2_f32();
    let f64_t = rank3_batched_2x2_f64();
    let c32_t = rank3_batched_2x2_c32();
    let c64_t = rank3_batched_2x2_c64();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        for (view, dtype) in [
            (
                TensorRead::from_view(TensorView::F32(f32_t.as_view())),
                DType::F32,
            ),
            (
                TensorRead::from_view(TensorView::F64(f64_t.as_view())),
                DType::F64,
            ),
            (
                TensorRead::from_view(TensorView::C32(c32_t.as_view())),
                DType::C32,
            ),
            (
                TensorRead::from_view(TensorView::C64(c64_t.as_view())),
                DType::C64,
            ),
        ] {
            let outs = backend.qr_read(view).unwrap();
            assert_eq!(
                outs.len(),
                2,
                "qr_read expected 2 outputs for dtype {dtype:?}"
            );
            assert_eq!(outs[0].dtype(), dtype);
            assert_eq!(outs[0].shape(), &[2, 2, 2]);
            assert_eq!(outs[1].shape(), &[2, 2, 2]);
        }
    });
}

/// `eigh_read` fallback: rank-3 view forces `to_contiguous` path for all dtypes.
#[test]
fn eigh_read_to_contiguous_fallback_rank3_all_dtypes() {
    // Use symmetric/Hermitian rank-3 inputs (batch of symmetric 2x2 matrices).
    let sym_f32 = TypedTensor::<f32>::from_vec_col_major(
        vec![2, 2, 2],
        // batch 0: [[4,1],[1,3]] col-major: [4,1,1,3]
        // batch 1: [[2,0],[0,5]] col-major: [2,0,0,5]
        vec![4.0_f32, 1.0, 1.0, 3.0, 2.0, 0.0, 0.0, 5.0],
    )
    .unwrap();
    let sym_f64 = TypedTensor::<f64>::from_vec_col_major(
        vec![2, 2, 2],
        vec![4.0_f64, 1.0, 1.0, 3.0, 2.0, 0.0, 0.0, 5.0],
    )
    .unwrap();
    let sym_c32 = TypedTensor::<Complex32>::from_vec_col_major(
        vec![2, 2, 2],
        vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(5.0, 0.0),
        ],
    )
    .unwrap();
    let sym_c64 = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2, 2],
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(5.0, 0.0),
        ],
    )
    .unwrap();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        // Real dtypes: eigenvalues are same dtype, eigenvectors same dtype.
        let outs = backend
            .eigh_read(TensorRead::from_view(TensorView::F32(sym_f32.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].dtype(), DType::F32);
        assert_eq!(outs[1].dtype(), DType::F32);
        assert_eq!(outs[0].shape(), &[2, 2]);
        assert_eq!(outs[1].shape(), &[2, 2, 2]);

        let outs = backend
            .eigh_read(TensorRead::from_view(TensorView::F64(sym_f64.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].dtype(), DType::F64);
        assert_eq!(outs[1].dtype(), DType::F64);
        assert_eq!(outs[0].shape(), &[2, 2]);
        assert_eq!(outs[1].shape(), &[2, 2, 2]);

        // Complex dtypes: eigenvalues are real, eigenvectors are complex.
        let outs = backend
            .eigh_read(TensorRead::from_view(TensorView::C32(sym_c32.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].dtype(), DType::F32);
        assert_eq!(outs[1].dtype(), DType::C32);
        assert_eq!(outs[0].shape(), &[2, 2]);
        assert_eq!(outs[1].shape(), &[2, 2, 2]);

        let outs = backend
            .eigh_read(TensorRead::from_view(TensorView::C64(sym_c64.as_view())))
            .unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].dtype(), DType::F64);
        assert_eq!(outs[1].dtype(), DType::C64);
        assert_eq!(outs[0].shape(), &[2, 2]);
        assert_eq!(outs[1].shape(), &[2, 2, 2]);
    });
}

/// `cholesky_read` fallback: rank-3 SPD view forces `to_contiguous` path for all dtypes.
#[test]
fn cholesky_read_to_contiguous_fallback_rank3_all_dtypes() {
    let f32_t = rank3_batched_spd_2x2_f32();
    let f64_t = rank3_batched_spd_2x2_f64();
    let c32_t = rank3_batched_spd_2x2_c32();
    let c64_t = rank3_batched_spd_2x2_c64();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        let out = backend
            .cholesky_read(TensorRead::from_view(TensorView::F32(f32_t.as_view())))
            .unwrap();
        assert!(matches!(out, Tensor::F32(_)));
        assert_eq!(out.shape(), &[2, 2, 2]);

        let out = backend
            .cholesky_read(TensorRead::from_view(TensorView::F64(f64_t.as_view())))
            .unwrap();
        assert!(matches!(out, Tensor::F64(_)));
        assert_eq!(out.shape(), &[2, 2, 2]);

        let out = backend
            .cholesky_read(TensorRead::from_view(TensorView::C32(c32_t.as_view())))
            .unwrap();
        assert!(matches!(out, Tensor::C32(_)));
        assert_eq!(out.shape(), &[2, 2, 2]);

        let out = backend
            .cholesky_read(TensorRead::from_view(TensorView::C64(c64_t.as_view())))
            .unwrap();
        assert!(matches!(out, Tensor::C64(_)));
        assert_eq!(out.shape(), &[2, 2, 2]);
    });
}

/// `lu_read` fallback: rank-3 view forces `to_contiguous` path for all dtypes.
#[test]
fn lu_read_to_contiguous_fallback_rank3_all_dtypes() {
    let f32_t = rank3_batched_2x2_f32();
    let f64_t = rank3_batched_2x2_f64();
    let c32_t = rank3_batched_2x2_c32();
    let c64_t = rank3_batched_2x2_c64();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        for (view, dtype) in [
            (
                TensorRead::from_view(TensorView::F32(f32_t.as_view())),
                DType::F32,
            ),
            (
                TensorRead::from_view(TensorView::F64(f64_t.as_view())),
                DType::F64,
            ),
            (
                TensorRead::from_view(TensorView::C32(c32_t.as_view())),
                DType::C32,
            ),
            (
                TensorRead::from_view(TensorView::C64(c64_t.as_view())),
                DType::C64,
            ),
        ] {
            let outs = backend.lu_read(view).unwrap();
            assert_eq!(
                outs.len(),
                4,
                "lu_read expected 4 outputs for dtype {dtype:?}"
            );
            assert_eq!(outs[0].dtype(), dtype);
            // P, L, U have shape [2, 2, 2]; parity is [2] (one scalar per batch element).
            assert_eq!(outs[0].shape(), &[2, 2, 2]);
            assert_eq!(outs[1].shape(), &[2, 2, 2]);
            assert_eq!(outs[2].shape(), &[2, 2, 2]);
            assert_eq!(outs[3].shape(), &[2]);
        }
    });
}

/// `full_piv_lu_read` fallback: rank-3 view forces `to_contiguous` path for all dtypes.
#[test]
fn full_piv_lu_read_to_contiguous_fallback_rank3_all_dtypes() {
    let f32_t = rank3_batched_2x2_f32();
    let f64_t = rank3_batched_2x2_f64();
    let c32_t = rank3_batched_2x2_c32();
    let c64_t = rank3_batched_2x2_c64();

    let mut backend = CpuBackend::new();
    with_cpu_linalg(&mut backend, |backend| {
        for (view, dtype) in [
            (
                TensorRead::from_view(TensorView::F32(f32_t.as_view())),
                DType::F32,
            ),
            (
                TensorRead::from_view(TensorView::F64(f64_t.as_view())),
                DType::F64,
            ),
            (
                TensorRead::from_view(TensorView::C32(c32_t.as_view())),
                DType::C32,
            ),
            (
                TensorRead::from_view(TensorView::C64(c64_t.as_view())),
                DType::C64,
            ),
        ] {
            let outs = backend.full_piv_lu_read(view).unwrap();
            assert_eq!(
                outs.len(),
                5,
                "full_piv_lu_read expected 5 outputs for dtype {dtype:?}"
            );
            // P_row, L, U, P_col each [2, 2, 2]; parity is [2].
            assert_eq!(outs[0].shape(), &[2, 2, 2]);
            assert_eq!(outs[1].shape(), &[2, 2, 2]);
            assert_eq!(outs[2].shape(), &[2, 2, 2]);
            assert_eq!(outs[3].shape(), &[2, 2, 2]);
            assert_eq!(outs[4].shape(), &[2]);
        }
    });
}
