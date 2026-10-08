use super::*;
use tenferro_tensor::BackendSessionHost;
use tenferro_tensor::TensorRead;

#[test]
fn test_dot_general_matmul() {
    let a = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
    );
    let b = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![3, 4],
            vec![
                1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let c = backend
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &DotGeneralConfig {
                    lhs_contracting_dims: [1].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(c.shape(), &[2, 4]);
    assert_eq!(get_f64(&c, &[0, 0]), 38.0);
    assert_eq!(get_f64(&c, &[1, 0]), 83.0);
    assert_eq!(get_f64(&c, &[0, 1]), 44.0);
    assert_eq!(get_f64(&c, &[1, 1]), 98.0);
    assert_eq!(get_f64(&c, &[0, 3]), 56.0);
    assert_eq!(get_f64(&c, &[1, 3]), 128.0);
}

#[test]
fn test_dot_general_with_conj_matches_materialized_complex_matmul() {
    let lhs_data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 0.5),
        Complex64::new(2.0, -1.0),
        Complex64::new(0.25, 4.0),
    ];
    let rhs_data = vec![
        Complex64::new(-2.0, 1.0),
        Complex64::new(1.5, -0.25),
        Complex64::new(0.5, 3.0),
        Complex64::new(-1.0, -2.0),
    ];
    let lhs = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(vec![2, 2], lhs_data.clone()).unwrap(),
    );
    let rhs = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(vec![2, 2], rhs_data.clone()).unwrap(),
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let mut backend = CpuBackend::new();

    let out = backend
        .with_backend_session(|__s| __s.dot_general_with_conj(&lhs, &rhs, &config, true, true))
        .unwrap()
        .unwrap();

    let lhs_conj: Vec<Complex64> = lhs_data.iter().map(|value| value.conj()).collect();
    let rhs_conj: Vec<Complex64> = rhs_data.iter().map(|value| value.conj()).collect();
    let expected = matmul_c64(&lhs_conj, &rhs_conj, 2, 2, 2);
    for col in 0..2 {
        for row in 0..2 {
            assert_c64_close(
                get_c64(&out, &[row, col]),
                expected[col_major_index(2, row, col)],
            );
        }
    }
}

#[test]
fn test_dot_general_read_accepts_tensor_and_view_inputs() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs_shape = [3usize, 2];
    let rhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_view = TensorView::f64(&rhs_shape, &rhs_data).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let mut backend = CpuBackend::new();

    let direct = backend
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_view(rhs_view.clone()),
                &config,
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(direct.shape(), &[2, 2]);
    assert_eq!(direct.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);

    let session = backend
        .with_backend_session(|exec| {
            exec.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_view(rhs_view),
                &config,
            )
        })
        .unwrap();
    let session = session.unwrap();
    assert_eq!(
        session.as_slice::<f64>().unwrap(),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn test_dot_general_read_accepts_transposed_host_view_input() {
    let lhs_source =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let lhs_view = lhs_source.as_view().transpose_view([1, 0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let mut backend = CpuBackend::new();

    let out = backend
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_view(TensorView::F64(lhs_view)),
                TensorRead::from_tensor(&rhs),
                &config,
            )
        })
        .unwrap()
        .unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[50.0, 122.0, 68.0, 167.0]);
}

#[test]
fn test_dot_general_read_into_writes_compact_and_strided_outputs() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs_shape = [3usize, 2];
    let rhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let mut backend = CpuBackend::new();

    let mut compact = Tensor::from_vec_col_major(vec![2, 2], vec![-1.0_f64; 4]).unwrap();
    backend
        .with_backend_session(|__s| {
            __s.dot_general_read_into(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_view(TensorView::f64(&rhs_shape, &rhs_data).unwrap()),
                &config,
                TensorWrite::from_tensor(&mut compact),
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        compact.as_slice::<f64>().unwrap(),
        &[22.0, 28.0, 49.0, 64.0]
    );

    let mut strided_data = [-1.0_f64; 8];
    {
        let out_view = TensorViewMut::F64(
            TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut strided_data).unwrap(),
        );
        backend
            .with_backend_session(|exec| {
                exec.dot_general_read_into(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_view(TensorView::f64(&rhs_shape, &rhs_data).unwrap()),
                    &config,
                    TensorWrite::from_view(out_view),
                )
            })
            .unwrap()
            .unwrap();
    }
    assert_eq!(
        strided_data,
        [-1.0, 22.0, 28.0, -1.0, 49.0, 64.0, -1.0, -1.0]
    );
}

#[test]
fn test_dot_general_read_into_rejects_output_shape_and_dtype_mismatch() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let mut backend = CpuBackend::new();

    let mut wrong_shape = Tensor::from_vec_col_major(vec![4], vec![0.0_f64; 4]).unwrap();
    let shape_err = backend
        .with_backend_session(|__s| {
            __s.dot_general_read_into(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                TensorWrite::from_tensor(&mut wrong_shape),
            )
        })
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        shape_err,
        Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        }
    ));

    let mut wrong_dtype = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f32; 4]).unwrap();
    let dtype_err = backend
        .with_backend_session(|__s| {
            __s.dot_general_read_into(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                TensorWrite::from_tensor(&mut wrong_dtype),
            )
        })
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        dtype_err,
        Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        }
    ));
}

#[test]
fn test_dot_general_read_into_accum_updates_existing_output() {
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let initial = [10.0_f64, 20.0, 30.0, 40.0];
    let mut out = Tensor::from_vec_col_major(vec![2, 2], initial.to_vec()).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let accum = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F64(2.0),
        beta: ContractionScalar::F64(-0.5),
    };
    let mut backend = CpuBackend::new();

    backend
        .with_backend_session(|__s| {
            __s.dot_general_read_into_accum(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                accum,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap()
        .unwrap();

    let expected_dot = [22.0, 28.0, 49.0, 64.0];
    for (actual, expected) in out
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(expected_dot.iter().zip(initial))
    {
        let (dot, previous) = expected;
        assert_f64_close(*actual, 2.0 * *dot - 0.5 * previous);
    }

    let mut strided_data = [-1.0_f64, 10.0, 20.0, -1.0, 30.0, 40.0, -1.0, -1.0];
    {
        let out_view = TensorViewMut::F64(
            TypedTensorViewMut::from_slice([2, 2], [1, 3], 1, &mut strided_data).unwrap(),
        );
        backend
            .with_backend_session(|exec| {
                exec.dot_general_read_into_accum_cached(
                    Some(7),
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &config,
                    accum,
                    TensorWrite::from_view(out_view),
                )
            })
            .unwrap()
            .unwrap();
    }
    assert_eq!(strided_data[0], -1.0);
    assert_eq!(strided_data[3], -1.0);
    assert_eq!(strided_data[6], -1.0);
    assert_eq!(strided_data[7], -1.0);
    for (actual, expected) in [
        strided_data[1],
        strided_data[2],
        strided_data[4],
        strided_data[5],
    ]
    .into_iter()
    .zip(expected_dot.iter().zip(initial))
    {
        let (dot, previous) = expected;
        assert_f64_close(actual, 2.0 * *dot - 0.5 * previous);
    }
}

#[test]
fn test_dot_general_read_into_accum_applies_complex_conj_and_scalars() {
    let lhs_data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 0.5),
        Complex64::new(2.0, -1.0),
        Complex64::new(0.25, 4.0),
    ];
    let rhs_data = vec![
        Complex64::new(-2.0, 1.0),
        Complex64::new(1.5, -0.25),
        Complex64::new(0.5, 3.0),
        Complex64::new(-1.0, -2.0),
    ];
    let initial = vec![
        Complex64::new(1.0, -1.0),
        Complex64::new(2.0, 0.5),
        Complex64::new(-3.0, 2.0),
        Complex64::new(0.25, -4.0),
    ];
    let lhs = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(vec![2, 2], lhs_data.clone()).unwrap(),
    );
    let rhs = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(vec![2, 2], rhs_data.clone()).unwrap(),
    );
    let mut out = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(vec![2, 2], initial.clone()).unwrap(),
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let alpha = Complex64::new(0.5, -1.0);
    let beta = Complex64::new(-0.25, 0.75);
    let accum = DotGeneralAccumulation {
        lhs_conj: true,
        rhs_conj: true,
        alpha: ContractionScalar::C64(alpha),
        beta: ContractionScalar::C64(beta),
    };
    let mut backend = CpuBackend::new();

    backend
        .with_backend_session(|__s| {
            __s.dot_general_read_into_accum(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                accum,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap()
        .unwrap();

    let lhs_conj: Vec<Complex64> = lhs_data.iter().map(|value| value.conj()).collect();
    let rhs_conj: Vec<Complex64> = rhs_data.iter().map(|value| value.conj()).collect();
    let dot = matmul_c64(&lhs_conj, &rhs_conj, 2, 2, 2);
    for index in 0..4 {
        assert_c64_close(
            out.as_slice::<Complex64>().unwrap()[index],
            alpha * dot[index] + beta * initial[index],
        );
    }
}

#[test]
fn test_dot_general_read_into_accum_rejects_scalar_dtype_mismatch() {
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![1, 1], vec![4.0_f64]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let accum = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F32(1.0),
        beta: ContractionScalar::F64(0.0),
    };
    let mut backend = CpuBackend::new();

    let err = backend
        .with_backend_session(|__s| {
            __s.dot_general_read_into_accum(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
                accum,
                TensorWrite::from_tensor(&mut out),
            )
        })
        .unwrap()
        .unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        }
    ));
}

#[test]
fn test_dot_general_read_into_accum_covers_supported_scalar_dtypes() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };

    let lhs_f32 = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let rhs_f32 = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f32, 6.0]).unwrap();
    let mut out_f32 = Tensor::from_vec_col_major(vec![2, 1], vec![7.0_f32, 8.0]).unwrap();
    CpuBackend::new()
        .with_backend_session(|__s| {
            __s.dot_general_read_into_accum(
                TensorRead::from_tensor(&lhs_f32),
                TensorRead::from_tensor(&rhs_f32),
                &config,
                DotGeneralAccumulation {
                    lhs_conj: false,
                    rhs_conj: false,
                    alpha: ContractionScalar::F32(2.0),
                    beta: ContractionScalar::F32(-0.5),
                },
                TensorWrite::from_tensor(&mut out_f32),
            )
        })
        .unwrap()
        .unwrap();
    let out_f32 = out_f32.as_slice::<f32>().unwrap();
    assert!((out_f32[0] - 42.5).abs() < 1.0e-5);
    assert!((out_f32[1] - 64.0).abs() < 1.0e-5);

    let lhs_c32 = Tensor::from_typed::<tenferro_tensor::Complex32>(
        TypedTensor::from_vec_col_major(
            vec![1, 2],
            vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)],
        )
        .unwrap(),
    );
    let rhs_c32 = Tensor::from_typed::<tenferro_tensor::Complex32>(
        TypedTensor::from_vec_col_major(
            vec![2, 1],
            vec![Complex32::new(0.5, -1.0), Complex32::new(3.0, 0.25)],
        )
        .unwrap(),
    );
    let mut out_c32 = Tensor::from_typed::<tenferro_tensor::Complex32>(
        TypedTensor::from_vec_col_major(vec![1, 1], vec![Complex32::new(-2.0, 0.5)]).unwrap(),
    );
    let alpha = Complex32::new(1.5, -0.25);
    let beta = Complex32::new(0.25, 0.5);
    CpuBackend::new()
        .with_backend_session(|__s| {
            __s.dot_general_read_into_accum(
                TensorRead::from_tensor(&lhs_c32),
                TensorRead::from_tensor(&rhs_c32),
                &config,
                DotGeneralAccumulation {
                    lhs_conj: true,
                    rhs_conj: false,
                    alpha: ContractionScalar::C32(alpha),
                    beta: ContractionScalar::C32(beta),
                },
                TensorWrite::from_tensor(&mut out_c32),
            )
        })
        .unwrap()
        .unwrap();
    let dot = lhs_c32.as_slice::<Complex32>().unwrap()[0].conj()
        * rhs_c32.as_slice::<Complex32>().unwrap()[0]
        + lhs_c32.as_slice::<Complex32>().unwrap()[1].conj()
            * rhs_c32.as_slice::<Complex32>().unwrap()[1];
    let expected = alpha * dot + beta * Complex32::new(-2.0, 0.5);
    let actual = out_c32.as_slice::<Complex32>().unwrap()[0];
    assert!((actual.re - expected.re).abs() < 1.0e-5);
    assert!((actual.im - expected.im).abs() < 1.0e-5);
}

#[cfg(feature = "blas")]
#[test]
fn test_dot_general_read_blas_negative_stride_view_falls_back() {
    let lhs_source =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let lhs_view = lhs_source
        .as_view()
        .slice_axis_view(1, StridedSliceSpec::reverse())
        .unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let mut backend = CpuBackend::new();

    let out = tenferro_tensor::BackendSessionHost::with_backend_session(&mut backend, |session| {
        session.dot_general_read(
            TensorRead::from_view(TensorView::F64(lhs_view)),
            TensorRead::from_tensor(&rhs),
            &config,
        )
    })
    .unwrap()
    .unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[68.0, 92.0, 95.0, 128.0]);
}

#[test]
fn test_dot_general_inner_product_returns_rank0_scalar() {
    let a = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
    );
    let b = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![4.0, 5.0, 6.0]).unwrap(),
    );
    let mut backend = CpuBackend::new();
    let c = backend
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &DotGeneralConfig {
                    lhs_contracting_dims: [0].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            )
        })
        .unwrap()
        .unwrap();
    assert!(c.shape().is_empty());
    assert_eq!(get_f64(&c, &[]), 32.0);
}

#[test]
fn test_dot_general_zero_sized_matmul_returns_empty_matrix() {
    let a =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![0, 0], Vec::new()).unwrap());
    let b =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![0, 0], Vec::new()).unwrap());
    let mut backend = CpuBackend::new();
    let c = backend
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &DotGeneralConfig {
                    lhs_contracting_dims: [1].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            )
        })
        .unwrap()
        .unwrap();

    assert_eq!(c.shape(), &[0, 0]);
    assert!(c
        .as_typed::<f64>()
        .expect("expected F64 tensor")
        .host_data()
        .unwrap()
        .is_empty());
}

#[test]
fn test_dot_general_zero_contracting_dim_returns_zero_filled_output() {
    let a =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![2, 0], Vec::new()).unwrap());
    let b =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![0, 3], Vec::new()).unwrap());
    let mut backend = CpuBackend::new();
    let c = backend
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &DotGeneralConfig {
                    lhs_contracting_dims: [1].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            )
        })
        .unwrap()
        .unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(
        c.as_typed::<f64>()
            .expect("expected F64 tensor")
            .host_data()
            .unwrap(),
        &[0.0; 6]
    );
}

#[test]
fn test_dot_general_falls_back_for_unfusable_lhs_batch_layout() {
    let a = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![2, 2, 2, 2],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );
    let b = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![2, 2, 2, 2],
            vec![
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let c = backend
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &DotGeneralConfig {
                    lhs_contracting_dims: [3].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [0, 2].as_slice().into(),
                    rhs_batch_dims: [2, 3].as_slice().into(),
                },
            )
        })
        .unwrap()
        .unwrap();

    assert_eq!(c.shape(), &[2, 2, 2, 2]);
    assert_eq!(get_f64(&c, &[0, 0, 0, 0]), 1.0);
    assert_eq!(get_f64(&c, &[1, 0, 0, 0]), 3.0);
    assert_eq!(get_f64(&c, &[0, 1, 0, 0]), 9.0);
    assert_eq!(get_f64(&c, &[1, 1, 0, 0]), 11.0);
    assert_eq!(get_f64(&c, &[0, 0, 1, 0]), 2.0);
    assert_eq!(get_f64(&c, &[1, 0, 1, 0]), 4.0);
    assert_eq!(get_f64(&c, &[0, 1, 1, 0]), 10.0);
    assert_eq!(get_f64(&c, &[1, 1, 1, 0]), 12.0);
    assert_eq!(get_f64(&c, &[0, 0, 0, 1]), 5.0);
    assert_eq!(get_f64(&c, &[1, 0, 0, 1]), 7.0);
    assert_eq!(get_f64(&c, &[0, 1, 0, 1]), 13.0);
    assert_eq!(get_f64(&c, &[1, 1, 0, 1]), 15.0);
    assert_eq!(get_f64(&c, &[0, 0, 1, 1]), 6.0);
    assert_eq!(get_f64(&c, &[1, 0, 1, 1]), 8.0);
    assert_eq!(get_f64(&c, &[0, 1, 1, 1]), 14.0);
    assert_eq!(get_f64(&c, &[1, 1, 1, 1]), 16.0);
}

#[test]
fn test_dot_general_falls_back_for_mixed_batch_orders() {
    let lhs_storage = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_storage = [10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0];
    let lhs_view =
        TypedTensorView::from_slice([1, 1, 2, 3], [1, 1, 3, 1], 0, &lhs_storage).unwrap();
    let rhs_view =
        TypedTensorView::from_slice([1, 1, 2, 3], [1, 1, 1, 2], 0, &rhs_storage).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2, 3].as_slice().into(),
        rhs_batch_dims: [2, 3].as_slice().into(),
    };
    let mut backend = CpuBackend::new();

    let output = backend
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_view(TensorView::F64(lhs_view)),
                TensorRead::from_view(TensorView::F64(rhs_view)),
                &config,
            )
        })
        .unwrap()
        .unwrap();

    assert_eq!(output.shape(), &[1, 1, 2, 3]);
    for b1 in 0..3 {
        for b0 in 0..2 {
            let lhs = lhs_storage[b0 * 3 + b1];
            let rhs = rhs_storage[b0 + b1 * 2];
            assert_eq!(
                get_f64(&output, &[0, 0, b0, b1]),
                lhs * rhs,
                "mixed batch coordinate ({b0}, {b1})"
            );
        }
    }
}

#[test]
fn test_transpose() {
    let t = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let tr = transpose(&t, &[1, 0]).unwrap();
    assert_eq!(tr.shape(), &[3, 2]);
    assert_eq!(get_f64(&tr, &[0, 0]), 1.0);
    assert_eq!(get_f64(&tr, &[0, 1]), 2.0);
    assert_eq!(get_f64(&tr, &[1, 0]), 3.0);
    assert_eq!(get_f64(&tr, &[1, 1]), 4.0);
    assert_eq!(get_f64(&tr, &[2, 0]), 5.0);
    assert_eq!(get_f64(&tr, &[2, 1]), 6.0);
}

#[test]
fn test_broadcast_in_dim() {
    let scalar =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![], vec![5.0]).unwrap());
    let broadcast = broadcast_in_dim(&scalar, &[3], &[]).unwrap();
    assert_eq!(broadcast.shape(), &[3]);
    assert_eq!(get_f64(&broadcast, &[0]), 5.0);
    assert_eq!(get_f64(&broadcast, &[1]), 5.0);
    assert_eq!(get_f64(&broadcast, &[2]), 5.0);

    let v = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
    );
    let m = broadcast_in_dim(&v, &[3, 2], &[0]).unwrap();
    assert_eq!(m.shape(), &[3, 2]);
    for j in 0..2 {
        assert_eq!(get_f64(&m, &[0, j]), 1.0);
        assert_eq!(get_f64(&m, &[1, j]), 2.0);
        assert_eq!(get_f64(&m, &[2, j]), 3.0);
    }
}

#[test]
fn test_tril_3x3() {
    let t = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
    );
    let lower = tril(&t, 0).unwrap();
    assert_eq!(lower.shape(), &[3, 3]);
    assert_eq!(
        lower
            .as_typed::<f64>()
            .expect("expected f64 tensor")
            .host_data()
            .unwrap(),
        &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
    );
}

#[test]
fn test_triu_3x3() {
    let t = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
    );
    let upper = triu(&t, 0).unwrap();
    assert_eq!(upper.shape(), &[3, 3]);
    assert_eq!(
        upper
            .as_typed::<f64>()
            .expect("expected f64 tensor")
            .host_data()
            .unwrap(),
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_triangular_masks_rectangular_batched_nonzero_diagonal() {
    let t = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 3, 2], (1..=12).map(f64::from).collect()).unwrap(),
    );

    let lower = tril(&t, 1).unwrap();
    assert_eq!(
        lower.as_slice::<f64>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 12.0]
    );

    let upper = triu(&t, 1).unwrap();
    assert_eq!(
        upper.as_slice::<f64>().unwrap(),
        &[0.0, 0.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0, 0.0, 11.0, 12.0]
    );
}

#[test]
fn test_tril_triu_zero_sized_batch_return_empty_tensor() {
    let t = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 2, 0], Vec::new()).unwrap(),
    );

    let lower = tril(&t, 0).unwrap();
    assert_eq!(lower.shape(), &[2, 2, 0]);
    assert!(lower
        .as_typed::<f64>()
        .expect("expected f64 tensor")
        .host_data()
        .unwrap()
        .is_empty());

    let upper = triu(&t, 0).unwrap();
    assert_eq!(upper.shape(), &[2, 2, 0]);
    assert!(upper
        .as_typed::<f64>()
        .expect("expected f64 tensor")
        .host_data()
        .unwrap()
        .is_empty());
}

#[test]
fn test_tril_triu_extreme_offsets_do_not_overflow() {
    let t = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );

    let lower_min = tril(&t, i64::MIN).unwrap();
    assert_eq!(
        lower_min
            .as_typed::<f64>()
            .expect("expected f64 tensor")
            .host_data()
            .unwrap(),
        &[0.0, 0.0, 0.0, 0.0]
    );

    let upper_min = triu(&t, i64::MIN).unwrap();
    assert_eq!(
        upper_min
            .as_typed::<f64>()
            .expect("expected f64 tensor")
            .host_data()
            .unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );

    let lower_max = tril(&t, i64::MAX).unwrap();
    assert_eq!(
        lower_max
            .as_typed::<f64>()
            .expect("expected f64 tensor")
            .host_data()
            .unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );

    let upper_max = triu(&t, i64::MAX).unwrap();
    assert_eq!(
        upper_max
            .as_typed::<f64>()
            .expect("expected f64 tensor")
            .host_data()
            .unwrap(),
        &[0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn test_triangular_masks_use_checked_index_arithmetic_contract() {
    let source = include_str!("../../structural.rs");
    let section_start = source
        .find("fn typed_triangular_mask")
        .expect("typed_triangular_mask should exist");
    let section = &source[section_start..];

    for needle in [
        "checked_triangular_extent(op, tensor.shape(), rows, cols)?",
        "checked_triangular_offset(op, batch_idx, block_size, col, rows, row_idx)?",
    ] {
        assert!(
            section.contains(needle),
            "triangular masks should use checked index arithmetic: missing {needle}"
        );
    }
}

#[test]
fn test_neg_and_conj() {
    let t = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![3.0, -7.0]).unwrap(),
    );
    let n = neg(&t).unwrap();
    assert_eq!(get_f64(&n, &[0]), -3.0);
    assert_eq!(get_f64(&n, &[1]), 7.0);

    let c = conj(&t).unwrap();
    assert_eq!(get_f64(&c, &[0]), 3.0);
    assert_eq!(get_f64(&c, &[1]), -7.0);
}

#[test]
fn test_cpu_backend_analytic_ops_real() {
    let mut backend = CpuBackend::new();

    let exp_input = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap(),
    );
    let exp_out = backend
        .with_backend_session(|__s| __s.exp_read(TensorRead::from_tensor(&exp_input)))
        .unwrap()
        .unwrap();
    assert_f64_close(get_f64(&exp_out, &[0]), 1.0);
    assert_f64_close(get_f64(&exp_out, &[1]), std::f64::consts::E);

    let log_input = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0, 4.0]).unwrap(),
    );
    let log_out = backend
        .with_backend_session(|__s| __s.log_read(TensorRead::from_tensor(&log_input)))
        .unwrap()
        .unwrap();
    assert_f64_close(get_f64(&log_out, &[0]), 0.0);
    assert_f64_close(get_f64(&log_out, &[1]), 4.0_f64.ln());

    let trig_input = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::FRAC_PI_2]).unwrap(),
    );
    let sin_out = backend
        .with_backend_session(|__s| __s.sin_read(TensorRead::from_tensor(&trig_input)))
        .unwrap()
        .unwrap();
    let cos_out = backend
        .with_backend_session(|__s| __s.cos_read(TensorRead::from_tensor(&trig_input)))
        .unwrap()
        .unwrap();
    assert_f64_close(get_f64(&sin_out, &[0]), 0.0);
    assert_f64_close(get_f64(&sin_out, &[1]), 1.0);
    assert_f64_close(get_f64(&cos_out, &[0]), 1.0);
    assert_f64_close(get_f64(&cos_out, &[1]), 0.0);

    let tanh_input = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap(),
    );
    let tanh_out = backend
        .with_backend_session(|__s| __s.tanh_read(TensorRead::from_tensor(&tanh_input)))
        .unwrap()
        .unwrap();
    assert_f64_close(get_f64(&tanh_out, &[0]), 0.0);
    assert_f64_close(get_f64(&tanh_out, &[1]), 1.0_f64.tanh());

    let sqrt_input = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0, 4.0]).unwrap(),
    );
    let sqrt_out = backend
        .with_backend_session(|__s| __s.sqrt_read(TensorRead::from_tensor(&sqrt_input)))
        .unwrap()
        .unwrap();
    let rsqrt_out = backend
        .with_backend_session(|__s| __s.rsqrt_read(TensorRead::from_tensor(&sqrt_input)))
        .unwrap()
        .unwrap();
    assert_f64_close(get_f64(&sqrt_out, &[0]), 1.0);
    assert_f64_close(get_f64(&sqrt_out, &[1]), 2.0);
    assert_f64_close(get_f64(&rsqrt_out, &[0]), 1.0);
    assert_f64_close(get_f64(&rsqrt_out, &[1]), 0.5);

    let expm1_out = backend
        .with_backend_session(|__s| {
            __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(&exp_input))
        })
        .unwrap()
        .unwrap();
    let log1p_out = backend
        .with_backend_session(|__s| __s.log1p_read(TensorRead::from_tensor(&log_input)))
        .unwrap()
        .unwrap();
    assert_f64_close(get_f64(&expm1_out, &[0]), 0.0);
    assert_f64_close(get_f64(&expm1_out, &[1]), 1.0_f64.exp_m1());
    assert_f64_close(get_f64(&log1p_out, &[0]), 2.0_f64.ln());
    assert_f64_close(get_f64(&log1p_out, &[1]), 5.0_f64.ln());

    let pow_base = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![2.0, 9.0]).unwrap(),
    );
    let pow_exp = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![3.0, 0.5]).unwrap(),
    );
    let pow_out = backend
        .with_backend_session(|__s| {
            __s.pow_read(
                TensorRead::from_tensor(&pow_base),
                TensorRead::from_tensor(&pow_exp),
            )
        })
        .unwrap()
        .unwrap();
    assert_f64_close(get_f64(&pow_out, &[0]), 8.0);
    assert_f64_close(get_f64(&pow_out, &[1]), 3.0);
}

#[test]
fn test_cpu_backend_analytic_ops_complex() {
    let mut backend = CpuBackend::new();

    let exp_input = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 1.0)],
        )
        .unwrap(),
    );
    let exp_out = backend
        .with_backend_session(|__s| __s.exp_read(TensorRead::from_tensor(&exp_input)))
        .unwrap()
        .unwrap();
    assert_c64_close(get_c64(&exp_out, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&exp_out, &[1]), Complex64::new(1.0, 1.0).exp());

    let log_input = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, -0.5)],
        )
        .unwrap(),
    );
    let log_out = backend
        .with_backend_session(|__s| __s.log_read(TensorRead::from_tensor(&log_input)))
        .unwrap()
        .unwrap();
    assert_c64_close(get_c64(&log_out, &[0]), Complex64::new(1.0, 0.0).ln());
    assert_c64_close(get_c64(&log_out, &[1]), Complex64::new(2.0, -0.5).ln());

    let trig_input = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, -0.25)],
        )
        .unwrap(),
    );
    let sin_out = backend
        .with_backend_session(|__s| __s.sin_read(TensorRead::from_tensor(&trig_input)))
        .unwrap()
        .unwrap();
    let cos_out = backend
        .with_backend_session(|__s| __s.cos_read(TensorRead::from_tensor(&trig_input)))
        .unwrap()
        .unwrap();
    let tanh_out = backend
        .with_backend_session(|__s| __s.tanh_read(TensorRead::from_tensor(&trig_input)))
        .unwrap()
        .unwrap();
    assert_c64_close(get_c64(&sin_out, &[0]), Complex64::new(0.0, 0.0).sin());
    assert_c64_close(get_c64(&sin_out, &[1]), Complex64::new(0.5, -0.25).sin());
    assert_c64_close(get_c64(&cos_out, &[0]), Complex64::new(0.0, 0.0).cos());
    assert_c64_close(get_c64(&cos_out, &[1]), Complex64::new(0.5, -0.25).cos());
    assert_c64_close(get_c64(&tanh_out, &[0]), Complex64::new(0.0, 0.0).tanh());
    assert_c64_close(get_c64(&tanh_out, &[1]), Complex64::new(0.5, -0.25).tanh());

    let sqrt_input = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(4.0, 3.0)],
        )
        .unwrap(),
    );
    let sqrt_out = backend
        .with_backend_session(|__s| __s.sqrt_read(TensorRead::from_tensor(&sqrt_input)))
        .unwrap()
        .unwrap();
    let rsqrt_out = backend
        .with_backend_session(|__s| __s.rsqrt_read(TensorRead::from_tensor(&sqrt_input)))
        .unwrap()
        .unwrap();
    assert_c64_close(get_c64(&sqrt_out, &[0]), Complex64::new(1.0, 0.0).sqrt());
    assert_c64_close(get_c64(&sqrt_out, &[1]), Complex64::new(4.0, 3.0).sqrt());
    assert_c64_close_tol(
        get_c64(&rsqrt_out, &[0]),
        Complex64::new(1.0, 0.0) / Complex64::new(1.0, 0.0).sqrt(),
        1.0e-12,
    );
    assert_c64_close_tol(
        get_c64(&rsqrt_out, &[1]),
        Complex64::new(1.0, 0.0) / Complex64::new(4.0, 3.0).sqrt(),
        1.0e-12,
    );

    let expm1_out = backend
        .with_backend_session(|__s| {
            __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(&exp_input))
        })
        .unwrap()
        .unwrap();
    let log1p_out = backend
        .with_backend_session(|__s| __s.log1p_read(TensorRead::from_tensor(&log_input)))
        .unwrap()
        .unwrap();
    assert_c64_close(
        get_c64(&expm1_out, &[0]),
        Complex64::new(0.0, 0.0).exp() - Complex64::new(1.0, 0.0),
    );
    assert_c64_close(
        get_c64(&expm1_out, &[1]),
        Complex64::new(1.0, 1.0).exp() - Complex64::new(1.0, 0.0),
    );
    assert_c64_close(
        get_c64(&log1p_out, &[0]),
        (Complex64::new(1.0, 0.0) + Complex64::new(1.0, 0.0)).ln(),
    );
    assert_c64_close(
        get_c64(&log1p_out, &[1]),
        (Complex64::new(2.0, -0.5) + Complex64::new(1.0, 0.0)).ln(),
    );

    let pow_base = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
        )
        .unwrap(),
    );
    let pow_exp = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(2.0, 0.0), Complex64::new(0.5, 0.25)],
        )
        .unwrap(),
    );
    let pow_out = backend
        .with_backend_session(|__s| {
            __s.pow_read(
                TensorRead::from_tensor(&pow_base),
                TensorRead::from_tensor(&pow_exp),
            )
        })
        .unwrap()
        .unwrap();
    assert_c64_close(
        get_c64(&pow_out, &[0]),
        Complex64::new(1.0, 1.0).powc(Complex64::new(2.0, 0.0)),
    );
    assert_c64_close(
        get_c64(&pow_out, &[1]),
        Complex64::new(2.0, -1.0).powc(Complex64::new(0.5, 0.25)),
    );
}

#[test]
fn test_extract_diagonal() {
    let square = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
    );
    let d = extract_diagonal(&square, 0, 1).unwrap();
    assert_eq!(d.shape(), &[3]);
    assert_eq!(get_f64(&d, &[0]), 1.0);
    assert_eq!(get_f64(&d, &[1]), 5.0);
    assert_eq!(get_f64(&d, &[2]), 9.0);

    let cube = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 3, 3], (1..=18).map(|x| x as f64).collect())
            .unwrap(),
    );
    let diag = extract_diagonal(&cube, 1, 2).unwrap();
    assert_eq!(diag.shape(), &[2, 3]);
    assert_eq!(get_f64(&diag, &[0, 0]), 1.0);
    assert_eq!(get_f64(&diag, &[1, 1]), 10.0);
    assert_eq!(get_f64(&diag, &[1, 2]), 18.0);
}

#[test]
fn test_embed_diagonal() {
    let v = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
    );
    let m = embed_diagonal(&v, 0, 1).unwrap();
    assert_eq!(m.shape(), &[3, 3]);
    assert_eq!(get_f64(&m, &[0, 0]), 1.0);
    assert_eq!(get_f64(&m, &[1, 1]), 2.0);
    assert_eq!(get_f64(&m, &[2, 2]), 3.0);
    assert_eq!(get_f64(&m, &[0, 1]), 0.0);
    assert_eq!(get_f64(&m, &[2, 0]), 0.0);
}

#[test]
fn test_cpu_backend_dispatches_tensor_backend_ops() {
    let a = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap(),
    );
    let b = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap(),
    );
    let mut backend = CpuBackend::new();
    let out = backend
        .with_backend_session(|__s| {
            __s.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        })
        .unwrap()
        .unwrap();
    assert_eq!(get_f64(&out, &[0]), 4.0);
    assert_eq!(get_f64(&out, &[1]), 6.0);
}

#[test]
fn test_tier2_elementwise_ops_real() {
    let lhs = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![8.0, -2.0, 9.0]).unwrap(),
    );
    let rhs = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![2.0, 5.0, 3.0]).unwrap(),
    );
    let pred = Tensor::from_typed::<bool>(
        TypedTensor::from_vec_col_major(vec![3], vec![false, true, true]).unwrap(),
    );
    let on_true = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![10.0, 20.0, 30.0]).unwrap(),
    );
    let on_false = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
    );
    let lower = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![-1.0, -1.0, 0.0]).unwrap(),
    );
    let upper = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3], vec![1.0, 0.25, 4.0]).unwrap(),
    );
    let mut backend = CpuBackend::new();

    let div = backend
        .with_backend_session(|__s| {
            __s.div_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        })
        .unwrap()
        .unwrap();
    assert_eq!(get_f64(&div, &[0]), 4.0);
    assert_eq!(get_f64(&div, &[1]), -0.4);
    assert_eq!(get_f64(&div, &[2]), 3.0);

    let abs = backend
        .with_backend_session(|__s| __s.abs_read(TensorRead::from_tensor(&lhs)))
        .unwrap()
        .unwrap();
    assert_eq!(get_f64(&abs, &[0]), 8.0);
    assert_eq!(get_f64(&abs, &[1]), 2.0);
    assert_eq!(get_f64(&abs, &[2]), 9.0);

    let sign = backend
        .with_backend_session(|__s| __s.sign_read(TensorRead::from_tensor(&lhs)))
        .unwrap()
        .unwrap();
    assert_eq!(get_f64(&sign, &[0]), 1.0);
    assert_eq!(get_f64(&sign, &[1]), -1.0);
    assert_eq!(get_f64(&sign, &[2]), 1.0);

    let maximum = backend
        .with_backend_session(|__s| {
            __s.maximum_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        })
        .unwrap()
        .unwrap();
    assert_eq!(get_f64(&maximum, &[0]), 8.0);
    assert_eq!(get_f64(&maximum, &[1]), 5.0);
    assert_eq!(get_f64(&maximum, &[2]), 9.0);

    let minimum = backend
        .with_backend_session(|__s| {
            __s.minimum_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        })
        .unwrap()
        .unwrap();
    assert_eq!(get_f64(&minimum, &[0]), 2.0);
    assert_eq!(get_f64(&minimum, &[1]), -2.0);
    assert_eq!(get_f64(&minimum, &[2]), 3.0);

    let eq = backend
        .with_backend_session(|__s| {
            __s.compare_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &CompareDir::Eq,
            )
        })
        .unwrap()
        .unwrap();
    assert!(!get_bool(&eq, &[0]));
    assert!(!get_bool(&eq, &[1]));
    assert!(!get_bool(&eq, &[2]));

    let lt = backend
        .with_backend_session(|__s| {
            __s.compare_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &CompareDir::Lt,
            )
        })
        .unwrap()
        .unwrap();
    assert!(!get_bool(&lt, &[0]));
    assert!(get_bool(&lt, &[1]));
    assert!(!get_bool(&lt, &[2]));

    let le = backend
        .with_backend_session(|__s| {
            __s.compare_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &CompareDir::Le,
            )
        })
        .unwrap()
        .unwrap();
    assert!(!get_bool(&le, &[0]));
    assert!(get_bool(&le, &[1]));
    assert!(!get_bool(&le, &[2]));

    let gt = backend
        .with_backend_session(|__s| {
            __s.compare_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &CompareDir::Gt,
            )
        })
        .unwrap()
        .unwrap();
    assert!(get_bool(&gt, &[0]));
    assert!(!get_bool(&gt, &[1]));
    assert!(get_bool(&gt, &[2]));

    let ge = backend
        .with_backend_session(|__s| {
            __s.compare_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &CompareDir::Ge,
            )
        })
        .unwrap()
        .unwrap();
    assert!(get_bool(&ge, &[0]));
    assert!(!get_bool(&ge, &[1]));
    assert!(get_bool(&ge, &[2]));

    let select = backend
        .with_backend_session(|__s| {
            __s.select_read(
                TensorRead::from_tensor(&pred),
                TensorRead::from_tensor(&on_true),
                TensorRead::from_tensor(&on_false),
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(get_f64(&select, &[0]), 1.0);
    assert_eq!(get_f64(&select, &[1]), 20.0);
    assert_eq!(get_f64(&select, &[2]), 30.0);

    let clamp = backend
        .with_backend_session(|__s| {
            __s.clamp_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&lower),
                TensorRead::from_tensor(&upper),
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(get_f64(&clamp, &[0]), 1.0);
    assert_eq!(get_f64(&clamp, &[1]), -1.0);
    assert_eq!(get_f64(&clamp, &[2]), 4.0);
}

#[test]
fn test_tier2_elementwise_ops_complex() {
    let input = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
        )
        .unwrap(),
    );
    let lhs = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap(),
    );
    let rhs = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 2.0)],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();

    let abs = backend
        .with_backend_session(|__s| __s.abs_read(TensorRead::from_tensor(&input)))
        .unwrap()
        .unwrap();
    assert_eq!(abs.dtype(), DType::F64);
    assert_eq!(get_f64(&abs, &[0]), 5.0);
    assert_eq!(get_f64(&abs, &[1]), 0.0);

    let sign = backend
        .with_backend_session(|__s| __s.sign_read(TensorRead::from_tensor(&input)))
        .unwrap()
        .unwrap();
    assert_c64_close(get_c64(&sign, &[0]), Complex64::new(0.6, 0.8));
    assert_c64_close(get_c64(&sign, &[1]), Complex64::new(0.0, 0.0));

    assert!(matches!(
        backend.with_backend_session(|__s| __s.maximum_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))).unwrap(),
        Err(crate::Error::Unsupported {
            op: "maximum",
            message,
        }) if message.contains("total order")
    ));
    assert!(matches!(
        backend.with_backend_session(|__s| __s.minimum_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))).unwrap(),
        Err(crate::Error::Unsupported {
            op: "minimum",
            message,
        }) if message.contains("total order")
    ));
}

/// The canonical fallback plans packed and borrowed canonical operands
/// directly: a borrowed operand at a nonzero offset and a strided destination
/// view must still produce the reference contraction (#1897).
#[test]
fn canonical_fallback_honors_operand_offsets_and_output_strides() {
    use tenferro_tensor::{
        DotGeneralConfig, TensorView, TensorViewMut, TensorWrite, TypedTensorView,
        TypedTensorViewMut,
    };

    // lhs [2, 3, 2] contracts its middle axis, so its free axes (0, 2) need
    // packing; rhs is a compact [3, 2] block at element offset 5 of a larger
    // buffer, already canonical and therefore borrowed.
    let lhs_data: Vec<f64> = (1..=12).map(f64::from).collect();
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 2], lhs_data.clone()).unwrap();
    let rhs_storage: Vec<f64> = (0..11).map(|i| f64::from(i) * 0.5 - 1.0).collect();
    let rhs = TypedTensorView::from_slice([3, 2], [1, 3], 5, &rhs_storage).unwrap();
    let rhs_data = &rhs_storage[5..11];
    // The [2, 2, 2] output keeps its fused row group (i, l) compact but pads
    // each GEMM column to a leading dimension of 6 in a 12-element buffer; BLAS
    // also requires a unit row stride.
    let mut out_storage = vec![-7.0_f64; 12];
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };

    let mut backend = CpuBackend::with_threads(1).unwrap();
    {
        let out =
            TypedTensorViewMut::from_slice([2, 2, 2], [1, 2, 6], 0, &mut out_storage).unwrap();
        backend
            .with_backend_session(|session| {
                session.dot_general_read_into(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_view(TensorView::F64(rhs)),
                    &config,
                    TensorWrite::from_view(TensorViewMut::F64(out)),
                )
            })
            .unwrap()
            .unwrap();
    }

    // out[i, l, n] = sum_j lhs[i, j, l] * rhs[j, n] at storage i + 2l + 6n.
    for n in 0..2 {
        for l in 0..2 {
            for i in 0..2 {
                let expected: f64 = (0..3)
                    .map(|j| lhs_data[i + 2 * j + 6 * l] * rhs_data[j + 3 * n])
                    .sum();
                assert_eq!(out_storage[i + 2 * l + 6 * n], expected, "({i}, {l}, {n})");
            }
        }
    }
    // The padding slots of each column are untouched.
    assert!(out_storage
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 6 >= 4)
        .all(|(_, &value)| value == -7.0));
}

#[test]
fn auto_batched_gemm_on_lanes_matches_reference_with_padded_batches() {
    use tenferro_tensor::{DotGeneralConfig, TensorViewMut, TensorWrite, TypedTensorViewMut};

    // 1024 tiny 4x4x4 products clear the Auto lane gate at 4 threads, so the
    // batch runs as one strided GEMM per lane. Each output item is padded to a
    // batch stride of 20, and the padding must survive the lane split.
    let (m, batch, pitch) = (4_usize, 1024_usize, 20_usize);
    let lhs_data: Vec<f64> = (0..m * m * batch).map(|i| (i % 13) as f64 - 6.0).collect();
    let rhs_data: Vec<f64> = (0..m * m * batch).map(|i| (i % 7) as f64 * 0.5).collect();
    let lhs = Tensor::from_vec_col_major(vec![m, m, batch], lhs_data.clone()).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![m, m, batch], rhs_data.clone()).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    };

    // A nonzero view offset shifts every lane's chunk split.
    for offset in [0_usize, 3] {
        let mut out_storage = vec![-7.0_f64; offset + pitch * batch];
        let mut backend = CpuBackend::with_threads(4).unwrap();
        {
            let out = TypedTensorViewMut::from_slice(
                [m, m, batch],
                [1, m as isize, pitch as isize],
                offset as isize,
                &mut out_storage,
            )
            .unwrap();
            backend
                .with_backend_session(|session| {
                    session.dot_general_read_into(
                        TensorRead::from_tensor(&lhs),
                        TensorRead::from_tensor(&rhs),
                        &config,
                        TensorWrite::from_view(TensorViewMut::F64(out)),
                    )
                })
                .unwrap()
                .unwrap();
        }

        for b in 0..batch {
            for j in 0..m {
                for i in 0..m {
                    let expected: f64 = (0..m)
                        .map(|k| lhs_data[i + m * k + m * m * b] * rhs_data[k + m * j + m * m * b])
                        .sum();
                    assert_eq!(
                        out_storage[offset + i + m * j + pitch * b],
                        expected,
                        "offset {offset}: ({i}, {j}, {b})"
                    );
                }
            }
        }
        assert!(out_storage
            .iter()
            .enumerate()
            .filter(|(index, _)| *index < offset || (index - offset) % pitch >= m * m)
            .all(|(_, &value)| value == -7.0));
    }
}

#[test]
fn auto_grouped_gemm_on_lanes_matches_reference_for_ordered_and_reversed_jobs() {
    use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
    use tenferro_tensor::{DType, DotGeneralAccumulation, TensorWrite};

    // 1024 4x4x4 jobs clear the Auto lane gate at 4 threads. Ordered outputs
    // (padded to a pitch of 20) run as one chunk per lane; reversed outputs
    // fall back to one call per job. Both must match the reference and leave
    // the padding untouched.
    let (m, jobs_len, pitch) = (4_usize, 1024_usize, 20_usize);
    let lhs_data: Vec<f64> = (0..m * m * jobs_len)
        .map(|i| (i % 13) as f64 - 6.0)
        .collect();
    let rhs_data: Vec<f64> = (0..m * m * jobs_len)
        .map(|i| (i % 7) as f64 * 0.5)
        .collect();
    let lhs = Tensor::from_vec_col_major(vec![m * m * jobs_len], lhs_data.clone()).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![m * m * jobs_len], rhs_data.clone()).unwrap();
    for reversed in [false, true] {
        let slot = |job: usize| if reversed { jobs_len - 1 - job } else { job };
        let jobs = (0..jobs_len)
            .map(|job| GroupedGemmJob::new(pitch * slot(job), m * m * job, m * m * job, m, m, m))
            .collect::<Vec<_>>();
        let mut output =
            Tensor::from_vec_col_major(vec![pitch * jobs_len], vec![-7.0_f64; pitch * jobs_len])
                .unwrap();
        let mut backend = CpuBackend::with_threads(4).unwrap();
        backend
            .with_backend_session(|session| {
                session.grouped_gemm_cached(
                    None,
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &GroupedGemmConfig::new(
                        &jobs,
                        DotGeneralAccumulation::overwrite(DType::F64).unwrap(),
                    ),
                    TensorWrite::from_tensor(&mut output),
                )
            })
            .unwrap()
            .unwrap();

        let out = output.as_slice::<f64>().unwrap();
        for job in 0..jobs_len {
            for j in 0..m {
                for i in 0..m {
                    let expected: f64 = (0..m)
                        .map(|k| {
                            lhs_data[m * m * job + i + m * k] * rhs_data[m * m * job + k + m * j]
                        })
                        .sum();
                    assert_eq!(
                        out[pitch * slot(job) + i + m * j],
                        expected,
                        "reversed={reversed} ({i}, {j}, {job})"
                    );
                }
            }
        }
        assert!(out
            .iter()
            .enumerate()
            .filter(|(index, _)| index % pitch >= m * m)
            .all(|(_, &value)| value == -7.0));
    }
}

#[test]
fn auto_allocated_batched_gemm_on_lanes_matches_reference() {
    use tenferro_tensor::DotGeneralConfig;

    // The allocating entry point writes an uninitialized pooled destination;
    // 1024 4x4x4 products clear the Auto lane gate at 4 threads, so each lane
    // fully overwrites its own chunk of that destination (#1898).
    let (m, batch) = (4_usize, 1024_usize);
    let lhs_data: Vec<f64> = (0..m * m * batch).map(|i| (i % 13) as f64 - 6.0).collect();
    let rhs_data: Vec<f64> = (0..m * m * batch).map(|i| (i % 7) as f64 * 0.5).collect();
    let lhs = Tensor::from_vec_col_major(vec![m, m, batch], lhs_data.clone()).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![m, m, batch], rhs_data.clone()).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    };

    let mut backend = CpuBackend::with_threads(4).unwrap();
    let output = backend
        .with_backend_session(|session| {
            session.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
        })
        .unwrap()
        .unwrap();

    assert_eq!(output.shape(), &[m, m, batch]);
    let out = output.as_slice::<f64>().unwrap();
    for b in 0..batch {
        for j in 0..m {
            for i in 0..m {
                let expected: f64 = (0..m)
                    .map(|k| lhs_data[i + m * k + m * m * b] * rhs_data[k + m * j + m * m * b])
                    .sum();
                assert_eq!(out[i + m * j + m * m * b], expected, "({i}, {j}, {b})");
            }
        }
    }
}
