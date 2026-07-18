use super::*;

#[test]
fn test_gather_1d_indices() {
    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
    );
    let start_indices = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 2, 4]).unwrap();

    let out = gather(&operand, &start_indices, &simple_gather_config()).unwrap();

    assert_eq!(out.shape(), &[3]);
    assert_eq!(get_f64(&out, &[0]), 10.0);
    assert_eq!(get_f64(&out, &[1]), 30.0);
    assert_eq!(get_f64(&out, &[2]), 50.0);
}

#[test]
fn test_gather_accepts_i64_indices() {
    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
    );
    let start_indices = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 2, 4]).unwrap();

    let out = gather(&operand, &start_indices, &simple_gather_config()).unwrap();

    assert_eq!(start_indices.dtype(), DType::I64);
    assert_eq!(
        start_indices.as_slice::<i64>().unwrap(),
        [0, 2, 4].as_slice()
    );
    assert_eq!(out.shape(), &[3]);
    assert_eq!(get_f64(&out, &[0]), 10.0);
    assert_eq!(get_f64(&out, &[1]), 30.0);
    assert_eq!(get_f64(&out, &[2]), 50.0);
}

#[test]
fn test_gather_with_implicit_index_vector_dim() {
    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
    );
    let start_indices = Tensor::from_vec_col_major(vec![3], vec![4_i64, 1, 0]).unwrap();
    let config = GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };

    let out = gather(&operand, &start_indices, &config).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(get_f64(&out, &[0]), 50.0);
    assert_eq!(get_f64(&out, &[1]), 20.0);
    assert_eq!(get_f64(&out, &[2]), 10.0);
}

#[test]
fn test_scatter_accepts_i64_indices() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3]).unwrap());
    let scatter_indices =
        Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 1, 2, 0, 1, 2]).unwrap();
    let updates =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![5.0, 6.0, 7.0]).unwrap());

    let out = scatter(
        &operand,
        &scatter_indices,
        &updates,
        &diagonal_scatter_config(),
    )
    .unwrap();

    assert_eq!(scatter_indices.dtype(), DType::I64);
    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 5.0);
    assert_eq!(get_f64(&out, &[1, 1]), 6.0);
    assert_eq!(get_f64(&out, &[2, 2]), 7.0);
}

#[test]
fn test_scatter_to_diagonal() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3]).unwrap());
    let scatter_indices =
        Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 1, 2, 0, 1, 2]).unwrap();
    let updates =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![5.0, 6.0, 7.0]).unwrap());

    let out = scatter(
        &operand,
        &scatter_indices,
        &updates,
        &diagonal_scatter_config(),
    )
    .unwrap();

    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 5.0);
    assert_eq!(get_f64(&out, &[1, 1]), 6.0);
    assert_eq!(get_f64(&out, &[2, 2]), 7.0);
    assert_eq!(get_f64(&out, &[1, 0]), 0.0);
    assert_eq!(get_f64(&out, &[0, 2]), 0.0);
}

#[test]
fn test_scatter_clamps_negative_and_out_of_bounds_windows() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![4]).unwrap());
    let scatter_indices = Tensor::from_vec_col_major(vec![3, 1], vec![-1_i64, 2, 4]).unwrap();
    let updates =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![5.0, 6.0, 7.0]).unwrap());
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let out = scatter(&operand, &scatter_indices, &updates, &config).unwrap();
    assert_eq!(out.shape(), &[4]);
    assert_eq!(get_f64(&out, &[0]), 5.0);
    assert_eq!(get_f64(&out, &[1]), 0.0);
    assert_eq!(get_f64(&out, &[2]), 6.0);
    assert_eq!(get_f64(&out, &[3]), 7.0);
}

#[test]
fn test_pad_adds_zero_edges() {
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let config = PadConfig {
        edge_padding_low: vec![1, 1],
        edge_padding_high: vec![1, 1],
        interior_padding: vec![0, 0],
    };

    let out = pad(&input, &config).unwrap();

    assert_eq!(out.shape(), &[4, 5]);
    assert_eq!(get_f64(&out, &[1, 1]), 1.0);
    assert_eq!(get_f64(&out, &[2, 1]), 2.0);
    assert_eq!(get_f64(&out, &[1, 2]), 3.0);
    assert_eq!(get_f64(&out, &[2, 3]), 6.0);
    assert_eq!(get_f64(&out, &[0, 0]), 0.0);
    assert_eq!(get_f64(&out, &[3, 4]), 0.0);
}

#[test]
fn test_pad_with_interior_spacing() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
    let config = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![1],
    };

    let out = pad(&input, &config).unwrap();
    assert_eq!(out.shape(), &[5]);
    assert_eq!(get_f64(&out, &[0]), 0.0);
    assert_eq!(get_f64(&out, &[1]), 1.0);
    assert_eq!(get_f64(&out, &[2]), 0.0);
    assert_eq!(get_f64(&out, &[3]), 2.0);
    assert_eq!(get_f64(&out, &[4]), 0.0);
}

#[test]
fn test_dynamic_slice_clamps_starts() {
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );
    let starts = Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]).unwrap();

    let out = dynamic_slice(&input, &starts, &[2, 2]).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64(&out, &[0, 0]), 11.0);
    assert_eq!(get_f64(&out, &[1, 0]), 12.0);
    assert_eq!(get_f64(&out, &[0, 1]), 15.0);
    assert_eq!(get_f64(&out, &[1, 1]), 16.0);
}

#[test]
fn test_dynamic_slice_accepts_i64_starts() {
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );
    let starts = Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]).unwrap();

    let out = dynamic_slice(&input, &starts, &[2, 2]).unwrap();

    assert_eq!(starts.dtype(), DType::I64);
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64(&out, &[0, 0]), 11.0);
    assert_eq!(get_f64(&out, &[1, 0]), 12.0);
    assert_eq!(get_f64(&out, &[0, 1]), 15.0);
    assert_eq!(get_f64(&out, &[1, 1]), 16.0);
}

#[test]
fn test_dynamic_update_slice_clamps_starts() {
    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![10.0, 11.0, 12.0, 13.0, 14.0]).unwrap(),
    );
    let update =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    let starts = Tensor::from_vec_col_major(vec![1], vec![4_i64]).unwrap();

    let out = dynamic_update_slice(&operand, &update, &starts).unwrap();

    assert_eq!(out.shape(), &[5]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[10.0, 11.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_slice_concatenate_and_reverse_edge_cases() {
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![4, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap(),
    );
    let config = SliceConfig {
        starts: vec![0, 0],
        limits: vec![4, 3],
        strides: vec![2, 2],
    };
    let mut backend = CpuBackend::new();
    let sliced = backend.slice(&input, &config).unwrap();
    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(get_f64(&sliced, &[0, 0]), 1.0);
    assert_eq!(get_f64(&sliced, &[1, 0]), 3.0);
    assert_eq!(get_f64(&sliced, &[0, 1]), 9.0);
    assert_eq!(get_f64(&sliced, &[1, 1]), 11.0);

    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![3.0, 4.0]).unwrap());
    let c = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![5.0, 6.0]).unwrap());
    let concatenated = backend.concatenate(&[&a, &b, &c], 1).unwrap();
    assert_eq!(concatenated.shape(), &[2, 3]);
    assert_eq!(get_f64(&concatenated, &[0, 0]), 1.0);
    assert_eq!(get_f64(&concatenated, &[1, 1]), 4.0);
    assert_eq!(get_f64(&concatenated, &[0, 2]), 5.0);

    let reversed = backend.reverse(&input, &[0, 1]).unwrap();
    assert_eq!(reversed.shape(), &[4, 3]);
    assert_eq!(get_f64(&reversed, &[0, 0]), 12.0);
    assert_eq!(get_f64(&reversed, &[3, 2]), 1.0);
}

#[test]
fn test_structural_convert_helper_returns_result() {
    let input =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.25_f32, -2.5_f32]).unwrap());

    let output = crate::structural::convert(&input, DType::F64).unwrap();

    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.dtype(), DType::F64);
    assert_eq!(get_f64(&output, &[0]), 1.25);
    assert_eq!(get_f64(&output, &[1]), -2.5);
}

#[test]
fn test_structural_convert_rejects_lossy_dtype_projection() {
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.25_f64, -2.5_f64]).unwrap());

    let err = crate::structural::convert(&input, DType::I32).unwrap_err();

    assert!(matches!(
        err,
        crate::Error::UnsupportedDTypeConversion {
            op: "convert",
            from: DType::F64,
            to: DType::I32,
            ..
        }
    ));
}

#[test]
fn test_backend_cast_supports_real_complex_and_precision_changes() {
    let mut backend = CpuBackend::new();
    let f32_input =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.25_f32, -2.5_f32]).unwrap());
    let f64_input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.25_f64, -2.5_f64]).unwrap());
    let i64_input =
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![1_i64, -2_i64]).unwrap());
    let c32_input = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.25, -0.5), Complex32::new(-2.5, 4.0)],
        )
        .unwrap(),
    );
    let c64_input = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.25, -0.5), Complex64::new(-2.5, 4.0)],
        )
        .unwrap(),
    );

    let cases = [
        (&f32_input, DType::F32),
        (&f32_input, DType::F64),
        (&f32_input, DType::I64),
        (&f32_input, DType::C32),
        (&f32_input, DType::C64),
        (&f64_input, DType::F32),
        (&f64_input, DType::F64),
        (&f64_input, DType::I64),
        (&f64_input, DType::C32),
        (&f64_input, DType::C64),
        (&i64_input, DType::F32),
        (&i64_input, DType::F64),
        (&i64_input, DType::I64),
        (&i64_input, DType::C32),
        (&i64_input, DType::C64),
        (&c32_input, DType::F32),
        (&c32_input, DType::F64),
        (&c32_input, DType::I64),
        (&c32_input, DType::C32),
        (&c32_input, DType::C64),
        (&c64_input, DType::F32),
        (&c64_input, DType::F64),
        (&c64_input, DType::I64),
        (&c64_input, DType::C32),
        (&c64_input, DType::C64),
    ];

    for (input, to) in cases {
        let output = backend.cast(input, to).unwrap();
        assert_eq!(output.shape(), &[2]);
        assert_eq!(output.dtype(), to);

        match (input.dtype(), &output) {
            (DType::F32, Tensor::F32(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.25, -2.5])
            }
            (DType::F32, Tensor::F64(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.25, -2.5])
            }
            (DType::F32, Tensor::I64(inner)) => assert_eq!(inner.host_data().unwrap(), &[1, -2]),
            (DType::F32, Tensor::C32(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex32::new(1.25, 0.0), Complex32::new(-2.5, 0.0)]
            ),
            (DType::F32, Tensor::C64(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex64::new(1.25, 0.0), Complex64::new(-2.5, 0.0)]
            ),
            (DType::F64, Tensor::F32(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.25, -2.5])
            }
            (DType::F64, Tensor::F64(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.25, -2.5])
            }
            (DType::F64, Tensor::I64(inner)) => assert_eq!(inner.host_data().unwrap(), &[1, -2]),
            (DType::F64, Tensor::C32(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex32::new(1.25, 0.0), Complex32::new(-2.5, 0.0)]
            ),
            (DType::F64, Tensor::C64(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex64::new(1.25, 0.0), Complex64::new(-2.5, 0.0)]
            ),
            (DType::I64, Tensor::F32(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.0, -2.0])
            }
            (DType::I64, Tensor::F64(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.0, -2.0])
            }
            (DType::I64, Tensor::I64(inner)) => assert_eq!(inner.host_data().unwrap(), &[1, -2]),
            (DType::I64, Tensor::C32(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex32::new(1.0, 0.0), Complex32::new(-2.0, 0.0)]
            ),
            (DType::I64, Tensor::C64(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)]
            ),
            (DType::C32, Tensor::F32(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.25, -2.5])
            }
            (DType::C32, Tensor::F64(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.25, -2.5])
            }
            (DType::C32, Tensor::I64(inner)) => assert_eq!(inner.host_data().unwrap(), &[1, -2]),
            (DType::C32, Tensor::C32(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex32::new(1.25, -0.5), Complex32::new(-2.5, 4.0)]
            ),
            (DType::C32, Tensor::C64(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex64::new(1.25, -0.5), Complex64::new(-2.5, 4.0)]
            ),
            (DType::C64, Tensor::F32(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.25, -2.5])
            }
            (DType::C64, Tensor::F64(inner)) => {
                assert_eq!(inner.host_data().unwrap(), &[1.25, -2.5])
            }
            (DType::C64, Tensor::I64(inner)) => assert_eq!(inner.host_data().unwrap(), &[1, -2]),
            (DType::C64, Tensor::C32(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex32::new(1.25, -0.5), Complex32::new(-2.5, 4.0)]
            ),
            (DType::C64, Tensor::C64(inner)) => assert_eq!(
                inner.host_data().unwrap(),
                &[Complex64::new(1.25, -0.5), Complex64::new(-2.5, 4.0)]
            ),
            _ => unreachable!("unexpected conversion case"),
        }
    }
}

#[test]
fn test_backend_cast_rejects_nonfinite_or_out_of_range_float_to_int_values() {
    let mut backend = CpuBackend::new();

    let f64_bad = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![4],
            vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                i32::MAX as f64 + 1.0,
            ],
        )
        .unwrap(),
    );
    let err = backend.cast(&f64_bad, DType::I32).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "cast",
            source: tenferro_tensor::ValidationError::InvalidArgument {
                argument: "value",
                message,
            },
        } if message.contains("finite") || message.contains("out of i32 range")
    ));

    let f32_bad =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![1], vec![i64::MAX as f32]).unwrap());
    let err = backend.cast(&f32_bad, DType::I64).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "cast",
            source,
        } if source.to_string().contains("out of i64 range")
    ));

    let c64_bad = Tensor::C64(
        TypedTensor::from_vec_col_major(vec![1], vec![Complex64::new(f64::INFINITY, 0.0)]).unwrap(),
    );
    let err = backend.cast(&c64_bad, DType::I64).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "cast",
            source,
        } if source.to_string().contains("finite")
    ));
}

#[test]
fn test_cpu_supports_i32_and_bool_structural_paths() {
    let mut backend = CpuBackend::new();

    let i32_tensor = Tensor::from_vec_col_major(vec![2], vec![-1_i32, 0]).unwrap();
    let i32_as_bool = backend.cast(&i32_tensor, DType::Bool).unwrap();
    assert_eq!(i32_as_bool.as_slice::<bool>().unwrap(), &[true, false]);

    let bool_tensor = Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    let bool_as_i64 = backend.convert(&bool_tensor, DType::I64).unwrap();
    assert_eq!(bool_as_i64.as_slice::<i64>().unwrap(), &[1, 0]);

    let bool_matrix =
        Tensor::from_vec_col_major(vec![2, 3], vec![true, false, false, true, true, false])
            .unwrap();
    let transposed = transpose(&bool_matrix, &[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(
        transposed.as_slice::<bool>().unwrap(),
        &[true, false, true, false, true, false]
    );

    let padded = pad(
        &bool_tensor,
        &PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        },
    )
    .unwrap();
    assert_eq!(
        padded.as_slice::<bool>().unwrap(),
        &[false, true, false, false]
    );

    let starts = Tensor::from_vec_col_major(vec![1], vec![1_i32]).unwrap();
    let sliced = dynamic_slice(
        &Tensor::from_vec_col_major(vec![3], vec![true, false, true]).unwrap(),
        &starts,
        &[2],
    )
    .unwrap();
    assert_eq!(sliced.as_slice::<bool>().unwrap(), &[false, true]);

    let upper = triu(
        &Tensor::from_vec_col_major(vec![2, 2], vec![true, true, false, true]).unwrap(),
        0,
    )
    .unwrap();
    assert_eq!(
        upper.as_slice::<bool>().unwrap(),
        &[true, false, false, true]
    );

    let i32_sum = reduce_sum(
        &Tensor::from_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap(),
        &[0],
    )
    .unwrap();
    assert_eq!(i32_sum.as_slice::<i32>().unwrap(), &[3, 7]);
}

#[test]
fn test_backend_default_and_buffer_pool_len() {
    let backend = CpuBackend::default();
    assert!(backend.num_threads() >= 1);
    assert_eq!(backend.buffer_pool_len(), 0);
}

#[test]
fn test_backend_buffer_pool_controls_report_and_update_limits() {
    let ctx = Arc::new(CpuContext::with_threads(1).unwrap());
    let mut backend = CpuBackend::from_context_with_buffer_pool_limit(ctx, 64);

    assert_eq!(backend.num_threads(), 1);
    assert_eq!(backend.buffer_pool_limit_bytes(), 64);
    assert_eq!(backend.buffer_pool_len(), 0);
    let stats = backend.buffer_pool_stats();
    assert_eq!(stats.buffers, 0);
    assert_eq!(stats.capacity_bytes, 0);
    let cache_stats = backend.buffer_pool_cache_stats();
    assert_eq!(cache_stats.entries, 0);
    assert_eq!(cache_stats.retained_bytes, 0);

    backend.set_buffer_pool_limit_bytes(0);
    assert_eq!(backend.buffer_pool_limit_bytes(), 0);
    backend.reset_buffer_pool();
    assert_eq!(backend.buffer_pool_len(), 0);
}

#[test]
fn test_backend_mul_neg_conj_dispatch() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap());
    let c = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();

    let prod = TensorElementwise::mul(&mut backend, &a, &b).unwrap();
    assert_eq!(get_f64(&prod, &[0]), 3.0);
    assert_eq!(get_f64(&prod, &[1]), -8.0);

    let negated = backend.neg(&a).unwrap();
    assert_eq!(get_f64(&negated, &[0]), -1.0);
    assert_eq!(get_f64(&negated, &[1]), 2.0);

    let conjugated = backend.conj(&c).unwrap();
    assert_c64_close(get_c64(&conjugated, &[0]), Complex64::new(1.0, -2.0));
    assert_c64_close(get_c64(&conjugated, &[1]), Complex64::new(-3.0, -0.5));
}

#[test]
fn test_backend_structural_ops_dispatch() {
    let mut backend = CpuBackend::new();
    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());

    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![5.0]).unwrap());
    let broadcast = backend.broadcast_in_dim(&scalar, &[2, 2], &[]).unwrap();
    assert_eq!(broadcast.shape(), &[2, 2]);
    assert_eq!(get_f64(&broadcast, &[0, 0]), 5.0);
    assert_eq!(get_f64(&broadcast, &[1, 1]), 5.0);

    let diag = backend.extract_diagonal(&a, 0, 1).unwrap();
    assert_eq!(diag.shape(), &[2]);
    assert_eq!(get_f64(&diag, &[0]), 1.0);
    assert_eq!(get_f64(&diag, &[1]), 4.0);

    let d = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![10.0, 20.0]).unwrap());
    let embedded = backend.embed_diagonal(&d, 0, 1).unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(get_f64(&embedded, &[0, 0]), 10.0);
    assert_eq!(get_f64(&embedded, &[1, 1]), 20.0);

    let tril_result = backend.tril(&a, 0).unwrap();
    assert_eq!(tril_result.shape(), &[2, 2]);
    assert_eq!(get_f64(&tril_result, &[0, 1]), 0.0);

    let triu_result = backend.triu(&a, 0).unwrap();
    assert_eq!(triu_result.shape(), &[2, 2]);
    assert_eq!(get_f64(&triu_result, &[1, 0]), 0.0);

    let summed = TensorReduction::reduce_sum(&mut backend, &a, &[0]).unwrap();
    assert_eq!(summed.shape(), &[2]);
    assert_eq!(get_f64(&summed, &[0]), 3.0);
    assert_eq!(get_f64(&summed, &[1]), 7.0);
}

#[test]
fn test_backend_dot_general_f32_c32_and_dtype_mismatch() {
    let mut backend = CpuBackend::new();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let a_f32 =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![1, 2], vec![1.0f32, 2.0]).unwrap());
    let b_f32 =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2, 1], vec![3.0f32, 4.0]).unwrap());
    let out_f32 = backend.dot_general(&a_f32, &b_f32, &config).unwrap();
    assert_eq!(out_f32.shape(), &[1, 1]);

    let a_c32 = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![1, 2],
            vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
        )
        .unwrap(),
    );
    let b_c32 = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2, 1],
            vec![Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)],
        )
        .unwrap(),
    );
    let out_c32 = backend.dot_general(&a_c32, &b_c32, &config).unwrap();
    assert_eq!(out_c32.shape(), &[1, 1]);

    let f64_t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
    let f32_t = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap());
    let err = backend.dot_general(&f64_t, &f32_t, &config).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        }
    ));
}

#[test]
fn test_backend_gather_scatter_dynamic_slice_dispatch() {
    let mut backend = CpuBackend::new();

    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
    );
    let start_indices = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 2, 4]).unwrap();
    let gathered = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap();
    assert_eq!(gathered.shape(), &[3]);
    assert_eq!(get_f64(&gathered, &[0]), 10.0);
    assert_eq!(get_f64(&gathered, &[2]), 50.0);

    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3]).unwrap());
    let scatter_indices =
        Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 1, 2, 0, 1, 2]).unwrap();
    let updates =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![5.0, 6.0, 7.0]).unwrap());
    let scattered = backend
        .scatter(
            &operand,
            &scatter_indices,
            &updates,
            &diagonal_scatter_config(),
        )
        .unwrap();
    assert_eq!(get_f64(&scattered, &[0, 0]), 5.0);
    assert_eq!(get_f64(&scattered, &[1, 1]), 6.0);
    assert_eq!(get_f64(&scattered, &[2, 2]), 7.0);

    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );
    let starts = Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]).unwrap();
    let ds = backend.dynamic_slice(&input, &starts, &[2, 2]).unwrap();
    assert_eq!(ds.shape(), &[2, 2]);
    assert_eq!(get_f64(&ds, &[0, 0]), 11.0);
    assert_eq!(get_f64(&ds, &[1, 1]), 16.0);
}
