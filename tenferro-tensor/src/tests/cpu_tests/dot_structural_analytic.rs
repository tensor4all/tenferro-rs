use super::*;

#[test]
fn test_dot_general_matmul() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 4],
        vec![
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
    ));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
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
    let lhs = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        lhs_data.clone(),
    ));
    let rhs = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        rhs_data.clone(),
    ));
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut backend = CpuBackend::new();

    let out = backend
        .dot_general_with_conj(&lhs, &rhs, &config, true, true)
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
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs_shape = [3usize, 2];
    let rhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_view = TensorView::f64(&rhs_shape, &rhs_data).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut backend = CpuBackend::new();

    let direct = backend
        .dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_view(rhs_view),
            &config,
        )
        .unwrap();
    assert_eq!(direct.shape(), &[2, 2]);
    assert_eq!(direct.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);

    let session = backend.with_backend_session(|exec| {
        exec.dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_view(rhs_view),
            &config,
        )
    });
    let session = session.unwrap();
    assert_eq!(
        session.as_slice::<f64>().unwrap(),
        &[22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn test_dot_general_inner_product_returns_rank0_scalar() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![4.0, 5.0, 6.0],
    ));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![0],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    assert!(c.shape().is_empty());
    assert_eq!(get_f64(&c, &[]), 32.0);
}

#[test]
fn test_dot_general_zero_sized_matmul_returns_empty_matrix() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 0], Vec::new()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 0], Vec::new()));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();

    assert_eq!(c.shape(), &[0, 0]);
    match c {
        Tensor::F64(inner) => assert!(inner.host_data().is_empty()),
        _ => panic!("expected F64 tensor"),
    }
}

#[test]
fn test_dot_general_zero_contracting_dim_returns_zero_filled_output() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 0], Vec::new()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 3], Vec::new()));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    match c {
        Tensor::F64(inner) => assert_eq!(inner.host_data(), &[0.0; 6]),
        _ => panic!("expected F64 tensor"),
    }
}

#[test]
fn test_dot_general_falls_back_for_unfusable_lhs_batch_layout() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2, 2],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2, 2],
        vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ],
    ));
    let mut backend = CpuBackend::new();
    let c = backend
        .dot_general(
            &a,
            &b,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![3],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![0, 2],
                rhs_batch_dims: vec![2, 3],
            },
        )
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
fn test_transpose() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
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
    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![5.0]));
    let broadcast = broadcast_in_dim(&scalar, &[3], &[]).unwrap();
    assert_eq!(broadcast.shape(), &[3]);
    assert_eq!(get_f64(&broadcast, &[0]), 5.0);
    assert_eq!(get_f64(&broadcast, &[1]), 5.0);
    assert_eq!(get_f64(&broadcast, &[2]), 5.0);

    let v = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
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
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let lower = tril(&t, 0).unwrap();
    assert_eq!(lower.shape(), &[3, 3]);
    assert_eq!(
        match &lower {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
    );
}

#[test]
fn test_triu_3x3() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let upper = triu(&t, 0).unwrap();
    assert_eq!(upper.shape(), &[3, 3]);
    assert_eq!(
        match &upper {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_tril_triu_zero_sized_batch_return_empty_tensor() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2, 0], Vec::new()));

    let lower = tril(&t, 0).unwrap();
    assert_eq!(lower.shape(), &[2, 2, 0]);
    match lower {
        Tensor::F64(inner) => assert!(inner.host_data().is_empty()),
        _ => panic!("expected f64 tensor"),
    }

    let upper = triu(&t, 0).unwrap();
    assert_eq!(upper.shape(), &[2, 2, 0]);
    match upper {
        Tensor::F64(inner) => assert!(inner.host_data().is_empty()),
        _ => panic!("expected f64 tensor"),
    }
}

#[test]
fn test_tril_triu_extreme_offsets_do_not_overflow() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));

    let lower_min = tril(&t, i64::MIN).unwrap();
    assert_eq!(
        match &lower_min {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[0.0, 0.0, 0.0, 0.0]
    );

    let upper_min = triu(&t, i64::MIN).unwrap();
    assert_eq!(
        match &upper_min {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[1.0, 2.0, 3.0, 4.0]
    );

    let lower_max = tril(&t, i64::MAX).unwrap();
    assert_eq!(
        match &lower_max {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[1.0, 2.0, 3.0, 4.0]
    );

    let upper_max = triu(&t, i64::MAX).unwrap();
    assert_eq!(
        match &upper_max {
            Tensor::F64(inner) => inner.host_data(),
            _ => panic!("expected f64 tensor"),
        },
        &[0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn test_neg_and_conj() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, -7.0]));
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

    let exp_input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![0.0, 1.0]));
    let exp_out = backend.exp(&exp_input).unwrap();
    assert_f64_close(get_f64(&exp_out, &[0]), 1.0);
    assert_f64_close(get_f64(&exp_out, &[1]), std::f64::consts::E);

    let log_input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 4.0]));
    let log_out = backend.log(&log_input).unwrap();
    assert_f64_close(get_f64(&log_out, &[0]), 0.0);
    assert_f64_close(get_f64(&log_out, &[1]), 4.0_f64.ln());

    let trig_input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![0.0, std::f64::consts::FRAC_PI_2],
    ));
    let sin_out = backend.sin(&trig_input).unwrap();
    let cos_out = backend.cos(&trig_input).unwrap();
    assert_f64_close(get_f64(&sin_out, &[0]), 0.0);
    assert_f64_close(get_f64(&sin_out, &[1]), 1.0);
    assert_f64_close(get_f64(&cos_out, &[0]), 1.0);
    assert_f64_close(get_f64(&cos_out, &[1]), 0.0);

    let tanh_input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![0.0, 1.0]));
    let tanh_out = backend.tanh(&tanh_input).unwrap();
    assert_f64_close(get_f64(&tanh_out, &[0]), 0.0);
    assert_f64_close(get_f64(&tanh_out, &[1]), 1.0_f64.tanh());

    let sqrt_input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 4.0]));
    let sqrt_out = backend.sqrt(&sqrt_input).unwrap();
    let rsqrt_out = backend.rsqrt(&sqrt_input).unwrap();
    assert_f64_close(get_f64(&sqrt_out, &[0]), 1.0);
    assert_f64_close(get_f64(&sqrt_out, &[1]), 2.0);
    assert_f64_close(get_f64(&rsqrt_out, &[0]), 1.0);
    assert_f64_close(get_f64(&rsqrt_out, &[1]), 0.5);

    let expm1_out = backend.expm1(&exp_input).unwrap();
    let log1p_out = backend.log1p(&log_input).unwrap();
    assert_f64_close(get_f64(&expm1_out, &[0]), 0.0);
    assert_f64_close(get_f64(&expm1_out, &[1]), 1.0_f64.exp_m1());
    assert_f64_close(get_f64(&log1p_out, &[0]), 2.0_f64.ln());
    assert_f64_close(get_f64(&log1p_out, &[1]), 5.0_f64.ln());

    let pow_base = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0, 9.0]));
    let pow_exp = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 0.5]));
    let pow_out = backend.pow(&pow_base, &pow_exp).unwrap();
    assert_f64_close(get_f64(&pow_out, &[0]), 8.0);
    assert_f64_close(get_f64(&pow_out, &[1]), 3.0);
}

#[test]
fn test_cpu_backend_analytic_ops_complex() {
    let mut backend = CpuBackend::new();

    let exp_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 1.0)],
    ));
    let exp_out = backend.exp(&exp_input).unwrap();
    assert_c64_close(get_c64(&exp_out, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&exp_out, &[1]), Complex64::new(1.0, 1.0).exp());

    let log_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, -0.5)],
    ));
    let log_out = backend.log(&log_input).unwrap();
    assert_c64_close(get_c64(&log_out, &[0]), Complex64::new(1.0, 0.0).ln());
    assert_c64_close(get_c64(&log_out, &[1]), Complex64::new(2.0, -0.5).ln());

    let trig_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, -0.25)],
    ));
    let sin_out = backend.sin(&trig_input).unwrap();
    let cos_out = backend.cos(&trig_input).unwrap();
    let tanh_out = backend.tanh(&trig_input).unwrap();
    assert_c64_close(get_c64(&sin_out, &[0]), Complex64::new(0.0, 0.0).sin());
    assert_c64_close(get_c64(&sin_out, &[1]), Complex64::new(0.5, -0.25).sin());
    assert_c64_close(get_c64(&cos_out, &[0]), Complex64::new(0.0, 0.0).cos());
    assert_c64_close(get_c64(&cos_out, &[1]), Complex64::new(0.5, -0.25).cos());
    assert_c64_close(get_c64(&tanh_out, &[0]), Complex64::new(0.0, 0.0).tanh());
    assert_c64_close(get_c64(&tanh_out, &[1]), Complex64::new(0.5, -0.25).tanh());

    let sqrt_input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(4.0, 3.0)],
    ));
    let sqrt_out = backend.sqrt(&sqrt_input).unwrap();
    let rsqrt_out = backend.rsqrt(&sqrt_input).unwrap();
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

    let expm1_out = backend.expm1(&exp_input).unwrap();
    let log1p_out = backend.log1p(&log_input).unwrap();
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

    let pow_base = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    ));
    let pow_exp = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.5, 0.25)],
    ));
    let pow_out = backend.pow(&pow_base, &pow_exp).unwrap();
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
    let square = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let d = extract_diagonal(&square, 0, 1).unwrap();
    assert_eq!(d.shape(), &[3]);
    assert_eq!(get_f64(&d, &[0]), 1.0);
    assert_eq!(get_f64(&d, &[1]), 5.0);
    assert_eq!(get_f64(&d, &[2]), 9.0);

    let cube = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 3, 3],
        (1..=18).map(|x| x as f64).collect(),
    ));
    let diag = extract_diagonal(&cube, 1, 2).unwrap();
    assert_eq!(diag.shape(), &[2, 3]);
    assert_eq!(get_f64(&diag, &[0, 0]), 1.0);
    assert_eq!(get_f64(&diag, &[1, 1]), 10.0);
    assert_eq!(get_f64(&diag, &[1, 2]), 18.0);
}

#[test]
fn test_embed_diagonal() {
    let v = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
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
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]));
    let mut backend = CpuBackend::new();
    let out = TensorBackend::add(&mut backend, &a, &b).unwrap();
    assert_eq!(get_f64(&out, &[0]), 4.0);
    assert_eq!(get_f64(&out, &[1]), 6.0);
}

#[test]
fn test_tier2_elementwise_ops_real() {
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![8.0, -2.0, 9.0],
    ));
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![2.0, 5.0, 3.0],
    ));
    let pred = Tensor::Bool(TypedTensor::from_vec_col_major(
        vec![3],
        vec![false, true, true],
    ));
    let on_true = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![10.0, 20.0, 30.0],
    ));
    let on_false = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    let lower = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![-1.0, -1.0, 0.0],
    ));
    let upper = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 0.25, 4.0],
    ));
    let mut backend = CpuBackend::new();

    let div = backend.div(&lhs, &rhs).unwrap();
    assert_eq!(get_f64(&div, &[0]), 4.0);
    assert_eq!(get_f64(&div, &[1]), -0.4);
    assert_eq!(get_f64(&div, &[2]), 3.0);

    let abs = backend.abs(&lhs).unwrap();
    assert_eq!(get_f64(&abs, &[0]), 8.0);
    assert_eq!(get_f64(&abs, &[1]), 2.0);
    assert_eq!(get_f64(&abs, &[2]), 9.0);

    let sign = backend.sign(&lhs).unwrap();
    assert_eq!(get_f64(&sign, &[0]), 1.0);
    assert_eq!(get_f64(&sign, &[1]), -1.0);
    assert_eq!(get_f64(&sign, &[2]), 1.0);

    let maximum = backend.maximum(&lhs, &rhs).unwrap();
    assert_eq!(get_f64(&maximum, &[0]), 8.0);
    assert_eq!(get_f64(&maximum, &[1]), 5.0);
    assert_eq!(get_f64(&maximum, &[2]), 9.0);

    let minimum = backend.minimum(&lhs, &rhs).unwrap();
    assert_eq!(get_f64(&minimum, &[0]), 2.0);
    assert_eq!(get_f64(&minimum, &[1]), -2.0);
    assert_eq!(get_f64(&minimum, &[2]), 3.0);

    let eq = backend.compare(&lhs, &rhs, &CompareDir::Eq).unwrap();
    assert!(!get_bool(&eq, &[0]));
    assert!(!get_bool(&eq, &[1]));
    assert!(!get_bool(&eq, &[2]));

    let lt = backend.compare(&lhs, &rhs, &CompareDir::Lt).unwrap();
    assert!(!get_bool(&lt, &[0]));
    assert!(get_bool(&lt, &[1]));
    assert!(!get_bool(&lt, &[2]));

    let le = backend.compare(&lhs, &rhs, &CompareDir::Le).unwrap();
    assert!(!get_bool(&le, &[0]));
    assert!(get_bool(&le, &[1]));
    assert!(!get_bool(&le, &[2]));

    let gt = backend.compare(&lhs, &rhs, &CompareDir::Gt).unwrap();
    assert!(get_bool(&gt, &[0]));
    assert!(!get_bool(&gt, &[1]));
    assert!(get_bool(&gt, &[2]));

    let ge = backend.compare(&lhs, &rhs, &CompareDir::Ge).unwrap();
    assert!(get_bool(&ge, &[0]));
    assert!(!get_bool(&ge, &[1]));
    assert!(get_bool(&ge, &[2]));

    let select = backend.select(&pred, &on_true, &on_false).unwrap();
    assert_eq!(get_f64(&select, &[0]), 1.0);
    assert_eq!(get_f64(&select, &[1]), 20.0);
    assert_eq!(get_f64(&select, &[2]), 30.0);

    let clamp = backend.clamp(&lhs, &lower, &upper).unwrap();
    assert_eq!(get_f64(&clamp, &[0]), 1.0);
    assert_eq!(get_f64(&clamp, &[1]), -1.0);
    assert_eq!(get_f64(&clamp, &[2]), 4.0);
}

#[test]
fn test_tier2_elementwise_ops_complex() {
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
    ));
    let lhs = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
    ));
    let rhs = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 2.0)],
    ));
    let mut backend = CpuBackend::new();

    let abs = backend.abs(&input).unwrap();
    assert_c64_close(get_c64(&abs, &[0]), Complex64::new(5.0, 0.0));
    assert_c64_close(get_c64(&abs, &[1]), Complex64::new(0.0, 0.0));

    let sign = backend.sign(&input).unwrap();
    assert_c64_close(get_c64(&sign, &[0]), Complex64::new(0.6, 0.8));
    assert_c64_close(get_c64(&sign, &[1]), Complex64::new(0.0, 0.0));

    let maximum = backend.maximum(&lhs, &rhs).unwrap();
    assert_c64_close(get_c64(&maximum, &[0]), Complex64::new(3.0, 4.0));
    assert_c64_close(get_c64(&maximum, &[1]), Complex64::new(0.0, 2.0));

    let minimum = backend.minimum(&lhs, &rhs).unwrap();
    assert_c64_close(get_c64(&minimum, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&minimum, &[1]), Complex64::new(1.0, 0.0));
}
