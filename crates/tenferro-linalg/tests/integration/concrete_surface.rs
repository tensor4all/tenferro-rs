use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::{
    EighOptions, LinalgBackend, QrOptions, RankRevealingQrOptions, SvdOptions, TensorLinalgExt,
    TensorReadLinalgExt, TypedTensorLinalgExt,
};
use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorScalar, TypedTensor};

#[test]
fn cpu_execution_session_implements_linalg_backend() {
    fn assert_linalg_backend<B: LinalgBackend>() {}

    assert_linalg_backend::<tenferro_cpu::CpuExecSession<'static>>();
}

#[test]
fn dynamic_and_read_surfaces_return_fixed_tuples() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let (_u, s, _vt) = input.svd(session).unwrap();
        let svdvals = input.svdvals(session).unwrap();
        let (_q, r) = TensorRead::from_tensor(&input).qr_read(session).unwrap();
        let (sign, logabsdet) = input.slogdet(session).unwrap();

        assert_eq!(s.as_slice::<f64>().unwrap(), &[4.0, 2.0]);
        assert_eq!(svdvals.as_slice::<f64>().unwrap(), &[4.0, 2.0]);
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(sign.as_slice::<f64>().unwrap(), &[1.0]);
        assert!((logabsdet.as_slice::<f64>().unwrap()[0] - 8.0_f64.ln()).abs() < 1.0e-12);
    });
}

#[test]
fn rank_revealing_qr_handles_interspersed_dependence() {
    // Columns are a, a, b. A prefix-only non-pivoted rank guard cannot retain
    // the later independent column, whereas CPQR must report rank two.
    let input = Tensor::from_vec_col_major(
        vec![4, 3],
        vec![
            1.0_f64, 2.0, 3.0, 4.0, // a
            1.0, 2.0, 3.0, 4.0, // duplicate a
            0.0, 1.0, 0.0, 0.0, // independent b
        ],
    )
    .unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let result = input
            .rank_revealing_qr(RankRevealingQrOptions::default().rtol(1.0e-12), session)
            .unwrap();
        assert_eq!(result.q.shape(), &[4, 3]);
        assert_eq!(result.r.shape(), &[3, 3]);
        assert_eq!(result.column_permutation.shape(), &[3]);
        assert_eq!(result.rank.shape(), &[] as &[usize]);
        assert_eq!(result.rank.as_slice::<i64>().unwrap(), &[2]);

        let permutation = result.column_permutation.as_slice::<i64>().unwrap();
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);

        let q = result.q.as_slice::<f64>().unwrap();
        let r = result.r.as_slice::<f64>().unwrap();
        let source = input.as_slice::<f64>().unwrap();
        for lhs_col in 0..3 {
            for rhs_col in 0..3 {
                let inner = (0..4)
                    .map(|row| q[row + lhs_col * 4] * q[row + rhs_col * 4])
                    .sum::<f64>();
                let expected = if lhs_col == rhs_col { 1.0 } else { 0.0 };
                assert!((inner - expected).abs() < 1.0e-11);
            }
        }
        for factor_col in 0..3 {
            let source_col = usize::try_from(permutation[factor_col]).unwrap();
            for row in 0..4 {
                let reconstructed = (0..3)
                    .map(|inner| q[row + inner * 4] * r[inner + factor_col * 3])
                    .sum::<f64>();
                assert!((reconstructed - source[row + source_col * 4]).abs() < 1.0e-11);
            }
        }
    });
}

#[test]
fn rank_revealing_qr_supports_all_float_and_complex_dtypes() {
    use num_complex::Complex32;

    let inputs = vec![
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 0.0, 1.0, 0.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 1.0, 0.0]).unwrap(),
        Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex32::new(1.0, 1.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 1.0),
                Complex32::new(0.0, 0.0),
            ],
        )
        .unwrap(),
        Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap(),
    ];
    let mut host = CpuBackend::new();
    host.with_backend_session(|session| {
        for input in &inputs {
            let result = input
                .rank_revealing_qr(RankRevealingQrOptions::default().rtol(1.0e-6), session)
                .unwrap();
            assert_eq!(result.q.dtype(), input.dtype());
            assert_eq!(result.r.dtype(), input.dtype());
            assert_eq!(result.rank.as_slice::<i64>().unwrap(), &[1]);
        }
    });
}

#[test]
fn rank_revealing_qr_handles_empty_dimensions_and_empty_batches() {
    let inputs: [(Vec<usize>, Vec<f64>); 3] = [
        (vec![0, 3], vec![]),
        (vec![3, 0], vec![]),
        (vec![2, 2, 0], vec![]),
    ];
    let mut host = CpuBackend::new();
    host.with_backend_session(|session| {
        for (shape, data) in inputs {
            let input = Tensor::from_vec_col_major(shape, data).unwrap();
            let result = input
                .rank_revealing_qr(RankRevealingQrOptions::default(), session)
                .unwrap();
            let ranks = result.rank.as_slice::<i64>().unwrap();
            if result.rank.shape().is_empty() {
                assert_eq!(ranks, &[0]);
            } else {
                assert!(ranks.is_empty());
            }
        }
    });
}

#[test]
fn rank_revealing_qr_rejects_non_finite_input() {
    let input = Tensor::from_vec_col_major(vec![2, 1], vec![f64::NAN, 1.0]).unwrap();
    let mut host = CpuBackend::new();
    host.with_backend_session(|session| {
        assert!(input
            .rank_revealing_qr(RankRevealingQrOptions::default(), session)
            .is_err());
    });
}

#[test]
fn rank_revealing_qr_rejects_invalid_tolerances() {
    let input = Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f64]).unwrap();
    let mut host = CpuBackend::new();
    host.with_backend_session(|session| {
        for options in [
            RankRevealingQrOptions::default().rtol(f64::NAN),
            RankRevealingQrOptions::default().rtol(-1.0),
            RankRevealingQrOptions::default().atol(f64::INFINITY),
        ] {
            assert!(input.rank_revealing_qr(options, session).is_err());
        }
    });
}

#[test]
fn rank_revealing_qr_zero_and_batched_metadata() {
    let input = Tensor::from_vec_col_major(vec![2, 2, 2], vec![0.0_f64; 8]).unwrap();
    let mut host = CpuBackend::new();
    host.with_backend_session(|session| {
        let result = input
            .rank_revealing_qr(RankRevealingQrOptions::default(), session)
            .unwrap();
        assert_eq!(result.q.shape(), &[2, 2, 2]);
        assert_eq!(result.r.shape(), &[2, 2, 2]);
        assert_eq!(result.column_permutation.shape(), &[2, 2]);
        assert_eq!(result.rank.shape(), &[2]);
        assert_eq!(result.rank.as_slice::<i64>().unwrap(), &[0, 0]);
    });
}

#[test]
fn typed_surface_exposes_associated_real_and_complex_outputs() {
    let real =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let complex = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let (_u, singular_values, _vt): (TypedTensor<f64>, TypedTensor<f64>, TypedTensor<f64>) =
            real.svd(session).unwrap();
        let (eigenvalues, _vectors): (TypedTensor<Complex64>, TypedTensor<Complex64>) =
            real.eig(session).unwrap();
        let (_cu, complex_singular_values, _cvt): (
            TypedTensor<Complex64>,
            TypedTensor<f64>,
            TypedTensor<Complex64>,
        ) = complex.svd(session).unwrap();
        let real_values: TypedTensor<f64> = real.svdvals(session).unwrap();
        let complex_values: TypedTensor<f64> = complex.svdvals(session).unwrap();

        assert_eq!(singular_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(complex_singular_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(real_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(complex_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(eigenvalues.shape(), &[2]);
    });
}

#[test]
fn typed_input_is_erased_as_a_borrowed_read() {
    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
    let read = f64::tensor_read(&input);
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let factor = read.cholesky_read(session).unwrap();
        assert_eq!(factor.shape(), &[2, 2]);
    });
}

#[test]
fn dynamic_composites_cover_inverse_pseudoinverse_eigenvalues_and_norm() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let det = input.det(session).unwrap();
        let inv = input.inv(session).unwrap();
        let pinv = input.pinv(session).unwrap();
        let eigvalsh = input.eigvalsh(session).unwrap();
        let eigvals = input.eigvals(session).unwrap();
        let norm = input.norm(None, Some(&[0, 1]), true, session).unwrap();

        assert!((det.as_slice::<f64>().unwrap()[0] - 8.0).abs() < 1.0e-12);
        assert_eq!(inv.as_slice::<f64>().unwrap(), &[0.5, 0.0, 0.0, 0.25]);
        assert_eq!(pinv.as_slice::<f64>().unwrap(), &[0.5, 0.0, 0.0, 0.25]);
        assert_eq!(eigvalsh.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
        assert_eq!(eigvals.dtype(), tenferro_tensor::DType::C64);
        assert_eq!(norm.shape(), &[1, 1]);
        assert!((norm.as_slice::<f64>().unwrap()[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
    });
}

#[test]
fn read_surface_accepts_a_strided_view_without_an_input_clone() {
    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .unwrap();
    let transposed = input.as_view().transpose_view([1, 0]).unwrap();
    let values_view = input.as_view().transpose_view([1, 0]).unwrap();
    let read = TensorRead::from_view(f64::tensor_view(transposed));
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let (_u, singular_values, _vt) = read.svd_read(session).unwrap();
        let values = TensorRead::from_view(f64::tensor_view(values_view))
            .svdvals_read(session)
            .unwrap();
        assert_eq!(singular_values.as_slice::<f64>().unwrap(), &[2.0, 1.0]);
        assert_eq!(values.as_slice::<f64>().unwrap(), &[2.0, 1.0]);
    });
}

#[test]
fn typed_complex_composites_return_real_outputs_where_required() {
    let input = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let (_sign, logabsdet): (TypedTensor<Complex64>, TypedTensor<f64>) =
            input.slogdet(session).unwrap();
        let norm: TypedTensor<f64> = input.norm(None, Some(&[0, 1]), false, session).unwrap();

        assert!((logabsdet.as_slice().unwrap()[0] - 8.0_f64.ln()).abs() < 1.0e-12);
        assert!((norm.as_slice().unwrap()[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
    });
}

#[test]
fn typed_solve_surfaces_accept_vector_and_matrix_rhs() {
    let a =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let vector = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 8.0]).unwrap();
    let matrix = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 8.0]).unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let vector_x = a.full_piv_lu_solve(&vector, session).unwrap();
        let matrix_x = a.full_piv_lu_solve(&matrix, session).unwrap();
        let triangular_x = a
            .triangular_solve(&matrix, true, false, false, false, session)
            .unwrap();

        assert_eq!(vector_x.as_slice().unwrap(), &[2.0, 2.0]);
        assert_eq!(matrix_x.as_slice().unwrap(), &[2.0, 2.0]);
        assert_eq!(triangular_x.as_slice().unwrap(), &[2.0, 2.0]);
    });
}

#[test]
fn concrete_norm_distinguishes_empty_axes_and_rejects_invalid_axes() {
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        let identity = input.norm(Some(2.0), Some(&[]), false, session).unwrap();
        let error = input
            .norm(Some(2.0), Some(&[1]), false, session)
            .unwrap_err();

        assert_eq!(identity.as_slice::<f64>().unwrap(), &[3.0, 4.0]);
        assert!(error.to_string().contains("axis 1"));
    });
}

#[test]
fn read_surface_covers_factorizations_and_composites() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![9.0_f64, 7.0]).unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        TensorRead::from_tensor(&a)
            .svd_with_options_read(SvdOptions::default(), session)
            .unwrap();
        TensorRead::from_tensor(&a)
            .qr_with_options_read(QrOptions::default(), session)
            .unwrap();
        TensorRead::from_tensor(&a).lu_read(session).unwrap();
        TensorRead::from_tensor(&a)
            .full_piv_lu_read(session)
            .unwrap();
        let x = TensorRead::from_tensor(&a)
            .full_piv_lu_solve_read(TensorRead::from_tensor(&b), session)
            .unwrap();
        TensorRead::from_tensor(&a)
            .solve_read(TensorRead::from_tensor(&b), session)
            .unwrap();
        TensorRead::from_tensor(&a)
            .eigh_with_options_read(EighOptions::default(), session)
            .unwrap();
        TensorRead::from_tensor(&a).eig_read(session).unwrap();
        TensorRead::from_tensor(&a).slogdet_read(session).unwrap();
        TensorRead::from_tensor(&a).det_read(session).unwrap();
        TensorRead::from_tensor(&a).inv_read(session).unwrap();
        let svdvals = TensorRead::from_tensor(&a).svdvals_read(session).unwrap();
        let eigvalsh = TensorRead::from_tensor(&a).eigvalsh_read(session).unwrap();
        let eigvals = TensorRead::from_tensor(&a).eigvals_read(session).unwrap();
        // Values-only reads must still produce the same eigenvalues as the
        // full decomposition surfaces (issue #1666). A = [[4, 1], [1, 3]] has
        // eigenvalues (7 ± √5) / 2.
        let svdvals_values = svdvals.as_slice::<f64>().unwrap();
        assert!((svdvals_values[0] - (7.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1.0e-12);
        assert!((svdvals_values[1] - (7.0 - 5.0_f64.sqrt()) / 2.0).abs() < 1.0e-12);
        let eigvalsh_values = eigvalsh.as_slice::<f64>().unwrap();
        assert!((eigvalsh_values[0] - (7.0 - 5.0_f64.sqrt()) / 2.0).abs() < 1.0e-12);
        assert!((eigvalsh_values[1] - (7.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1.0e-12);
        assert_eq!(eigvals.dtype(), tenferro_tensor::DType::C64);
        assert_eq!(eigvals.shape(), &[2]);
        TensorRead::from_tensor(&a).pinv_read(session).unwrap();
        TensorRead::from_tensor(&a)
            .pinv_with_rtol_read(1.0e-12, session)
            .unwrap();
        TensorRead::from_tensor(&a)
            .norm_read(Some(2.0), Some(&[0, 1]), false, session)
            .unwrap();

        let actual = x.as_slice::<f64>().unwrap();
        assert!((actual[0] - 20.0 / 11.0).abs() < 1.0e-12);
        assert!((actual[1] - 19.0 / 11.0).abs() < 1.0e-12);
    });
}

#[test]
fn read_surface_solve_read_into_writes_the_caller_buffer_directly() {
    // The concrete `solve_read_into` keeps the direct write path: the CPU
    // exec session solves into the caller's buffer without an allocate-then-
    // copy round trip (issue #1680 Phase 3).
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 1.0, 0.0, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![7.0_f64, 4.0]).unwrap();
    let mut host = CpuBackend::new();

    let mut owned = Tensor::from_vec_col_major([2, 1], vec![-9.0_f64; 2]).unwrap();
    host.with_backend_session(|session| {
        TensorRead::from_tensor(&a).solve_read_into(
            TensorRead::from_tensor(&b),
            tenferro_tensor::TensorWrite::from_tensor(&mut owned),
            session,
        )
    })
    .unwrap();
    let solved = owned.as_slice::<f64>().unwrap();
    assert!((solved[0] - 7.0 / 3.0).abs() < 1.0e-12);
    assert!((solved[1] - 5.0 / 6.0).abs() < 1.0e-12);

    // Strided output: only the destination entries are touched.
    let mut storage = vec![-17.0_f64; 8];
    let view =
        tenferro_tensor::TypedTensorViewMut::from_slice([2, 1], [1, 4], 1, &mut storage).unwrap();
    host.with_backend_session(|session| {
        TensorRead::from_tensor(&a).solve_read_into(
            TensorRead::from_tensor(&b),
            tenferro_tensor::TensorWrite::from_view(tenferro_tensor::TensorViewMut::F64(view)),
            session,
        )
    })
    .unwrap();
    assert_eq!(storage[0], -17.0);
    assert!((storage[1] - 7.0 / 3.0).abs() < 1.0e-12);
    assert!((storage[2] - 5.0 / 6.0).abs() < 1.0e-12);
    assert_eq!(storage[3], -17.0);
}

#[test]
fn typed_surface_covers_all_receiver_adapters() {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![9.0, 7.0]).unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        a.svd_with_options(SvdOptions::default(), session).unwrap();
        a.qr(session).unwrap();
        a.qr_with_options(QrOptions::default(), session).unwrap();
        a.lu(session).unwrap();
        a.full_piv_lu(session).unwrap();
        a.solve(&b, session).unwrap();
        a.cholesky(session).unwrap();
        a.eigh(session).unwrap();
        a.eigh_with_options(EighOptions::default(), session)
            .unwrap();
        a.det(session).unwrap();
        a.inv(session).unwrap();
        a.eigvalsh(session).unwrap();
        a.eigvals(session).unwrap();
        a.pinv(session).unwrap();
        a.pinv_with_rtol(1.0e-12, session).unwrap();
        let norm = a
            .norm(Some(f64::INFINITY), Some(&[0, 1]), false, session)
            .unwrap();

        assert_eq!(norm.as_slice().unwrap(), &[5.0]);
    });
}

#[test]
fn concrete_norm_covers_orders_axis_permutation_and_validation() {
    let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let tensor = Tensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
    )
    .unwrap();
    let mut host = CpuBackend::new();

    host.with_backend_session(|session| {
        for order in [
            Some(0.0),
            Some(1.0),
            Some(-1.0),
            Some(2.0),
            Some(-2.0),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ] {
            matrix.norm(order, Some(&[0, 1]), false, session).unwrap();
        }
        tensor.norm(None, Some(&[2, 0]), true, session).unwrap();
        tensor.norm(Some(3.0), None, false, session).unwrap();

        assert!(tensor.norm(None, Some(&[0, 0]), false, session).is_err());
        assert!(tensor.norm(Some(0.0), None, false, session).is_ok());
    });
}
