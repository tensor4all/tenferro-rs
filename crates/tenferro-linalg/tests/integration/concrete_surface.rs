use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{
    EighOptions, LinalgBackend, QrOptions, RankRevealingQrOptions, SvdOptions, TensorLinalgExt,
    TensorReadLinalgExt, TypedTensorLinalgExt,
};
use tenferro_tensor::{Tensor, TensorRead, TensorScalar, TypedTensor};

#[test]
fn cpu_execution_session_implements_linalg_backend() {
    fn assert_linalg_backend<B: LinalgBackend>() {}

    assert_linalg_backend::<tenferro_cpu::CpuEngine>();
}

#[test]
fn dynamic_and_read_surfaces_return_fixed_tuples() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        let (_u, s, _vt) = input.svd(&mut host).unwrap();
        let svdvals = input.svdvals(&mut host).unwrap();
        let (_q, r) = TensorRead::from_tensor(&input).qr_read(&mut host).unwrap();
        let (sign, logabsdet) = input.slogdet(&mut host).unwrap();

        assert_eq!(s.as_slice::<f64>().unwrap(), &[4.0, 2.0]);
        assert_eq!(svdvals.as_slice::<f64>().unwrap(), &[4.0, 2.0]);
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(sign.as_slice::<f64>().unwrap(), &[1.0]);
        assert!((logabsdet.as_slice::<f64>().unwrap()[0] - 8.0_f64.ln()).abs() < 1.0e-12);
    };
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
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        let result = input
            .rank_revealing_qr(RankRevealingQrOptions::default().rtol(1.0e-12), &mut host)
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
    };
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
    let mut host = tenferro_cpu::CpuEngine::serial();
    {
        for input in &inputs {
            let result = input
                .rank_revealing_qr(RankRevealingQrOptions::default().rtol(1.0e-6), &mut host)
                .unwrap();
            assert_eq!(result.q.dtype(), input.dtype());
            assert_eq!(result.r.dtype(), input.dtype());
            assert_eq!(result.rank.as_slice::<i64>().unwrap(), &[1]);
        }
    };
}

#[test]
fn rank_revealing_qr_handles_empty_dimensions_and_empty_batches() {
    let inputs: [(Vec<usize>, Vec<f64>); 3] = [
        (vec![0, 3], vec![]),
        (vec![3, 0], vec![]),
        (vec![2, 2, 0], vec![]),
    ];
    let mut host = tenferro_cpu::CpuEngine::serial();
    {
        for (shape, data) in inputs {
            let input = Tensor::from_vec_col_major(shape, data).unwrap();
            let result = input
                .rank_revealing_qr(RankRevealingQrOptions::default(), &mut host)
                .unwrap();
            let ranks = result.rank.as_slice::<i64>().unwrap();
            if result.rank.shape().is_empty() {
                assert_eq!(ranks, &[0]);
            } else {
                assert!(ranks.is_empty());
            }
        }
    };
}

#[test]
fn rank_revealing_qr_rejects_non_finite_input() {
    let input = Tensor::from_vec_col_major(vec![2, 1], vec![f64::NAN, 1.0]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();
    {
        assert!(input
            .rank_revealing_qr(RankRevealingQrOptions::default(), &mut host)
            .is_err());
    };
}

#[test]
fn rank_revealing_qr_rejects_invalid_tolerances() {
    let input = Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f64]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();
    {
        for options in [
            RankRevealingQrOptions::default().rtol(f64::NAN),
            RankRevealingQrOptions::default().rtol(-1.0),
            RankRevealingQrOptions::default().atol(f64::INFINITY),
        ] {
            assert!(input.rank_revealing_qr(options, &mut host).is_err());
        }
    };
}

#[test]
fn rank_revealing_qr_zero_and_batched_metadata() {
    let input = Tensor::from_vec_col_major(vec![2, 2, 2], vec![0.0_f64; 8]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();
    {
        let result = input
            .rank_revealing_qr(RankRevealingQrOptions::default(), &mut host)
            .unwrap();
        assert_eq!(result.q.shape(), &[2, 2, 2]);
        assert_eq!(result.r.shape(), &[2, 2, 2]);
        assert_eq!(result.column_permutation.shape(), &[2, 2]);
        assert_eq!(result.rank.shape(), &[2]);
        assert_eq!(result.rank.as_slice::<i64>().unwrap(), &[0, 0]);
    };
}

/// The compile-time CPU provider this build selected.
fn rrqr_provider_backends() -> Vec<CpuEngine> {
    vec![tenferro_cpu::CpuEngine::serial()]
}

/// An all-zero item between nonzero items gets the canonical zero-rank factors from one batched
/// provider call, on every route, while the nonzero items keep their own factors and ranks.
#[test]
fn rank_revealing_qr_mixes_zero_and_nonzero_items_in_one_batch() {
    // Items: diag(3, 2) (rank 2), zeros (rank 0), [[1, 1], [1, 1]] (rank 1).
    let data = vec![
        3.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let input = Tensor::from_vec_col_major(vec![2, 2, 3], data).unwrap();
    for mut host in rrqr_provider_backends() {
        {
            let result = input
                .rank_revealing_qr(RankRevealingQrOptions::default().rtol(1.0e-12), &mut host)
                .unwrap();
            assert_eq!(result.rank.as_slice::<i64>().unwrap(), &[2, 0, 1]);
            let q = result.q.as_slice::<f64>().unwrap();
            let r = result.r.as_slice::<f64>().unwrap();
            let p = result.column_permutation.as_slice::<i64>().unwrap();
            assert_eq!(
                &q[4..8],
                &[1.0, 0.0, 0.0, 1.0],
                "zero item: identity Q columns"
            );
            assert_eq!(&r[4..8], &[0.0; 4], "zero item: zero R");
            assert_eq!(&p[2..4], &[0, 1], "zero item: identity permutation");
        };
    }
}

/// A non-finite item is reported before any item is factored, on every route.
#[test]
fn rank_revealing_qr_reports_non_finite_input_first_on_every_route() {
    let data = vec![1.0_f64, 0.0, 0.0, 1.0, f64::INFINITY, 0.0, 0.0, 1.0];
    let input = Tensor::from_vec_col_major(vec![2, 2, 2], data).unwrap();
    for mut host in rrqr_provider_backends() {
        {
            let error = input
                .rank_revealing_qr(RankRevealingQrOptions::default(), &mut host)
                .unwrap_err();
            assert!(error.to_string().contains("non-finite input"), "{error}");
        };
    }
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
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        let (_u, singular_values, _vt): (TypedTensor<f64>, TypedTensor<f64>, TypedTensor<f64>) =
            real.svd(&mut host).unwrap();
        let (eigenvalues, _vectors): (TypedTensor<Complex64>, TypedTensor<Complex64>) =
            real.eig(&mut host).unwrap();
        let (_cu, complex_singular_values, _cvt): (
            TypedTensor<Complex64>,
            TypedTensor<f64>,
            TypedTensor<Complex64>,
        ) = complex.svd(&mut host).unwrap();
        let real_values: TypedTensor<f64> = real.svdvals(&mut host).unwrap();
        let complex_values: TypedTensor<f64> = complex.svdvals(&mut host).unwrap();

        assert_eq!(singular_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(complex_singular_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(real_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(complex_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(eigenvalues.shape(), &[2]);
    };
}

#[test]
fn typed_input_is_erased_as_a_borrowed_read() {
    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
    let read = f64::tensor_read(&input);
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        let factor = read.cholesky_read(&mut host).unwrap();
        assert_eq!(factor.shape(), &[2, 2]);
    };
}

#[test]
fn dynamic_composites_cover_inverse_pseudoinverse_eigenvalues_and_norm() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        let det = input.det(&mut host).unwrap();
        let inv = input.inv(&mut host).unwrap();
        let pinv = input.pinv(&mut host).unwrap();
        let eigvalsh = input.eigvalsh(&mut host).unwrap();
        let eigvals = input.eigvals(&mut host).unwrap();
        let norm = input.norm(None, Some(&[0, 1]), true, &mut host).unwrap();

        assert!((det.as_slice::<f64>().unwrap()[0] - 8.0).abs() < 1.0e-12);
        assert_eq!(inv.as_slice::<f64>().unwrap(), &[0.5, 0.0, 0.0, 0.25]);
        assert_eq!(pinv.as_slice::<f64>().unwrap(), &[0.5, 0.0, 0.0, 0.25]);
        assert_eq!(eigvalsh.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
        assert_eq!(eigvals.dtype(), tenferro_tensor::DType::C64);
        assert_eq!(norm.shape(), &[1, 1]);
        assert!((norm.as_slice::<f64>().unwrap()[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
    };
}

#[test]
fn read_surface_accepts_a_strided_view_without_an_input_clone() {
    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .unwrap();
    let transposed = input.as_view().transpose_view([1, 0]).unwrap();
    let values_view = input.as_view().transpose_view([1, 0]).unwrap();
    let read = TensorRead::from_view(f64::tensor_view(transposed));
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        let (_u, singular_values, _vt) = read.svd_read(&mut host).unwrap();
        let values = TensorRead::from_view(f64::tensor_view(values_view))
            .svdvals_read(&mut host)
            .unwrap();
        assert_eq!(singular_values.as_slice::<f64>().unwrap(), &[2.0, 1.0]);
        assert_eq!(values.as_slice::<f64>().unwrap(), &[2.0, 1.0]);
    };
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
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        let (_sign, logabsdet): (TypedTensor<Complex64>, TypedTensor<f64>) =
            input.slogdet(&mut host).unwrap();
        let norm: TypedTensor<f64> = input.norm(None, Some(&[0, 1]), false, &mut host).unwrap();

        assert!((logabsdet.as_slice().unwrap()[0] - 8.0_f64.ln()).abs() < 1.0e-12);
        assert!((norm.as_slice().unwrap()[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
    };
}

#[test]
fn typed_solve_surfaces_accept_vector_and_matrix_rhs() {
    let a =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let vector = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 8.0]).unwrap();
    let matrix = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 8.0]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        let vector_x = a.full_piv_lu_solve(&vector, &mut host).unwrap();
        let matrix_x = a.full_piv_lu_solve(&matrix, &mut host).unwrap();
        let triangular_x = a
            .triangular_solve(&matrix, true, false, false, false, &mut host)
            .unwrap();

        assert_eq!(vector_x.as_slice().unwrap(), &[2.0, 2.0]);
        assert_eq!(matrix_x.as_slice().unwrap(), &[2.0, 2.0]);
        assert_eq!(triangular_x.as_slice().unwrap(), &[2.0, 2.0]);
    };
}

#[test]
fn concrete_norm_distinguishes_empty_axes_and_rejects_invalid_axes() {
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let empty = Tensor::from_vec_col_major([0, 2], Vec::<f64>::new()).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();
    assert_eq!(host.num_threads(), 1);

    {
        let identity = input.norm(Some(2.0), Some(&[]), false, &mut host).unwrap();
        let error = input
            .norm(Some(2.0), Some(&[1]), false, &mut host)
            .unwrap_err();
        let duplicate_nonempty = input
            .norm(Some(2.0), Some(&[0, 0]), false, &mut host)
            .unwrap_err();
        let duplicate_empty = empty
            .norm(None, Some(&[0, 0]), false, &mut host)
            .unwrap_err();

        assert_eq!(identity.as_slice::<f64>().unwrap(), &[3.0, 4.0]);
        assert!(error.to_string().contains("axis 1"));
        for error in [duplicate_nonempty, duplicate_empty] {
            assert!(matches!(
                error,
                tenferro_tensor::Error::Validation {
                    op: "norm",
                    source: tenferro_tensor::ValidationError::DuplicateAxis {
                        axis: 0,
                        role: "dim"
                    }
                }
            ));
        }
    };
}

#[test]
fn read_surface_covers_factorizations_and_composites() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![9.0_f64, 7.0]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        TensorRead::from_tensor(&a)
            .svd_with_options_read(SvdOptions::default(), &mut host)
            .unwrap();
        TensorRead::from_tensor(&a)
            .qr_with_options_read(QrOptions::default(), &mut host)
            .unwrap();
        TensorRead::from_tensor(&a).lu_read(&mut host).unwrap();
        TensorRead::from_tensor(&a)
            .full_piv_lu_read(&mut host)
            .unwrap();
        let x = TensorRead::from_tensor(&a)
            .full_piv_lu_solve_read(TensorRead::from_tensor(&b), &mut host)
            .unwrap();
        TensorRead::from_tensor(&a)
            .solve_read(TensorRead::from_tensor(&b), &mut host)
            .unwrap();
        TensorRead::from_tensor(&a)
            .eigh_with_options_read(EighOptions::default(), &mut host)
            .unwrap();
        TensorRead::from_tensor(&a).eig_read(&mut host).unwrap();
        TensorRead::from_tensor(&a).slogdet_read(&mut host).unwrap();
        TensorRead::from_tensor(&a).det_read(&mut host).unwrap();
        TensorRead::from_tensor(&a).inv_read(&mut host).unwrap();
        let svdvals = TensorRead::from_tensor(&a).svdvals_read(&mut host).unwrap();
        let eigvalsh = TensorRead::from_tensor(&a)
            .eigvalsh_read(&mut host)
            .unwrap();
        let eigvals = TensorRead::from_tensor(&a).eigvals_read(&mut host).unwrap();
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
        TensorRead::from_tensor(&a).pinv_read(&mut host).unwrap();
        TensorRead::from_tensor(&a)
            .pinv_with_rtol_read(1.0e-12, &mut host)
            .unwrap();
        TensorRead::from_tensor(&a)
            .norm_read(Some(2.0), Some(&[0, 1]), false, &mut host)
            .unwrap();

        let actual = x.as_slice::<f64>().unwrap();
        assert!((actual[0] - 20.0 / 11.0).abs() < 1.0e-12);
        assert!((actual[1] - 19.0 / 11.0).abs() < 1.0e-12);
    };
}

#[test]
fn read_surface_solve_read_into_writes_the_caller_buffer_directly() {
    // The concrete `solve_read_into` keeps the direct write path: the CPU
    // exec session solves into the caller's buffer without an allocate-then-
    // copy round trip (issue #1680 Phase 3).
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 1.0, 0.0, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 1], vec![7.0_f64, 4.0]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();

    let mut owned = Tensor::from_vec_col_major([2, 1], vec![-9.0_f64; 2]).unwrap();
    {
        TensorRead::from_tensor(&a).solve_read_into(
            TensorRead::from_tensor(&b),
            tenferro_tensor::TensorWrite::from_tensor(&mut owned),
            &mut host,
        )
    }
    .unwrap();
    let solved = owned.as_slice::<f64>().unwrap();
    assert!((solved[0] - 7.0 / 3.0).abs() < 1.0e-12);
    assert!((solved[1] - 5.0 / 6.0).abs() < 1.0e-12);

    // Strided output: only the destination entries are touched.
    let mut storage = vec![-17.0_f64; 8];
    let view =
        tenferro_tensor::TypedTensorViewMut::from_slice([2, 1], [1, 4], 1, &mut storage).unwrap();
    {
        TensorRead::from_tensor(&a).solve_read_into(
            TensorRead::from_tensor(&b),
            tenferro_tensor::TensorWrite::from_view(tenferro_tensor::TensorViewMut::F64(view)),
            &mut host,
        )
    }
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
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        a.svd_with_options(SvdOptions::default(), &mut host)
            .unwrap();
        a.qr(&mut host).unwrap();
        a.qr_with_options(QrOptions::default(), &mut host).unwrap();
        a.lu(&mut host).unwrap();
        a.full_piv_lu(&mut host).unwrap();
        a.solve(&b, &mut host).unwrap();
        a.cholesky(&mut host).unwrap();
        a.eigh(&mut host).unwrap();
        a.eigh_with_options(EighOptions::default(), &mut host)
            .unwrap();
        a.det(&mut host).unwrap();
        a.inv(&mut host).unwrap();
        a.eigvalsh(&mut host).unwrap();
        a.eigvals(&mut host).unwrap();
        a.pinv(&mut host).unwrap();
        a.pinv_with_rtol(1.0e-12, &mut host).unwrap();
        let norm = a
            .norm(Some(f64::INFINITY), Some(&[0, 1]), false, &mut host)
            .unwrap();

        assert_eq!(norm.as_slice().unwrap(), &[5.0]);
    };
}

#[test]
fn concrete_spectral_norm_preserves_signed_and_complex_input_across_surfaces() {
    let signed = Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 1.0, 1.0, -1.0]).unwrap();
    let complex = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, -1.0),
            ],
        )
        .unwrap(),
    );
    let typed_signed =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0_f64, 1.0, 1.0, -1.0]).unwrap();
    let typed_complex = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, -1.0),
        ],
    )
    .unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    assert_eq!(backend.num_threads(), 1);

    {
        for order in [Some(2.0), Some(-2.0)] {
            for value in [
                signed
                    .norm(order, Some(&[0, 1]), false, &mut backend)
                    .unwrap(),
                TensorRead::from_tensor(&signed)
                    .norm_read(order, Some(&[0, 1]), false, &mut backend)
                    .unwrap(),
                complex
                    .norm(order, Some(&[0, 1]), false, &mut backend)
                    .unwrap(),
                TensorRead::from_tensor(&complex)
                    .norm_read(order, Some(&[0, 1]), false, &mut backend)
                    .unwrap(),
                TensorRead::from_view(tenferro_tensor::TensorView::F64(
                    typed_signed.as_view().transpose_view([1, 0]).unwrap(),
                ))
                .norm_read(order, Some(&[1, 0]), true, &mut backend)
                .unwrap(),
                TensorRead::from_view(tenferro_tensor::TensorView::C64(
                    typed_complex.as_view().transpose_view([1, 0]).unwrap(),
                ))
                .norm_read(order, Some(&[0, 1]), false, &mut backend)
                .unwrap(),
            ] {
                assert!((value.as_slice::<f64>().unwrap()[0] - 2.0_f64.sqrt()).abs() < 1.0e-12);
            }

            let signed_value = typed_signed
                .norm(order, Some(&[0, 1]), false, &mut backend)
                .unwrap();
            let complex_value = typed_complex
                .norm(order, Some(&[0, 1]), false, &mut backend)
                .unwrap();
            assert!((signed_value.as_slice().unwrap()[0] - 2.0_f64.sqrt()).abs() < 1.0e-12);
            assert!((complex_value.as_slice().unwrap()[0] - 2.0_f64.sqrt()).abs() < 1.0e-12);
        }
    };
}

#[test]
fn concrete_norm_covers_orders_axis_permutation_and_validation() {
    let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let tensor = Tensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
    )
    .unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();

    {
        for order in [
            Some(0.0),
            Some(1.0),
            Some(-1.0),
            Some(2.0),
            Some(-2.0),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ] {
            matrix.norm(order, Some(&[0, 1]), false, &mut host).unwrap();
        }
        tensor.norm(None, Some(&[2, 0]), true, &mut host).unwrap();
        tensor.norm(Some(3.0), None, false, &mut host).unwrap();

        assert!(tensor.norm(None, Some(&[0, 0]), false, &mut host).is_err());
        assert!(tensor.norm(Some(0.0), None, false, &mut host).is_ok());
    };
}

#[test]
fn complex_pinv_matches_the_moore_penrose_inverse() {
    // #2049: for a complex input the singular values are real, so the inverted
    // values that scale `v` are real too, while `v` is complex; the multiply
    // requires one dtype.
    let input = Tensor::from_typed::<Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(4.0, 1.0),
            ],
        )
        .unwrap(),
    );
    let mut host = tenferro_cpu::CpuEngine::serial();
    {
        let pinv = input.pinv(&mut host).unwrap();
        assert_eq!(pinv.dtype(), tenferro_tensor::DType::C64);
        let values = pinv.as_slice::<Complex64>().unwrap().to_vec();
        // diag(2, 4+i) has the pseudoinverse diag(1/2, 1/(4+i)) = diag(1/2, (4-i)/17).
        let expected = [
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(4.0 / 17.0, -1.0 / 17.0),
        ];
        for (index, (actual, expected)) in values.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).norm() < 1.0e-12,
                "pinv[{index}]: {actual:?} against {expected:?}"
            );
        }

        let with_rtol = input.pinv_with_rtol(1.0e-12, &mut host).unwrap();
        for (index, (actual, expected)) in with_rtol
            .as_slice::<Complex64>()
            .unwrap()
            .iter()
            .zip(values.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).norm() < 1.0e-12,
                "pinv_with_rtol[{index}]: {actual:?} against {expected:?}"
            );
        }
    }
}

/// Assert the public `eig` column gauge on a square `[n, n]` eigenvector
/// matrix: every column has unit 2-norm.
fn assert_unit_norm_eigenvector_columns(vectors: &Tensor) {
    let [n, columns_count] = <[usize; 2]>::try_from(vectors.shape()).unwrap();
    assert_eq!(n, columns_count);
    let columns = vectors.as_slice::<Complex64>().unwrap();
    for column in 0..n {
        let norm = (0..n)
            .map(|row| columns[column * n + row].norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1.0e-12,
            "eigenvector column {column} must have unit 2-norm, got {norm}"
        );
    }
}

#[test]
fn eig_enforces_unit_norm_eigenvector_columns_and_refuses_non_finite_ones() {
    // #2148: the vendored oracle records unit 2-norm eigenvector columns for
    // tenferro and PyTorch, so the seam is where that becomes true for every
    // provider, and a column a provider left non-finite is a typed numerical
    // failure rather than NaN.
    let diagonalizable =
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 1.0, 0.0, 2.0]).unwrap();
    let mut host = tenferro_cpu::CpuEngine::serial();
    {
        let (values, vectors) = diagonalizable.eig(&mut host).unwrap();
        assert_eq!(values.shape(), &[2]);
        assert_eq!(vectors.shape(), &[2, 2]);
        assert_unit_norm_eigenvector_columns(&vectors);
    };

    // A finite defective (Jordan) input: unit-norm columns or a typed
    // numerical failure - never NaN. Checking finiteness alone would miss a
    // provider that returned a finite but non-unit column, so assert the same
    // norm contract on the `Ok` branch here too.
    let defective = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 1.0, 0.0, 1.0]).unwrap();
    match defective.eig(&mut host) {
        Ok((_, vectors)) => assert_unit_norm_eigenvector_columns(&vectors),
        Err(error) => assert_eq!(error.kind(), tenferro_tensor::ErrorKind::NumericalFailure),
    };
}
