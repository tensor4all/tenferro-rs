use num_complex::{Complex32, Complex64};
use tenferro_ops::ext_op::invoke_extension_shape_inference;
use tenferro_runtime::extension::ExtensionOp;
use tenferro_tensor::{DType, Error, Tensor, TypedTensor};

use super::{
    apply_eigh_gauge, apply_qr_gauge, apply_svd_gauge, canonical_svd_gauge_layout, promote_dtypes,
    EighGauge, LinalgExtensionOp, LinalgOp, QrGauge, SvdGauge, LINALG_EXTENSION_FAMILY_ID,
};

#[test]
fn session_support_admits_every_cpu_linear_algebra_op() {
    use tenferro_cpu::CpuBackend;

    // The CPU exec session runs every linalg kernel; admission must never be
    // narrowed below the session executor's real support (issue #1665), or
    // `apply_eager` would fall back to the compiled-program path for CPU
    // solve/svd/eigh.
    let ops = [
        LinalgOp::Cholesky,
        LinalgOp::Lu,
        LinalgOp::LuFactor,
        LinalgOp::LuSolvePrepared {
            transpose_a: false,
            conjugate_a: false,
        },
        LinalgOp::LuSolvePrepared {
            transpose_a: false,
            conjugate_a: true,
        },
        LinalgOp::LuSolvePrepared {
            transpose_a: true,
            conjugate_a: false,
        },
        LinalgOp::LuSolvePrepared {
            transpose_a: true,
            conjugate_a: true,
        },
        LinalgOp::SignDetFromLuFactor,
        LinalgOp::LogAbsDetFromLuFactor,
        LinalgOp::FullPivLu,
        LinalgOp::FullPivLuSolve { transpose_a: false },
        LinalgOp::FullPivLuSolve { transpose_a: true },
        LinalgOp::Svd {
            derivative_eps: 0.0,
            gauge: SvdGauge::Raw,
        },
        LinalgOp::SvdVals {
            derivative_eps: 0.0,
        },
        LinalgOp::Qr {
            gauge: QrGauge::Raw,
        },
        LinalgOp::Eigh {
            derivative_eps: 0.0,
            gauge: EighGauge::Raw,
        },
        LinalgOp::EighVals {
            derivative_eps: 0.0,
        },
        LinalgOp::Eig {
            input_dtype: DType::F64,
        },
        LinalgOp::EigVals {
            input_dtype: DType::F64,
        },
        LinalgOp::TriangularSolve {
            left_side: false,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        },
    ];
    for op in ops {
        let admitted = super::linalg_session_supported::<CpuBackend>(&LinalgExtensionOp::new(op));
        assert!(
            admitted,
            "the CPU linalg session executor runs every op; admission must not narrow: {op:?}"
        );
    }

    // SvdFull is conservatively rejected on CPU: the backend type does not
    // carry its provider kind (faer vs BLAS), and BLAS has no in-session
    // full-matrices SVD, so admission must not over-claim.
    let svd_full = LinalgExtensionOp::new(LinalgOp::SvdFull);
    assert!(
        !super::linalg_session_supported::<CpuBackend>(&svd_full),
        "SvdFull must not be admitted on CPU (BLAS has no in-session full SVD)"
    );
}

#[test]
fn infer_output_meta_returns_error_on_input_count_mismatch() {
    let op = LinalgExtensionOp::new(LinalgOp::Cholesky);

    let err = invoke_extension_shape_inference(&op, &[], &[]).unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            op: "extension",
            source,
        } if source.to_string().contains(LINALG_EXTENSION_FAMILY_ID)
    ));
}

#[test]
fn extension_dtype_promotion_delegates_to_canonical_tensor_rules() {
    let source = include_str!("../extension.rs");
    assert!(
        !source.contains("fn promote_dtype("),
        "linalg extension metadata must not duplicate the canonical dtype promotion lattice"
    );

    let dtypes = [
        DType::Bool,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
        DType::C32,
        DType::C64,
    ];
    for lhs in dtypes {
        for rhs in dtypes {
            assert_eq!(
                promote_dtypes(&[lhs, rhs]),
                tenferro_tensor::validate::promote_dtype(lhs, rhs),
                "promotion mismatch for {lhs:?}, {rhs:?}"
            );
        }
    }
}

#[test]
fn decomposition_value_outputs_prune_to_values_only_ops() {
    let svd = LinalgExtensionOp::new(LinalgOp::Svd {
        derivative_eps: 1.0e-12,
        gauge: SvdGauge::CanonicalPivot,
    });
    let pruned_svd = svd
        .prune_outputs(&[false, true, false])
        .expect("S-only SVD should prune to SvdVals");
    let pruned_svd = pruned_svd
        .as_any()
        .downcast_ref::<LinalgExtensionOp>()
        .expect("pruned SVD op should stay in linalg family");
    assert_eq!(
        pruned_svd.op(),
        LinalgOp::SvdVals {
            derivative_eps: 1.0e-12
        }
    );

    let eigh = LinalgExtensionOp::new(LinalgOp::Eigh {
        derivative_eps: 1.0e-12,
        gauge: EighGauge::Raw,
    });
    let pruned_eigh = eigh
        .prune_outputs(&[true, false])
        .expect("eigenvalue-only Hermitian eigendecomposition should prune to EighVals");
    let pruned_eigh = pruned_eigh
        .as_any()
        .downcast_ref::<LinalgExtensionOp>()
        .expect("pruned Eigh op should stay in linalg family");
    assert_eq!(
        pruned_eigh.op(),
        LinalgOp::EighVals {
            derivative_eps: 1.0e-12
        }
    );

    let eig = LinalgExtensionOp::new(LinalgOp::Eig {
        input_dtype: DType::F64,
    });
    let pruned_eig = eig
        .prune_outputs(&[true, false])
        .expect("eigenvalue-only general eigendecomposition should prune to EigVals");
    let pruned_eig = pruned_eig
        .as_any()
        .downcast_ref::<LinalgExtensionOp>()
        .expect("pruned Eig op should stay in linalg family");
    assert_eq!(
        pruned_eig.op(),
        LinalgOp::EigVals {
            input_dtype: DType::F64
        }
    );
}

#[test]
fn canonical_pivot_svd_gauge_flips_real_vectors_and_vt_rows_together() {
    let mut outputs = vec![
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, -2.0, -3.0, 0.5]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 1.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap(),
    ];

    apply_svd_gauge(SvdGauge::CanonicalPivot, &mut outputs).unwrap();

    assert_eq!(
        outputs[0].as_slice::<f64>().unwrap(),
        &[-1.0, 2.0, 3.0, -0.5]
    );
    assert_eq!(
        outputs[2].as_slice::<f64>().unwrap(),
        &[-10.0, -20.0, -30.0, -40.0]
    );
}

#[test]
fn canonical_pivot_svd_gauge_removes_complex_pivot_phase() {
    let mut outputs = vec![
        Tensor::C64(
            TypedTensor::from_vec_col_major(
                vec![2, 1],
                vec![Complex64::new(1.0, 1.0), Complex64::new(0.1, 0.0)],
            )
            .unwrap(),
        ),
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        Tensor::C64(
            TypedTensor::from_vec_col_major(
                vec![1, 2],
                vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 4.0)],
            )
            .unwrap(),
        ),
    ];

    apply_svd_gauge(SvdGauge::CanonicalPivot, &mut outputs).unwrap();

    let scale = 2.0_f64.sqrt();
    let u = outputs[0].as_slice::<Complex64>().unwrap();
    assert!((u[0].re - scale).abs() < 1.0e-12);
    assert!(u[0].im.abs() < 1.0e-12);
    assert!((u[1].re - 0.1 / scale).abs() < 1.0e-12);
    assert!((u[1].im + 0.1 / scale).abs() < 1.0e-12);

    let vt = outputs[2].as_slice::<Complex64>().unwrap();
    assert!((vt[0].re - scale).abs() < 1.0e-12);
    assert!((vt[0].im - scale).abs() < 1.0e-12);
    assert!((vt[1].re + 1.0 / scale).abs() < 1.0e-12);
    assert!((vt[1].im - 7.0 / scale).abs() < 1.0e-12);
}

#[test]
fn canonical_svd_gauge_rejects_batch_product_overflow() {
    let error = canonical_svd_gauge_layout(1, 1, 1, &[usize::MAX, 2])
        .expect_err("overflowing SVD batch shape should be rejected");

    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-linalg.svd",
            ..
        }
    ));
    assert!(error.to_string().contains("canonical SVD batch"));
}

#[test]
fn canonical_svd_gauge_rejects_u_batch_span_overflow() {
    let error = canonical_svd_gauge_layout(usize::MAX, 2, 1, &[])
        .expect_err("overflowing U batch span should be rejected");

    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-linalg.svd",
            ..
        }
    ));
    assert!(error.to_string().contains("canonical SVD U batch"));
}

#[test]
fn canonical_svd_gauge_rejects_vt_batch_span_overflow() {
    let error = canonical_svd_gauge_layout(1, usize::MAX, 2, &[])
        .expect_err("overflowing VT batch span should be rejected");

    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-linalg.svd",
            ..
        }
    ));
    assert!(error.to_string().contains("canonical SVD VT batch"));
}

#[test]
fn canonical_svd_gauge_rejects_u_storage_span_overflow() {
    let error = canonical_svd_gauge_layout(usize::MAX, 1, 1, &[2])
        .expect_err("overflowing U storage span should be rejected");

    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-linalg.svd",
            ..
        }
    ));
    assert!(error.to_string().contains("canonical SVD U storage"));
}

#[test]
fn canonical_svd_gauge_rejects_vt_storage_span_overflow() {
    let error = canonical_svd_gauge_layout(1, 1, usize::MAX, &[2])
        .expect_err("overflowing VT storage span should be rejected");

    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-linalg.svd",
            ..
        }
    ));
    assert!(error.to_string().contains("canonical SVD VT storage"));
}

#[test]
fn canonical_svd_gauge_rejects_short_u_storage() {
    let layout = canonical_svd_gauge_layout(1, 1, 1, &[2]).unwrap();
    let error = layout
        .validate_storage(1, 2)
        .expect_err("short U storage should be rejected before gauge indexing");

    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-linalg.svd",
            ..
        }
    ));
    assert!(error
        .to_string()
        .contains("expected U storage length 2, got 1"));
}

#[test]
fn canonical_pivot_svd_gauge_handles_batched_f32_and_c32_outputs() {
    let mut real_outputs = vec![
        Tensor::from_vec_col_major(vec![2, 1, 2], vec![-2.0_f32, 1.0, 0.5, -3.0]).unwrap(),
        Tensor::from_vec_col_major(vec![1, 2], vec![2.0_f32, 3.0]).unwrap(),
        Tensor::from_vec_col_major(vec![1, 2, 2], vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap(),
    ];

    apply_svd_gauge(SvdGauge::CanonicalPivot, &mut real_outputs).unwrap();

    assert_eq!(
        real_outputs[0].as_slice::<f32>().unwrap(),
        &[2.0, -1.0, -0.5, 3.0]
    );
    assert_eq!(
        real_outputs[2].as_slice::<f32>().unwrap(),
        &[-10.0, -20.0, -30.0, -40.0]
    );

    let mut complex_outputs = vec![
        Tensor::C32(
            TypedTensor::from_vec_col_major(
                vec![1, 1, 2],
                vec![Complex32::new(1.0, 1.0), Complex32::new(0.0, -2.0)],
            )
            .unwrap(),
        ),
        Tensor::from_vec_col_major(vec![1, 2], vec![2.0_f32, 3.0]).unwrap(),
        Tensor::C32(
            TypedTensor::from_vec_col_major(
                vec![1, 1, 2],
                vec![Complex32::new(2.0, 0.0), Complex32::new(3.0, 0.0)],
            )
            .unwrap(),
        ),
    ];

    apply_svd_gauge(SvdGauge::CanonicalPivot, &mut complex_outputs).unwrap();

    let scale = 2.0_f32.sqrt();
    let u = complex_outputs[0].as_slice::<Complex32>().unwrap();
    assert!((u[0].re - scale).abs() < 1.0e-6);
    assert!(u[0].im.abs() < 1.0e-6);
    assert!((u[1].re - 2.0).abs() < 1.0e-6);
    assert!(u[1].im.abs() < 1.0e-6);
    let vt = complex_outputs[2].as_slice::<Complex32>().unwrap();
    assert!((vt[0].re - scale).abs() < 1.0e-6);
    assert!((vt[0].im - scale).abs() < 1.0e-6);
    assert!(vt[1].re.abs() < 1.0e-6);
    assert!((vt[1].im + 3.0).abs() < 1.0e-6);
}

#[test]
fn canonical_pivot_svd_gauge_accepts_zero_batch() {
    let mut outputs = vec![
        Tensor::from_vec_col_major(vec![2, 1, 0], Vec::<f64>::new()).unwrap(),
        Tensor::from_vec_col_major(vec![1, 0], Vec::<f64>::new()).unwrap(),
        Tensor::from_vec_col_major(vec![1, 2, 0], Vec::<f64>::new()).unwrap(),
    ];

    apply_svd_gauge(SvdGauge::CanonicalPivot, &mut outputs).unwrap();

    assert!(outputs[0].as_slice::<f64>().unwrap().is_empty());
    assert!(outputs[2].as_slice::<f64>().unwrap().is_empty());
}

#[test]
fn canonical_pivot_eigh_gauge_flips_real_eigenvector_columns() {
    let mut outputs = vec![
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![-0.9_f64, 0.2, -0.8, 0.1]).unwrap(),
    ];
    let before_reconstruction = reconstruct_eigh_f64(
        outputs[0].as_slice::<f64>().unwrap(),
        outputs[1].as_slice::<f64>().unwrap(),
        2,
    );

    apply_eigh_gauge(EighGauge::CanonicalPivot, &mut outputs).unwrap();

    assert_eq!(
        outputs[1].as_slice::<f64>().unwrap(),
        &[0.9, -0.2, 0.8, -0.1]
    );
    let after_reconstruction = reconstruct_eigh_f64(
        outputs[0].as_slice::<f64>().unwrap(),
        outputs[1].as_slice::<f64>().unwrap(),
        2,
    );
    assert_slice_close_f64(&after_reconstruction, &before_reconstruction, 1.0e-12);
}

#[test]
fn canonical_pivot_eigh_gauge_removes_complex_eigenvector_phase() {
    let mut outputs = vec![
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Tensor::C64(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex64::new(1.0, 1.0),
                    Complex64::new(0.1, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .unwrap(),
        ),
    ];
    let before_reconstruction = reconstruct_eigh_c64(
        outputs[0].as_slice::<f64>().unwrap(),
        outputs[1].as_slice::<Complex64>().unwrap(),
        2,
    );

    apply_eigh_gauge(EighGauge::CanonicalPivot, &mut outputs).unwrap();

    let scale = 2.0_f64.sqrt();
    let vectors = outputs[1].as_slice::<Complex64>().unwrap();
    assert!((vectors[0].re - scale).abs() < 1.0e-12);
    assert!(vectors[0].im.abs() < 1.0e-12);
    assert!((vectors[1].re - 0.1 / scale).abs() < 1.0e-12);
    assert!((vectors[1].im + 0.1 / scale).abs() < 1.0e-12);
    let after_reconstruction = reconstruct_eigh_c64(
        outputs[0].as_slice::<f64>().unwrap(),
        outputs[1].as_slice::<Complex64>().unwrap(),
        2,
    );
    assert_slice_close_c64(&after_reconstruction, &before_reconstruction, 1.0e-12);
}

#[test]
fn eigh_gauge_covers_f32_and_c32_paths() {
    let mut real_outputs = vec![
        Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 3.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![-0.7_f32, 0.2, 0.5, -0.8]).unwrap(),
    ];
    let before_real = reconstruct_eigh_f64(
        &real_outputs[0]
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
        &real_outputs[1]
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
        2,
    );

    apply_eigh_gauge(EighGauge::CanonicalPivot, &mut real_outputs).unwrap();

    assert_eq!(
        real_outputs[1].as_slice::<f32>().unwrap(),
        &[0.7, -0.2, -0.5, 0.8]
    );
    let after_real = reconstruct_eigh_f64(
        &real_outputs[0]
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
        &real_outputs[1]
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
        2,
    );
    assert_slice_close_f64(&after_real, &before_real, 1.0e-6);

    let mut complex_outputs = vec![
        Tensor::from_vec_col_major(vec![2], vec![2.0_f32, 3.0]).unwrap(),
        Tensor::C32(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex32::new(1.0, 1.0),
                    Complex32::new(0.1, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                ],
            )
            .unwrap(),
        ),
    ];

    apply_eigh_gauge(EighGauge::CanonicalPivot, &mut complex_outputs).unwrap();

    let scale = 2.0_f32.sqrt();
    let vectors = complex_outputs[1].as_slice::<Complex32>().unwrap();
    assert!((vectors[0].re - scale).abs() < 1.0e-6);
    assert!(vectors[0].im.abs() < 1.0e-6);
    assert!((vectors[1].re - 0.1 / scale).abs() < 1.0e-6);
    assert!((vectors[1].im + 0.1 / scale).abs() < 1.0e-6);
    assert_eq!(vectors[2], Complex32::new(0.0, 0.0));
    assert_eq!(vectors[3], Complex32::new(0.0, 0.0));
}

#[test]
fn positive_diagonal_qr_gauge_flips_real_q_columns_and_r_rows() {
    let mut outputs = vec![
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![-5.0_f64, 0.0, 6.0, -7.0]).unwrap(),
    ];
    let before_product = matmul_f64(
        outputs[0].as_slice::<f64>().unwrap(),
        outputs[1].as_slice::<f64>().unwrap(),
        2,
        2,
        2,
    );

    apply_qr_gauge(QrGauge::PositiveDiagonal, &mut outputs).unwrap();

    assert_eq!(
        outputs[0].as_slice::<f64>().unwrap(),
        &[-1.0, -2.0, -3.0, -4.0]
    );
    assert_eq!(
        outputs[1].as_slice::<f64>().unwrap(),
        &[5.0, -0.0, -6.0, 7.0]
    );
    let after_product = matmul_f64(
        outputs[0].as_slice::<f64>().unwrap(),
        outputs[1].as_slice::<f64>().unwrap(),
        2,
        2,
        2,
    );
    assert_slice_close_f64(&after_product, &before_product, 1.0e-12);
}

#[test]
fn positive_diagonal_qr_gauge_removes_complex_diagonal_phase() {
    let mut outputs = vec![
        Tensor::C64(
            TypedTensor::from_vec_col_major(
                vec![2, 1],
                vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
            )
            .unwrap(),
        ),
        Tensor::C64(
            TypedTensor::from_vec_col_major(
                vec![1, 2],
                vec![Complex64::new(1.0, 1.0), Complex64::new(3.0, 4.0)],
            )
            .unwrap(),
        ),
    ];
    let before_product = matmul_c64(
        outputs[0].as_slice::<Complex64>().unwrap(),
        outputs[1].as_slice::<Complex64>().unwrap(),
        2,
        1,
        2,
    );

    apply_qr_gauge(QrGauge::PositiveDiagonal, &mut outputs).unwrap();

    let scale = 2.0_f64.sqrt();
    let q = outputs[0].as_slice::<Complex64>().unwrap();
    assert!((q[0].re - scale).abs() < 1.0e-12);
    assert!((q[0].im - scale).abs() < 1.0e-12);
    assert!((q[1].re + 1.0 / scale).abs() < 1.0e-12);
    assert!((q[1].im - 1.0 / scale).abs() < 1.0e-12);

    let r = outputs[1].as_slice::<Complex64>().unwrap();
    assert!((r[0].re - scale).abs() < 1.0e-12);
    assert!(r[0].im.abs() < 1.0e-12);
    assert!((r[1].re - 7.0 / scale).abs() < 1.0e-12);
    assert!((r[1].im - 1.0 / scale).abs() < 1.0e-12);
    let after_product = matmul_c64(
        outputs[0].as_slice::<Complex64>().unwrap(),
        outputs[1].as_slice::<Complex64>().unwrap(),
        2,
        1,
        2,
    );
    assert_slice_close_c64(&after_product, &before_product, 1.0e-12);
}

#[test]
fn qr_gauge_covers_f32_and_c32_paths() {
    let mut real_outputs = vec![
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![-5.0_f32, 0.0, 6.0, -7.0]).unwrap(),
    ];

    apply_qr_gauge(QrGauge::PositiveDiagonal, &mut real_outputs).unwrap();

    assert_eq!(
        real_outputs[0].as_slice::<f32>().unwrap(),
        &[-1.0, -2.0, -3.0, -4.0]
    );
    assert_eq!(
        real_outputs[1].as_slice::<f32>().unwrap(),
        &[5.0, -0.0, -6.0, 7.0]
    );

    let mut complex_outputs = vec![
        Tensor::C32(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex32::new(2.0, 0.0),
                    Complex32::new(0.0, 1.0),
                    Complex32::new(0.5, 0.0),
                    Complex32::new(1.0, 0.0),
                ],
            )
            .unwrap(),
        ),
        Tensor::C32(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex32::new(1.0, 1.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(3.0, 4.0),
                    Complex32::new(0.0, 0.0),
                ],
            )
            .unwrap(),
        ),
    ];

    apply_qr_gauge(QrGauge::PositiveDiagonal, &mut complex_outputs).unwrap();

    let scale = 2.0_f32.sqrt();
    let q = complex_outputs[0].as_slice::<Complex32>().unwrap();
    assert!((q[0].re - scale).abs() < 1.0e-6);
    assert!((q[0].im - scale).abs() < 1.0e-6);
    assert!((q[1].re + 1.0 / scale).abs() < 1.0e-6);
    assert!((q[1].im - 1.0 / scale).abs() < 1.0e-6);
    assert_eq!(q[2], Complex32::new(0.5, 0.0));
    assert_eq!(q[3], Complex32::new(1.0, 0.0));

    let r = complex_outputs[1].as_slice::<Complex32>().unwrap();
    assert!((r[0].re - scale).abs() < 1.0e-6);
    assert!(r[0].im.abs() < 1.0e-6);
    assert!((r[2].re - 7.0 / scale).abs() < 1.0e-6);
    assert!((r[2].im - 1.0 / scale).abs() < 1.0e-6);
    assert_eq!(r[3], Complex32::new(0.0, 0.0));
}

#[test]
fn raw_gauges_leave_outputs_unchanged() {
    let mut eigh_outputs = vec![
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 2], vec![-1.0_f64, 0.0, 0.0, -1.0]).unwrap(),
    ];
    let original_vectors = eigh_outputs[1].as_slice::<f64>().unwrap().to_vec();

    apply_eigh_gauge(EighGauge::Raw, &mut eigh_outputs).unwrap();

    assert_eq!(eigh_outputs[1].as_slice::<f64>().unwrap(), original_vectors);

    let mut qr_outputs = vec![
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap(),
        Tensor::from_vec_col_major(vec![1, 2], vec![-3.0_f64, 4.0]).unwrap(),
    ];
    let original_q = qr_outputs[0].as_slice::<f64>().unwrap().to_vec();
    let original_r = qr_outputs[1].as_slice::<f64>().unwrap().to_vec();

    apply_qr_gauge(QrGauge::Raw, &mut qr_outputs).unwrap();

    assert_eq!(qr_outputs[0].as_slice::<f64>().unwrap(), original_q);
    assert_eq!(qr_outputs[1].as_slice::<f64>().unwrap(), original_r);
}

#[test]
fn gauge_validation_errors_cover_malformed_outputs() {
    assert_invalid_config(apply_eigh_gauge(EighGauge::CanonicalPivot, &mut []));

    let mut bad_eigh_rank = vec![
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap(),
        Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f64]).unwrap(),
    ];
    assert_invalid_config(apply_eigh_gauge(
        EighGauge::CanonicalPivot,
        &mut bad_eigh_rank,
    ));

    let mut bad_eigh_shape = vec![
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 0.0]).unwrap(),
    ];
    assert_invalid_config(apply_eigh_gauge(
        EighGauge::CanonicalPivot,
        &mut bad_eigh_shape,
    ));

    let mut bad_eigh_dtype = vec![
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        Tensor::from_vec_col_major(vec![1, 1], vec![1_i32]).unwrap(),
    ];
    assert!(matches!(
        apply_eigh_gauge(EighGauge::CanonicalPivot, &mut bad_eigh_dtype),
        Err(Error::Unsupported { .. })
    ));

    assert_invalid_config(apply_qr_gauge(QrGauge::PositiveDiagonal, &mut []));

    let mut bad_qr_rank = vec![
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f64]).unwrap(),
    ];
    assert_invalid_config(apply_qr_gauge(QrGauge::PositiveDiagonal, &mut bad_qr_rank));

    let mut bad_qr_shape = vec![
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
        Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 0.0]).unwrap(),
    ];
    assert_invalid_config(apply_qr_gauge(QrGauge::PositiveDiagonal, &mut bad_qr_shape));

    let mut bad_qr_dtype = vec![
        Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f64]).unwrap(),
        Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f32]).unwrap(),
    ];
    assert!(matches!(
        apply_qr_gauge(QrGauge::PositiveDiagonal, &mut bad_qr_dtype),
        Err(Error::Validation {
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
            ..
        })
    ));
}

fn reconstruct_eigh_f64(values: &[f64], vectors: &[f64], n: usize) -> Vec<f64> {
    let mut output = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            let mut sum = 0.0;
            for eig in 0..n {
                sum += vectors[row + n * eig] * values[eig] * vectors[col + n * eig];
            }
            output[row + n * col] = sum;
        }
    }
    output
}

fn reconstruct_eigh_c64(values: &[f64], vectors: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for eig in 0..n {
                sum += vectors[row + n * eig] * values[eig] * vectors[col + n * eig].conj();
            }
            output[row + n * col] = sum;
        }
    }
    output
}

fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut output = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += lhs[row + m * inner] * rhs[inner + k * col];
            }
            output[row + m * col] = sum;
        }
    }
    output
}

fn matmul_c64(
    lhs: &[Complex64],
    rhs: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for row in 0..m {
            let mut sum = Complex64::new(0.0, 0.0);
            for inner in 0..k {
                sum += lhs[row + m * inner] * rhs[inner + k * col];
            }
            output[row + m * col] = sum;
        }
    }
    output
}

fn assert_slice_close_f64(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "index {index}: actual={actual}, expected={expected}, tol={tol}"
        );
    }
}

fn assert_slice_close_c64(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() <= tol,
            "index {index}: actual={actual:?}, expected={expected:?}, tol={tol}"
        );
    }
}

fn assert_invalid_config(result: tenferro_tensor::Result<()>) {
    assert!(matches!(result, Err(Error::Validation { .. })));
}
