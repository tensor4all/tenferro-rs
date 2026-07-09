use std::sync::Arc;

use num_complex::Complex64;
use tenferro_runtime::extension::ExtensionOp;
use tenferro_tensor::{
    Buffer, BufferHandle, DType, DeviceId, DeviceKind, Error, GpuBackendKind, MemoryKind,
    Placement, Tensor, TypedTensor,
};

use super::{apply_svd_gauge, promote_dtypes, LinalgExtensionOp, LinalgOp, SvdGauge};

#[test]
fn eager_linalg_rejects_cuda_tensor_when_cuda_feature_is_disabled() {
    let tensor = Tensor::F64(
        TypedTensor::from_buffer_col_major(
            vec![2, 2],
            Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, 4))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
            },
        )
        .unwrap(),
    );
    let op = LinalgExtensionOp::new(LinalgOp::Cholesky);

    let err = op
        .host_reference()
        .expect("linalg host reference")
        .execute(&[&tensor])
        .unwrap_err();

    match err {
        Error::BackendFailure { op, message } => {
            assert_eq!(op, "linalg_host_reference");
            assert!(message.contains("cuda feature"));
            assert!(message.contains("download"));
        }
        other => panic!("expected BackendFailure, got {other:?}"),
    }
}

#[test]
fn infer_output_meta_returns_error_on_input_count_mismatch() {
    let op = LinalgExtensionOp::new(LinalgOp::Cholesky);

    let err = op.infer_output_meta(&[], &[]).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "tenferro-linalg",
            ..
        }
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
