use super::*;
use crate::core::AdMode;
use ::tidu::expert::Tape;
use num_complex::Complex64;
use tenferro_internal_ad_core::DynAdTensor;
use tenferro_internal_frontend_core::ScalarType;

#[test]
fn eager_linalg_ad_results_expose_erased_carrier_metadata() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a =
        DenseTensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let da =
        DenseTensor::<f64>::from_slice(&[0.1, 0.0, 0.0, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let ad_a = AdTensor::new_forward(a, da).unwrap();

    let out_svd = crate::ops::ad::svd(&ad_a).unwrap();
    assert_eq!(out_svd.u.mode(), AdMode::Forward);
    assert_eq!(out_svd.s.scalar_type(), ScalarType::F64);
    assert_eq!(out_svd.vt.mode(), AdMode::Forward);

    let out_qr = crate::ops::ad::qr(&ad_a).unwrap();
    assert_eq!(out_qr.q.mode(), AdMode::Forward);
    assert_eq!(out_qr.r.mode(), AdMode::Forward);
}

#[test]
fn eager_complex_svd_uses_real_singular_value_carrier() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = DenseTensor::<num_complex::Complex64>::from_slice(
        &[
            num_complex::Complex64::new(2.0, 0.0),
            num_complex::Complex64::new(1.0, -1.0),
            num_complex::Complex64::new(0.0, 1.0),
            num_complex::Complex64::new(3.0, 0.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let da = DenseTensor::<num_complex::Complex64>::from_slice(
        &[
            num_complex::Complex64::new(0.1, 0.0),
            num_complex::Complex64::new(0.0, 0.1),
            num_complex::Complex64::new(-0.2, 0.0),
            num_complex::Complex64::new(0.0, -0.2),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let ad_a = AdTensor::new_forward(a, da).unwrap();

    let out_svd = crate::ops::ad::svd(&ad_a).unwrap();
    assert_eq!(out_svd.u.scalar_type(), ScalarType::C64);
    assert_eq!(out_svd.s.scalar_type(), ScalarType::F64);
    assert_eq!(out_svd.vt.scalar_type(), ScalarType::C64);
    assert_eq!(out_svd.u.mode(), AdMode::Forward);
    assert_eq!(out_svd.s.mode(), AdMode::Forward);
    assert_eq!(out_svd.vt.mode(), AdMode::Forward);
}

#[test]
fn eager_linalg_dyn_entrypoints_accept_erased_inputs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a =
        DenseTensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let da =
        DenseTensor::<f64>::from_slice(&[0.1, 0.0, 0.0, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let b = DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let db = DenseTensor::<f64>::from_slice(&[0.0, 0.2], &[2], MemoryOrder::ColumnMajor).unwrap();

    let dyn_a = DynAdTensor::from(AdTensor::new_forward(a, da).unwrap());
    let dyn_b = DynAdTensor::from(AdTensor::new_forward(b, db).unwrap());

    let out_svd = tenferro_internal_ad_linalg::eager::svd_dyn((&dyn_a).into()).unwrap();
    assert_eq!(out_svd.u.mode(), AdMode::Forward);
    assert_eq!(out_svd.s.scalar_type(), ScalarType::F64);
    assert_eq!(out_svd.vt.mode(), AdMode::Forward);

    let tape = Tape::<crate::DynTensor>::new();
    let dyn_reverse = DynAdTensor::from(reverse_leaf_f64(
        DenseTensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
        &tape,
    ));
    let out_svd_reverse =
        tenferro_internal_ad_linalg::eager::svd_dyn((&dyn_reverse).into()).unwrap();
    assert_reverse_on_tape(&out_svd_reverse.u, &tape);
    assert_reverse_on_tape(&out_svd_reverse.s, &tape);
    assert_reverse_on_tape(&out_svd_reverse.vt, &tape);

    let out_qr = tenferro_internal_ad_linalg::eager::qr_dyn((&dyn_a).into()).unwrap();
    assert_eq!(out_qr.q.scalar_type(), ScalarType::F64);
    assert_eq!(out_qr.q.mode(), AdMode::Forward);
    assert_eq!(out_qr.r.mode(), AdMode::Forward);

    let out_lstsq =
        tenferro_internal_ad_linalg::eager::lstsq_dyn((&dyn_a).into(), (&dyn_b).into()).unwrap();
    assert_eq!(out_lstsq.x.scalar_type(), ScalarType::F64);
    assert_eq!(out_lstsq.x.mode(), AdMode::Forward);
    assert_eq!(out_lstsq.residual.mode(), AdMode::Forward);
}

#[test]
fn eager_linalg_dyn_single_output_entrypoints_preserve_reverse_metadata() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let matrix = DynAdTensor::from(reverse_leaf_f64(
        DenseTensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
        &tape,
    ));
    let rhs_real = DynAdTensor::from(reverse_leaf_f64(
        DenseTensor::<f64>::from_slice(&[2.0, 0.0, 0.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
        &tape,
    ));
    let triangular = DynAdTensor::from(reverse_leaf_f64(
        DenseTensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
        &tape,
    ));
    let complex_matrix = DynAdTensor::from(reverse_leaf_c64(
        DenseTensor::<Complex64>::from_slice(
            &[
                Complex64::new(2.0, 0.5),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, -0.25),
                Complex64::new(3.0, 0.75),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        &tape,
    ));
    let complex_rhs = DynAdTensor::from(reverse_leaf_c64(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        &tape,
    ));

    let chol = tenferro_internal_ad_linalg::eager::cholesky_dyn((&matrix).into()).unwrap();
    assert_reverse_on_tape(&chol, &tape);

    let solve = tenferro_internal_ad_linalg::eager::solve_dyn((&matrix).into(), (&rhs_real).into())
        .unwrap();
    assert_reverse_on_tape(&solve, &tape);

    let solve_triangular = tenferro_internal_ad_linalg::eager::solve_triangular_dyn(
        (&triangular).into(),
        (&rhs_real).into(),
    )
    .unwrap();
    assert_reverse_on_tape(&solve_triangular, &tape);

    let complex_solve = tenferro_internal_ad_linalg::eager::solve_dyn(
        (&complex_matrix).into(),
        (&complex_rhs).into(),
    )
    .unwrap();
    assert_reverse_on_tape(&complex_solve, &tape);

    let complex_solve_triangular = tenferro_internal_ad_linalg::eager::solve_triangular_dyn(
        (&complex_matrix).into(),
        (&complex_rhs).into(),
    )
    .unwrap();
    assert_reverse_on_tape(&complex_solve_triangular, &tape);

    let inv = tenferro_internal_ad_linalg::eager::inv_dyn((&matrix).into()).unwrap();
    assert_reverse_on_tape(&inv, &tape);

    let det = tenferro_internal_ad_linalg::eager::det_dyn((&matrix).into()).unwrap();
    assert_reverse_on_tape(&det, &tape);

    let pinv = tenferro_internal_ad_linalg::eager::pinv_dyn((&matrix).into()).unwrap();
    assert_reverse_on_tape(&pinv, &tape);

    let matrix_exp = tenferro_internal_ad_linalg::eager::matrix_exp_dyn((&matrix).into()).unwrap();
    assert_reverse_on_tape(&matrix_exp, &tape);

    let norm = tenferro_internal_ad_linalg::eager::norm_dyn((&matrix).into()).unwrap();
    assert_reverse_on_tape(&norm, &tape);
}
