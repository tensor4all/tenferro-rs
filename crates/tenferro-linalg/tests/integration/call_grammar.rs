//! The #2552 call grammar on tenferro-linalg: every operation has an options type with
//! `Default` and `with_*` setters, and no public option carries a `bool`.

use tenferro_linalg::{
    CholeskyOptions, DetOptions, Diagonal, EigOptions, EighGauge, EighOptions, EigvalsOptions,
    EigvalshOptions, FullPivLuOptions, FullPivLuSolveOptions, HouseholderQrOptions, InvOptions,
    KeepDims, LstsqOptions, LuOptions, NormOptions, PinvOptions, QrGauge, QrOptions,
    RankRevealingQrOptions, Side, SlogdetOptions, SolveOptions, SvdDriver, SvdGauge, SvdOptions,
    SvdShape, SvdvalsOptions, Transpose, TriangularSolveOptions, Uplo,
    DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
};

#[test]
fn svd_options_carry_shape_gauge_driver_and_derivative_eps() {
    let defaults = SvdOptions::default();
    assert_eq!(defaults.shape, SvdShape::Thin);
    assert_eq!(defaults.gauge, SvdGauge::Raw);
    assert_eq!(defaults.driver, SvdDriver::Auto);
    assert_eq!(
        defaults.derivative_eps,
        DEFAULT_DECOMPOSITION_DERIVATIVE_EPS
    );

    let options = SvdOptions::default()
        .with_shape(SvdShape::Full)
        .with_gauge(SvdGauge::CanonicalPivot)
        .with_driver(SvdDriver::Jacobi)
        .with_derivative_eps(1.0e-9);
    assert_eq!(options.shape, SvdShape::Full);
    assert_eq!(options.gauge, SvdGauge::CanonicalPivot);
    assert_eq!(options.driver, SvdDriver::Jacobi);
    assert_eq!(options.derivative_eps, 1.0e-9);
}

#[test]
fn qr_family_options_use_with_setters() {
    assert_eq!(QrOptions::default().gauge, QrGauge::Raw);
    assert_eq!(
        QrOptions::default()
            .with_gauge(QrGauge::PositiveDiagonal)
            .gauge,
        QrGauge::PositiveDiagonal
    );

    let rrqr = RankRevealingQrOptions::default()
        .with_gauge(QrGauge::PositiveDiagonal)
        .with_rtol(1.0e-10)
        .with_atol(1.0e-14);
    assert_eq!(rrqr.gauge, QrGauge::PositiveDiagonal);
    assert_eq!(rrqr.rtol, 1.0e-10);
    assert_eq!(rrqr.atol, 1.0e-14);
    assert_eq!(RankRevealingQrOptions::default().rtol, 0.0);
    assert_eq!(RankRevealingQrOptions::default().atol, 0.0);

    let _ = HouseholderQrOptions::default();
}

#[test]
fn eigh_options_use_with_setters() {
    let options = EighOptions::default()
        .with_gauge(EighGauge::CanonicalPivot)
        .with_derivative_eps(1.0e-10);
    assert_eq!(options.gauge, EighGauge::CanonicalPivot);
    assert_eq!(options.derivative_eps, 1.0e-10);
    assert_eq!(EighOptions::default().gauge, EighGauge::Raw);
}

#[test]
fn triangular_solve_options_replace_four_bool_flags() {
    // The defaults are the all-`false` flags of the former signature, the same defaults as
    // `jax.lax.linalg.triangular_solve` (see OPEN-QUESTIONS-2552.md).
    let defaults = TriangularSolveOptions::default();
    assert_eq!(defaults.side, Side::Right);
    assert_eq!(defaults.uplo, Uplo::Upper);
    assert_eq!(defaults.transpose, Transpose::NoTranspose);
    assert_eq!(defaults.diagonal, Diagonal::NonUnit);

    let options = TriangularSolveOptions::default()
        .with_side(Side::Left)
        .with_uplo(Uplo::Lower)
        .with_transpose(Transpose::Transpose)
        .with_diagonal(Diagonal::Unit);
    assert_eq!(options.side, Side::Left);
    assert_eq!(options.uplo, Uplo::Lower);
    assert_eq!(options.transpose, Transpose::Transpose);
    assert_eq!(options.diagonal, Diagonal::Unit);
}

#[test]
fn norm_options_take_signed_axes_and_a_keep_dims_enum() {
    let defaults = NormOptions::default();
    assert_eq!(defaults.ord, None);
    assert_eq!(defaults.axes, None);
    assert_eq!(defaults.keep_dims, KeepDims::Drop);

    let options = NormOptions::default()
        .with_ord(2.0)
        .with_axes(vec![-2, -1])
        .with_keep_dims(KeepDims::Keep);
    assert_eq!(options.ord, Some(2.0));
    assert_eq!(options.axes.as_deref(), Some(&[-2_isize, -1][..]));
    assert_eq!(options.keep_dims, KeepDims::Keep);
}

#[test]
fn pinv_options_default_to_the_dtype_tolerance() {
    assert_eq!(PinvOptions::default().rtol, None);
    assert_eq!(PinvOptions::default().with_rtol(1.0e-8).rtol, Some(1.0e-8));
}

#[test]
fn option_less_operations_have_empty_options() {
    let _ = SvdvalsOptions::default();
    let _ = LuOptions::default();
    let _ = FullPivLuOptions::default();
    let _ = FullPivLuSolveOptions::default();
    let _ = SolveOptions::default();
    let _ = CholeskyOptions::default();
    let _ = EigOptions::default();
    let _ = EigvalsOptions::default();
    let _ = EigvalshOptions::default();
    let _ = SlogdetOptions::default();
    let _ = DetOptions::default();
    let _ = InvOptions::default();
    let _ = LstsqOptions::default();
}
