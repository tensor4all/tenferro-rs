use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_runtime::extension::apply;
use tenferro_runtime::{
    CompareDir, DType, DotGeneralConfig, Error, ErrorPhase, Result, TraceContext, TraceValue,
    TracedTensor,
};

use crate::extension::{
    validate_derivative_eps, EighOptions, LinalgExtensionOp, LinalgOp, QrOptions, SvdOptions,
};
use crate::rank_revealing_qr::validate_rank_revealing_qr_options;
use crate::validation::{ensure_float_or_complex, validate_lstsq, validate_square_matrix};
use crate::{RankRevealingQrOptions, RankRevealingQrResult};

/// Crate-internal traced linalg bodies; the public grammar is
/// `TracedLinalgOps` (#2552), which forwards here.
pub(crate) trait TracedLinalg {
    fn svd(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    fn svd_with_options(
        &self,
        options: SvdOptions,
    ) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    fn svd_full(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    fn qr(&self) -> Result<(TracedTensor, TracedTensor)>;

    fn householder_qr(&self) -> Result<crate::HouseholderQr<TracedTensor>>;

    fn qr_with_options(&self, options: QrOptions) -> Result<(TracedTensor, TracedTensor)>;

    fn rank_revealing_qr(
        &self,
        options: RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<TracedTensor>>;

    fn eigh(&self) -> Result<(TracedTensor, TracedTensor)>;

    fn eigh_with_options(&self, options: EighOptions) -> Result<(TracedTensor, TracedTensor)>;

    fn cholesky(&self) -> Result<TracedTensor>;

    fn lu(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)>;

    fn full_piv_lu(
        &self,
    ) -> Result<(
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
    )>;
    fn eig(&self) -> Result<(TracedTensor, TracedTensor)>;

    fn solve(&self, b: &TracedTensor) -> Result<TracedTensor>;

    fn lstsq(&self, b: &TracedTensor) -> Result<TracedTensor>;

    fn full_piv_lu_solve(&self, b: &TracedTensor) -> Result<TracedTensor>;

    fn triangular_solve(
        &self,
        b: &TracedTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<TracedTensor>;
    fn slogdet(&self) -> Result<(TracedTensor, TracedTensor)>;

    fn det(&self) -> Result<TracedTensor>;

    fn inv(&self) -> Result<TracedTensor>;

    fn eigvalsh(&self) -> Result<TracedTensor>;

    fn eigvals(&self) -> Result<TracedTensor>;

    fn pinv(&self) -> Result<TracedTensor>;

    fn pinv_with_rtol(&self, rtol: f64) -> Result<TracedTensor>;

    fn norm(&self, ord: Option<f64>, dim: Option<&[usize]>, keepdim: bool) -> Result<TracedTensor>;
}

/// Crate-internal trace-value linalg construction used by `TracedLinalg`.
pub(crate) trait TraceContextLinalg {
    fn svd(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    fn svd_with_options(
        &self,
        a: TraceValue,
        options: SvdOptions,
    ) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    fn svd_full(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    fn qr(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor)>;

    fn householder_qr(&self, a: TraceValue) -> Result<crate::HouseholderQr<TracedTensor>>;

    fn qr_with_options(
        &self,
        a: TraceValue,
        options: QrOptions,
    ) -> Result<(TracedTensor, TracedTensor)>;

    fn rank_revealing_qr(
        &self,
        a: TraceValue,
        options: RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<TracedTensor>>;

    fn eigh(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor)>;

    fn eigh_with_options(
        &self,
        a: TraceValue,
        options: EighOptions,
    ) -> Result<(TracedTensor, TracedTensor)>;

    fn cholesky(&self, a: TraceValue) -> Result<TracedTensor>;

    fn lu(&self, a: TraceValue)
        -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)>;

    fn full_piv_lu(
        &self,
        a: TraceValue,
    ) -> Result<(
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
    )>;

    fn eig(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor)>;

    fn solve(&self, a: TraceValue, b: TraceValue) -> Result<TracedTensor>;

    fn lstsq(&self, a: TraceValue, b: TraceValue) -> Result<TracedTensor>;

    fn full_piv_lu_solve(&self, a: TraceValue, b: TraceValue) -> Result<TracedTensor>;

    fn triangular_solve(
        &self,
        a: TraceValue,
        b: TraceValue,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<TracedTensor>;

    fn slogdet(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor)>;

    fn det(&self, a: TraceValue) -> Result<TracedTensor>;

    fn inv(&self, a: TraceValue) -> Result<TracedTensor>;

    fn eigvalsh(&self, a: TraceValue) -> Result<TracedTensor>;

    fn eigvals(&self, a: TraceValue) -> Result<TracedTensor>;

    fn pinv(&self, a: TraceValue) -> Result<TracedTensor>;

    fn pinv_with_rtol(&self, a: TraceValue, rtol: f64) -> Result<TracedTensor>;

    fn norm(
        &self,
        a: TraceValue,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<TracedTensor>;
}

impl TraceContextLinalg for TraceContext {
    fn svd(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        svd(&self.derive_tensor(a)?)
    }

    fn svd_with_options(
        &self,
        a: TraceValue,
        options: SvdOptions,
    ) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        svd_with_options(&self.derive_tensor(a)?, options)
    }

    fn svd_full(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        svd_full(&self.derive_tensor(a)?)
    }

    fn qr(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor)> {
        qr(&self.derive_tensor(a)?)
    }

    fn householder_qr(&self, a: TraceValue) -> Result<crate::HouseholderQr<TracedTensor>> {
        householder_qr(&self.derive_tensor(a)?)
    }

    fn qr_with_options(
        &self,
        a: TraceValue,
        options: QrOptions,
    ) -> Result<(TracedTensor, TracedTensor)> {
        qr_with_options(&self.derive_tensor(a)?, options)
    }

    fn rank_revealing_qr(
        &self,
        a: TraceValue,
        options: RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<TracedTensor>> {
        rank_revealing_qr(&self.derive_tensor(a)?, options)
    }

    fn eigh(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor)> {
        eigh(&self.derive_tensor(a)?)
    }

    fn eigh_with_options(
        &self,
        a: TraceValue,
        options: EighOptions,
    ) -> Result<(TracedTensor, TracedTensor)> {
        eigh_with_options(&self.derive_tensor(a)?, options)
    }

    fn cholesky(&self, a: TraceValue) -> Result<TracedTensor> {
        cholesky(&self.derive_tensor(a)?)
    }

    fn lu(
        &self,
        a: TraceValue,
    ) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
        lu(&self.derive_tensor(a)?)
    }

    fn full_piv_lu(
        &self,
        a: TraceValue,
    ) -> Result<(
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
    )> {
        full_piv_lu(&self.derive_tensor(a)?)
    }

    fn eig(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor)> {
        eig(&self.derive_tensor(a)?)
    }

    fn solve(&self, a: TraceValue, b: TraceValue) -> Result<TracedTensor> {
        solve(&self.derive_tensor(a)?, &self.derive_tensor(b)?)
    }

    fn lstsq(&self, a: TraceValue, b: TraceValue) -> Result<TracedTensor> {
        lstsq(&self.derive_tensor(a)?, &self.derive_tensor(b)?)
    }

    fn full_piv_lu_solve(&self, a: TraceValue, b: TraceValue) -> Result<TracedTensor> {
        full_piv_lu_solve(&self.derive_tensor(a)?, &self.derive_tensor(b)?)
    }

    fn triangular_solve(
        &self,
        a: TraceValue,
        b: TraceValue,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<TracedTensor> {
        triangular_solve(
            &self.derive_tensor(a)?,
            &self.derive_tensor(b)?,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )
    }

    fn slogdet(&self, a: TraceValue) -> Result<(TracedTensor, TracedTensor)> {
        slogdet(&self.derive_tensor(a)?)
    }

    fn det(&self, a: TraceValue) -> Result<TracedTensor> {
        det(&self.derive_tensor(a)?)
    }

    fn inv(&self, a: TraceValue) -> Result<TracedTensor> {
        inv(&self.derive_tensor(a)?)
    }

    fn eigvalsh(&self, a: TraceValue) -> Result<TracedTensor> {
        eigvalsh(&self.derive_tensor(a)?)
    }

    fn eigvals(&self, a: TraceValue) -> Result<TracedTensor> {
        eigvals(&self.derive_tensor(a)?)
    }

    fn pinv(&self, a: TraceValue) -> Result<TracedTensor> {
        pinv(&self.derive_tensor(a)?)
    }

    fn pinv_with_rtol(&self, a: TraceValue, rtol: f64) -> Result<TracedTensor> {
        pinv_with_rtol(&self.derive_tensor(a)?, rtol)
    }

    fn norm(
        &self,
        a: TraceValue,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<TracedTensor> {
        norm(&self.derive_tensor(a)?, ord, dim, keepdim)
    }
}

impl TracedLinalg for TracedTensor {
    fn svd(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        self.context().svd(self.trace_value())
    }

    fn svd_with_options(
        &self,
        options: SvdOptions,
    ) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        self.context().svd_with_options(self.trace_value(), options)
    }

    fn svd_full(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        self.context().svd_full(self.trace_value())
    }

    fn qr(&self) -> Result<(TracedTensor, TracedTensor)> {
        self.context().qr(self.trace_value())
    }

    fn householder_qr(&self) -> Result<crate::HouseholderQr<TracedTensor>> {
        self.context().householder_qr(self.trace_value())
    }

    fn qr_with_options(&self, options: QrOptions) -> Result<(TracedTensor, TracedTensor)> {
        self.context().qr_with_options(self.trace_value(), options)
    }

    fn rank_revealing_qr(
        &self,
        options: RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<TracedTensor>> {
        self.context()
            .rank_revealing_qr(self.trace_value(), options)
    }

    fn eigh(&self) -> Result<(TracedTensor, TracedTensor)> {
        self.context().eigh(self.trace_value())
    }

    fn eigh_with_options(&self, options: EighOptions) -> Result<(TracedTensor, TracedTensor)> {
        self.context()
            .eigh_with_options(self.trace_value(), options)
    }

    fn cholesky(&self) -> Result<TracedTensor> {
        self.context().cholesky(self.trace_value())
    }

    fn lu(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
        self.context().lu(self.trace_value())
    }

    fn full_piv_lu(
        &self,
    ) -> Result<(
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
    )> {
        self.context().full_piv_lu(self.trace_value())
    }

    fn eig(&self) -> Result<(TracedTensor, TracedTensor)> {
        self.context().eig(self.trace_value())
    }

    fn solve(&self, b: &TracedTensor) -> Result<TracedTensor> {
        self.context().solve(self.trace_value(), b.trace_value())
    }

    fn lstsq(&self, b: &TracedTensor) -> Result<TracedTensor> {
        self.context().lstsq(self.trace_value(), b.trace_value())
    }

    fn full_piv_lu_solve(&self, b: &TracedTensor) -> Result<TracedTensor> {
        self.context()
            .full_piv_lu_solve(self.trace_value(), b.trace_value())
    }

    fn triangular_solve(
        &self,
        b: &TracedTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<TracedTensor> {
        self.context().triangular_solve(
            self.trace_value(),
            b.trace_value(),
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )
    }

    fn slogdet(&self) -> Result<(TracedTensor, TracedTensor)> {
        self.context().slogdet(self.trace_value())
    }

    fn det(&self) -> Result<TracedTensor> {
        self.context().det(self.trace_value())
    }

    fn inv(&self) -> Result<TracedTensor> {
        self.context().inv(self.trace_value())
    }

    fn eigvalsh(&self) -> Result<TracedTensor> {
        self.context().eigvalsh(self.trace_value())
    }

    fn eigvals(&self) -> Result<TracedTensor> {
        self.context().eigvals(self.trace_value())
    }

    fn pinv(&self) -> Result<TracedTensor> {
        self.context().pinv(self.trace_value())
    }

    fn pinv_with_rtol(&self, rtol: f64) -> Result<TracedTensor> {
        self.context().pinv_with_rtol(self.trace_value(), rtol)
    }

    fn norm(&self, ord: Option<f64>, dim: Option<&[usize]>, keepdim: bool) -> Result<TracedTensor> {
        self.context().norm(self.trace_value(), ord, dim, keepdim)
    }
}

pub(crate) fn svd(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    svd_with_options(a, SvdOptions::default())
}

pub(crate) fn svd_with_options(
    a: &TracedTensor,
    options: SvdOptions,
) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    validate_derivative_eps("svd_with_options", options.derivative_eps)?;
    ensure_float_or_complex("svd", a.dtype())?;
    three_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Svd {
                derivative_eps: options.derivative_eps,
                gauge: options.gauge,
                driver: options.driver,
            })),
            &[a],
        )?,
        "svd",
    )
}

pub(crate) fn svd_full(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    ensure_float_or_complex("svd_full", a.dtype())?;
    three_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::SvdFull)), &[a])?,
        "svd_full",
    )
}

pub(crate) fn qr(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    qr_with_options(a, QrOptions::default())
}

pub(crate) fn householder_qr(a: &TracedTensor) -> Result<crate::HouseholderQr<TracedTensor>> {
    ensure_float_or_complex("householder_qr", a.dtype())?;
    let mut outputs = apply(
        Arc::new(LinalgExtensionOp::new(LinalgOp::HouseholderQrFactor)),
        &[a],
    )?
    .into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(packed), Some(coeff), None) => Ok(
            crate::HouseholderQr::<TracedTensor>::from_traced_outputs(packed, coeff),
        ),
        _ => Err(unexpected_output_count("householder_qr", 2)),
    }
}

pub(crate) fn qr_with_options(
    a: &TracedTensor,
    options: QrOptions,
) -> Result<(TracedTensor, TracedTensor)> {
    ensure_float_or_complex("qr", a.dtype())?;
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Qr {
                gauge: options.gauge,
            })),
            &[a],
        )?,
        "qr",
    )
}

pub(crate) fn rank_revealing_qr(
    a: &TracedTensor,
    options: RankRevealingQrOptions,
) -> Result<RankRevealingQrResult<TracedTensor>> {
    validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
    ensure_float_or_complex("rank_revealing_qr", a.dtype())?;
    let (q, r, column_permutation, rank) = four_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::RankRevealingQr {
                gauge: options.gauge,
                rtol: options.rtol,
                atol: options.atol,
            })),
            &[a],
        )?,
        "rank_revealing_qr",
    )?;
    Ok(RankRevealingQrResult {
        q,
        r,
        column_permutation,
        rank,
    })
}

pub(crate) fn eigh(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    eigh_with_options(a, EighOptions::default())
}

pub(crate) fn eigh_with_options(
    a: &TracedTensor,
    options: EighOptions,
) -> Result<(TracedTensor, TracedTensor)> {
    validate_derivative_eps("eigh_with_options", options.derivative_eps)?;
    ensure_float_or_complex("eigh", a.dtype())?;
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eigh {
                derivative_eps: options.derivative_eps,
                gauge: options.gauge,
                driver: options.driver,
            })),
            &[a],
        )?,
        "eigh",
    )
}

pub(crate) fn cholesky(a: &TracedTensor) -> Result<TracedTensor> {
    ensure_float_or_complex("cholesky", a.dtype())?;
    one_output(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Cholesky)), &[a])?,
        "cholesky",
    )
}

pub(crate) fn lu(
    a: &TracedTensor,
) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
    four_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Lu)), &[a])?,
        "lu",
    )
}

pub(crate) fn full_piv_lu(
    a: &TracedTensor,
) -> Result<(
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
)> {
    five_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLu)), &[a])?,
        "full_piv_lu",
    )
}

pub(crate) fn eig(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eig {
                input_dtype: a.dtype(),
            })),
            &[a],
        )?,
        "eig",
    )
}

pub(crate) fn solve(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    // One fused op factors and solves in a single backend call and saves
    // (x, packed LU, pivots), so AD reuses the factors for the adjoint solve.
    let mut outputs = apply(
        Arc::new(LinalgExtensionOp::new(LinalgOp::LuFactorSolve)),
        &[a, b],
    )?
    .into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(x), Some(_packed_lu), Some(_pivots), None) => Ok(x),
        _ => Err(unexpected_output_count("lu_factor_solve", 3)),
    }
}

pub(crate) fn lstsq(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    validate_lstsq(
        "lstsq",
        a.dtype(),
        a.rank(),
        b.rank(),
        || {
            let shape = require_concrete_shape("lstsq", a)?;
            Ok((shape[0], shape[1]))
        },
        |message| {
            Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
                "lstsq", "shape", message,
            ))
        },
    )?;
    let (q, r) = qr(a)?;
    let qh = q.conj()?.transpose(&matrix_transpose_perm(q.rank()))?;
    let qh_b = matmul_preserve_trailing_batch(&qh, b)?;
    triangular_solve(&r, &qh_b, true, false, false, false)
}

pub(crate) fn full_piv_lu_solve(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLuSolve {
                transpose_a: false,
            })),
            &[a, b],
        )?,
        "full_piv_lu_solve",
    )
}

pub(crate) fn triangular_solve(
    a: &TracedTensor,
    b: &TracedTensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            })),
            &[a, b],
        )?,
        "triangular_solve",
    )
}

pub(crate) fn slogdet(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    if let Some(shape) = a.try_concrete_shape() {
        validate_square_matrix("slogdet", &shape)?;
    }
    if let Some(empty) = slogdet_empty_square(a)? {
        return Ok(empty);
    }
    let mut factor_outputs =
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::LuFactor)), &[a])?.into_iter();
    let (packed_lu, parity) = match (
        factor_outputs.next(),
        factor_outputs.next(),
        factor_outputs.next(),
        factor_outputs.next(),
    ) {
        (Some(packed_lu), Some(_pivots), Some(parity), None) => (packed_lu, parity),
        _ => return Err(unexpected_output_count("lu_factor", 3)),
    };
    let mut sign_outputs = apply(
        Arc::new(LinalgExtensionOp::new(LinalgOp::SignDetFromLuFactor)),
        &[a, &packed_lu, &parity],
    )?
    .into_iter();
    let sign = match (sign_outputs.next(), sign_outputs.next()) {
        (Some(sign), None) => sign,
        _ => return Err(unexpected_output_count("signdet_from_lu_factor", 1)),
    };
    let mut logabsdet_outputs = apply(
        Arc::new(LinalgExtensionOp::new(LinalgOp::LogAbsDetFromLuFactor)),
        &[a, &packed_lu],
    )?
    .into_iter();
    let logabsdet = match (logabsdet_outputs.next(), logabsdet_outputs.next()) {
        (Some(logabsdet), None) => logabsdet,
        _ => return Err(unexpected_output_count("logabsdet_from_lu_factor", 1)),
    };
    Ok((sign, logabsdet))
}

pub(crate) fn det(a: &TracedTensor) -> Result<TracedTensor> {
    // JAX's recipe: `sign, logdet = slogdet(a); return sign * exp(logdet)`
    // (`_det` in `jax/_src/numpy/linalg.py`). Multiplying the LU diagonal
    // instead overflows before the magnitudes cancel when the determinant
    // spans an extreme dynamic range, so the two recipes disagree there.
    let (sign, logabsdet) = slogdet(a)?;
    &sign * &logabsdet.exp()?
}

fn slogdet_empty_square(a: &TracedTensor) -> Result<Option<(TracedTensor, TracedTensor)>> {
    let ctx = a.context();
    let Some(shape) = a.try_concrete_shape() else {
        return Ok(None);
    };
    if shape.len() < 2 || shape[0] != 0 || shape[1] != 0 {
        return Ok(None);
    }
    let batch_shape = shape[2..].to_vec();
    Ok(Some((
        filled_real(ctx, a.dtype(), batch_shape.clone(), 1.0)?,
        filled_real(ctx, real_values_dtype(a.dtype()), batch_shape, 0.0)?,
    )))
}

pub(crate) fn inv(a: &TracedTensor) -> Result<TracedTensor> {
    ensure_min_rank("inv", a.rank(), 2)?;
    let shape = require_concrete_shape("inv", a)?;
    let eye = eye_like(a, shape[0])?;
    solve(a, &eye)
}

pub(crate) fn eigvalsh(a: &TracedTensor) -> Result<TracedTensor> {
    ensure_float_or_complex("eigvalsh", a.dtype())?;
    eigh_values(a)
}

pub(crate) fn eigvals(a: &TracedTensor) -> Result<TracedTensor> {
    eig_values(a)
}

pub(crate) fn pinv(a: &TracedTensor) -> Result<TracedTensor> {
    ensure_float_or_complex("pinv", a.dtype())?;
    let shape = require_concrete_shape("pinv", a)?;
    let max_dim = match (shape.first(), shape.get(1)) {
        (Some(&m), Some(&n)) => m.max(n),
        (Some(&m), None) => m,
        _ => 0,
    };
    pinv_with_rtol(a, default_pinv_rtol(a.dtype(), max_dim))
}

pub(crate) fn pinv_with_rtol(a: &TracedTensor, rtol: f64) -> Result<TracedTensor> {
    ensure_float_or_complex("pinv_with_rtol", a.dtype())?;
    require_concrete_shape("pinv_with_rtol", a)?;
    let (u, s, vt) = svd(a)?;
    let abs_s = s.abs()?;
    let s_max = abs_s.reduce_max(Some(&[0]))?;
    let s_max_shape = s_max.concrete_shape()?;
    let threshold_scalar = broadcast_scalar(
        scalar_real(s.context(), s.dtype(), rtol.max(0.0))?,
        &s_max_shape,
    )?;
    let threshold = (&s_max * &threshold_scalar)?;
    let s_shape = s.concrete_shape()?;
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, &s_shape)?;
    let mask = abs_s.compare(&threshold, CompareDir::Gt)?;
    let mask = mask.convert(s.dtype())?;
    let ones = ones_like(&s)?;
    let neg_mask = (-&mask)?;
    let denom = (&s + &(&ones + &neg_mask)?)?;
    let s_inv = (&mask / &denom)?;

    let v = vt.conj()?.transpose(&matrix_transpose_perm(vt.rank()))?;
    let uh = u.conj()?.transpose(&matrix_transpose_perm(u.rank()))?;
    // See the concrete builder: a complex input has real singular values, so
    // `s_inv` is real while `v` is complex, and the multiply requires one dtype
    // (#2049).
    let s_inv = if s_inv.dtype() == v.dtype() {
        s_inv
    } else {
        s_inv.convert(v.dtype())?
    };
    let vs = scale_matrix_columns(&v, &s_inv)?;
    matmul_preserve_trailing_batch(&vs, &uh)
}

pub(crate) fn norm(
    a: &TracedTensor,
    ord: Option<f64>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> Result<TracedTensor> {
    ensure_float_or_complex("norm", a.dtype())?;
    let shape = require_concrete_shape("norm", a)?;
    let axes = dim.map_or_else(|| (0..a.rank()).collect::<Vec<_>>(), |dims| dims.to_vec());
    validate_axes("norm", a.rank(), &axes)?;
    if axes.is_empty() {
        return Ok(a.clone());
    }
    if reduced_axes_have_zero_extent(&shape, &axes)
        && let Some(zero) =
            zero_norm_for_empty_reduction(a.context(), a.dtype(), &shape, &axes, keepdim, ord)?
    {
        return Ok(zero);
    }

    let out = if can_square_without_abs(a.dtype(), axes.len(), ord) {
        frobenius_norm(a, &axes)?
    } else {
        match axes.len() {
            1 => vector_norm(a, axes[0], ord)?,
            2 => matrix_norm(a, &axes, ord)?,
            _ => {
                let abs = a.abs()?;
                match ord {
                    None => frobenius_norm(&abs, &axes)?,
                    Some(p) if p == f64::INFINITY => abs.reduce_max(Some(&axes))?,
                    Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(Some(&axes))?,
                    Some(0.0) => count_nonzero(&abs, &axes)?,
                    Some(p) => p_norm(&abs, &axes, p)?,
                }
            }
        }
    };
    restore_keepdim(out, &shape, &axes, keepdim)
}

fn unexpected_output_count(name: &str, expected: usize) -> Error {
    Error::Internal(format!("{name} must produce exactly {expected} outputs"))
}

fn one_output(outputs: Vec<TracedTensor>, name: &str) -> Result<TracedTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(unexpected_output_count(name, 1)),
    }
}

fn two_outputs(outputs: Vec<TracedTensor>, name: &str) -> Result<(TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(lhs), Some(rhs), None) => Ok((lhs, rhs)),
        _ => Err(unexpected_output_count(name, 2)),
    }
}

fn three_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), None) => Ok((first, second, third)),
        _ => Err(unexpected_output_count(name, 3)),
    }
}

fn four_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), Some(fourth), None) => {
            Ok((first, second, third, fourth))
        }
        _ => Err(unexpected_output_count(name, 4)),
    }
}

fn five_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), Some(fourth), Some(fifth), None) => {
            Ok((first, second, third, fourth, fifth))
        }
        _ => Err(unexpected_output_count(name, 5)),
    }
}

fn scalar_real(ctx: &TraceContext, dtype: DType, value: f64) -> Result<TracedTensor> {
    match dtype {
        DType::F64 => ctx.from_vec_col_major(vec![], vec![value]),
        DType::F32 => ctx.from_vec_col_major(vec![], vec![value as f32]),
        DType::I32 => ctx.from_vec_col_major(vec![], vec![value.round() as i32]),
        DType::I64 => ctx.from_vec_col_major(vec![], vec![value.round() as i64]),
        DType::Bool => ctx.from_vec_col_major(vec![], vec![value != 0.0]),
        DType::C64 => ctx.from_vec_col_major(vec![], vec![Complex64::new(value, 0.0)]),
        DType::C32 => ctx.from_vec_col_major(vec![], vec![Complex32::new(value as f32, 0.0)]),
        // An externally defined scalar has no traced scalar literal.
        DType::External(_) => Err(Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "scalar_real",
                "dtype",
                "an externally defined scalar has no traced scalar literal",
            ),
        )),
    }
}

fn filled_real(
    ctx: &TraceContext,
    dtype: DType,
    shape: Vec<usize>,
    value: f64,
) -> Result<TracedTensor> {
    let len = tenferro_tensor::validate::checked_shape_product("slogdet", "output shape", &shape)?;
    match dtype {
        DType::F64 => ctx.from_vec_col_major(shape, vec![value; len]),
        DType::F32 => ctx.from_vec_col_major(shape, vec![value as f32; len]),
        DType::I32 => ctx.from_vec_col_major(shape, vec![value.round() as i32; len]),
        DType::I64 => ctx.from_vec_col_major(shape, vec![value.round() as i64; len]),
        DType::Bool => ctx.from_vec_col_major(shape, vec![value != 0.0; len]),
        DType::C64 => ctx.from_vec_col_major(shape, vec![Complex64::new(value, 0.0); len]),
        DType::C32 => ctx.from_vec_col_major(shape, vec![Complex32::new(value as f32, 0.0); len]),
        // An externally defined scalar has no traced filled literal.
        DType::External(_) => Err(Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "filled_real",
                "dtype",
                "an externally defined scalar has no traced filled literal",
            ),
        )),
    }
}

pub(crate) fn real_values_dtype(dtype: DType) -> DType {
    match dtype {
        DType::C64 => DType::F64,
        DType::C32 => DType::F32,
        other => other,
    }
}

fn can_square_without_abs(dtype: DType, axes_len: usize, ord: Option<f64>) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
        && (ord.is_none() || (ord == Some(2.0) && axes_len != 2))
}

fn ensure_min_rank(op: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual < expected {
        return Err(Error::TensorRuntime(tenferro_tensor::Error::rank_mismatch(
            op, expected, actual,
        )));
    }
    Ok(())
}

fn validate_axes(op: &'static str, rank: usize, axes: &[usize]) -> Result<()> {
    tenferro_tensor::validate::validate_unique_axes(op, "dim", rank, axes)
        .map_err(Error::TensorRuntime)
}

fn require_concrete_shape(op: &'static str, input: &TracedTensor) -> Result<Vec<usize>> {
    input.try_concrete_shape().ok_or_else(|| {
        Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            op,
            "shape",
            "symbolic shape is not supported by this traced linalg helper",
        ))
    })
}

fn zero_scalar(ctx: &TraceContext, dtype: DType) -> Result<TracedTensor> {
    scalar_real(ctx, dtype, 0.0)
}

fn one_scalar(ctx: &TraceContext, dtype: DType) -> Result<TracedTensor> {
    scalar_real(ctx, dtype, 1.0)
}

fn ones_like(input: &TracedTensor) -> Result<TracedTensor> {
    let shape = input.concrete_shape()?;
    broadcast_scalar(one_scalar(input.context(), input.dtype())?, &shape)
}

fn eye_like(anchor: &TracedTensor, size: usize) -> Result<TracedTensor> {
    let mut vector_shape = vec![size];
    let anchor_shape = anchor.concrete_shape()?;
    vector_shape.extend_from_slice(&anchor_shape[2..]);
    let diagonal = broadcast_scalar(one_scalar(anchor.context(), anchor.dtype())?, &vector_shape)?;
    diagonal.embed_diag(0, 1)
}

fn broadcast_scalar(input: TracedTensor, shape: &[usize]) -> Result<TracedTensor> {
    let input_shape = input.concrete_shape()?;
    if input_shape == shape {
        return Ok(input);
    }
    input.broadcast_in_dim(shape, &[])
}

fn broadcast_batch_scalar_to_leading_axis(
    input: &TracedTensor,
    shape: &[usize],
) -> Result<TracedTensor> {
    let input_shape = input.concrete_shape()?;
    if input_shape == shape {
        return Ok(input.clone());
    }
    let dims: Vec<usize> = (1..shape.len()).collect();
    input.broadcast_in_dim(shape, &dims)
}

fn matmul_preserve_trailing_batch(lhs: &TracedTensor, rhs: &TracedTensor) -> Result<TracedTensor> {
    let rank = lhs.rank();
    let batch_dims: Vec<usize> = (2..rank).collect();
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: batch_dims.clone().iter().copied().collect(),
            rhs_batch_dims: batch_dims.iter().copied().collect(),
        },
    )
}

fn matrix_transpose_perm(rank: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

fn frobenius_norm(abs: &TracedTensor, axes: &[usize]) -> Result<TracedTensor> {
    abs.reduce_sum_squares(Some(axes))?.sqrt()
}

fn p_norm(abs: &TracedTensor, axes: &[usize], p: f64) -> Result<TracedTensor> {
    if !p.is_finite() || p == 0.0 {
        return Err(Error::invalid_argument(
            "norm",
            ErrorPhase::GraphBuild,
            "p",
            format!("p-norm order must be finite and nonzero, got {p}"),
        ));
    }
    if p == 2.0 {
        return frobenius_norm(abs, axes);
    }
    let power = abs.pow(&scalar_real(abs.context(), abs.dtype(), p)?)?;
    let inv_p = scalar_real(abs.context(), abs.dtype(), 1.0 / p)?;
    power.reduce_sum(Some(axes))?.pow(&inv_p)
}

fn reduced_axes_have_zero_extent(shape: &[usize], axes: &[usize]) -> bool {
    axes.iter().any(|&axis| shape[axis] == 0)
}

fn zero_norm_for_empty_reduction(
    ctx: &TraceContext,
    dtype: DType,
    input_shape: &[usize],
    axes: &[usize],
    keepdim: bool,
    ord: Option<f64>,
) -> Result<Option<TracedTensor>> {
    if !empty_reduction_norm_is_zero(axes.len(), ord) {
        return Ok(None);
    }
    let output_shape = reduction_shape(input_shape, axes, keepdim);
    zero_traced_tensor(ctx, real_norm_dtype(dtype)?, output_shape).map(Some)
}

fn empty_reduction_norm_is_zero(axis_count: usize, ord: Option<f64>) -> bool {
    match ord {
        None => true,
        Some(0.0) => true,
        Some(p) if p.is_infinite() => true,
        Some(p) if p.is_finite() && p > 0.0 => axis_count != 2 || p != 2.0,
        _ => false,
    }
}

fn reduction_shape(input_shape: &[usize], axes: &[usize], keepdim: bool) -> Vec<usize> {
    if keepdim {
        let mut shape = input_shape.to_vec();
        for &axis in axes {
            shape[axis] = 1;
        }
        return shape;
    }
    let mut reduced = vec![false; input_shape.len()];
    for &axis in axes {
        reduced[axis] = true;
    }
    input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (!reduced[axis]).then_some(dim))
        .collect()
}

fn real_norm_dtype(dtype: DType) -> Result<DType> {
    match dtype {
        DType::F32 | DType::F64 => Ok(dtype),
        DType::C32 => Ok(DType::F32),
        DType::C64 => Ok(DType::F64),
        _ => Err(Error::TensorRuntime(
            tenferro_tensor::Error::unsupported_dtype(
                "norm",
                dtype,
                "norm supports only floating-point and complex dtypes",
            ),
        )),
    }
}

fn zero_traced_tensor(ctx: &TraceContext, dtype: DType, shape: Vec<usize>) -> Result<TracedTensor> {
    let len = checked_element_count("norm", &shape)?;
    match dtype {
        DType::F32 => ctx.from_vec_col_major(shape, vec![0.0_f32; len]),
        DType::F64 => ctx.from_vec_col_major(shape, vec![0.0_f64; len]),
        DType::C32 => ctx.from_vec_col_major(shape, vec![Complex32::new(0.0, 0.0); len]),
        DType::C64 => ctx.from_vec_col_major(shape, vec![Complex64::new(0.0, 0.0); len]),
        _ => Err(Error::TensorRuntime(
            tenferro_tensor::Error::unsupported_dtype(
                "norm",
                dtype,
                "norm supports only floating-point and complex dtypes",
            ),
        )),
    }
}

fn checked_element_count(op: &'static str, shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
                op,
                "shape",
                "shape element count overflow",
            ))
        })
    })
}

fn default_pinv_rtol(dtype: DType, max_dim: usize) -> f64 {
    let eps = match dtype {
        DType::F32 | DType::C32 => f32::EPSILON as f64,
        DType::F64 | DType::C64 => f64::EPSILON,
        DType::I32 | DType::I64 | DType::Bool => 0.0,
        // INVARIANT: the decomposition rejects an externally defined scalar before
        // it asks for a tolerance, so this value is never used.
        DType::External(_) => unreachable!("the decomposition validates its dtype first"),
    };
    eps * max_dim as f64
}

fn vector_norm(a: &TracedTensor, axis: usize, ord: Option<f64>) -> Result<TracedTensor> {
    let abs = a.abs()?;
    match ord {
        None => frobenius_norm(&abs, &[axis]),
        Some(0.0) => count_nonzero(&abs, &[axis]),
        Some(p) if p == f64::INFINITY => abs.reduce_max(Some(&[axis])),
        Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(Some(&[axis])),
        Some(p) => p_norm(&abs, &[axis], p),
    }
}

fn matrix_norm(a: &TracedTensor, axes: &[usize], ord: Option<f64>) -> Result<TracedTensor> {
    let matrix = move_axes_to_front(a, axes)?;
    if matches!(ord, Some(2.0) | Some(-2.0)) {
        let singular_values = svd_values(&matrix)?.abs()?;
        return if ord == Some(2.0) {
            singular_values.reduce_max(Some(&[0]))
        } else {
            singular_values.reduce_min(Some(&[0]))
        };
    }

    let abs = matrix.abs()?;
    match ord {
        None => frobenius_norm(&abs, &[0, 1]),
        Some(p) if p == f64::INFINITY => matrix_row_sum_norm(&abs, true),
        Some(p) if p == f64::NEG_INFINITY => matrix_row_sum_norm(&abs, false),
        Some(1.0) => matrix_col_sum_norm(&abs, true),
        Some(-1.0) => matrix_col_sum_norm(&abs, false),
        Some(0.0) => count_nonzero(&abs, &[0, 1]),
        Some(p) => p_norm(&abs, &[0, 1], p),
    }
}

fn svd_values(a: &TracedTensor) -> Result<TracedTensor> {
    let (_u, s, _vt) = three_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Svd {
                derivative_eps: SvdOptions::default().derivative_eps,
                gauge: SvdOptions::default().gauge,
                driver: SvdOptions::default().driver,
            })),
            &[a],
        )?,
        "svd_values",
    )?;
    Ok(s)
}

fn eigh_values(a: &TracedTensor) -> Result<TracedTensor> {
    let (values, _vectors) = two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eigh {
                derivative_eps: EighOptions::default().derivative_eps,
                gauge: EighOptions::default().gauge,
                driver: EighOptions::default().driver,
            })),
            &[a],
        )?,
        "eigh_values",
    )?;
    Ok(values)
}

fn eig_values(a: &TracedTensor) -> Result<TracedTensor> {
    let (values, _vectors) = two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eig {
                input_dtype: a.dtype(),
            })),
            &[a],
        )?,
        "eig_values",
    )?;
    Ok(values)
}

fn scale_matrix_columns(matrix: &TracedTensor, scale: &TracedTensor) -> Result<TracedTensor> {
    let matrix_shape = matrix.concrete_shape()?;
    let scale_shape_input = scale.concrete_shape()?;
    let mut scale_shape = vec![1, scale_shape_input[0]];
    scale_shape.extend_from_slice(&matrix_shape[2..]);
    let dims: Vec<usize> = (0..matrix_shape.len()).collect();
    let scale = scale
        .reshape(&scale_shape)?
        .broadcast_in_dim(&matrix_shape, &dims)?;
    matrix * &scale
}

fn count_nonzero(abs: &TracedTensor, axes: &[usize]) -> Result<TracedTensor> {
    // `ord = 0` counts `x != 0` (#2359); equality with zero, then negated,
    // keeps NaN entries because IEEE 754 makes `NaN > 0` false while
    // `NaN != 0` is true.
    let is_zero = abs.compare(&zero_scalar(abs.context(), abs.dtype())?, CompareDir::Eq)?;
    let is_zero = is_zero.convert(abs.dtype())?;
    let ones = ones_like(abs)?;
    let mask = (&ones - &is_zero)?;
    mask.reduce_sum(Some(axes))
}

fn matrix_row_sum_norm(abs: &TracedTensor, take_max: bool) -> Result<TracedTensor> {
    let row_sums = abs.reduce_sum(Some(&[1]))?;
    if take_max {
        row_sums.reduce_max(Some(&[0]))
    } else {
        row_sums.reduce_min(Some(&[0]))
    }
}

fn matrix_col_sum_norm(abs: &TracedTensor, take_max: bool) -> Result<TracedTensor> {
    let col_sums = abs.reduce_sum(Some(&[0]))?;
    if take_max {
        col_sums.reduce_max(Some(&[0]))
    } else {
        col_sums.reduce_min(Some(&[0]))
    }
}

fn move_axes_to_front(tensor: &TracedTensor, axes: &[usize]) -> Result<TracedTensor> {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        return Ok(tensor.clone());
    }

    let mut selected = vec![false; tensor.rank()];
    for &axis in axes {
        selected[axis] = true;
    }

    let mut perm = Vec::with_capacity(tensor.rank());
    perm.extend_from_slice(axes);
    for (axis, is_selected) in selected.iter().enumerate().take(tensor.rank()) {
        if !*is_selected {
            perm.push(axis);
        }
    }
    tensor.transpose(&perm)
}

fn restore_keepdim(
    reduced: TracedTensor,
    original_shape: &[usize],
    axes: &[usize],
    keepdim: bool,
) -> Result<TracedTensor> {
    if !keepdim {
        return Ok(reduced);
    }
    let mut kept_shape = original_shape.to_vec();
    for &axis in axes {
        kept_shape[axis] = 1;
    }
    reduced.reshape(&kept_shape)
}

#[cfg(test)]
mod tests {
    use super::p_norm;
    use tenferro_runtime::TraceContext;

    #[test]
    fn p_norm_rejects_zero_and_non_finite_orders() {
        let trace = TraceContext::new();
        let x = trace
            .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
            .unwrap();
        let abs = x.abs().unwrap();

        for p in [0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = p_norm(&abs, &[0], p).unwrap_err();
            assert!(
                err.to_string().contains("finite") || err.to_string().contains("nonzero"),
                "expected finite nonzero order error, got {err:?}"
            );
        }
    }
}
