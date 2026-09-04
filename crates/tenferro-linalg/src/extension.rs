use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::with_cpu_exec_session;
use tenferro_extension_macros::define_extension_runtime;
use tenferro_ops::SymDim;
use tenferro_runtime::extension::{ExtensionExecutionContext, ExtensionOp};
use tenferro_tensor::{BackendSession, DType, Error, ErrorKind, Tensor, TensorBackend, TensorRead};

#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::with_cuda_exec_session;

use crate::backend::LinalgBackend;
use crate::RankRevealingQrOptions;

mod gauge;
#[cfg(all(test, not(feature = "cuda")))]
mod tests;

pub(crate) use gauge::{apply_eigh_gauge, apply_qr_gauge};

pub const LINALG_EXTENSION_FAMILY_ID: &str = "tenferro-linalg.linalg.v1";

/// Default derivative regularization used by decomposition AD rules.
///
/// This epsilon is used only when differentiating decomposition formulas with
/// repeated or nearly repeated spectral values. It is not a solver tolerance.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{SvdOptions, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS};
///
/// let options = SvdOptions::default();
/// assert_eq!(options.derivative_eps, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS);
/// ```
pub const DEFAULT_DECOMPOSITION_DERIVATIVE_EPS: f64 = 1e-12;

/// Singular-vector gauge convention used by [`SvdOptions`].
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{SvdGauge, SvdOptions};
///
/// let options = SvdOptions::default().gauge(SvdGauge::CanonicalPivot);
/// assert_eq!(options.gauge, SvdGauge::CanonicalPivot);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SvdGauge {
    /// Leave the backend's raw singular vector signs or phases unchanged.
    Raw,
    /// Make each left singular vector's max-absolute pivot entry positive-real
    /// and adjust the matching `VT` row so reconstruction is preserved.
    CanonicalPivot,
}

/// Eigenvector gauge convention used by [`EighOptions`].
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{EighGauge, EighOptions};
///
/// let options = EighOptions::default().gauge(EighGauge::CanonicalPivot);
/// assert_eq!(options.gauge, EighGauge::CanonicalPivot);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EighGauge {
    /// Leave the backend's raw eigenvector signs or phases unchanged.
    Raw,
    /// Make each eigenvector's max-absolute pivot entry positive-real.
    CanonicalPivot,
}

/// QR factor gauge convention used by [`QrOptions`].
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{QrGauge, QrOptions};
///
/// let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
/// assert_eq!(options.gauge, QrGauge::PositiveDiagonal);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QrGauge {
    /// Leave the backend's raw QR signs or phases unchanged.
    Raw,
    /// Make each `R` diagonal entry positive-real, compensating `Q`.
    PositiveDiagonal,
}

/// Options for singular value decomposition.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{SvdGauge, SvdOptions};
///
/// let options = SvdOptions::default()
///     .gauge(SvdGauge::CanonicalPivot)
///     .derivative_eps(1.0e-10);
/// assert_eq!(options.gauge, SvdGauge::CanonicalPivot);
/// assert_eq!(options.derivative_eps, 1.0e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvdOptions {
    /// Singular-vector gauge convention.
    pub gauge: SvdGauge,
    /// AD derivative regularization for repeated or nearly repeated singular values.
    pub derivative_eps: f64,
}

impl Default for SvdOptions {
    fn default() -> Self {
        Self {
            gauge: SvdGauge::Raw,
            derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
        }
    }
}

impl SvdOptions {
    /// Return options with the requested singular-vector gauge.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{SvdGauge, SvdOptions};
    ///
    /// let options = SvdOptions::default().gauge(SvdGauge::CanonicalPivot);
    /// assert_eq!(options.gauge, SvdGauge::CanonicalPivot);
    /// ```
    pub fn gauge(mut self, gauge: SvdGauge) -> Self {
        self.gauge = gauge;
        self
    }

    /// Return options with an explicit derivative epsilon.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::SvdOptions;
    ///
    /// let options = SvdOptions::default().derivative_eps(1.0e-9);
    /// assert_eq!(options.derivative_eps, 1.0e-9);
    /// ```
    pub fn derivative_eps(mut self, derivative_eps: f64) -> Self {
        self.derivative_eps = derivative_eps;
        self
    }
}

/// Options for Hermitian eigenvalue decomposition.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::EighOptions;
///
/// let options = EighOptions::default().derivative_eps(1.0e-10);
/// assert_eq!(options.derivative_eps, 1.0e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EighOptions {
    /// Eigenvector gauge convention.
    pub gauge: EighGauge,
    /// AD derivative regularization for repeated or nearly repeated eigenvalues.
    pub derivative_eps: f64,
}

impl Default for EighOptions {
    fn default() -> Self {
        Self {
            gauge: EighGauge::Raw,
            derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
        }
    }
}

impl EighOptions {
    /// Return options with the requested eigenvector gauge.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{EighGauge, EighOptions};
    ///
    /// let options = EighOptions::default().gauge(EighGauge::CanonicalPivot);
    /// assert_eq!(options.gauge, EighGauge::CanonicalPivot);
    /// ```
    pub fn gauge(mut self, gauge: EighGauge) -> Self {
        self.gauge = gauge;
        self
    }

    /// Return options with an explicit derivative epsilon.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::EighOptions;
    ///
    /// let options = EighOptions::default().derivative_eps(1.0e-9);
    /// assert_eq!(options.derivative_eps, 1.0e-9);
    /// ```
    pub fn derivative_eps(mut self, derivative_eps: f64) -> Self {
        self.derivative_eps = derivative_eps;
        self
    }
}

/// Options for QR decomposition.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{QrGauge, QrOptions};
///
/// let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
/// assert_eq!(options.gauge, QrGauge::PositiveDiagonal);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QrOptions {
    /// QR sign or phase convention.
    pub gauge: QrGauge,
}

impl Default for QrOptions {
    fn default() -> Self {
        Self {
            gauge: QrGauge::Raw,
        }
    }
}

impl QrOptions {
    /// Return options with the requested QR gauge.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::{QrGauge, QrOptions};
    ///
    /// let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    /// assert_eq!(options.gauge, QrGauge::PositiveDiagonal);
    /// ```
    pub fn gauge(mut self, gauge: QrGauge) -> Self {
        self.gauge = gauge;
        self
    }
}

pub(crate) fn validate_derivative_eps(
    op: &'static str,
    derivative_eps: f64,
) -> tenferro_tensor::Result<()> {
    if derivative_eps.is_finite() && derivative_eps > 0.0 {
        Ok(())
    } else {
        Err(Error::invalid_argument(
            op,
            "derivative_eps",
            format!("must be positive and finite, got {derivative_eps}"),
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
#[allow(dead_code)]
pub(crate) enum LinalgOp {
    Cholesky,
    Lu,
    LuFactor,
    LuSolvePrepared {
        transpose_a: bool,
        conjugate_a: bool,
    },
    SignDetFromLuFactor,
    LogAbsDetFromLuFactor,
    FullPivLu,
    FullPivLuSolve {
        transpose_a: bool,
    },
    /// Solve `a @ x = b` with partial-pivot LU (same kernel as
    /// `LinalgBackend::solve`). Two inputs (matrix, rhs) to one output.
    /// Only the eager surface (autodiff feature) constructs this variant;
    /// the traced `solve` composite stays LuFactor + LuSolvePrepared.
    #[cfg_attr(not(feature = "autodiff"), allow(dead_code))]
    Solve,
    Svd {
        derivative_eps: f64,
        gauge: SvdGauge,
    },
    /// Full-matrices SVD: `U` is `m x m` and `Vh` is `n x n`, so the trailing
    /// `Vh` rows span the input's right nullspace. Value-only: AD is
    /// intentionally unsupported (see the linalg AD support manifest).
    SvdFull,
    SvdVals {
        derivative_eps: f64,
    },
    Qr {
        gauge: QrGauge,
    },
    RankRevealingQr {
        gauge: QrGauge,
        rtol: f64,
        atol: f64,
    },
    HouseholderQrFactor,
    HouseholderQrFromFactors,
    HouseholderQrAppend,
    HouseholderQrR {
        gauge: QrGauge,
    },
    HouseholderQrQColumns {
        start: usize,
        end: usize,
        gauge: QrGauge,
    },
    /// Internal AD residual operation for symbolic full thin-Q recovery.
    HouseholderQrThinQ {
        gauge: QrGauge,
    },
    /// Internal linear operation for abstract-state column append.
    HouseholderQrAppendTangent,
    /// Internal transpose operation that splits an appended state cotangent.
    HouseholderQrSplitTangent {
        right: bool,
    },
    Eigh {
        derivative_eps: f64,
        gauge: EighGauge,
    },
    EighVals {
        derivative_eps: f64,
    },
    Eig {
        input_dtype: DType,
    },
    EigVals {
        input_dtype: DType,
    },
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    },
}

impl LinalgOp {
    fn output_count(self) -> usize {
        match self {
            Self::Cholesky
            | Self::EighVals { .. }
            | Self::EigVals { .. }
            | Self::FullPivLuSolve { .. }
            | Self::LogAbsDetFromLuFactor
            | Self::LuSolvePrepared { .. }
            | Self::SignDetFromLuFactor
            | Self::Solve
            | Self::SvdVals { .. }
            | Self::TriangularSolve { .. } => 1,
            Self::Svd { .. } | Self::SvdFull => 3,
            Self::RankRevealingQr { .. } | Self::Lu => 4,
            Self::Qr { .. }
            | Self::HouseholderQrFactor
            | Self::HouseholderQrFromFactors
            | Self::HouseholderQrAppend
            | Self::Eigh { .. }
            | Self::Eig { .. } => 2,
            Self::HouseholderQrR { .. }
            | Self::HouseholderQrQColumns { .. }
            | Self::HouseholderQrThinQ { .. }
            | Self::HouseholderQrAppendTangent
            | Self::HouseholderQrSplitTangent { .. } => 1,
            Self::LuFactor => 3,
            Self::FullPivLu => 5,
        }
    }

    fn input_count(self) -> usize {
        match self {
            Self::FullPivLuSolve { .. }
            | Self::Solve
            | Self::TriangularSolve { .. }
            | Self::HouseholderQrFromFactors
            | Self::HouseholderQrR { .. }
            | Self::HouseholderQrQColumns { .. }
            | Self::HouseholderQrThinQ { .. } => 2,
            Self::LogAbsDetFromLuFactor => 2,
            Self::SignDetFromLuFactor | Self::HouseholderQrAppend => 3,
            Self::HouseholderQrAppendTangent => 4,
            Self::HouseholderQrSplitTangent { .. } => 3,
            Self::LuSolvePrepared { .. } => 4,
            _ => 1,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Cholesky => 0,
            Self::Lu => 1,
            Self::FullPivLu => 2,
            Self::FullPivLuSolve { .. } => 3,
            Self::Svd { .. } => 4,
            Self::Qr { .. } => 5,
            Self::Eigh { .. } => 6,
            Self::Eig { .. } => 7,
            Self::TriangularSolve { .. } => 9,
            Self::LuFactor => 10,
            Self::LuSolvePrepared { .. } => 11,
            Self::SvdVals { .. } => 12,
            Self::EighVals { .. } => 13,
            Self::EigVals { .. } => 14,
            Self::SvdFull => 15,
            Self::LogAbsDetFromLuFactor => 16,
            Self::SignDetFromLuFactor => 17,
            Self::Solve => 18,
            Self::HouseholderQrFactor => 19,
            Self::HouseholderQrFromFactors => 20,
            Self::HouseholderQrAppend => 21,
            Self::HouseholderQrR { .. } => 22,
            Self::HouseholderQrQColumns { .. } => 23,
            Self::HouseholderQrThinQ { .. } => 24,
            Self::HouseholderQrAppendTangent => 25,
            Self::HouseholderQrSplitTangent { .. } => 26,
            Self::RankRevealingQr { .. } => 27,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub(crate) struct LinalgExtensionOp {
    op: LinalgOp,
}

impl LinalgExtensionOp {
    pub(crate) fn new(op: LinalgOp) -> Self {
        Self { op }
    }

    pub(crate) fn op(&self) -> LinalgOp {
        self.op
    }
}

impl ExtensionOp for LinalgExtensionOp {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(self.op.tag());
        match self.op {
            LinalgOp::Svd {
                derivative_eps,
                gauge,
            } => {
                hasher.write_u64(derivative_eps.to_bits());
                hash_svd_gauge(hasher, gauge);
            }
            LinalgOp::SvdVals { derivative_eps } | LinalgOp::EighVals { derivative_eps } => {
                hasher.write_u64(derivative_eps.to_bits());
            }
            LinalgOp::Qr { gauge }
            | LinalgOp::HouseholderQrR { gauge }
            | LinalgOp::HouseholderQrThinQ { gauge } => {
                hash_qr_gauge(hasher, gauge);
            }
            LinalgOp::RankRevealingQr { gauge, rtol, atol } => {
                hash_qr_gauge(hasher, gauge);
                hasher.write_u64(rtol.to_bits());
                hasher.write_u64(atol.to_bits());
            }
            LinalgOp::HouseholderQrQColumns { start, end, gauge } => {
                hasher.write_usize(start);
                hasher.write_usize(end);
                hash_qr_gauge(hasher, gauge);
            }
            LinalgOp::Eigh {
                derivative_eps,
                gauge,
            } => {
                hasher.write_u64(derivative_eps.to_bits());
                hash_eigh_gauge(hasher, gauge);
            }
            LinalgOp::Eig { input_dtype } | LinalgOp::EigVals { input_dtype } => {
                hash_dtype(hasher, input_dtype);
            }
            LinalgOp::FullPivLuSolve { transpose_a }
            | LinalgOp::HouseholderQrSplitTangent { right: transpose_a } => {
                hasher.write_u8(u8::from(transpose_a));
            }
            LinalgOp::LuSolvePrepared {
                transpose_a,
                conjugate_a,
            } => {
                hasher.write_u8(u8::from(transpose_a));
                hasher.write_u8(u8::from(conjugate_a));
            }
            LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => {
                hasher.write_u8(u8::from(left_side));
                hasher.write_u8(u8::from(lower));
                hasher.write_u8(u8::from(transpose_a));
                hasher.write_u8(u8::from(unit_diagonal));
            }
            LinalgOp::Cholesky
            | LinalgOp::Lu
            | LinalgOp::LuFactor
            | LinalgOp::LogAbsDetFromLuFactor
            | LinalgOp::SignDetFromLuFactor
            | LinalgOp::FullPivLu
            | LinalgOp::SvdFull
            | LinalgOp::Solve
            | LinalgOp::HouseholderQrFactor
            | LinalgOp::HouseholderQrFromFactors
            | LinalgOp::HouseholderQrAppend
            | LinalgOp::HouseholderQrAppendTangent => {}
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|that| self == that)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        self.op.input_count()
    }

    fn output_count(&self) -> usize {
        self.op.output_count()
    }

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn prune_outputs(&self, live_outputs: &[bool]) -> Option<Arc<dyn ExtensionOp>> {
        match self.op {
            LinalgOp::Svd { derivative_eps, .. } if live_outputs == [false, true, false] => {
                Some(Arc::new(Self::new(LinalgOp::SvdVals { derivative_eps })))
            }
            LinalgOp::Eigh { derivative_eps, .. } if live_outputs == [true, false] => {
                Some(Arc::new(Self::new(LinalgOp::EighVals { derivative_eps })))
            }
            LinalgOp::Eig { input_dtype } if live_outputs == [true, false] => {
                Some(Arc::new(Self::new(LinalgOp::EigVals { input_dtype })))
            }
            _ => None,
        }
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtypes = (0..self.input_count())
            .map(|input| ctx.input_dtype(input))
            .collect::<Result<Vec<_>, _>>()?;
        let input_shapes = (0..self.input_count())
            .map(|input| ctx.input_shape(input))
            .collect::<Result<Vec<_>, _>>()?;
        let metas = match self.op {
            LinalgOp::Cholesky => {
                require_matrix_meta("tenferro-linalg.cholesky", input_shapes[0])?;
                vec![(promote_dtypes(&input_dtypes), input_shapes[0].to_vec())]
            }
            LinalgOp::FullPivLuSolve { .. } => {
                require_matrix_meta("tenferro-linalg.full_piv_lu_solve", input_shapes[0])?;
                require_matrix_meta("tenferro-linalg.full_piv_lu_solve", input_shapes[1])?;
                vec![(promote_dtypes(&input_dtypes), input_shapes[1].to_vec())]
            }
            LinalgOp::Solve => {
                require_matrix_meta("tenferro-linalg.solve", input_shapes[0])?;
                require_matrix_meta("tenferro-linalg.solve", input_shapes[1])?;
                vec![(promote_dtypes(&input_dtypes), input_shapes[1].to_vec())]
            }
            LinalgOp::TriangularSolve { .. } => {
                require_matrix_meta("tenferro-linalg.triangular_solve", input_shapes[0])?;
                require_matrix_meta("tenferro-linalg.triangular_solve", input_shapes[1])?;
                vec![(promote_dtypes(&input_dtypes), input_shapes[1].to_vec())]
            }
            LinalgOp::LuSolvePrepared { .. } => {
                require_matrix_meta("tenferro-linalg.lu_solve_prepared_lu", input_shapes[0])?;
                require_matrix_meta("tenferro-linalg.lu_solve_prepared_rhs", input_shapes[3])?;
                vec![(
                    promote_dtypes(&[input_dtypes[0], input_dtypes[3]]),
                    input_shapes[3].to_vec(),
                )]
            }
            LinalgOp::Lu => lu_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::LuFactor => lu_factor_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::SignDetFromLuFactor => {
                vec![signdet_from_lu_factor_meta(
                    input_dtypes[0],
                    input_shapes[0],
                    input_shapes[1],
                    input_shapes[2],
                )?]
            }
            LinalgOp::LogAbsDetFromLuFactor => {
                vec![logabsdet_from_lu_factor_meta(
                    input_dtypes[0],
                    input_shapes[0],
                    input_shapes[1],
                )?]
            }
            LinalgOp::FullPivLu => full_piv_lu_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::Svd { .. } => svd_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::SvdFull => svd_full_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::SvdVals { .. } => {
                vec![svd_values_meta(input_dtypes[0], input_shapes[0])?]
            }
            LinalgOp::Qr { .. } => qr_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::RankRevealingQr { .. } => {
                rank_revealing_qr_meta(input_dtypes[0], input_shapes[0])?
            }
            LinalgOp::HouseholderQrFactor => {
                householder_qr_factor_meta(input_dtypes[0], input_shapes[0])?
            }
            LinalgOp::HouseholderQrFromFactors => {
                householder_qr_from_factors_meta(&input_dtypes, &input_shapes)?
            }
            LinalgOp::HouseholderQrAppend => {
                householder_qr_append_meta(&input_dtypes, &input_shapes)?
            }
            LinalgOp::HouseholderQrR { .. } => vec![householder_qr_r_meta(
                &input_dtypes,
                input_shapes[0],
                input_shapes[1],
            )?],
            LinalgOp::HouseholderQrQColumns { start, end, .. } => {
                vec![householder_qr_q_columns_meta(
                    &input_dtypes,
                    input_shapes[0],
                    input_shapes[1],
                    start,
                    end,
                )?]
            }
            LinalgOp::HouseholderQrThinQ { .. } => {
                vec![householder_qr_thin_q_meta(
                    &input_dtypes,
                    input_shapes[0],
                    input_shapes[1],
                )?]
            }
            LinalgOp::HouseholderQrAppendTangent => {
                vec![householder_qr_append_tangent_meta(
                    &input_dtypes,
                    &input_shapes,
                )?]
            }
            LinalgOp::HouseholderQrSplitTangent { right } => {
                vec![householder_qr_split_tangent_meta(
                    &input_dtypes,
                    &input_shapes,
                    right,
                )?]
            }
            LinalgOp::Eigh { .. } => eigh_meta(input_dtypes[0], input_shapes[0])?,
            LinalgOp::EighVals { .. } => vec![eigh_values_meta(input_dtypes[0], input_shapes[0])?],
            LinalgOp::Eig { input_dtype } => eig_meta(input_dtype, input_shapes[0])?,
            LinalgOp::EigVals { input_dtype } => {
                vec![eig_values_meta(input_dtype, input_shapes[0])?]
            }
        };
        Ok(metas)
    }
}

pub(crate) fn execute_linalg_extension_reads<B: BackendSession + ?Sized>(
    op: &LinalgExtensionOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    execute_linalg_extension_reads_on_session(op, inputs, ctx.backend_mut())
}

pub(crate) fn execute_linalg_extension_reads_owner<B: TensorBackend>(
    op: &LinalgExtensionOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let (backend, caches) = ctx.parts_mut();
    backend.with_backend_session(|session| {
        let mut session_ctx = ExtensionExecutionContext::new(session, caches);
        execute_linalg_extension_reads(op, inputs, &mut session_ctx)
    })
}

fn execute_linalg_extension_reads_on_session<B: BackendSession + ?Sized>(
    op: &LinalgExtensionOp,
    inputs: &[TensorRead<'_>],
    session: &mut B,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if let Some(result) = with_cpu_exec_session(session, |session| {
        execute_linalg_extension_reads_in_session(op, inputs, session)
    }) {
        return result;
    }
    #[cfg(feature = "cuda")]
    if let Some(result) = with_cuda_exec_session(session, |session| {
        execute_linalg_extension_reads_in_session(op, inputs, session)
    }) {
        return result;
    }
    Err(Error::unsupported(
        "linalg_extension",
        "selected backend session does not expose a linalg execution capability",
    ))
}

fn execute_linalg_extension_reads_in_session<S: LinalgBackend>(
    op: &LinalgExtensionOp,
    inputs: &[TensorRead<'_>],
    session: &mut S,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if op.op() == LinalgOp::HouseholderQrAppendTangent {
        let left = session.to_contiguous_read(inputs[0].clone())?;
        let right = session.to_contiguous_read(inputs[1].clone())?;
        return Ok(vec![session.concatenate(&[&left, &right], 1)?]);
    }
    if let LinalgOp::HouseholderQrSplitTangent { right } = op.op() {
        let cotangent_shape = inputs[0].clone().tensor_view().shape().to_vec();
        let left_shape = inputs[1].clone().tensor_view().shape().to_vec();
        let right_shape = inputs[2].clone().tensor_view().shape().to_vec();
        let config =
            householder_qr_split_config(&cotangent_shape, &left_shape, &right_shape, right)?;
        let cotangent = session.to_contiguous_read(inputs[0].clone())?;
        return Ok(vec![session.slice(&cotangent, &config)?]);
    }
    if op.op() == LinalgOp::Cholesky {
        return Ok(vec![session.cholesky_read(inputs[0].clone())?]);
    }
    if let LinalgOp::TriangularSolve {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    } = op.op()
    {
        match session.triangular_solve_read(
            inputs[0].clone(),
            inputs[1].clone(),
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        ) {
            Ok(output) => return Ok(vec![output]),
            Err(error) if error.kind() == ErrorKind::Unsupported => {}
            Err(error) => return Err(error),
        }
    }

    // Linalg kernels operate on compact tensors; materialization is explicit
    // here so borrowed views cannot bypass provider errors.
    let materialized_inputs = inputs
        .iter()
        .cloned()
        .map(|input| session.to_contiguous_read(input))
        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
    let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
    execute_linalg(op.op(), &input_refs, session)
}

fn linalg_session_supported<B: BackendSession + 'static>(op: &LinalgExtensionOp) -> bool {
    // The `supports_session` contract (capability.rs) requires that an op is
    // admitted to a scheduler session only when the session executor genuinely
    // executes it without returning `Unsupported`. Admission is exactly
    // per-op/per-backend so `apply_eager` keeps the native prepared path for
    // every op the session can actually run (issue #1665).
    let type_id = std::any::TypeId::of::<B>();
    if type_id == std::any::TypeId::of::<tenferro_cpu::CpuBackend>() {
        // The CPU backend type does not carry its provider kind (faer vs BLAS)
        // at this type-only seam, and the BLAS provider does not implement
        // in-session full-matrices SVD, so SvdFull is conservatively rejected
        // and falls back to the compiled path. Every other CPU linalg kernel
        // runs in-session on both faer and BLAS providers.
        return op.op() != LinalgOp::SvdFull;
    }
    #[cfg(feature = "cuda")]
    {
        if type_id == std::any::TypeId::of::<tenferro_gpu::cuda::CudaBackend>() {
            return match op.op() {
                // Complete-pivoting LU and general eig have no CUDA kernels.
                LinalgOp::FullPivLu | LinalgOp::FullPivLuSolve { .. } => false,
                LinalgOp::Eig { .. } | LinalgOp::EigVals { .. } => false,
                // Plain partial-pivot solve runs in-session via cuSOLVER
                // getrf plus prepared pivot/triangular solves
                // (`gpu/linalg.rs::solve` = lu_factor + lu_solve_prepared, no
                // Unsupported path for F32/F64/C32/C64), so it is admitted.
                LinalgOp::Solve => true,
                // Full-matrices SVD falls back to the default `svd_full`
                // impl, which reports `Unsupported`.
                LinalgOp::SvdFull | LinalgOp::RankRevealingQr { .. } => false,
                // Conjugate-only prepared LU solve is unsupported on CUDA.
                LinalgOp::LuSolvePrepared {
                    transpose_a: false,
                    conjugate_a: true,
                } => false,
                _ => true,
            };
        }
    }
    false
}

fn execute_linalg_extension_in_session(
    op: &LinalgExtensionOp,
    session: &mut dyn BackendSession,
    _extension_caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // Reuse the existing session executor that the eager and scheduler paths
    // already share; it downcasts the borrowed session to the CPU/CUDA exec
    // session and runs the same forward kernel for every LinalgOp.
    execute_linalg_extension_reads_on_session(op, inputs, session)
}

define_extension_runtime! {
    runtime = LinalgRuntime,
    family_id = LINALG_EXTENSION_FAMILY_ID,
    op_type = LinalgExtensionOp,
    execute = execute_linalg_extension_reads_owner,
    execute_reads = execute_linalg_extension_reads_owner,
    execute_in_session = execute_linalg_extension_in_session,
    session_supported = linalg_session_supported,
    backend_bound = TensorBackend,
}

fn execute_linalg<B: LinalgBackend>(
    op: LinalgOp,
    inputs: &[&Tensor],
    backend: &mut B,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match op {
        LinalgOp::Cholesky => Ok(vec![backend.cholesky(inputs[0])?]),
        LinalgOp::Lu => backend.lu(inputs[0]),
        LinalgOp::LuFactor => backend.lu_factor(inputs[0]),
        LinalgOp::SignDetFromLuFactor => Ok(vec![signdet_from_lu_factor(
            inputs[0].dtype(),
            inputs[1],
            inputs[2],
            backend,
        )?]),
        LinalgOp::LogAbsDetFromLuFactor => Ok(vec![logabsdet_from_lu_factor(inputs[1], backend)?]),
        LinalgOp::LuSolvePrepared {
            transpose_a,
            conjugate_a,
        } => Ok(vec![backend.lu_solve_prepared(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            transpose_a,
            conjugate_a,
        )?]),
        LinalgOp::FullPivLu => backend.full_piv_lu(inputs[0]),
        LinalgOp::FullPivLuSolve { transpose_a } => Ok(vec![backend.full_piv_lu_solve(
            inputs[0],
            inputs[1],
            transpose_a,
        )?]),
        LinalgOp::Solve => Ok(vec![backend.solve(inputs[0], inputs[1])?]),
        LinalgOp::Svd {
            derivative_eps,
            gauge,
        } => backend.svd_with_options(
            inputs[0],
            SvdOptions {
                derivative_eps,
                gauge,
            },
        ),
        LinalgOp::SvdFull => backend.svd_full(inputs[0]),
        LinalgOp::SvdVals { .. } => Ok(vec![backend.svd_values(inputs[0])?]),
        LinalgOp::Qr { gauge } => backend.qr_with_options(inputs[0], QrOptions { gauge }),
        LinalgOp::RankRevealingQr { gauge, rtol, atol } => {
            backend.rank_revealing_qr(inputs[0], RankRevealingQrOptions { gauge, rtol, atol })
        }
        LinalgOp::HouseholderQrFactor => {
            let state = backend.householder_qr(inputs[0])?;
            Ok(vec![state.packed, state.coeff])
        }
        LinalgOp::HouseholderQrFromFactors => {
            let state = backend.householder_qr_from_factors(inputs[0], inputs[1])?;
            Ok(vec![state.packed, state.coeff])
        }
        LinalgOp::HouseholderQrAppend => {
            let state = backend.householder_qr_append(inputs[0], inputs[1], inputs[2])?;
            Ok(vec![state.packed, state.coeff])
        }
        LinalgOp::HouseholderQrR { gauge } => Ok(vec![backend.householder_qr_r(
            inputs[0],
            inputs[1],
            QrOptions { gauge },
        )?]),
        LinalgOp::HouseholderQrQColumns { start, end, gauge } => Ok(vec![backend
            .householder_qr_q_columns(inputs[0], inputs[1], start..end, QrOptions { gauge })?]),
        LinalgOp::HouseholderQrThinQ { gauge } => {
            let end = inputs[1].shape().first().copied().ok_or_else(|| {
                Error::rank_mismatch("tenferro-linalg.householder_qr_thin_q", 1, 0)
            })?;
            Ok(vec![backend.householder_qr_q_columns(
                inputs[0],
                inputs[1],
                0..end,
                QrOptions { gauge },
            )?])
        }
        LinalgOp::HouseholderQrAppendTangent => {
            Ok(vec![backend.concatenate(&[inputs[0], inputs[1]], 1)?])
        }
        LinalgOp::HouseholderQrSplitTangent { right } => {
            let config = householder_qr_split_config(
                inputs[0].shape(),
                inputs[1].shape(),
                inputs[2].shape(),
                right,
            )?;
            Ok(vec![backend.slice(inputs[0], &config)?])
        }
        LinalgOp::Eigh {
            derivative_eps,
            gauge,
        } => backend.eigh_with_options(
            inputs[0],
            EighOptions {
                derivative_eps,
                gauge,
            },
        ),
        LinalgOp::EighVals { .. } => Ok(vec![backend.eigh_values(inputs[0])?]),
        LinalgOp::Eig { .. } => backend.eig(inputs[0]),
        LinalgOp::EigVals { .. } => Ok(vec![backend.eig_values(inputs[0])?]),
        LinalgOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        } => Ok(vec![backend.triangular_solve(
            inputs[0],
            inputs[1],
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )?]),
    }
}

fn signdet_from_lu_factor<B: LinalgBackend + ?Sized>(
    input_dtype: DType,
    packed_lu: &Tensor,
    parity: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let diag = backend.extract_diagonal(packed_lu, 0, 1)?;
    let det_u = backend.reduce_prod_read(TensorRead::from_tensor(&diag), &[0])?;
    let det = backend.mul_read(
        TensorRead::from_tensor(parity),
        TensorRead::from_tensor(&det_u),
    )?;
    if matches!(input_dtype, DType::C32 | DType::C64) {
        let abs = backend.abs_read(TensorRead::from_tensor(&det))?;
        let abs = backend.convert(&abs, input_dtype)?;
        backend.div_read(TensorRead::from_tensor(&det), TensorRead::from_tensor(&abs))
    } else {
        backend.sign_read(TensorRead::from_tensor(&det))
    }
}

fn logabsdet_from_lu_factor<B: LinalgBackend + ?Sized>(
    packed_lu: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let diag = backend.extract_diagonal(packed_lu, 0, 1)?;
    let abs = backend.abs_read(TensorRead::from_tensor(&diag))?;
    let log = backend.log_read(TensorRead::from_tensor(&abs))?;
    backend.reduce_sum_read(TensorRead::from_tensor(&log), &[0])
}

pub(crate) fn apply_svd_gauge(
    gauge: SvdGauge,
    outputs: &mut [Tensor],
) -> tenferro_tensor::Result<()> {
    match gauge {
        SvdGauge::Raw => Ok(()),
        SvdGauge::CanonicalPivot => apply_canonical_pivot_svd_gauge(outputs),
    }
}

fn apply_canonical_pivot_svd_gauge(outputs: &mut [Tensor]) -> tenferro_tensor::Result<()> {
    if outputs.len() != 3 {
        return Err(Error::invalid_argument(
            "tenferro-linalg.svd",
            "outputs",
            format!(
                "canonical SVD gauge expected three outputs, got {}",
                outputs.len()
            ),
        ));
    }

    let (u_slice, rest) = outputs.split_at_mut(1);
    let (singular_slice, vt_slice) = rest.split_at_mut(1);
    let u = &mut u_slice[0];
    let singular_values = &singular_slice[0];
    let vt = &mut vt_slice[0];
    let u_shape = u.shape().to_vec();
    let s_shape = singular_values.shape().to_vec();
    let vt_shape = vt.shape().to_vec();
    if u_shape.len() < 2 || vt_shape.len() < 2 || s_shape.is_empty() {
        return Err(Error::invalid_argument(
            "tenferro-linalg.svd",
            "outputs",
            format!(
                "canonical SVD gauge expected U rank >= 2, S rank >= 1, VT rank >= 2; got U={u_shape:?}, S={s_shape:?}, VT={vt_shape:?}"
            ),
        ));
    }

    let m = u_shape[0];
    let k = u_shape[1];
    let n = vt_shape[1];
    if s_shape[0] != k
        || vt_shape[0] != k
        || u_shape[2..] != vt_shape[2..]
        || s_shape[1..] != u_shape[2..]
    {
        return Err(Error::invalid_argument(
            "tenferro-linalg.svd",
            "outputs",
            format!(
                "canonical SVD gauge expected compatible compact SVD shapes, got U={u_shape:?}, S={s_shape:?}, VT={vt_shape:?}"
            ),
        ));
    }
    let layout = canonical_svd_gauge_layout(m, k, n, &u_shape[2..])?;

    match (u, vt) {
        (Tensor::F64(u), Tensor::F64(vt)) => {
            canonicalize_svd_gauge_f64(u.host_data_mut()?, vt.host_data_mut()?, layout)
        }
        (Tensor::F32(u), Tensor::F32(vt)) => {
            canonicalize_svd_gauge_f32(u.host_data_mut()?, vt.host_data_mut()?, layout)
        }
        (Tensor::C64(u), Tensor::C64(vt)) => {
            canonicalize_svd_gauge_c64(u.host_data_mut()?, vt.host_data_mut()?, layout)
        }
        (Tensor::C32(u), Tensor::C32(vt)) => {
            canonicalize_svd_gauge_c32(u.host_data_mut()?, vt.host_data_mut()?, layout)
        }
        (u, vt) => Err(Error::dtype_mismatch(
            "tenferro-linalg.svd",
            u.dtype(),
            vt.dtype(),
        )),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanonicalSvdGaugeLayout {
    m: usize,
    k: usize,
    batch_count: usize,
    u_batch_len: usize,
    vt_batch_len: usize,
    u_len: usize,
    vt_len: usize,
}

impl CanonicalSvdGaugeLayout {
    fn validate_storage(self, u_len: usize, vt_len: usize) -> tenferro_tensor::Result<()> {
        if u_len != self.u_len {
            return Err(Error::invalid_argument(
                "tenferro-linalg.svd",
                "U storage",
                format!(
                    "canonical SVD gauge expected U storage length {}, got {u_len}",
                    self.u_len
                ),
            ));
        }
        if vt_len != self.vt_len {
            return Err(Error::invalid_argument(
                "tenferro-linalg.svd",
                "VT storage",
                format!(
                    "canonical SVD gauge expected VT storage length {}, got {vt_len}",
                    self.vt_len
                ),
            ));
        }
        Ok(())
    }
}

fn canonical_svd_gauge_layout(
    m: usize,
    k: usize,
    n: usize,
    batch_shape: &[usize],
) -> tenferro_tensor::Result<CanonicalSvdGaugeLayout> {
    let batch_count = tenferro_tensor::validate::checked_shape_product(
        "tenferro-linalg.svd",
        "canonical SVD batch",
        batch_shape,
    )?;
    let u_batch_len = tenferro_tensor::validate::checked_shape_product(
        "tenferro-linalg.svd",
        "canonical SVD U batch",
        &[m, k],
    )?;
    let vt_batch_len = tenferro_tensor::validate::checked_shape_product(
        "tenferro-linalg.svd",
        "canonical SVD VT batch",
        &[k, n],
    )?;
    let u_len = tenferro_tensor::validate::checked_shape_product(
        "tenferro-linalg.svd",
        "canonical SVD U storage",
        &[u_batch_len, batch_count],
    )?;
    let vt_len = tenferro_tensor::validate::checked_shape_product(
        "tenferro-linalg.svd",
        "canonical SVD VT storage",
        &[vt_batch_len, batch_count],
    )?;
    Ok(CanonicalSvdGaugeLayout {
        m,
        k,
        batch_count,
        u_batch_len,
        vt_batch_len,
        u_len,
        vt_len,
    })
}

fn canonicalize_svd_gauge_f64(
    u: &mut [f64],
    vt: &mut [f64],
    layout: CanonicalSvdGaugeLayout,
) -> tenferro_tensor::Result<()> {
    layout.validate_storage(u.len(), vt.len())?;
    if layout.batch_count == 0 || layout.u_batch_len == 0 || layout.vt_batch_len == 0 {
        return Ok(());
    }
    for (u_batch, vt_batch) in u
        .chunks_exact_mut(layout.u_batch_len)
        .zip(vt.chunks_exact_mut(layout.vt_batch_len))
    {
        for (col, u_column) in u_batch.chunks_exact_mut(layout.m).enumerate() {
            let pivot = max_abs_pivot_f64(u_column);
            let pivot_value = u_column[pivot];
            if pivot_value < 0.0 {
                for value in u_column {
                    *value = -*value;
                }
                for vt_column in vt_batch.chunks_exact_mut(layout.k) {
                    vt_column[col] = -vt_column[col];
                }
            }
        }
    }
    Ok(())
}

fn canonicalize_svd_gauge_f32(
    u: &mut [f32],
    vt: &mut [f32],
    layout: CanonicalSvdGaugeLayout,
) -> tenferro_tensor::Result<()> {
    layout.validate_storage(u.len(), vt.len())?;
    if layout.batch_count == 0 || layout.u_batch_len == 0 || layout.vt_batch_len == 0 {
        return Ok(());
    }
    for (u_batch, vt_batch) in u
        .chunks_exact_mut(layout.u_batch_len)
        .zip(vt.chunks_exact_mut(layout.vt_batch_len))
    {
        for (col, u_column) in u_batch.chunks_exact_mut(layout.m).enumerate() {
            let pivot = max_abs_pivot_f32(u_column);
            let pivot_value = u_column[pivot];
            if pivot_value < 0.0 {
                for value in u_column {
                    *value = -*value;
                }
                for vt_column in vt_batch.chunks_exact_mut(layout.k) {
                    vt_column[col] = -vt_column[col];
                }
            }
        }
    }
    Ok(())
}

fn canonicalize_svd_gauge_c64(
    u: &mut [Complex64],
    vt: &mut [Complex64],
    layout: CanonicalSvdGaugeLayout,
) -> tenferro_tensor::Result<()> {
    layout.validate_storage(u.len(), vt.len())?;
    if layout.batch_count == 0 || layout.u_batch_len == 0 || layout.vt_batch_len == 0 {
        return Ok(());
    }
    for (u_batch, vt_batch) in u
        .chunks_exact_mut(layout.u_batch_len)
        .zip(vt.chunks_exact_mut(layout.vt_batch_len))
    {
        for (col, u_column) in u_batch.chunks_exact_mut(layout.m).enumerate() {
            let pivot = max_abs_pivot_c64(u_column);
            let pivot_value = u_column[pivot];
            let pivot_norm = pivot_value.norm();
            if pivot_norm == 0.0 {
                continue;
            }
            let phase = pivot_value.conj() / pivot_norm;
            let vt_phase = phase.conj();
            for value in u_column {
                *value *= phase;
            }
            for vt_column in vt_batch.chunks_exact_mut(layout.k) {
                vt_column[col] *= vt_phase;
            }
        }
    }
    Ok(())
}

fn canonicalize_svd_gauge_c32(
    u: &mut [Complex32],
    vt: &mut [Complex32],
    layout: CanonicalSvdGaugeLayout,
) -> tenferro_tensor::Result<()> {
    layout.validate_storage(u.len(), vt.len())?;
    if layout.batch_count == 0 || layout.u_batch_len == 0 || layout.vt_batch_len == 0 {
        return Ok(());
    }
    for (u_batch, vt_batch) in u
        .chunks_exact_mut(layout.u_batch_len)
        .zip(vt.chunks_exact_mut(layout.vt_batch_len))
    {
        for (col, u_column) in u_batch.chunks_exact_mut(layout.m).enumerate() {
            let pivot = max_abs_pivot_c32(u_column);
            let pivot_value = u_column[pivot];
            let pivot_norm = pivot_value.norm();
            if pivot_norm == 0.0 {
                continue;
            }
            let phase = pivot_value.conj() / pivot_norm;
            let vt_phase = phase.conj();
            for value in u_column {
                *value *= phase;
            }
            for vt_column in vt_batch.chunks_exact_mut(layout.k) {
                vt_column[col] *= vt_phase;
            }
        }
    }
    Ok(())
}

fn max_abs_pivot_f64(u_column: &[f64]) -> usize {
    let mut pivot = 0;
    let mut pivot_abs = u_column[0].abs();
    for (row, value) in u_column.iter().enumerate().skip(1) {
        let candidate_abs = value.abs();
        if candidate_abs > pivot_abs {
            pivot = row;
            pivot_abs = candidate_abs;
        }
    }
    pivot
}

fn max_abs_pivot_f32(u_column: &[f32]) -> usize {
    let mut pivot = 0;
    let mut pivot_abs = u_column[0].abs();
    for (row, value) in u_column.iter().enumerate().skip(1) {
        let candidate_abs = value.abs();
        if candidate_abs > pivot_abs {
            pivot = row;
            pivot_abs = candidate_abs;
        }
    }
    pivot
}

fn max_abs_pivot_c64(u_column: &[Complex64]) -> usize {
    let mut pivot = 0;
    let mut pivot_abs = u_column[0].norm_sqr();
    for (row, value) in u_column.iter().enumerate().skip(1) {
        let candidate_abs = value.norm_sqr();
        if candidate_abs > pivot_abs {
            pivot = row;
            pivot_abs = candidate_abs;
        }
    }
    pivot
}

fn max_abs_pivot_c32(u_column: &[Complex32]) -> usize {
    let mut pivot = 0;
    let mut pivot_abs = u_column[0].norm_sqr();
    for (row, value) in u_column.iter().enumerate().skip(1) {
        let candidate_abs = value.norm_sqr();
        if candidate_abs > pivot_abs {
            pivot = row;
            pivot_abs = candidate_abs;
        }
    }
    pivot
}

fn require_matrix_meta(op: &'static str, shape: &[SymDim]) -> tenferro_tensor::Result<()> {
    if shape.len() < 2 {
        return Err(Error::rank_mismatch(op, 2, shape.len()));
    }
    Ok(())
}

fn matrix_meta_parts<'a>(
    op: &'static str,
    shape: &'a [SymDim],
) -> tenferro_tensor::Result<(SymDim, SymDim, &'a [SymDim])> {
    require_matrix_meta(op, shape)?;
    Ok((shape[0].clone(), shape[1].clone(), &shape[2..]))
}

fn lu_meta(dtype: DType, shape: &[SymDim]) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.lu", shape)?;
    let k = m.clone().min(n.clone());
    Ok(vec![
        (dtype, matrix_shape(m.clone(), m, batch)),
        (dtype, matrix_shape(shape[0].clone(), k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
        (dtype, batch.to_vec()),
    ])
}

fn lu_factor_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.lu_factor", shape)?;
    let k = m.min(n);
    Ok(vec![
        (dtype, shape.to_vec()),
        (DType::I32, vector_shape(k, batch)),
        (dtype, batch.to_vec()),
    ])
}

fn signdet_from_lu_factor_meta(
    input_dtype: DType,
    input_shape: &[SymDim],
    packed_shape: &[SymDim],
    parity_shape: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    let (_, _, batch) = matrix_meta_parts("tenferro-linalg.signdet_from_lu_factor", input_shape)?;
    require_matrix_meta(
        "tenferro-linalg.signdet_from_lu_factor_packed",
        packed_shape,
    )?;
    if parity_shape.len() != batch.len() {
        return Err(Error::rank_mismatch(
            "tenferro-linalg.signdet_from_lu_factor_parity",
            batch.len(),
            parity_shape.len(),
        ));
    }
    Ok((input_dtype, batch.to_vec()))
}

fn logabsdet_from_lu_factor_meta(
    input_dtype: DType,
    input_shape: &[SymDim],
    packed_shape: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    let (_, _, batch) = matrix_meta_parts("tenferro-linalg.logabsdet_from_lu_factor", input_shape)?;
    require_matrix_meta(
        "tenferro-linalg.logabsdet_from_lu_factor_packed",
        packed_shape,
    )?;
    Ok((singular_values_dtype(input_dtype), batch.to_vec()))
}

fn full_piv_lu_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.full_piv_lu", shape)?;
    Ok(vec![
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
        (singular_values_dtype(dtype), batch.to_vec()),
    ])
}

fn svd_meta(dtype: DType, shape: &[SymDim]) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.svd", shape)?;
    let k = m.clone().min(n.clone());
    Ok(vec![
        (dtype, matrix_shape(m, k.clone(), batch)),
        (singular_values_dtype(dtype), vector_shape(k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
    ])
}

fn svd_full_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.svd_full", shape)?;
    let k = m.clone().min(n.clone());
    Ok(vec![
        (dtype, matrix_shape(m.clone(), m, batch)),
        (singular_values_dtype(dtype), vector_shape(k, batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
    ])
}

fn svd_values_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.svd_values", shape)?;
    let k = m.min(n);
    Ok((singular_values_dtype(dtype), vector_shape(k, batch)))
}

fn qr_meta(dtype: DType, shape: &[SymDim]) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.qr", shape)?;
    let k = m.clone().min(n.clone());
    Ok(vec![
        (dtype, matrix_shape(m, k.clone(), batch)),
        (dtype, matrix_shape(k, n, batch)),
    ])
}

fn rank_revealing_qr_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (m, n, batch) = matrix_meta_parts("tenferro-linalg.rank_revealing_qr", shape)?;
    let k = m.clone().min(n.clone());
    Ok(vec![
        (dtype, matrix_shape(m, k.clone(), batch)),
        (dtype, matrix_shape(k, n.clone(), batch)),
        (DType::I64, vector_shape(n, batch)),
        (DType::I64, batch.to_vec()),
    ])
}

fn require_householder_rank2(op: &'static str, shape: &[SymDim]) -> tenferro_tensor::Result<()> {
    if shape.len() != 2 {
        return Err(Error::rank_mismatch(op, 2, shape.len()));
    }
    Ok(())
}

fn householder_qr_factor_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    require_householder_rank2("tenferro-linalg.householder_qr", shape)?;
    let k = shape[0].clone().min(shape[1].clone());
    Ok(vec![(dtype, shape.to_vec()), (dtype, vec![k])])
}

fn householder_qr_from_factors_meta(
    dtypes: &[DType],
    shapes: &[&[SymDim]],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    const OP: &str = "tenferro-linalg.householder_qr_from_factors";
    require_householder_rank2(OP, shapes[0])?;
    require_householder_rank2(OP, shapes[1])?;
    if dtypes[0] != dtypes[1] {
        return Err(Error::dtype_mismatch(OP, dtypes[0], dtypes[1]));
    }
    require_static_extent_equal(OP, "q.cols/r.rows", &shapes[0][1], &shapes[1][0])?;
    if let (Some(q_cols), Some(q_rows), Some(r_cols)) = (
        shapes[0][1].constant_value(),
        shapes[0][0].constant_value(),
        shapes[1][1].constant_value(),
    ) {
        if q_cols > q_rows.min(r_cols) {
            return Err(Error::invalid_argument(
                OP,
                "shape",
                "Q column count exceeds min(Q rows, R columns)",
            ));
        }
    }
    let m = shapes[0][0].clone();
    let n = shapes[1][1].clone();
    let k = m.clone().min(n.clone());
    Ok(vec![(dtypes[0], vec![m, n]), (dtypes[0], vec![k])])
}

fn householder_qr_append_meta(
    dtypes: &[DType],
    shapes: &[&[SymDim]],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    const OP: &str = "tenferro-linalg.householder_qr_append";
    require_householder_state_meta(OP, dtypes, shapes[0], shapes[1])?;
    require_householder_rank2(OP, shapes[2])?;
    if dtypes[0] != dtypes[2] {
        return Err(Error::dtype_mismatch(OP, dtypes[0], dtypes[2]));
    }
    require_static_extent_equal(OP, "rows", &shapes[0][0], &shapes[2][0])?;
    let m = shapes[0][0].clone();
    let width = shapes[0][1].clone() + shapes[2][1].clone();
    let k = m.clone().min(width.clone());
    Ok(vec![(dtypes[0], vec![m, width]), (dtypes[0], vec![k])])
}

fn require_static_extent_equal(
    op: &'static str,
    field: &'static str,
    lhs: &SymDim,
    rhs: &SymDim,
) -> tenferro_tensor::Result<()> {
    if let (Some(lhs), Some(rhs)) = (lhs.constant_value(), rhs.constant_value()) {
        if lhs != rhs {
            return Err(Error::invalid_argument(
                op,
                field,
                format!("expected equal extents, got {lhs} and {rhs}"),
            ));
        }
    }
    Ok(())
}

fn require_householder_state_meta(
    op: &'static str,
    dtypes: &[DType],
    packed: &[SymDim],
    coeff: &[SymDim],
) -> tenferro_tensor::Result<()> {
    require_householder_rank2(op, packed)?;
    if coeff.len() != 1 {
        return Err(Error::rank_mismatch(op, 1, coeff.len()));
    }
    if dtypes[0] != dtypes[1] {
        return Err(Error::dtype_mismatch(op, dtypes[0], dtypes[1]));
    }
    let expected = packed[0].clone().min(packed[1].clone());
    require_static_extent_equal(op, "coeff", &coeff[0], &expected)
}

fn householder_qr_r_meta(
    dtypes: &[DType],
    packed: &[SymDim],
    coeff: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    require_householder_state_meta("tenferro-linalg.householder_qr_r", dtypes, packed, coeff)?;
    Ok((dtypes[0], vec![coeff[0].clone(), packed[1].clone()]))
}

fn householder_qr_q_columns_meta(
    dtypes: &[DType],
    packed: &[SymDim],
    coeff: &[SymDim],
    start: usize,
    end: usize,
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    require_householder_state_meta(
        "tenferro-linalg.householder_qr_q_columns",
        dtypes,
        packed,
        coeff,
    )?;
    if start > end {
        return Err(Error::invalid_argument(
            "tenferro-linalg.householder_qr_q_columns",
            "range",
            format!("invalid Q-column range {start}..{end}"),
        ));
    }
    if coeff[0].constant_value().is_some_and(|k| end > k) {
        return Err(Error::invalid_argument(
            "tenferro-linalg.householder_qr_q_columns",
            "range",
            format!("Q-column range {start}..{end} exceeds thin-Q width"),
        ));
    }
    Ok((
        dtypes[0],
        vec![packed[0].clone(), SymDim::from(end - start)],
    ))
}

fn householder_qr_thin_q_meta(
    dtypes: &[DType],
    packed: &[SymDim],
    coeff: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    require_householder_state_meta(
        "tenferro-linalg.householder_qr_thin_q",
        dtypes,
        packed,
        coeff,
    )?;
    Ok((dtypes[0], vec![packed[0].clone(), coeff[0].clone()]))
}

fn householder_qr_split_config(
    cotangent: &[usize],
    left: &[usize],
    right_shape: &[usize],
    take_right: bool,
) -> tenferro_tensor::Result<tenferro_tensor::SliceConfig> {
    const OP: &str = "tenferro-linalg.householder_qr_split_tangent";
    for shape in [cotangent, left, right_shape] {
        if shape.len() != 2 {
            return Err(Error::rank_mismatch(OP, 2, shape.len()));
        }
    }
    let total_width = left[1]
        .checked_add(right_shape[1])
        .ok_or_else(|| Error::invalid_argument(OP, "shape", "column range overflow"))?;
    if cotangent[0] != left[0] || cotangent[0] != right_shape[0] || cotangent[1] != total_width {
        return Err(Error::invalid_argument(
            OP,
            "shape",
            "cotangent shape does not match appended factors",
        ));
    }
    let selected = if take_right { right_shape } else { left };
    let start = if take_right { left[1] } else { 0 };
    let end = start
        .checked_add(selected[1])
        .ok_or_else(|| Error::invalid_argument(OP, "shape", "column range overflow"))?;
    Ok(tenferro_tensor::SliceConfig {
        starts: vec![0, start],
        limits: vec![selected[0], end],
        strides: vec![1, 1],
    })
}

fn householder_qr_append_tangent_meta(
    dtypes: &[DType],
    shapes: &[&[SymDim]],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    const OP: &str = "tenferro-linalg.householder_qr_append_tangent";
    for shape in shapes {
        require_householder_rank2(OP, shape)?;
    }
    if dtypes.iter().any(|dtype| *dtype != dtypes[0]) {
        return Err(Error::dtype_mismatch(OP, dtypes[0], dtypes[1]));
    }
    require_static_extent_equal(OP, "rows", &shapes[0][0], &shapes[1][0])?;
    require_static_extent_equal(OP, "left tangent", &shapes[0][0], &shapes[2][0])?;
    require_static_extent_equal(OP, "right tangent", &shapes[1][0], &shapes[3][0])?;
    require_static_extent_equal(OP, "anchor rows", &shapes[2][0], &shapes[3][0])?;
    Ok((
        dtypes[0],
        vec![
            shapes[2][0].clone(),
            shapes[2][1].clone() + shapes[3][1].clone(),
        ],
    ))
}

fn householder_qr_split_tangent_meta(
    dtypes: &[DType],
    shapes: &[&[SymDim]],
    right: bool,
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    const OP: &str = "tenferro-linalg.householder_qr_split_tangent";
    for shape in shapes {
        require_householder_rank2(OP, shape)?;
    }
    if dtypes.iter().any(|dtype| *dtype != dtypes[0]) {
        return Err(Error::dtype_mismatch(OP, dtypes[0], dtypes[1]));
    }
    require_static_extent_equal(OP, "left rows", &shapes[0][0], &shapes[1][0])?;
    require_static_extent_equal(OP, "right rows", &shapes[0][0], &shapes[2][0])?;
    let expected_width = shapes[1][1].clone() + shapes[2][1].clone();
    require_static_extent_equal(OP, "width", &shapes[0][1], &expected_width)?;
    let selected = if right { shapes[2] } else { shapes[1] };
    Ok((dtypes[0], selected.to_vec()))
}

fn eigh_meta(dtype: DType, shape: &[SymDim]) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.eigh", shape)?;
    Ok(vec![
        (singular_values_dtype(dtype), vector_shape(n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
    ])
}

fn eigh_values_meta(
    dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.eigh_values", shape)?;
    Ok((singular_values_dtype(dtype), vector_shape(n, batch)))
}

fn eig_meta(
    input_dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    let dtype = eig_output_dtype(input_dtype);
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.eig", shape)?;
    Ok(vec![
        (dtype, vector_shape(n.clone(), batch)),
        (dtype, matrix_shape(n.clone(), n, batch)),
    ])
}

fn eig_values_meta(
    input_dtype: DType,
    shape: &[SymDim],
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    let dtype = eig_output_dtype(input_dtype);
    let (n, _, batch) = matrix_meta_parts("tenferro-linalg.eig_values", shape)?;
    Ok((dtype, vector_shape(n, batch)))
}

fn matrix_shape(rows: SymDim, cols: SymDim, batch: &[SymDim]) -> Vec<SymDim> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(batch);
    shape
}

fn vector_shape(len: SymDim, batch: &[SymDim]) -> Vec<SymDim> {
    let mut shape = vec![len];
    shape.extend_from_slice(batch);
    shape
}

fn eig_output_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F64 | DType::C64 => DType::C64,
        DType::F32 | DType::C32 => DType::C32,
        DType::I32 | DType::I64 | DType::Bool => DType::C64,
    }
}

fn singular_values_dtype(dtype: DType) -> DType {
    match dtype {
        DType::C64 => DType::F64,
        DType::C32 => DType::F32,
        other => other,
    }
}

fn promote_dtypes(dtypes: &[DType]) -> DType {
    dtypes
        .iter()
        .copied()
        .reduce(tenferro_tensor::validate::promote_dtype)
        .unwrap_or(DType::F64)
}

fn hash_dtype(hasher: &mut dyn Hasher, dtype: DType) {
    let tag = match dtype {
        DType::F64 => 0,
        DType::F32 => 1,
        DType::I64 => 2,
        DType::C64 => 3,
        DType::C32 => 4,
        DType::I32 => 5,
        DType::Bool => 6,
    };
    hasher.write_u8(tag);
}

fn hash_svd_gauge(hasher: &mut dyn Hasher, gauge: SvdGauge) {
    let tag = match gauge {
        SvdGauge::Raw => 0,
        SvdGauge::CanonicalPivot => 1,
    };
    hasher.write_u8(tag);
}

fn hash_eigh_gauge(hasher: &mut dyn Hasher, gauge: EighGauge) {
    let tag = match gauge {
        EighGauge::Raw => 0,
        EighGauge::CanonicalPivot => 1,
    };
    hasher.write_u8(tag);
}

fn hash_qr_gauge(hasher: &mut dyn Hasher, gauge: QrGauge) {
    let tag = match gauge {
        QrGauge::Raw => 0,
        QrGauge::PositiveDiagonal => 1,
    };
    hasher.write_u8(tag);
}
