// Solve residual policy reference: PyTorch 8dd3b763, derivatives.yaml's
// _linalg_solve_ex and FunctionsManual.cpp::linalg_solve_backward. The tracked
// implementation runs the fused LuFactorSolve op, whose saved factors feed the
// LuSolvePrepared adjoint solve.
use std::sync::Arc;

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::{
    apply_eager_with_targeted_extension_in_session, EagerExtensionBackendKind, EagerExtensionTarget,
};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_cpu::CpuEngine;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaEngine;
use tenferro_runtime::{ErrorPhase, ExtensionModule};

use crate::eager_composites;
use crate::extension::{
    extension_module, validate_derivative_eps, EighOptions, LinalgExtensionOp, LinalgOp, QrOptions,
    SvdOptions,
};
use crate::rank_revealing_qr::validate_rank_revealing_qr_options;
use crate::{RankRevealingQrOptions, RankRevealingQrResult};

/// Crate-internal eager linalg bodies; the public grammar is
/// `EagerLinalgOps` (#2552), which forwards here.
pub(crate) trait EagerLinalg {
    fn cholesky(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    fn svd(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;

    fn svd_with_options(
        &mut self,
        input: &EagerTensor,
        options: SvdOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;

    fn svd_full(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;

    fn qr(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    fn qr_with_options(
        &mut self,
        input: &EagerTensor,
        options: QrOptions,
    ) -> Result<(EagerTensor, EagerTensor)>;

    fn triangular_solve(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<EagerTensor>;

    fn solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor>;

    fn lstsq(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor>;

    fn lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)>;

    fn slogdet(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    fn det(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    fn inv(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    fn eigvalsh(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    fn eigvals(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    fn eigh(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    fn eigh_with_options(
        &mut self,
        input: &EagerTensor,
        options: EighOptions,
    ) -> Result<(EagerTensor, EagerTensor)>;

    fn eig(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    fn pinv(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    fn pinv_with_rtol(&mut self, input: &EagerTensor, rtol: f64) -> Result<EagerTensor>;

    fn norm(
        &mut self,
        input: &EagerTensor,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<EagerTensor>;

    fn full_piv_lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
    )>;

    fn full_piv_lu_solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor)
        -> Result<EagerTensor>;

    fn rank_revealing_qr(
        &mut self,
        input: &EagerTensor,
        options: RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<EagerTensor>>;

    fn householder_qr(&mut self, input: &EagerTensor) -> Result<crate::HouseholderQr<EagerTensor>>;
}

impl EagerLinalg for EagerSession<'_> {
    fn svd(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        self.svd_with_options(input, SvdOptions::default())
    }

    fn svd_with_options(
        &mut self,
        input: &EagerTensor,
        options: SvdOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        validate_derivative_eps("svd_with_options", options.derivative_eps)?;
        three_outputs(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::Svd {
                    derivative_eps: options.derivative_eps,
                    gauge: options.gauge,
                    driver: options.driver,
                },
                &[input],
            )?,
            "svd",
        )
    }

    fn svd_full(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        three_outputs(
            apply_linalg_eager_in_session(self, LinalgOp::SvdFull, &[input])?,
            "svd_full",
        )
    }

    fn qr(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        self.qr_with_options(input, QrOptions::default())
    }

    fn qr_with_options(
        &mut self,
        input: &EagerTensor,
        options: QrOptions,
    ) -> Result<(EagerTensor, EagerTensor)> {
        two_outputs(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::Qr {
                    gauge: options.gauge,
                },
                &[input],
            )?,
            "qr",
        )
    }

    fn cholesky(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(self, LinalgOp::Cholesky, &[input])?,
            "cholesky",
        )
    }

    fn triangular_solve(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::TriangularSolve {
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                },
                &[matrix, rhs],
            )?,
            "triangular_solve",
        )
    }

    fn solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
        if !matrix.tracks_grad() && !rhs.tracks_grad() {
            return one_output(
                apply_linalg_eager_in_session(self, LinalgOp::Solve, &[matrix, rhs])?,
                "solve",
            );
        }
        validate_tracked_solve_inputs(matrix, rhs)?;
        factor_solve_output(apply_linalg_eager_in_session(
            self,
            LinalgOp::LuFactorSolve,
            &[matrix, rhs],
        )?)
    }

    fn lstsq(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::lstsq(self, matrix, rhs)
    }

    fn lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)> {
        let mut outputs = apply_linalg_eager_in_session(self, LinalgOp::Lu, &[input])?.into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(p), Some(l), Some(u), Some(parity), None) => Ok((p, l, u, parity)),
            _ => Err(Error::Internal(
                "lu eager op returned an unexpected number of outputs".into(),
            )),
        }
    }

    fn slogdet(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        eager_composites::slogdet(self, input)
    }

    fn det(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::det(self, input)
    }

    fn inv(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::inv(self, input)
    }

    fn eigvalsh(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::EighVals {
                    derivative_eps: crate::extension::DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    driver: EighOptions::default().driver,
                },
                &[input],
            )?,
            "eigvalsh",
        )
    }

    fn eigvals(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::EigVals {
                    input_dtype: input.dtype(),
                },
                &[input],
            )?,
            "eigvals",
        )
    }

    fn eigh(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        self.eigh_with_options(input, EighOptions::default())
    }

    fn eigh_with_options(
        &mut self,
        input: &EagerTensor,
        options: EighOptions,
    ) -> Result<(EagerTensor, EagerTensor)> {
        validate_derivative_eps("eigh_with_options", options.derivative_eps)?;
        two_outputs(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::Eigh {
                    derivative_eps: options.derivative_eps,
                    gauge: options.gauge,
                    driver: options.driver,
                },
                &[input],
            )?,
            "eigh",
        )
    }

    fn eig(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        two_outputs(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::Eig {
                    input_dtype: input.dtype(),
                },
                &[input],
            )?,
            "eig",
        )
    }

    fn pinv(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::pinv(self, input)
    }

    fn pinv_with_rtol(&mut self, input: &EagerTensor, rtol: f64) -> Result<EagerTensor> {
        eager_composites::pinv_with_rtol(self, input, rtol)
    }

    fn norm(
        &mut self,
        input: &EagerTensor,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<EagerTensor> {
        eager_composites::norm(self, input, ord, dim, keepdim)
    }

    fn full_piv_lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
    )> {
        let mut outputs =
            apply_linalg_eager_in_session(self, LinalgOp::FullPivLu, &[input])?.into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => Ok((p, l, u, q, parity)),
            _ => Err(Error::Internal(
                "full_piv_lu eager op returned an unexpected number of outputs".into(),
            )),
        }
    }

    fn full_piv_lu_solve(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
    ) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::FullPivLuSolve { transpose_a: false },
                &[matrix, rhs],
            )?,
            "full_piv_lu_solve",
        )
    }

    fn rank_revealing_qr(
        &mut self,
        input: &EagerTensor,
        options: RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<EagerTensor>> {
        validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
        let mut outputs = apply_linalg_eager_in_session(
            self,
            LinalgOp::RankRevealingQr {
                gauge: options.gauge,
                rtol: options.rtol,
                atol: options.atol,
            },
            &[input],
        )?
        .into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(q), Some(r), Some(column_permutation), Some(rank), None) => {
                Ok(RankRevealingQrResult {
                    q,
                    r,
                    column_permutation,
                    rank,
                })
            }
            _ => Err(Error::Internal(
                "rank_revealing_qr eager op returned an unexpected number of outputs".into(),
            )),
        }
    }

    fn householder_qr(&mut self, input: &EagerTensor) -> Result<crate::HouseholderQr<EagerTensor>> {
        crate::householder::eager_state(apply_linalg_eager_in_session(
            self,
            LinalgOp::HouseholderQrFactor,
            &[input],
        )?)
    }
}

pub(crate) fn apply_linalg_eager_in_session(
    session: &mut EagerSession<'_>,
    op: LinalgOp,
    inputs: &[&EagerTensor],
) -> Result<Vec<EagerTensor>> {
    let op = Arc::new(LinalgExtensionOp::new(op));
    apply_eager_with_targeted_extension_in_session(session, op, inputs, eager_extension_module)
}

fn eager_extension_module(target: EagerExtensionTarget) -> Result<Arc<dyn ExtensionModule>> {
    let EagerExtensionTarget {
        engine_id,
        backend_kind,
    } = target;
    match backend_kind {
        EagerExtensionBackendKind::Cpu => {
            extension_module::<CpuEngine>(engine_id).map_err(eager_runtime_config_error)
        }
        #[cfg(feature = "cuda")]
        EagerExtensionBackendKind::Cuda => {
            extension_module::<CudaEngine>(engine_id).map_err(eager_runtime_config_error)
        }
    }
}

fn eager_runtime_config_error(source: tenferro_runtime::RuntimeConfigError) -> Error {
    Error::runtime_state_source(
        "tenferro_linalg::eager_extension_module",
        ErrorPhase::Execution,
        source,
    )
}

fn validate_tracked_solve_inputs(a: &EagerTensor, b: &EagerTensor) -> Result<()> {
    if !a.same_context(b) {
        return Err(Error::ContextMismatch {
            lhs: a.ctx_id(),
            rhs: b.ctx_id(),
        });
    }
    crate::validation::validate_solve_inputs(a.dtype(), a.shape(), b.dtype(), b.shape())
}

// One fused factor+solve retains LU/pivots and X for backward, while the
// explicit A operand preserves higher-order semantics (PyTorch's
// _linalg_solve_ex / FunctionsManual.cpp::linalg_solve_backward).
fn factor_solve_output(outputs: Vec<EagerTensor>) -> Result<EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(x), Some(_packed_lu), Some(_pivots), None) => Ok(x),
        _ => Err(Error::Internal(
            "lu_factor_solve eager op returned an unexpected number of outputs".into(),
        )),
    }
}

fn three_outputs(
    outputs: Vec<EagerTensor>,
    name: &str,
) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), None) => Ok((first, second, third)),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}

pub(crate) fn one_output(outputs: Vec<EagerTensor>, name: &str) -> Result<EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}

fn two_outputs(outputs: Vec<EagerTensor>, name: &str) -> Result<(EagerTensor, EagerTensor)> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(lhs), Some(rhs), None) => Ok((lhs, rhs)),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}
