use tenferro_linalg::{
    CholeskyExResult, EigenResult, InvExResult, LinalgScalar, LuFactorExResult, LuFactorResult,
    LuPivot, LuResult, NormKind, QrResult, SolveExResult,
};
use tenferro_tensor::Tensor;

use crate::runtime::contracts::*;
use crate::runtime::dispatch::*;
use crate::{Error, Result};

pub mod ad {
    pub use tenferro_internal_ad_linalg::eager::{
        cholesky_dyn, cholesky_ex_dyn, det_dyn, eig_dyn, eigen_dyn, inv_dyn, inv_ex_dyn, lstsq_dyn,
        lu_dyn, lu_factor_dyn, lu_factor_ex_dyn, lu_solve_dyn, matrix_exp_dyn, norm_dyn, pinv_dyn,
        qr_dyn, slogdet_dyn, solve_dyn, solve_ex_dyn, solve_triangular_dyn, svd_dyn,
    };
    pub use tenferro_internal_ad_ops::ad::*;
}

mod linalg;

pub use linalg::*;
