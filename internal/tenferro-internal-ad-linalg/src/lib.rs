//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

mod linearized;

pub use tenferro_internal_error::{Error, Result};

pub(crate) mod runtime {
    pub mod contracts {
        pub use tenferro_internal_runtime::contracts::*;
    }

    pub mod dispatch {
        pub use tenferro_internal_runtime::dispatch::*;
    }
}

pub(crate) mod structured {
    pub use tenferro_internal_frontend_core::StructuredTensor;
}

mod ops;

pub mod eager {
    pub use crate::ops::linalg::ad::eager::{
        cholesky_dyn, cholesky_ex_dyn, det_dyn, eig_dyn, eigen_dyn, inv_dyn, inv_ex_dyn, lstsq_dyn,
        lu_dyn, lu_factor_dyn, lu_factor_ex_dyn, lu_solve_dyn, matrix_exp_dyn, norm_dyn, pinv_dyn,
        qr_dyn, slogdet_dyn, solve_dyn, solve_ex_dyn, solve_triangular_dyn, svd_dyn,
    };
}

pub mod results {
    pub use crate::ops::linalg::results::{
        DynCholeskyExResult, DynEigResult, DynEigenResult, DynInvExResult, DynLstsqResult,
        DynLuFactorExResult, DynLuFactorResult, DynLuResult, DynQrResult, DynSlogdetResult,
        DynSolveExResult, DynSvdResult,
    };
}

#[doc(hidden)]
pub mod __typed_ad {
    pub use crate::ops::linalg::ad::{
        cholesky_ad, det_ad, eig_ad, eigen_ad, inv_ad, lstsq_ad, lu_ad, qr_ad, slogdet_ad, svd_ad,
        CholeskyAdBuilder, DetAdBuilder, EigAdBuilder, EigenAdBuilder, InvAdBuilder,
        LstsqAdBuilder, LuAdBuilder, QrAdBuilder, SlogdetAdBuilder, SvdAdBuilder,
    };
}

#[doc(hidden)]
pub mod __typed_eager {
    pub use crate::ops::linalg::ad::eager::*;
}

#[doc(hidden)]
pub mod __typed_results {
    pub use crate::ops::linalg::results::*;
}
