pub(crate) use crate::error::unsupported_dtype;

mod cholesky;
mod eig;
mod eigh;
mod full_piv_lu;
mod helpers;
mod lu;
mod qr;
mod solve;
mod svd;
mod triangular_solve;

pub(crate) use cholesky::{cholesky, cholesky_compact_data};
pub(crate) use eig::{eig, eig_values};
pub(crate) use eigh::{eigh, eigh_values};
pub(crate) use full_piv_lu::{full_piv_lu, full_piv_lu_solve};
pub(crate) use lu::{lu, lu_factor};
pub(crate) use qr::{
    append_2d as householder_qr_append, compact_factor_2d as householder_qr,
    from_factors_2d as householder_qr_from_factors, q_columns_2d as householder_qr_q_columns, qr,
    rank_revealing_qr, raw_r_2d as householder_qr_r,
};
pub(crate) use solve::{solve, solve_from_views, solve_into};
pub(crate) use svd::{svd, svd_values};
pub(crate) use triangular_solve::triangular_solve;
