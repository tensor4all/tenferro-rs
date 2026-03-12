use tenferro_device::Result;

use super::{
    as_i32, check_info_cholesky, check_info_nonnegative, check_info_success, check_len,
    fill_zero_upper, lwork_from_query_f32, lwork_from_query_f64, pivots_to_forward_perm, split_lu,
    write_real_eig_general_output, BlasLapackBackend, LinalgBackend,
};

mod decompositions;
mod linear_systems;
mod spectral;

use decompositions::impl_real_decompositions;
use linear_systems::impl_real_linear_systems;
use spectral::impl_real_spectral;

macro_rules! impl_lapack_backend_real {
    (
        $ty:ty,
        $gesvd:ident,
        $geqrf:ident,
        $orgqr:ident,
        $getrf:ident,
        $potrf:ident,
        $syev:ident,
        $gesv:ident,
        $trtrs:ident,
        $geev:ident,
        $gemm:path,
        $lwork_from_query:ident
    ) => {
        impl LinalgBackend<$ty> for BlasLapackBackend {
            type Real = $ty;

            impl_real_decompositions!(
                $ty,
                $gesvd,
                $geqrf,
                $orgqr,
                $getrf,
                $potrf,
                $lwork_from_query
            );
            impl_real_linear_systems!($ty, $gesv, $trtrs, $gemm);
            impl_real_spectral!($ty, $syev, $geev, $lwork_from_query);
        }
    };
}

pub(crate) use impl_lapack_backend_real;
