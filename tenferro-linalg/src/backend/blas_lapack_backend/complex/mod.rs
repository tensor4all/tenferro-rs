use tenferro_device::Result;

use super::{
    as_i32, check_info_cholesky, check_info_nonnegative, check_info_success, check_len,
    fill_zero_upper, lwork_from_query_c32, lwork_from_query_c64, pivots_to_forward_perm, split_lu,
    BlasLapackBackend, LinalgBackend,
};

mod decompositions;
mod linear_systems;
mod spectral;

use decompositions::impl_complex_decompositions;
use linear_systems::impl_complex_linear_systems;
use spectral::impl_complex_spectral;

macro_rules! impl_lapack_backend_complex {
    (
        $complex_ty:ty,
        $real_ty:ty,
        $gesvd:ident,
        $geqrf:ident,
        $ungqr:ident,
        $getrf:ident,
        $potrf:ident,
        $heev:ident,
        $gesv:ident,
        $trtrs:ident,
        $geev:ident,
        $gemm:path,
        $lwork_from_query:ident
    ) => {
        impl LinalgBackend<$complex_ty> for BlasLapackBackend {
            type Real = $real_ty;

            impl_complex_decompositions!(
                $complex_ty,
                $real_ty,
                $gesvd,
                $geqrf,
                $ungqr,
                $getrf,
                $potrf,
                $lwork_from_query
            );
            impl_complex_linear_systems!($complex_ty, $real_ty, $gesv, $trtrs, $gemm);
            impl_complex_spectral!($complex_ty, $real_ty, $heev, $geev, $lwork_from_query);
        }
    };
}

pub(crate) use impl_lapack_backend_complex;
