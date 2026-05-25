use num_complex::{Complex32, Complex64};

use crate::cpu::CpuContext;

pub(crate) trait FaerGemm: Sized {
    #[allow(clippy::too_many_arguments)]
    unsafe fn strided_gemm(
        ctx: &CpuContext,
        alpha: Self,
        a_ptr: *const Self,
        m: usize,
        k: usize,
        a_rs: isize,
        a_cs: isize,
        b_ptr: *const Self,
        n: usize,
        b_rs: isize,
        b_cs: isize,
        beta: Self,
        c_ptr: *mut Self,
        c_rs: isize,
        c_cs: isize,
    ) {
        unsafe {
            Self::strided_gemm_with_conj(
                ctx, alpha, a_ptr, m, k, a_rs, a_cs, false, b_ptr, n, b_rs, b_cs, false, beta,
                c_ptr, c_rs, c_cs,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn strided_gemm_with_conj(
        ctx: &CpuContext,
        alpha: Self,
        a_ptr: *const Self,
        m: usize,
        k: usize,
        a_rs: isize,
        a_cs: isize,
        conj_a: bool,
        b_ptr: *const Self,
        n: usize,
        b_rs: isize,
        b_cs: isize,
        conj_b: bool,
        beta: Self,
        c_ptr: *mut Self,
        c_rs: isize,
        c_cs: isize,
    );
}

macro_rules! impl_faer_gemm {
    ($ty:ty) => {
        impl FaerGemm for $ty {
            unsafe fn strided_gemm_with_conj(
                ctx: &CpuContext,
                alpha: $ty,
                a_ptr: *const $ty,
                m: usize,
                k: usize,
                a_rs: isize,
                a_cs: isize,
                conj_a: bool,
                b_ptr: *const $ty,
                n: usize,
                b_rs: isize,
                b_cs: isize,
                conj_b: bool,
                beta: $ty,
                c_ptr: *mut $ty,
                c_rs: isize,
                c_cs: isize,
            ) {
                use faer::{Accum, Conj, MatMut, MatRef};
                let a_rs = super::normalize_singleton_stride(a_rs, m, k);
                let a_cs = super::normalize_singleton_stride(a_cs, k, m);
                let b_rs = super::normalize_singleton_stride(b_rs, k, n);
                let b_cs = super::normalize_singleton_stride(b_cs, n, k);
                let c_rs = super::normalize_singleton_stride(c_rs, m, 1);
                let c_cs = super::normalize_singleton_stride(c_cs, n, m);
                let a_mat = MatRef::<$ty>::from_raw_parts(a_ptr, m, k, a_rs, a_cs);
                let b_mat = MatRef::<$ty>::from_raw_parts(b_ptr, k, n, b_rs, b_cs);
                let zero = <$ty as num_traits::Zero>::zero();
                let one = <$ty as num_traits::One>::one();
                let accum = if beta == zero {
                    Accum::Replace
                } else {
                    if beta != one {
                        let mut col_off = 0isize;
                        for _ in 0..n {
                            let mut off = col_off;
                            for _ in 0..m {
                                *c_ptr.offset(off) *= beta;
                                off += c_rs;
                            }
                            col_off += c_cs;
                        }
                    }
                    Accum::Add
                };
                let mut c_mat = MatMut::<$ty>::from_raw_parts_mut(c_ptr, m, n, c_rs, c_cs);
                let conj_a = if conj_a { Conj::Yes } else { Conj::No };
                let conj_b = if conj_b { Conj::Yes } else { Conj::No };
                faer::linalg::matmul::matmul_with_conj(
                    &mut c_mat,
                    accum,
                    &a_mat,
                    conj_a,
                    &b_mat,
                    conj_b,
                    alpha,
                    ctx.faer_par(),
                );
            }
        }
    };
}

impl_faer_gemm!(f64);
impl_faer_gemm!(f32);
impl_faer_gemm!(Complex64);
impl_faer_gemm!(Complex32);
