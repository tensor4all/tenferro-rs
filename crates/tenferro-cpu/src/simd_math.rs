//! Vectorized kernels for the elementwise analytic operations.
//!
//! `tenferro-cpu` evaluates `exp`/`log`/`sin`/`cos`/`tanh`/`expm1`/`log1p` one element at a time
//! through `strided_kernel::map_into`, and Rust does not vectorize calls into libm, so those paths
//! run at the speed of the scalar C library. This module maps an operation onto `elem-simd-math`,
//! which selects the best SIMD level for the running CPU.
//!
//! `strided_basic::map_into_uninit_with_kernel` owns everything else: which flat runs exist, how
//! the destination slices are built, and how many threads run the operation. It calls back here
//! once per run, so the kernel sits inside the parallel leaf rather than beside it. Every other
//! layout keeps the `map_into` path, so the two agree element for element; only the rounding
//! differs, by at most the bound documented for the kernel (1.0 ULP for these operations).
//!
//! `pow` is deliberately absent. SLEEF's `pow` — the source of `elem-simd-math` — is slower than the
//! platform `libm`, and glibc's is also correctly rounded, so `pow` keeps the scalar path. The
//! measurements are in that crate's `docs/findings-pow.md`.

use core::any::TypeId;
use core::mem::MaybeUninit;

use elem_simd_math::{f32 as kernels32, f64 as kernels64};

/// Which vectorized kernel an operation should try, resolved once at the dispatch boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VectorKernel {
    /// The operation has no vectorized kernel.
    None,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Expm1,
    Log1p,
}

/// Applies `kernel` to `src`, writing `dst`.
///
/// Returns `false` when there is no vectorized kernel for `T`, leaving `dst` untouched.
pub(crate) fn apply<T: 'static>(
    kernel: VectorKernel,
    src: &[T],
    dst: &mut [MaybeUninit<T>],
) -> bool {
    if kernel == VectorKernel::None || src.is_empty() || src.len() != dst.len() {
        return false;
    }
    // SAFETY: `TypeId` is injective, so each comparison proves that `T` is that scalar; the
    // reinterpreted slices then have the element type and the length the callee expects. `dst` is a
    // full-overwrite destination, so no element is read before it is written.
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let src = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<f64>(), src.len()) };
        let dst = unsafe {
            core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<MaybeUninit<f64>>(), dst.len())
        };
        let result = match kernel {
            VectorKernel::Exp => kernels64::exp(src, dst),
            VectorKernel::Log => kernels64::log(src, dst),
            VectorKernel::Sin => kernels64::sin(src, dst),
            VectorKernel::Cos => kernels64::cos(src, dst),
            VectorKernel::Tanh => kernels64::tanh(src, dst),
            VectorKernel::Expm1 => kernels64::expm1(src, dst),
            VectorKernel::Log1p => kernels64::log1p(src, dst),
            VectorKernel::None => unreachable!("guarded above"),
        };
        result.is_ok()
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        let src = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<f32>(), src.len()) };
        let dst = unsafe {
            core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<MaybeUninit<f32>>(), dst.len())
        };
        let result = match kernel {
            VectorKernel::Exp => kernels32::exp(src, dst),
            VectorKernel::Log => kernels32::log(src, dst),
            VectorKernel::Sin => kernels32::sin(src, dst),
            VectorKernel::Cos => kernels32::cos(src, dst),
            VectorKernel::Tanh => kernels32::tanh(src, dst),
            VectorKernel::Expm1 => kernels32::expm1(src, dst),
            VectorKernel::Log1p => kernels32::log1p(src, dst),
            VectorKernel::None => unreachable!("guarded above"),
        };
        result.is_ok()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel the module offers, so that each `match` arm in `apply` is exercised.
    const KERNELS: [VectorKernel; 7] = [
        VectorKernel::Exp,
        VectorKernel::Log,
        VectorKernel::Sin,
        VectorKernel::Cos,
        VectorKernel::Tanh,
        VectorKernel::Expm1,
        VectorKernel::Log1p,
    ];

    /// `src` is positive, so `log` and `log1p` are defined over all of it.
    fn samples<T: Copy>(lift: fn(f64) -> T) -> Vec<T> {
        (0..64).map(|i| lift(0.125 + i as f64 * 0.03125)).collect()
    }

    /// Compares a kernel result with a scalar one within a few ULP of the element type.
    #[track_caller]
    fn check<T: Copy>(kernel: VectorKernel, got: T, want: f64, eps: f64, widen: fn(T) -> f64) {
        let got = widen(got);
        let tolerance = 4.0 * eps * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "{kernel:?}: got {got}, want {want}"
        );
    }

    #[test]
    fn f64_kernels_match_the_scalar_definition_over_a_compact_buffer() {
        let src: Vec<f64> = samples(|x| x);
        let scalar: [fn(f64) -> f64; 7] = [
            f64::exp,
            f64::ln,
            f64::sin,
            f64::cos,
            f64::tanh,
            f64::exp_m1,
            f64::ln_1p,
        ];
        for (&kernel, &f) in KERNELS.iter().zip(scalar.iter()) {
            let mut dst = vec![MaybeUninit::<f64>::uninit(); src.len()];
            assert!(apply(kernel, &src, &mut dst));
            for (i, &x) in src.iter().enumerate() {
                // SAFETY: `apply` returned `true`, so every element was written.
                let got = unsafe { dst[i].assume_init() };
                check(kernel, got, f(x), f64::EPSILON, |v| v);
            }
        }
    }

    #[test]
    fn f32_kernels_match_the_scalar_definition_over_a_compact_buffer() {
        let src: Vec<f32> = samples(|x| x as f32);
        let scalar: [fn(f32) -> f32; 7] = [
            f32::exp,
            f32::ln,
            f32::sin,
            f32::cos,
            f32::tanh,
            f32::exp_m1,
            f32::ln_1p,
        ];
        for (&kernel, &f) in KERNELS.iter().zip(scalar.iter()) {
            let mut dst = vec![MaybeUninit::<f32>::uninit(); src.len()];
            assert!(apply(kernel, &src, &mut dst));
            for (i, &x) in src.iter().enumerate() {
                // SAFETY: as above.
                let got = unsafe { dst[i].assume_init() };
                check(kernel, got, f(x) as f64, f32::EPSILON as f64, |v| v as f64);
            }
        }
    }

    #[test]
    fn apply_declines_unsupported_kernels_and_length_mismatches() {
        let src = vec![1.0f64, 2.0];
        let mut dst = vec![MaybeUninit::<f64>::uninit(); 2];
        assert!(!apply(VectorKernel::None, &src, &mut dst));
        let mut short = vec![MaybeUninit::<f64>::uninit(); 1];
        assert!(!apply(VectorKernel::Exp, &src, &mut short));
        let mut empty = Vec::<MaybeUninit<f64>>::new();
        assert!(!apply(VectorKernel::Exp, &[], &mut empty));
        // A dtype without a kernel leaves the destination untouched.
        let isrc = vec![1i32, 2];
        let mut idst = vec![MaybeUninit::<i32>::uninit(); 2];
        assert!(!apply(VectorKernel::Exp, &isrc, &mut idst));
    }
}
