//! Vectorized kernels for the elementwise analytic operations.
//!
//! `tenferro-cpu` evaluates `exp`/`log`/`sin`/`cos`/`tanh`/`expm1`/`log1p` one element at a time
//! through `strided_kernel::map_into`, and Rust does not vectorize calls into libm, so those paths
//! run at the speed of the scalar C library. When both the input and the output of an operation are
//! compact column-major buffers, this module hands the flat slices to `elem-simd-math`, which
//! selects the best SIMD level for the running CPU. Every other layout keeps the `map_into` path
//! unchanged, so the two agree element for element; only the rounding differs, by at most the error
//! bound documented for the kernel (1.0 ULP for these operations).
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

/// Whether `dims`/`strides` describe a compact column-major buffer, i.e. whether the flat buffer
/// order equals the logical element order.
pub(crate) fn is_compact_column_major(dims: &[usize], strides: &[isize]) -> bool {
    let mut expected = 1isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if stride != expected {
            return false;
        }
        match expected.checked_mul(dim as isize) {
            Some(next) => expected = next,
            None => return false,
        }
    }
    true
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

    #[test]
    fn compact_column_major_separates_compact_from_strided_layouts() {
        // `[2, 3]` column-major is the layout the CPU backend allocates.
        assert!(is_compact_column_major(&[2, 3], &[1, 2]));
        // Row-major strides over the same shape are not compact.
        assert!(!is_compact_column_major(&[2, 3], &[3, 1]));
        // Scalar and size-1 dims are trivially compact.
        assert!(is_compact_column_major(&[], &[]));
        assert!(is_compact_column_major(&[1], &[1]));
        assert!(is_compact_column_major(&[7], &[1]));
        // A broadcast stride is not compact.
        assert!(!is_compact_column_major(&[4, 4], &[1, 0]));
    }

    #[test]
    fn apply_matches_the_scalar_definition_over_a_compact_buffer() {
        let src: Vec<f64> = (0..64).map(|i| (i as f64) * 0.125 - 3.0).collect();
        let mut dst = vec![MaybeUninit::<f64>::uninit(); src.len()];
        assert!(apply(VectorKernel::Exp, &src, &mut dst));
        for (i, &x) in src.iter().enumerate() {
            // SAFETY: `apply` returned `true`, so every element was written.
            let got = unsafe { dst[i].assume_init() };
            assert!((got - x.exp()).abs() <= 4.0 * f64::EPSILON * got.abs().max(1.0));
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
