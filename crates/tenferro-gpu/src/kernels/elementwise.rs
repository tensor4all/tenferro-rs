// INVARIANT: CubeCL's kernel DSL intentionally emits equal-expression forms
// that are simplified by the device compiler, not by host-side Clippy.
#![allow(clippy::eq_op)]
// INVARIANT: the complex-division kernel keeps `x = x * y` form (Baudin–Smith
// scaling) because compound assignment is not part of the kernel DSL subset.
#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

use crate::kernels::helpers::{
    flat_to_tensor_index, multi_to_tensor_index, nan_propagating_max, nan_propagating_min,
    wrapping_add, wrapping_mul, wrapping_neg, wrapping_sub,
};

pub(crate) const COMPARE_EQ: usize = 0;
pub(crate) const COMPARE_LT: usize = 1;
pub(crate) const COMPARE_LE: usize = 2;
pub(crate) const COMPARE_GT: usize = 3;
pub(crate) const COMPARE_GE: usize = 4;
pub(crate) const MIXED_ADD: usize = 0;
pub(crate) const MIXED_SUB: usize = 1;
pub(crate) const MIXED_MUL: usize = 2;
pub(crate) const MIXED_DIV: usize = 3;

macro_rules! binary_float_kernel {
    ($name:ident, $op:tt) => {
        #[cube(launch_unchecked)]
        pub fn $name<F: Float>(out: &mut Array<F>, lhs: &Array<F>, rhs: &Array<F>) {
            if ABSOLUTE_POS < out.len() {
                out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] $op rhs[ABSOLUTE_POS];
            }
        }
    };
}

macro_rules! binary_float_complex_kernel {
    ($float_name:ident, $complex_name:ident, $op:tt) => {
        #[cube(launch_unchecked)]
        pub fn $float_name<F: Float>(out: &mut Array<F>, lhs: &Array<F>, rhs: &Array<F>) {
            if ABSOLUTE_POS < out.len() {
                out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] $op rhs[ABSOLUTE_POS];
            }
        }

        #[cube(launch_unchecked)]
        pub fn $complex_name<C: ComplexCore>(
            out: &mut Array<C>,
            lhs: &Array<C>,
            rhs: &Array<C>,
        ) {
            if ABSOLUTE_POS < out.len() {
                out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] $op rhs[ABSOLUTE_POS];
            }
        }
    };
}

#[cube]
fn broadcast_source_index<E: CubePrimitive>(
    out_flat: usize,
    out: &Tensor<E>,
    input: &Tensor<E>,
    #[comptime] dims: Sequence<usize>,
    #[comptime] output_rank: usize,
) -> usize {
    let input_rank = dims.len();
    let out_idx = flat_to_tensor_index(out_flat, out, output_rank);
    let mut input_idx = Array::<usize>::new(input_rank);
    #[unroll]
    for src_axis in 0..input_rank {
        let dst_axis = comptime! { *dims.index(src_axis) };
        let src_dim = input.shape(src_axis);
        input_idx[src_axis] = out_idx[dst_axis];
        if src_dim == 1 {
            input_idx[src_axis] = 0;
        }
    }
    multi_to_tensor_index(&input_idx, input, input_rank)
}

macro_rules! broadcast_multiply_kernel {
    ($name:ident, $bound:path) => {
        #[cube(launch_unchecked)]
        pub fn $name<E: $bound>(
            out: &mut Tensor<E>,
            lhs: &Tensor<E>,
            rhs: &Tensor<E>,
            #[comptime] lhs_dims: Sequence<usize>,
            #[comptime] rhs_dims: Sequence<usize>,
            #[comptime] output_rank: usize,
        ) {
            if ABSOLUTE_POS < out.len() {
                let lhs_idx = broadcast_source_index(ABSOLUTE_POS, out, lhs, lhs_dims, output_rank);
                let rhs_idx = broadcast_source_index(ABSOLUTE_POS, out, rhs, rhs_dims, output_rank);
                out[ABSOLUTE_POS] = lhs[lhs_idx] * rhs[rhs_idx];
            }
        }
    };
}

broadcast_multiply_kernel!(broadcast_multiply_float, Float);
broadcast_multiply_kernel!(broadcast_multiply_complex, ComplexCore);

#[cube(launch_unchecked)]
pub fn broadcast_multiply_int<I: Int>(
    out: &mut Tensor<I>,
    lhs: &Tensor<I>,
    rhs: &Tensor<I>,
    #[comptime] lhs_dims: Sequence<usize>,
    #[comptime] rhs_dims: Sequence<usize>,
    #[comptime] output_rank: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let lhs_idx = broadcast_source_index(ABSOLUTE_POS, out, lhs, lhs_dims, output_rank);
        let rhs_idx = broadcast_source_index(ABSOLUTE_POS, out, rhs, rhs_dims, output_rank);
        out[ABSOLUTE_POS] = wrapping_mul::<I>(lhs[lhs_idx], rhs[rhs_idx]);
    }
}

macro_rules! unary_float_kernel {
    ($name:ident, $method:ident) => {
        #[cube(launch_unchecked)]
        pub fn $name<F: Float>(out: &mut Array<F>, input: &Array<F>) {
            if ABSOLUTE_POS < out.len() {
                out[ABSOLUTE_POS] = input[ABSOLUTE_POS].$method();
            }
        }
    };
}

macro_rules! unary_both_kernel {
    ($float_name:ident, $complex_name:ident, |$value:ident| $body:expr) => {
        #[cube(launch_unchecked)]
        pub fn $float_name<F: Float>(out: &mut Array<F>, input: &Array<F>) {
            if ABSOLUTE_POS < out.len() {
                let $value = input[ABSOLUTE_POS];
                out[ABSOLUTE_POS] = $body;
            }
        }

        #[cube(launch_unchecked)]
        pub fn $complex_name<C: ComplexCore>(out: &mut Array<C>, input: &Array<C>) {
            if ABSOLUTE_POS < out.len() {
                let $value = input[ABSOLUTE_POS];
                out[ABSOLUTE_POS] = $body;
            }
        }
    };
}

/// `y[i] <- coefficients[0] * x[strided(i)] + coefficients[1] * y[i]` for a
/// strided or offset `x`.
///
/// The compact `x` layout is served by cuBLAS `geam`; this kernel covers the
/// arbitrary-stride source layouts no BLAS-1 vendor entry can address, so the
/// caller never has to canonicalize `x` into scratch first. `y` stays compact
/// per the shared accumulate contract and is addressed as `y_offset + i`.
///
/// INVARIANT: the coefficients arrive as a two-element device array rather than
/// kernel scalars because the CUDA dialect's kernel-argument type info panics
/// while sizing a complex scalar parameter; the array form is the same explicit
/// device-constant boundary the in-place scaling kernels use.
macro_rules! axpby_strided_source_kernel {
    ($name:ident, $bound:ident) => {
        #[cube(launch_unchecked)]
        pub fn $name<E: $bound>(
            y: &mut Array<E>,
            x: &Array<E>,
            coefficients: &Array<E>,
            #[comptime] dims: Sequence<usize>,
            #[comptime] x_strides: Sequence<i64>,
            x_offset: i64,
            y_offset: i64,
            #[comptime] len: usize,
            #[comptime] rank: usize,
        ) {
            if ABSOLUTE_POS < len {
                let mut flat = ABSOLUTE_POS;
                let mut x_index = x_offset;
                #[unroll]
                for axis in 0..rank {
                    let dim = comptime! { *dims.index(axis) };
                    let coordinate = flat % dim;
                    flat /= dim;
                    let x_stride = comptime! { *x_strides.index(axis) };
                    x_index += (coordinate as i64) * x_stride;
                }
                let y_index = usize::cast_from(y_offset) + ABSOLUTE_POS;
                y[y_index] =
                    coefficients[0] * x[usize::cast_from(x_index)] + coefficients[1] * y[y_index];
            }
        }
    };
}

axpby_strided_source_kernel!(axpby_strided_source_float, Float);
axpby_strided_source_kernel!(axpby_strided_source_complex, ComplexCore);

binary_float_complex_kernel!(add_float, add_complex, +);
binary_float_complex_kernel!(sub_float, sub_complex, -);
binary_float_complex_kernel!(mul_float, mul_complex, *);
// Complex division uses `div_complex_parts` below: `ComplexCore`'s `/` lowers to
// the unscaled textbook formula, which overflows/underflows for extreme operands.
binary_float_kernel!(div_float, /);

#[cube(launch_unchecked)]
pub fn add_int<I: Int>(out: &mut Array<I>, lhs: &Array<I>, rhs: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = wrapping_add::<I>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn sub_int<I: Int>(out: &mut Array<I>, lhs: &Array<I>, rhs: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = wrapping_sub::<I>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn mul_int<I: Int>(out: &mut Array<I>, lhs: &Array<I>, rhs: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = wrapping_mul::<I>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

macro_rules! scalar_binary_float_kernel {
    ($name:ident, |$lhs:ident, $rhs:ident| $body:expr) => {
        #[cube(launch_unchecked)]
        pub fn $name<F: Float>(
            out: &mut Array<F>,
            lhs: &Array<F>,
            rhs: &Array<F>,
            #[comptime] lhs_scalar: bool,
        ) {
            if ABSOLUTE_POS < out.len() {
                let lhs_idx = if lhs_scalar { 0 } else { ABSOLUTE_POS };
                let rhs_idx = if lhs_scalar { ABSOLUTE_POS } else { 0 };
                let $lhs = lhs[lhs_idx];
                let $rhs = rhs[rhs_idx];
                out[ABSOLUTE_POS] = $body;
            }
        }
    };
}

scalar_binary_float_kernel!(scalar_div_float, |x, y| x / y);
scalar_binary_float_kernel!(scalar_pow_float, |x, y| x.powf(y));
scalar_binary_float_kernel!(scalar_rem_float, |x, y| {
    let remainder = x - (x / y).trunc() * y;
    if remainder == F::new(0.0f32) {
        x * F::new(0.0f32)
    } else {
        remainder
    }
});

#[cube(launch_unchecked)]
pub fn scalar_real_complex_binary<F: Float>(
    out: &mut Array<F>,
    real: &Array<F>,
    complex: &Array<F>,
    #[comptime] real_lhs: bool,
    #[comptime] mode: usize,
) {
    let complex_idx = ABSOLUTE_POS * 2;
    if complex_idx < out.len() {
        let scalar = real[0];
        let re = complex[complex_idx];
        let im = complex[complex_idx + 1];
        let zero = F::new(0.0f32);
        // INVARIANT: These unsimplified component expressions must evaluate in the
        // same order as CPU `num_complex` after promotion to `Complex(real, +0)`.
        // Keep zero cross terms: they determine NaN, infinity, overflow, and signed zero.
        let (out_re, out_im) = if mode == MIXED_ADD {
            if real_lhs {
                (scalar + re, zero + im)
            } else {
                (re + scalar, im + zero)
            }
        } else if mode == MIXED_SUB {
            if real_lhs {
                (scalar - re, zero - im)
            } else {
                (re - scalar, im - zero)
            }
        } else if mode == MIXED_MUL {
            if real_lhs {
                (scalar * re - zero * im, scalar * im + zero * re)
            } else {
                (re * scalar - im * zero, re * zero + im * scalar)
            }
        } else if !real_lhs {
            let norm_sqr = scalar * scalar + zero * zero;
            (
                (re * scalar + im * zero) / norm_sqr,
                (im * scalar - re * zero) / norm_sqr,
            )
        } else {
            let norm_sqr = re * re + im * im;
            (
                (scalar * re + zero * im) / norm_sqr,
                (zero * re - scalar * im) / norm_sqr,
            )
        };
        out[complex_idx] = out_re;
        out[complex_idx + 1] = out_im;
    }
}

#[cube(launch_unchecked)]
pub fn scalar_div_int_checked<I: Int>(
    out: &mut Array<I>,
    lhs: &Array<I>,
    rhs: &Array<I>,
    err: &mut Array<i32>,
    #[comptime] lhs_scalar: bool,
) {
    if ABSOLUTE_POS < out.len() {
        let lhs_idx = if lhs_scalar { 0 } else { ABSOLUTE_POS };
        let rhs_idx = if lhs_scalar { ABSOLUTE_POS } else { 0 };
        let x = lhs[lhs_idx];
        let y = rhs[rhs_idx];
        let zero = I::new(0);
        let minus_one = wrapping_sub::<I>(zero, I::new(1));
        if y == zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else if y == minus_one {
            out[ABSOLUTE_POS] = wrapping_neg::<I>(x);
        } else {
            out[ABSOLUTE_POS] = x / y;
        }
    }
}

#[cube(launch_unchecked)]
pub fn scalar_rem_int_checked<I: Int>(
    out: &mut Array<I>,
    lhs: &Array<I>,
    rhs: &Array<I>,
    err: &mut Array<i32>,
    #[comptime] lhs_scalar: bool,
) {
    if ABSOLUTE_POS < out.len() {
        let lhs_idx = if lhs_scalar { 0 } else { ABSOLUTE_POS };
        let rhs_idx = if lhs_scalar { ABSOLUTE_POS } else { 0 };
        let x = lhs[lhs_idx];
        let y = rhs[rhs_idx];
        let zero = I::new(0);
        let minus_one = wrapping_sub::<I>(zero, I::new(1));
        if y == zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else if y == minus_one {
            out[ABSOLUTE_POS] = zero;
        } else {
            let quotient = x / y;
            out[ABSOLUTE_POS] = wrapping_sub::<I>(x, wrapping_mul::<I>(quotient, y));
        }
    }
}

#[cube(launch_unchecked)]
pub fn div_int_checked<I: Int>(
    out: &mut Array<I>,
    lhs: &Array<I>,
    rhs: &Array<I>,
    err: &mut Array<i32>,
) {
    if ABSOLUTE_POS < out.len() {
        let x = lhs[ABSOLUTE_POS];
        let y = rhs[ABSOLUTE_POS];
        let zero = I::new(0);
        let minus_one = wrapping_sub::<I>(zero, I::new(1));
        if y == zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else if y == minus_one {
            out[ABSOLUTE_POS] = wrapping_neg::<I>(x);
        } else {
            out[ABSOLUTE_POS] = x / y;
        }
    }
}

#[cube(launch_unchecked)]
pub fn rem_float<F: Float>(out: &mut Array<F>, lhs: &Array<F>, rhs: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        let x = lhs[ABSOLUTE_POS];
        let y = rhs[ABSOLUTE_POS];
        let remainder = x - (x / y).trunc() * y;
        out[ABSOLUTE_POS] = if remainder == F::new(0.0f32) {
            x * F::new(0.0f32)
        } else {
            remainder
        };
    }
}

#[cube(launch_unchecked)]
pub fn rem_int_checked<I: Int>(
    out: &mut Array<I>,
    lhs: &Array<I>,
    rhs: &Array<I>,
    err: &mut Array<i32>,
) {
    if ABSOLUTE_POS < out.len() {
        let x = lhs[ABSOLUTE_POS];
        let y = rhs[ABSOLUTE_POS];
        let zero = I::new(0);
        let minus_one = wrapping_sub::<I>(zero, I::new(1));
        if y == zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else if y == minus_one {
            out[ABSOLUTE_POS] = zero;
        } else {
            let quotient = x / y;
            out[ABSOLUTE_POS] = wrapping_sub::<I>(x, wrapping_mul::<I>(quotient, y));
        }
    }
}
unary_both_kernel!(neg_float, neg_complex, |value| -value);
#[cube(launch_unchecked)]
pub fn neg_int<I: Int>(out: &mut Array<I>, input: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = wrapping_neg::<I>(input[ABSOLUTE_POS]);
    }
}
unary_float_kernel!(exp_float, exp);
unary_float_kernel!(log_float, ln);
unary_float_kernel!(sin_float, sin);
unary_float_kernel!(cos_float, cos);
unary_float_kernel!(tanh_float, tanh);
unary_float_kernel!(sqrt_float, sqrt);

#[cube(launch_unchecked)]
pub fn rsqrt_float<F: Float>(out: &mut Array<F>, input: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].inverse_sqrt();
    }
}

#[cube(launch_unchecked)]
pub fn expm1_float<F: Float>(out: &mut Array<F>, input: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].exp_m1();
    }
}

#[cube(launch_unchecked)]
pub fn log1p_float<F: Float>(out: &mut Array<F>, input: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].log1p();
    }
}

#[cube(launch_unchecked)]
pub fn abs_float<F: Float<WithScalar<F> = F>>(out: &mut Array<F>, input: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        let value = input[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = value.abs();
    }
}

#[cube(launch_unchecked)]
pub fn abs_complex32(out: &mut Array<f32>, input: &Array<num_complex::Complex32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].abs();
    }
}

#[cube(launch_unchecked)]
pub fn abs_complex64(out: &mut Array<f64>, input: &Array<num_complex::Complex64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].abs();
    }
}

#[cube(launch_unchecked)]
pub fn abs_int<I: Int>(out: &mut Array<I>, input: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        let value = input[ABSOLUTE_POS];
        let zero = I::new(0);
        out[ABSOLUTE_POS] = if value < zero {
            wrapping_neg::<I>(value)
        } else {
            value
        };
    }
}

#[cube(launch_unchecked)]
pub fn sign_float<F: Float>(out: &mut Array<F>, input: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        let value = input[ABSOLUTE_POS];
        let zero = F::new(0.0_f32);
        let one = F::new(1.0_f32);
        out[ABSOLUTE_POS] = if value != value {
            value
        } else if value == zero {
            zero
        } else if value > zero {
            one
        } else {
            -one
        };
    }
}

#[cube(launch_unchecked)]
pub fn sign_complex<C: cubecl::frontend::ComplexMath>(out: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < out.len() {
        let value = input[ABSOLUTE_POS];
        let zero = C::cast_from(0.0_f32);
        out[ABSOLUTE_POS] = if value == zero {
            zero
        } else {
            value / C::cast_from(value.abs())
        };
    }
}

#[cube(launch_unchecked)]
pub fn sign_int<I: Int>(out: &mut Array<I>, input: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        let value = input[ABSOLUTE_POS];
        let zero = I::new(0);
        out[ABSOLUTE_POS] = if value == zero {
            zero
        } else if value > zero {
            I::new(1)
        } else {
            wrapping_sub::<I>(zero, I::new(1))
        };
    }
}

#[cube(launch_unchecked)]
pub fn maximum_float<F: Float>(out: &mut Array<F>, lhs: &Array<F>, rhs: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = nan_propagating_max::<F>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn maximum_int<I: Int>(out: &mut Array<I>, lhs: &Array<I>, rhs: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        let x = lhs[ABSOLUTE_POS];
        let y = rhs[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = if x >= y { x } else { y };
    }
}

#[cube(launch_unchecked)]
pub fn minimum_float<F: Float>(out: &mut Array<F>, lhs: &Array<F>, rhs: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = nan_propagating_min::<F>(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn minimum_int<I: Int>(out: &mut Array<I>, lhs: &Array<I>, rhs: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        let x = lhs[ABSOLUTE_POS];
        let y = rhs[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = if x <= y { x } else { y };
    }
}

#[cube(launch_unchecked)]
pub fn pow_float<F: Float>(out: &mut Array<F>, lhs: &Array<F>, rhs: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS].powf(rhs[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn pow_int_checked<I: Int>(
    out: &mut Array<I>,
    lhs: &Array<I>,
    rhs: &Array<I>,
    err: &mut Array<i32>,
) {
    if ABSOLUTE_POS < out.len() {
        let zero = I::new(0);
        let one = I::new(1);
        let two = I::new(2);
        let mut exp = rhs[ABSOLUTE_POS];
        if exp < zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else {
            let mut base = lhs[ABSOLUTE_POS];
            let mut acc = one;
            while exp > zero {
                let quotient = exp / two;
                let remainder = wrapping_sub::<I>(exp, wrapping_mul::<I>(quotient, two));
                if remainder != zero {
                    acc = wrapping_mul::<I>(acc, base);
                }
                exp = quotient;
                if exp > zero {
                    base = wrapping_mul::<I>(base, base);
                }
            }
            out[ABSOLUTE_POS] = acc;
        }
    }
}

#[cube(launch_unchecked)]
pub fn scalar_pow_int_checked<I: Int>(
    out: &mut Array<I>,
    lhs: &Array<I>,
    rhs: &Array<I>,
    err: &mut Array<i32>,
    #[comptime] lhs_scalar: bool,
) {
    if ABSOLUTE_POS < out.len() {
        let lhs_idx = if lhs_scalar { 0 } else { ABSOLUTE_POS };
        let rhs_idx = if lhs_scalar { ABSOLUTE_POS } else { 0 };
        let zero = I::new(0);
        let one = I::new(1);
        let two = I::new(2);
        let mut exp = rhs[rhs_idx];
        if exp < zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else {
            let mut base = lhs[lhs_idx];
            let mut acc = one;
            while exp > zero {
                let quotient = exp / two;
                let remainder = wrapping_sub::<I>(exp, wrapping_mul::<I>(quotient, two));
                if remainder != zero {
                    acc = wrapping_mul::<I>(acc, base);
                }
                exp = quotient;
                if exp > zero {
                    base = wrapping_mul::<I>(base, base);
                }
            }
            out[ABSOLUTE_POS] = acc;
        }
    }
}

#[cube(launch_unchecked)]
pub fn compare_float_bool<F: Float>(
    out: &mut Array<bool>,
    lhs: &Array<F>,
    rhs: &Array<F>,
    #[comptime] mode: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let x = lhs[ABSOLUTE_POS];
        let y = rhs[ABSOLUTE_POS];
        let pred = match mode {
            COMPARE_EQ => x == y,
            COMPARE_LT => x < y,
            COMPARE_LE => x <= y,
            COMPARE_GT => x > y,
            COMPARE_GE => x >= y,
            _ => false,
        };
        out[ABSOLUTE_POS] = pred;
    }
}

#[cube(launch_unchecked)]
pub fn compare_int_bool<I: Int>(
    out: &mut Array<bool>,
    lhs: &Array<I>,
    rhs: &Array<I>,
    #[comptime] mode: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let x = lhs[ABSOLUTE_POS];
        let y = rhs[ABSOLUTE_POS];
        let pred = match mode {
            COMPARE_EQ => x == y,
            COMPARE_LT => x < y,
            COMPARE_LE => x <= y,
            COMPARE_GT => x > y,
            COMPARE_GE => x >= y,
            _ => false,
        };
        out[ABSOLUTE_POS] = pred;
    }
}

#[cube(launch_unchecked)]
pub fn select_bool_float<F: Float>(
    out: &mut Array<F>,
    pred: &Array<bool>,
    on_true: &Array<F>,
    on_false: &Array<F>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = if pred[ABSOLUTE_POS] {
            on_true[ABSOLUTE_POS]
        } else {
            on_false[ABSOLUTE_POS]
        };
    }
}

#[cube(launch_unchecked)]
pub fn select_bool_int<I: Int>(
    out: &mut Array<I>,
    pred: &Array<bool>,
    on_true: &Array<I>,
    on_false: &Array<I>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = if pred[ABSOLUTE_POS] {
            on_true[ABSOLUTE_POS]
        } else {
            on_false[ABSOLUTE_POS]
        };
    }
}

#[cube(launch_unchecked)]
pub fn clamp_float<F: Float>(
    out: &mut Array<F>,
    input: &Array<F>,
    lower: &Array<F>,
    upper: &Array<F>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].clamp(lower[ABSOLUTE_POS], upper[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn conj_complex<C: ComplexCore>(out: &mut Array<C>, input: &Array<C>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].conj();
    }
}

/// `|x|` written without the `Abs` expander, whose result type is a
/// `WithScalar` that cannot be compared or recombined with `F`.
#[cube]
fn complex_div_abs<F: Float>(x: F) -> F {
    if x < F::new(0.0f32) {
        -x
    } else {
        x
    }
}

/// `sign(x)` preserving the sign of zero, as Julia Base uses it.
#[cube]
fn complex_div_sign<F: Float>(x: F) -> F {
    let mut s = F::new(1.0f32);
    if x == F::new(0.0f32) {
        s = x;
    } else if x < F::new(0.0f32) {
        s = F::new(-1.0f32);
    }
    s
}

/// One Baudin–Smith numerator/denominator term, ported from Julia Base
/// `robust_cdiv2`.
#[cube]
fn complex_div_term<F: Float>(a: F, b: F, c: F, d: F, r: F, t: F) -> F {
    if r != F::new(0.0f32) {
        let br = b * r;
        if br != F::new(0.0f32) {
            (a + br) * t
        } else {
            a * t + (b * t) * r
        }
    } else {
        (a + d * (b / c)) * t
    }
}

/// Scale-robust complex division over the interleaved real/imaginary parts.
///
/// `out`, `lhs` and `rhs` hold the parts of `Complex<f32>` / `Complex<f64>`
/// operands: element `i` is parts `2*i` (real) and `2*i + 1` (imaginary).
/// Ported from Julia Base `complex.jl` (`/`, `cdiv`, `robust_cdiv1/2`,
/// `scaling_cdiv`, `scaleargs_cdiv`) and matching the host reference
/// `strided_basic::complex_div`.
#[cube(launch_unchecked)]
pub fn div_complex_parts<F: Float>(out: &mut Array<F>, lhs: &Array<F>, rhs: &Array<F>) {
    let elements = out.len() / 2;
    if ABSOLUTE_POS < elements {
        let i = ABSOLUTE_POS * 2;
        let mut a = lhs[i];
        let mut b = lhs[i + 1];
        let mut c = rhs[i];
        let mut d = rhs[i + 1];
        if complex_div_abs::<F>(c) > F::max_value() || complex_div_abs::<F>(d) > F::max_value() {
            let a_finite = a == a && complex_div_abs::<F>(a) <= F::max_value();
            let b_finite = b == b && complex_div_abs::<F>(b) <= F::max_value();
            if a_finite && b_finite {
                out[i] = F::new(0.0f32) * complex_div_sign::<F>(a) * complex_div_sign::<F>(c);
                out[i + 1] = -F::new(0.0f32) * complex_div_sign::<F>(b) * complex_div_sign::<F>(d);
            } else {
                // `F::NAN` codegens to an undefined `NaN` identifier on CUDA.
                let nan = F::new(0.0f32) / F::new(0.0f32);
                out[i] = nan;
                out[i + 1] = nan;
            }
        } else {
            let abs_a = complex_div_abs::<F>(a);
            let abs_b = complex_div_abs::<F>(b);
            let abs_c = complex_div_abs::<F>(c);
            let abs_d = complex_div_abs::<F>(d);
            let ab = if abs_a >= abs_b { abs_a } else { abs_b };
            let cd = if abs_c >= abs_d { abs_c } else { abs_d };
            let half = F::new(0.5f32);
            let two = F::new(2.0f32);
            let half_ov = F::max_value() * half;
            let two_un_eps = F::MIN_POSITIVE * two / F::EPSILON;
            let mut scale = F::new(1.0f32);
            if ab >= half_ov || ab <= two_un_eps || cd >= half_ov || cd <= two_un_eps {
                let big = two / (F::EPSILON * F::EPSILON);
                if ab >= half_ov {
                    a = a * half;
                    b = b * half;
                    scale = scale * two;
                } else if ab <= two_un_eps {
                    a = a * big;
                    b = b * big;
                    scale = scale / big;
                }
                if cd >= half_ov {
                    c = c * half;
                    d = d * half;
                    scale = scale * half;
                } else if cd <= two_un_eps {
                    c = c * big;
                    d = d * big;
                    scale = scale * big;
                }
            }
            if complex_div_abs::<F>(d) <= complex_div_abs::<F>(c) {
                let r = d / c;
                let t = F::new(1.0f32) / (c + d * r);
                out[i] = complex_div_term::<F>(a, b, c, d, r, t) * scale;
                out[i + 1] = complex_div_term::<F>(b, -a, c, d, r, t) * scale;
            } else {
                let r = c / d;
                let t = F::new(1.0f32) / (c * r + d);
                out[i] = complex_div_term::<F>(b, a, d, c, r, t) * scale;
                out[i + 1] = -complex_div_term::<F>(a, -b, d, c, r, t) * scale;
            }
        }
    }
}
