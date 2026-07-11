use cubecl::prelude::*;

use crate::kernels::helpers::{flat_to_tensor_index, multi_to_tensor_index};

pub(crate) const COMPARE_EQ: usize = 0;
pub(crate) const COMPARE_LT: usize = 1;
pub(crate) const COMPARE_LE: usize = 2;
pub(crate) const COMPARE_GT: usize = 3;
pub(crate) const COMPARE_GE: usize = 4;
pub(crate) const MIXED_ADD: usize = 0;
pub(crate) const MIXED_SUB: usize = 1;
pub(crate) const MIXED_MUL: usize = 2;
pub(crate) const MIXED_DIV: usize = 3;

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

macro_rules! binary_float_int_complex_kernel {
    ($float_name:ident, $int_name:ident, $complex_name:ident, $op:tt) => {
        binary_float_complex_kernel!($float_name, $complex_name, $op);

        #[cube(launch_unchecked)]
        pub fn $int_name<I: Int>(out: &mut Array<I>, lhs: &Array<I>, rhs: &Array<I>) {
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
broadcast_multiply_kernel!(broadcast_multiply_int, Int);
broadcast_multiply_kernel!(broadcast_multiply_complex, ComplexCore);

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

binary_float_int_complex_kernel!(add_float, add_int, add_complex, +);
binary_float_int_complex_kernel!(sub_float, sub_int, sub_complex, -);
binary_float_int_complex_kernel!(mul_float, mul_int, mul_complex, *);
binary_float_complex_kernel!(div_float, div_complex, /);

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
scalar_binary_float_kernel!(scalar_rem_float, |x, y| x - (x / y).trunc() * y);

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
        let (out_re, out_im) = if mode == MIXED_ADD {
            (re + scalar, im)
        } else if mode == MIXED_SUB {
            if real_lhs {
                (scalar - re, zero - im)
            } else {
                (re - scalar, im)
            }
        } else if mode == MIXED_MUL {
            (re * scalar, im * scalar)
        } else if !real_lhs {
            (re / scalar, im / scalar)
        } else {
            let norm_sqr = re * re + im * im;
            (scalar * re / norm_sqr, -(scalar * im / norm_sqr))
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
        let minus_one = zero - I::new(1);
        if y == zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else if y == minus_one {
            out[ABSOLUTE_POS] = zero - x;
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
        let minus_one = zero - I::new(1);
        if y == zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else if y == minus_one {
            out[ABSOLUTE_POS] = zero;
        } else {
            let quotient = x / y;
            out[ABSOLUTE_POS] = x - quotient * y;
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
        let minus_one = zero - I::new(1);
        if y == zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else if y == minus_one {
            out[ABSOLUTE_POS] = zero - x;
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
        out[ABSOLUTE_POS] = x - (x / y).trunc() * y;
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
        let minus_one = zero - I::new(1);
        if y == zero {
            err[0] = 1;
            out[ABSOLUTE_POS] = zero;
        } else if y == minus_one {
            out[ABSOLUTE_POS] = zero;
        } else {
            let quotient = x / y;
            out[ABSOLUTE_POS] = x - quotient * y;
        }
    }
}
unary_both_kernel!(neg_float, neg_complex, |value| -value);
#[cube(launch_unchecked)]
pub fn neg_int<I: Int>(out: &mut Array<I>, input: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = I::new(0) - input[ABSOLUTE_POS];
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
        out[ABSOLUTE_POS] = if value < zero { zero - value } else { value };
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
pub fn sign_int<I: Int>(out: &mut Array<I>, input: &Array<I>) {
    if ABSOLUTE_POS < out.len() {
        let value = input[ABSOLUTE_POS];
        let zero = I::new(0);
        out[ABSOLUTE_POS] = if value == zero {
            zero
        } else if value > zero {
            I::new(1)
        } else {
            zero - I::new(1)
        };
    }
}

#[cube(launch_unchecked)]
pub fn maximum_float<F: Float>(out: &mut Array<F>, lhs: &Array<F>, rhs: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS].max(rhs[ABSOLUTE_POS]);
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
        out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS].min(rhs[ABSOLUTE_POS]);
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
                let remainder = exp - quotient * two;
                if remainder != zero {
                    acc = acc * base;
                }
                exp = quotient;
                if exp > zero {
                    base = base * base;
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
