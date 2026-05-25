//! Traced tensor operations.
//!
//! This module is the public namespace for operations that build traced tensor
//! graphs. The operation names stay independent of execution mode; the module
//! name identifies the tensor family.

use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::{CompareDir, DType, DotGeneralConfig};

pub use crate::traced::{TracedTensor, TracedTensorId};

/// Convert a traced tensor to a different dtype.
pub fn convert(input: &TracedTensor, to: DType) -> TracedTensor {
    input.convert(to)
}

/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// # let y = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let z = tenferro::traced_tensor::add(&x, &y);
/// ```
pub fn add(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    lhs.add(rhs)
}

macro_rules! unary_fn {
    ($name:ident, $method:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro::TracedTensor;
        /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]);
        #[doc = concat!(
                    "let y = tenferro::traced_tensor::",
                    stringify!($name),
                    "(&x);"
                )]
        /// ```
        pub fn $name(input: &TracedTensor) -> TracedTensor {
            input.$method()
        }
    };
}

macro_rules! binary_method_fn {
    ($name:ident, $method:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro::TracedTensor;
        /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
        /// # let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
        #[doc = concat!(
                    "let z = tenferro::traced_tensor::",
                    stringify!($name),
                    "(&x, &y);"
                )]
        /// ```
        pub fn $name(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
            lhs.$method(rhs)
        }
    };
}

binary_method_fn!(
    mul,
    mul,
    "Elementwise multiplication with NumPy-style broadcasting."
);
binary_method_fn!(
    div,
    div,
    "Elementwise division with NumPy-style broadcasting."
);
binary_method_fn!(pow, pow, "Elementwise power with NumPy-style broadcasting.");

unary_fn!(neg, neg, "Elementwise negation.");
unary_fn!(abs, abs, "Elementwise absolute value.");
unary_fn!(sign, sign, "Elementwise sign.");
unary_fn!(conj, conj, "Elementwise complex conjugate.");
unary_fn!(exp, exp, "Elementwise exponential.");
unary_fn!(log, log, "Elementwise natural logarithm.");
unary_fn!(sin, sin, "Elementwise sine.");
unary_fn!(cos, cos, "Elementwise cosine.");
unary_fn!(tanh, tanh, "Elementwise hyperbolic tangent.");
unary_fn!(sqrt, sqrt, "Elementwise square root.");
unary_fn!(rsqrt, rsqrt, "Elementwise reciprocal square root.");
unary_fn!(expm1, expm1, "Elementwise `exp(x) - 1`.");
unary_fn!(log1p, log1p, "Elementwise `log(1 + x)`.");

/// Elementwise subtraction with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
/// # let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
/// let z = tenferro::traced_tensor::sub(&x, &y);
/// ```
pub fn sub(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    add(lhs, &neg(rhs))
}

/// Elementwise maximum with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
/// # let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
/// let z = tenferro::traced_tensor::maximum(&x, &y);
/// ```
pub fn maximum(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    crate::traced::apply_broadcast_binary_op(StdTensorOp::Maximum, lhs, rhs)
}

/// Elementwise minimum with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
/// # let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
/// let z = tenferro::traced_tensor::minimum(&x, &y);
/// ```
pub fn minimum(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    crate::traced::apply_broadcast_binary_op(StdTensorOp::Minimum, lhs, rhs)
}

/// Elementwise comparison with NumPy-style broadcasting.
///
/// Current traced comparison follows the primitive numeric-mask dtype policy.
///
/// # Examples
///
/// ```rust
/// # use tenferro::{CompareDir, TracedTensor};
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
/// # let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
/// let z = tenferro::traced_tensor::compare(&x, &y, CompareDir::Gt);
/// ```
pub fn compare(lhs: &TracedTensor, rhs: &TracedTensor, dir: CompareDir) -> TracedTensor {
    crate::traced::apply_broadcast_binary_op(StdTensorOp::Compare(dir), lhs, rhs)
}

/// Select values from `on_true` or `on_false` using a condition tensor.
///
/// This corresponds to NumPy `where(condition, x, y)`.
///
/// # Examples
///
/// ```rust
/// # use tenferro::{CompareDir, TracedTensor};
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
/// # let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
/// # let condition = tenferro::traced_tensor::compare(&x, &y, CompareDir::Gt);
/// let z = tenferro::traced_tensor::where_select(&condition, &x, &y);
/// ```
pub fn where_select(
    condition: &TracedTensor,
    on_true: &TracedTensor,
    on_false: &TracedTensor,
) -> TracedTensor {
    crate::traced::apply_broadcast_ternary_op(StdTensorOp::Select, condition, on_true, on_false)
}

/// Clamp values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![-2.0_f64, 4.0]);
/// # let lower = TracedTensor::from_vec_col_major(vec![], vec![0.0_f64]);
/// # let upper = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
/// let z = tenferro::traced_tensor::clamp(&x, &lower, &upper);
/// ```
pub fn clamp(input: &TracedTensor, lower: &TracedTensor, upper: &TracedTensor) -> TracedTensor {
    crate::traced::apply_broadcast_ternary_op(StdTensorOp::Clamp, input, lower, upper)
}

/// Matrix multiplication helper for rank-2 traced tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// # let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
/// let c = tenferro::traced_tensor::matmul(&a, &b);
/// ```
pub fn matmul(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![a.rank - 1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    a.dot_general(b, config)
}
