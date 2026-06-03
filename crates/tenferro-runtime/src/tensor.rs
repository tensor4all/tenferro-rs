//! Concrete tensor operations.
//!
//! `tenferro-tensor` owns storage and backend traits. This runtime crate
//! provides backend-parametric helper functions over those tensor types.

use tenferro_ops::broadcast::{broadcast_input_plan, broadcast_shape, broadcast_shapes};
use tenferro_tensor::{CompareDir, DType, DotGeneralConfig, Error, Result, TensorBackend};

pub use tenferro_tensor::Tensor;

/// Convert a tensor to a different dtype.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, DType, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let y = tensor::convert(&x, DType::F32, &mut backend).unwrap();
/// ```
pub fn convert(input: &Tensor, to: DType, backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|exec| exec.convert(input, to))
}

/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// # let y = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let z = tensor::add(&x, &y, &mut backend).unwrap();
/// ```
pub fn add(lhs: &Tensor, rhs: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_binary(lhs, rhs, backend)?;
    backend.with_backend_session(|exec| exec.add(&lhs, &rhs))
}

macro_rules! unary_fn {
    ($name:ident, $method:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro_cpu::CpuBackend;
        /// use tenferro_runtime::{tensor, Tensor};
        /// # let mut backend = CpuBackend::new();
        /// # let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]);
        #[doc = concat!("let y = tensor::", stringify!($name), "(&x, &mut backend).unwrap();")]
        /// ```
        pub fn $name(input: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
            backend.with_backend_session(|exec| exec.$method(input))
        }
    };
}

macro_rules! binary_fn {
    ($name:ident, $method:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro_cpu::CpuBackend;
        /// use tenferro_runtime::{tensor, Tensor};
        /// # let mut backend = CpuBackend::new();
        /// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
        /// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
        #[doc = concat!("let z = tensor::", stringify!($name), "(&x, &y, &mut backend).unwrap();")]
        /// ```
        pub fn $name(
            lhs: &Tensor,
            rhs: &Tensor,
            backend: &mut impl TensorBackend,
        ) -> Result<Tensor> {
            let (lhs, rhs) = broadcast_binary(lhs, rhs, backend)?;
            backend.with_backend_session(|exec| exec.$method(&lhs, &rhs))
        }
    };
}

binary_fn!(
    mul,
    mul,
    "Elementwise multiplication with NumPy-style broadcasting."
);
binary_fn!(
    div,
    div,
    "Elementwise division with NumPy-style broadcasting."
);
binary_fn!(pow, pow, "Elementwise power with NumPy-style broadcasting.");
binary_fn!(
    maximum,
    maximum,
    "Elementwise maximum with NumPy-style broadcasting."
);
binary_fn!(
    minimum,
    minimum,
    "Elementwise minimum with NumPy-style broadcasting."
);

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
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
/// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
/// let z = tensor::sub(&x, &y, &mut backend).unwrap();
/// ```
pub fn sub(lhs: &Tensor, rhs: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_binary(lhs, rhs, backend)?;
    let neg_rhs = backend.with_backend_session(|exec| exec.neg(&rhs))?;
    backend.with_backend_session(|exec| exec.add(&lhs, &neg_rhs))
}

/// Elementwise comparison with NumPy-style broadcasting.
///
/// The result is a bool tensor.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, CompareDir, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
/// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
/// let z = tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();
/// assert_eq!(z.as_slice::<bool>().unwrap(), &[true, false]);
/// ```
pub fn compare(
    lhs: &Tensor,
    rhs: &Tensor,
    dir: CompareDir,
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_binary(lhs, rhs, backend)?;
    backend.with_backend_session(|exec| exec.compare(&lhs, &rhs, &dir))
}

/// Select values from `on_true` or `on_false` using a condition tensor.
///
/// This corresponds to NumPy `where(condition, x, y)`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, CompareDir, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
/// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
/// # let condition = tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();
/// let z = tensor::where_select(&condition, &x, &y, &mut backend).unwrap();
/// ```
pub fn where_select(
    condition: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    let (condition, on_true, on_false) = broadcast_ternary(condition, on_true, on_false, backend)?;
    backend.with_backend_session(|exec| exec.select(&condition, &on_true, &on_false))
}

/// Clamp values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![-2.0_f64, 4.0]);
/// # let lower = Tensor::from_vec_col_major(vec![], vec![0.0_f64]);
/// # let upper = Tensor::from_vec_col_major(vec![], vec![3.0_f64]);
/// let z = tensor::clamp(&x, &lower, &upper, &mut backend).unwrap();
/// ```
pub fn clamp(
    input: &Tensor,
    lower: &Tensor,
    upper: &Tensor,
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    let (input, lower, upper) = broadcast_ternary(input, lower, upper, backend)?;
    backend.with_backend_session(|exec| exec.clamp(&input, &lower, &upper))
}

/// Matrix multiplication helper for rank-2 tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// # let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
/// let c = tensor::matmul(&a, &b, &mut backend).unwrap();
/// ```
pub fn matmul(a: &Tensor, b: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![a.shape().len() - 1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    backend.with_backend_session(|exec| exec.dot_general(a, b, &config))
}

/// Reshape a tensor without changing element order.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let y = tensor::reshape(&x, &[4], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[4]);
/// ```
pub fn reshape(
    input: &Tensor,
    shape: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    backend.with_backend_session(|exec| exec.reshape(input, shape))
}

/// Permute tensor axes.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// let y = tensor::transpose(&x, &[1, 0], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[3, 2]);
/// ```
pub fn transpose(
    input: &Tensor,
    perm: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    backend.with_backend_session(|exec| exec.transpose(input, perm))
}

/// Sum a tensor over one or more axes.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{tensor, Tensor};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let y = tensor::reduce_sum(&x, &[0], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[2]);
/// ```
pub fn reduce_sum(
    input: &Tensor,
    axes: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    backend.with_backend_session(|exec| exec.reduce_sum(input, axes))
}

fn broadcast_binary(
    lhs: &Tensor,
    rhs: &Tensor,
    backend: &mut impl TensorBackend,
) -> Result<(Tensor, Tensor)> {
    let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
    Ok((
        broadcast_to(lhs, &shape, backend)?,
        broadcast_to(rhs, &shape, backend)?,
    ))
}

fn broadcast_ternary(
    first: &Tensor,
    second: &Tensor,
    third: &Tensor,
    backend: &mut impl TensorBackend,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(broadcast_error)?;
    Ok((
        broadcast_to(first, &shape, backend)?,
        broadcast_to(second, &shape, backend)?,
        broadcast_to(third, &shape, backend)?,
    ))
}

fn broadcast_to(
    input: &Tensor,
    target_shape: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    let input_shape = input.shape();
    if input_shape == target_shape {
        return Ok(input.clone());
    }

    let plan = broadcast_input_plan(input_shape, target_shape).map_err(broadcast_error)?;
    let source = if plan.source_shape == input_shape {
        input.clone()
    } else {
        backend.with_backend_session(|exec| exec.reshape(input, &plan.source_shape))?
    };
    backend.with_backend_session(|exec| exec.broadcast_in_dim(&source, target_shape, &plan.dims))
}

fn broadcast_error(err: impl std::fmt::Display) -> Error {
    Error::backend_failure("broadcast", err.to_string())
}
