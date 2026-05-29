//! Typed tensor operations.
//!
//! Operation families that are no longer part of core, including einsum, live
//! in their extension crates.

use tenferro_ops::broadcast::{broadcast_input_plan, broadcast_shape, broadcast_shapes};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, Error, Result, Tensor, TensorBackend, TensorRead, TensorScalar,
};

pub use tenferro_tensor::TypedTensor;

/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]);
/// let z = typed_tensor::add(&x, &y, &mut backend).unwrap();
/// ```
pub fn add<T: TensorScalar>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let (lhs, rhs) = broadcast_binary_read(lhs, rhs, backend)?;
    let out =
        backend.with_backend_session(|exec| exec.add_read(lhs.tensor_read(), rhs.tensor_read()))?;
    try_into_typed_result("add", out)
}

macro_rules! unary_fn {
    ($name:ident, $method:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro_cpu::CpuBackend;
        /// use tenferro_runtime::{typed_tensor, TypedTensor};
        /// # let mut backend = CpuBackend::new();
        /// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 4.0]);
        #[doc = concat!("let y = typed_tensor::", stringify!($name), "(&x, &mut backend).unwrap();")]
        /// ```
        pub fn $name<T: TensorScalar>(
            input: &TypedTensor<T>,
            backend: &mut impl TensorBackend,
        ) -> Result<TypedTensor<T>> {
            let out = backend.with_backend_session(|exec| exec.$method(T::tensor_read(input)))?;
            try_into_typed_result(stringify!($name), out)
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
        /// use tenferro_runtime::{typed_tensor, TypedTensor};
        /// # let mut backend = CpuBackend::new();
        /// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]);
        /// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]);
        #[doc = concat!("let z = typed_tensor::", stringify!($name), "(&x, &y, &mut backend).unwrap();")]
        /// ```
        pub fn $name<T: TensorScalar>(
            lhs: &TypedTensor<T>,
            rhs: &TypedTensor<T>,
            backend: &mut impl TensorBackend,
        ) -> Result<TypedTensor<T>> {
            let (lhs, rhs) = broadcast_binary_read(lhs, rhs, backend)?;
            let out =
                backend.with_backend_session(|exec| exec.$method(lhs.tensor_read(), rhs.tensor_read()))?;
            try_into_typed_result(stringify!($name), out)
        }
    };
}

binary_fn!(
    mul,
    mul_read,
    "Elementwise multiplication with NumPy-style broadcasting."
);
binary_fn!(
    div,
    div_read,
    "Elementwise division with NumPy-style broadcasting."
);
binary_fn!(
    pow,
    pow_read,
    "Elementwise power with NumPy-style broadcasting."
);
binary_fn!(
    maximum,
    maximum_read,
    "Elementwise maximum with NumPy-style broadcasting."
);
binary_fn!(
    minimum,
    minimum_read,
    "Elementwise minimum with NumPy-style broadcasting."
);

unary_fn!(neg, neg_read, "Elementwise negation.");
unary_fn!(abs, abs_read, "Elementwise absolute value.");
unary_fn!(sign, sign_read, "Elementwise sign.");
unary_fn!(conj, conj_read, "Elementwise complex conjugate.");
unary_fn!(exp, exp_read, "Elementwise exponential.");
unary_fn!(log, log_read, "Elementwise natural logarithm.");
unary_fn!(sin, sin_read, "Elementwise sine.");
unary_fn!(cos, cos_read, "Elementwise cosine.");
unary_fn!(tanh, tanh_read, "Elementwise hyperbolic tangent.");
unary_fn!(sqrt, sqrt_read, "Elementwise square root.");
unary_fn!(rsqrt, rsqrt_read, "Elementwise reciprocal square root.");
unary_fn!(expm1, expm1_read, "Elementwise `exp(x) - 1`.");
unary_fn!(log1p, log1p_read, "Elementwise `log(1 + x)`.");

/// Elementwise subtraction with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]);
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]);
/// let z = typed_tensor::sub(&x, &y, &mut backend).unwrap();
/// ```
pub fn sub<T: TensorScalar>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let (lhs, rhs) = broadcast_binary_read(lhs, rhs, backend)?;
    let neg_rhs = backend.with_backend_session(|exec| exec.neg_read(rhs.tensor_read()))?;
    let out = backend.with_backend_session(|exec| {
        exec.add_read(lhs.tensor_read(), TensorRead::from_tensor(&neg_rhs))
    })?;
    try_into_typed_result("sub", out)
}

/// Elementwise comparison with NumPy-style broadcasting.
///
/// The result is a bool tensor.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, CompareDir, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]);
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]);
/// let z = typed_tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();
/// assert_eq!(z.host_data(), &[true, false]);
/// ```
pub fn compare<T: TensorScalar>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: CompareDir,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<bool>> {
    let (lhs, rhs) = broadcast_binary_read(lhs, rhs, backend)?;
    let out = backend.with_backend_session(|exec| {
        exec.compare_read(lhs.tensor_read(), rhs.tensor_read(), &dir)
    })?;
    try_into_typed_result("compare", out)
}

/// Select values from `on_true` or `on_false` using a condition tensor.
///
/// This corresponds to NumPy `where(condition, x, y)`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, CompareDir, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]);
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]);
/// # let condition = typed_tensor::compare(&x, &y, CompareDir::Gt, &mut backend).unwrap();
/// let z = typed_tensor::where_select(&condition, &x, &y, &mut backend).unwrap();
/// ```
pub fn where_select<T: TensorScalar>(
    condition: &TypedTensor<bool>,
    on_true: &TypedTensor<T>,
    on_false: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let (condition, on_true, on_false) =
        broadcast_ternary_read(condition, on_true, on_false, backend)?;
    let out = backend.with_backend_session(|exec| {
        exec.select_read(
            condition.tensor_read(),
            on_true.tensor_read(),
            on_false.tensor_read(),
        )
    })?;
    try_into_typed_result("where_select", out)
}

/// Clamp values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-2.0, 4.0]);
/// # let lower = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0]);
/// # let upper = TypedTensor::<f64>::from_vec_col_major(vec![], vec![3.0]);
/// let z = typed_tensor::clamp(&x, &lower, &upper, &mut backend).unwrap();
/// ```
pub fn clamp<T: TensorScalar>(
    input: &TypedTensor<T>,
    lower: &TypedTensor<T>,
    upper: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let (input, lower, upper) = broadcast_ternary_read(input, lower, upper, backend)?;
    let out = backend.with_backend_session(|exec| {
        exec.clamp_read(
            input.tensor_read(),
            lower.tensor_read(),
            upper.tensor_read(),
        )
    })?;
    try_into_typed_result("clamp", out)
}

/// Matrix multiplication helper for rank-2 typed tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]);
/// # let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0; 6]);
/// let c = typed_tensor::matmul(&a, &b, &mut backend).unwrap();
/// ```
pub fn matmul<T: TensorScalar>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![a.shape().len() - 1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let out = backend.with_backend_session(|exec| {
        exec.dot_general_read(T::tensor_read(a), T::tensor_read(b), &config)
    })?;
    try_into_typed_result("matmul", out)
}

/// Sum elements across one or more axes.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_row_major(
///     vec![2, 3],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
/// );
/// let row_sums = typed_tensor::reduce_sum(&x, &[1], &mut backend).unwrap();
/// assert_eq!(row_sums.host_data(), &[6.0, 15.0]);
/// ```
pub fn reduce_sum<T: TensorScalar>(
    input: &TypedTensor<T>,
    axes: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let out =
        backend.with_backend_session(|exec| exec.reduce_sum_read(T::tensor_read(input), axes))?;
    try_into_typed_result("reduce_sum", out)
}

/// Reshape a typed tensor through the backend structural operation.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]);
/// let y = typed_tensor::reshape(&x, &[3, 2], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[3, 2]);
/// ```
pub fn reshape<T: TensorScalar>(
    input: &TypedTensor<T>,
    shape: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let out =
        backend.with_backend_session(|exec| exec.reshape_read(T::tensor_read(input), shape))?;
    try_into_typed_result("reshape", out)
}

/// Permute typed tensor axes through the backend structural operation.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]);
/// let y = typed_tensor::transpose(&x, &[1, 0], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[3, 2]);
/// ```
pub fn transpose<T: TensorScalar>(
    input: &TypedTensor<T>,
    perm: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let out =
        backend.with_backend_session(|exec| exec.transpose_read(T::tensor_read(input), perm))?;
    try_into_typed_result("transpose", out)
}

/// Broadcast a typed tensor into a larger shape.
///
/// `dims` maps each input axis to its output axis, following the concrete
/// backend `broadcast_in_dim` contract.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{typed_tensor, TypedTensor};
/// # let mut backend = CpuBackend::new();
/// let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]);
/// let matrix = typed_tensor::broadcast_in_dim(&row, &[2, 3], &[1], &mut backend).unwrap();
/// assert_eq!(matrix.shape(), &[2, 3]);
/// ```
pub fn broadcast_in_dim<T: TensorScalar>(
    input: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let out = backend.with_backend_session(|exec| {
        exec.broadcast_in_dim_read(T::tensor_read(input), shape, dims)
    })?;
    try_into_typed_result("broadcast_in_dim", out)
}

enum ReadInput<'a> {
    Borrowed(TensorRead<'a>),
    Owned(Tensor),
}

impl ReadInput<'_> {
    fn tensor_read(&self) -> TensorRead<'_> {
        match self {
            Self::Borrowed(read) => read.clone(),
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
        }
    }
}

fn broadcast_binary_read<'a, T: TensorScalar>(
    lhs: &'a TypedTensor<T>,
    rhs: &'a TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<(ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
    Ok((
        broadcast_to_read(lhs, &shape, backend)?,
        broadcast_to_read(rhs, &shape, backend)?,
    ))
}

fn broadcast_ternary_read<'a, C: TensorScalar, T: TensorScalar>(
    first: &'a TypedTensor<C>,
    second: &'a TypedTensor<T>,
    third: &'a TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<(ReadInput<'a>, ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(broadcast_error)?;
    Ok((
        broadcast_to_read(first, &shape, backend)?,
        broadcast_to_read(second, &shape, backend)?,
        broadcast_to_read(third, &shape, backend)?,
    ))
}

fn broadcast_to_read<'a, T: TensorScalar>(
    input: &'a TypedTensor<T>,
    target_shape: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<ReadInput<'a>> {
    if input.shape() == target_shape {
        return Ok(ReadInput::Borrowed(T::tensor_read(input)));
    }

    let plan = broadcast_input_plan(input.shape(), target_shape).map_err(broadcast_error)?;
    let source = if plan.source_shape == input.shape() {
        ReadInput::Borrowed(T::tensor_read(input))
    } else {
        let reshaped = backend.with_backend_session(|exec| {
            exec.reshape_read(T::tensor_read(input), &plan.source_shape)
        })?;
        ReadInput::Owned(reshaped)
    };
    let out = backend.with_backend_session(|exec| {
        exec.broadcast_in_dim_read(source.tensor_read(), target_shape, &plan.dims)
    })?;
    Ok(ReadInput::Owned(out))
}

fn broadcast_error(err: impl std::fmt::Display) -> Error {
    Error::backend_failure("broadcast", err.to_string())
}

fn try_into_typed_result<T: TensorScalar>(
    op: &'static str,
    tensor: Tensor,
) -> Result<TypedTensor<T>> {
    let actual = tensor.dtype();
    T::try_into_typed(tensor).ok_or(Error::DTypeMismatch {
        op,
        lhs: T::dtype(),
        rhs: actual,
    })
}
