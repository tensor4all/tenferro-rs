//! Typed tensor operations.
//!
//! Operation families that are no longer part of core, including einsum, live
//! in their extension crates.

use tenferro_tensor::{CompareDir, Error, Result, Tensor, TensorBackend, TensorScalar};

pub use tenferro_tensor::TypedTensor;

/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{typed_tensor, CpuBackend, TypedTensor};
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
    let out = crate::tensor::add(&erase(lhs), &erase(rhs), backend)?;
    try_into_typed_result("add", out)
}

macro_rules! unary_fn {
    ($name:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro_runtime::{typed_tensor, CpuBackend, TypedTensor};
        /// # let mut backend = CpuBackend::new();
        /// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 4.0]);
        #[doc = concat!("let y = typed_tensor::", stringify!($name), "(&x, &mut backend).unwrap();")]
        /// ```
        pub fn $name<T: TensorScalar>(
            input: &TypedTensor<T>,
            backend: &mut impl TensorBackend,
        ) -> Result<TypedTensor<T>> {
            let out = crate::tensor::$name(&erase(input), backend)?;
            try_into_typed_result(stringify!($name), out)
        }
    };
}

macro_rules! binary_fn {
    ($name:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro_runtime::{typed_tensor, CpuBackend, TypedTensor};
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
            let out = crate::tensor::$name(&erase(lhs), &erase(rhs), backend)?;
            try_into_typed_result(stringify!($name), out)
        }
    };
}

binary_fn!(
    mul,
    "Elementwise multiplication with NumPy-style broadcasting."
);
binary_fn!(div, "Elementwise division with NumPy-style broadcasting.");
binary_fn!(pow, "Elementwise power with NumPy-style broadcasting.");
binary_fn!(
    maximum,
    "Elementwise maximum with NumPy-style broadcasting."
);
binary_fn!(
    minimum,
    "Elementwise minimum with NumPy-style broadcasting."
);

unary_fn!(neg, "Elementwise negation.");
unary_fn!(abs, "Elementwise absolute value.");
unary_fn!(sign, "Elementwise sign.");
unary_fn!(conj, "Elementwise complex conjugate.");
unary_fn!(exp, "Elementwise exponential.");
unary_fn!(log, "Elementwise natural logarithm.");
unary_fn!(sin, "Elementwise sine.");
unary_fn!(cos, "Elementwise cosine.");
unary_fn!(tanh, "Elementwise hyperbolic tangent.");
unary_fn!(sqrt, "Elementwise square root.");
unary_fn!(rsqrt, "Elementwise reciprocal square root.");
unary_fn!(expm1, "Elementwise `exp(x) - 1`.");
unary_fn!(log1p, "Elementwise `log(1 + x)`.");

/// Elementwise subtraction with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{typed_tensor, CpuBackend, TypedTensor};
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
    let out = crate::tensor::sub(&erase(lhs), &erase(rhs), backend)?;
    try_into_typed_result("sub", out)
}

/// Elementwise comparison with NumPy-style broadcasting.
///
/// The result is a bool tensor.
///
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{typed_tensor, CompareDir, CpuBackend, TypedTensor};
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
    let out = crate::tensor::compare(&erase(lhs), &erase(rhs), dir, backend)?;
    try_into_typed_result("compare", out)
}

/// Select values from `on_true` or `on_false` using a condition tensor.
///
/// This corresponds to NumPy `where(condition, x, y)`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{typed_tensor, CompareDir, CpuBackend, TypedTensor};
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
    let out = crate::tensor::where_select(
        &erase(condition),
        &erase(on_true),
        &erase(on_false),
        backend,
    )?;
    try_into_typed_result("where_select", out)
}

/// Clamp values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{typed_tensor, CpuBackend, TypedTensor};
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
    let out = crate::tensor::clamp(&erase(input), &erase(lower), &erase(upper), backend)?;
    try_into_typed_result("clamp", out)
}

/// Matrix multiplication helper for rank-2 typed tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{typed_tensor, CpuBackend, TypedTensor};
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
    let out = crate::tensor::matmul(&erase(a), &erase(b), backend)?;
    try_into_typed_result("matmul", out)
}

fn erase<T: TensorScalar>(input: &TypedTensor<T>) -> Tensor {
    T::into_tensor(input.shape.clone(), input.host_data().to_vec())
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
