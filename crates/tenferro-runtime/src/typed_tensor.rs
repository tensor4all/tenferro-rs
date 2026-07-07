//! Typed tensor operation extension traits.
//!
//! Operation families that are no longer part of core, including einsum, live
//! in their extension crates.

use tenferro_ops::broadcast::{broadcast_input_plan, broadcast_shape, broadcast_shapes};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{CompareDir, Error, Result, Tensor, TensorBackend, TensorRead, TensorScalar};

use crate::{TypedTensorMaskOpsExt, TypedTensorOpsExt};
use tenferro_tensor::TypedTensor;

impl<T: TensorScalar> TypedTensorOpsExt<T> for TypedTensor<T> {
    fn add<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        add(self, rhs, backend)
    }

    fn sub<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        sub(self, rhs, backend)
    }

    fn mul<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        mul(self, rhs, backend)
    }

    fn div<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        div(self, rhs, backend)
    }

    fn rem<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        rem(self, rhs, backend)
    }

    fn pow<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        pow(self, rhs, backend)
    }

    fn maximum<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        maximum(self, rhs, backend)
    }

    fn minimum<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        minimum(self, rhs, backend)
    }

    fn neg<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        neg(self, backend)
    }

    fn abs<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        abs(self, backend)
    }

    fn sign<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        sign(self, backend)
    }

    fn conj<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        conj(self, backend)
    }

    fn exp<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        exp(self, backend)
    }

    fn log<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        log(self, backend)
    }

    fn sin<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        sin(self, backend)
    }

    fn cos<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        cos(self, backend)
    }

    fn tanh<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        tanh(self, backend)
    }

    fn sqrt<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        sqrt(self, backend)
    }

    fn rsqrt<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        rsqrt(self, backend)
    }

    fn expm1<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        expm1(self, backend)
    }

    fn log1p<B: TensorBackend>(&self, backend: &mut B) -> Result<TypedTensor<T>> {
        log1p(self, backend)
    }

    fn compare<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        dir: CompareDir,
        backend: &mut B,
    ) -> Result<TypedTensor<bool>> {
        compare(self, rhs, dir, backend)
    }

    fn clamp<B: TensorBackend>(
        &self,
        lower: &TypedTensor<T>,
        upper: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        clamp(self, lower, upper, backend)
    }

    fn matmul<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        matmul(self, rhs, backend)
    }

    fn reduce_sum<B: TensorBackend>(
        &self,
        axes: &[usize],
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        reduce_sum(self, axes, backend)
    }

    fn reshape<B: TensorBackend>(
        &self,
        shape: &[usize],
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        reshape(self, shape, backend)
    }

    fn transpose<B: TensorBackend>(
        &self,
        perm: &[usize],
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        transpose(self, perm, backend)
    }

    fn broadcast_in_dim<B: TensorBackend>(
        &self,
        shape: &[usize],
        dims: &[usize],
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        broadcast_in_dim(self, shape, dims, backend)
    }
}

impl TypedTensorMaskOpsExt for TypedTensor<bool> {
    fn where_select<T: TensorScalar, B: TensorBackend>(
        &self,
        on_true: &TypedTensor<T>,
        on_false: &TypedTensor<T>,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        where_select(self, on_true, on_false, backend)
    }
}

/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
/// let z = x.add(&y, &mut backend).unwrap();
/// ```
fn add<T: TensorScalar>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let (lhs, rhs) = broadcast_binary_read(lhs, rhs, backend)?;
    let out =
        backend.with_backend_session(|exec| exec.add_read(lhs.tensor_read(), rhs.tensor_read()))?;
    into_typed_result("add", out)
}

macro_rules! unary_fn {
    ($name:ident, $method:ident, $summary:literal) => {
        #[doc = $summary]
        ///
        /// # Examples
        ///
        /// ```rust
        /// # use tenferro_cpu::CpuBackend;
        /// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
        /// # let mut backend = CpuBackend::new();
        /// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 4.0]).unwrap();
        #[doc = concat!("let y = x.", stringify!($name), "(&mut backend).unwrap();")]
        /// ```
        fn $name<T: TensorScalar>(
            input: &TypedTensor<T>,
            backend: &mut impl TensorBackend,
        ) -> Result<TypedTensor<T>> {
            let out = backend.with_backend_session(|exec| exec.$method(T::tensor_read(input)))?;
            into_typed_result(stringify!($name), out)
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
        /// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
        /// # let mut backend = CpuBackend::new();
        /// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
        /// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
        #[doc = concat!("let z = x.", stringify!($name), "(&y, &mut backend).unwrap();")]
        /// ```
        fn $name<T: TensorScalar>(
            lhs: &TypedTensor<T>,
            rhs: &TypedTensor<T>,
            backend: &mut impl TensorBackend,
        ) -> Result<TypedTensor<T>> {
            let (lhs, rhs) = broadcast_binary_read(lhs, rhs, backend)?;
            let out = backend
                .with_backend_session(|exec| exec.$method(lhs.tensor_read(), rhs.tensor_read()))?;
            into_typed_result(stringify!($name), out)
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
    rem,
    rem_read,
    "Elementwise remainder with NumPy-style broadcasting."
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
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
/// let z = x.sub(&y, &mut backend).unwrap();
/// ```
fn sub<T: TensorScalar>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let (lhs, rhs) = broadcast_binary_read(lhs, rhs, backend)?;
    let out =
        backend.with_backend_session(|exec| exec.sub_read(lhs.tensor_read(), rhs.tensor_read()))?;
    into_typed_result("sub", out)
}

/// Elementwise comparison with NumPy-style broadcasting.
///
/// The result is a bool tensor.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{CompareDir, TypedTensor, TypedTensorMaskOpsExt, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
/// let z = x.compare(&y, CompareDir::Gt, &mut backend).unwrap();
/// assert_eq!(z.host_data().unwrap(), &[true, false]);
/// ```
fn compare<T: TensorScalar>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: CompareDir,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<bool>> {
    let (lhs, rhs) = broadcast_binary_read(lhs, rhs, backend)?;
    let out = backend.with_backend_session(|exec| {
        exec.compare_read(lhs.tensor_read(), rhs.tensor_read(), &dir)
    })?;
    into_typed_result("compare", out)
}

/// Select values from `on_true` or `on_false` using a condition tensor.
///
/// This corresponds to NumPy `where(condition, x, y)`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{CompareDir, TypedTensor, TypedTensorMaskOpsExt, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
/// # let condition = x.compare(&y, CompareDir::Gt, &mut backend).unwrap();
/// let z = condition.where_select(&x, &y, &mut backend).unwrap();
/// ```
fn where_select<T: TensorScalar>(
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
    into_typed_result("where_select", out)
}

/// Clamp values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-2.0, 4.0]).unwrap();
/// # let lower = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0]).unwrap();
/// # let upper = TypedTensor::<f64>::from_vec_col_major(vec![], vec![3.0]).unwrap();
/// let z = x.clamp(&lower, &upper, &mut backend).unwrap();
/// ```
fn clamp<T: TensorScalar>(
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
    into_typed_result("clamp", out)
}

/// Matrix multiplication helper for rank-2 typed tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
/// # let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0; 6]).unwrap();
/// let c = a.matmul(&b, &mut backend).unwrap();
/// ```
fn matmul<T: TensorScalar>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let config = matmul_config_for_shapes("matmul", a.shape(), b.shape())?;
    let out = backend.with_backend_session(|exec| {
        exec.dot_general_read(T::tensor_read(a), T::tensor_read(b), &config)
    })?;
    into_typed_result("matmul", out)
}

/// Sum elements across one or more axes.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_col_major(
///     vec![2, 3],
///     vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
/// )?;
/// let row_sums = x.reduce_sum(&[1], &mut backend).unwrap();
/// assert_eq!(row_sums.host_data()?, &[6.0, 15.0]);
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
fn reduce_sum<T: TensorScalar>(
    input: &TypedTensor<T>,
    axes: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let out =
        backend.with_backend_session(|exec| exec.reduce_sum_read(T::tensor_read(input), axes))?;
    into_typed_result("reduce_sum", out)
}

/// Reshape a typed tensor through the backend structural operation.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
/// let y = x.reshape(&[3, 2], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[3, 2]);
/// ```
fn reshape<T: TensorScalar>(
    input: &TypedTensor<T>,
    shape: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let out =
        backend.with_backend_session(|exec| exec.reshape_read(T::tensor_read(input), shape))?;
    into_typed_result("reshape", out)
}

/// Permute typed tensor axes through the backend structural operation.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
/// let y = x.transpose(&[1, 0], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[3, 2]);
/// ```
fn transpose<T: TensorScalar>(
    input: &TypedTensor<T>,
    perm: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let out =
        backend.with_backend_session(|exec| exec.transpose_read(T::tensor_read(input), perm))?;
    into_typed_result("transpose", out)
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
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
/// let matrix = row.broadcast_in_dim(&[2, 3], &[1], &mut backend).unwrap();
/// assert_eq!(matrix.shape(), &[2, 3]);
/// ```
fn broadcast_in_dim<T: TensorScalar>(
    input: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    let out = backend.with_backend_session(|exec| {
        exec.broadcast_in_dim_read(T::tensor_read(input), shape, dims)
    })?;
    into_typed_result("broadcast_in_dim", out)
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

fn into_typed_result<T: TensorScalar>(op: &'static str, tensor: Tensor) -> Result<TypedTensor<T>> {
    let actual = tensor.dtype();
    T::into_typed(tensor).map_err(|_| Error::DTypeMismatch {
        op,
        lhs: T::dtype(),
        rhs: actual,
    })
}
