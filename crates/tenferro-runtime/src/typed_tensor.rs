//! Typed tensor operation extension traits.
//!
//! Operation families that are no longer part of core, including einsum, live
//! in their extension crates.

use tenferro_ops::broadcast::{
    broadcast_input_plan, broadcast_shape, broadcast_shapes, BroadcastError,
};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{
    BackendSession, CompareDir, DType, Error, Result, Tensor, TensorBackend, TensorRead,
    TensorScalar, ValidationError,
};

use crate::{TypedTensorMaskOpsExt, TypedTensorOpsExt, TypedTensorSessionOpsExt};
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

impl<T: TensorScalar> TypedTensorSessionOpsExt<T> for TypedTensor<T> {
    fn add_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.add_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("add", out)
    }

    fn mul_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.mul_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("mul", out)
    }

    fn exp_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.exp_read(T::tensor_read(self))?;
        into_typed_result("exp", out)
    }

    fn reduce_sum_in(
        &self,
        axes: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.reduce_sum_read(T::tensor_read(self), axes)?;
        into_typed_result("reduce_sum", out)
    }

    fn sub_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.sub_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("sub", out)
    }

    fn div_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.div_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("div", out)
    }

    fn rem_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.rem_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("rem", out)
    }

    fn pow_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.pow_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("pow", out)
    }

    fn maximum_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.maximum_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("maximum", out)
    }

    fn minimum_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.minimum_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("minimum", out)
    }

    fn neg_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.neg_read(T::tensor_read(self))?;
        into_typed_result("neg", out)
    }

    fn abs_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.abs_read(T::tensor_read(self))?;
        into_typed_result("abs", out)
    }

    fn sign_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.sign_read(T::tensor_read(self))?;
        into_typed_result("sign", out)
    }

    fn conj_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.conj_read(T::tensor_read(self))?;
        into_typed_result("conj", out)
    }

    fn log_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.log_read(T::tensor_read(self))?;
        into_typed_result("log", out)
    }

    fn expm1_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.expm1_read(T::tensor_read(self))?;
        into_typed_result("expm1", out)
    }

    fn log1p_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.log1p_read(T::tensor_read(self))?;
        into_typed_result("log1p", out)
    }

    fn sin_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.sin_read(T::tensor_read(self))?;
        into_typed_result("sin", out)
    }

    fn cos_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.cos_read(T::tensor_read(self))?;
        into_typed_result("cos", out)
    }

    fn tanh_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.tanh_read(T::tensor_read(self))?;
        into_typed_result("tanh", out)
    }

    fn sqrt_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.sqrt_read(T::tensor_read(self))?;
        into_typed_result("sqrt", out)
    }

    fn rsqrt_in(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.rsqrt_read(T::tensor_read(self))?;
        into_typed_result("rsqrt", out)
    }

    fn compare_in(
        &self,
        rhs: &TypedTensor<T>,
        dir: CompareDir,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<bool>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.compare_read(lhs.tensor_read(), rhs.tensor_read(), &dir)?;
        into_typed_result("compare", out)
    }

    fn clamp_in(
        &self,
        lower: &TypedTensor<T>,
        upper: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (input, lower, upper) = broadcast_ternary_in_read(self, lower, upper, session)?;
        let out = session.clamp_read(
            input.tensor_read(),
            lower.tensor_read(),
            upper.tensor_read(),
        )?;
        into_typed_result("clamp", out)
    }

    fn matmul_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let config = matmul_config_for_shapes("matmul", self.shape(), rhs.shape())?;
        let out = session.dot_general_read(T::tensor_read(self), T::tensor_read(rhs), &config)?;
        into_typed_result("matmul", out)
    }

    fn reshape_in(
        &self,
        shape: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.reshape_read(T::tensor_read(self), shape)?;
        into_typed_result("reshape", out)
    }

    fn transpose_in(
        &self,
        perm: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.transpose_read(T::tensor_read(self), perm)?;
        into_typed_result("transpose", out)
    }

    fn broadcast_in_dim_in(
        &self,
        shape: &[usize],
        dims: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.broadcast_in_dim_read(T::tensor_read(self), shape, dims)?;
        into_typed_result("broadcast_in_dim", out)
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
    backend.with_backend_session(|session| lhs.add_in(rhs, session))
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
            backend.with_backend_session(|session| input.$method(session))
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
            backend.with_backend_session(|session| lhs.$method(rhs, session))
        }
    };
}

binary_fn!(
    div,
    div_in,
    "Elementwise division with NumPy-style broadcasting."
);
binary_fn!(
    rem,
    rem_in,
    "Elementwise remainder with NumPy-style broadcasting."
);
binary_fn!(
    pow,
    pow_in,
    "Elementwise power with NumPy-style broadcasting."
);
binary_fn!(
    maximum,
    maximum_in,
    "Elementwise maximum with NumPy-style broadcasting."
);
binary_fn!(
    minimum,
    minimum_in,
    "Elementwise minimum with NumPy-style broadcasting."
);

unary_fn!(neg, neg_in, "Elementwise negation.");
unary_fn!(abs, abs_in, "Elementwise absolute value.");
unary_fn!(sign, sign_in, "Elementwise sign.");
unary_fn!(conj, conj_in, "Elementwise complex conjugate.");
unary_fn!(log, log_in, "Elementwise natural logarithm.");
unary_fn!(sin, sin_in, "Elementwise sine.");
unary_fn!(cos, cos_in, "Elementwise cosine.");
unary_fn!(tanh, tanh_in, "Elementwise hyperbolic tangent.");
unary_fn!(sqrt, sqrt_in, "Elementwise square root.");
unary_fn!(rsqrt, rsqrt_in, "Elementwise reciprocal square root.");
unary_fn!(expm1, expm1_in, "Elementwise `exp(x) - 1`.");
unary_fn!(log1p, log1p_in, "Elementwise `log(1 + x)`.");

/// Elementwise multiplication with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
/// # let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
/// let z = x.mul(&y, &mut backend).unwrap();
/// ```
fn mul<T: TensorScalar>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    backend.with_backend_session(|session| lhs.mul_in(rhs, session))
}

/// Elementwise exponential.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 4.0]).unwrap();
/// let y = x.exp(&mut backend).unwrap();
/// ```
fn exp<T: TensorScalar>(
    input: &TypedTensor<T>,
    backend: &mut impl TensorBackend,
) -> Result<TypedTensor<T>> {
    backend.with_backend_session(|session| input.exp_in(session))
}

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
    backend.with_backend_session(|session| lhs.sub_in(rhs, session))
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
    backend.with_backend_session(|session| lhs.compare_in(rhs, dir, session))
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
    backend.with_backend_session(|session| input.clamp_in(lower, upper, session))
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
    backend.with_backend_session(|session| a.matmul_in(b, session))
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
    backend.with_backend_session(|session| input.reduce_sum_in(axes, session))
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
    backend.with_backend_session(|session| input.reshape_in(shape, session))
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
    backend.with_backend_session(|session| input.transpose_in(perm, session))
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
    backend.with_backend_session(|session| input.broadcast_in_dim_in(shape, dims, session))
}

// INVARIANT: this private adapter keeps borrowed reads borrowed and owns only
// the explicit fallback tensor; it is never exposed or cloned.
#[allow(clippy::large_enum_variant)]
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

fn broadcast_to_in_read<'a, T: TensorScalar>(
    input: &'a TypedTensor<T>,
    target_shape: &[usize],
    session: &mut dyn BackendSession,
) -> Result<ReadInput<'a>> {
    if input.shape() == target_shape {
        return Ok(ReadInput::Borrowed(T::tensor_read(input)));
    }

    let plan = broadcast_input_plan(input.shape(), target_shape).map_err(broadcast_error)?;
    let source = if plan.source_shape == input.shape() {
        ReadInput::Borrowed(T::tensor_read(input))
    } else {
        let reshaped = session.reshape_read(T::tensor_read(input), &plan.source_shape)?;
        ReadInput::Owned(reshaped)
    };
    let out = session.broadcast_in_dim_read(source.tensor_read(), target_shape, &plan.dims)?;
    Ok(ReadInput::Owned(out))
}

fn broadcast_binary_in_read<'a, T: TensorScalar>(
    lhs: &'a TypedTensor<T>,
    rhs: &'a TypedTensor<T>,
    session: &mut dyn BackendSession,
) -> Result<(ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
    Ok((
        broadcast_to_in_read(lhs, &shape, session)?,
        broadcast_to_in_read(rhs, &shape, session)?,
    ))
}

fn broadcast_ternary_in_read<'a, C: TensorScalar, T: TensorScalar>(
    first: &'a TypedTensor<C>,
    second: &'a TypedTensor<T>,
    third: &'a TypedTensor<T>,
    session: &mut dyn BackendSession,
) -> Result<(ReadInput<'a>, ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(broadcast_error)?;
    Ok((
        broadcast_to_in_read(first, &shape, session)?,
        broadcast_to_in_read(second, &shape, session)?,
        broadcast_to_in_read(third, &shape, session)?,
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

fn broadcast_error(err: BroadcastError) -> Error {
    match err {
        BroadcastError::IncompatibleBinary { lhs, rhs } => {
            Error::shape_mismatch("broadcast", lhs, rhs)
        }
        BroadcastError::IncompatibleInput { input, output } => {
            Error::shape_mismatch("broadcast", input, output)
        }
        BroadcastError::RankTooLarge { input, output } => {
            Error::rank_mismatch("broadcast", output.len(), input.len())
        }
    }
}

fn into_typed_result<T: TensorScalar>(op: &'static str, tensor: Tensor) -> Result<TypedTensor<T>> {
    let actual = tensor.dtype();
    T::into_typed(tensor).map_err(|_| {
        Error::validation(
            op,
            ValidationError::DTypeMismatch {
                expected: core_dtype(T::dtype()),
                actual: core_dtype(actual),
            },
        )
    })
}

fn core_dtype(dtype: DType) -> tenferro_tensor::core::DType {
    match dtype {
        DType::F32 => tenferro_tensor::core::DType::F32,
        DType::F64 => tenferro_tensor::core::DType::F64,
        DType::I32 => tenferro_tensor::core::DType::I32,
        DType::I64 => tenferro_tensor::core::DType::I64,
        DType::Bool => tenferro_tensor::core::DType::Bool,
        DType::C32 => tenferro_tensor::core::DType::C32,
        DType::C64 => tenferro_tensor::core::DType::C64,
    }
}
