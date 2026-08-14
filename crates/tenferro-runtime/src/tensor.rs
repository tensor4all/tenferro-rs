//! Concrete tensor operation extension trait.
//!
//! `tenferro-tensor` owns storage and backend traits. This runtime crate
//! provides backend-parametric operation methods through [`TensorOpsExt`].

use tenferro_ops::broadcast::{
    broadcast_error_to_validation, broadcast_input_plan, broadcast_shape, broadcast_shapes,
};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{BackendSession, CompareDir, DType, Error, Result, TensorBackend};

use crate::{TensorOpsExt, TensorSessionOpsExt};
use tenferro_tensor::Tensor;

impl TensorOpsExt for Tensor {
    fn convert<B: TensorBackend>(&self, to: DType, backend: &mut B) -> Result<Tensor> {
        convert(self, to, backend)
    }

    fn cast<B: TensorBackend>(&self, to: DType, backend: &mut B) -> Result<Tensor> {
        cast(self, to, backend)
    }

    fn add<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        add(self, rhs, backend)
    }

    fn sub<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        sub(self, rhs, backend)
    }

    fn mul<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        mul(self, rhs, backend)
    }

    fn div<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        div(self, rhs, backend)
    }

    fn rem<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        rem(self, rhs, backend)
    }

    fn pow<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        pow(self, rhs, backend)
    }

    fn maximum<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        maximum(self, rhs, backend)
    }

    fn minimum<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        minimum(self, rhs, backend)
    }

    fn neg<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        neg(self, backend)
    }

    fn abs<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        abs(self, backend)
    }

    fn sign<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        sign(self, backend)
    }

    fn conj<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        conj(self, backend)
    }

    fn exp<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        exp(self, backend)
    }

    fn log<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        log(self, backend)
    }

    fn sin<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        sin(self, backend)
    }

    fn cos<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        cos(self, backend)
    }

    fn tanh<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        tanh(self, backend)
    }

    fn sqrt<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        sqrt(self, backend)
    }

    fn rsqrt<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        rsqrt(self, backend)
    }

    fn expm1<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        expm1(self, backend)
    }

    fn log1p<B: TensorBackend>(&self, backend: &mut B) -> Result<Tensor> {
        log1p(self, backend)
    }

    fn compare<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        backend: &mut B,
    ) -> Result<Tensor> {
        compare(self, rhs, dir, backend)
    }

    fn where_select<B: TensorBackend>(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        backend: &mut B,
    ) -> Result<Tensor> {
        where_select(self, on_true, on_false, backend)
    }

    fn clamp<B: TensorBackend>(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        backend: &mut B,
    ) -> Result<Tensor> {
        clamp(self, lower, upper, backend)
    }

    fn matmul<B: TensorBackend>(&self, rhs: &Tensor, backend: &mut B) -> Result<Tensor> {
        matmul(self, rhs, backend)
    }

    fn reshape<B: TensorBackend>(&self, shape: &[usize], backend: &mut B) -> Result<Tensor> {
        reshape(self, shape, backend)
    }

    fn transpose<B: TensorBackend>(&self, perm: &[usize], backend: &mut B) -> Result<Tensor> {
        transpose(self, perm, backend)
    }

    fn reduce_sum<B: TensorBackend>(&self, axes: &[usize], backend: &mut B) -> Result<Tensor> {
        reduce_sum(self, axes, backend)
    }
}

impl TensorSessionOpsExt for Tensor {
    fn add_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.add(&lhs, &rhs)
    }

    fn mul_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.mul(&lhs, &rhs)
    }

    fn exp_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.exp(self)
    }

    fn reduce_sum_in(&self, axes: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.reduce_sum(self, axes)
    }

    fn convert_in(&self, to: DType, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.convert(self, to)
    }

    fn cast_in(&self, to: DType, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.cast(self, to)
    }

    fn sub_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.sub(&lhs, &rhs)
    }

    fn div_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.div(&lhs, &rhs)
    }

    fn rem_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.rem(&lhs, &rhs)
    }

    fn pow_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.pow(&lhs, &rhs)
    }

    fn maximum_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.maximum(&lhs, &rhs)
    }

    fn minimum_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.minimum(&lhs, &rhs)
    }

    fn neg_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.neg(self)
    }

    fn abs_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.abs(self)
    }

    fn sign_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sign(self)
    }

    fn conj_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.conj(self)
    }

    fn log_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.log(self)
    }

    fn expm1_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.expm1(self)
    }

    fn log1p_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.log1p(self)
    }

    fn sin_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sin(self)
    }

    fn cos_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.cos(self)
    }

    fn tanh_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.tanh(self)
    }

    fn sqrt_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sqrt(self)
    }

    fn rsqrt_in(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.rsqrt(self)
    }

    fn compare_in(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.compare(&lhs, &rhs, &dir)
    }

    fn where_select_in(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (condition, on_true, on_false) =
            broadcast_ternary_in(self, on_true, on_false, session)?;
        session.select(&condition, &on_true, &on_false)
    }

    fn clamp_in(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (input, lower, upper) = broadcast_ternary_in(self, lower, upper, session)?;
        session.clamp(&input, &lower, &upper)
    }

    fn matmul_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let config = matmul_config_for_shapes("matmul", self.shape(), rhs.shape())?;
        session.dot_general(self, rhs, &config)
    }

    fn reshape_in(&self, shape: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.reshape(self, shape)
    }

    fn transpose_in(&self, perm: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.transpose(self, perm)
    }
}

/// Convert a tensor to a different dtype using the checked conversion lattice.
///
/// Use [`cast`] for explicit lossy dtype projection.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{DType, Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let y = x.convert(DType::C64, &mut backend).unwrap();
/// assert_eq!(y.dtype(), DType::C64);
/// ```
///
/// # Errors
///
/// Returns an error when the requested conversion is outside tenferro's checked
/// dtype-promotion lattice, or when the backend does not support the requested
/// conversion.
fn convert(input: &Tensor, to: DType, backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|session| input.convert_in(to, session))
}

/// Cast a tensor to a different dtype using explicit dtype projection.
///
/// Unlike [`convert`], `cast` may truncate, narrow precision, project complex
/// values to their real component, or use boolean truthiness where the backend
/// supports the requested projection.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{DType, Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![1.2_f64, -2.8]).unwrap();
/// let y = x.cast(DType::I32, &mut backend).unwrap();
/// assert_eq!(y.as_slice::<i32>().unwrap(), &[1, -2]);
/// ```
///
/// # Errors
///
/// Returns an error when the backend does not support the requested explicit
/// dtype projection.
fn cast(input: &Tensor, to: DType, backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|session| input.cast_in(to, session))
}

/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// # let y = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
/// let z = x.add(&y, &mut backend).unwrap();
/// ```
fn add(lhs: &Tensor, rhs: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
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
        /// use tenferro_runtime::{Tensor, TensorOpsExt};
        /// # let mut backend = CpuBackend::new();
        /// # let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap();
        #[doc = concat!("let y = x.", stringify!($name), "(&mut backend).unwrap();")]
        /// ```
        fn $name(input: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
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
        /// use tenferro_runtime::{Tensor, TensorOpsExt};
        /// # let mut backend = CpuBackend::new();
        /// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
        /// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
        #[doc = concat!("let z = x.", stringify!($name), "(&y, &mut backend).unwrap();")]
        /// ```
        fn $name(lhs: &Tensor, rhs: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
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
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
/// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
/// let z = x.mul(&y, &mut backend).unwrap();
/// ```
fn mul(lhs: &Tensor, rhs: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|session| lhs.mul_in(rhs, session))
}

/// Elementwise exponential.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap();
/// let y = x.exp(&mut backend).unwrap();
/// ```
fn exp(input: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|session| input.exp_in(session))
}

/// Elementwise subtraction with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
/// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
/// let z = x.sub(&y, &mut backend).unwrap();
/// ```
fn sub(lhs: &Tensor, rhs: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
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
/// use tenferro_runtime::{CompareDir, Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
/// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
/// let z = x.compare(&y, CompareDir::Gt, &mut backend).unwrap();
/// assert_eq!(z.as_slice::<bool>().unwrap(), &[true, false]);
/// ```
fn compare(
    lhs: &Tensor,
    rhs: &Tensor,
    dir: CompareDir,
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
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
/// use tenferro_runtime::{CompareDir, Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
/// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
/// # let condition = x.compare(&y, CompareDir::Gt, &mut backend).unwrap();
/// let z = condition.where_select(&x, &y, &mut backend).unwrap();
/// ```
fn where_select(
    condition: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    backend.with_backend_session(|session| condition.where_select_in(on_true, on_false, session))
}

/// Clamp values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![-2.0_f64, 4.0]).unwrap();
/// # let lower = Tensor::from_vec_col_major(vec![], vec![0.0_f64]).unwrap();
/// # let upper = Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
/// let z = x.clamp(&lower, &upper, &mut backend).unwrap();
/// ```
fn clamp(
    input: &Tensor,
    lower: &Tensor,
    upper: &Tensor,
    backend: &mut impl TensorBackend,
) -> Result<Tensor> {
    backend.with_backend_session(|session| input.clamp_in(lower, upper, session))
}

/// Matrix multiplication helper for rank-2 tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// # let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
/// let c = a.matmul(&b, &mut backend).unwrap();
/// ```
fn matmul(a: &Tensor, b: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|session| a.matmul_in(b, session))
}

/// Reshape a tensor without changing element order.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
/// let y = x.reshape(&[4], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[4]);
/// ```
fn reshape(input: &Tensor, shape: &[usize], backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|session| input.reshape_in(shape, session))
}

/// Permute tensor axes.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let y = x.transpose(&[1, 0], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[3, 2]);
/// ```
fn transpose(input: &Tensor, perm: &[usize], backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|session| input.transpose_in(perm, session))
}

/// Sum a tensor over one or more axes.
///
/// # Examples
///
/// ```rust
/// # use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
/// let y = x.reduce_sum(&[0], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[2]);
/// ```
fn reduce_sum(input: &Tensor, axes: &[usize], backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|session| input.reduce_sum_in(axes, session))
}

fn broadcast_to_in(
    input: &Tensor,
    target_shape: &[usize],
    session: &mut dyn BackendSession,
) -> Result<Tensor> {
    let input_shape = input.shape();
    if input_shape == target_shape {
        return input.duplicate();
    }

    let plan = broadcast_input_plan(input_shape, target_shape).map_err(broadcast_error)?;
    let source = if plan.source_shape == input_shape {
        input.duplicate()?
    } else {
        session.reshape(input, &plan.source_shape)?
    };
    session.broadcast_in_dim(&source, target_shape, &plan.dims)
}

fn broadcast_binary_in(
    lhs: &Tensor,
    rhs: &Tensor,
    session: &mut dyn BackendSession,
) -> Result<(Tensor, Tensor)> {
    let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
    Ok((
        broadcast_to_in(lhs, &shape, session)?,
        broadcast_to_in(rhs, &shape, session)?,
    ))
}

fn broadcast_ternary_in(
    first: &Tensor,
    second: &Tensor,
    third: &Tensor,
    session: &mut dyn BackendSession,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(broadcast_error)?;
    Ok((
        broadcast_to_in(first, &shape, session)?,
        broadcast_to_in(second, &shape, session)?,
        broadcast_to_in(third, &shape, session)?,
    ))
}

fn broadcast_error(err: tenferro_ops::broadcast::BroadcastError) -> Error {
    Error::validation("broadcast", broadcast_error_to_validation(err))
}
