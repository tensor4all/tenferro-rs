//! Concrete tensor operation extension trait.
//!
//! `tenferro-tensor` owns storage and backend traits. This runtime crate
//! provides backend-parametric operation methods through [`TensorOpsExt`].

use tenferro_ops::broadcast::{broadcast_input_plan, broadcast_shape, broadcast_shapes};
use tenferro_tensor::{CompareDir, DType, DotGeneralConfig, Error, Result, TensorBackend};

use crate::TensorOpsExt;
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
    backend.with_backend_session(|exec| exec.convert(input, to))
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
    backend.with_backend_session(|exec| exec.cast(input, to))
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
        /// use tenferro_runtime::{Tensor, TensorOpsExt};
        /// # let mut backend = CpuBackend::new();
        /// # let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap();
        #[doc = concat!("let y = x.", stringify!($name), "(&mut backend).unwrap();")]
        /// ```
        fn $name(input: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
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
        /// use tenferro_runtime::{Tensor, TensorOpsExt};
        /// # let mut backend = CpuBackend::new();
        /// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
        /// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
        #[doc = concat!("let z = x.", stringify!($name), "(&y, &mut backend).unwrap();")]
        /// ```
        fn $name(lhs: &Tensor, rhs: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
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
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
/// # let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
/// let z = x.sub(&y, &mut backend).unwrap();
/// ```
fn sub(lhs: &Tensor, rhs: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
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
    let (condition, on_true, on_false) = broadcast_ternary(condition, on_true, on_false, backend)?;
    backend.with_backend_session(|exec| exec.select(&condition, &on_true, &on_false))
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
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// # let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
/// let c = a.matmul(&b, &mut backend).unwrap();
/// ```
fn matmul(a: &Tensor, b: &Tensor, backend: &mut impl TensorBackend) -> Result<Tensor> {
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
/// use tenferro_runtime::{Tensor, TensorOpsExt};
/// # let mut backend = CpuBackend::new();
/// # let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
/// let y = x.reshape(&[4], &mut backend).unwrap();
/// assert_eq!(y.shape(), &[4]);
/// ```
fn reshape(input: &Tensor, shape: &[usize], backend: &mut impl TensorBackend) -> Result<Tensor> {
    backend.with_backend_session(|exec| exec.reshape(input, shape))
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
    backend.with_backend_session(|exec| exec.transpose(input, perm))
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
