use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::eager::EagerTensor;
use crate::eager_ops::{broadcast_binary, broadcast_ternary};
use crate::error::Result;
use crate::CompareDir;

impl EagerTensor {
    /// Elementwise absolute value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![-1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.abs().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the dtype has no
    /// absolute-value implementation, or a typed backend/runtime-state error.
    pub fn abs(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Abs)
    }

    /// Elementwise complex conjugate.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.conj().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, -2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when conjugation is not
    /// defined for the dtype, or a typed backend/runtime-state error.
    pub fn conj(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Conj)
    }

    /// Elementwise sign.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![-2.0_f64, 3.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.sign().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[-1.0, 1.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when sign is not
    /// defined for the dtype, or a typed backend/runtime-state error.
    pub fn sign(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sign)
    }

    /// Elementwise natural logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.log().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[0.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype, or a typed backend/runtime-state error during execution.
    pub fn log(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Log)
    }

    /// Elementwise square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![4.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.sqrt().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype, or a typed backend/runtime-state error during execution.
    pub fn sqrt(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sqrt)
    }

    /// Elementwise reciprocal square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![4.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.rsqrt().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[0.5]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype, or a typed backend/runtime-state error during execution.
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Rsqrt)
    }

    /// Elementwise sine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.sin().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[0.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype, or a typed backend/runtime-state error during execution.
    pub fn sin(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sin)
    }

    /// Elementwise cosine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.cos().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype, or a typed backend/runtime-state error during execution.
    pub fn cos(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Cos)
    }

    /// Elementwise hyperbolic tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.tanh().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[0.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype, or a typed backend/runtime-state error during execution.
    pub fn tanh(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Tanh)
    }

    /// Elementwise `exp(x) - 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.expm1().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[0.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype, or a typed backend/runtime-state error during execution.
    pub fn expm1(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Expm1)
    }

    /// Elementwise `log(1 + x)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.log1p().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[0.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype, or a typed backend/runtime-state error during execution.
    pub fn log1p(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Log1p)
    }

    /// Elementwise division.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![8.0_f64, -6.0, 9.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 3.0, 3.0]).unwrap(), ctx.clone()).unwrap();
    /// let z = x.div(&y).unwrap();
    ///
    /// assert_eq!(z.materialized().unwrap().as_slice::<f64>().unwrap(), &[4.0, -2.0, 3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] or
    /// `ValidationError::DTypeMismatch` for
    /// incompatible operands, or a typed backend/runtime-state error. Addition
    /// does not have a zero-divisor failure; numerical zero-divisor errors are
    /// specific to division and remainder.
    pub fn div(&self, other: &Self) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("div", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Div)
    }

    /// Elementwise remainder.
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] or
    /// `ValidationError::DTypeMismatch` for
    /// incompatible operands, or a typed backend/runtime-state error.
    /// Subtraction does not have a zero-divisor failure; numerical
    /// zero-divisor errors are specific to division and remainder.
    pub fn rem(&self, other: &Self) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("rem", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Rem)
    }

    /// Elementwise power.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let base = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(), ctx.clone()).unwrap();
    /// let exp = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = base.pow(&exp).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[8.0, 9.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] or
    /// `ValidationError::DTypeMismatch` for
    /// incompatible operands, `NumericalFailure` for a checked invalid power,
    /// or a typed backend/runtime-state error.
    pub fn pow(&self, other: &Self) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("pow", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Pow)
    }

    /// Elementwise maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 5.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let z = x.maximum(&y).unwrap();
    ///
    /// assert_eq!(z.materialized().unwrap().as_slice::<f64>().unwrap(), &[3.0, 5.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] or
    /// `ValidationError::DTypeMismatch` for
    /// incompatible operands, or a typed unsupported/backend/runtime-state
    /// error.
    pub fn maximum(&self, other: &Self) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("maximum", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Maximum)
    }

    /// Elementwise minimum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 5.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let z = x.minimum(&y).unwrap();
    ///
    /// assert_eq!(z.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] or
    /// `ValidationError::DTypeMismatch` for
    /// incompatible operands, or a typed unsupported/backend/runtime-state
    /// error.
    pub fn minimum(&self, other: &Self) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("minimum", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Minimum)
    }

    /// Elementwise comparison.
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] or
    /// `ValidationError::DTypeMismatch` for
    /// incompatible operands, or a typed unsupported/backend/runtime-state
    /// error.
    pub fn compare(&self, other: &Self, dir: CompareDir) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("compare", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Compare(dir))
    }

    /// Select values from `on_true` or `on_false` using `condition`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let condition = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![false, true]).unwrap(), ctx.clone()).unwrap();
    /// let on_true = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![10.0_f64, 20.0]).unwrap(), ctx.clone()).unwrap();
    /// let on_false = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::select(&condition, &on_true, &on_false).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 20.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] when the three operands do not
    /// broadcast, `DTypeMismatch` for incompatible value dtypes, or a typed
    /// backend/runtime-state error.
    pub fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self> {
        Self::where_select(condition, on_true, on_false)
    }

    /// Select values from `on_true` or `on_false` using `condition`.
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] when the three operands do not
    /// broadcast, `DTypeMismatch` for incompatible value dtypes, or a typed
    /// backend/runtime-state error.
    pub fn where_select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self> {
        let (condition, on_true, on_false) =
            broadcast_ternary("where_select", condition, on_true, on_false)?;
        condition.ternary_op(&on_true, &on_false, StdTensorOp::Select)
    }

    /// Clamp values elementwise between lower and upper bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![-2.0_f64, 0.5, 5.0]).unwrap(), ctx.clone()).unwrap();
    /// let lower = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 0.0, 1.0]).unwrap(), ctx.clone()).unwrap();
    /// let upper = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.clamp(&lower, &upper).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[-1.0, 0.5, 4.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    ///
    /// Returns [`crate::error::Error::ContextMismatch`] for different eager runtimes,
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] when the three operands do not
    /// broadcast, `DTypeMismatch` for incompatible bounds, or a typed
    /// unsupported/backend/runtime-state error.
    pub fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self> {
        let (input, lower, upper) = broadcast_ternary("clamp", self, lower, upper)?;
        input.ternary_op(&lower, &upper, StdTensorOp::Clamp)
    }
}
