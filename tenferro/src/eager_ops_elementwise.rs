use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::eager::EagerTensor;
use crate::error::Result;

impl EagerTensor {
    /// Elementwise absolute value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![-1.0_f64, 2.0]), ctx.clone());
    /// let y = x.abs().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn abs(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Abs)
    }

    /// Elementwise complex conjugate.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![1.0_f64, -2.0]), ctx.clone());
    /// let y = x.conj().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, -2.0]);
    /// ```
    pub fn conj(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Conj)
    }

    /// Elementwise sign.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![-2.0_f64, 3.0]), ctx.clone());
    /// let y = x.sign().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[-1.0, 1.0]);
    /// ```
    pub fn sign(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sign)
    }

    /// Elementwise natural logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![1.0_f64]), ctx.clone());
    /// let y = x.log().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn log(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Log)
    }

    /// Elementwise square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![4.0_f64]), ctx.clone());
    /// let y = x.sqrt().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[2.0]);
    /// ```
    pub fn sqrt(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sqrt)
    }

    /// Elementwise reciprocal square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![4.0_f64]), ctx.clone());
    /// let y = x.rsqrt().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.5]);
    /// ```
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Rsqrt)
    }

    /// Elementwise sine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![0.0_f64]), ctx.clone());
    /// let y = x.sin().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn sin(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sin)
    }

    /// Elementwise cosine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![0.0_f64]), ctx.clone());
    /// let y = x.cos().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0]);
    /// ```
    pub fn cos(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Cos)
    }

    /// Elementwise hyperbolic tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![0.0_f64]), ctx.clone());
    /// let y = x.tanh().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn tanh(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Tanh)
    }

    /// Elementwise `exp(x) - 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![0.0_f64]), ctx.clone());
    /// let y = x.expm1().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn expm1(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Expm1)
    }

    /// Elementwise `log(1 + x)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![0.0_f64]), ctx.clone());
    /// let y = x.log1p().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn log1p(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Log1p)
    }

    /// Elementwise division.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![3], vec![8.0_f64, -6.0, 9.0]), ctx.clone());
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec(vec![3], vec![2.0_f64, 3.0, 3.0]), ctx.clone());
    /// let z = x.div(&y).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[4.0, -2.0, 3.0]);
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, StdTensorOp::Div)
    }

    /// Elementwise power.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let base = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![2.0_f64, 3.0]), ctx.clone());
    /// let exp = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![3.0_f64, 2.0]), ctx.clone());
    /// let y = base.pow(&exp).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[8.0, 9.0]);
    /// ```
    pub fn pow(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, StdTensorOp::Pow)
    }

    /// Elementwise maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![1.0_f64, 5.0]), ctx.clone());
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]), ctx.clone());
    /// let z = x.maximum(&y).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[3.0, 5.0]);
    /// ```
    pub fn maximum(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, StdTensorOp::Maximum)
    }

    /// Elementwise minimum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![1.0_f64, 5.0]), ctx.clone());
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]), ctx.clone());
    /// let z = x.minimum(&y).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    /// ```
    pub fn minimum(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, StdTensorOp::Minimum)
    }

    /// Select values from `on_true` or `on_false` using `condition`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let condition = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![0.0_f64, 1.0]), ctx.clone());
    /// let on_true = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![10.0_f64, 20.0]), ctx.clone());
    /// let on_false = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
    /// let y = EagerTensor::select(&condition, &on_true, &on_false).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 20.0]);
    /// ```
    pub fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self> {
        condition.ternary_op(on_true, on_false, StdTensorOp::Select)
    }

    /// Clamp values elementwise between lower and upper bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![3], vec![-2.0_f64, 0.5, 5.0]), ctx.clone());
    /// let lower = EagerTensor::from_tensor_in(Tensor::from_vec(vec![3], vec![-1.0_f64, 0.0, 1.0]), ctx.clone());
    /// let upper = EagerTensor::from_tensor_in(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 4.0]), ctx.clone());
    /// let y = x.clamp(&lower, &upper).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[-1.0, 0.5, 4.0]);
    /// ```
    pub fn clamp(&self, lower: &Self, upper: &Self) -> Result<Self> {
        self.ternary_op(lower, upper, StdTensorOp::Clamp)
    }
}
