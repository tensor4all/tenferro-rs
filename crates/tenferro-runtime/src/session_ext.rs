//! Engine-explicit operations on [`Tensor`].

use num_complex::Complex64;

use crate::{
    CompareDir, DType, DotGeneralConfig, Engine, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, Tensor,
};

/// AD-free tensor operations on [`Tensor`], run on a borrowed engine.
///
/// Every method takes the receiver first and the engine last, with the
/// arguments and config types of the eager `EagerSession` method of the same
/// name, so AD-free and eager code differ only in where the engine comes
/// from: the caller holds one `&mut dyn Engine` and borrows it for each call.
/// Nothing is entered and nothing is locked, so there is no entry error and a
/// single call needs one `?`. Borrow one engine for a whole sequence instead of
/// building one per operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
///
/// let mut engine = CpuEngine::serial();
/// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let y = {
///     let lower = x.tril(0, &mut engine)?;
///     lower.reduce_sum(None, &mut engine)
/// }?;
/// assert_eq!(y.as_slice::<f64>()?, &[7.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait TensorSessionOpsExt {
    /// Elementwise addition with NumPy-style broadcasting on an engine.
    ///
    /// The broadcast (reshape + `broadcast_in_dim`, or a copy when shapes
    /// already match) and the add itself all run in the caller's engine;
    /// this op never opens an engine of its own.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    /// let sum = a.add(&b, &mut engine)?;
    /// assert_eq!(sum.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with a
    /// [`ShapeMismatch`](tenferro_tensor::ValidationError::ShapeMismatch) or
    /// `DTypeMismatch` payload when operands are incompatible, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn add(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise multiplication with NumPy-style broadcasting on an engine.
    ///
    /// Like [`Self::add`], broadcast and multiply run on the one engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![4], vec![3.0_f64; 4]).unwrap();
    /// let product = a.mul(&b, &mut engine)?;
    /// assert_eq!(product.as_slice::<f64>().unwrap(), &[6.0; 4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn mul(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise exponential on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    /// let y = x.exp(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!((y[0] - 1.0).abs() < 1.0e-12);
    /// assert!((y[1] - std::f64::consts::E).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn exp(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Sum over the selected axes on an engine. `None` reduces every
    /// axis and `Some(&[])` keeps the input shape, as in the eager and traced
    /// reduction family.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let sums = x.reduce_sum(Some(&[1]), &mut engine)?;
    /// assert_eq!(sums.as_slice::<f64>().unwrap(), &[3.0, 3.0]);
    /// let total = x.reduce_sum(None, &mut engine)?;
    /// assert_eq!(total.as_slice::<f64>().unwrap(), &[6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// or `DuplicateAxis` for invalid reductions, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reduce_sum(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Convert to a different dtype using the checked conversion lattice on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{DType, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.convert(DType::C64, &mut engine)?;
    /// assert_eq!(y.dtype(), DType::C64);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// conversion is outside the checked lattice,
    /// [`tenferro_tensor::Error::Validation`] with `DTypeMismatch` or
    /// `InvalidArgument` for invalid tensor metadata, or
    /// [`tenferro_tensor::Error::BackendSource`] when the backend reports a
    /// typed failure.
    fn convert(&self, to: DType, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Cast to a different dtype using explicit lossy projection on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{DType, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.2_f64, -2.8]).unwrap();
    /// let y = x.cast(DType::I32, &mut engine)?;
    /// assert_eq!(y.as_slice::<i32>().unwrap(), &[1, -2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// requested cast is unsupported, [`tenferro_tensor::Error::Validation`]
    /// with `DTypeMismatch` or `InvalidArgument` for invalid tensor metadata,
    /// or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cast(&self, to: DType, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise subtraction with NumPy-style broadcasting on an engine.
    ///
    /// Like [`Self::add`], the broadcast and the subtraction run on the one
    /// engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    /// let y = a.sub(&b, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, -4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn sub(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise division with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 8.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let y = a.div(&b, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[2.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for shape/dtype incompatibility,
    /// [`tenferro_tensor::Error::Extension`] with a numerical classification
    /// for a detected zero divisor, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn div(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise remainder with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let y = a.rem(&b, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for shape/dtype incompatibility, a numerical
    /// [`tenferro_tensor::Error::Extension`] for a detected zero divisor, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn rem(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise power with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 2.0]).unwrap();
    /// let y = a.pow(&b, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[8.0, 9.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible metadata, a numerical
    /// [`tenferro_tensor::Error::Extension`] for a detected negative integer
    /// exponent, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn pow(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise maximum with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    /// let y = a.maximum(&b, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[2.0, 8.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn maximum(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise minimum with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    /// let y = a.minimum(&b, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn minimum(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise negation on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    /// let y = x.neg(&mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the dtype is not
    /// supported by the operation, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn neg(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise absolute value on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![-1.0_f64, 2.0]).unwrap();
    /// let y = x.abs(&mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sign on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    /// let y = x.sign(&mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, -1.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise complex conjugate on an engine.
    ///
    /// For real dtypes the conjugate is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    /// let y = x.conj(&mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, -2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise natural logarithm on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, std::f64::consts::E]).unwrap();
    /// let y = x.log(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `exp(x) - 1` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    /// let y = x.expm1(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - (std::f64::consts::E - 1.0)).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn expm1(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `log(1 + x)` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::E - 1.0]).unwrap();
    /// let y = x.log1p(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log1p(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise error function `erf(x)` on the engine, for real `F32`/`F64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    /// let y = x.erf(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert_eq!(y[0], 0.0);
    /// assert!((y[1] - 0.842_700_792_949_714_9).abs() < 1.0e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn erf(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sine on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::FRAC_PI_2]).unwrap();
    /// let y = x.sin(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sin(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise cosine on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::PI]).unwrap();
    /// let y = x.cos(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!((y[0] - 1.0).abs() < 1.0e-12);
    /// assert!((y[1] + 1.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cos(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise hyperbolic tangent on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    /// let y = x.tanh(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 0.7615941559557649).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn tanh(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise square root on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 9.0]).unwrap();
    /// let y = x.sqrt(&mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise reciprocal square root on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 1.0]).unwrap();
    /// let y = x.rsqrt(&mut engine)?;
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!((y[0] - 0.5).abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn rsqrt(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise comparison with NumPy-style broadcasting on an engine.
    ///
    /// The result is a bool tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{CompareDir, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    /// let y = a.compare(&b, CompareDir::Gt, &mut engine)?;
    /// assert_eq!(y.as_slice::<bool>().unwrap(), &[true, false]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible shape/dtype metadata, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn compare(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Select values from `on_true` or `on_false` using this tensor as condition on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let condition = Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    /// let on_true = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let on_false = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    /// let y = condition.where_select(&on_true, &on_false, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` when the condition and branches are incompatible, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn where_select(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Clamp values elementwise between lower and upper bounds on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![-2.0_f64, 4.0]).unwrap();
    /// let lower = Tensor::from_vec_col_major(vec![], vec![0.0_f64]).unwrap();
    /// let upper = Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let y = x.clamp(&lower, &upper, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[0.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` when bounds are incompatible with the input, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn clamp(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Rank-2 matrix multiplication on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    /// let c = a.matmul(&b, &mut engine)?;
    /// assert_eq!(c.shape(), &[2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch`,
    /// `ShapeMismatch`, or `DTypeMismatch` for incompatible matrices, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn matmul(&self, rhs: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Reshape without changing element order on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let y = x.reshape(&[4], &mut engine)?;
    /// assert_eq!(y.shape(), &[4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch`, `RankMismatch`, or `InvalidArgument` when element
    /// counts or ranks are invalid, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reshape(&self, shape: &[usize], session: &mut dyn Engine)
        -> tenferro_tensor::Result<Tensor>;
    /// Permute axes on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let y = x.transpose(&[1, 0], &mut engine)?;
    /// assert_eq!(y.shape(), &[3, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `InvalidPermutationLength`, `AxisOutOfBounds`, or `DuplicateAxis` for
    /// an invalid permutation, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn transpose(
        &self,
        perm: &[usize],
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Gather slices of this tensor at `indices` on the engine (StableHLO `gather`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{GatherConfig, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0])?;
    /// let indices = Tensor::from_vec_col_major(vec![2, 1], vec![2_i64, 0])?;
    /// let config = GatherConfig {
    ///     offset_dims: vec![],
    ///     collapsed_slice_dims: vec![0],
    ///     start_index_map: vec![0],
    ///     index_vector_dim: 1,
    ///     slice_sizes: vec![1],
    /// };
    /// let y = x.gather(&indices, config, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[30.0, 10.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid configuration, index dtype or out-of-range index, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn gather(
        &self,
        indices: &Tensor,
        config: GatherConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Scatter `updates` into a copy of this tensor at `indices` on the engine (StableHLO `scatter`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{ScatterConfig, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![4], vec![0.0_f64; 4])?;
    /// let indices = Tensor::from_vec_col_major(vec![2, 1], vec![1_i64, 3])?;
    /// let updates = Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0])?;
    /// let config = ScatterConfig {
    ///     update_window_dims: vec![],
    ///     inserted_window_dims: vec![0],
    ///     scatter_dims_to_operand_dims: vec![0],
    ///     index_vector_dim: 1,
    /// };
    /// let y = x.scatter(&indices, &updates, config, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[0.0, 5.0, 0.0, 7.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid configuration, index dtype or update shape, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn scatter(
        &self,
        indices: &Tensor,
        updates: &Tensor,
        config: ScatterConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Slice this tensor with explicit start, limit and stride per axis on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{SliceConfig, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let config = SliceConfig { starts: vec![1], limits: vec![3], strides: vec![1] };
    /// let y = x.slice(config, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[2.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for bounds or strides that do not fit the input, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn slice(
        &self,
        config: SliceConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Slice this tensor at runtime `starts` (an integer tensor) with static `sizes` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let starts = Tensor::from_vec_col_major(vec![1], vec![1_i64])?;
    /// let y = x.dynamic_slice(&starts, &[2], &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[2.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for a start-index dtype or rank mismatch or sizes larger than the input, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn dynamic_slice(
        &self,
        starts: &Tensor,
        sizes: &[usize],
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Pad this tensor with zeros (edge and interior padding per axis) on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{PadConfig, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let config = PadConfig {
    ///     edge_padding_low: vec![1],
    ///     edge_padding_high: vec![1],
    ///     interior_padding: vec![1],
    /// };
    /// let y = x.pad(config, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[0.0, 1.0, 0.0, 2.0, 0.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for a padding configuration whose length does not match the input rank, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn pad(&self, config: PadConfig, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Concatenate tensors along `axis` on an engine.
    ///
    /// This has no receiver, like the eager `EagerSession::concatenate`: call it as
    /// `Tensor::concatenate(&[&a, &b], axis, session)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// let b = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let y = Tensor::concatenate(&[&a, &b], 0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an empty input list, an axis out of range, or mismatched shapes or dtypes, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn concatenate(
        inputs: &[&Tensor],
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>
    where
        Self: Sized;
    /// Reverse the elements along `axes` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    /// let y = x.reverse(&[0], &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[3.0, 2.0, 1.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an out-of-range or repeated axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reverse(&self, axes: &[usize], session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Take the maximum over the selected axes on an engine. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 5.0, 3.0, 2.0])?;
    /// let y = x.reduce_max(Some(&[0]), &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[5.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an out-of-range or repeated axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    /// An unsupported dtype returns [`tenferro_tensor::Error::Unsupported`].
    fn reduce_max(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Take the minimum over the selected axes on an engine. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 5.0, 3.0, 2.0])?;
    /// let y = x.reduce_min(Some(&[0]), &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an out-of-range or repeated axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    /// An unsupported dtype returns [`tenferro_tensor::Error::Unsupported`].
    fn reduce_min(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Multiply over the selected axes on an engine. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 5.0, 3.0, 2.0])?;
    /// let y = x.reduce_prod(Some(&[0]), &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[5.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an out-of-range or repeated axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    /// An unsupported dtype returns [`tenferro_tensor::Error::Unsupported`].
    fn reduce_prod(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Sum elementwise squares over the selected axes on the engine (`f32`/`f64`). `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 5.0, 3.0, 2.0])?;
    /// let y = x.reduce_sum_squares(Some(&[0]), &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[26.0, 13.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an out-of-range or repeated axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    /// An unsupported dtype returns [`tenferro_tensor::Error::Unsupported`].
    fn reduce_sum_squares(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Broadcast this tensor into `shape`, mapping input axis `i` to output axis `dims[i]`, on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let y = x.broadcast_in_dim(&[2, 2], &[0], &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 2.0, 1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for a dimension mapping that does not fit the input or target shape, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn broadcast_in_dim(
        &self,
        shape: &[usize],
        dims: &[usize],
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Keep the lower triangle (on and below diagonal `k`) of the trailing matrix axes on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let y = x.tril(0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 2.0, 0.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an input of rank below 2, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn tril(&self, k: i64, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Keep the upper triangle (on and above diagonal `k`) of the trailing matrix axes on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let y = x.triu(0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 0.0, 3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an input of rank below 2, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn triu(&self, k: i64, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Extract the diagonal along `axis_a` and `axis_b` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let y = x.extract_diag(0, 1, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for equal or out-of-range axes, or axes of different extent, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn extract_diag(
        &self,
        axis_a: usize,
        axis_b: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Embed this tensor along the diagonal of `axis_a` and `axis_b` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let y = x.embed_diag(0, 1, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 0.0, 0.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for invalid diagonal axes, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn embed_diag(
        &self,
        axis_a: usize,
        axis_b: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Contract this tensor with `rhs` on the engine (StableHLO `dot_general`).
    ///
    /// The output layout is `[lhs free..., rhs free..., batch...]` (batch axes trail).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{DotGeneralConfig, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let lhs = Tensor::from_vec_col_major(vec![1, 2], vec![2.0_f64, 3.0])?;
    /// let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 5.0])?;
    /// let config = DotGeneralConfig {
    ///     lhs_contracting_dims: [1].as_slice().into(),
    ///     rhs_contracting_dims: [0].as_slice().into(),
    ///     lhs_batch_dims: [].as_slice().into(),
    ///     rhs_batch_dims: [].as_slice().into(),
    /// };
    /// let y = lhs.dot_general(&rhs, config, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[23.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for incompatible contraction or batch dimensions or dtypes, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn dot_general(
        &self,
        rhs: &Tensor,
        config: DotGeneralConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Contract with optional conjugation of either operand on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{DotGeneralConfig, Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// use num_complex::Complex64;
    /// let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(0.0, 1.0)])?;
    /// let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(0.0, 1.0)])?;
    /// let config = DotGeneralConfig {
    ///     lhs_contracting_dims: [1].as_slice().into(),
    ///     rhs_contracting_dims: [0].as_slice().into(),
    ///     lhs_batch_dims: [].as_slice().into(),
    ///     rhs_batch_dims: [].as_slice().into(),
    /// };
    /// let y = lhs.dot_general_with_conj(&rhs, config, true, false, &mut engine)?;
    /// assert_eq!(y.as_slice::<Complex64>()?, &[Complex64::new(1.0, 0.0)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for incompatible contraction or batch dimensions or dtypes, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn dot_general_with_conj(
        &self,
        rhs: &Tensor,
        config: DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Multiply by a real scalar on the engine, with the eager `scale_real` dtype rules.
    ///
    /// Integer dtypes round the factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let y = x.scale_real(2.0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[2.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `InvalidArgument` for a
    /// non-finite factor, an integer factor out of range, or an external dtype,
    /// or [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn scale_real(&self, factor: f64, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Multiply a complex tensor by a complex scalar on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// use num_complex::Complex64;
    /// let x = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])?;
    /// let y = x.scale_complex(Complex64::new(0.0, 1.0), &mut engine)?;
    /// assert_eq!(y.as_slice::<Complex64>()?, &[Complex64::new(-2.0, 1.0)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `InvalidArgument` when
    /// the input dtype is not complex, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn scale_complex(
        &self,
        factor: Complex64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Logistic sigmoid `1 / (1 + exp(-x))` on the engine, overflow-free.
    ///
    /// Evaluated as `1 / (1 + e)` for `x > 0` and `e / (1 + e)` otherwise, with
    /// `e = exp(-|x|)`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![-700.0_f64, 0.0, 1000.0])?;
    /// let y = x.sigmoid(&mut engine)?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[1], 0.5);
    /// assert!(y[0] > 0.0 && y[0] < 1e-300);
    /// assert_eq!(y[2], 1.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn sigmoid(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// SiLU (swish) `x * sigmoid(x)` on an engine.
    ///
    /// Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 0.0, 1.0])?;
    /// let y = x.silu(&mut engine)?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[1], 0.0);
    /// assert!((y[2] - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn silu(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Softplus `log(1 + exp(x))` on the engine, in the stable form `max(x, 0) + log1p(exp(-|x|))`.
    ///
    /// Real `F32`/`F64` only; never overflows.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![-1000.0_f64, 0.0, 1000.0])?;
    /// let y = x.softplus(&mut engine)?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[0], 0.0);
    /// assert!((y[1] - 2.0_f64.ln()).abs() < 1e-15);
    /// assert_eq!(y[2], 1000.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn softplus(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Exact GELU `x/2 * (1 + erf(x / sqrt(2)))` on an engine.
    ///
    /// Real `F32`/`F64` only (PyTorch `approximate="none"`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 0.0, 1.0])?;
    /// let y = x.gelu(&mut engine)?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[1], 0.0);
    /// assert!((y[2] - 0.841_344_746_068_542_9).abs() < 1e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn gelu(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// GELU tanh approximation on the engine (PyTorch `approximate="tanh"`).
    ///
    /// `x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`; real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 0.0, 1.0])?;
    /// let y = x.gelu_tanh(&mut engine)?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[1], 0.0);
    /// assert!((y[2] - 0.841_191_990_608_276_8).abs() < 1e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn gelu_tanh(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Arithmetic mean over `axes` on the engine (`None` reduces every axis).
    ///
    /// Float and complex dtypes. The sum is divided by the element count; a mean
    /// over zero elements is `NaN`, and `Some(&[])` is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let y = x.reduce_mean(Some(&[1]), &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[2.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for integer or
    /// `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `AxisOutOfBounds` or `DuplicateAxis` for invalid axes, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reduce_mean(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Max-subtracted softmax along `axis` on an engine.
    ///
    /// Real `F32`/`F64` only. A slice that is entirely `-inf` returns zeros
    /// instead of `NaN`; a `NaN` or `+inf` entry makes its slice `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, f64::NEG_INFINITY])?;
    /// let y = x.softmax(0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 0.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `AxisOutOfBounds` for an invalid axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn softmax(&self, axis: usize, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    /// Max-subtracted log-softmax along `axis` on an engine.
    ///
    /// Real `F32`/`F64` only. A slice that is entirely `-inf` returns `-inf`
    /// instead of `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0])?;
    /// let y = x.log_softmax(0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[-std::f64::consts::LN_2; 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `AxisOutOfBounds` for an invalid axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn log_softmax(&self, axis: usize, session: &mut dyn Engine)
        -> tenferro_tensor::Result<Tensor>;
    /// Softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// `mask` broadcasts to the input shape. Masked-out entries are `0` whatever
    /// their value; a slice with no unmasked entry is all zeros.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 1.0, f64::NAN])?;
    /// let mask = Tensor::from_vec_col_major(vec![3], vec![true, true, false])?;
    /// let y = x.masked_softmax(&mask, 0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[0.5, 0.5, 0.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `DTypeMismatch` for a non-`Bool` mask, `ShapeMismatch` for a mask that
    /// does not broadcast to the input, or `AxisOutOfBounds` for an invalid
    /// axis, or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn masked_softmax(
        &self,
        mask: &Tensor,
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Log-softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// Masked-out entries are `-inf`; a slice with no unmasked entry is all `-inf`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 1.0, 5.0])?;
    /// let mask = Tensor::from_vec_col_major(vec![3], vec![true, true, false])?;
    /// let y = x.masked_log_softmax(&mask, 0, &mut engine)?;
    /// let ln_half = -std::f64::consts::LN_2;
    /// assert_eq!(y.as_slice::<f64>()?, &[ln_half, ln_half, f64::NEG_INFINITY]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `DTypeMismatch` for a non-`Bool` mask, `ShapeMismatch` for a mask that
    /// does not broadcast to the input, or `AxisOutOfBounds` for an invalid
    /// axis, or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn masked_log_softmax(
        &self,
        mask: &Tensor,
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Layer normalization along `axis` with optional affine `weight` / `bias`, on an engine.
    ///
    /// `(x - mean) / sqrt(var + eps) * weight + bias` with the biased variance;
    /// `weight` and `bias` are rank-1 of length `shape[axis]`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?;
    /// let bias = Tensor::from_vec_col_major(vec![2], vec![10.0_f64, 10.0])?;
    /// let y = x.layer_norm(0, None, Some(&bias), 0.0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[9.0, 11.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `AxisOutOfBounds` for an invalid axis, `InvalidArgument` for a negative
    /// or non-finite `eps`, or `DTypeMismatch` / `ShapeMismatch` for a weight or
    /// bias that is not a same-dtype vector of the axis length, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn layer_norm(
        &self,
        axis: usize,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// RMS normalization along `axis` with optional affine `weight` / `bias`, on an engine.
    ///
    /// `x / sqrt(mean(x^2) + eps) * weight + bias`; `weight` and `bias` are rank-1
    /// of length `shape[axis]`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
    /// let weight = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 1.0])?;
    /// let y = x.rms_norm(0, Some(&weight), None, 0.0, &mut engine)?;
    /// let y = y.as_slice::<f64>()?;
    /// let rms = 12.5_f64.sqrt();
    /// assert!((y[0] - 6.0 / rms).abs() < 1e-15 && (y[1] - 4.0 / rms).abs() < 1e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `AxisOutOfBounds` for an invalid axis, `InvalidArgument` for a negative
    /// or non-finite `eps`, or `DTypeMismatch` / `ShapeMismatch` for a weight or
    /// bias that is not a same-dtype vector of the axis length, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn rms_norm(
        &self,
        axis: usize,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    /// NumPy-style `take_along_axis` over `gather`, on an engine.
    ///
    /// `out[.., i, ..] = self[.., indices[.., i, ..], ..]` along `axis`. `indices`
    /// (I32/I64) has the input's rank; every other dimension is either the
    /// input's extent (batch-varying indices) or `1` (the whole extent is taken).
    /// Indices must be in bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// // Per-batch row gather: out[i, j, b] = x[idx[i, b], j, b].
    /// let x = Tensor::from_vec_col_major(vec![2, 2, 2], (0..8).map(f64::from).collect::<Vec<_>>())?;
    /// let idx = Tensor::from_vec_col_major(vec![2, 1, 2], vec![1_i64, 0, 0, 0])?;
    /// let y = x.take_along_axis(&idx, 0, &mut engine)?;
    /// assert_eq!(y.as_slice::<f64>()?, &[1.0, 0.0, 3.0, 2.0, 4.0, 4.0, 6.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` or
    /// `ShapeMismatch` for incompatible index shapes, `AxisOutOfBounds` for an
    /// invalid axis, or `InvalidArgument` when taking from a zero-length axis;
    /// [`tenferro_tensor::Error::UnsupportedDType`] for a non-integer index
    /// dtype; or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn take_along_axis(
        &self,
        indices: &Tensor,
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
}
