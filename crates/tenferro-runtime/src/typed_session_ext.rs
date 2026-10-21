//! Engine-explicit operations on [`TypedTensor`].

use num_complex::Complex64;

use crate::{CompareDir, DotGeneralConfig, Engine, TensorScalar, TypedTensor};

/// AD-free tensor operations on [`TypedTensor`], run on a borrowed engine.
///
/// The methods mirror [`crate::TensorSessionOpsExt`] for a statically known
/// scalar type, with the engine last. Indexing, padding, concatenation and the
/// triangular/diagonal masks are included: they go through the `*_read` hooks,
/// so a typed tensor is borrowed rather than copied before the kernel.
/// Selection with a bool mask is
/// [`crate::TypedTensorMaskSessionOpsExt::where_select`].
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
///
/// let mut engine = CpuEngine::serial();
/// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -3.0])?;
/// let y = x.reduce_max(None, &mut engine)?;
/// assert_eq!(y.host_data()?, &[1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait TypedTensorSessionOpsExt<T: TensorScalar> {
    /// Elementwise addition with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
    /// let sum = a.add(&b, &mut engine)?;
    /// assert_eq!(sum.host_data().unwrap(), &[4.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn add(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise addition into a caller-owned output on an engine.
    ///
    /// The caller keeps the output's storage, so the operation allocates nothing for
    /// it: this is the caller-output route [`Self::add`] cannot offer, the one the
    /// dispatch floor in #2139 is measured against. `out` is written in place and must
    /// have the broadcast result shape with a write-injective layout; the session
    /// rejects anything else before writing, leaving `out` unchanged on failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
    /// let mut out = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 0.0]).unwrap();
    ///  a.add_into(&b, &mut out, &mut engine)?;
    /// assert_eq!(out.host_data().unwrap(), &[4.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands or an output of the wrong shape,
    /// `ValidationError::OverlappingMutableLayout` for a non-injective output layout,
    /// or [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn add_into(
        &self,
        rhs: &TypedTensor<T>,
        out: &mut TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<()>;
    /// Elementwise multiplication with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![2.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![3.0; 4]).unwrap();
    /// let product = a.mul(&b, &mut engine)?;
    /// assert_eq!(product.host_data().unwrap(), &[6.0; 4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn mul(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise exponential on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = x.exp(&mut engine)?;
    /// let y = y.host_data().unwrap();
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
    fn exp(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Sum over the selected axes on an engine. `None` reduces every
    /// axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let sums = x.reduce_sum(Some(&[1]), &mut engine)?;
    /// assert_eq!(sums.host_data().unwrap(), &[3.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// for an axis outside the input rank or `DuplicateAxis` when `axes`
    /// repeats an axis, or [`tenferro_tensor::Error::BackendSource`] for a
    /// typed backend failure.
    fn reduce_sum(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise subtraction with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = a.sub(&b, &mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[1.0, -4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn sub(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise division with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 8.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let y = a.div(&b, &mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[2.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected zero divisor, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn div(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise remainder with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![5.0, 7.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let y = a.rem(&b, &mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[1.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected zero divisor, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn rem(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise power with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 2.0]).unwrap();
    /// let y = a.pow(&b, &mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[8.0, 9.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected negative integer exponent, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn pow(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise maximum with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = a.maximum(&b, &mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[2.0, 8.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn maximum(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise minimum with NumPy-style broadcasting on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = a.minimum(&b, &mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[1.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn minimum(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise negation on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = x.neg(&mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[-1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn neg(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise absolute value on an engine.
    ///
    /// The result has the real counterpart dtype `T::Real`: complex magnitude
    /// is real, and real or integer inputs keep their own dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-1.0, 2.0])?;
    /// let y = x.abs(&mut engine)?;
    /// assert_eq!(y.host_data()?, &[1.0, 2.0]);
    ///
    /// let z = TypedTensor::<Complex64>::from_vec_col_major(vec![1], vec![Complex64::new(3.0, 4.0)])?;
    /// let magnitude: TypedTensor<f64> = z.abs(&mut engine)?;
    /// assert_eq!(magnitude.host_data()?, &[5.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T::Real>>;
    /// Elementwise sign on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = x.sign(&mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[1.0, -1.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise complex conjugate on an engine.
    ///
    /// For real dtypes the conjugate is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = x.conj(&mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[1.0, -2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise natural logarithm on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, std::f64::consts::E]).unwrap();
    /// let y = x.log(&mut engine)?;
    /// let y = y.host_data().unwrap();
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
    fn log(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `exp(x) - 1` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = x.expm1(&mut engine)?;
    /// let y = y.host_data().unwrap();
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
    fn expm1(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `log(1 + x)` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::E - 1.0]).unwrap();
    /// let y = x.log1p(&mut engine)?;
    /// let y = y.host_data().unwrap();
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
    fn log1p(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise error function `erf(x)` on the engine, for real `f32`/`f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = x.erf(&mut engine)?;
    /// let y = y.host_data().unwrap();
    /// assert_eq!(y[0], 0.0);
    /// assert!((y[1] - 0.842_700_792_949_714_9).abs() < 1.0e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for a complex or
    /// integer element type, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn erf(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sine on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::FRAC_PI_2]).unwrap();
    /// let y = x.sin(&mut engine)?;
    /// let y = y.host_data().unwrap();
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
    fn sin(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise cosine on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::PI]).unwrap();
    /// let y = x.cos(&mut engine)?;
    /// let y = y.host_data().unwrap();
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
    fn cos(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise hyperbolic tangent on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = x.tanh(&mut engine)?;
    /// let y = y.host_data().unwrap();
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
    fn tanh(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise square root on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 9.0]).unwrap();
    /// let y = x.sqrt(&mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[2.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise reciprocal square root on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 1.0]).unwrap();
    /// let y = x.rsqrt(&mut engine)?;
    /// let y = y.host_data().unwrap();
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
    fn rsqrt(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise comparison with NumPy-style broadcasting on an engine.
    ///
    /// The result is a bool typed tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{CompareDir, TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = a.compare(&b, CompareDir::Gt, &mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[true, false]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when broadcasting the operands is
    /// impossible, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn compare(
        &self,
        rhs: &TypedTensor<T>,
        dir: CompareDir,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<bool>>;
    /// Clamp values elementwise between lower and upper bounds on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-2.0, 4.0]).unwrap();
    /// let lower = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0]).unwrap();
    /// let upper = TypedTensor::<f64>::from_vec_col_major(vec![], vec![3.0]).unwrap();
    /// let y = x.clamp(&lower, &upper, &mut engine)?;
    /// assert_eq!(y.host_data().unwrap(), &[0.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when a bound cannot broadcast to
    /// the input, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn clamp(
        &self,
        lower: &TypedTensor<T>,
        upper: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Rank-2 matrix multiplication on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0; 6]).unwrap();
    /// let c = a.matmul(&b, &mut engine)?;
    /// assert_eq!(c.shape(), &[2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` when
    /// either operand is not rank two or `ShapeMismatch::ContractedDimensions`
    /// when the inner dimensions differ, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn matmul(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Reshape through the backend structural operation on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let y = x.reshape(&[3, 2], &mut engine)?;
    /// assert_eq!(y.shape(), &[3, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::ReshapeElementCount` when the element counts differ,
    /// `IntegerOverflow` when shape arithmetic overflows, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reshape(
        &self,
        shape: &[usize],
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Permute axes through the backend structural operation on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let y = x.transpose(&[1, 0], &mut engine)?;
    /// assert_eq!(y.shape(), &[3, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `InvalidPermutationLength` when `perm` has the wrong length,
    /// `AxisOutOfBounds` for an invalid axis, or `DuplicateAxis` for a
    /// repeated axis, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn transpose(
        &self,
        perm: &[usize],
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Broadcast into a larger shape on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    /// let matrix = row.broadcast_in_dim(&[2, 3], &[1], &mut engine)?;
    /// assert_eq!(matrix.shape(), &[2, 3]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` when
    /// `dims` does not match the input rank, `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid mapping, or
    /// `ShapeMismatch::IncompatibleShapes` when known dimensions cannot
    /// broadcast. [`tenferro_tensor::Error::BackendSource`] reports a typed
    /// backend failure.
    fn broadcast_in_dim(
        &self,
        shape: &[usize],
        dims: &[usize],
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Take the maximum over the selected axes on an engine. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0])?;
    /// let y = x.reduce_max(Some(&[0]), &mut engine)?;
    /// assert_eq!(y.host_data()?, &[5.0, 3.0]);
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
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Take the minimum over the selected axes on an engine. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0])?;
    /// let y = x.reduce_min(Some(&[0]), &mut engine)?;
    /// assert_eq!(y.host_data()?, &[1.0, 2.0]);
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
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Multiply over the selected axes on an engine. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0])?;
    /// let y = x.reduce_prod(Some(&[0]), &mut engine)?;
    /// assert_eq!(y.host_data()?, &[5.0, 6.0]);
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
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Sum elementwise squares over the selected axes on the engine (`f32`/`f64`). `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0])?;
    /// let y = x.reduce_sum_squares(Some(&[0]), &mut engine)?;
    /// assert_eq!(y.host_data()?, &[26.0, 13.0]);
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
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Contract this tensor with `rhs` on the engine (StableHLO `dot_general`).
    ///
    /// The output layout is `[lhs free..., rhs free..., batch...]` (batch axes trail).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// use tenferro_runtime::DotGeneralConfig;
    /// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![1, 2], vec![2.0, 3.0])?;
    /// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 5.0])?;
    /// let config = DotGeneralConfig {
    ///     lhs_contracting_dims: [1].as_slice().into(),
    ///     rhs_contracting_dims: [0].as_slice().into(),
    ///     lhs_batch_dims: [].as_slice().into(),
    ///     rhs_batch_dims: [].as_slice().into(),
    /// };
    /// let y = lhs.dot_general(&rhs, config, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[23.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for incompatible contraction or batch dimensions, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn dot_general(
        &self,
        rhs: &TypedTensor<T>,
        config: DotGeneralConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Contract with optional conjugation of either operand on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// use num_complex::Complex64;
    /// use tenferro_runtime::DotGeneralConfig;
    /// let lhs = TypedTensor::<Complex64>::from_vec_col_major(vec![1, 1], vec![Complex64::new(0.0, 1.0)])?;
    /// let rhs = TypedTensor::<Complex64>::from_vec_col_major(vec![1, 1], vec![Complex64::new(0.0, 1.0)])?;
    /// let config = DotGeneralConfig {
    ///     lhs_contracting_dims: [1].as_slice().into(),
    ///     rhs_contracting_dims: [0].as_slice().into(),
    ///     lhs_batch_dims: [].as_slice().into(),
    ///     rhs_batch_dims: [].as_slice().into(),
    /// };
    /// let y = lhs.dot_general_with_conj(&rhs, config, true, false, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[Complex64::new(1.0, 0.0)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for incompatible contraction or batch dimensions, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn dot_general_with_conj(
        &self,
        rhs: &TypedTensor<T>,
        config: DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Multiply by a real scalar on the engine, with the eager `scale_real` dtype rules.
    ///
    /// Integer dtypes round the factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// let y = x.scale_real(2.0, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[2.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `InvalidArgument` for a
    /// non-finite factor or an integer factor out of range, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn scale_real(
        &self,
        factor: f64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Multiply a complex tensor by a complex scalar on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// use num_complex::Complex64;
    /// let x = TypedTensor::<Complex64>::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])?;
    /// let y = x.scale_complex(Complex64::new(0.0, 1.0), &mut engine)?;
    /// assert_eq!(y.host_data()?, &[Complex64::new(-2.0, 1.0)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `InvalidArgument` when
    /// `T` is not complex, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn scale_complex(
        &self,
        factor: Complex64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Logistic sigmoid `1 / (1 + exp(-x))` on the engine, overflow-free.
    ///
    /// Evaluated as `1 / (1 + e)` for `x > 0` and `e / (1 + e)` otherwise, with
    /// `e = exp(-|x|)`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-700.0, 0.0, 1000.0])?;
    /// let y = x.sigmoid(&mut engine)?;
    /// let y = y.host_data()?;
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
    fn sigmoid(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// SiLU (swish) `x * sigmoid(x)` on an engine.
    ///
    /// Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-1.0, 0.0, 1.0])?;
    /// let y = x.silu(&mut engine)?;
    /// let y = y.host_data()?;
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
    fn silu(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Softplus `log(1 + exp(x))` on the engine, in the stable form `max(x, 0) + log1p(exp(-|x|))`.
    ///
    /// Real `F32`/`F64` only; never overflows.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-1000.0, 0.0, 1000.0])?;
    /// let y = x.softplus(&mut engine)?;
    /// let y = y.host_data()?;
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
    fn softplus(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Exact GELU `x/2 * (1 + erf(x / sqrt(2)))` on an engine.
    ///
    /// Real `F32`/`F64` only (PyTorch `approximate="none"`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-1.0, 0.0, 1.0])?;
    /// let y = x.gelu(&mut engine)?;
    /// let y = y.host_data()?;
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
    fn gelu(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// GELU tanh approximation on the engine (PyTorch `approximate="tanh"`).
    ///
    /// `x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`; real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-1.0, 0.0, 1.0])?;
    /// let y = x.gelu_tanh(&mut engine)?;
    /// let y = y.host_data()?;
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
    fn gelu_tanh(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Arithmetic mean over `axes` on the engine (`None` reduces every axis).
    ///
    /// Float and complex dtypes. The sum is divided by the element count; a mean
    /// over zero elements is `NaN`, and `Some(&[])` is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let y = x.reduce_mean(None, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[2.5]);
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
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Max-subtracted softmax along `axis` on an engine.
    ///
    /// Real `F32`/`F64` only. A slice that is entirely `-inf` returns zeros
    /// instead of `NaN`; a `NaN` or `+inf` entry makes its slice `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 1.0])?;
    /// let y = x.softmax(0, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[0.5, 0.5]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `AxisOutOfBounds` for an invalid axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn softmax(
        &self,
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Max-subtracted log-softmax along `axis` on an engine.
    ///
    /// Real `F32`/`F64` only. A slice that is entirely `-inf` returns `-inf`
    /// instead of `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 1.0])?;
    /// let y = x.log_softmax(0, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[-std::f64::consts::LN_2; 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDType`] for complex,
    /// integer, or `Bool` input, [`tenferro_tensor::Error::Validation`] with
    /// `AxisOutOfBounds` for an invalid axis, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn log_softmax(
        &self,
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// `mask` broadcasts to the input shape. Masked-out entries are `0` whatever
    /// their value; a slice with no unmasked entry is all zeros.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
    /// let mask = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![false, false])?;
    /// let y = x.masked_softmax(&mask, 0, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[0.0, 0.0]);
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
        mask: &TypedTensor<bool>,
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Log-softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// Masked-out entries are `-inf`; a slice with no unmasked entry is all `-inf`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
    /// let mask = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false])?;
    /// let y = x.masked_log_softmax(&mask, 0, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[0.0, f64::NEG_INFINITY]);
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
        mask: &TypedTensor<bool>,
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Layer normalization along `axis` with optional affine `weight` / `bias`, on an engine.
    ///
    /// `(x - mean) / sqrt(var + eps) * weight + bias` with the biased variance;
    /// `weight` and `bias` are rank-1 of length `shape[axis]`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 3.0])?;
    /// let y = x.layer_norm(0, None, None, 0.0, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[-1.0, 1.0]);
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
        weight: Option<&TypedTensor<T>>,
        bias: Option<&TypedTensor<T>>,
        eps: f64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// RMS normalization along `axis` with optional affine `weight` / `bias`, on an engine.
    ///
    /// `x / sqrt(mean(x^2) + eps) * weight + bias`; `weight` and `bias` are rank-1
    /// of length `shape[axis]`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 0.0])?;
    /// let y = x.rms_norm(0, None, None, 1e-6, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[0.0, 0.0]);
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
        weight: Option<&TypedTensor<T>>,
        bias: Option<&TypedTensor<T>>,
        eps: f64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;

    /// Gather windows of this tensor at `indices` on an engine (StableHLO `gather`).
    ///
    /// Mirrors [`crate::TensorSessionOpsExt::gather`]; the index tensor stays
    /// dtype-erased, as on that surface.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{GatherConfig, Tensor, TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// // Row 1 of a 3x2 matrix.
    /// let idx = Tensor::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// let config = GatherConfig {
    ///     offset_dims: vec![1],
    ///     collapsed_slice_dims: vec![0],
    ///     start_index_map: vec![0],
    ///     index_vector_dim: 1,
    ///     slice_sizes: vec![1, 2],
    /// };
    /// let row = x.gather(&idx, config, &mut engine)?;
    /// assert_eq!(row.host_data()?, &[2.0, 5.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn gather(
        &self,
        indices: &tenferro_tensor::Tensor,
        config: crate::GatherConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Scatter `updates` into a copy of this tensor at `indices` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{ScatterConfig, Tensor, TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![0.0; 6])?;
    /// // Add `[10, 20]` into row 1.
    /// let idx = Tensor::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// let updates = Tensor::from_vec_col_major(vec![1, 2], vec![10.0_f64, 20.0])?;
    /// let config = ScatterConfig {
    ///     update_window_dims: vec![1],
    ///     inserted_window_dims: vec![0],
    ///     scatter_dims_to_operand_dims: vec![0],
    ///     index_vector_dim: 1,
    /// };
    /// let y = x.scatter(&idx, &updates, config, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[0.0, 10.0, 0.0, 0.0, 20.0, 0.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn scatter(
        &self,
        indices: &tenferro_tensor::Tensor,
        updates: &tenferro_tensor::Tensor,
        config: crate::ScatterConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Slice this tensor with start, limit and stride per axis on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{SliceConfig, TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// let config = SliceConfig {
    ///     starts: vec![1, 0],
    ///     limits: vec![3, 2],
    ///     strides: vec![1, 1],
    /// };
    /// let y = x.slice(config, &mut engine)?;
    /// assert_eq!(y.shape(), &[2, 2]);
    /// assert_eq!(y.host_data()?, &[2.0, 3.0, 5.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn slice(
        &self,
        config: crate::SliceConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Slice at runtime `starts` with static `sizes` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0])?;
    /// // A start of 4 is clamped so the window of 2 fits.
    /// let starts = Tensor::from_vec_col_major(vec![1], vec![4_i64])?;
    /// let y = x.dynamic_slice(&starts, &[2], &mut engine)?;
    /// assert_eq!(y.host_data()?, &[3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn dynamic_slice(
        &self,
        starts: &tenferro_tensor::Tensor,
        sizes: &[usize],
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Pad this tensor with zeros on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{PadConfig, TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// let config = PadConfig {
    ///     edge_padding_low: vec![1],
    ///     edge_padding_high: vec![0],
    ///     interior_padding: vec![0],
    /// };
    /// let y = x.pad(config, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[0.0, 1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn pad(
        &self,
        config: crate::PadConfig,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Concatenate typed tensors along `axis` on an engine.
    ///
    /// N-ary, so it stays an associated function, matching the erased surface.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0])?;
    /// let y = TypedTensor::concatenate(&[&a, &b], 0, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[1.0, 2.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn concatenate(
        inputs: &[&TypedTensor<T>],
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>
    where
        Self: Sized;
    /// Reverse the elements along `axes` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0])?;
    /// let y = x.reverse(&[0], &mut engine)?;
    /// assert_eq!(y.host_data()?, &[3.0, 2.0, 1.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn reverse(
        &self,
        axes: &[usize],
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Zero the elements above the `k`-th diagonal on an engine (keep the lower triangle).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let lower = x.tril(0, &mut engine)?;
    /// assert_eq!(lower.host_data()?, &[1.0, 2.0, 0.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn tril(&self, k: i64, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Zero the elements below the `k`-th diagonal on an engine (keep the upper triangle).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let upper = x.triu(0, &mut engine)?;
    /// assert_eq!(upper.host_data()?, &[1.0, 0.0, 3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn triu(&self, k: i64, session: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Extract the diagonal over `axis_a` and `axis_b` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let diagonal = x.extract_diag(0, 1, &mut engine)?;
    /// assert_eq!(diagonal.host_data()?, &[1.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn extract_diag(
        &self,
        axis_a: usize,
        axis_b: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Embed this tensor along the diagonal of `axis_a` and `axis_b` on an engine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// let y = x.embed_diag(0, 1, &mut engine)?;
    /// assert_eq!(y.shape(), &[2, 2]);
    /// assert_eq!(y.host_data()?, &[1.0, 0.0, 0.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an invalid shape,
    /// axis, dtype or configuration, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn embed_diag(
        &self,
        axis_a: usize,
        axis_b: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// NumPy-style `take_along_axis` over `gather`, on an engine.
    ///
    /// Mirrors [`crate::TensorSessionOpsExt::take_along_axis`]; the index
    /// tensor stays dtype-erased, as on that surface.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{Tensor, TypedTensor, TypedTensorSessionOpsExt};
    ///
    /// let mut engine = CpuEngine::serial();
    /// // Per-batch row gather: out[i, j, b] = x[idx[i, b], j, b].
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2, 2], (0..8).map(f64::from).collect())?;
    /// let idx = Tensor::from_vec_col_major(vec![2, 1, 2], vec![1_i64, 0, 0, 0])?;
    /// let y = x.take_along_axis(&idx, 0, &mut engine)?;
    /// assert_eq!(y.host_data()?, &[1.0, 0.0, 3.0, 2.0, 4.0, 4.0, 6.0, 6.0]);
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
        indices: &tenferro_tensor::Tensor,
        axis: usize,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

/// Engine-explicit bool-mask operations for typed tensors.
///
/// This trait keeps `where_select` available as a method on bool
/// `TypedTensor`s while preserving the crate-root extension-trait surface. It
/// is public because downstream users call it directly; the implementation
/// helper in the private `typed_tensor` module is not a compatibility API.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_runtime::{TypedTensor, TypedTensorMaskSessionOpsExt};
///
/// let mut engine = CpuEngine::serial();
/// let condition =
///     TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false]).unwrap();
/// let on_true = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
/// let on_false = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
/// let selected = condition.where_select(&on_true, &on_false, &mut engine)?;
/// assert_eq!(selected.host_data().unwrap(), &[1.0, 4.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait TypedTensorMaskSessionOpsExt {
    /// Select typed values using this bool tensor as condition.
    ///
    /// The condition broadcasts against both branches (NumPy rules).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_runtime::{TypedTensor, TypedTensorMaskSessionOpsExt};
    ///
    ///
    /// let mut engine = CpuEngine::serial();
    /// let mask = TypedTensor::<bool>::from_vec_col_major(vec![], vec![false])?;
    /// let x = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![1, 2])?;
    /// let y = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![3, 4])?;
    /// let picked = mask.where_select(&x, &y, &mut engine)?;
    /// assert_eq!(picked.host_data()?, &[3, 4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when the condition or either branch
    /// cannot broadcast to the other operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn where_select<U: TensorScalar>(
        &self,
        on_true: &TypedTensor<U>,
        on_false: &TypedTensor<U>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<U>>;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    /// Method names declared between a trait header and its closing brace.
    fn method_names<'a>(source: &'a str, header: &str) -> BTreeSet<&'a str> {
        let body = source
            .split_once(header)
            .unwrap_or_else(|| panic!("trait header not found: {header}"))
            .1;
        let body = body.split_once("\n}\n").expect("trait body ends").0;
        body.lines()
            .filter_map(|line| line.strip_prefix("    fn "))
            .filter_map(|rest| rest.split_once('(').map(|(name, _)| name))
            .collect()
    }

    /// #2214: an operation may not land on the erased surface alone.
    ///
    /// The exemption list is the reviewed set of erased-only operations:
    /// `convert` and `cast` change the dtype of a dtype-erased tensor (a typed
    /// tensor has one dtype), and `where_select` is on the bool-mask trait.
    /// Anything else missing here is a new gap.
    #[test]
    fn typed_surface_covers_the_erased_surface_except_documented_exemptions() {
        let erased = method_names(
            include_str!("session_ext.rs"),
            "pub trait TensorSessionOpsExt {",
        );
        let typed = method_names(
            include_str!("typed_session_ext.rs"),
            "pub trait TypedTensorSessionOpsExt<T: TensorScalar> {",
        );
        let exempt: BTreeSet<&str> = ["convert", "cast", "where_select"].into_iter().collect();
        let missing: Vec<&&str> = erased
            .difference(&typed)
            .filter(|name| !exempt.contains(*name))
            .collect();
        assert!(
            missing.is_empty(),
            "the typed surface is missing {missing:?}; add the wrapper and its read hook, or extend the reviewed exemption list"
        );
    }
}
