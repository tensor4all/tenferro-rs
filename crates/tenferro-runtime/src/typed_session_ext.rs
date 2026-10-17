//! Backend-explicit session operations on [`TypedTensor`].

use num_complex::Complex64;

use crate::{BackendSession, CompareDir, DotGeneralConfig, TensorScalar, TypedTensor};

/// AD-free tensor operations on [`TypedTensor`], run inside a borrowed backend session.
///
/// The methods mirror [`crate::TensorSessionOpsExt`] for a statically known
/// scalar type, with the session last. Operations whose backend hooks take
/// owned dtype-erased tensors (indexing, padding, concatenation, triangular
/// and diagonal masks) are offered on [`crate::Tensor`] only: a typed form
/// would have to copy the input first. Move a typed tensor into that surface
/// with `Tensor::from_typed`, which does not copy. Selection with a bool mask
/// is [`crate::TypedTensorMaskSessionOpsExt::where_select`].
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
/// use tenferro_tensor::BackendSessionHost;
///
/// let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -3.0])?;
/// let y = backend.with_backend_session(|session| x.reduce_max(None, session))??;
/// assert_eq!(y.host_data()?, &[1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait TypedTensorSessionOpsExt<T: TensorScalar> {
    /// Elementwise addition with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
    /// let sum = backend.with_backend_session(|session| a.add(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise addition into a caller-owned output inside a session.
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
    /// let mut out = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 0.0]).unwrap();
    /// backend.with_backend_session(|session| a.add_into(&b, &mut out, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<()>;
    /// Elementwise multiplication with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![2.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![3.0; 4]).unwrap();
    /// let product = backend.with_backend_session(|session| a.mul(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise exponential inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.exp(session))??;
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
    fn exp(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Sum over the selected axes inside a session. `None` reduces every
    /// axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let sums = backend.with_backend_session(|session| x.reduce_sum(Some(&[1]), session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise subtraction with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.sub(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise division with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 8.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.div(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise remainder with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![5.0, 7.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.rem(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise power with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.pow(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise maximum with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.maximum(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise minimum with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.minimum(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise negation inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.neg(session))??;
    /// assert_eq!(y.host_data().unwrap(), &[-1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn neg(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise absolute value inside a session.
    ///
    /// The result has the real counterpart dtype `T::Real`: complex magnitude
    /// is real, and real or integer inputs keep their own dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-1.0, 2.0])?;
    /// let y = backend.with_backend_session(|session| x.abs(session))??;
    /// assert_eq!(y.host_data()?, &[1.0, 2.0]);
    ///
    /// let z = TypedTensor::<Complex64>::from_vec_col_major(vec![1], vec![Complex64::new(3.0, 4.0)])?;
    /// let magnitude: TypedTensor<f64> = backend.with_backend_session(|session| z.abs(session))??;
    /// assert_eq!(magnitude.host_data()?, &[5.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T::Real>>;
    /// Elementwise sign inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sign(session))??;
    /// assert_eq!(y.host_data().unwrap(), &[1.0, -1.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise complex conjugate inside a session.
    ///
    /// For real dtypes the conjugate is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.conj(session))??;
    /// assert_eq!(y.host_data().unwrap(), &[1.0, -2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise natural logarithm inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, std::f64::consts::E]).unwrap();
    /// let y = backend.with_backend_session(|session| x.log(session))??;
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
    fn log(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `exp(x) - 1` inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.expm1(session))??;
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
    fn expm1(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `log(1 + x)` inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::E - 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.log1p(session))??;
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
    fn log1p(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise error function `erf(x)` inside a session, for real `f32`/`f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.erf(session))??;
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
    fn erf(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sine inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::FRAC_PI_2]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sin(session))??;
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
    fn sin(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise cosine inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::PI]).unwrap();
    /// let y = backend.with_backend_session(|session| x.cos(session))??;
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
    fn cos(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise hyperbolic tangent inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.tanh(session))??;
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
    fn tanh(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise square root inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 9.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sqrt(session))??;
    /// assert_eq!(y.host_data().unwrap(), &[2.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise reciprocal square root inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.rsqrt(session))??;
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
    fn rsqrt(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise comparison with NumPy-style broadcasting inside a session.
    ///
    /// The result is a bool typed tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{CompareDir, TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.compare(&b, CompareDir::Gt, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<bool>>;
    /// Clamp values elementwise between lower and upper bounds inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-2.0, 4.0]).unwrap();
    /// let lower = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0]).unwrap();
    /// let upper = TypedTensor::<f64>::from_vec_col_major(vec![], vec![3.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.clamp(&lower, &upper, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Rank-2 matrix multiplication inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0; 6]).unwrap();
    /// let c = backend.with_backend_session(|session| a.matmul(&b, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Reshape through the backend structural operation inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let y = backend.with_backend_session(|session| x.reshape(&[3, 2], session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Permute axes through the backend structural operation inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let y = backend.with_backend_session(|session| x.transpose(&[1, 0], session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Broadcast into a larger shape inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    /// let matrix = backend.with_backend_session(|session| row.broadcast_in_dim(&[2, 3], &[1], session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Take the maximum over the selected axes inside a session. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0])?;
    /// let y = backend.with_backend_session(|session| x.reduce_max(Some(&[0]), session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Take the minimum over the selected axes inside a session. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0])?;
    /// let y = backend.with_backend_session(|session| x.reduce_min(Some(&[0]), session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Multiply over the selected axes inside a session. `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0])?;
    /// let y = backend.with_backend_session(|session| x.reduce_prod(Some(&[0]), session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Sum elementwise squares over the selected axes inside a session (`f32`/`f64`). `None` reduces every axis and `Some(&[])` keeps the input shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 5.0, 3.0, 2.0])?;
    /// let y = backend.with_backend_session(|session| x.reduce_sum_squares(Some(&[0]), session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Contract this tensor with `rhs` inside a session (StableHLO `dot_general`).
    ///
    /// The output layout is `[lhs free..., rhs free..., batch...]` (batch axes trail).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// use tenferro_runtime::DotGeneralConfig;
    /// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![1, 2], vec![2.0, 3.0])?;
    /// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 5.0])?;
    /// let config = DotGeneralConfig {
    ///     lhs_contracting_dims: [1].as_slice().iter().copied().collect(),
    ///     rhs_contracting_dims: [0].as_slice().iter().copied().collect(),
    ///     lhs_batch_dims: [].as_slice().iter().copied().collect(),
    ///     rhs_batch_dims: [].as_slice().iter().copied().collect(),
    /// };
    /// let y = backend.with_backend_session(|session| lhs.dot_general(&rhs, config, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Contract with optional conjugation of either operand inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// use num_complex::Complex64;
    /// use tenferro_runtime::DotGeneralConfig;
    /// let lhs = TypedTensor::<Complex64>::from_vec_col_major(vec![1, 1], vec![Complex64::new(0.0, 1.0)])?;
    /// let rhs = TypedTensor::<Complex64>::from_vec_col_major(vec![1, 1], vec![Complex64::new(0.0, 1.0)])?;
    /// let config = DotGeneralConfig {
    ///     lhs_contracting_dims: [1].as_slice().iter().copied().collect(),
    ///     rhs_contracting_dims: [0].as_slice().iter().copied().collect(),
    ///     lhs_batch_dims: [].as_slice().iter().copied().collect(),
    ///     rhs_batch_dims: [].as_slice().iter().copied().collect(),
    /// };
    /// let y = backend.with_backend_session(|session| lhs.dot_general_with_conj(&rhs, config, true, false, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Multiply by a real scalar inside a session, with the eager `scale_real` dtype rules.
    ///
    /// Integer dtypes round the factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
    /// let y = backend.with_backend_session(|session| x.scale_real(2.0, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Multiply a complex tensor by a complex scalar inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// use num_complex::Complex64;
    /// let x = TypedTensor::<Complex64>::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])?;
    /// let y = backend.with_backend_session(|session| x.scale_complex(Complex64::new(0.0, 1.0), session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Logistic sigmoid `1 / (1 + exp(-x))` inside a session, overflow-free.
    ///
    /// Evaluated as `1 / (1 + e)` for `x > 0` and `e / (1 + e)` otherwise, with
    /// `e = exp(-|x|)`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-700.0, 0.0, 1000.0])?;
    /// let y = backend.with_backend_session(|session| x.sigmoid(session))??;
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
    fn sigmoid(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// SiLU (swish) `x * sigmoid(x)` inside a session.
    ///
    /// Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-1.0, 0.0, 1.0])?;
    /// let y = backend.with_backend_session(|session| x.silu(session))??;
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
    fn silu(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Softplus `log(1 + exp(x))` inside a session, in the stable form `max(x, 0) + log1p(exp(-|x|))`.
    ///
    /// Real `F32`/`F64` only; never overflows.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-1000.0, 0.0, 1000.0])?;
    /// let y = backend.with_backend_session(|session| x.softplus(session))??;
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
    fn softplus(&self, session: &mut dyn BackendSession)
        -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Exact GELU `x/2 * (1 + erf(x / sqrt(2)))` inside a session.
    ///
    /// Real `F32`/`F64` only (PyTorch `approximate="none"`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-1.0, 0.0, 1.0])?;
    /// let y = backend.with_backend_session(|session| x.gelu(session))??;
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
    fn gelu(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// GELU tanh approximation inside a session (PyTorch `approximate="tanh"`).
    ///
    /// `x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`; real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-1.0, 0.0, 1.0])?;
    /// let y = backend.with_backend_session(|session| x.gelu_tanh(session))??;
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
    fn gelu_tanh(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Arithmetic mean over `axes` inside a session (`None` reduces every axis).
    ///
    /// Float and complex dtypes. The sum is divided by the element count; a mean
    /// over zero elements is `NaN`, and `Some(&[])` is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let y = backend.with_backend_session(|session| x.reduce_mean(None, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Max-subtracted softmax along `axis` inside a session.
    ///
    /// Real `F32`/`F64` only. A slice that is entirely `-inf` returns zeros
    /// instead of `NaN`; a `NaN` or `+inf` entry makes its slice `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 1.0])?;
    /// let y = backend.with_backend_session(|session| x.softmax(0, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Max-subtracted log-softmax along `axis` inside a session.
    ///
    /// Real `F32`/`F64` only. A slice that is entirely `-inf` returns `-inf`
    /// instead of `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 1.0])?;
    /// let y = backend.with_backend_session(|session| x.log_softmax(0, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// `mask` broadcasts to the input shape. Masked-out entries are `0` whatever
    /// their value; a slice with no unmasked entry is all zeros.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
    /// let mask = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![false, false])?;
    /// let y = backend.with_backend_session(|session| x.masked_softmax(&mask, 0, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Log-softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// Masked-out entries are `-inf`; a slice with no unmasked entry is all `-inf`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
    /// let mask = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false])?;
    /// let y = backend.with_backend_session(|session| x.masked_log_softmax(&mask, 0, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Layer normalization along `axis` with optional affine `weight` / `bias`, inside a session.
    ///
    /// `(x - mean) / sqrt(var + eps) * weight + bias` with the biased variance;
    /// `weight` and `bias` are rank-1 of length `shape[axis]`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 3.0])?;
    /// let y = backend.with_backend_session(|session| x.layer_norm(0, None, None, 0.0, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// RMS normalization along `axis` with optional affine `weight` / `bias`, inside a session.
    ///
    /// `x / sqrt(mean(x^2) + eps) * weight + bias`; `weight` and `bias` are rank-1
    /// of length `shape[axis]`. Real `F32`/`F64` only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 0.0])?;
    /// let y = backend.with_backend_session(|session| x.rms_norm(0, None, None, 1e-6, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

/// Backend-explicit bool-mask session operations for typed tensors.
///
/// This trait keeps `where_select` available as a method on bool
/// `TypedTensor`s while preserving the crate-root extension-trait surface. It
/// is public because downstream users call it directly; the implementation
/// helper in the private `typed_tensor` module is not a compatibility API.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorMaskSessionOpsExt};
/// use tenferro_tensor::BackendSessionHost;
///
/// let mut backend = CpuBackend::new();
/// let condition =
///     TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false]).unwrap();
/// let on_true = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
/// let on_false = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
/// let selected = backend
///     .with_backend_session(|session| condition.where_select(&on_true, &on_false, session))?
///     .unwrap();
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorMaskSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let mask = TypedTensor::<bool>::from_vec_col_major(vec![], vec![false])?;
    /// let x = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![1, 2])?;
    /// let y = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![3, 4])?;
    /// let picked = backend.with_backend_session(|session| mask.where_select(&x, &y, session))??;
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
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<U>>;
}
