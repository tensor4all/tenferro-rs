//! Receiver-first concrete linear algebra surfaces.
//!
//! Owned tensors use [`TensorLinalgExt`], borrowed tensors and views use the
//! `_read` methods on [`TensorReadLinalgExt`], and typed tensors use
//! [`TypedTensorLinalgExt`]. All methods dispatch internally to the built-in
//! CPU/CUDA execution sessions through an erased `&mut dyn BackendSession`
//! (issue #1680 Phase 3); third-party [`LinalgBackend`] implementations
//! remain supported through the SPI trait, but the concrete op path is
//! built-in-session only.

use num_complex::{Complex32, Complex64};
use tenferro_cpu::with_cpu_exec_session;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::with_cuda_exec_session;
use tenferro_tensor::{
    BackendSession, CompareDir, DType, DotGeneralConfig, Tensor, TensorRead, TensorScalar,
    TensorWrite, TypedTensor,
};

use crate::extension::{
    apply_eigh_gauge, apply_qr_gauge, apply_svd_gauge, validate_derivative_eps, EighOptions,
    QrOptions, SvdOptions,
};
use crate::LinalgBackend;

/// Scalar types supported by statically typed linear algebra methods.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::LinalgScalar;
///
/// fn accepts_linalg_scalar<T: LinalgScalar>() {}
/// accepts_linalg_scalar::<f64>();
/// ```
pub trait LinalgScalar: TensorScalar + private::Sealed {
    /// Complex counterpart used by general eigendecomposition.
    type Complex: TensorScalar;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

impl LinalgScalar for f32 {
    type Complex = Complex32;
}
impl LinalgScalar for f64 {
    type Complex = Complex64;
}
impl LinalgScalar for Complex32 {
    type Complex = Complex32;
}
impl LinalgScalar for Complex64 {
    type Complex = Complex64;
}

/// Fixed typed output tuple for singular value decomposition.
///
/// # Examples
///
/// ```rust
/// let _: Option<tenferro_linalg::TypedSvd<f64>> = None;
/// ```
pub type TypedSvd<T> = (
    TypedTensor<T>,
    TypedTensor<<T as TensorScalar>::Real>,
    TypedTensor<T>,
);
/// Fixed typed output tuple for LU decomposition.
///
/// # Examples
///
/// ```rust
/// let _: Option<tenferro_linalg::TypedLu<f64>> = None;
/// ```
pub type TypedLu<T> = (
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<<T as TensorScalar>::Real>,
);
/// Fixed typed output tuple for complete-pivot LU decomposition.
///
/// # Examples
///
/// ```rust
/// let _: Option<tenferro_linalg::TypedFullPivLu<f64>> = None;
/// ```
pub type TypedFullPivLu<T> = (
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<T>,
    TypedTensor<<T as TensorScalar>::Real>,
);
/// Fixed typed output tuple for general eigendecomposition.
///
/// # Examples
///
/// ```rust
/// let _: Option<tenferro_linalg::TypedEig<f64>> = None;
/// ```
pub type TypedEig<T> = (
    TypedTensor<<T as LinalgScalar>::Complex>,
    TypedTensor<<T as LinalgScalar>::Complex>,
);

/// Linear algebra methods for dtype-erased owned tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_linalg::TensorLinalgExt;
/// use tenferro_tensor::{BackendSessionHost, Tensor};
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
/// let mut host = CpuBackend::new();
/// let (_u, singular_values, _vt) = host.with_backend_session(|session| a.svd(session))?;
/// assert_eq!(singular_values.as_slice::<f64>()?, &[4.0, 2.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorLinalgExt {
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or output-contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_u, s, _vt) = host.with_backend_session(|session| a.svd(session))?;
    /// assert_eq!(s.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    /// Compute singular values without allocating singular-vector outputs.
    ///
    /// # Errors
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected
    /// backend has no values-only capability, rather than silently computing a
    /// full decomposition and discarding its vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| a.svdvals(session))?;
    /// assert_eq!(values.as_slice::<f64>()?, &[4.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svdvals(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for matrix metadata or options, plus SVD backend errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{SvdOptions, TensorLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_u, s, _vt) = host.with_backend_session(|session| { a.svd_with_options(SvdOptions::default(), session) })?;
    /// assert_eq!(s.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd_with_options(
        &self,
        options: SvdOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or output-contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (q, r) = host.with_backend_session(|session| a.qr(session))?;
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<(Tensor, Tensor)>;

    /// Initialize opaque compact Householder QR state.
    ///
    /// # Errors
    ///
    /// Returns validation errors for non-matrix or unsupported-dtype input and
    /// typed provider errors when compact QR is unavailable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::TensorLinalgExt;
    /// use tenferro_tensor::{BackendSessionHost, Tensor};
    /// let a = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
    /// let mut host = CpuBackend::new();
    /// let qr = host.with_backend_session(|session| a.householder_qr(session))?;
    /// assert!(format!("{qr:?}").starts_with("HouseholderQr"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn householder_qr(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<crate::HouseholderQr<Tensor>>;

    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for matrix metadata or options, plus QR backend errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{QrOptions, TensorLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (q, r) = host.with_backend_session(|session| { a.qr_with_options(QrOptions::default(), session) })?;
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr_with_options(
        &self,
        options: QrOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or output-contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_p, l, u, parity) = host.with_backend_session(|session| a.lu(session))?;
    /// assert_eq!(l.shape(), &[2, 2]);
    /// assert_eq!(u.shape(), &[2, 2]);
    /// assert_eq!(parity.shape(), &[]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn lu(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or output-contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (p, _l, _u, q, parity) = host.with_backend_session(|session| a.full_piv_lu(session))?;
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(parity.shape(), &[]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for incompatible inputs or a backend/singular-system error.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| a.full_piv_lu_solve(&b, session))?;
    /// assert_eq!(x.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu_solve(
        &self,
        b: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for incompatible inputs or a backend/singular-system error.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| a.solve(&b, session))?;
    /// assert_eq!(x.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn solve(
        &self,
        b: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for a non-square input or backend/numerical errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let l = host.with_backend_session(|session| a.cholesky(session))?;
    /// assert_eq!(l.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn cholesky(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, convergence, or output-contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| a.eigh(session))?;
    /// assert_eq!(values.as_slice::<f64>()?, &[1.0, 3.0]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for matrix metadata or options, plus Eigh backend errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{EighOptions, TensorLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| { a.eigh_with_options(EighOptions::default(), session) })?;
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh_with_options(
        &self,
        options: EighOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, convergence, or output-contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| a.eig(session))?;
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eig(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for incompatible inputs/flags or backend/numerical errors.
    #[allow(clippy::too_many_arguments)]
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| { a.triangular_solve(&b, true, false, false, false, session) })?;
    /// assert_eq!(x.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn triangular_solve(
        &self,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or LU output-contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (sign, logabsdet) = host.with_backend_session(|session| a.slogdet(session))?;
    /// assert_eq!(sign.as_slice::<f64>()?, &[1.0]);
    /// assert_eq!(logabsdet.shape(), &[]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn slogdet(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns the validation, backend, numerical, or contract errors from [`Self::slogdet`].
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let determinant = host.with_backend_session(|session| a.det(session))?;
    /// assert!((determinant.as_slice::<f64>()?[0] - 8.0).abs() < 1.0e-12);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn det(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, or singular-solve numerical errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let inverse = host.with_backend_session(|session| a.inv(session))?;
    /// assert_eq!(inverse.as_slice::<f64>()?, &[0.5, 0.0, 0.0, 0.25]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn inv(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns the validation, backend, convergence, or contract errors from [`Self::eigh`].
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| a.eigvalsh(session))?;
    /// assert_eq!(values.as_slice::<f64>()?, &[1.0, 3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvalsh(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns the validation, backend, convergence, or contract errors from [`Self::eig`].
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| a.eigvals(session))?;
    /// assert_eq!(values.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvals(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or SVD contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let pseudoinverse = host.with_backend_session(|session| a.pinv(session))?;
    /// assert_eq!(pseudoinverse.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn pinv(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns a validation error for invalid `rtol`, plus errors from [`Self::pinv`].
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let pseudoinverse = host.with_backend_session(|session| a.pinv_with_rtol(1.0e-12, session))?;
    /// assert_eq!(pseudoinverse.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn pinv_with_rtol(
        &self,
        rtol: f64,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for axes/order combinations or required backend operations.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let frobenius = host.with_backend_session(|session| a.norm(None, None, false, session))?;
    /// assert_eq!(frobenius.shape(), &[]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn norm(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
}

/// Linear algebra methods for borrowed tensor reads.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_linalg::TensorReadLinalgExt;
/// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
///
/// let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
/// let mut host = CpuBackend::new();
/// let (_q, r) = host.with_backend_session(|session| {
///     TensorRead::from_tensor(&input).qr_read(session)
/// })?;
/// assert_eq!(r.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorReadLinalgExt {
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, same-placement materialization, backend, numerical, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_u, s, _vt) = host.with_backend_session(|session| {
    ///     TensorRead::from_tensor(&a).svd_read(session)
    /// })?;
    /// assert_eq!(s.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    /// Compute singular values from a borrowed tensor read target.
    ///
    /// Eligible faer host views are consumed without a full input copy. A
    /// provider that requires owned compact storage may materialize explicitly;
    /// unsupported providers return a typed error.
    ///
    /// # Errors
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected
    /// backend has no borrowed values-only capability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| TensorRead::from_tensor(&a).svdvals_read(session))?;
    /// assert_eq!(values.as_slice::<f64>()?, &[4.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svdvals_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus read/materialization or SVD errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{SvdOptions, TensorReadLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_u, s, _vt) = host.with_backend_session(|session| {
    ///     TensorRead::from_tensor(&a).svd_with_options_read(SvdOptions::default(), session)
    /// })?;
    /// assert_eq!(s.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd_with_options_read(
        self,
        options: SvdOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, same-placement materialization, backend, numerical, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (q, r) = host.with_backend_session(|session| { TensorRead::from_tensor(&a).qr_read(session) })?;
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr_read(self, session: &mut dyn BackendSession)
        -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus read/materialization or QR errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{QrOptions, TensorReadLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (q, r) = host.with_backend_session(|session| {
    ///     TensorRead::from_tensor(&a).qr_with_options_read(QrOptions::default(), session)
    /// })?;
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr_with_options_read(
        self,
        options: QrOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, same-placement materialization, backend, numerical, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_p, l, u, _parity) = host.with_backend_session(|session| { TensorRead::from_tensor(&a).lu_read(session) })?;
    /// assert_eq!(l.shape(), &[2, 2]);
    /// assert_eq!(u.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn lu_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, same-placement materialization, backend, numerical, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (p, _l, _u, q, _parity) = host.with_backend_session(|session| { TensorRead::from_tensor(&a).full_piv_lu_read(session) })?;
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input validation, read/materialization, backend, or singular errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| {
    ///     TensorRead::from_tensor(&a).full_piv_lu_solve_read(TensorRead::from_tensor(&b), session)
    /// })?;
    /// assert_eq!(x.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu_solve_read(
        self,
        b: TensorRead<'_>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input validation, read/materialization, backend, or singular errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| { TensorRead::from_tensor(&a).solve_read(TensorRead::from_tensor(&b), session) })?;
    /// assert_eq!(x.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn solve_read(
        self,
        b: TensorRead<'_>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Solve into a caller-owned destination without allocating the result at
    /// the public API boundary.
    ///
    /// Backends with a native path may write directly into a compatible target;
    /// the trait default preserves the ordinary solve-read plus copy behavior.
    ///
    /// # Errors
    /// Returns `tenferro_tensor_core::ShapeMismatch` or
    /// `tenferro_tensor_core::ValidationError::DTypeMismatch` for incompatible
    /// destination metadata, `tenferro_tensor_core::ValidationError::InvalidArgument`
    /// for aliasing or placement violations, `Error::Unsupported` when the
    /// extension implementation or provider is unavailable, and `Error::Singular`
    /// for a singular system.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorWrite};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0])?;
    /// # let mut out = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    /// # let mut host = CpuBackend::new();
    /// host.with_backend_session(|session| {
    ///         TensorRead::from_tensor(&a).solve_read_into(
    ///             TensorRead::from_tensor(&b),
    ///             TensorWrite::from_tensor(&mut out),
    ///             session,
    ///         )
    ///     })?;
    /// assert_eq!(out.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn solve_read_into(
        self,
        b: TensorRead<'_>,
        out: TensorWrite<'_>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<()>
    where
        Self: Sized,
    {
        let _ = (self, b, out, session);
        Err(tenferro_tensor::Error::unsupported(
            "solve_read_into",
            "this tensor-read extension implementation does not accept borrowed solve targets",
        ))
    }
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns matrix validation, read/materialization, backend, or positive-definiteness errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let l = host.with_backend_session(|session| { TensorRead::from_tensor(&a).cholesky_read(session) })?;
    /// assert_eq!(l.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn cholesky_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, convergence, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| { TensorRead::from_tensor(&a).eigh_read(session) })?;
    /// assert_eq!(values.as_slice::<f64>()?, &[1.0, 3.0]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus read or Eigh backend errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{EighOptions, TensorReadLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| {
    ///     TensorRead::from_tensor(&a).eigh_with_options_read(EighOptions::default(), session)
    /// })?;
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh_with_options_read(
        self,
        options: EighOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, convergence, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| { TensorRead::from_tensor(&a).eig_read(session) })?;
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eig_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input/flag validation, read, backend, or singular errors.
    #[allow(clippy::too_many_arguments)]
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| {
    ///     TensorRead::from_tensor(&a).triangular_solve_read(
    ///         TensorRead::from_tensor(&b),
    ///         true,
    ///         false,
    ///         false,
    ///         false,
    ///         session,
    ///     )
    /// })?;
    /// assert_eq!(x.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn triangular_solve_read(
        self,
        b: TensorRead<'_>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, numerical, or LU contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (sign, logabsdet) = host.with_backend_session(|session| { TensorRead::from_tensor(&a).slogdet_read(session) })?;
    /// assert_eq!(sign.as_slice::<f64>()?, &[1.0]);
    /// assert_eq!(logabsdet.shape(), &[]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn slogdet_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, numerical, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let determinant = host.with_backend_session(|session| { TensorRead::from_tensor(&a).det_read(session) })?;
    /// assert!((determinant.as_slice::<f64>()?[0] - 8.0).abs() < 1.0e-12);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn det_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, or singular-solve errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let inverse = host.with_backend_session(|session| { TensorRead::from_tensor(&a).inv_read(session) })?;
    /// assert_eq!(inverse.as_slice::<f64>()?, &[0.5, 0.0, 0.0, 0.25]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn inv_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, convergence, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| { TensorRead::from_tensor(&a).eigvalsh_read(session) })?;
    /// assert_eq!(values.as_slice::<f64>()?, &[1.0, 3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvalsh_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, convergence, or contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| { TensorRead::from_tensor(&a).eigvals_read(session) })?;
    /// assert_eq!(values.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvals_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, numerical, or SVD contract errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let pseudoinverse = host.with_backend_session(|session| { TensorRead::from_tensor(&a).pinv_read(session) })?;
    /// assert_eq!(pseudoinverse.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn pinv_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns a validation error for invalid `rtol`, plus errors from [`Self::pinv_read`].
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let pseudoinverse = host.with_backend_session(|session| { TensorRead::from_tensor(&a).pinv_with_rtol_read(1.0e-12, session) })?;
    /// assert_eq!(pseudoinverse.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn pinv_with_rtol_read(
        self,
        rtol: f64,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for axes/order combinations or required read/backend operations.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TensorReadLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let frobenius = host.with_backend_session(|session| { TensorRead::from_tensor(&a).norm_read(None, None, false, session) })?;
    /// assert_eq!(frobenius.shape(), &[]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn norm_read(
        self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
}

/// Linear algebra methods with statically typed inputs and outputs.
///
/// Singular values, Hermitian eigenvalues, determinant log-magnitudes, and
/// norms use `T::Real`; general eigenvalues and eigenvectors use `T::Complex`.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_linalg::TypedTensorLinalgExt;
/// use tenferro_tensor::{BackendSessionHost, TypedTensor};
///
/// let input = TypedTensor::<f64>::from_vec_col_major(
///     vec![2, 2],
///     vec![2.0, 0.0, 0.0, 4.0],
/// )?;
/// let mut host = CpuBackend::new();
/// let (_u, singular_values, _vt) = host.with_backend_session(|session| input.svd(session))?;
/// assert_eq!(singular_values.as_slice()?, &[4.0, 2.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TypedTensorLinalgExt<T: LinalgScalar> {
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_u, s, _vt) = host.with_backend_session(|session| a.svd(session))?;
    /// assert_eq!(s.as_slice()?, &[4.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedSvd<T>>;
    /// Compute singular values without allocating singular-vector outputs.
    ///
    /// # Errors
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected
    /// backend has no values-only capability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| a.svdvals(session))?;
    /// assert_eq!(values.as_slice()?, &[4.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svdvals(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus backend or typed-output errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{SvdOptions, TypedTensorLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_u, s, _vt) = host.with_backend_session(|session| { a.svd_with_options(SvdOptions::default(), session) })?;
    /// assert_eq!(s.as_slice()?, &[4.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd_with_options(
        &self,
        options: SvdOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedSvd<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (q, r) = host.with_backend_session(|session| a.qr(session))?;
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus backend or typed-output errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{QrOptions, TypedTensorLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (q, r) = host.with_backend_session(|session| { a.qr_with_options(QrOptions::default(), session) })?;
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr_with_options(
        &self,
        options: QrOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (_p, l, u, _parity) = host.with_backend_session(|session| a.lu(session))?;
    /// assert_eq!(l.shape(), &[2, 2]);
    /// assert_eq!(u.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn lu(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedLu<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (p, _l, _u, q, _parity) = host.with_backend_session(|session| a.full_piv_lu(session))?;
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedFullPivLu<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input validation, backend, singular, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0])?;
    /// # let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![-1.0, 5.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| a.full_piv_lu_solve(&b, session))?;
    /// assert_eq!(x.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu_solve(
        &self,
        b: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input validation, backend, singular, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 8.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| a.solve(&b, session))?;
    /// assert_eq!(x.as_slice()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn solve(
        &self,
        b: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns matrix validation, backend, positive-definiteness, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 2.0, 2.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let l = host.with_backend_session(|session| a.cholesky(session))?;
    /// assert_eq!(l.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn cholesky(&self, session: &mut dyn BackendSession)
        -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, convergence, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| a.eigh(session))?;
    /// assert_eq!(values.as_slice()?, &[1.0, 3.0]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus backend or typed-output errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{EighOptions, TypedTensorLinalgExt};
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| { a.eigh_with_options(EighOptions::default(), session) })?;
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh_with_options(
        &self,
        options: EighOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, convergence, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (values, vectors) = host.with_backend_session(|session| a.eig(session))?;
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eig(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedEig<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input/flag validation, backend, singular, or typed-output errors.
    #[allow(clippy::too_many_arguments)]
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 1.0, 3.0])?;
    /// # let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 9.0])?;
    /// # let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| { a.triangular_solve(&b, true, false, false, false, session) })?;
    /// assert_eq!(x.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn triangular_solve(
        &self,
        b: &TypedTensor<T>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let (sign, logabsdet) = host.with_backend_session(|session| a.slogdet(session))?;
    /// assert_eq!(sign.as_slice()?, &[1.0]);
    /// assert_eq!(logabsdet.shape(), &[]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn slogdet(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<<T as TensorScalar>::Real>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let determinant = host.with_backend_session(|session| a.det(session))?;
    /// assert!((determinant.as_slice()?[0] - 8.0).abs() < 1.0e-12);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn det(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, singular-solve, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let inverse = host.with_backend_session(|session| a.inv(session))?;
    /// assert_eq!(inverse.as_slice()?, &[0.5, 0.0, 0.0, 0.25]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn inv(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, convergence, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| a.eigvalsh(session))?;
    /// assert_eq!(values.as_slice()?, &[1.0, 3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvalsh(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, convergence, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0])?;
    /// # let mut host = CpuBackend::new();
    /// let values = host.with_backend_session(|session| a.eigvals(session))?;
    /// assert_eq!(values.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvals(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T::Complex>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let pseudoinverse = host.with_backend_session(|session| a.pinv(session))?;
    /// assert_eq!(pseudoinverse.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn pinv(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns a validation error for invalid `rtol`, plus backend or typed-output errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let pseudoinverse = host.with_backend_session(|session| { a.pinv_with_rtol(1.0e-12, session) })?;
    /// assert_eq!(pseudoinverse.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn pinv_with_rtol(
        &self,
        rtol: f64,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for axes/order combinations, backend, or typed-output errors.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::TypedTensorLinalgExt;
    /// # use tenferro_tensor::{BackendSessionHost, TypedTensor};
    /// # let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 4.0])?;
    /// # let mut host = CpuBackend::new();
    /// let frobenius = host.with_backend_session(|session| a.norm(None, None, false, session))?;
    /// assert_eq!(frobenius.shape(), &[]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn norm(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>>;
}

impl TensorLinalgExt for Tensor {
    fn svd(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "svd", |backend| three(backend.svd(self)?, "svd"))
    }
    fn svdvals(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "svdvals", |backend| backend.svd_values(self))
    }

    fn svd_with_options(
        &self,
        options: SvdOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "svd_with_options", |backend| {
            three(backend.svd_with_options(self, options)?, "svd_with_options")
        })
    }
    fn qr(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "qr", |backend| two(backend.qr(self)?, "qr"))
    }
    fn householder_qr(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<crate::HouseholderQr<Tensor>> {
        with_linalg_backend(session, "householder_qr", |backend| {
            backend
                .householder_qr(self)
                .map(crate::HouseholderQr::from_backend)
        })
    }
    fn qr_with_options(
        &self,
        options: QrOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "qr_with_options", |backend| {
            two(backend.qr_with_options(self, options)?, "qr_with_options")
        })
    }
    fn lu(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "lu", |backend| four(backend.lu(self)?, "lu"))
    }
    fn full_piv_lu(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "full_piv_lu", |backend| {
            five(backend.full_piv_lu(self)?, "full_piv_lu")
        })
    }
    fn full_piv_lu_solve(
        &self,
        b: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "full_piv_lu_solve", |backend| {
            backend.full_piv_lu_solve(self, b, false)
        })
    }
    fn solve(
        &self,
        b: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "solve", |backend| backend.solve(self, b))
    }
    fn cholesky(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "cholesky", |backend| backend.cholesky(self))
    }
    fn eigh(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "eigh", |backend| two(backend.eigh(self)?, "eigh"))
    }
    fn eigh_with_options(
        &self,
        options: EighOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "eigh_with_options", |backend| {
            two(
                backend.eigh_with_options(self, options)?,
                "eigh_with_options",
            )
        })
    }
    fn eig(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "eig", |backend| two(backend.eig(self)?, "eig"))
    }
    fn triangular_solve(
        &self,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "triangular_solve", |backend| {
            backend.triangular_solve(self, b, left_side, lower, transpose_a, unit_diagonal)
        })
    }
    fn slogdet(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "slogdet", |backend| {
            slogdet_from_lu(four(backend.lu(self)?, "slogdet")?, backend)
        })
    }
    fn det(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "det", |backend| {
            det_impl(self.slogdet(backend)?, backend)
        })
    }
    fn inv(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "inv", |backend| inv_owned(self, backend))
    }
    fn eigvalsh(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "eigvalsh", |backend| backend.eigh_values(self))
    }
    fn eigvals(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "eigvals", |backend| backend.eig_values(self))
    }
    fn pinv(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "pinv", |backend| pinv_owned(self, None, backend))
    }
    fn pinv_with_rtol(
        &self,
        rtol: f64,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "pinv_with_rtol", |backend| {
            pinv_owned(self, Some(rtol), backend)
        })
    }
    fn norm(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "norm", |backend| {
            norm_from_read(TensorRead::from_tensor(self), ord, dim, keepdim, backend)
        })
    }
}

impl TensorReadLinalgExt for TensorRead<'_> {
    fn svd_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "svd_read", |backend| {
            three(backend.svd_read(self)?, "svd_read")
        })
    }
    fn svdvals_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "svdvals_read", |backend| {
            backend.svd_values_read(self)
        })
    }

    fn svd_with_options_read(
        self,
        options: SvdOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "svd_with_options_read", |backend| {
            validate_derivative_eps("svd_with_options_read", options.derivative_eps)?;
            let mut out = backend.svd_read(self)?;
            apply_svd_gauge(options.gauge, &mut out)?;
            three(out, "svd_with_options_read")
        })
    }
    fn qr_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "qr_read", |backend| {
            two(backend.qr_read(self)?, "qr_read")
        })
    }
    fn qr_with_options_read(
        self,
        options: QrOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "qr_with_options_read", |backend| {
            let mut out = backend.qr_read(self)?;
            apply_qr_gauge(options.gauge, &mut out)?;
            two(out, "qr_with_options_read")
        })
    }
    fn lu_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "lu_read", |backend| {
            four(backend.lu_read(self)?, "lu_read")
        })
    }
    fn full_piv_lu_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "full_piv_lu_read", |backend| {
            five(backend.full_piv_lu_read(self)?, "full_piv_lu_read")
        })
    }
    fn full_piv_lu_solve_read(
        self,
        b: TensorRead<'_>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "full_piv_lu_solve_read", |backend| {
            let b_is_vector = b.shape().len() + 1 == self.shape().len();
            let original_b_shape = b.shape().to_vec();
            let vector_as_matrix = if b_is_vector {
                let mut shape = vec![b.shape()[0], 1];
                shape.extend_from_slice(&b.shape()[1..]);
                Some(backend.reshape_read(b.clone(), &shape)?)
            } else {
                None
            };
            let b = vector_as_matrix.as_ref().map_or(b, TensorRead::from_tensor);
            let (p, l, u, q, _parity) = self.full_piv_lu_read(backend)?;
            let pb = linalg_matmul_read(&p, b, false, backend)?;
            let z = backend.triangular_solve_read(
                TensorRead::from_tensor(&l),
                TensorRead::from_tensor(&pb),
                true,
                true,
                false,
                true,
            )?;
            let w = backend.triangular_solve_read(
                TensorRead::from_tensor(&u),
                TensorRead::from_tensor(&z),
                true,
                false,
                false,
                false,
            )?;
            let mut perm: Vec<usize> = (0..q.shape().len()).collect();
            perm.swap(0, 1);
            let qt = transpose(&q, &perm, backend)?;
            let solution = linalg_matmul_read(&qt, TensorRead::from_tensor(&w), false, backend)?;
            if b_is_vector {
                reshape(&solution, &original_b_shape, backend)
            } else {
                Ok(solution)
            }
        })
    }
    fn solve_read(
        self,
        b: TensorRead<'_>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "solve_read", |backend| backend.solve_read(self, b))
    }
    fn solve_read_into(
        self,
        b: TensorRead<'_>,
        out: TensorWrite<'_>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<()> {
        with_linalg_backend(session, "solve_read_into", |backend| {
            backend.solve_read_into(self, b, out)
        })
    }
    fn cholesky_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "cholesky_read", |backend| {
            backend.cholesky_read(self)
        })
    }
    fn eigh_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "eigh_read", |backend| {
            two(backend.eigh_read(self)?, "eigh_read")
        })
    }
    fn eigh_with_options_read(
        self,
        options: EighOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "eigh_with_options_read", |backend| {
            validate_derivative_eps("eigh_with_options_read", options.derivative_eps)?;
            let mut out = backend.eigh_read(self)?;
            apply_eigh_gauge(options.gauge, &mut out)?;
            two(out, "eigh_with_options_read")
        })
    }
    fn eig_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "eig_read", |backend| {
            two(backend.eig_read(self)?, "eig_read")
        })
    }
    fn triangular_solve_read(
        self,
        b: TensorRead<'_>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "triangular_solve_read", |backend| {
            backend.triangular_solve_read(self, b, left_side, lower, transpose_a, unit_diagonal)
        })
    }
    fn slogdet_read(
        self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "slogdet_read", |backend| {
            slogdet_from_lu(four(backend.lu_read(self)?, "slogdet_read")?, backend)
        })
    }
    fn det_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "det_read", |backend| {
            det_impl(self.slogdet_read(backend)?, backend)
        })
    }
    fn inv_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "inv_read", |backend| inv_read(self, backend))
    }
    fn eigvalsh_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "eigvalsh_read", |backend| {
            backend.eigh_values_read(self)
        })
    }
    fn eigvals_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "eigvals_read", |backend| {
            let materialized = backend.to_contiguous_read(self)?;
            backend.eig_values(&materialized)
        })
    }
    fn pinv_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "pinv_read", |backend| {
            pinv_read(self, None, backend)
        })
    }
    fn pinv_with_rtol_read(
        self,
        rtol: f64,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "pinv_with_rtol_read", |backend| {
            pinv_read(self, Some(rtol), backend)
        })
    }
    fn norm_read(
        self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "norm_read", |backend| {
            norm_from_read(self, ord, dim, keepdim, backend)
        })
    }
}

impl<T: LinalgScalar> TypedTensorLinalgExt<T> for TypedTensor<T> {
    fn svd(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedSvd<T>> {
        with_linalg_backend(session, "svd", |backend| {
            typed_svd(T::tensor_read(self).svd_read(backend)?)
        })
    }

    fn svdvals(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>> {
        with_linalg_backend(session, "svdvals", |backend| {
            typed_output::<<T as TensorScalar>::Real>(T::tensor_read(self).svdvals_read(backend)?)
        })
    }

    fn svd_with_options(
        &self,
        options: SvdOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedSvd<T>> {
        with_linalg_backend(session, "svd_with_options", |backend| {
            typed_svd(T::tensor_read(self).svd_with_options_read(options, backend)?)
        })
    }

    fn qr(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
        with_linalg_backend(session, "qr", |backend| {
            typed_pair_same(T::tensor_read(self).qr_read(backend)?)
        })
    }

    fn qr_with_options(
        &self,
        options: QrOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
        with_linalg_backend(session, "qr_with_options", |backend| {
            typed_pair_same(T::tensor_read(self).qr_with_options_read(options, backend)?)
        })
    }

    fn lu(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedLu<T>> {
        with_linalg_backend(session, "lu", |backend| {
            let (p, l, u, parity) = T::tensor_read(self).lu_read(backend)?;
            Ok((
                typed_output::<T>(p)?,
                typed_output::<T>(l)?,
                typed_output::<T>(u)?,
                typed_output::<<T as TensorScalar>::Real>(parity)?,
            ))
        })
    }

    fn full_piv_lu_solve(
        &self,
        b: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        with_linalg_backend(session, "full_piv_lu_solve", |backend| {
            typed_output::<T>(
                T::tensor_read(self).full_piv_lu_solve_read(T::tensor_read(b), backend)?,
            )
        })
    }

    fn full_piv_lu(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedFullPivLu<T>> {
        with_linalg_backend(session, "full_piv_lu", |backend| {
            let (p, l, u, q, parity) = T::tensor_read(self).full_piv_lu_read(backend)?;
            Ok((
                typed_output::<T>(p)?,
                typed_output::<T>(l)?,
                typed_output::<T>(u)?,
                typed_output::<T>(q)?,
                typed_output::<<T as TensorScalar>::Real>(parity)?,
            ))
        })
    }

    fn solve(
        &self,
        b: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        with_linalg_backend(session, "solve", |backend| {
            typed_output::<T>(T::tensor_read(self).solve_read(T::tensor_read(b), backend)?)
        })
    }

    fn cholesky(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        with_linalg_backend(session, "cholesky", |backend| {
            typed_output::<T>(T::tensor_read(self).cholesky_read(backend)?)
        })
    }

    fn eigh(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)> {
        with_linalg_backend(session, "eigh", |backend| {
            typed_eigh(T::tensor_read(self).eigh_read(backend)?)
        })
    }

    fn eigh_with_options(
        &self,
        options: EighOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)> {
        with_linalg_backend(session, "eigh_with_options", |backend| {
            typed_eigh(T::tensor_read(self).eigh_with_options_read(options, backend)?)
        })
    }

    fn eig(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedEig<T>> {
        with_linalg_backend(session, "eig", |backend| {
            let (values, vectors) = T::tensor_read(self).eig_read(backend)?;
            Ok((
                typed_output::<T::Complex>(values)?,
                typed_output::<T::Complex>(vectors)?,
            ))
        })
    }

    fn triangular_solve(
        &self,
        b: &TypedTensor<T>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        with_linalg_backend(session, "triangular_solve", |backend| {
            typed_output::<T>(T::tensor_read(self).triangular_solve_read(
                T::tensor_read(b),
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                backend,
            )?)
        })
    }

    fn slogdet(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<<T as TensorScalar>::Real>)> {
        with_linalg_backend(session, "slogdet", |backend| {
            let (sign, logabsdet) = T::tensor_read(self).slogdet_read(backend)?;
            Ok((
                typed_output::<T>(sign)?,
                typed_output::<<T as TensorScalar>::Real>(logabsdet)?,
            ))
        })
    }

    fn det(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>> {
        with_linalg_backend(session, "det", |backend| {
            typed_output::<T>(T::tensor_read(self).det_read(backend)?)
        })
    }

    fn inv(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>> {
        with_linalg_backend(session, "inv", |backend| {
            typed_output::<T>(T::tensor_read(self).inv_read(backend)?)
        })
    }

    fn eigvalsh(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>> {
        with_linalg_backend(session, "eigvalsh", |backend| {
            typed_output::<<T as TensorScalar>::Real>(T::tensor_read(self).eigvalsh_read(backend)?)
        })
    }

    fn eigvals(
        &self,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T::Complex>> {
        with_linalg_backend(session, "eigvals", |backend| {
            typed_output::<T::Complex>(T::tensor_read(self).eigvals_read(backend)?)
        })
    }

    fn pinv(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>> {
        with_linalg_backend(session, "pinv", |backend| {
            typed_output::<T>(T::tensor_read(self).pinv_read(backend)?)
        })
    }

    fn pinv_with_rtol(
        &self,
        rtol: f64,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        with_linalg_backend(session, "pinv_with_rtol", |backend| {
            typed_output::<T>(T::tensor_read(self).pinv_with_rtol_read(rtol, backend)?)
        })
    }

    fn norm(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>> {
        with_linalg_backend(session, "norm", |backend| {
            typed_output::<<T as TensorScalar>::Real>(
                T::tensor_read(self).norm_read(ord, dim, keepdim, backend)?,
            )
        })
    }
}

fn typed_svd<T: LinalgScalar>(
    (u, s, vt): (Tensor, Tensor, Tensor),
) -> tenferro_tensor::Result<TypedSvd<T>> {
    Ok((
        typed_output::<T>(u)?,
        typed_output::<<T as TensorScalar>::Real>(s)?,
        typed_output::<T>(vt)?,
    ))
}

fn typed_pair_same<T: LinalgScalar>(
    (a, b): (Tensor, Tensor),
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
    Ok((typed_output::<T>(a)?, typed_output::<T>(b)?))
}

fn typed_eigh<T: LinalgScalar>(
    (values, vectors): (Tensor, Tensor),
) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)> {
    Ok((
        typed_output::<<T as TensorScalar>::Real>(values)?,
        typed_output::<T>(vectors)?,
    ))
}

/// Run a concrete linalg body against the built-in linalg execution sessions
/// (CPU/CUDA) carried by `session`, returning a typed capability error when
/// the session does not expose a linalg execution capability.
///
/// This is the built-in dispatch shared by the concrete linalg surface;
/// callers never downcast themselves (issue #1680 Phase 3). Third-party
/// [`LinalgBackend`] implementations remain supported through the SPI trait,
/// but the concrete op path is built-in-session only. The composite bodies
/// run on the borrowed `&mut dyn LinalgBackend` exactly as before.
pub(crate) fn with_linalg_backend<X>(
    session: &mut dyn BackendSession,
    op: &'static str,
    f: impl FnOnce(&mut dyn LinalgBackend) -> tenferro_tensor::Result<X>,
) -> tenferro_tensor::Result<X> {
    // The capability branches are mutually exclusive, so `f` runs exactly
    // once. Probe the marker first, then re-extract the same exec session and
    // run the composite body on it (FnOnce cannot be captured by several
    // branch closures).
    if with_cpu_exec_session(session, |_| ()).is_some() {
        return with_cpu_exec_session(session, |exec| f(exec as &mut dyn LinalgBackend))
            .expect("marker probe matched a CPU execution session");
    }
    #[cfg(feature = "cuda")]
    if with_cuda_exec_session(session, |_| ()).is_some() {
        return with_cuda_exec_session(session, |exec| f(exec as &mut dyn LinalgBackend))
            .expect("marker probe matched a CUDA execution session");
    }
    Err(tenferro_tensor::Error::unsupported(
        op,
        "selected backend session does not expose a linalg execution capability",
    ))
}

fn typed_output<T: TensorScalar>(tensor: Tensor) -> tenferro_tensor::Result<TypedTensor<T>> {
    if tensor.dtype() != T::dtype() {
        return Err(tenferro_tensor::Error::Internal(format!(
            "typed linalg backend contract expected {:?}, got {:?}",
            T::dtype(),
            tensor.dtype()
        )));
    }
    T::into_typed(tensor)
}

fn arity(name: &'static str, expected: usize, actual: usize) -> tenferro_tensor::Error {
    tenferro_tensor::Error::Internal(format!(
        "{name} backend contract expected {expected} outputs, got {actual}"
    ))
}
fn two(mut out: Vec<Tensor>, name: &'static str) -> tenferro_tensor::Result<(Tensor, Tensor)> {
    if out.len() != 2 {
        return Err(arity(name, 2, out.len()));
    }
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    Ok((a, b))
}
fn three(
    mut out: Vec<Tensor>,
    name: &'static str,
) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
    if out.len() != 3 {
        return Err(arity(name, 3, out.len()));
    }
    let c = out.pop().unwrap();
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    Ok((a, b, c))
}
fn four(
    mut out: Vec<Tensor>,
    name: &'static str,
) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)> {
    if out.len() != 4 {
        return Err(arity(name, 4, out.len()));
    }
    let d = out.pop().unwrap();
    let c = out.pop().unwrap();
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    Ok((a, b, c, d))
}
fn five(
    mut out: Vec<Tensor>,
    name: &'static str,
) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    if out.len() != 5 {
        return Err(arity(name, 5, out.len()));
    }
    let e = out.pop().unwrap();
    let d = out.pop().unwrap();
    let c = out.pop().unwrap();
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    Ok((a, b, c, d, e))
}

// Composite implementations are kept below the surface adapters so every
// backend-visible operation remains explicit and testable.
fn slogdet_from_lu<B: LinalgBackend + ?Sized>(
    (_p, _l, u, parity): (Tensor, Tensor, Tensor, Tensor),
    backend: &mut B,
) -> tenferro_tensor::Result<(Tensor, Tensor)> {
    let diag = backend.extract_diagonal(&u, 0, 1)?;
    let sign = backend.sign_read(TensorRead::from_tensor(&diag))?;
    let sign_u = backend.reduce_prod_read(TensorRead::from_tensor(&sign), &[0])?;
    let sign = backend.mul_read(
        TensorRead::from_tensor(&parity),
        TensorRead::from_tensor(&sign_u),
    )?;
    let abs = backend.abs_read(TensorRead::from_tensor(&diag))?;
    let log = backend.log_read(TensorRead::from_tensor(&abs))?;
    let logabsdet = backend.reduce_sum_read(TensorRead::from_tensor(&log), &[0])?;
    Ok((sign, logabsdet))
}
fn det_impl<B: LinalgBackend + ?Sized>(
    (sign, logabsdet): (Tensor, Tensor),
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let magnitude = backend.exp_read(TensorRead::from_tensor(&logabsdet))?;
    backend.mul_read(
        TensorRead::from_tensor(&sign),
        TensorRead::from_tensor(&magnitude),
    )
}
fn inv_owned<B: LinalgBackend + ?Sized>(
    a: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let eye = eye_like(a.dtype(), a.shape(), backend)?;
    backend.solve(a, &eye)
}
fn inv_read<B: LinalgBackend + ?Sized>(
    a: TensorRead<'_>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let eye = eye_like(a.dtype(), a.shape(), backend)?;
    backend.solve_read(a, TensorRead::from_tensor(&eye))
}
fn pinv_owned<B: LinalgBackend + ?Sized>(
    a: &Tensor,
    rtol: Option<f64>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    ensure_float_or_complex("pinv", a.dtype())?;
    let outputs = three(backend.svd(a)?, "pinv")?;
    pinv_from_svd(
        outputs,
        rtol.unwrap_or_else(|| default_pinv_rtol(a.dtype(), a.shape())),
        backend,
    )
}
fn pinv_read<B: LinalgBackend + ?Sized>(
    a: TensorRead<'_>,
    rtol: Option<f64>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    ensure_float_or_complex("pinv", a.dtype())?;
    let default_rtol = default_pinv_rtol(a.dtype(), a.shape());
    let outputs = three(backend.svd_read(a)?, "pinv_read")?;
    pinv_from_svd(outputs, rtol.unwrap_or(default_rtol), backend)
}
fn norm_from_read<B: LinalgBackend + ?Sized>(
    input: TensorRead<'_>,
    ord: Option<f64>,
    dim: Option<&[usize]>,
    keepdim: bool,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    ensure_float_or_complex("norm", input.dtype())?;
    let original_shape = input.shape().to_vec();
    let axes = dim.map_or_else(
        || (0..original_shape.len()).collect::<Vec<_>>(),
        <[usize]>::to_vec,
    );
    if axes.is_empty() {
        return backend.to_contiguous_read(input);
    }
    validate_axes("norm", original_shape.len(), &axes)?;
    let reduced = if can_square_without_abs(input.dtype(), axes.len(), ord) {
        frobenius_norm_read(input.clone(), &axes, backend)?
    } else {
        let abs = backend.abs_read(input)?;
        match axes.len() {
            1 => norm_over_axes(&abs, &axes, ord, backend)?,
            2 => matrix_norm(&abs, &axes, ord, backend)?,
            _ => norm_over_axes(&abs, &axes, ord, backend)?,
        }
    };
    if !keepdim {
        return Ok(reduced);
    }
    let mut shape = original_shape;
    for &axis in &axes {
        shape[axis] = 1;
    }
    reshape(&reduced, &shape, backend)
}

fn scalar_real(dtype: DType, value: f64) -> tenferro_tensor::Result<Tensor> {
    match dtype {
        DType::F32 => Tensor::from_vec_col_major(vec![], vec![value as f32]),
        DType::F64 => Tensor::from_vec_col_major(vec![], vec![value]),
        DType::C32 => Tensor::from_vec_col_major(vec![], vec![Complex32::new(value as f32, 0.0)]),
        DType::C64 => Tensor::from_vec_col_major(vec![], vec![Complex64::new(value, 0.0)]),
        _ => Err(crate::error::unsupported_dtype("linalg_scalar", dtype)),
    }
}

fn broadcast<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    shape: &[usize],
    dims: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    if input.shape() == shape {
        return input.duplicate();
    }
    backend.broadcast_in_dim_read(TensorRead::from_tensor(input), shape, dims)
}

fn eye_like<B: LinalgBackend + ?Sized>(
    dtype: DType,
    shape: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    if shape.len() < 2 {
        return Err(tenferro_tensor::Error::rank_mismatch("inv", 2, shape.len()));
    }
    let mut diagonal_shape = vec![shape[0]];
    diagonal_shape.extend_from_slice(&shape[2..]);
    let scalar = scalar_real(dtype, 1.0)?;
    let diagonal = broadcast(&scalar, &diagonal_shape, &[], backend)?;
    backend.embed_diagonal(&diagonal, 0, 1)
}

fn ensure_float_or_complex(op: &'static str, dtype: DType) -> tenferro_tensor::Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::C32 | DType::C64 => Ok(()),
        _ => Err(crate::error::unsupported_dtype(op, dtype)),
    }
}

fn can_square_without_abs(dtype: DType, axes_len: usize, ord: Option<f64>) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
        && (ord.is_none() || (ord == Some(2.0) && axes_len != 2))
}

fn validate_axes(op: &'static str, rank: usize, axes: &[usize]) -> tenferro_tensor::Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(tenferro_tensor::Error::axis_out_of_bounds(op, axis, rank));
        }
        if seen[axis] {
            return Err(tenferro_tensor::Error::duplicate_axis(op, axis, "dim"));
        }
        seen[axis] = true;
    }
    Ok(())
}

fn default_pinv_rtol(dtype: DType, shape: &[usize]) -> f64 {
    let max_dim = shape
        .first()
        .copied()
        .unwrap_or(0)
        .max(shape.get(1).copied().unwrap_or(0));
    let eps = match dtype {
        DType::F32 | DType::C32 => f32::EPSILON as f64,
        DType::F64 | DType::C64 => f64::EPSILON,
        _ => 0.0,
    };
    eps * max_dim as f64
}

fn pinv_from_svd<B: LinalgBackend + ?Sized>(
    (u, s, vt): (Tensor, Tensor, Tensor),
    rtol: f64,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let abs_s = backend.abs_read(TensorRead::from_tensor(&s))?;
    let s_max = reduce_max(&abs_s, &[0], backend)?;
    let threshold_scalar = scalar_real(abs_s.dtype(), rtol.max(0.0))?;
    let threshold = binary_mul(&s_max, &threshold_scalar, backend)?;
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, s.shape(), backend)?;
    let mask = {
        let mask = backend.compare_read(
            TensorRead::from_tensor(&abs_s),
            TensorRead::from_tensor(&threshold),
            &CompareDir::Gt,
        )?;
        backend.convert(&mask, s.dtype())?
    };
    let ones = ones_like(&s, backend)?;
    let neg_mask = backend.neg_read(TensorRead::from_tensor(&mask))?;
    let offset = binary_add(&ones, &neg_mask, backend)?;
    let denom = binary_add(&s, &offset, backend)?;
    let s_inv = backend.div_read(
        TensorRead::from_tensor(&mask),
        TensorRead::from_tensor(&denom),
    )?;
    let v = conjugate_transpose(&vt, backend)?;
    let uh = conjugate_transpose(&u, backend)?;
    let vs = scale_matrix_columns(&v, &s_inv, backend)?;
    matmul_preserve_trailing_batch(&vs, &uh, backend)
}

fn ones_like<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let one = scalar_real(input.dtype(), 1.0)?;
    broadcast(&one, input.shape(), &[], backend)
}

fn binary_add<B: LinalgBackend + ?Sized>(
    lhs: &Tensor,
    rhs: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.add_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
}

fn binary_mul<B: LinalgBackend + ?Sized>(
    lhs: &Tensor,
    rhs: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.mul_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
}

fn reduce_max<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.reduce_max_read(TensorRead::from_tensor(input), axes)
}

fn reduce_sum<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.reduce_sum_read(TensorRead::from_tensor(input), axes)
}

fn reduce_min<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.reduce_min_read(TensorRead::from_tensor(input), axes)
}

fn reshape<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    shape: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.reshape_read(TensorRead::from_tensor(input), shape)
}

fn transpose<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    perm: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.transpose_read(TensorRead::from_tensor(input), perm)
}

fn broadcast_batch_scalar_to_leading_axis<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    shape: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let dims: Vec<usize> = (1..shape.len()).collect();
    broadcast(input, shape, &dims, backend)
}

fn conjugate_transpose<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let conj = backend.conj_read(TensorRead::from_tensor(input))?;
    let mut perm: Vec<usize> = (0..input.shape().len()).collect();
    perm.swap(0, 1);
    transpose(&conj, &perm, backend)
}

fn scale_matrix_columns<B: LinalgBackend + ?Sized>(
    matrix: &Tensor,
    scale: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let mut scale_shape = vec![1, scale.shape()[0]];
    scale_shape.extend_from_slice(&matrix.shape()[2..]);
    let reshaped = reshape(scale, &scale_shape, backend)?;
    let dims: Vec<usize> = (0..matrix.shape().len()).collect();
    let expanded = broadcast(&reshaped, matrix.shape(), &dims, backend)?;
    binary_mul(matrix, &expanded, backend)
}

fn matmul_preserve_trailing_batch<B: LinalgBackend + ?Sized>(
    lhs: &Tensor,
    rhs: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let batch: Vec<usize> = (2..lhs.shape().len()).collect();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: batch.clone(),
        rhs_batch_dims: batch,
    };
    backend.dot_general_read(
        TensorRead::from_tensor(lhs),
        TensorRead::from_tensor(rhs),
        &config,
    )
}

fn linalg_matmul_read<B: LinalgBackend + ?Sized>(
    lhs: &Tensor,
    rhs: TensorRead<'_>,
    rhs_is_vector: bool,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let lhs_batch_dims: Vec<usize> = (2..lhs.shape().len()).collect();
    let rhs_batch_start = if rhs_is_vector { 1 } else { 2 };
    let rhs_batch_dims: Vec<usize> = (rhs_batch_start..rhs.shape().len()).collect();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims,
        rhs_batch_dims,
    };
    backend.dot_general_read(TensorRead::from_tensor(lhs), rhs, &config)
}

fn frobenius_norm<B: LinalgBackend + ?Sized>(
    abs: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let sum = backend.reduce_sum_squares_read(TensorRead::from_tensor(abs), axes)?;
    backend.sqrt_read(TensorRead::from_tensor(&sum))
}

fn frobenius_norm_read<B: LinalgBackend + ?Sized>(
    input: TensorRead<'_>,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let sum = backend.reduce_sum_squares_read(input, axes)?;
    backend.sqrt_read(TensorRead::from_tensor(&sum))
}

fn p_norm<B: LinalgBackend + ?Sized>(
    abs: &Tensor,
    axes: &[usize],
    p: f64,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    if !p.is_finite() || p == 0.0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "norm",
            "p",
            format!("p-norm order must be finite and nonzero, got {p}"),
        ));
    }
    if p == 2.0 {
        return frobenius_norm(abs, axes, backend);
    }
    let power = scalar_real(abs.dtype(), p)?;
    let powered = backend.pow_read(
        TensorRead::from_tensor(abs),
        TensorRead::from_tensor(&power),
    )?;
    let sum = reduce_sum(&powered, axes, backend)?;
    let inverse = scalar_real(abs.dtype(), 1.0 / p)?;
    backend.pow_read(
        TensorRead::from_tensor(&sum),
        TensorRead::from_tensor(&inverse),
    )
}

fn count_nonzero<B: LinalgBackend + ?Sized>(
    abs: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let zero = scalar_real(abs.dtype(), 0.0)?;
    let zero = broadcast(&zero, abs.shape(), &[], backend)?;
    let mask = backend.compare_read(
        TensorRead::from_tensor(abs),
        TensorRead::from_tensor(&zero),
        &CompareDir::Gt,
    )?;
    let mask = backend.convert(&mask, abs.dtype())?;
    reduce_sum(&mask, axes, backend)
}

fn norm_over_axes<B: LinalgBackend + ?Sized>(
    abs: &Tensor,
    axes: &[usize],
    ord: Option<f64>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    match ord {
        None => frobenius_norm(abs, axes, backend),
        Some(p) if p == f64::INFINITY => reduce_max(abs, axes, backend),
        Some(p) if p == f64::NEG_INFINITY => reduce_min(abs, axes, backend),
        Some(0.0) => count_nonzero(abs, axes, backend),
        Some(p) => p_norm(abs, axes, p, backend),
    }
}

fn matrix_norm<B: LinalgBackend + ?Sized>(
    abs: &Tensor,
    axes: &[usize],
    ord: Option<f64>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let matrix = move_axes_to_front(abs, axes, backend)?;
    match ord {
        None => frobenius_norm(&matrix, &[0, 1], backend),
        Some(p) if p == f64::INFINITY => row_sum_norm(&matrix, true, backend),
        Some(p) if p == f64::NEG_INFINITY => row_sum_norm(&matrix, false, backend),
        Some(1.0) => col_sum_norm(&matrix, true, backend),
        Some(-1.0) => col_sum_norm(&matrix, false, backend),
        Some(2.0) | Some(-2.0) => {
            let (_, singular_values, _) = three(backend.svd(&matrix)?, "norm")?;
            if ord == Some(2.0) {
                reduce_max(&singular_values, &[0], backend)
            } else {
                reduce_min(&singular_values, &[0], backend)
            }
        }
        Some(0.0) => count_nonzero(&matrix, &[0, 1], backend),
        Some(p) => p_norm(&matrix, &[0, 1], p, backend),
    }
}

fn row_sum_norm<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    take_max: bool,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let sums = reduce_sum(input, &[1], backend)?;
    if take_max {
        reduce_max(&sums, &[0], backend)
    } else {
        reduce_min(&sums, &[0], backend)
    }
}

fn col_sum_norm<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    take_max: bool,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let sums = reduce_sum(input, &[0], backend)?;
    if take_max {
        reduce_max(&sums, &[0], backend)
    } else {
        reduce_min(&sums, &[0], backend)
    }
}

fn move_axes_to_front<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        return input.duplicate();
    }
    let mut selected = vec![false; input.shape().len()];
    for &axis in axes {
        selected[axis] = true;
    }
    let mut perm = axes.to_vec();
    perm.extend(
        selected
            .iter()
            .enumerate()
            .filter_map(|(axis, selected)| (!selected).then_some(axis)),
    );
    transpose(input, &perm, backend)
}
