// Solve residual policy reference: PyTorch 8dd3b763, derivatives.yaml's
// _linalg_solve_ex and FunctionsManual.cpp::linalg_solve_backward. The tracked
// implementation runs the fused LuFactorSolve op, whose saved factors feed the
// LuSolvePrepared adjoint solve.
use std::sync::Arc;

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::{
    apply_eager_with_targeted_extension_in_session, apply_eager_with_targeted_extension_session,
    EagerExtensionBackendKind, EagerExtensionTarget,
};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_cpu::CpuEngine;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaEngine;
#[cfg(feature = "webgpu")]
use tenferro_gpu::webgpu::WebGpuBackend;
use tenferro_runtime::{ErrorPhase, ExtensionModule};

use crate::eager_composites;
use crate::extension::{
    extension_module, validate_derivative_eps, EighOptions, LinalgExtensionOp, LinalgOp, QrOptions,
    SvdOptions,
};
use crate::rank_revealing_qr::validate_rank_revealing_qr_options;
use crate::{RankRevealingQrOptions, RankRevealingQrResult};

/// Tensor-owned linear solve. Other eager linear-algebra operations use
/// [`EagerSessionLinalgExt`] inside [`tenferro_ad::EagerRuntime::with_eager_session`].
///
/// The tensor-owned entry preserves calling-thread `no_grad` behavior for
/// tracked solves; it must not be called from inside a borrowed session.
///
/// # Examples
/// ```rust
/// # use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// # use tenferro_linalg::EagerTensorLinalgExt;
/// # let ctx = EagerRuntime::new()?;
/// # let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?, ctx.clone())?;
/// # let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?, ctx)?;
/// let x = a.solve(&b)?;
/// assert_eq!(x.shape(), &[1, 1]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub trait EagerTensorLinalgExt {
    /// The CUDA provider does not inspect the cuSOLVER `getrf` status flag on
    /// the forward solve path, so it performs no per-call non-singularity check
    /// and issues no host round trip. An exactly singular input is left to the
    /// caller's verification, as in `torch.linalg.solve`. CPU providers may
    /// still report their own exactly singular factor as `Error::Singular`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible matrix, batch, or dtype
    /// metadata, `Error::Extension` for an unsupported dtype or singular
    /// system, and `Error::RuntimeState` when the backend is unavailable.
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::EagerTensorLinalgExt;
    /// # let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// # let a = EagerTensor::from_tensor_in(
    /// #     Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap(),
    /// #     ctx.clone(),
    /// # )?;
    /// # let b = EagerTensor::from_tensor_in(
    /// #     Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0]).unwrap(),
    /// #     ctx,
    /// # )?;
    /// let x = a.solve(&b)?;
    /// assert_eq!(x.value()?.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    fn solve(&self, b: &EagerTensor) -> Result<EagerTensor>;
}

impl EagerTensorLinalgExt for EagerTensor {
    fn solve(&self, b: &EagerTensor) -> Result<EagerTensor> {
        solve(self, b)
    }
}

/// Linear algebra operations on a runtime-bound borrowed eager session.
///
/// # Examples
/// ```rust
/// use tenferro_ad::{EagerRuntime, Tensor};
/// use tenferro_linalg::EagerSessionLinalgExt;
/// let ctx = EagerRuntime::new()?;
/// let factor = ctx.with_eager_session(|session| {
///     let input = session.constant_from(Tensor::from_vec_col_major(vec![1, 1], vec![4.0_f64])?)?;
///     session.cholesky(&input)
/// })?;
/// assert_eq!(factor.value()?.as_slice::<f64>()?, &[2.0]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub trait EagerSessionLinalgExt {
    /// Compute the lower Cholesky factor without reopening the eager backend.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let factor = ctx.with_eager_session(|session| {
    ///     let input = session.constant_from(Tensor::from_vec_col_major(vec![1, 1], vec![9.0_f64])?)?;
    ///     session.cholesky(&input)
    /// })?;
    /// assert_eq!(factor.value()?.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, module or unsupported-executor errors.
    fn cholesky(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Compute a thin singular value decomposition in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let (_u, values, _vt) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![3.0_f64])?)?;
    ///     s.svd(&a)
    /// })?;
    /// assert_eq!(values.value()?.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn svd(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;

    /// Compute thin SVD with an explicit gauge, driver, and derivative regularizer.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, SvdOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let (_u, values, _vt) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![3.0_f64])?)?;
    ///     s.svd_with_options(&a, SvdOptions::default())
    /// })?;
    /// assert_eq!(values.value()?.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed invalid-tolerance, extension, backend, or unsupported errors.
    fn svd_with_options(
        &mut self,
        input: &EagerTensor,
        options: SvdOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;

    /// Compute full-matrices SVD, including the right nullspace.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let (u, values, vh) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 2], vec![1.0_f64, 1.0])?)?;
    ///     s.svd_full(&a)
    /// })?;
    /// assert_eq!(u.shape(), &[1, 1]);
    /// assert_eq!(values.shape(), &[1]);
    /// assert_eq!(vh.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` when the
    /// input is not a (batched) matrix, `Error::UnsupportedAdRule` when a
    /// traced input needs a derivative this decomposition does not provide,
    /// `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn svd_full(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;

    /// Compute a thin QR decomposition in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let (q, r) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![3.0_f64])?)?;
    ///     s.qr(&a)
    /// })?;
    /// assert_eq!(q.shape(), &[1, 1]);
    /// assert_eq!(r.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn qr(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    /// Compute QR with an explicit post-processing gauge.
    ///
    /// The CUDA provider does not read the cuSOLVER `geqrf`/`orgqr` status
    /// flags, so forward QR issues no host round trip and reports neither a
    /// vendor parameter error nor a rank deficiency. A rank-deficient input
    /// still yields valid factors.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, QrOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let (q, r) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![3.0_f64])?)?;
    ///     s.qr_with_options(&a, QrOptions::default())
    /// })?;
    /// assert_eq!(q.shape(), &[1, 1]);
    /// assert_eq!(r.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported errors.
    fn qr_with_options(
        &mut self,
        input: &EagerTensor,
        options: QrOptions,
    ) -> Result<(EagerTensor, EagerTensor)>;

    /// Solve a triangular system in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let x = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.triangular_solve(&a, &b, true, true, false, false)
    /// })?;
    /// assert_eq!(x.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn triangular_solve(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<EagerTensor>;

    /// Solve a square linear system in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let x = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.solve(&a, &b)
    /// })?;
    /// assert_eq!(x.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// The CUDA provider does not inspect the cuSOLVER `getrf` status flag on
    /// the forward solve path, so it performs no per-call non-singularity check
    /// and issues no host round trip. An exactly singular input is left to the
    /// caller's verification, as in `torch.linalg.solve`. CPU providers may
    /// still report their own exactly singular factor as `Error::Singular`.
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor>;

    /// Solve a full-column-rank least-squares problem inside this session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let x = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![1.0_f64, 2.0])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![2.0_f64, 4.0])?)?;
    ///     s.lstsq(&a, &b)
    /// })?;
    /// assert!((x.value()?.as_slice::<f64>()?[0] - 2.0).abs() < 1e-12);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed rank/shape/dtype validation, extension, backend, or unsupported errors.
    fn lstsq(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor>;

    /// Factor a matrix without leaving this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let (_p, _l, u, _parity) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.lu(&a)
    /// })?;
    /// assert_eq!(u.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)>;

    /// Compute determinant sign and logarithm of its absolute value in this session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let (sign, logabs) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.slogdet(&a)
    /// })?;
    /// assert_eq!(sign.value()?.as_slice::<f64>()?, &[1.0]);
    /// assert!((logabs.value()?.as_slice::<f64>()?[0] - 2.0_f64.ln()).abs() < 1e-12);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn slogdet(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    /// Compute a determinant inside the borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let det = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.det(&a)
    /// })?;
    /// assert_eq!(det.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn det(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Invert a square matrix inside this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let inverse = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.inv(&a)
    /// })?;
    /// assert_eq!(inverse.value()?.as_slice::<f64>()?, &[0.5]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed rank/shape/dtype validation, extension, backend, or unsupported errors.
    fn inv(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Compute eigenvalues of a Hermitian matrix in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let values = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigvalsh(&a)
    /// })?;
    /// assert_eq!(values.value()?.as_slice::<f64>()?, &[4.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn eigvalsh(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Compute general eigenvalues in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let values = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigvals(&a)
    /// })?;
    /// assert_eq!(values.value()?.as_slice::<num_complex::Complex64>()?, &[num_complex::Complex64::new(4.0, 0.0)]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn eigvals(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Compute eigenvalues and vectors of a Hermitian matrix in this session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let (values, vectors) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigh(&a)
    /// })?;
    /// assert_eq!(values.value()?.as_slice::<f64>()?, &[4.0]);
    /// assert_eq!(vectors.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, unsupported-dtype, extension, or backend errors.
    fn eigh(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    /// Compute Hermitian eigendecomposition with explicit gauge and tolerance.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, EighOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let (values, _vectors) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigh_with_options(&a, EighOptions::default())
    /// })?;
    /// assert_eq!(values.value()?.as_slice::<f64>()?, &[4.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::InvalidArgument` for an
    /// invalid tolerance or with `ValidationError::ShapeMismatch` for a
    /// non-square input, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn eigh_with_options(
        &mut self,
        input: &EagerTensor,
        options: EighOptions,
    ) -> Result<(EagerTensor, EagerTensor)>;

    /// Compute general eigenvalues and eigenvectors inside this session.
    /// Eigenvectors come back with unit 2-norm columns, whichever provider computed them;
    /// a column the provider left non-finite is reported as a numerical failure instead
    /// of being returned as `NaN` (#2148).
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let (values, vectors) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eig(&a)
    /// })?;
    /// assert_eq!(values.value()?.as_slice::<num_complex::Complex64>()?, &[num_complex::Complex64::new(4.0, 0.0)]);
    /// assert_eq!(vectors.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` or
    /// `ValidationError::ShapeMismatch` for a non-square (batched) matrix,
    /// `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn eig(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    /// Compute the Moore-Penrose pseudoinverse using the default tolerance.
    ///
    /// The cutoff is applied to an accuracy-preserving default SVD
    /// (`SvdDriver::Auto`), so the retained rank matches the CPU reference.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let inverse = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.pinv(&a)
    /// })?;
    /// assert_eq!(inverse.value()?.as_slice::<f64>()?, &[0.5]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` when the
    /// input is not a (batched) matrix, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype) from the
    /// underlying SVD, or `Error::TensorRuntime` for a backend failure.
    fn pinv(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Compute the pseudoinverse with an explicit relative tolerance.
    ///
    /// Like [`Self::pinv`], the threshold is applied to an accuracy-preserving
    /// default SVD (`SvdDriver::Auto`).
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let inverse = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.pinv_with_rtol(&a, 1.0e-12)
    /// })?;
    /// assert_eq!(inverse.value()?.as_slice::<f64>()?, &[0.5]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` when the
    /// input is not a (batched) matrix, `ValidationError::InvalidArgument` for a
    /// negative or non-finite tolerance, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype) from
    /// the underlying SVD, or `Error::TensorRuntime` for a backend failure.
    fn pinv_with_rtol(&mut self, input: &EagerTensor, rtol: f64) -> Result<EagerTensor>;

    /// Compute a vector, matrix, or tensor norm in this session.
    /// An empty axis list is a no-op and clones the input without dispatch.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2], vec![3.0_f64, 4.0])?)?;
    ///     s.norm(&a, Some(2.0), Some(&[0]), false)
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<f64>()?, &[5.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed unsupported-dtype, invalid-axis/order, SVD, or backend errors.
    fn norm(
        &mut self,
        input: &EagerTensor,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<EagerTensor>;

    /// Compute complete-pivot LU factors `(P, L, U, Q, parity)` in this session.
    /// Reconstruction uses `A = P^T * L * U * Q`; scalar `parity` is real
    /// (`F32` for `F32`/`C32` inputs and `F64` for `F64`/`C64`).
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let (p, _l, _u, q, parity) = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?)?;
    ///     s.full_piv_lu(&a)
    /// })?;
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(parity.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed rank/shape, unsupported-provider, numerical, or output-count errors.
    fn full_piv_lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
    )>;

    /// Solve a linear system using complete-pivot LU in this session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let x = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.full_piv_lu_solve(&a, &b)
    /// })?;
    /// assert_eq!(x.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed rank/shape, unsupported-provider, singularity, or backend errors.
    fn full_piv_lu_solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor)
        -> Result<EagerTensor>;

    /// Compute column-pivoted rank-revealing QR in this session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, RankRevealingQrOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?)?;
    ///     s.rank_revealing_qr(&a, RankRevealingQrOptions::default())
    /// })?;
    /// assert_eq!(result.column_permutation.shape(), &[2]);
    /// assert_eq!(result.rank.value()?.as_slice::<i64>()?, &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch`,
    /// `ValidationError::DTypeMismatch` or `ValidationError::InvalidArgument` for
    /// an invalid rank, dtype or tolerance, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn rank_revealing_qr(
        &mut self,
        input: &EagerTensor,
        options: RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<EagerTensor>>;

    /// Initialize compact Householder QR state in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::EagerSessionLinalgExt;
    /// let ctx = EagerRuntime::new()?;
    /// let state = ctx.with_eager_session(|s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![1.0_f64, 2.0])?)?;
    ///     s.householder_qr(&a)
    /// })?;
    /// assert!(format!("{state:?}").starts_with("HouseholderQr"));
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed invalid metadata, unsupported executor, or backend errors.
    fn householder_qr(&mut self, input: &EagerTensor) -> Result<crate::HouseholderQr<EagerTensor>>;
}

impl EagerSessionLinalgExt for EagerSession<'_> {
    fn svd(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        self.svd_with_options(input, SvdOptions::default())
    }

    fn svd_with_options(
        &mut self,
        input: &EagerTensor,
        options: SvdOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        validate_derivative_eps("svd_with_options", options.derivative_eps)?;
        three_outputs(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::Svd {
                    derivative_eps: options.derivative_eps,
                    gauge: options.gauge,
                    driver: options.driver,
                },
                &[input],
            )?,
            "svd",
        )
    }

    fn svd_full(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        three_outputs(
            apply_linalg_eager_in_session(self, LinalgOp::SvdFull, &[input])?,
            "svd_full",
        )
    }

    fn qr(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        self.qr_with_options(input, QrOptions::default())
    }

    fn qr_with_options(
        &mut self,
        input: &EagerTensor,
        options: QrOptions,
    ) -> Result<(EagerTensor, EagerTensor)> {
        two_outputs(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::Qr {
                    gauge: options.gauge,
                },
                &[input],
            )?,
            "qr",
        )
    }

    fn cholesky(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(self, LinalgOp::Cholesky, &[input])?,
            "cholesky",
        )
    }

    fn triangular_solve(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::TriangularSolve {
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                },
                &[matrix, rhs],
            )?,
            "triangular_solve",
        )
    }

    fn solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
        if !matrix.tracks_grad() && !rhs.tracks_grad() {
            return one_output(
                apply_linalg_eager_in_session(self, LinalgOp::Solve, &[matrix, rhs])?,
                "solve",
            );
        }
        validate_tracked_solve_inputs(matrix, rhs)?;
        factor_solve_output(apply_linalg_eager_in_session(
            self,
            LinalgOp::LuFactorSolve,
            &[matrix, rhs],
        )?)
    }

    fn lstsq(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::lstsq(self, matrix, rhs)
    }

    fn lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)> {
        let mut outputs = apply_linalg_eager_in_session(self, LinalgOp::Lu, &[input])?.into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(p), Some(l), Some(u), Some(parity), None) => Ok((p, l, u, parity)),
            _ => Err(Error::Internal(
                "lu eager op returned an unexpected number of outputs".into(),
            )),
        }
    }

    fn slogdet(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        eager_composites::slogdet(self, input)
    }

    fn det(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::det(self, input)
    }

    fn inv(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::inv(self, input)
    }

    fn eigvalsh(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::EighVals {
                    derivative_eps: crate::extension::DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    driver: EighOptions::default().driver,
                },
                &[input],
            )?,
            "eigvalsh",
        )
    }

    fn eigvals(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::EigVals {
                    input_dtype: input.dtype(),
                },
                &[input],
            )?,
            "eigvals",
        )
    }

    fn eigh(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        self.eigh_with_options(input, EighOptions::default())
    }

    fn eigh_with_options(
        &mut self,
        input: &EagerTensor,
        options: EighOptions,
    ) -> Result<(EagerTensor, EagerTensor)> {
        validate_derivative_eps("eigh_with_options", options.derivative_eps)?;
        two_outputs(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::Eigh {
                    derivative_eps: options.derivative_eps,
                    gauge: options.gauge,
                    driver: options.driver,
                },
                &[input],
            )?,
            "eigh",
        )
    }

    fn eig(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        two_outputs(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::Eig {
                    input_dtype: input.dtype(),
                },
                &[input],
            )?,
            "eig",
        )
    }

    fn pinv(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::pinv(self, input)
    }

    fn pinv_with_rtol(&mut self, input: &EagerTensor, rtol: f64) -> Result<EagerTensor> {
        eager_composites::pinv_with_rtol(self, input, rtol)
    }

    fn norm(
        &mut self,
        input: &EagerTensor,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<EagerTensor> {
        eager_composites::norm(self, input, ord, dim, keepdim)
    }

    fn full_piv_lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
    )> {
        let mut outputs =
            apply_linalg_eager_in_session(self, LinalgOp::FullPivLu, &[input])?.into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => Ok((p, l, u, q, parity)),
            _ => Err(Error::Internal(
                "full_piv_lu eager op returned an unexpected number of outputs".into(),
            )),
        }
    }

    fn full_piv_lu_solve(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
    ) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::FullPivLuSolve { transpose_a: false },
                &[matrix, rhs],
            )?,
            "full_piv_lu_solve",
        )
    }

    fn rank_revealing_qr(
        &mut self,
        input: &EagerTensor,
        options: RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<EagerTensor>> {
        validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
        let mut outputs = apply_linalg_eager_in_session(
            self,
            LinalgOp::RankRevealingQr {
                gauge: options.gauge,
                rtol: options.rtol,
                atol: options.atol,
            },
            &[input],
        )?
        .into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(q), Some(r), Some(column_permutation), Some(rank), None) => {
                Ok(RankRevealingQrResult {
                    q,
                    r,
                    column_permutation,
                    rank,
                })
            }
            _ => Err(Error::Internal(
                "rank_revealing_qr eager op returned an unexpected number of outputs".into(),
            )),
        }
    }

    fn householder_qr(&mut self, input: &EagerTensor) -> Result<crate::HouseholderQr<EagerTensor>> {
        crate::householder::eager_state(apply_linalg_eager_in_session(
            self,
            LinalgOp::HouseholderQrFactor,
            &[input],
        )?)
    }
}

pub(crate) fn apply_linalg_eager_in_session(
    session: &mut EagerSession<'_>,
    op: LinalgOp,
    inputs: &[&EagerTensor],
) -> Result<Vec<EagerTensor>> {
    let op = Arc::new(LinalgExtensionOp::new(op));
    apply_eager_with_targeted_extension_in_session(session, op, inputs, eager_extension_module)
}

pub(crate) fn apply_linalg_eager(
    op: LinalgOp,
    inputs: &[&EagerTensor],
) -> Result<Vec<EagerTensor>> {
    let op = Arc::new(LinalgExtensionOp::new(op));
    apply_eager_with_targeted_extension_session(op, inputs, eager_extension_module)
}

fn eager_extension_module(target: EagerExtensionTarget) -> Result<Arc<dyn ExtensionModule>> {
    let EagerExtensionTarget {
        engine_id,
        backend_kind,
    } = target;
    match backend_kind {
        EagerExtensionBackendKind::Cpu => {
            extension_module::<CpuEngine>(engine_id).map_err(eager_runtime_config_error)
        }
        #[cfg(feature = "cuda")]
        EagerExtensionBackendKind::Cuda => {
            extension_module::<CudaEngine>(engine_id).map_err(eager_runtime_config_error)
        }
        #[cfg(feature = "webgpu")]
        EagerExtensionBackendKind::WebGpu => {
            extension_module::<WebGpuBackend>(engine_id).map_err(eager_runtime_config_error)
        }
    }
}

fn eager_runtime_config_error(source: tenferro_runtime::RuntimeConfigError) -> Error {
    Error::runtime_state_source(
        "tenferro_linalg::eager_extension_module",
        ErrorPhase::Execution,
        source,
    )
}

/// Solve a linear system for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new()?;
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap(),
///     ctx.clone(),
/// ).unwrap();
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let x = a.solve(&b)?;
/// assert_eq!(x.shape(), &[2, 1]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// The CUDA provider does not inspect the cuSOLVER `getrf` status flag on
/// the forward solve path, so it performs no per-call non-singularity check
/// and issues no host round trip. An exactly singular input is left to the
/// caller's verification, as in `torch.linalg.solve`. CPU providers may
/// still report their own exactly singular factor as `Error::Singular`.
///
/// # Errors
///
/// Returns `Error::Validation` for incompatible matrix, batch, or dtype
/// metadata, `Error::Extension` for an unsupported dtype or singular system,
/// and `Error::RuntimeState` when the backend is unavailable.
pub fn solve(a: &EagerTensor, b: &EagerTensor) -> Result<EagerTensor> {
    if !a.tracks_grad() && !b.tracks_grad() {
        return one_output(apply_linalg_eager(LinalgOp::Solve, &[a, b])?, "solve");
    }
    validate_tracked_solve_inputs(a, b)?;
    factor_solve_output(apply_linalg_eager(LinalgOp::LuFactorSolve, &[a, b])?)
}

fn validate_tracked_solve_inputs(a: &EagerTensor, b: &EagerTensor) -> Result<()> {
    if !a.same_context(b) {
        return Err(Error::ContextMismatch {
            lhs: a.ctx_id(),
            rhs: b.ctx_id(),
        });
    }
    crate::validation::validate_solve_inputs(a.dtype(), a.shape(), b.dtype(), b.shape())
}

// One fused factor+solve retains LU/pivots and X for backward, while the
// explicit A operand preserves higher-order semantics (PyTorch's
// _linalg_solve_ex / FunctionsManual.cpp::linalg_solve_backward).
fn factor_solve_output(outputs: Vec<EagerTensor>) -> Result<EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(x), Some(_packed_lu), Some(_pivots), None) => Ok(x),
        _ => Err(Error::Internal(
            "lu_factor_solve eager op returned an unexpected number of outputs".into(),
        )),
    }
}

fn three_outputs(
    outputs: Vec<EagerTensor>,
    name: &str,
) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), None) => Ok((first, second, third)),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}

pub(crate) fn one_output(outputs: Vec<EagerTensor>, name: &str) -> Result<EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}

fn two_outputs(outputs: Vec<EagerTensor>, name: &str) -> Result<(EagerTensor, EagerTensor)> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(lhs), Some(rhs), None) => Ok((lhs, rhs)),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}
