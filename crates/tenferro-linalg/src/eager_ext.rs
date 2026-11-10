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
use tenferro_runtime::{ErrorPhase, ExecutionEngine, ExtensionModule};

use crate::eager_composites;
use crate::extension::{
    extension_module, validate_derivative_eps, EighOptions, LinalgExtensionOp, LinalgOp, QrOptions,
    SvdOptions,
};
use crate::rank_revealing_qr::validate_rank_revealing_qr_options;
use crate::{
    CholeskyOptions, DetOptions, Diagonal, EigOptions, EigvalsOptions, EigvalshOptions,
    FullPivLuOptions, FullPivLuSolveOptions, HouseholderQrOptions, InvOptions, KeepDims,
    LstsqOptions, LuOptions, NormOptions, PinvOptions, RankRevealingQrOptions,
    RankRevealingQrResult, Side, SlogdetOptions, SolveOptions, SvdShape, Transpose,
    TriangularSolveOptions, Uplo,
};

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
/// let x = a.solve(&mut tenferro_cpu::CpuEngine::serial(), &b)?;
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
    /// # let ctx = EagerRuntime::new()?;
    /// # let a = EagerTensor::from_tensor_in(
    /// #     Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap(),
    /// #     ctx.clone(),
    /// # )?;
    /// # let b = EagerTensor::from_tensor_in(
    /// #     Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0]).unwrap(),
    /// #     ctx,
    /// # )?;
    /// let x = a.solve(&mut tenferro_cpu::CpuEngine::serial(), &b)?;
    /// assert_eq!(x.value()?.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    fn solve(&self, engine: &mut dyn ExecutionEngine, b: &EagerTensor) -> Result<EagerTensor>;
}

impl EagerTensorLinalgExt for EagerTensor {
    fn solve(&self, engine: &mut dyn ExecutionEngine, b: &EagerTensor) -> Result<EagerTensor> {
        solve(engine, self, b)
    }
}

/// Linear algebra operations on a runtime-bound borrowed eager session.
///
/// Every operation `op(&x, ..)` has a configurable form `op_with(&x, .., &Options)`; the
/// short form is the `_with` form with default options.
///
/// # Examples
/// ```rust
/// use tenferro_ad::{EagerRuntime, Tensor};
/// use tenferro_linalg::EagerSessionLinalgExt;
/// let ctx = EagerRuntime::new()?;
/// let factor = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
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
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![9.0_f64])?)?;
    ///     s.cholesky(&a)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn cholesky(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::cholesky`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, CholeskyOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![9.0_f64])?)?;
    ///     s.cholesky_with(&a, &CholeskyOptions::default())
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn cholesky_with(
        &mut self,
        input: &EagerTensor,
        options: &CholeskyOptions,
    ) -> Result<EagerTensor>;

    /// Compute an SVD in this borrowed session.
    ///
    /// The default options request the thin factorization with the default gauge, driver, and
    /// derivative regularizer. `SvdShape::Full` requests the full-matrices factorization,
    /// including the right nullspace; it does not combine with a non-default gauge, driver, or
    /// derivative epsilon.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![3.0_f64])?)?;
    ///     s.svd(&a)
    /// })?;
    /// let (_u, values, _vt) = r;
    /// assert_eq!(values.value()?.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed invalid-tolerance, extension, backend, or unsupported errors, and
    /// `Error::Unsupported` for `SvdShape::Full` combined with a non-default gauge, driver, or
    /// derivative epsilon.
    fn svd(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;

    /// Configurable form of [`Self::svd`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, SvdShape, SvdOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 2], vec![1.0_f64, 1.0])?)?;
    ///     s.svd_with(&a, &SvdOptions::default().with_shape(SvdShape::Full))
    /// })?;
    /// let (u, values, vh) = r;
    /// assert_eq!(u.shape(), &[1, 1]);
    /// assert_eq!(values.shape(), &[1]);
    /// assert_eq!(vh.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed invalid-tolerance, extension, backend, or unsupported errors, and
    /// `Error::Unsupported` for `SvdShape::Full` combined with a non-default gauge, driver, or
    /// derivative epsilon.
    fn svd_with(
        &mut self,
        input: &EagerTensor,
        options: &SvdOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;

    /// Compute a thin QR decomposition in this borrowed session.
    ///
    /// The CUDA provider does not read the cuSOLVER `geqrf`/`orgqr` status
    /// flags, so forward QR issues no host round trip and reports neither a
    /// vendor parameter error nor a rank deficiency. A rank-deficient input
    /// still yields valid factors.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![3.0_f64])?)?;
    ///     s.qr(&a)
    /// })?;
    /// let (q, r) = r;
    /// assert_eq!(q.shape(), &[1, 1]);
    /// assert_eq!(r.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn qr(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    /// Configurable form of [`Self::qr`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, QrOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![3.0_f64])?)?;
    ///     s.qr_with(&a, &QrOptions::default())
    /// })?;
    /// let (q, r) = r;
    /// assert_eq!(q.shape(), &[1, 1]);
    /// assert_eq!(r.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn qr_with(
        &mut self,
        input: &EagerTensor,
        options: &QrOptions,
    ) -> Result<(EagerTensor, EagerTensor)>;

    /// Solve a triangular system in this borrowed session.
    ///
    /// The default options solve `X op(A) = B` with the upper triangle, no transpose, and a
    /// stored diagonal; set the `TriangularSolveOptions` fields to change the side, triangle,
    /// transpose, or diagonal.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.triangular_solve(&a, &b)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn triangular_solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::triangular_solve`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, Side, TriangularSolveOptions, Uplo};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.triangular_solve_with(&a, &b, &TriangularSolveOptions::default().with_side(Side::Left).with_uplo(Uplo::Lower))
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn triangular_solve_with(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        options: &TriangularSolveOptions,
    ) -> Result<EagerTensor>;

    /// Solve a square linear system in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.solve(&a, &b)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// The CUDA provider does not inspect the cuSOLVER `getrf` status flag on
    /// the forward solve path, so it performs no per-call non-singularity check
    /// and issues no host round trip. An exactly singular input is left to the
    /// caller's verification, as in `torch.linalg.solve`. CPU providers may
    /// still report their own exactly singular factor as `Error::Singular`.
    ///
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::solve`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, SolveOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.solve_with(&a, &b, &SolveOptions::default())
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn solve_with(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        options: &SolveOptions,
    ) -> Result<EagerTensor>;

    /// Solve a full-column-rank least-squares problem inside this session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![1.0_f64, 2.0])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![2.0_f64, 4.0])?)?;
    ///     s.lstsq(&a, &b)
    /// })?;
    /// assert!((r.value()?.as_slice::<f64>()?[0] - 2.0).abs() < 1e-12);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed rank/shape/dtype validation, extension, backend, or unsupported errors.
    fn lstsq(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::lstsq`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, LstsqOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![1.0_f64, 2.0])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![2.0_f64, 4.0])?)?;
    ///     s.lstsq_with(&a, &b, &LstsqOptions::default())
    /// })?;
    /// assert!((r.value()?.as_slice::<f64>()?[0] - 2.0).abs() < 1e-12);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed rank/shape/dtype validation, extension, backend, or unsupported errors.
    fn lstsq_with(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        options: &LstsqOptions,
    ) -> Result<EagerTensor>;

    /// Factor a matrix without leaving this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.lu(&a)
    /// })?;
    /// let (_p, _l, u, _parity) = r;
    /// assert_eq!(u.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)>;

    /// Configurable form of [`Self::lu`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, LuOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.lu_with(&a, &LuOptions::default())
    /// })?;
    /// let (_p, _l, u, _parity) = r;
    /// assert_eq!(u.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn lu_with(
        &mut self,
        input: &EagerTensor,
        options: &LuOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)>;

    /// Compute determinant sign and logarithm of its absolute value in this session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.slogdet(&a)
    /// })?;
    /// let (sign, logabs) = r;
    /// assert_eq!(sign.value()?.as_slice::<f64>()?, &[1.0]);
    /// assert!((logabs.value()?.as_slice::<f64>()?[0] - 2.0_f64.ln()).abs() < 1e-12);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn slogdet(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    /// Configurable form of [`Self::slogdet`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, SlogdetOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.slogdet_with(&a, &SlogdetOptions::default())
    /// })?;
    /// let (sign, logabs) = r;
    /// assert_eq!(sign.value()?.as_slice::<f64>()?, &[1.0]);
    /// assert!((logabs.value()?.as_slice::<f64>()?[0] - 2.0_f64.ln()).abs() < 1e-12);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn slogdet_with(
        &mut self,
        input: &EagerTensor,
        options: &SlogdetOptions,
    ) -> Result<(EagerTensor, EagerTensor)>;

    /// Compute a determinant inside the borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.det(&a)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn det(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::det`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, DetOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.det_with(&a, &DetOptions::default())
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn det_with(&mut self, input: &EagerTensor, options: &DetOptions) -> Result<EagerTensor>;

    /// Invert a square matrix inside this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.inv(&a)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[0.5]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed rank/shape/dtype validation, extension, backend, or unsupported errors.
    fn inv(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::inv`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, InvOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.inv_with(&a, &InvOptions::default())
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[0.5]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed rank/shape/dtype validation, extension, backend, or unsupported errors.
    fn inv_with(&mut self, input: &EagerTensor, options: &InvOptions) -> Result<EagerTensor>;

    /// Compute eigenvalues of a Hermitian matrix in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigvalsh(&a)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[4.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn eigvalsh(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::eigvalsh`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, EigvalshOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigvalsh_with(&a, &EigvalshOptions::default())
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[4.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn eigvalsh_with(
        &mut self,
        input: &EagerTensor,
        options: &EigvalshOptions,
    ) -> Result<EagerTensor>;

    /// Compute general eigenvalues in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigvals(&a)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<num_complex::Complex64>()?, &[num_complex::Complex64::new(4.0, 0.0)]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn eigvals(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::eigvals`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, EigvalsOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigvals_with(&a, &EigvalsOptions::default())
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<num_complex::Complex64>()?, &[num_complex::Complex64::new(4.0, 0.0)]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed validation, extension, backend, or unsupported-executor errors.
    fn eigvals_with(
        &mut self,
        input: &EagerTensor,
        options: &EigvalsOptions,
    ) -> Result<EagerTensor>;

    /// Compute eigenvalues and vectors of a Hermitian matrix in this session.
    ///
    /// The options choose the eigenvector gauge, the derivative regularizer, and the driver.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigh(&a)
    /// })?;
    /// let (values, vectors) = r;
    /// assert_eq!(values.value()?.as_slice::<f64>()?, &[4.0]);
    /// assert_eq!(vectors.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::InvalidArgument` for an
    /// invalid tolerance or with `ValidationError::ShapeMismatch` for a
    /// non-square input, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn eigh(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    /// Configurable form of [`Self::eigh`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, EighOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eigh_with(&a, &EighOptions::default())
    /// })?;
    /// let (values, vectors) = r;
    /// assert_eq!(values.value()?.as_slice::<f64>()?, &[4.0]);
    /// assert_eq!(vectors.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::InvalidArgument` for an
    /// invalid tolerance or with `ValidationError::ShapeMismatch` for a
    /// non-square input, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn eigh_with(
        &mut self,
        input: &EagerTensor,
        options: &EighOptions,
    ) -> Result<(EagerTensor, EagerTensor)>;

    /// Compute general eigenvalues and eigenvectors inside this session.
    /// Eigenvectors come back with unit 2-norm columns, whichever provider computed them;
    /// a column the provider left non-finite is reported as a numerical failure instead
    /// of being returned as `NaN` (#2148).
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eig(&a)
    /// })?;
    /// let (values, vectors) = r;
    /// assert_eq!(values.value()?.as_slice::<num_complex::Complex64>()?, &[num_complex::Complex64::new(4.0, 0.0)]);
    /// assert_eq!(vectors.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` or
    /// `ValidationError::ShapeMismatch` for a non-square (batched) matrix,
    /// `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn eig(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)>;

    /// Configurable form of [`Self::eig`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, EigOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.eig_with(&a, &EigOptions::default())
    /// })?;
    /// let (values, vectors) = r;
    /// assert_eq!(values.value()?.as_slice::<num_complex::Complex64>()?, &[num_complex::Complex64::new(4.0, 0.0)]);
    /// assert_eq!(vectors.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` or
    /// `ValidationError::ShapeMismatch` for a non-square (batched) matrix,
    /// `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn eig_with(
        &mut self,
        input: &EagerTensor,
        options: &EigOptions,
    ) -> Result<(EagerTensor, EagerTensor)>;

    /// Compute the Moore-Penrose pseudoinverse.
    ///
    /// With `PinvOptions::rtol` unset the cutoff is the default tolerance. The cutoff is applied
    /// to an accuracy-preserving default SVD (`SvdDriver::Auto`), so the retained rank matches
    /// the CPU reference.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.pinv(&a)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[0.5]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` when the
    /// input is not a (batched) matrix, `ValidationError::InvalidArgument` for a
    /// negative or non-finite tolerance, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype) from the
    /// underlying SVD, or `Error::TensorRuntime` for a backend failure.
    fn pinv(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::pinv`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, PinvOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     s.pinv_with(&a, &PinvOptions::default().with_rtol(1.0e-12))
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[0.5]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` when the
    /// input is not a (batched) matrix, `ValidationError::InvalidArgument` for a
    /// negative or non-finite tolerance, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype) from the
    /// underlying SVD, or `Error::TensorRuntime` for a backend failure.
    fn pinv_with(&mut self, input: &EagerTensor, options: &PinvOptions) -> Result<EagerTensor>;

    /// Compute a vector, matrix, or tensor norm in this session.
    /// An empty axis list is a no-op and clones the input without dispatch.
    /// `ord = Some(0.0)` counts the entries that are not equal to zero: `NaN`
    /// counts as nonzero, both `+0.0` and `-0.0` count as zero, and `±inf`
    /// count as nonzero (#2359). Axes in `NormOptions::axes` may be negative.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2], vec![3.0_f64, 4.0])?)?;
    ///     s.norm(&a)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[5.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed unsupported-dtype, invalid-axis/order, SVD, or backend errors.
    fn norm(&mut self, input: &EagerTensor) -> Result<EagerTensor>;

    /// Configurable form of [`Self::norm`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, NormOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2], vec![3.0_f64, 4.0])?)?;
    ///     s.norm_with(&a, &NormOptions::default().with_ord(2.0).with_axes(vec![-1]))
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[5.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed unsupported-dtype, invalid-axis/order, SVD, or backend errors.
    fn norm_with(&mut self, input: &EagerTensor, options: &NormOptions) -> Result<EagerTensor>;

    /// Compute complete-pivot LU factors `(P, L, U, Q, parity)` in this session.
    /// Reconstruction uses `A = P^T * L * U * Q`; scalar `parity` is real
    /// (`F32` for `F32`/`C32` inputs and `F64` for `F64`/`C64`).
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?)?;
    ///     s.full_piv_lu(&a)
    /// })?;
    /// let (p, _l, _u, q, parity) = r;
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(parity.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
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

    /// Configurable form of [`Self::full_piv_lu`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, FullPivLuOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?)?;
    ///     s.full_piv_lu_with(&a, &FullPivLuOptions::default())
    /// })?;
    /// let (p, _l, _u, q, parity) = r;
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(parity.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed rank/shape, unsupported-provider, numerical, or output-count errors.
    fn full_piv_lu_with(
        &mut self,
        input: &EagerTensor,
        options: &FullPivLuOptions,
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
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.full_piv_lu_solve(&a, &b)
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed rank/shape, unsupported-provider, singularity, or backend errors.
    fn full_piv_lu_solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor)
        -> Result<EagerTensor>;

    /// Configurable form of [`Self::full_piv_lu_solve`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, FullPivLuSolveOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![2.0_f64])?)?;
    ///     let b = s.constant_from(Tensor::from_vec_col_major([1, 1], vec![4.0_f64])?)?;
    ///     s.full_piv_lu_solve_with(&a, &b, &FullPivLuSolveOptions::default())
    /// })?;
    /// assert_eq!(r.value()?.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed rank/shape, unsupported-provider, singularity, or backend errors.
    fn full_piv_lu_solve_with(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        options: &FullPivLuSolveOptions,
    ) -> Result<EagerTensor>;

    /// Compute column-pivoted rank-revealing QR in this session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?)?;
    ///     s.rank_revealing_qr(&a)
    /// })?;
    /// assert_eq!(r.column_permutation.shape(), &[2]);
    /// assert_eq!(r.rank.value()?.as_slice::<i64>()?, &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch`,
    /// `ValidationError::DTypeMismatch` or `ValidationError::InvalidArgument` for
    /// an invalid rank, dtype or tolerance, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn rank_revealing_qr(
        &mut self,
        input: &EagerTensor,
    ) -> Result<RankRevealingQrResult<EagerTensor>>;

    /// Configurable form of [`Self::rank_revealing_qr`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, RankRevealingQrOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?)?;
    ///     s.rank_revealing_qr_with(&a, &RankRevealingQrOptions::default())
    /// })?;
    /// assert_eq!(r.column_permutation.shape(), &[2]);
    /// assert_eq!(r.rank.value()?.as_slice::<i64>()?, &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch`,
    /// `ValidationError::DTypeMismatch` or `ValidationError::InvalidArgument` for
    /// an invalid rank, dtype or tolerance, `Error::Extension` carrying the linalg failure
    /// (for example `Error::NonConvergence` or an unsupported dtype), or
    /// `Error::TensorRuntime` for a backend failure.
    fn rank_revealing_qr_with(
        &mut self,
        input: &EagerTensor,
        options: &RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<EagerTensor>>;

    /// Initialize compact Householder QR state in this borrowed session.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![1.0_f64, 2.0])?)?;
    ///     s.householder_qr(&a)
    /// })?;
    /// assert!(format!("{r:?}").starts_with("HouseholderQr"));
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed invalid metadata, unsupported executor, or backend errors.
    fn householder_qr(&mut self, input: &EagerTensor) -> Result<crate::HouseholderQr<EagerTensor>>;

    /// Configurable form of [`Self::householder_qr`], taking explicit options.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_linalg::{EagerSessionLinalgExt, HouseholderQrOptions};
    /// let ctx = EagerRuntime::new()?;
    /// let r = ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
    ///     let a = s.constant_from(Tensor::from_vec_col_major([2, 1], vec![1.0_f64, 2.0])?)?;
    ///     s.householder_qr_with(&a, &HouseholderQrOptions::default())
    /// })?;
    /// assert!(format!("{r:?}").starts_with("HouseholderQr"));
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns typed invalid metadata, unsupported executor, or backend errors.
    fn householder_qr_with(
        &mut self,
        input: &EagerTensor,
        options: &HouseholderQrOptions,
    ) -> Result<crate::HouseholderQr<EagerTensor>>;
}

impl EagerSessionLinalgExt for EagerSession<'_> {
    fn cholesky(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        self.cholesky_with(input, &CholeskyOptions::default())
    }

    fn cholesky_with(
        &mut self,
        input: &EagerTensor,
        _options: &CholeskyOptions,
    ) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(self, LinalgOp::Cholesky, &[input])?,
            "cholesky",
        )
    }

    fn svd(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        self.svd_with(input, &SvdOptions::default())
    }

    fn svd_with(
        &mut self,
        input: &EagerTensor,
        options: &SvdOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        crate::options::ensure_full_svd_options("svd", options)?;
        match options.shape {
            SvdShape::Thin => {
                validate_derivative_eps("svd_with", options.derivative_eps)?;
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
            SvdShape::Full => three_outputs(
                apply_linalg_eager_in_session(self, LinalgOp::SvdFull, &[input])?,
                "svd_full",
            ),
        }
    }

    fn qr(&mut self, input: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
        self.qr_with(input, &QrOptions::default())
    }

    fn qr_with(
        &mut self,
        input: &EagerTensor,
        options: &QrOptions,
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

    fn triangular_solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
        self.triangular_solve_with(matrix, rhs, &TriangularSolveOptions::default())
    }

    fn triangular_solve_with(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        options: &TriangularSolveOptions,
    ) -> Result<EagerTensor> {
        one_output(
            apply_linalg_eager_in_session(
                self,
                LinalgOp::TriangularSolve {
                    left_side: options.side == Side::Left,
                    lower: options.uplo == Uplo::Lower,
                    transpose_a: options.transpose == Transpose::Transpose,
                    unit_diagonal: options.diagonal == Diagonal::Unit,
                },
                &[matrix, rhs],
            )?,
            "triangular_solve",
        )
    }

    fn solve(&mut self, matrix: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
        self.solve_with(matrix, rhs, &SolveOptions::default())
    }

    fn solve_with(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        _options: &SolveOptions,
    ) -> Result<EagerTensor> {
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
        self.lstsq_with(matrix, rhs, &LstsqOptions::default())
    }

    fn lstsq_with(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        _options: &LstsqOptions,
    ) -> Result<EagerTensor> {
        eager_composites::lstsq(self, matrix, rhs)
    }

    fn lu(
        &mut self,
        input: &EagerTensor,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)> {
        self.lu_with(input, &LuOptions::default())
    }

    fn lu_with(
        &mut self,
        input: &EagerTensor,
        _options: &LuOptions,
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
        self.slogdet_with(input, &SlogdetOptions::default())
    }

    fn slogdet_with(
        &mut self,
        input: &EagerTensor,
        _options: &SlogdetOptions,
    ) -> Result<(EagerTensor, EagerTensor)> {
        eager_composites::slogdet(self, input)
    }

    fn det(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        self.det_with(input, &DetOptions::default())
    }

    fn det_with(&mut self, input: &EagerTensor, _options: &DetOptions) -> Result<EagerTensor> {
        eager_composites::det(self, input)
    }

    fn inv(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        self.inv_with(input, &InvOptions::default())
    }

    fn inv_with(&mut self, input: &EagerTensor, _options: &InvOptions) -> Result<EagerTensor> {
        eager_composites::inv(self, input)
    }

    fn eigvalsh(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        self.eigvalsh_with(input, &EigvalshOptions::default())
    }

    fn eigvalsh_with(
        &mut self,
        input: &EagerTensor,
        _options: &EigvalshOptions,
    ) -> Result<EagerTensor> {
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
        self.eigvals_with(input, &EigvalsOptions::default())
    }

    fn eigvals_with(
        &mut self,
        input: &EagerTensor,
        _options: &EigvalsOptions,
    ) -> Result<EagerTensor> {
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
        self.eigh_with(input, &EighOptions::default())
    }

    fn eigh_with(
        &mut self,
        input: &EagerTensor,
        options: &EighOptions,
    ) -> Result<(EagerTensor, EagerTensor)> {
        validate_derivative_eps("eigh_with", options.derivative_eps)?;
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
        self.eig_with(input, &EigOptions::default())
    }

    fn eig_with(
        &mut self,
        input: &EagerTensor,
        _options: &EigOptions,
    ) -> Result<(EagerTensor, EagerTensor)> {
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
        self.pinv_with(input, &PinvOptions::default())
    }

    fn pinv_with(&mut self, input: &EagerTensor, options: &PinvOptions) -> Result<EagerTensor> {
        match options.rtol {
            None => eager_composites::pinv(self, input),
            Some(rtol) => eager_composites::pinv_with_rtol(self, input, rtol),
        }
    }

    fn norm(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        self.norm_with(input, &NormOptions::default())
    }

    fn norm_with(&mut self, input: &EagerTensor, options: &NormOptions) -> Result<EagerTensor> {
        let axes = crate::options::resolve_norm_axes(
            "norm",
            options.axes.as_deref(),
            input.shape().len(),
        )?;
        eager_composites::norm(
            self,
            input,
            options.ord,
            axes.as_deref(),
            options.keep_dims == KeepDims::Keep,
        )
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
        self.full_piv_lu_with(input, &FullPivLuOptions::default())
    }

    fn full_piv_lu_with(
        &mut self,
        input: &EagerTensor,
        _options: &FullPivLuOptions,
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
        self.full_piv_lu_solve_with(matrix, rhs, &FullPivLuSolveOptions::default())
    }

    fn full_piv_lu_solve_with(
        &mut self,
        matrix: &EagerTensor,
        rhs: &EagerTensor,
        _options: &FullPivLuSolveOptions,
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
    ) -> Result<RankRevealingQrResult<EagerTensor>> {
        self.rank_revealing_qr_with(input, &RankRevealingQrOptions::default())
    }

    fn rank_revealing_qr_with(
        &mut self,
        input: &EagerTensor,
        options: &RankRevealingQrOptions,
    ) -> Result<RankRevealingQrResult<EagerTensor>> {
        validate_rank_revealing_qr_options("rank_revealing_qr", *options)?;
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
        self.householder_qr_with(input, &HouseholderQrOptions::default())
    }

    fn householder_qr_with(
        &mut self,
        input: &EagerTensor,
        _options: &HouseholderQrOptions,
    ) -> Result<crate::HouseholderQr<EagerTensor>> {
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
    engine: &mut dyn ExecutionEngine,
    op: LinalgOp,
    inputs: &[&EagerTensor],
) -> Result<Vec<EagerTensor>> {
    let op = Arc::new(LinalgExtensionOp::new(op));
    apply_eager_with_targeted_extension_session(engine, op, inputs, eager_extension_module)
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
/// let x = a.solve(&mut tenferro_cpu::CpuEngine::serial(), &b)?;
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
pub fn solve(
    engine: &mut dyn ExecutionEngine,
    a: &EagerTensor,
    b: &EagerTensor,
) -> Result<EagerTensor> {
    if !a.tracks_grad() && !b.tracks_grad() {
        return one_output(
            apply_linalg_eager(engine, LinalgOp::Solve, &[a, b])?,
            "solve",
        );
    }
    validate_tracked_solve_inputs(a, b)?;
    factor_solve_output(apply_linalg_eager(
        engine,
        LinalgOp::LuFactorSolve,
        &[a, b],
    )?)
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
