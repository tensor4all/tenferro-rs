//! Injectable CPU linear-algebra kernels.
//!
//! A [`CpuLinalgKernels`] implementation replaces tenferro-linalg's built-in
//! faer/LAPACK kernels op by op on one CPU backend. Install it with
//! [`install_linalg_kernels`] on a [`CpuProviderBundleBuilder`]; the CPU
//! session then asks it first for every primitive it dispatches (Cholesky,
//! triangular solve, LU and full-pivot LU, solve, SVD, QR and rank-revealing
//! QR, eigh, eig, and their value-only forms, including the `_read`
//! variants). A method returns [`CpuLinalgOutcome::Unsupported`], its default,
//! before producing anything, and the built-in kernel runs instead. The
//! Householder family, `lu_factor` and prepared LU solves, and `_into`
//! outputs always use the built-in kernels. Composites (`det`, `slogdet`, `inv`, `lstsq`, `pinv`,
//! norms) are built from the primitives and follow them.
//!
//! A kernel runs inside the session's entered [`CpuExecutionContext`] on host
//! operands and must return exactly what the built-in kernel returns for the
//! same input: the same outputs in the same order, shapes, dtypes, pivot and
//! ordering conventions, and batch handling (trailing batch axes). It may run
//! in parallel only in [`tenferro_cpu::ParallelMode::Inner`], within
//! `thread_budget()`, on the context's pool
//! ([`CpuExecutionContext::rayon_pool`]).
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//! use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuProviderBundle};
//! use tenferro_linalg::cpu_kernels::{install_linalg_kernels, CpuLinalgKernels};
//!
//! /// Declines everything, so the built-in kernels run.
//! #[derive(Debug)]
//! struct Nothing;
//! impl CpuLinalgKernels for Nothing {}
//!
//! let builder = CpuProviderBundle::builder(CpuBackendKind::default_compiled());
//! let bundle = install_linalg_kernels(builder, Arc::new(Nothing)).build()?;
//! let backend = CpuBackend::new().with_provider_bundle(bundle)?;
//! # let _ = backend;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::sync::Arc;

use tenferro_cpu::provider::CpuProviderUnsupported;
use tenferro_cpu::{CpuExecutionContext, CpuProviderBundleBuilder};
use tenferro_tensor::{Tensor, TensorView};

use crate::RankRevealingQrOptions;

/// Result of a [`CpuLinalgKernels`] call.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuProviderUnsupported;
/// use tenferro_linalg::cpu_kernels::CpuLinalgOutcome;
/// let o: CpuLinalgOutcome<u8> = CpuLinalgOutcome::Unsupported(CpuProviderUnsupported::RuntimeUnavailable);
/// assert!(matches!(o, CpuLinalgOutcome::Unsupported(_)));
/// ```
#[derive(Debug)]
pub enum CpuLinalgOutcome<T> {
    /// The kernel produced the outputs.
    Executed(T),
    /// The kernel declined; nothing was produced and the built-in kernel
    /// runs.
    Unsupported(CpuProviderUnsupported),
}

/// Flags of a triangular solve (`op(A) X = B` or `X op(A) = B`).
///
/// # Examples
///
/// ```
/// let o = tenferro_linalg::cpu_kernels::TriangularSolveOptions {
///     left_side: true, lower: true, transpose_a: false, unit_diagonal: false,
/// };
/// assert!(o.lower);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriangularSolveOptions {
    /// Solve `op(A) X = B` (else `X op(A) = B`).
    pub left_side: bool,
    /// `A` is lower triangular.
    pub lower: bool,
    /// Use `A^T`.
    pub transpose_a: bool,
    /// The diagonal of `A` is taken as ones.
    pub unit_diagonal: bool,
}

fn declined<T>() -> tenferro_tensor::Result<CpuLinalgOutcome<T>> {
    Ok(CpuLinalgOutcome::Unsupported(
        CpuProviderUnsupported::RuntimeUnavailable,
    ))
}

/// Replacement CPU linear-algebra kernels. Every method defaults to
/// [`CpuLinalgOutcome::Unsupported`]; implement the ones the provider
/// handles. See the [module documentation](self) for the contract.
///
/// # Errors
///
/// Methods return errors only for failures after committing to the
/// operation (a runtime or storage failure); declining is
/// [`CpuLinalgOutcome::Unsupported`].
///
/// # Examples
///
/// ```
/// use tenferro_linalg::cpu_kernels::CpuLinalgKernels;
/// #[derive(Debug)]
/// struct Nothing;
/// impl CpuLinalgKernels for Nothing {}
/// let _: &dyn CpuLinalgKernels = &Nothing;
/// ```
#[allow(unused_variables)]
pub trait CpuLinalgKernels: std::fmt::Debug + Send + Sync + 'static {
    /// Cholesky factor, as the built-in `cholesky`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn cholesky(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        declined()
    }

    /// Triangular solve, as the built-in `triangular_solve`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn triangular_solve(
        &self,
        context: &CpuExecutionContext<'_>,
        a: TensorView<'_>,
        b: TensorView<'_>,
        options: TriangularSolveOptions,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        declined()
    }

    /// Partial-pivot LU, as the built-in `lu`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn lu(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        declined()
    }

    /// Full-pivot LU, as the built-in `full_piv_lu`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn full_piv_lu(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        declined()
    }

    /// Linear solve `A X = B`, as the built-in `solve`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn solve(
        &self,
        context: &CpuExecutionContext<'_>,
        a: TensorView<'_>,
        b: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        declined()
    }

    /// Thin SVD, as the built-in `svd`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn svd(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        declined()
    }

    /// Full SVD, as the built-in `svd_full`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn svd_full(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        declined()
    }

    /// Singular values, as the built-in `svd_values`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn svd_values(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        declined()
    }

    /// Thin QR, as the built-in `qr`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn qr(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        declined()
    }

    /// Column-pivoted QR, as the built-in `rank_revealing_qr`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn rank_revealing_qr(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
        options: RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        declined()
    }

    /// Hermitian eigendecomposition, as the built-in `eigh`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn eigh(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        declined()
    }

    /// Hermitian eigenvalues, as the built-in `eigh_values`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn eigh_values(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        declined()
    }

    /// General eigendecomposition, as the built-in `eig`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn eig(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Vec<Tensor>>> {
        declined()
    }

    /// General eigenvalues, as the built-in `eig_values`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendFailure`] or
    /// [`tenferro_tensor::Error::BackendSource`] when the kernel's runtime or a
    /// host buffer fails after it committed to the operation.
    fn eig_values(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> tenferro_tensor::Result<CpuLinalgOutcome<Tensor>> {
        declined()
    }
}

/// The provider-bundle extension that carries a [`CpuLinalgKernels`].
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tenferro_linalg::cpu_kernels::{CpuLinalgKernels, CpuLinalgKernelsSlot};
/// #[derive(Debug)]
/// struct Nothing;
/// impl CpuLinalgKernels for Nothing {}
/// let slot = CpuLinalgKernelsSlot(Arc::new(Nothing));
/// let _ = &slot.0;
/// ```
#[derive(Debug)]
pub struct CpuLinalgKernelsSlot(pub Arc<dyn CpuLinalgKernels>);

/// Install `kernels` on a provider bundle under construction.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tenferro_cpu::{CpuBackendKind, CpuProviderBundle};
/// use tenferro_linalg::cpu_kernels::{install_linalg_kernels, CpuLinalgKernels, CpuLinalgKernelsSlot};
/// #[derive(Debug)]
/// struct Nothing;
/// impl CpuLinalgKernels for Nothing {}
/// let bundle = install_linalg_kernels(
///     CpuProviderBundle::builder(CpuBackendKind::default_compiled()),
///     Arc::new(Nothing),
/// )
/// .build()?;
/// assert!(bundle.extension::<CpuLinalgKernelsSlot>().is_some());
/// # Ok::<(), tenferro_cpu::CpuProviderBundleBuildError>(())
/// ```
pub fn install_linalg_kernels(
    builder: CpuProviderBundleBuilder,
    kernels: Arc<dyn CpuLinalgKernels>,
) -> CpuProviderBundleBuilder {
    builder.extension(Arc::new(CpuLinalgKernelsSlot(kernels)))
}
