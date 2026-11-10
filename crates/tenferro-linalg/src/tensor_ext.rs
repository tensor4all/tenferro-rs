//! Receiver-first concrete linear algebra surfaces.
//!
//! Owned tensors, borrowed [`TensorRead`]s and typed tensors or views all use
//! [`TensorLinalgExt`]. All methods dispatch internally to the built-in
//! CPU/CUDA execution sessions through an erased `&mut dyn Engine`
//! (issue #1680 Phase 3); third-party [`LinalgBackend`] implementations
//! remain supported through the SPI trait, but the concrete op path is
//! built-in-session only.

use num_complex::{Complex32, Complex64};
use tenferro_cpu::linalg_interop::PoolScalar;
use tenferro_cpu::with_cpu_engine;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::raw::with_cuda_engine;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, Engine, Tensor, TensorRead, TensorScalar, TensorWrite,
};

use crate::extension::{EighOptions, QrOptions, SvdOptions};
use crate::input::LinalgInput;
use crate::options::{ensure_full_svd_options, resolve_norm_axes};
use crate::traced::real_values_dtype;
use crate::validation::validate_square_matrix;
use crate::{
    CholeskyOptions, DetOptions, Diagonal, Eig, EigOptions, Eigh, EigvalsOptions, EigvalshOptions,
    FullPivLu, FullPivLuOptions, FullPivLuSolveOptions, HouseholderQrOptions, InvOptions, KeepDims,
    Lu, LuOptions, NormOptions, PinvOptions, Qr, Side, Slogdet, SlogdetOptions, SolveOptions, Svd,
    SvdShape, SvdvalsOptions, Transpose, TriangularSolveOptions, Uplo,
};
use crate::{LinalgBackend, RankRevealingQrOptions, RankRevealingQrResult};

type Res<T> = tenferro_tensor::Result<T>;

/// Scalar types supported by statically typed linear algebra methods: `f32`, `f64`,
/// `Complex32` and `Complex64`.
///
/// The trait is sealed: its supertrait lives in a private module, so no other type can implement
/// it. It names the two output scalars a factorization of `Self` can return.
///
/// | `Self` | [`RealScalar`](LinalgScalar::RealScalar) | [`Complex`](LinalgScalar::Complex) |
/// | --- | --- | --- |
/// | `f32` | `f32` | `Complex32` |
/// | `f64` | `f64` | `Complex64` |
/// | `Complex32` | `f32` | `Complex32` |
/// | `Complex64` | `f64` | `Complex64` |
///
/// # Examples
///
/// ```rust
/// use num_complex::Complex64;
/// use tenferro_linalg::LinalgScalar;
///
/// fn accepts_linalg_scalar<T: LinalgScalar>() {}
/// accepts_linalg_scalar::<f64>();
///
/// fn spectrum_scalar<T: LinalgScalar>() -> &'static str {
///     std::any::type_name::<T::RealScalar>()
/// }
/// assert_eq!(spectrum_scalar::<Complex64>(), "f64");
/// ```
pub trait LinalgScalar: TensorScalar + private::Sealed {
    /// Complex counterpart used by general eigendecomposition.
    type Complex: TensorScalar + PoolScalar + Default;
    /// Real scalar of singular values and Hermitian eigenvalues.
    type RealScalar: TensorScalar + PoolScalar + Default;
}

mod private {
    /// The sealing supertrait of [`LinalgScalar`](super::LinalgScalar).
    ///
    /// `pub` inside a private module: nameable as a supertrait, unreachable outside the crate.
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

impl LinalgScalar for f32 {
    type Complex = Complex32;
    type RealScalar = f32;
}
impl LinalgScalar for f64 {
    type Complex = Complex64;
    type RealScalar = f64;
}
impl LinalgScalar for Complex32 {
    type Complex = Complex32;
    type RealScalar = f32;
}
impl LinalgScalar for Complex64 {
    type Complex = Complex64;
    type RealScalar = f64;
}

/// Linear algebra methods for every concrete input kind.
///
/// Implemented for owned [`Tensor`], borrowed [`TensorRead`] and typed
/// [`TypedTensorView`](tenferro_tensor::TypedTensorView) through the blanket impl over
/// [`LinalgInput`]. Each operation `op(exec)` is the short form of
/// `op_with(&OpOptions::default(), exec)`. Outputs follow the input: dtype-erased input gives
/// [`Tensor`], a typed view gives typed tensors (`Self::Real` for singular values, norms and
/// parity, `Self::Complex` for general eigenvalues). Every method dispatches to the built-in
/// CPU/CUDA execution sessions through an erased `&mut dyn Engine`.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_linalg::{SvdOptions, SvdShape, TensorLinalgExt};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
/// let mut host = tenferro_cpu::CpuEngine::serial();
/// let singular_values = a.svd(&mut host)?.s;
/// assert_eq!(singular_values.as_slice::<f64>()?, &[4.0, 2.0]);
/// let u = a.svd_with(&SvdOptions::default().with_shape(SvdShape::Full), &mut host)?.u;
/// assert_eq!(u.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorLinalgExt: LinalgInput {
    /// Compute the singular value decomposition `(U, S, Vt)`.
    ///
    /// Short form of [`Self::svd_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::svd_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.svd(&mut host)?;
    /// let s = r.s;
    /// assert_eq!(s.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd(&self, exec: &mut dyn Engine) -> Res<Svd<Self::Out, Self::Real>> {
        self.svd_with(&SvdOptions::default(), exec)
    }

    /// Compute the singular value decomposition `(U, S, Vt)`.
    ///
    /// `SvdShape::Thin` is the thin decomposition and honours the gauge, driver and
    /// `derivative_eps` of the options. `SvdShape::Full` returns `U` shaped `m x m` and `Vt`
    /// shaped `n x n`, so the trailing `Vt` rows span the input's right nullspace and the
    /// trailing `U` columns span its left nullspace; it combines only with the default gauge,
    /// driver and `derivative_eps`, and the thin decomposition is never substituted for it.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected backend does not support the operation,
    /// plus validation, numerical, or output-contract errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{SvdOptions, SvdShape, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.svd_with(&SvdOptions::default().with_shape(SvdShape::Full), &mut host)?;
    /// let (u, s, vt) = (r.u, r.s, r.vh);
    /// assert_eq!(u.shape(), &[1, 1]);
    /// assert_eq!(s.shape(), &[1]);
    /// assert_eq!(vt.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd_with(
        &self,
        options: &SvdOptions,
        exec: &mut dyn Engine,
    ) -> Res<Svd<Self::Out, Self::Real>> {
        ensure_full_svd_options("svd", options)?;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "svd", |backend| {
            three(
                match options.shape {
                    SvdShape::Thin => backend.svd_with_options_read(a, *options)?,
                    SvdShape::Full => backend.svd_full_read(a)?,
                },
                "svd",
            )
        })?;
        Ok(Svd {
            u: Self::linalg_out(outs.0)?,
            s: Self::linalg_real(outs.1)?,
            vh: Self::linalg_out(outs.2)?,
        })
    }

    /// Compute singular values without allocating singular-vector outputs.
    ///
    /// Short form of [`Self::svdvals_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::svdvals_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.svdvals(&mut host)?;
    /// assert_eq!(r.as_slice::<f64>()?, &[4.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svdvals(&self, exec: &mut dyn Engine) -> Res<Self::Real> {
        self.svdvals_with(&SvdvalsOptions::default(), exec)
    }

    /// Compute singular values without allocating singular-vector outputs.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected backend has no values-only
    /// capability, rather than silently computing a full decomposition and discarding its vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{SvdvalsOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.svdvals_with(&SvdvalsOptions::default(), &mut host)?;
    /// assert_eq!(r.as_slice::<f64>()?, &[4.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svdvals_with(&self, options: &SvdvalsOptions, exec: &mut dyn Engine) -> Res<Self::Real> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "svdvals", |backend| backend.svd_values_read(a))?;
        Self::linalg_real(outs)
    }

    /// Compute the thin QR decomposition `(Q, R)`.
    ///
    /// Short form of [`Self::qr_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::qr_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.qr(&mut host)?;
    /// let (q, r) = (r.q, r.r);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr(&self, exec: &mut dyn Engine) -> Res<Qr<Self::Out>> {
        self.qr_with(&QrOptions::default(), exec)
    }

    /// Compute the thin QR decomposition `(Q, R)`.
    ///
    /// The CUDA provider does not read the cuSOLVER `geqrf`/`orgqr` status flags, so forward QR
    /// reports neither a vendor parameter error nor a rank deficiency: shapes and dtypes are
    /// validated up front, and a rank-deficient input still yields its valid factors.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected backend does not support the operation,
    /// plus validation, numerical, or output-contract errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{QrOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.qr_with(&QrOptions::default(), &mut host)?;
    /// let (q, r) = (r.q, r.r);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr_with(&self, options: &QrOptions, exec: &mut dyn Engine) -> Res<Qr<Self::Out>> {
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "qr", |backend| {
            two(backend.qr_with_options_read(a, *options)?, "qr")
        })?;
        Ok(Qr {
            q: Self::linalg_out(outs.0)?,
            r: Self::linalg_out(outs.1)?,
        })
    }

    /// Initialize opaque compact Householder QR state.
    ///
    /// Short form of [`Self::householder_qr_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::householder_qr_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.householder_qr(&mut host)?;
    /// assert!(format!("{r:?}").starts_with("HouseholderQr"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn householder_qr(&self, exec: &mut dyn Engine) -> Res<crate::HouseholderQr<Tensor>> {
        self.householder_qr_with(&HouseholderQrOptions::default(), exec)
    }

    /// Initialize opaque compact Householder QR state.
    ///
    /// The state is always a `HouseholderQr<Tensor>`, also for a typed input.
    ///
    /// # Errors
    ///
    /// Returns validation errors for non-matrix or unsupported-dtype input and typed provider errors
    /// when compact QR is unavailable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{HouseholderQrOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.householder_qr_with(&HouseholderQrOptions::default(), &mut host)?;
    /// assert!(format!("{r:?}").starts_with("HouseholderQr"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn householder_qr_with(
        &self,
        options: &HouseholderQrOptions,
        exec: &mut dyn Engine,
    ) -> Res<crate::HouseholderQr<Tensor>> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "householder_qr", |backend| {
            match a.as_tensor() {
                Some(t) => backend.householder_qr(t),
                None => {
                    let shape = a.shape().to_vec();
                    let compact = backend.reshape_read(a, &shape)?;
                    backend.householder_qr(&compact)
                }
            }
            .map(crate::HouseholderQr::from_backend)
        })?;
        Ok(outs)
    }

    /// Compute column-pivoted rank-revealing QR with fixed-shape tensor metadata.
    ///
    /// Short form of [`Self::rank_revealing_qr_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::rank_revealing_qr_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.rank_revealing_qr(&mut host)?;
    /// assert_eq!(r.rank.as_slice::<i64>()?, &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn rank_revealing_qr(
        &self,
        exec: &mut dyn Engine,
    ) -> Res<RankRevealingQrResult<Self::Out, Self::Index>> {
        self.rank_revealing_qr_with(&RankRevealingQrOptions::default(), exec)
    }

    /// Compute column-pivoted rank-revealing QR with fixed-shape tensor metadata.
    ///
    /// # Errors
    ///
    /// Returns validation errors for rank, dtype, or invalid tolerances; numerical failure for
    /// non-finite input or diagonal values; and explicit unsupported/backend errors when the selected
    /// provider cannot execute RRQR.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{RankRevealingQrOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.rank_revealing_qr_with(&RankRevealingQrOptions::default().with_rtol(1e-12), &mut host)?;
    /// assert_eq!(r.column_permutation.shape(), &[2]);
    /// assert_eq!(r.rank.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn rank_revealing_qr_with(
        &self,
        options: &RankRevealingQrOptions,
        exec: &mut dyn Engine,
    ) -> Res<RankRevealingQrResult<Self::Out, Self::Index>> {
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "rank_revealing_qr", |backend| {
            rank_revealing_qr_result(backend.rank_revealing_qr_read(a, *options)?)
        })?;
        Ok(RankRevealingQrResult {
            q: Self::linalg_out(outs.q)?,
            r: Self::linalg_out(outs.r)?,
            column_permutation: Self::linalg_index(outs.column_permutation)?,
            rank: Self::linalg_index(outs.rank)?,
        })
    }

    /// Compute the partial-pivot LU decomposition `(P, L, U, parity)`.
    ///
    /// Short form of [`Self::lu_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::lu_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.lu(&mut host)?;
    /// let (l, u, parity) = (r.l, r.u, r.parity);
    /// assert_eq!(l.shape(), &[2, 2]);
    /// assert_eq!(u.shape(), &[2, 2]);
    /// assert_eq!(parity.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn lu(&self, exec: &mut dyn Engine) -> Res<Lu<Self::Out, Self::Real>> {
        self.lu_with(&LuOptions::default(), exec)
    }

    /// Compute the partial-pivot LU decomposition `(P, L, U, parity)`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected backend does not support the operation,
    /// plus validation, numerical, or output-contract errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{LuOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.lu_with(&LuOptions::default(), &mut host)?;
    /// let (l, u) = (r.l, r.u);
    /// assert_eq!(l.shape(), &[2, 2]);
    /// assert_eq!(u.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn lu_with(
        &self,
        options: &LuOptions,
        exec: &mut dyn Engine,
    ) -> Res<Lu<Self::Out, Self::Real>> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "lu", |backend| four(backend.lu_read(a)?, "lu"))?;
        Ok(Lu {
            p: Self::linalg_out(outs.0)?,
            l: Self::linalg_out(outs.1)?,
            u: Self::linalg_out(outs.2)?,
            parity: Self::linalg_real(outs.3)?,
        })
    }

    /// Compute the complete-pivot LU decomposition `(P, L, U, Q, parity)`.
    ///
    /// Short form of [`Self::full_piv_lu_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::full_piv_lu_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.full_piv_lu(&mut host)?;
    /// let (p, q, parity) = (r.p, r.q, r.parity);
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// assert_eq!(parity.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu(&self, exec: &mut dyn Engine) -> Res<FullPivLu<Self::Out, Self::Real>> {
        self.full_piv_lu_with(&FullPivLuOptions::default(), exec)
    }

    /// Compute the complete-pivot LU decomposition `(P, L, U, Q, parity)`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected backend does not support the operation,
    /// plus validation, numerical, or output-contract errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{FullPivLuOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.full_piv_lu_with(&FullPivLuOptions::default(), &mut host)?;
    /// let (p, q) = (r.p, r.q);
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(q.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu_with(
        &self,
        options: &FullPivLuOptions,
        exec: &mut dyn Engine,
    ) -> Res<FullPivLu<Self::Out, Self::Real>> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "full_piv_lu", |backend| {
            five(backend.full_piv_lu_read(a)?, "full_piv_lu")
        })?;
        Ok(FullPivLu {
            p: Self::linalg_out(outs.0)?,
            l: Self::linalg_out(outs.1)?,
            u: Self::linalg_out(outs.2)?,
            q: Self::linalg_out(outs.3)?,
            parity: Self::linalg_real(outs.4)?,
        })
    }

    /// Solve `A X = B` through the complete-pivot LU decomposition.
    ///
    /// Short form of [`Self::full_piv_lu_solve_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::full_piv_lu_solve_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.full_piv_lu_solve(&b, &mut host)?;
    /// assert_eq!(r.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu_solve<B: LinalgInput>(&self, b: &B, exec: &mut dyn Engine) -> Res<Self::Out> {
        self.full_piv_lu_solve_with(b, &FullPivLuSolveOptions::default(), exec)
    }

    /// Solve `A X = B` through the complete-pivot LU decomposition.
    ///
    /// A rank-deficient system (a pivot at or below `eps * max|pivot|`) is rejected by the backend solve.
    ///
    /// # Errors
    ///
    /// Returns validation errors for incompatible inputs or a backend/singular-system error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{FullPivLuSolveOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.full_piv_lu_solve_with(&b, &FullPivLuSolveOptions::default(), &mut host)?;
    /// assert_eq!(r.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu_solve_with<B: LinalgInput>(
        &self,
        b: &B,
        options: &FullPivLuSolveOptions,
        exec: &mut dyn Engine,
    ) -> Res<Self::Out> {
        let _ = options;
        let a = self.linalg_read();
        let b = b.linalg_read();
        let outs = with_linalg_backend(exec, "full_piv_lu_solve", |backend| {
            full_piv_lu_solve_impl(a, b, backend)
        })?;
        Self::linalg_out(outs)
    }

    /// Solve `A X = B`.
    ///
    /// Short form of [`Self::solve_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::solve_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.solve(&b, &mut host)?;
    /// assert_eq!(r.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn solve<B: LinalgInput>(&self, b: &B, exec: &mut dyn Engine) -> Res<Self::Out> {
        self.solve_with(b, &SolveOptions::default(), exec)
    }

    /// Solve `A X = B`.
    ///
    /// The CUDA provider does not inspect the cuSOLVER `getrf` status flag on the forward solve
    /// path, so it performs no per-call non-singularity check and issues no host round trip. An
    /// exactly singular input is left to the caller's verification, as in `torch.linalg.solve`. CPU
    /// providers may still report their own exactly singular factor as `Error::Singular`.
    ///
    /// # Errors
    ///
    /// Returns validation errors for incompatible inputs or a backend/singular-system error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{SolveOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.solve_with(&b, &SolveOptions::default(), &mut host)?;
    /// assert_eq!(r.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn solve_with<B: LinalgInput>(
        &self,
        b: &B,
        options: &SolveOptions,
        exec: &mut dyn Engine,
    ) -> Res<Self::Out> {
        let _ = options;
        let a = self.linalg_read();
        let b = b.linalg_read();
        let outs = with_linalg_backend(exec, "solve", |backend| backend.solve_read(a, b))?;
        Self::linalg_out(outs)
    }

    /// Solve `A X = B` into a caller-owned destination without allocating the result at the
    /// public API boundary.
    ///
    /// Backends with a native path may write directly into a compatible target; otherwise the
    /// ordinary solve plus copy behaviour applies. The CUDA singularity note of
    /// [`Self::solve_with`] applies here too.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_tensor_core::ShapeMismatch` or
    /// `tenferro_tensor_core::ValidationError::DTypeMismatch` for incompatible destination
    /// metadata, `tenferro_tensor_core::ValidationError::InvalidArgument` for aliasing or
    /// placement violations, `Error::Unsupported` when the extension implementation or provider
    /// is unavailable, and `Error::Singular` for a singular system.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{SolveOptions, TensorLinalgExt};
    /// # use tenferro_tensor::{Tensor, TensorWrite};
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0])?;
    /// # let mut out = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// a.solve_into(&b, &SolveOptions::default(), &mut host, TensorWrite::from_tensor(&mut out))?;
    /// assert_eq!(out.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn solve_into<B: LinalgInput>(
        &self,
        b: &B,
        options: &SolveOptions,
        exec: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Res<()> {
        let _ = options;
        let a = self.linalg_read();
        let b = b.linalg_read();
        with_linalg_backend(exec, "solve_into", |backend| {
            backend.solve_read_into(a, b, out)
        })
    }

    /// Compute the lower Cholesky factor.
    ///
    /// Short form of [`Self::cholesky_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::cholesky_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.cholesky(&mut host)?;
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn cholesky(&self, exec: &mut dyn Engine) -> Res<Self::Out> {
        self.cholesky_with(&CholeskyOptions::default(), exec)
    }

    /// Compute the lower Cholesky factor.
    ///
    /// # Errors
    ///
    /// Returns validation errors for a non-square input, or backend/positive-definiteness errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{CholeskyOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.cholesky_with(&CholeskyOptions::default(), &mut host)?;
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn cholesky_with(&self, options: &CholeskyOptions, exec: &mut dyn Engine) -> Res<Self::Out> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "cholesky", |backend| backend.cholesky_read(a))?;
        Self::linalg_out(outs)
    }

    /// Compute the Hermitian eigendecomposition `(values, vectors)`.
    ///
    /// Short form of [`Self::eigh_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::eigh_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.eigh(&mut host)?;
    /// let (values, vectors) = (r.values, r.vectors);
    /// assert_eq!(values.as_slice::<f64>()?, &[1.0, 3.0]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh(&self, exec: &mut dyn Engine) -> Res<Eigh<Self::Out, Self::Real>> {
        self.eigh_with(&EighOptions::default(), exec)
    }

    /// Compute the Hermitian eigendecomposition `(values, vectors)`.
    ///
    /// # Errors
    ///
    /// Returns validation, unsupported-backend, convergence, or output-contract errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{EighOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.eigh_with(&EighOptions::default(), &mut host)?;
    /// let (values, vectors) = (r.values, r.vectors);
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh_with(
        &self,
        options: &EighOptions,
        exec: &mut dyn Engine,
    ) -> Res<Eigh<Self::Out, Self::Real>> {
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "eigh", |backend| {
            two(backend.eigh_with_options_read(a, *options)?, "eigh")
        })?;
        Ok(Eigh {
            values: Self::linalg_real(outs.0)?,
            vectors: Self::linalg_out(outs.1)?,
        })
    }

    /// Compute the general eigendecomposition `(values, vectors)`.
    ///
    /// Short form of [`Self::eig_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::eig_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.eig(&mut host)?;
    /// let (values, vectors) = (r.values, r.vectors);
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eig(&self, exec: &mut dyn Engine) -> Res<Eig<Self::Complex>> {
        self.eig_with(&EigOptions::default(), exec)
    }

    /// Compute the general eigendecomposition `(values, vectors)`.
    ///
    /// Eigenvectors come back with unit 2-norm columns, whichever provider computed them; a column
    /// the provider left non-finite is reported as a numerical failure instead of being returned as
    /// `NaN` (#2148).
    ///
    /// # Errors
    ///
    /// Returns validation, unsupported-backend, convergence, or output-contract errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{EigOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.eig_with(&EigOptions::default(), &mut host)?;
    /// let (values, vectors) = (r.values, r.vectors);
    /// assert_eq!(values.shape(), &[2]);
    /// assert_eq!(vectors.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eig_with(&self, options: &EigOptions, exec: &mut dyn Engine) -> Res<Eig<Self::Complex>> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "eig", |backend| two(backend.eig_read(a)?, "eig"))?;
        Ok(Eig {
            values: Self::linalg_complex(outs.0)?,
            vectors: Self::linalg_complex(outs.1)?,
        })
    }

    /// Compute the eigenvalues of a general matrix.
    ///
    /// Short form of [`Self::eigvals_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::eigvals_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.eigvals(&mut host)?;
    /// assert_eq!(r.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvals(&self, exec: &mut dyn Engine) -> Res<Self::Complex> {
        self.eigvals_with(&EigvalsOptions::default(), exec)
    }

    /// Compute the eigenvalues of a general matrix.
    ///
    /// # Errors
    ///
    /// Returns the validation, backend, convergence, or contract errors of [`Self::eig`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{EigvalsOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.eigvals_with(&EigvalsOptions::default(), &mut host)?;
    /// assert_eq!(r.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvals_with(&self, options: &EigvalsOptions, exec: &mut dyn Engine) -> Res<Self::Complex> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "eigvals", |backend| backend.eig_values_read(a))?;
        Self::linalg_complex(outs)
    }

    /// Compute the eigenvalues of a Hermitian matrix.
    ///
    /// Short form of [`Self::eigvalsh_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::eigvalsh_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.eigvalsh(&mut host)?;
    /// assert_eq!(r.as_slice::<f64>()?, &[1.0, 3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvalsh(&self, exec: &mut dyn Engine) -> Res<Self::Real> {
        self.eigvalsh_with(&EigvalshOptions::default(), exec)
    }

    /// Compute the eigenvalues of a Hermitian matrix.
    ///
    /// # Errors
    ///
    /// Returns the validation, backend, convergence, or contract errors of [`Self::eigh`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{EigvalshOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.eigvalsh_with(&EigvalshOptions::default(), &mut host)?;
    /// assert_eq!(r.as_slice::<f64>()?, &[1.0, 3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigvalsh_with(&self, options: &EigvalshOptions, exec: &mut dyn Engine) -> Res<Self::Real> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "eigvalsh", |backend| backend.eigh_values_read(a))?;
        Self::linalg_real(outs)
    }

    /// Solve a triangular system `op(A) X = B` or `X op(A) = B`.
    ///
    /// Short form of [`Self::triangular_solve_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::triangular_solve_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![1, 2], vec![4.0_f64, 9.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// // The default options solve `X A = B` with the upper triangle of `A`.
    /// let r = a.triangular_solve(&b, &mut host)?;
    /// assert_eq!(r.shape(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn triangular_solve<B: LinalgInput>(&self, b: &B, exec: &mut dyn Engine) -> Res<Self::Out> {
        self.triangular_solve_with(b, &TriangularSolveOptions::default(), exec)
    }

    /// Solve a triangular system `op(A) X = B` or `X op(A) = B`.
    ///
    /// The side, triangle, transposition and diagonal handling come from the options.
    ///
    /// # Errors
    ///
    /// Returns validation errors for incompatible inputs/options or backend/singular errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{Side, TriangularSolveOptions, Uplo, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0])?;
    /// # let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.triangular_solve_with(&b, &TriangularSolveOptions::default().with_side(Side::Left).with_uplo(Uplo::Upper), &mut host)?;
    /// assert_eq!(r.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn triangular_solve_with<B: LinalgInput>(
        &self,
        b: &B,
        options: &TriangularSolveOptions,
        exec: &mut dyn Engine,
    ) -> Res<Self::Out> {
        let a = self.linalg_read();
        let b = b.linalg_read();
        let outs = with_linalg_backend(exec, "triangular_solve", |backend| {
            backend.triangular_solve_read(
                a,
                b,
                options.side == Side::Left,
                options.uplo == Uplo::Lower,
                options.transpose == Transpose::Transpose,
                options.diagonal == Diagonal::Unit,
            )
        })?;
        Self::linalg_out(outs)
    }

    /// Compute the sign and the log-absolute value of the determinant.
    ///
    /// Short form of [`Self::slogdet_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::slogdet_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.slogdet(&mut host)?;
    /// let (sign, logabsdet) = (r.sign, r.logabsdet);
    /// assert_eq!(sign.as_slice::<f64>()?, &[1.0]);
    /// assert_eq!(logabsdet.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn slogdet(&self, exec: &mut dyn Engine) -> Res<Slogdet<Self::Out, Self::Real>> {
        self.slogdet_with(&SlogdetOptions::default(), exec)
    }

    /// Compute the sign and the log-absolute value of the determinant.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the selected backend does not support the operation,
    /// plus validation, numerical, or output-contract errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{SlogdetOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.slogdet_with(&SlogdetOptions::default(), &mut host)?;
    /// let sign = r.sign;
    /// assert_eq!(sign.as_slice::<f64>()?, &[1.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn slogdet_with(
        &self,
        options: &SlogdetOptions,
        exec: &mut dyn Engine,
    ) -> Res<Slogdet<Self::Out, Self::Real>> {
        let _ = options;
        let a = self.linalg_read();
        let outs = slogdet_tensors(a, "slogdet", exec)?;
        Ok(Slogdet {
            sign: Self::linalg_out(outs.0)?,
            logabsdet: Self::linalg_real(outs.1)?,
        })
    }

    /// Compute the determinant.
    ///
    /// Short form of [`Self::det_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::det_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.det(&mut host)?;
    /// assert!((r.as_slice::<f64>()?[0] - 8.0).abs() < 1.0e-12);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn det(&self, exec: &mut dyn Engine) -> Res<Self::Out> {
        self.det_with(&DetOptions::default(), exec)
    }

    /// Compute the determinant.
    ///
    /// # Errors
    ///
    /// Returns the validation, backend, numerical, or contract errors of [`Self::slogdet`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{DetOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.det_with(&DetOptions::default(), &mut host)?;
    /// assert!((r.as_slice::<f64>()?[0] - 8.0).abs() < 1.0e-12);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn det_with(&self, options: &DetOptions, exec: &mut dyn Engine) -> Res<Self::Out> {
        let _ = options;
        let a = self.linalg_read();
        let outs = det_tensors(a, exec)?;
        Self::linalg_out(outs)
    }

    /// Compute the matrix inverse.
    ///
    /// Short form of [`Self::inv_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::inv_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.inv(&mut host)?;
    /// assert_eq!(r.as_slice::<f64>()?, &[0.5, 0.0, 0.0, 0.25]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn inv(&self, exec: &mut dyn Engine) -> Res<Self::Out> {
        self.inv_with(&InvOptions::default(), exec)
    }

    /// Compute the matrix inverse.
    ///
    /// # Errors
    ///
    /// Returns validation, unsupported-backend, or singular-solve numerical errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{InvOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.inv_with(&InvOptions::default(), &mut host)?;
    /// assert_eq!(r.as_slice::<f64>()?, &[0.5, 0.0, 0.0, 0.25]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn inv_with(&self, options: &InvOptions, exec: &mut dyn Engine) -> Res<Self::Out> {
        let _ = options;
        let a = self.linalg_read();
        let outs = with_linalg_backend(exec, "inv", |backend| inv_read(a, backend))?;
        Self::linalg_out(outs)
    }

    /// Compute the Moore-Penrose pseudoinverse.
    ///
    /// Short form of [`Self::pinv_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::pinv_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.pinv(&mut host)?;
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn pinv(&self, exec: &mut dyn Engine) -> Res<Self::Out> {
        self.pinv_with(&PinvOptions::default(), exec)
    }

    /// Compute the Moore-Penrose pseudoinverse.
    ///
    /// `rtol: None` uses the dtype-default cutoff; `Some(rtol)` an explicit relative cutoff. The
    /// cutoff thresholds an SVD whose default driver is accuracy-preserving (`SvdDriver::Auto`), so a
    /// rank-deficient rank is retained or dropped the same way as on CPU. CPU providers and CUDA
    /// share this contract.
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid `rtol`, plus validation, unsupported-backend,
    /// numerical, or SVD contract errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{PinvOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.pinv_with(&PinvOptions::default().with_rtol(1.0e-12), &mut host)?;
    /// assert_eq!(r.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn pinv_with(&self, options: &PinvOptions, exec: &mut dyn Engine) -> Res<Self::Out> {
        let a = self.linalg_read();
        let outs =
            with_linalg_backend(exec, "pinv", |backend| pinv_read(a, options.rtol, backend))?;
        Self::linalg_out(outs)
    }

    /// Compute a vector, matrix, or tensor norm.
    ///
    /// Short form of [`Self::norm_with`] with default options.
    ///
    /// # Errors
    ///
    /// As [`Self::norm_with`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.norm(&mut host)?;
    /// assert_eq!(r.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn norm(&self, exec: &mut dyn Engine) -> Res<Self::Real> {
        self.norm_with(&NormOptions::default(), exec)
    }

    /// Compute a vector, matrix, or tensor norm.
    ///
    /// `ord: None` is the Frobenius/Euclidean norm. `ord: Some(0.0)` is the count of entries that are
    /// not equal to zero: `NaN` counts as nonzero, both `+0.0` and `-0.0` count as zero, and `±inf`
    /// count as nonzero (#2359; see `docs/spec/tensor-semantics.md`). Negative `axes` count from the
    /// end; `axes: None` reduces every axis.
    ///
    /// # Errors
    ///
    /// Returns validation errors for axes/order combinations or required backend operations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_linalg::{KeepDims, NormOptions, TensorLinalgExt};
    /// # use tenferro_tensor::Tensor;
    /// # let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 4.0])?;
    /// # let mut host = tenferro_cpu::CpuEngine::serial();
    /// let r = a.norm_with(&NormOptions::default().with_keep_dims(KeepDims::Keep), &mut host)?;
    /// assert_eq!(r.shape(), &[1, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn norm_with(&self, options: &NormOptions, exec: &mut dyn Engine) -> Res<Self::Real> {
        let a = self.linalg_read();
        let dim = resolve_norm_axes("norm", options.axes.as_deref(), a.shape().len())?;
        let outs = with_linalg_backend(exec, "norm", |backend| {
            norm_from_read(
                a,
                options.ord,
                dim.as_deref(),
                options.keep_dims == KeepDims::Keep,
                backend,
            )
        })?;
        Self::linalg_real(outs)
    }
}

impl<X: LinalgInput> TensorLinalgExt for X {}

fn rank_revealing_qr_result(
    outputs: Vec<Tensor>,
) -> tenferro_tensor::Result<RankRevealingQrResult<Tensor>> {
    let (q, r, column_permutation, rank) = four(outputs, "rank_revealing_qr")?;
    Ok(RankRevealingQrResult {
        q,
        r,
        column_permutation,
        rank,
    })
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
    session: &mut dyn Engine,
    op: &'static str,
    f: impl FnOnce(&mut dyn LinalgBackend) -> tenferro_tensor::Result<X>,
) -> tenferro_tensor::Result<X> {
    // The capability branches are mutually exclusive, so `f` runs exactly
    // once. Probe the marker first, then re-extract the same exec session and
    // run the composite body on it (FnOnce cannot be captured by several
    // branch closures).
    if with_cpu_engine(session, |_| ()).is_some() {
        return with_cpu_engine(session, |exec| f(exec as &mut dyn LinalgBackend))
            .expect("marker probe matched a CPU execution session");
    }
    #[cfg(feature = "cuda")]
    if with_cuda_engine(session, |_| ()).is_some() {
        return with_cuda_engine(session, |exec| f(exec as &mut dyn LinalgBackend))
            .expect("marker probe matched a CUDA execution session");
    }
    Err(tenferro_tensor::Error::unsupported(
        op,
        "selected backend session does not expose a linalg execution capability",
    ))
}

/// The empty-determinant shortcut, taken after the backend's own input-residency boundary.
///
/// An empty square input needs no factorization, but it still crosses the same boundary the LU
/// route would: a host session narrows it to host once and reads the output shape and dtype from
/// that host view, and a device session applies the device-residency check. Neither copies or
/// factorizes the input.
fn empty_square_slogdet_on(
    session: &mut dyn Engine,
    op: &'static str,
    input: TensorRead<'_>,
) -> tenferro_tensor::Result<(Tensor, Tensor)> {
    if with_cpu_engine(session, |_| ()).is_some() {
        return with_cpu_engine(session, |exec| {
            let input = input.try_host(op)?;
            empty_square_slogdet(input.shape(), input.dtype(), exec as &mut dyn LinalgBackend)
        })
        .expect("marker probe matched a CPU execution session");
    }
    #[cfg(feature = "cuda")]
    if with_cuda_engine(session, |_| ()).is_some() {
        return with_cuda_engine(session, |backend| {
            crate::gpu::ensure_device_resident_read(backend, op, &input)?;
            empty_square_slogdet(
                input.shape(),
                input.dtype(),
                backend as &mut dyn LinalgBackend,
            )
        })
        .expect("marker probe matched a CUDA execution session");
    }
    Err(tenferro_tensor::Error::unsupported(
        op,
        "selected backend session does not expose a linalg execution capability",
    ))
}

/// The complete-pivot LU solve on borrowed views.
///
/// Every surface goes through the backend solve, which rejects a rank-deficient system (a pivot
/// at or below `eps * max|pivot|`). A borrowed view is made compact first.
fn full_piv_lu_solve_impl(
    a: TensorRead<'_>,
    b: TensorRead<'_>,
    backend: &mut dyn LinalgBackend,
) -> tenferro_tensor::Result<Tensor> {
    let a_owned;
    let a = match a.as_tensor() {
        Some(a) => a,
        None => {
            let shape = a.shape().to_vec();
            a_owned = backend.reshape_read(a, &shape)?;
            &a_owned
        }
    };
    let b_owned;
    let b = match b.as_tensor() {
        Some(b) => b,
        None => {
            let shape = b.shape().to_vec();
            b_owned = backend.reshape_read(b, &shape)?;
            &b_owned
        }
    };
    backend.full_piv_lu_solve(a, b, false)
}

/// Sign and log-absolute determinant of a borrowed square matrix, with the empty `0 x 0`
/// shortcut taken after the backend's own input-residency boundary.
fn slogdet_tensors(
    a: TensorRead<'_>,
    op: &'static str,
    exec: &mut dyn Engine,
) -> tenferro_tensor::Result<(Tensor, Tensor)> {
    validate_square_matrix(op, a.shape())?;
    if a.shape()[0] == 0 && a.shape()[1] == 0 {
        return empty_square_slogdet_on(exec, op, a);
    }
    with_linalg_backend(exec, op, |backend| {
        slogdet_from_lu(four(backend.lu_read(a)?, op)?, backend)
    })
}

/// Determinant of a borrowed square matrix as `sign * exp(logabsdet)`.
fn det_tensors(a: TensorRead<'_>, exec: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
    validate_square_matrix("det", a.shape())?;
    with_linalg_backend(exec, "det", |backend| {
        let slogdet = slogdet_tensors(a, "det", backend)?;
        det_impl(slogdet, backend)
    })
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
fn inv_read<B: LinalgBackend + ?Sized>(
    a: TensorRead<'_>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let eye = eye_like(a.dtype(), a.shape(), backend)?;
    backend.solve_read(a, TensorRead::from_tensor(&eye))
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
    validate_axes("norm", original_shape.len(), &axes)?;
    if axes.is_empty() {
        return backend.to_contiguous_read(input);
    }
    let reduced = if ord.is_none() && axes.len() == original_shape.len() {
        // Full Frobenius reduction: one fused `|x|^2` pass with no `abs`
        // temporary and no leading-axis layout copy. Every backend that owns
        // the norm surface implements the fused primitive.
        let squared = backend.norm_squared_read(input.clone())?;
        backend.sqrt_read(TensorRead::from_tensor(&squared))?
    } else if can_square_without_abs(input.dtype(), axes.len(), ord) {
        frobenius_norm_read(input.clone(), &axes, backend)?
    } else if axes.len() == 2 {
        matrix_norm(input, &axes, ord, backend)?
    } else {
        let abs = backend.abs_read(input)?;
        norm_over_axes(&abs, &axes, ord, backend)?
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

/// Sign and log-absolute determinant of the empty `0 x 0` batch, whose
/// determinant is the empty product `1` and whose log-absolute determinant is
/// therefore `0`. The batch axes become the output shape.
fn empty_square_slogdet<B: LinalgBackend + ?Sized>(
    shape: &[usize],
    dtype: DType,
    backend: &mut B,
) -> tenferro_tensor::Result<(Tensor, Tensor)> {
    let batch_shape = &shape[2..];
    let sign_scalar = resident(scalar_real(dtype, 1.0)?, backend)?;
    let sign = broadcast(&sign_scalar, batch_shape, &[], backend)?;
    let log_scalar = resident(scalar_real(real_values_dtype(dtype), 0.0)?, backend)?;
    let logabsdet = broadcast(&log_scalar, batch_shape, &[], backend)?;
    Ok((sign, logabsdet))
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

/// Make a freshly built host tensor resident on `backend`.
///
/// The composite helpers below build their internal constants on the host - a
/// ones scalar, a pseudoinverse threshold, a norm order - and a device backend
/// rejects a host tensor at the first operation that reads it; the caller has no
/// way to supply a replacement (#2147). This is the concrete-session counterpart
/// of the eager composites' `constant_from_host`: on a CPU backend it is a host
/// copy, and every constant here is rank-0 or a broadcast of one, so the copy is
/// a scalar.
fn resident<B: LinalgBackend + ?Sized>(
    tensor: Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.upload_host_tensor(TensorRead::from_tensor(&tensor))
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
    let scalar = resident(scalar_real(dtype, 1.0)?, backend)?;
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
    tenferro_tensor::validate::validate_unique_axes(op, "dim", rank, axes)
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
    let threshold_scalar = resident(scalar_real(abs_s.dtype(), rtol.max(0.0))?, backend)?;
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
    // A complex input has real singular values, so `s_inv` is real while `v` is
    // complex, and `scale_matrix_columns` multiplies them through `binary_mul`,
    // which requires one dtype. Casting the real factor into the complex type is
    // exact (#2049).
    let s_inv = if s_inv.dtype() == v.dtype() {
        s_inv
    } else {
        backend.convert(&s_inv, v.dtype())?
    };
    let vs = scale_matrix_columns(&v, &s_inv, backend)?;
    matmul_preserve_trailing_batch(&vs, &uh, backend)
}

fn ones_like<B: LinalgBackend + ?Sized>(
    input: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let one = resident(scalar_real(input.dtype(), 1.0)?, backend)?;
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

fn matmul_preserve_trailing_batch<B: Engine + ?Sized>(
    lhs: &Tensor,
    rhs: &Tensor,
    session: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let batch: Vec<usize> = (2..lhs.shape().len()).collect();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: batch.clone().iter().copied().collect(),
        rhs_batch_dims: batch.iter().copied().collect(),
    };
    session.dot_general_read(
        TensorRead::from_tensor(lhs),
        TensorRead::from_tensor(rhs),
        &config,
    )
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
    let power = resident(scalar_real(abs.dtype(), p)?, backend)?;
    let powered = backend.pow_read(
        TensorRead::from_tensor(abs),
        TensorRead::from_tensor(&power),
    )?;
    let sum = reduce_sum(&powered, axes, backend)?;
    let inverse = resident(scalar_real(abs.dtype(), 1.0 / p)?, backend)?;
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
    let zero = resident(scalar_real(abs.dtype(), 0.0)?, backend)?;
    let zero = broadcast(&zero, abs.shape(), &[], backend)?;
    // `ord = 0` counts `x != 0` (#2359). Equality with zero, then negated,
    // keeps NaN entries in the count: IEEE 754 makes `NaN > 0` false while
    // `NaN != 0` is true, so the former `Gt` mask dropped them.
    let is_zero = backend.compare_read(
        TensorRead::from_tensor(abs),
        TensorRead::from_tensor(&zero),
        &CompareDir::Eq,
    )?;
    let is_zero = backend.convert(&is_zero, abs.dtype())?;
    let ones = ones_like(abs, backend)?;
    let mask = backend.sub_read(
        TensorRead::from_tensor(&ones),
        TensorRead::from_tensor(&is_zero),
    )?;
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
    input: TensorRead<'_>,
    axes: &[usize],
    ord: Option<f64>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let matrix = move_axes_to_front(input, axes, backend)?;
    if matches!(ord, Some(2.0) | Some(-2.0)) {
        let (_, singular_values, _) = three(backend.svd(&matrix)?, "norm")?;
        return if ord == Some(2.0) {
            reduce_max(&singular_values, &[0], backend)
        } else {
            reduce_min(&singular_values, &[0], backend)
        };
    }

    let abs = backend.abs_read(TensorRead::from_tensor(&matrix))?;
    match ord {
        None => frobenius_norm(&abs, &[0, 1], backend),
        Some(p) if p == f64::INFINITY => row_sum_norm(&abs, true, backend),
        Some(p) if p == f64::NEG_INFINITY => row_sum_norm(&abs, false, backend),
        Some(1.0) => col_sum_norm(&abs, true, backend),
        Some(-1.0) => col_sum_norm(&abs, false, backend),
        Some(0.0) => count_nonzero(&abs, &[0, 1], backend),
        Some(p) => p_norm(&abs, &[0, 1], p, backend),
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
    input: TensorRead<'_>,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        return backend.to_contiguous_read(input);
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
    backend.transpose_read(input, &perm)
}
