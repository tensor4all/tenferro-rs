//! Crate-internal concrete linear algebra bodies behind `LinalgOps` (#2552).
//!
//! Owned tensors use [`OwnedLinalg`] and borrowed reads use [`ReadLinalg`];
//! the public grammar forwards to one of them. All methods dispatch internally to the built-in
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
use crate::lane;
use crate::traced::real_values_dtype;
use crate::validation::validate_square_matrix;
use crate::{LinalgBackend, RankRevealingQrOptions, RankRevealingQrResult};

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

pub(crate) mod private {
    #[cfg(feature = "native")]
    use tcontract_exec::Exec;
    #[cfg(feature = "native")]
    use tenferro_cpu::linalg_interop::BufferPool;
    #[cfg(feature = "native")]
    use tenferro_tensor::{TypedTensor, TypedTensorView};

    /// `(U, S, Vᴴ)` of the borrowed SVD entry.
    #[cfg(feature = "native")]
    pub type Svd<T> = (
        TypedTensor<T>,
        TypedTensor<<T as super::LinalgScalar>::RealScalar>,
        TypedTensor<T>,
    );

    /// The sealing supertrait of [`LinalgScalar`](super::LinalgScalar).
    ///
    /// `pub` inside a private module: nameable as a supertrait, unreachable outside the crate.
    /// Its methods route the public borrowed-`Exec` entries to the compiled provider route's typed
    /// bodies per concrete type, which is what lets the public bound stay free of the provider's
    /// own scalar traits (whose `Real`/`Complex` names would otherwise collide with
    /// `LinalgScalar::Complex`).
    pub trait Sealed {
        /// The body of the public `svd_on_exec`.
        ///
        /// # Errors
        ///
        /// As the public `svd_on_exec`: `ValidationError::RankMismatch` below rank 2,
        /// `Error::Unsupported` for a negative stride, `Error::RuntimeState` for non-host storage,
        /// `ValidationError::InvalidArgument` for an overflowing count or out-of-range layout, and
        /// a provider failure mapped by `map_error`.
        #[cfg(feature = "native")]
        fn svd_on_exec(
            exec: &mut Exec<'_>,
            buffers: &mut BufferPool,
            view: &TypedTensorView<'_, Self>,
            full: bool,
        ) -> tenferro_tensor::Result<Svd<Self>>
        where
            Self: super::LinalgScalar + Sized;
        /// The body of the public `svd_values_on_exec`.
        ///
        /// # Errors
        ///
        /// As the public `svd_values_on_exec`: `ValidationError::RankMismatch` below rank 2,
        /// `Error::Unsupported` for a negative stride, `Error::RuntimeState` for non-host storage,
        /// `ValidationError::InvalidArgument` for an overflowing count or out-of-range layout, and
        /// a provider failure mapped by `map_error`.
        #[cfg(feature = "native")]
        fn svd_values_on_exec(
            exec: &mut Exec<'_>,
            buffers: &mut BufferPool,
            view: &TypedTensorView<'_, Self>,
        ) -> tenferro_tensor::Result<TypedTensor<<Self as super::LinalgScalar>::RealScalar>>
        where
            Self: super::LinalgScalar + Sized;
        /// The body of the public `qr_on_exec`.
        ///
        /// # Errors
        ///
        /// As the public `qr_on_exec`: `ValidationError::RankMismatch` below rank 2,
        /// `Error::Unsupported` for a negative stride, `Error::RuntimeState` for non-host storage,
        /// `ValidationError::InvalidArgument` for an overflowing count or out-of-range layout, and
        /// a provider failure mapped by `map_error`.
        #[cfg(feature = "native")]
        fn qr_on_exec(
            exec: &mut Exec<'_>,
            buffers: &mut BufferPool,
            view: &TypedTensorView<'_, Self>,
        ) -> tenferro_tensor::Result<(TypedTensor<Self>, TypedTensor<Self>)>
        where
            Self: super::LinalgScalar + Sized;
        /// The body of the public `eigh_on_exec`.
        ///
        /// # Errors
        ///
        /// As the public `eigh_on_exec`: `ValidationError::RankMismatch` below rank 2, `ValidationError::ShapeMismatch` for a non-square matrix,
        /// `Error::Unsupported` for a negative stride, `Error::RuntimeState` for non-host storage,
        /// `ValidationError::InvalidArgument` for an overflowing count or out-of-range layout, and
        /// a provider failure mapped by `map_error`.
        #[cfg(feature = "native")]
        #[allow(clippy::type_complexity)]
        fn eigh_on_exec(
            exec: &mut Exec<'_>,
            buffers: &mut BufferPool,
            view: &TypedTensorView<'_, Self>,
        ) -> tenferro_tensor::Result<(
            TypedTensor<<Self as super::LinalgScalar>::RealScalar>,
            TypedTensor<Self>,
        )>
        where
            Self: super::LinalgScalar + Sized;
        /// The body of the public `eigh_values_on_exec`.
        ///
        /// # Errors
        ///
        /// As the public `eigh_values_on_exec`: `ValidationError::RankMismatch` below rank 2, `ValidationError::ShapeMismatch` for a non-square matrix,
        /// `Error::Unsupported` for a negative stride, `Error::RuntimeState` for non-host storage,
        /// `ValidationError::InvalidArgument` for an overflowing count or out-of-range layout, and
        /// a provider failure mapped by `map_error`.
        #[cfg(feature = "native")]
        fn eigh_values_on_exec(
            exec: &mut Exec<'_>,
            buffers: &mut BufferPool,
            view: &TypedTensorView<'_, Self>,
        ) -> tenferro_tensor::Result<TypedTensor<<Self as super::LinalgScalar>::RealScalar>>
        where
            Self: super::LinalgScalar + Sized;
        /// The body of the public `eig_values_on_exec`.
        ///
        /// # Errors
        ///
        /// As the public `eig_values_on_exec`: `ValidationError::RankMismatch` below rank 2, `ValidationError::ShapeMismatch` for a non-square matrix,
        /// `Error::Unsupported` for a negative stride, `Error::RuntimeState` for non-host storage,
        /// `ValidationError::InvalidArgument` for an overflowing count or out-of-range layout, and
        /// a provider failure mapped by `map_error`.
        #[cfg(feature = "native")]
        fn eig_values_on_exec(
            exec: &mut Exec<'_>,
            buffers: &mut BufferPool,
            view: &TypedTensorView<'_, Self>,
        ) -> tenferro_tensor::Result<TypedTensor<<Self as super::LinalgScalar>::Complex>>
        where
            Self: super::LinalgScalar + Sized;
    }

    macro_rules! impl_sealed {
        ($($scalar:ty),*) => {$(
            impl Sealed for $scalar {
                #[cfg(feature = "native")]
                fn svd_on_exec(
                    exec: &mut Exec<'_>,
                    buffers: &mut BufferPool,
                    view: &TypedTensorView<'_, Self>,
                    full: bool,
                ) -> tenferro_tensor::Result<Svd<Self>> {
                    crate::cpu::linalg::faer::svd_typed(exec, buffers, view, full)
                }
                #[cfg(feature = "native")]
                fn svd_values_on_exec(
                    exec: &mut Exec<'_>,
                    buffers: &mut BufferPool,
                    view: &TypedTensorView<'_, Self>,
                ) -> tenferro_tensor::Result<TypedTensor<<Self as super::LinalgScalar>::RealScalar>> {
                    crate::cpu::linalg::faer::svd_values_typed(exec, buffers, view)
                }
                #[cfg(feature = "native")]
                fn qr_on_exec(
                    exec: &mut Exec<'_>,
                    buffers: &mut BufferPool,
                    view: &TypedTensorView<'_, Self>,
                ) -> tenferro_tensor::Result<(TypedTensor<Self>, TypedTensor<Self>)> {
                    crate::cpu::linalg::faer::qr_typed(exec, buffers, view)
                }
                #[cfg(feature = "native")]
                fn eigh_on_exec(
                    exec: &mut Exec<'_>,
                    buffers: &mut BufferPool,
                    view: &TypedTensorView<'_, Self>,
                ) -> tenferro_tensor::Result<(
                    TypedTensor<<Self as super::LinalgScalar>::RealScalar>,
                    TypedTensor<Self>,
                )> {
                    crate::cpu::linalg::faer::eigh_typed(exec, buffers, view)
                }
                #[cfg(feature = "native")]
                fn eigh_values_on_exec(
                    exec: &mut Exec<'_>,
                    buffers: &mut BufferPool,
                    view: &TypedTensorView<'_, Self>,
                ) -> tenferro_tensor::Result<TypedTensor<<Self as super::LinalgScalar>::RealScalar>> {
                    crate::cpu::linalg::faer::eigh_values_typed(exec, buffers, view)
                }
                #[cfg(feature = "native")]
                fn eig_values_on_exec(
                    exec: &mut Exec<'_>,
                    buffers: &mut BufferPool,
                    view: &TypedTensorView<'_, Self>,
                ) -> tenferro_tensor::Result<TypedTensor<<Self as super::LinalgScalar>::Complex>> {
                    crate::cpu::linalg::faer::eig_values_typed(exec, buffers, view)
                }
            }
        )*};
    }
    impl_sealed!(f32, f64, num_complex::Complex32, num_complex::Complex64);
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

/// Crate-internal bodies of the owned-tensor linalg entries; the public
/// grammar is `LinalgOps` (#2552), which forwards here.
pub(crate) trait OwnedLinalg {
    fn svd(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    fn svdvals(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn svd_with_options(
        &self,
        options: SvdOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    fn svd_full(
        &self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    fn qr(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)>;

    fn householder_qr(
        &self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<crate::HouseholderQr<Tensor>>;

    fn qr_with_options(
        &self,
        options: QrOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn rank_revealing_qr(
        &self,
        options: RankRevealingQrOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<RankRevealingQrResult<Tensor>>;
    fn lu(
        &self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)>;
    fn full_piv_lu(
        &self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    fn full_piv_lu_solve(
        &self,
        b: &Tensor,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    fn solve(&self, b: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn cholesky(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn eigh(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn eigh_with_options(
        &self,
        options: EighOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn eig(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    #[allow(clippy::too_many_arguments)]
    fn triangular_solve(
        &self,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    fn slogdet(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn det(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn inv(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn eigvalsh(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn eigvals(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn pinv(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn pinv_with_rtol(
        &self,
        rtol: f64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    fn norm(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
}

/// Crate-internal bodies of the borrowed-read linalg entries; the public
/// grammar is `LinalgOps` (#2552), which forwards here.
pub(crate) trait ReadLinalg {
    fn svd_read(
        self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    fn svdvals_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn svd_with_options_read(
        self,
        options: SvdOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    fn svd_full_read(
        self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    fn qr_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn qr_with_options_read(
        self,
        options: QrOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn rank_revealing_qr_read(
        self,
        options: RankRevealingQrOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<RankRevealingQrResult<Tensor>>;
    fn lu_read(
        self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)>;
    fn full_piv_lu_read(
        self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    fn full_piv_lu_solve_read(
        self,
        b: TensorRead<'_>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    fn solve_read(
        self,
        b: TensorRead<'_>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    fn solve_read_into(
        self,
        b: TensorRead<'_>,
        out: TensorWrite<'_>,
        session: &mut dyn Engine,
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
    fn cholesky_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn eigh_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn eigh_with_options_read(
        self,
        options: EighOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn eig_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    #[allow(clippy::too_many_arguments)]
    fn triangular_solve_read(
        self,
        b: TensorRead<'_>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    fn slogdet_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    fn det_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn inv_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn eigvalsh_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn eigvals_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn pinv_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
    fn pinv_with_rtol_read(
        self,
        rtol: f64,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
    fn norm_read(
        self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
}

impl OwnedLinalg for Tensor {
    fn svd(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        if let Some(result) =
            lane::try_lane_svd(session, TensorRead::from_tensor(self), false, None)
        {
            return result;
        }
        with_linalg_backend(session, "svd", |backend| three(backend.svd(self)?, "svd"))
    }
    fn svdvals(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        if let Some(result) = lane::try_lane_svdvals(session, TensorRead::from_tensor(self)) {
            return result;
        }
        with_linalg_backend(session, "svdvals", |backend| backend.svd_values(self))
    }

    fn svd_with_options(
        &self,
        options: SvdOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_svd(
            session,
            TensorRead::from_tensor(self),
            false,
            Some(("svd_with_options", options)),
        ) {
            return result;
        }
        with_linalg_backend(session, "svd_with_options", |backend| {
            three(backend.svd_with_options(self, options)?, "svd_with_options")
        })
    }
    fn svd_full(
        &self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_svd(session, TensorRead::from_tensor(self), true, None)
        {
            return result;
        }
        with_linalg_backend(session, "svd_full", |backend| {
            three(backend.svd_full(self)?, "svd_full")
        })
    }
    fn qr(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_qr(session, TensorRead::from_tensor(self)) {
            return result;
        }
        with_linalg_backend(session, "qr", |backend| two(backend.qr(self)?, "qr"))
    }
    fn householder_qr(
        &self,
        session: &mut dyn Engine,
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
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "qr_with_options", |backend| {
            two(backend.qr_with_options(self, options)?, "qr_with_options")
        })
    }
    fn rank_revealing_qr(
        &self,
        options: RankRevealingQrOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<RankRevealingQrResult<Tensor>> {
        with_linalg_backend(session, "rank_revealing_qr", |backend| {
            rank_revealing_qr_result(backend.rank_revealing_qr(self, options)?)
        })
    }
    fn lu(
        &self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "lu", |backend| four(backend.lu(self)?, "lu"))
    }
    fn full_piv_lu(
        &self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "full_piv_lu", |backend| {
            five(backend.full_piv_lu(self)?, "full_piv_lu")
        })
    }
    fn full_piv_lu_solve(
        &self,
        b: &Tensor,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "full_piv_lu_solve", |backend| {
            backend.full_piv_lu_solve(self, b, false)
        })
    }
    fn solve(&self, b: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "solve", |backend| backend.solve(self, b))
    }
    fn cholesky(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "cholesky", |backend| backend.cholesky(self))
    }
    fn eigh(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_eigh(session, TensorRead::from_tensor(self), None) {
            return result;
        }
        with_linalg_backend(session, "eigh", |backend| two(backend.eigh(self)?, "eigh"))
    }
    fn eigh_with_options(
        &self,
        options: EighOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_eigh(
            session,
            TensorRead::from_tensor(self),
            Some(("eigh_with_options", options)),
        ) {
            return result;
        }
        with_linalg_backend(session, "eigh_with_options", |backend| {
            two(
                backend.eigh_with_options(self, options)?,
                "eigh_with_options",
            )
        })
    }
    fn eig(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "eig", |backend| two(backend.eig(self)?, "eig"))
    }
    fn triangular_solve(
        &self,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "triangular_solve", |backend| {
            backend.triangular_solve(self, b, left_side, lower, transpose_a, unit_diagonal)
        })
    }
    fn slogdet(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        validate_square_matrix("slogdet", self.shape())?;
        if self.shape()[0] == 0 && self.shape()[1] == 0 {
            return empty_square_slogdet_on(session, "slogdet", TensorRead::from_tensor(self));
        }
        with_linalg_backend(session, "slogdet", |backend| {
            slogdet_from_lu(four(backend.lu(self)?, "slogdet")?, backend)
        })
    }
    fn det(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        validate_square_matrix("det", self.shape())?;
        with_linalg_backend(session, "det", |backend| {
            det_impl(self.slogdet(backend)?, backend)
        })
    }
    fn inv(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "inv", |backend| inv_owned(self, backend))
    }
    fn eigvalsh(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        if let Some(result) = lane::try_lane_eigvalsh(session, TensorRead::from_tensor(self)) {
            return result;
        }
        with_linalg_backend(session, "eigvalsh", |backend| backend.eigh_values(self))
    }
    fn eigvals(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        if let Some(result) = lane::try_lane_eigvals(session, TensorRead::from_tensor(self)) {
            return result;
        }
        with_linalg_backend(session, "eigvals", |backend| backend.eig_values(self))
    }
    fn pinv(&self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "pinv", |backend| pinv_owned(self, None, backend))
    }
    fn pinv_with_rtol(
        &self,
        rtol: f64,
        session: &mut dyn Engine,
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
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "norm", |backend| {
            norm_from_read(TensorRead::from_tensor(self), ord, dim, keepdim, backend)
        })
    }
}

impl ReadLinalg for TensorRead<'_> {
    fn svd_read(
        self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_svd(session, self.clone(), false, None) {
            return result;
        }
        with_linalg_backend(session, "svd_read", |backend| {
            three(backend.svd_read(self)?, "svd_read")
        })
    }
    fn svdvals_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        if let Some(result) = lane::try_lane_svdvals(session, self.clone()) {
            return result;
        }
        with_linalg_backend(session, "svdvals_read", |backend| {
            backend.svd_values_read(self)
        })
    }

    fn svd_with_options_read(
        self,
        options: SvdOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_svd(
            session,
            self.clone(),
            false,
            Some(("svd_with_options_read", options)),
        ) {
            return result;
        }
        with_linalg_backend(session, "svd_with_options_read", |backend| {
            three(
                backend.svd_with_options_read(self, options)?,
                "svd_with_options_read",
            )
        })
    }
    fn svd_full_read(
        self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_svd(session, self.clone(), true, None) {
            return result;
        }
        with_linalg_backend(session, "svd_full_read", |backend| {
            three(backend.svd_full_read(self)?, "svd_full_read")
        })
    }
    fn qr_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_qr(session, self.clone()) {
            return result;
        }
        with_linalg_backend(session, "qr_read", |backend| {
            two(backend.qr_read(self)?, "qr_read")
        })
    }
    fn qr_with_options_read(
        self,
        options: QrOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        with_linalg_backend(session, "qr_with_options_read", |backend| {
            two(
                backend.qr_with_options_read(self, options)?,
                "qr_with_options_read",
            )
        })
    }
    fn rank_revealing_qr_read(
        self,
        options: RankRevealingQrOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<RankRevealingQrResult<Tensor>> {
        with_linalg_backend(session, "rank_revealing_qr_read", |backend| {
            rank_revealing_qr_result(backend.rank_revealing_qr_read(self, options)?)
        })
    }
    fn lu_read(
        self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "lu_read", |backend| {
            four(backend.lu_read(self)?, "lu_read")
        })
    }
    fn full_piv_lu_read(
        self,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        with_linalg_backend(session, "full_piv_lu_read", |backend| {
            five(backend.full_piv_lu_read(self)?, "full_piv_lu_read")
        })
    }
    fn full_piv_lu_solve_read(
        self,
        b: TensorRead<'_>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "full_piv_lu_solve_read", |backend| {
            // Every surface goes through the backend solve, which rejects a rank-deficient
            // system (a pivot at or below `eps * max|pivot|`). A borrowed view is made compact
            // first.
            let a_owned;
            let a = match self.as_tensor() {
                Some(a) => a,
                None => {
                    let shape = self.shape().to_vec();
                    a_owned = backend.reshape_read(self, &shape)?;
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
        })
    }
    fn solve_read(
        self,
        b: TensorRead<'_>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "solve_read", |backend| backend.solve_read(self, b))
    }
    fn solve_read_into(
        self,
        b: TensorRead<'_>,
        out: TensorWrite<'_>,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<()> {
        with_linalg_backend(session, "solve_read_into", |backend| {
            backend.solve_read_into(self, b, out)
        })
    }
    fn cholesky_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "cholesky_read", |backend| {
            backend.cholesky_read(self)
        })
    }
    fn eigh_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_eigh(session, self.clone(), None) {
            return result;
        }
        with_linalg_backend(session, "eigh_read", |backend| {
            two(backend.eigh_read(self)?, "eigh_read")
        })
    }
    fn eigh_with_options_read(
        self,
        options: EighOptions,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        if let Some(result) = lane::try_lane_eigh(
            session,
            self.clone(),
            Some(("eigh_with_options_read", options)),
        ) {
            return result;
        }
        with_linalg_backend(session, "eigh_with_options_read", |backend| {
            two(
                backend.eigh_with_options_read(self, options)?,
                "eigh_with_options_read",
            )
        })
    }
    fn eig_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
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
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "triangular_solve_read", |backend| {
            backend.triangular_solve_read(self, b, left_side, lower, transpose_a, unit_diagonal)
        })
    }
    fn slogdet_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        validate_square_matrix("slogdet_read", self.shape())?;
        if self.shape()[0] == 0 && self.shape()[1] == 0 {
            return empty_square_slogdet_on(session, "slogdet_read", self);
        }
        with_linalg_backend(session, "slogdet_read", |backend| {
            slogdet_from_lu(four(backend.lu_read(self)?, "slogdet_read")?, backend)
        })
    }
    fn det_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        validate_square_matrix("det_read", self.shape())?;
        with_linalg_backend(session, "det_read", |backend| {
            det_impl(self.slogdet_read(backend)?, backend)
        })
    }
    fn inv_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "inv_read", |backend| inv_read(self, backend))
    }
    fn eigvalsh_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        if let Some(result) = lane::try_lane_eigvalsh(session, self.clone()) {
            return result;
        }
        with_linalg_backend(session, "eigvalsh_read", |backend| {
            backend.eigh_values_read(self)
        })
    }
    fn eigvals_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        if let Some(result) = lane::try_lane_eigvals(session, self.clone()) {
            return result;
        }
        with_linalg_backend(session, "eigvals_read", |backend| {
            backend.eig_values_read(self)
        })
    }
    fn pinv_read(self, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "pinv_read", |backend| {
            pinv_read(self, None, backend)
        })
    }
    fn pinv_with_rtol_read(
        self,
        rtol: f64,
        session: &mut dyn Engine,
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
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_linalg_backend(session, "norm_read", |backend| {
            norm_from_read(self, ord, dim, keepdim, backend)
        })
    }
}

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
    matmul_preserve_trailing_batch_read(lhs, TensorRead::from_tensor(rhs), session)
}

fn matmul_preserve_trailing_batch_read<B: Engine + ?Sized>(
    lhs: &Tensor,
    rhs: TensorRead<'_>,
    session: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let batch: Vec<usize> = (2..lhs.shape().len()).collect();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: batch.clone().iter().copied().collect(),
        rhs_batch_dims: batch.iter().copied().collect(),
    };
    session.dot_general_read(TensorRead::from_tensor(lhs), rhs, &config)
}

/// Compact Householder QR of a borrowed operand.
///
/// The compact factorization takes an owned tensor, so a borrowed operand is
/// first made compact (same placement and dtype) by the backend.
pub(crate) fn householder_qr_read(
    a: TensorRead<'_>,
    session: &mut dyn Engine,
) -> tenferro_tensor::Result<crate::HouseholderQr<Tensor>> {
    with_linalg_backend(session, "householder_qr", |backend| {
        let compact = backend.to_contiguous_read(a)?;
        backend
            .householder_qr(&compact)
            .map(crate::HouseholderQr::from_backend)
    })
}

/// Least squares `argmin_x |a x - b|` for a tall or square full-column-rank `a`.
///
/// The same composition the traced and eager `lstsq` build: thin QR `a = q r`,
/// then the triangular solve `r x = qᴴ b`. `a` is read like `qr_read` reads it;
/// `b` is read in place by the contraction.
pub(crate) fn lstsq_read(
    a: TensorRead<'_>,
    b: TensorRead<'_>,
    session: &mut dyn Engine,
) -> tenferro_tensor::Result<Tensor> {
    const OP: &str = "lstsq";
    ensure_float_or_complex(OP, a.dtype())?;
    for rank in [a.shape().len(), b.shape().len()] {
        if rank < 2 {
            return Err(tenferro_tensor::Error::rank_mismatch(OP, 2, rank));
        }
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m < n {
        return Err(tenferro_tensor::Error::invalid_argument(
            OP,
            "shape",
            format!(
                "lstsq requires a tall or square matrix (rows {m} >= cols {n}); \
                 underdetermined (wide) systems are not supported"
            ),
        ));
    }
    with_linalg_backend(session, OP, |backend| {
        let (q, r) = two(backend.qr_read(a)?, OP)?;
        let qh = conjugate_transpose(&q, backend)?;
        let qh_b = matmul_preserve_trailing_batch_read(&qh, b, backend)?;
        backend.triangular_solve(&r, &qh_b, true, false, false, false)
    })
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
