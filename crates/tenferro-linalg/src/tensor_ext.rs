//! Receiver-first concrete linear algebra surfaces.
//!
//! Owned tensors use [`TensorLinalgExt`], borrowed tensors and views use the
//! `_read` methods on [`TensorReadLinalgExt`], and typed tensors use
//! [`TypedTensorLinalgExt`]. All methods reuse a caller-owned backend.

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, Tensor, TensorRead, TensorScalar, TypedTensor,
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
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
/// let mut backend = CpuBackend::new();
/// let (_u, singular_values, _vt) = a.svd(&mut backend)?;
/// assert_eq!(singular_values.as_slice::<f64>()?, &[4.0, 2.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorLinalgExt {
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or output-contract errors.
    fn svd<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for matrix metadata or options, plus SVD backend errors.
    fn svd_with_options<B: LinalgBackend>(
        &self,
        options: SvdOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or output-contract errors.
    fn qr<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for matrix metadata or options, plus QR backend errors.
    fn qr_with_options<B: LinalgBackend>(
        &self,
        options: QrOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or output-contract errors.
    fn lu<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or output-contract errors.
    fn full_piv_lu<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for incompatible inputs or a backend/singular-system error.
    fn full_piv_lu_solve<B: LinalgBackend>(
        &self,
        b: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for incompatible inputs or a backend/singular-system error.
    fn solve<B: LinalgBackend>(
        &self,
        b: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for a non-square input or backend/numerical errors.
    fn cholesky<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, convergence, or output-contract errors.
    fn eigh<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for matrix metadata or options, plus Eigh backend errors.
    fn eigh_with_options<B: LinalgBackend>(
        &self,
        options: EighOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, convergence, or output-contract errors.
    fn eig<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for incompatible inputs/flags or backend/numerical errors.
    #[allow(clippy::too_many_arguments)]
    fn triangular_solve<B: LinalgBackend>(
        &self,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or LU output-contract errors.
    fn slogdet<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns the validation, backend, numerical, or contract errors from [`Self::slogdet`].
    fn det<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, or singular-solve numerical errors.
    fn inv<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns the validation, backend, convergence, or contract errors from [`Self::eigh`].
    fn eigvalsh<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns the validation, backend, convergence, or contract errors from [`Self::eig`].
    fn eigvals<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, unsupported-backend, numerical, or SVD contract errors.
    fn pinv<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns a validation error for invalid `rtol`, plus errors from [`Self::pinv`].
    fn pinv_with_rtol<B: LinalgBackend>(
        &self,
        rtol: f64,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for axes/order combinations or required backend operations.
    fn norm<B: LinalgBackend>(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
}

/// Linear algebra methods for borrowed tensor reads.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_linalg::TensorReadLinalgExt;
/// use tenferro_tensor::{Tensor, TensorRead};
///
/// let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
/// let mut backend = CpuBackend::new();
/// let (_q, r) = TensorRead::from_tensor(&input).qr_read(&mut backend)?;
/// assert_eq!(r.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorReadLinalgExt {
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, same-placement materialization, backend, numerical, or contract errors.
    fn svd_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus read/materialization or SVD errors.
    fn svd_with_options_read<B: LinalgBackend>(
        self,
        options: SvdOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, same-placement materialization, backend, numerical, or contract errors.
    fn qr_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus read/materialization or QR errors.
    fn qr_with_options_read<B: LinalgBackend>(
        self,
        options: QrOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, same-placement materialization, backend, numerical, or contract errors.
    fn lu_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, same-placement materialization, backend, numerical, or contract errors.
    fn full_piv_lu_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input validation, read/materialization, backend, or singular errors.
    fn full_piv_lu_solve_read<B: LinalgBackend>(
        self,
        b: TensorRead<'_>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input validation, read/materialization, backend, or singular errors.
    fn solve_read<B: LinalgBackend>(
        self,
        b: TensorRead<'_>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns matrix validation, read/materialization, backend, or positive-definiteness errors.
    fn cholesky_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, convergence, or contract errors.
    fn eigh_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus read or Eigh backend errors.
    fn eigh_with_options_read<B: LinalgBackend>(
        self,
        options: EighOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, convergence, or contract errors.
    fn eig_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input/flag validation, read, backend, or singular errors.
    #[allow(clippy::too_many_arguments)]
    fn triangular_solve_read<B: LinalgBackend>(
        self,
        b: TensorRead<'_>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, numerical, or LU contract errors.
    fn slogdet_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, numerical, or contract errors.
    fn det_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, or singular-solve errors.
    fn inv_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, convergence, or contract errors.
    fn eigvalsh_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, convergence, or contract errors.
    fn eigvals_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, read/materialization, backend, numerical, or SVD contract errors.
    fn pinv_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns a validation error for invalid `rtol`, plus errors from [`Self::pinv_read`].
    fn pinv_with_rtol_read<B: LinalgBackend>(
        self,
        rtol: f64,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for axes/order combinations or required read/backend operations.
    fn norm_read<B: LinalgBackend>(
        self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        backend: &mut B,
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
/// use tenferro_tensor::TypedTensor;
///
/// let input = TypedTensor::<f64>::from_vec_col_major(
///     vec![2, 2],
///     vec![2.0, 0.0, 0.0, 4.0],
/// )?;
/// let mut backend = CpuBackend::new();
/// let (_u, singular_values, _vt) = input.svd(&mut backend)?;
/// assert_eq!(singular_values.as_slice()?, &[4.0, 2.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TypedTensorLinalgExt<T: LinalgScalar> {
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    fn svd<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedSvd<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus backend or typed-output errors.
    fn svd_with_options<B: LinalgBackend>(
        &self,
        options: SvdOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedSvd<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    fn qr<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus backend or typed-output errors.
    fn qr_with_options<B: LinalgBackend>(
        &self,
        options: QrOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    fn lu<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedLu<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    fn full_piv_lu<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedFullPivLu<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input validation, backend, singular, or typed-downcast errors.
    fn full_piv_lu_solve<B: LinalgBackend>(
        &self,
        b: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input validation, backend, singular, or typed-downcast errors.
    fn solve<B: LinalgBackend>(
        &self,
        b: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns matrix validation, backend, positive-definiteness, or typed-downcast errors.
    fn cholesky<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, convergence, output-contract, or typed-downcast errors.
    fn eigh<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for metadata/options, plus backend or typed-output errors.
    fn eigh_with_options<B: LinalgBackend>(
        &self,
        options: EighOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, convergence, output-contract, or typed-downcast errors.
    fn eig<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedEig<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns incompatible-input/flag validation, backend, singular, or typed-output errors.
    #[allow(clippy::too_many_arguments)]
    fn triangular_solve<B: LinalgBackend>(
        &self,
        b: &TypedTensor<T>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    fn slogdet<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<<T as TensorScalar>::Real>)>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    fn det<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, singular-solve, or typed-downcast errors.
    fn inv<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, convergence, output-contract, or typed-downcast errors.
    fn eigvalsh<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, convergence, output-contract, or typed-downcast errors.
    fn eigvals<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T::Complex>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation, backend, numerical, output-contract, or typed-downcast errors.
    fn pinv<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns a validation error for invalid `rtol`, plus backend or typed-output errors.
    fn pinv_with_rtol<B: LinalgBackend>(
        &self,
        rtol: f64,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// # Errors
    /// Returns `tenferro_tensor::Error::Unsupported` when the selected backend does not support the operation.
    /// Returns validation errors for axes/order combinations, backend, or typed-output errors.
    fn norm<B: LinalgBackend>(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>>;
}

impl TensorLinalgExt for Tensor {
    fn svd<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        three(backend.svd(self)?, "svd")
    }
    fn svd_with_options<B: LinalgBackend>(
        &self,
        options: SvdOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        three(backend.svd_with_options(self, options)?, "svd_with_options")
    }
    fn qr<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        two(backend.qr(self)?, "qr")
    }
    fn qr_with_options<B: LinalgBackend>(
        &self,
        options: QrOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        two(backend.qr_with_options(self, options)?, "qr_with_options")
    }
    fn lu<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)> {
        four(backend.lu(self)?, "lu")
    }
    fn full_piv_lu<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        five(backend.full_piv_lu(self)?, "full_piv_lu")
    }
    fn full_piv_lu_solve<B: LinalgBackend>(
        &self,
        b: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        backend.full_piv_lu_solve(self, b, false)
    }
    fn solve<B: LinalgBackend>(
        &self,
        b: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        backend.solve(self, b)
    }
    fn cholesky<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        backend.cholesky(self)
    }
    fn eigh<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        two(backend.eigh(self)?, "eigh")
    }
    fn eigh_with_options<B: LinalgBackend>(
        &self,
        options: EighOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        two(
            backend.eigh_with_options(self, options)?,
            "eigh_with_options",
        )
    }
    fn eig<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        two(backend.eig(self)?, "eig")
    }
    fn triangular_solve<B: LinalgBackend>(
        &self,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        backend.triangular_solve(self, b, left_side, lower, transpose_a, unit_diagonal)
    }
    fn slogdet<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        slogdet_from_lu(four(backend.lu(self)?, "slogdet")?, backend)
    }
    fn det<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        det_impl(self.slogdet(backend)?, backend)
    }
    fn inv<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        inv_owned(self, backend)
    }
    fn eigvalsh<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        Ok(self.eigh(backend)?.0)
    }
    fn eigvals<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        Ok(self.eig(backend)?.0)
    }
    fn pinv<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        pinv_owned(self, None, backend)
    }
    fn pinv_with_rtol<B: LinalgBackend>(
        &self,
        rtol: f64,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        pinv_owned(self, Some(rtol), backend)
    }
    fn norm<B: LinalgBackend>(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        norm_from_read(TensorRead::from_tensor(self), ord, dim, keepdim, backend)
    }
}

impl TensorReadLinalgExt for TensorRead<'_> {
    fn svd_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        three(backend.svd_read(self)?, "svd_read")
    }
    fn svd_with_options_read<B: LinalgBackend>(
        self,
        options: SvdOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor)> {
        validate_derivative_eps("svd_with_options_read", options.derivative_eps)?;
        let mut out = backend.svd_read(self)?;
        apply_svd_gauge(options.gauge, &mut out)?;
        three(out, "svd_with_options_read")
    }
    fn qr_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        two(backend.qr_read(self)?, "qr_read")
    }
    fn qr_with_options_read<B: LinalgBackend>(
        self,
        options: QrOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        let mut out = backend.qr_read(self)?;
        apply_qr_gauge(options.gauge, &mut out)?;
        two(out, "qr_with_options_read")
    }
    fn lu_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor)> {
        four(backend.lu_read(self)?, "lu_read")
    }
    fn full_piv_lu_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        five(backend.full_piv_lu_read(self)?, "full_piv_lu_read")
    }
    fn full_piv_lu_solve_read<B: LinalgBackend>(
        self,
        b: TensorRead<'_>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let b_is_vector = b.shape().len() + 1 == self.shape().len();
        let original_b_shape = b.shape().to_vec();
        let vector_as_matrix = if b_is_vector {
            let mut shape = vec![b.shape()[0], 1];
            shape.extend_from_slice(&b.shape()[1..]);
            Some(backend.with_backend_session(|exec| exec.reshape_read(b.clone(), &shape))?)
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
    }
    fn solve_read<B: LinalgBackend>(
        self,
        b: TensorRead<'_>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        backend.solve_read(self, b)
    }
    fn cholesky_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        backend.cholesky_read(self)
    }
    fn eigh_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        two(backend.eigh_read(self)?, "eigh_read")
    }
    fn eigh_with_options_read<B: LinalgBackend>(
        self,
        options: EighOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        validate_derivative_eps("eigh_with_options_read", options.derivative_eps)?;
        let mut out = backend.eigh_read(self)?;
        apply_eigh_gauge(options.gauge, &mut out)?;
        two(out, "eigh_with_options_read")
    }
    fn eig_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        two(backend.eig_read(self)?, "eig_read")
    }
    fn triangular_solve_read<B: LinalgBackend>(
        self,
        b: TensorRead<'_>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        backend.triangular_solve_read(self, b, left_side, lower, transpose_a, unit_diagonal)
    }
    fn slogdet_read<B: LinalgBackend>(
        self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        slogdet_from_lu(four(backend.lu_read(self)?, "slogdet_read")?, backend)
    }
    fn det_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        det_impl(self.slogdet_read(backend)?, backend)
    }
    fn inv_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        inv_read(self, backend)
    }
    fn eigvalsh_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        Ok(self.eigh_read(backend)?.0)
    }
    fn eigvals_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        Ok(self.eig_read(backend)?.0)
    }
    fn pinv_read<B: LinalgBackend>(self, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
        pinv_read(self, None, backend)
    }
    fn pinv_with_rtol_read<B: LinalgBackend>(
        self,
        rtol: f64,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        pinv_read(self, Some(rtol), backend)
    }
    fn norm_read<B: LinalgBackend>(
        self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        norm_from_read(self, ord, dim, keepdim, backend)
    }
}

impl<T: LinalgScalar> TypedTensorLinalgExt<T> for TypedTensor<T> {
    fn svd<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedSvd<T>> {
        typed_svd(T::tensor_read(self).svd_read(backend)?)
    }

    fn svd_with_options<B: LinalgBackend>(
        &self,
        options: SvdOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedSvd<T>> {
        typed_svd(T::tensor_read(self).svd_with_options_read(options, backend)?)
    }

    fn qr<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
        typed_pair_same(T::tensor_read(self).qr_read(backend)?)
    }

    fn qr_with_options<B: LinalgBackend>(
        &self,
        options: QrOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
        typed_pair_same(T::tensor_read(self).qr_with_options_read(options, backend)?)
    }

    fn lu<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedLu<T>> {
        let (p, l, u, parity) = T::tensor_read(self).lu_read(backend)?;
        Ok((
            typed_output::<T>(p)?,
            typed_output::<T>(l)?,
            typed_output::<T>(u)?,
            typed_output::<<T as TensorScalar>::Real>(parity)?,
        ))
    }

    fn full_piv_lu_solve<B: LinalgBackend>(
        &self,
        b: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_output::<T>(T::tensor_read(self).full_piv_lu_solve_read(T::tensor_read(b), backend)?)
    }

    fn full_piv_lu<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedFullPivLu<T>> {
        let (p, l, u, q, parity) = T::tensor_read(self).full_piv_lu_read(backend)?;
        Ok((
            typed_output::<T>(p)?,
            typed_output::<T>(l)?,
            typed_output::<T>(u)?,
            typed_output::<T>(q)?,
            typed_output::<<T as TensorScalar>::Real>(parity)?,
        ))
    }

    fn solve<B: LinalgBackend>(
        &self,
        b: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_output::<T>(T::tensor_read(self).solve_read(T::tensor_read(b), backend)?)
    }

    fn cholesky<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_output::<T>(T::tensor_read(self).cholesky_read(backend)?)
    }

    fn eigh<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)> {
        typed_eigh(T::tensor_read(self).eigh_read(backend)?)
    }

    fn eigh_with_options<B: LinalgBackend>(
        &self,
        options: EighOptions,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<<T as TensorScalar>::Real>, TypedTensor<T>)> {
        typed_eigh(T::tensor_read(self).eigh_with_options_read(options, backend)?)
    }

    fn eig<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedEig<T>> {
        let (values, vectors) = T::tensor_read(self).eig_read(backend)?;
        Ok((
            typed_output::<T::Complex>(values)?,
            typed_output::<T::Complex>(vectors)?,
        ))
    }

    fn triangular_solve<B: LinalgBackend>(
        &self,
        b: &TypedTensor<T>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_output::<T>(T::tensor_read(self).triangular_solve_read(
            T::tensor_read(b),
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
            backend,
        )?)
    }

    fn slogdet<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<<T as TensorScalar>::Real>)> {
        let (sign, logabsdet) = T::tensor_read(self).slogdet_read(backend)?;
        Ok((
            typed_output::<T>(sign)?,
            typed_output::<<T as TensorScalar>::Real>(logabsdet)?,
        ))
    }

    fn det<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_output::<T>(T::tensor_read(self).det_read(backend)?)
    }

    fn inv<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_output::<T>(T::tensor_read(self).inv_read(backend)?)
    }

    fn eigvalsh<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>> {
        typed_output::<<T as TensorScalar>::Real>(T::tensor_read(self).eigvalsh_read(backend)?)
    }

    fn eigvals<B: LinalgBackend>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T::Complex>> {
        typed_output::<T::Complex>(T::tensor_read(self).eigvals_read(backend)?)
    }

    fn pinv<B: LinalgBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_output::<T>(T::tensor_read(self).pinv_read(backend)?)
    }

    fn pinv_with_rtol<B: LinalgBackend>(
        &self,
        rtol: f64,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_output::<T>(T::tensor_read(self).pinv_with_rtol_read(rtol, backend)?)
    }

    fn norm<B: LinalgBackend>(
        &self,
        ord: Option<f64>,
        dim: Option<&[usize]>,
        keepdim: bool,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<<T as TensorScalar>::Real>> {
        typed_output::<<T as TensorScalar>::Real>(
            T::tensor_read(self).norm_read(ord, dim, keepdim, backend)?,
        )
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
fn slogdet_from_lu<B: LinalgBackend>(
    (_p, _l, u, parity): (Tensor, Tensor, Tensor, Tensor),
    backend: &mut B,
) -> tenferro_tensor::Result<(Tensor, Tensor)> {
    let diag = backend.with_backend_session(|exec| exec.extract_diagonal(&u, 0, 1))?;
    let sign_u = backend.with_backend_session(|exec| {
        let sign = exec.sign_read(TensorRead::from_tensor(&diag))?;
        exec.reduce_prod_read(TensorRead::from_tensor(&sign), &[0])
    })?;
    let sign = backend.with_backend_session(|exec| {
        exec.mul_read(
            TensorRead::from_tensor(&parity),
            TensorRead::from_tensor(&sign_u),
        )
    })?;
    let logabsdet = backend.with_backend_session(|exec| {
        let abs = exec.abs_read(TensorRead::from_tensor(&diag))?;
        let log = exec.log_read(TensorRead::from_tensor(&abs))?;
        exec.reduce_sum_read(TensorRead::from_tensor(&log), &[0])
    })?;
    Ok((sign, logabsdet))
}
fn det_impl<B: LinalgBackend>(
    (sign, logabsdet): (Tensor, Tensor),
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|exec| {
        let magnitude = exec.exp_read(TensorRead::from_tensor(&logabsdet))?;
        exec.mul_read(
            TensorRead::from_tensor(&sign),
            TensorRead::from_tensor(&magnitude),
        )
    })
}
fn inv_owned<B: LinalgBackend>(a: &Tensor, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
    let eye = eye_like(a.dtype(), a.shape(), backend)?;
    backend.solve(a, &eye)
}
fn inv_read<B: LinalgBackend>(
    a: TensorRead<'_>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let eye = eye_like(a.dtype(), a.shape(), backend)?;
    backend.solve_read(a, TensorRead::from_tensor(&eye))
}
fn pinv_owned<B: LinalgBackend>(
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
fn pinv_read<B: LinalgBackend>(
    a: TensorRead<'_>,
    rtol: Option<f64>,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    ensure_float_or_complex("pinv", a.dtype())?;
    let default_rtol = default_pinv_rtol(a.dtype(), a.shape());
    let outputs = three(backend.svd_read(a)?, "pinv_read")?;
    pinv_from_svd(outputs, rtol.unwrap_or(default_rtol), backend)
}
fn norm_from_read<B: LinalgBackend>(
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
        return backend.with_backend_session(|exec| exec.to_contiguous_read(input));
    }
    validate_axes("norm", original_shape.len(), &axes)?;
    let reduced = if can_square_without_abs(input.dtype(), axes.len(), ord) {
        frobenius_norm_read(input.clone(), &axes, backend)?
    } else {
        let abs = backend.with_backend_session(|exec| exec.abs_read(input))?;
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

fn broadcast<B: LinalgBackend>(
    input: &Tensor,
    shape: &[usize],
    dims: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    if input.shape() == shape {
        return Ok(input.clone());
    }
    backend.with_backend_session(|exec| {
        exec.broadcast_in_dim_read(TensorRead::from_tensor(input), shape, dims)
    })
}

fn eye_like<B: LinalgBackend>(
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
    backend.with_backend_session(|exec| exec.embed_diagonal(&diagonal, 0, 1))
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

fn pinv_from_svd<B: LinalgBackend>(
    (u, s, vt): (Tensor, Tensor, Tensor),
    rtol: f64,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let abs_s = backend.with_backend_session(|exec| exec.abs_read(TensorRead::from_tensor(&s)))?;
    let s_max = reduce_max(&abs_s, &[0], backend)?;
    let threshold_scalar = scalar_real(abs_s.dtype(), rtol.max(0.0))?;
    let threshold = binary_mul(&s_max, &threshold_scalar, backend)?;
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, s.shape(), backend)?;
    let mask = backend.with_backend_session(|exec| {
        let mask = exec.compare_read(
            TensorRead::from_tensor(&abs_s),
            TensorRead::from_tensor(&threshold),
            &CompareDir::Gt,
        )?;
        exec.convert(&mask, s.dtype())
    })?;
    let ones = ones_like(&s, backend)?;
    let neg_mask =
        backend.with_backend_session(|exec| exec.neg_read(TensorRead::from_tensor(&mask)))?;
    let offset = binary_add(&ones, &neg_mask, backend)?;
    let denom = binary_add(&s, &offset, backend)?;
    let s_inv = backend.with_backend_session(|exec| {
        exec.div_read(
            TensorRead::from_tensor(&mask),
            TensorRead::from_tensor(&denom),
        )
    })?;
    let v = conjugate_transpose(&vt, backend)?;
    let uh = conjugate_transpose(&u, backend)?;
    let vs = scale_matrix_columns(&v, &s_inv, backend)?;
    matmul_preserve_trailing_batch(&vs, &uh, backend)
}

fn ones_like<B: LinalgBackend>(input: &Tensor, backend: &mut B) -> tenferro_tensor::Result<Tensor> {
    let one = scalar_real(input.dtype(), 1.0)?;
    broadcast(&one, input.shape(), &[], backend)
}

fn binary_add<B: LinalgBackend>(
    lhs: &Tensor,
    rhs: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|exec| {
        exec.add_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
    })
}

fn binary_mul<B: LinalgBackend>(
    lhs: &Tensor,
    rhs: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|exec| {
        exec.mul_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
    })
}

fn reduce_max<B: LinalgBackend>(
    input: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|exec| exec.reduce_max_read(TensorRead::from_tensor(input), axes))
}

fn reduce_sum<B: LinalgBackend>(
    input: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|exec| exec.reduce_sum_read(TensorRead::from_tensor(input), axes))
}

fn reduce_min<B: LinalgBackend>(
    input: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|exec| exec.reduce_min_read(TensorRead::from_tensor(input), axes))
}

fn reshape<B: LinalgBackend>(
    input: &Tensor,
    shape: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|exec| exec.reshape_read(TensorRead::from_tensor(input), shape))
}

fn transpose<B: LinalgBackend>(
    input: &Tensor,
    perm: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|exec| exec.transpose_read(TensorRead::from_tensor(input), perm))
}

fn broadcast_batch_scalar_to_leading_axis<B: LinalgBackend>(
    input: &Tensor,
    shape: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let dims: Vec<usize> = (1..shape.len()).collect();
    broadcast(input, shape, &dims, backend)
}

fn conjugate_transpose<B: LinalgBackend>(
    input: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let conj =
        backend.with_backend_session(|exec| exec.conj_read(TensorRead::from_tensor(input)))?;
    let mut perm: Vec<usize> = (0..input.shape().len()).collect();
    perm.swap(0, 1);
    transpose(&conj, &perm, backend)
}

fn scale_matrix_columns<B: LinalgBackend>(
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

fn matmul_preserve_trailing_batch<B: LinalgBackend>(
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
    backend.with_backend_session(|exec| {
        exec.dot_general_read(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            &config,
        )
    })
}

fn linalg_matmul_read<B: LinalgBackend>(
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
    backend.with_backend_session(|exec| {
        exec.dot_general_read(TensorRead::from_tensor(lhs), rhs, &config)
    })
}

fn frobenius_norm<B: LinalgBackend>(
    abs: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let squared = backend.with_backend_session(|exec| {
        exec.mul_read(TensorRead::from_tensor(abs), TensorRead::from_tensor(abs))
    })?;
    let sum = reduce_sum(&squared, axes, backend)?;
    backend.with_backend_session(|exec| exec.sqrt_read(TensorRead::from_tensor(&sum)))
}

fn frobenius_norm_read<B: LinalgBackend>(
    input: TensorRead<'_>,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let rhs = input.clone();
    let squared = backend.with_backend_session(|exec| exec.mul_read(input, rhs))?;
    let sum = reduce_sum(&squared, axes, backend)?;
    backend.with_backend_session(|exec| exec.sqrt_read(TensorRead::from_tensor(&sum)))
}

fn p_norm<B: LinalgBackend>(
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
    let powered = backend.with_backend_session(|exec| {
        exec.pow_read(
            TensorRead::from_tensor(abs),
            TensorRead::from_tensor(&power),
        )
    })?;
    let sum = reduce_sum(&powered, axes, backend)?;
    let inverse = scalar_real(abs.dtype(), 1.0 / p)?;
    backend.with_backend_session(|exec| {
        exec.pow_read(
            TensorRead::from_tensor(&sum),
            TensorRead::from_tensor(&inverse),
        )
    })
}

fn count_nonzero<B: LinalgBackend>(
    abs: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let zero = scalar_real(abs.dtype(), 0.0)?;
    let zero = broadcast(&zero, abs.shape(), &[], backend)?;
    let mask = backend.with_backend_session(|exec| {
        exec.compare_read(
            TensorRead::from_tensor(abs),
            TensorRead::from_tensor(&zero),
            &CompareDir::Gt,
        )
    })?;
    let mask = backend.with_backend_session(|exec| exec.convert(&mask, abs.dtype()))?;
    reduce_sum(&mask, axes, backend)
}

fn norm_over_axes<B: LinalgBackend>(
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

fn matrix_norm<B: LinalgBackend>(
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

fn row_sum_norm<B: LinalgBackend>(
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

fn col_sum_norm<B: LinalgBackend>(
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

fn move_axes_to_front<B: LinalgBackend>(
    input: &Tensor,
    axes: &[usize],
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        return Ok(input.clone());
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
