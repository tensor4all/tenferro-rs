//! Public concrete tensor einsum extension API.

use tenferro_tensor::{
    DType, DotGeneralAccumulation, Tensor, TensorBackend, TensorRead, TensorScalar, TensorWrite,
    TypedTensor, TypedTensorView, TypedTensorWrite,
};

use crate::eager::{
    eager_einsum_exec, eager_einsum_exec_read, eager_einsum_exec_read_into,
    eager_einsum_exec_read_into_accum, eager_einsum_read_subscripts, eager_einsum_subscripts,
    plan_subscripts,
};
use crate::{ContractionTree, EinsumSubscripts, Error, Result, Subscripts};

const TENSOR_EINSUM_OP: &str = "TensorEinsumExt::einsum";
const TENSOR_EINSUM_INTO_OP: &str = "TensorEinsumIntoExt::einsum_into";
const TENSOR_READ_EINSUM_OP: &str = "TensorReadEinsumExt::einsum_read";
const TENSOR_READ_EINSUM_INTO_OP: &str = "TensorReadEinsumIntoExt::einsum_read_into";
const TYPED_TENSOR_EINSUM_OP: &str = "TypedTensorEinsumExt::einsum";
const TYPED_TENSOR_EINSUM_INTO_OP: &str = "TypedTensorEinsumIntoExt::einsum_into";
const TYPED_TENSOR_READ_EINSUM_OP: &str = "TypedTensorReadEinsumExt::einsum_read";
const TYPED_TENSOR_READ_EINSUM_INTO_OP: &str = "TypedTensorReadEinsumIntoExt::einsum_read_into";
const PLAN_PREPARE_OP: &str = "ConcreteEinsumPlan::prepare";
const PLAN_EXECUTE_OP: &str = "ConcreteEinsumPlan::execute";

/// Backend-explicit einsum methods for dtype-erased concrete tensors.
///
/// Implementations are provided for slices and fixed-size arrays of
/// [`Tensor`] references, so both `inputs.as_slice().einsum(...)` and
/// `[&lhs, &rhs].einsum(...)` work.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TensorEinsumExt;
/// use tenferro_tensor::Tensor;
///
/// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
/// let mut backend = CpuBackend::new();
///
/// let out = [&lhs, &rhs].einsum("ij,jk->ik", &mut backend)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
pub trait TensorEinsumExt {
    /// Execute an einsum from string notation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with shape, rank, or dtype payloads for an invalid
    /// contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor>;

    /// Execute an einsum from parsed integer-label subscripts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with shape, rank, or dtype payloads for an
    /// invalid contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor>;
}

impl TensorEinsumExt for [&Tensor] {
    fn einsum<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor> {
        let subscripts = parse_subscripts(subscripts, TENSOR_EINSUM_OP)?;
        eager_einsum_subscripts(backend, self, &subscripts).map_err(Error::from)
    }

    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor> {
        let subscripts = Subscripts::from(subscripts);
        eager_einsum_subscripts(backend, self, &subscripts).map_err(Error::from)
    }
}

impl<const N: usize> TensorEinsumExt for [&Tensor; N] {
    fn einsum<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor> {
        self.as_slice().einsum(subscripts, backend)
    }

    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor> {
        self.as_slice().einsum_subscripts(subscripts, backend)
    }
}

/// Backend-explicit preallocated-output einsum methods for dtype-erased tensors.
pub trait TensorEinsumIntoExt {
    /// Execute an einsum from string notation into caller-provided output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with a shape, rank, or dtype payload when inputs or
    /// output do not match, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_into<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    /// Execute an einsum from parsed integer-label subscripts into caller-provided output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a shape, rank, or dtype payload when
    /// inputs or output do not match, or [`Error::Tensor`] for a typed backend
    /// failure.
    fn einsum_into_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()>;
}

impl TensorEinsumIntoExt for [&Tensor] {
    fn einsum_into<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let subscripts = parse_subscripts(subscripts, TENSOR_EINSUM_INTO_OP)?;
        tensor_einsum_into_subscripts(backend, self, &subscripts, out, TENSOR_EINSUM_INTO_OP)
    }

    fn einsum_into_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let subscripts = Subscripts::from(subscripts);
        tensor_einsum_into_subscripts(backend, self, &subscripts, out, TENSOR_EINSUM_INTO_OP)
    }
}

impl<const N: usize> TensorEinsumIntoExt for [&Tensor; N] {
    fn einsum_into<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice().einsum_into(subscripts, backend, out)
    }

    fn einsum_into_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice()
            .einsum_into_subscripts(subscripts, backend, out)
    }
}

/// Backend-explicit einsum methods for typed concrete tensors.
///
/// The result keeps the same scalar type as the inputs. Mixed dtypes should use
/// [`TensorEinsumExt`] on dtype-erased [`Tensor`] values instead.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TypedTensorEinsumExt;
/// use tenferro_tensor::TypedTensor;
///
/// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
/// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 4], vec![1.0; 12]).unwrap();
/// let mut backend = CpuBackend::new();
///
/// let out = [&lhs, &rhs].einsum("ij,jk->ik", &mut backend)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
pub trait TypedTensorEinsumExt<T: TensorScalar> {
    /// Execute an einsum from string notation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with shape or rank payloads for an invalid
    /// contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum<B: TensorBackend>(&self, subscripts: &str, backend: &mut B)
        -> Result<TypedTensor<T>>;

    /// Execute an einsum from parsed integer-label subscripts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with shape or rank payloads for an invalid
    /// contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>>;
}

impl<T: TensorScalar> TypedTensorEinsumExt<T> for [&TypedTensor<T>] {
    fn einsum<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        let subscripts = parse_subscripts(subscripts, TYPED_TENSOR_EINSUM_OP)?;
        typed_einsum_subscripts(backend, self, &subscripts, TYPED_TENSOR_EINSUM_OP)
    }

    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        let subscripts = Subscripts::from(subscripts);
        typed_einsum_subscripts(backend, self, &subscripts, TYPED_TENSOR_EINSUM_OP)
    }
}

impl<T: TensorScalar, const N: usize> TypedTensorEinsumExt<T> for [&TypedTensor<T>; N] {
    fn einsum<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum(subscripts, backend)
    }

    fn einsum_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_subscripts(subscripts, backend)
    }
}

/// Backend-explicit einsum methods for typed borrowed views.
///
/// The `_read` suffix distinguishes this borrowed-view surface from
/// [`TypedTensorEinsumExt`], whose unsuffixed methods accept only owned compact
/// typed tensors.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TypedTensorReadEinsumExt;
/// use tenferro_tensor::TypedTensor;
///
/// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
/// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
/// let mut backend = CpuBackend::new();
/// let result = [lhs.as_view(), rhs.as_view()].einsum_read("i,i->", &mut backend)?;
/// assert_eq!(result.as_slice()?, &[11.0]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
pub trait TypedTensorReadEinsumExt<T: TensorScalar> {
    /// Execute an einsum from string notation over typed borrowed views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::TypedTensorReadEinsumExt;
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0])?;
    /// let mut backend = CpuBackend::new();
    /// let result = [input.as_view()].einsum_read("i->i", &mut backend)?;
    /// assert_eq!(result.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with shape or rank payloads for incompatible
    /// views, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
    ) -> Result<TypedTensor<T>>;

    /// Execute an einsum from parsed integer-label subscripts over typed
    /// borrowed views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::{EinsumSubscripts, TypedTensorReadEinsumExt};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0])?;
    /// let subscripts = EinsumSubscripts::new(&[&[0]], &[0]);
    /// let mut backend = CpuBackend::new();
    /// let result = [input.as_view()].einsum_read_subscripts(&subscripts, &mut backend)?;
    /// assert_eq!(result.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with shape or rank payloads for
    /// incompatible views, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>>;
}

impl<'a, T: TensorScalar> TypedTensorReadEinsumExt<T> for [TypedTensorView<'a, T>] {
    fn einsum_read<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        let subscripts = parse_subscripts(subscripts, TYPED_TENSOR_READ_EINSUM_OP)?;
        typed_view_einsum_subscripts(backend, self, &subscripts, TYPED_TENSOR_READ_EINSUM_OP)
    }

    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        let subscripts = Subscripts::from(subscripts);
        typed_view_einsum_subscripts(backend, self, &subscripts, TYPED_TENSOR_READ_EINSUM_OP)
    }
}

impl<'a, T: TensorScalar, const N: usize> TypedTensorReadEinsumExt<T>
    for [TypedTensorView<'a, T>; N]
{
    fn einsum_read<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_read(subscripts, backend)
    }

    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_read_subscripts(subscripts, backend)
    }
}

/// Backend-explicit preallocated-output einsum methods for typed concrete tensors.
pub trait TypedTensorEinsumIntoExt<T: TensorScalar> {
    /// Execute an einsum from string notation into caller-provided typed output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with a shape, rank, or dtype payload when inputs or
    /// output do not match, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_into<'out, B, O>(&self, subscripts: &str, backend: &mut B, out: O) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>;

    /// Execute an einsum from parsed integer-label subscripts into caller-provided typed output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a shape, rank, or dtype payload when
    /// inputs or output do not match, or [`Error::Tensor`] for a typed backend
    /// failure.
    fn einsum_into_subscripts<'out, B, O>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: O,
    ) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>;
}

impl<T: TensorScalar> TypedTensorEinsumIntoExt<T> for [&TypedTensor<T>] {
    fn einsum_into<'out, B, O>(&self, subscripts: &str, backend: &mut B, out: O) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let subscripts = parse_subscripts(subscripts, TYPED_TENSOR_EINSUM_INTO_OP)?;
        typed_einsum_into_subscripts(
            backend,
            self,
            &subscripts,
            out.into(),
            TYPED_TENSOR_EINSUM_INTO_OP,
        )
    }

    fn einsum_into_subscripts<'out, B, O>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: O,
    ) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let subscripts = Subscripts::from(subscripts);
        typed_einsum_into_subscripts(
            backend,
            self,
            &subscripts,
            out.into(),
            TYPED_TENSOR_EINSUM_INTO_OP,
        )
    }
}

impl<T: TensorScalar, const N: usize> TypedTensorEinsumIntoExt<T> for [&TypedTensor<T>; N] {
    fn einsum_into<'out, B, O>(&self, subscripts: &str, backend: &mut B, out: O) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice().einsum_into(subscripts, backend, out)
    }

    fn einsum_into_subscripts<'out, B, O>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: O,
    ) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice()
            .einsum_into_subscripts(subscripts, backend, out)
    }
}

/// Backend-explicit preallocated-output einsum methods for typed borrowed views.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TypedTensorReadEinsumIntoExt;
/// use tenferro_tensor::TypedTensor;
///
/// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
/// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
/// let mut output = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0])?;
/// let mut backend = CpuBackend::new();
/// [lhs.as_view(), rhs.as_view()].einsum_read_into("i,i->", &mut backend, &mut output)?;
/// assert_eq!(output.as_slice()?, &[11.0]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
pub trait TypedTensorReadEinsumIntoExt<T: TensorScalar> {
    /// Execute an einsum from string notation over typed borrowed views into a
    /// caller-provided typed output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::TypedTensorReadEinsumIntoExt;
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0])?;
    /// let mut output = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0; 2])?;
    /// let mut backend = CpuBackend::new();
    /// [input.as_view()].einsum_read_into("i->i", &mut backend, &mut output)?;
    /// assert_eq!(output.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with a shape, rank, or dtype payload when inputs or
    /// output do not match, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read_into<'out, B, O>(&self, subscripts: &str, backend: &mut B, out: O) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>;

    /// Execute an einsum from parsed integer-label subscripts over typed
    /// borrowed views into a caller-provided typed output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::{EinsumSubscripts, TypedTensorReadEinsumIntoExt};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0])?;
    /// let mut output = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0; 2])?;
    /// let subscripts = EinsumSubscripts::new(&[&[0]], &[0]);
    /// let mut backend = CpuBackend::new();
    /// [input.as_view()].einsum_read_into_subscripts(&subscripts, &mut backend, &mut output)?;
    /// assert_eq!(output.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a shape, rank, or dtype payload when
    /// inputs or output do not match, or [`Error::Tensor`] for a typed backend
    /// failure.
    fn einsum_read_into_subscripts<'out, B, O>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: O,
    ) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>;
}

impl<'a, T: TensorScalar> TypedTensorReadEinsumIntoExt<T> for [TypedTensorView<'a, T>] {
    fn einsum_read_into<'out, B, O>(&self, subscripts: &str, backend: &mut B, out: O) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let subscripts = parse_subscripts(subscripts, TYPED_TENSOR_READ_EINSUM_INTO_OP)?;
        typed_view_einsum_into_subscripts(
            backend,
            self,
            &subscripts,
            out.into(),
            TYPED_TENSOR_READ_EINSUM_INTO_OP,
        )
    }

    fn einsum_read_into_subscripts<'out, B, O>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: O,
    ) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let subscripts = Subscripts::from(subscripts);
        typed_view_einsum_into_subscripts(
            backend,
            self,
            &subscripts,
            out.into(),
            TYPED_TENSOR_READ_EINSUM_INTO_OP,
        )
    }
}

impl<'a, T: TensorScalar, const N: usize> TypedTensorReadEinsumIntoExt<T>
    for [TypedTensorView<'a, T>; N]
{
    fn einsum_read_into<'out, B, O>(&self, subscripts: &str, backend: &mut B, out: O) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice().einsum_read_into(subscripts, backend, out)
    }

    fn einsum_read_into_subscripts<'out, B, O>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: O,
    ) -> Result<()>
    where
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice()
            .einsum_read_into_subscripts(subscripts, backend, out)
    }
}

/// Backend-explicit einsum methods for [`TensorRead`] inputs.
///
/// Use this surface when an input is a borrowed tensor view rather than an
/// owned compact [`Tensor`]. The `_read` suffix follows the repository-wide
/// convention for APIs that explicitly accept read-oriented borrowed inputs.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TensorReadEinsumExt;
/// use tenferro_tensor::{Tensor, TensorRead, TensorView};
///
/// let shape = [2, 3];
/// let data = [1.0_f64; 6];
/// let rhs = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
/// let inputs = [
///     TensorRead::from_view(TensorView::f64(&shape, &data)?),
///     TensorRead::from_tensor(&rhs),
/// ];
/// let mut backend = CpuBackend::new();
///
/// let out = inputs.einsum_read("ij,j->i", &mut backend)?;
/// assert_eq!(out.shape(), &[2]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
pub trait TensorReadEinsumExt {
    /// Execute an einsum from string notation over read-only tensor inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with shape, rank, or dtype payloads for an invalid
    /// contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor>;

    /// Execute an einsum from parsed integer-label subscripts over read-only
    /// tensor inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with shape, rank, or dtype payloads for an
    /// invalid contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor>;
}

impl<'a> TensorReadEinsumExt for [TensorRead<'a>] {
    fn einsum_read<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor> {
        let subscripts = parse_subscripts(subscripts, TENSOR_READ_EINSUM_OP)?;
        eager_einsum_read_subscripts(backend, self, &subscripts).map_err(Error::from)
    }

    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor> {
        let subscripts = Subscripts::from(subscripts);
        eager_einsum_read_subscripts(backend, self, &subscripts).map_err(Error::from)
    }
}

impl<'a, const N: usize> TensorReadEinsumExt for [TensorRead<'a>; N] {
    fn einsum_read<B: TensorBackend>(&self, subscripts: &str, backend: &mut B) -> Result<Tensor> {
        self.as_slice().einsum_read(subscripts, backend)
    }

    fn einsum_read_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
    ) -> Result<Tensor> {
        self.as_slice().einsum_read_subscripts(subscripts, backend)
    }
}

/// Backend-explicit preallocated-output einsum methods for [`TensorRead`] inputs.
pub trait TensorReadEinsumIntoExt {
    /// Execute an einsum from string notation over read-only inputs into caller-provided output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with a shape, rank, or dtype payload when inputs or
    /// output do not match, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read_into<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    /// Execute an einsum from parsed integer-label subscripts over read-only inputs into output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a shape, rank, or dtype payload when
    /// inputs or output do not match, or [`Error::Tensor`] for a typed backend
    /// failure.
    fn einsum_read_into_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()>;
}

impl<'a> TensorReadEinsumIntoExt for [TensorRead<'a>] {
    fn einsum_read_into<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let subscripts = parse_subscripts(subscripts, TENSOR_READ_EINSUM_INTO_OP)?;
        tensor_read_einsum_into_subscripts(
            backend,
            self,
            &subscripts,
            out,
            TENSOR_READ_EINSUM_INTO_OP,
        )
    }

    fn einsum_read_into_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let subscripts = Subscripts::from(subscripts);
        tensor_read_einsum_into_subscripts(
            backend,
            self,
            &subscripts,
            out,
            TENSOR_READ_EINSUM_INTO_OP,
        )
    }
}

impl<'a, const N: usize> TensorReadEinsumIntoExt for [TensorRead<'a>; N] {
    fn einsum_read_into<B: TensorBackend>(
        &self,
        subscripts: &str,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice().einsum_read_into(subscripts, backend, out)
    }

    fn einsum_read_into_subscripts<B: TensorBackend>(
        &self,
        subscripts: &EinsumSubscripts,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice()
            .einsum_read_into_subscripts(subscripts, backend, out)
    }
}

/// Prepared concrete einsum plan for repeated executions with fixed input
/// dtype and shape metadata.
///
/// Preparing a plan parses and optimizes the contraction tree once. Execution
/// validates the later inputs against the prepared dtype and shape contract,
/// then runs the stored tree without re-planning.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::ConcreteEinsumPlan;
/// use tenferro_tensor::Tensor;
///
/// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
/// let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
///
/// let mut backend = CpuBackend::new();
/// let out = plan.execute([&lhs, &rhs], &mut backend)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
#[derive(Debug)]
pub struct ConcreteEinsumPlan {
    tree: ContractionTree,
    inputs: Vec<ConcreteEinsumInputSpec>,
}

impl ConcreteEinsumPlan {
    /// Prepare a plan from dtype-erased concrete tensor inputs and string
    /// notation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] for rank, shape, or dtype contract violations, or
    /// [`Error::Planning`] when no valid contraction tree can be built.
    pub fn prepare<'a, I>(inputs: I, subscripts: &str) -> Result<Self>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let subscripts = parse_subscripts(subscripts, PLAN_PREPARE_OP)?;
        Self::prepare_subscripts_internal(input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from dtype-erased concrete tensor inputs and parsed
    /// integer-label subscripts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for rank, shape, or dtype contract
    /// violations, or [`Error::Planning`] when no valid contraction tree can be
    /// built.
    pub fn prepare_subscripts<'a, I>(inputs: I, subscripts: &EinsumSubscripts) -> Result<Self>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let subscripts = Subscripts::from(subscripts);
        Self::prepare_subscripts_internal(input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from typed concrete tensor inputs and string notation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] for rank or shape contract violations, or
    /// [`Error::Planning`] when no valid contraction tree can be built.
    pub fn prepare_typed<'a, T, I>(inputs: I, subscripts: &str) -> Result<Self>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
    {
        let subscripts = parse_subscripts(subscripts, PLAN_PREPARE_OP)?;
        Self::prepare_subscripts_internal(typed_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from typed concrete tensor inputs and parsed integer-label
    /// subscripts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for rank or shape contract violations, or
    /// [`Error::Planning`] when no valid contraction tree can be built.
    pub fn prepare_typed_subscripts<'a, T, I>(
        inputs: I,
        subscripts: &EinsumSubscripts,
    ) -> Result<Self>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
    {
        let subscripts = Subscripts::from(subscripts);
        Self::prepare_subscripts_internal(typed_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from read-only tensor inputs and string notation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] for rank, shape, or dtype contract violations, or
    /// [`Error::Planning`] when no valid contraction tree can be built.
    pub fn prepare_read<'a, I>(inputs: I, subscripts: &str) -> Result<Self>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let subscripts = parse_subscripts(subscripts, PLAN_PREPARE_OP)?;
        Self::prepare_subscripts_internal(read_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Prepare a plan from read-only tensor inputs and parsed integer-label
    /// subscripts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for rank, shape, or dtype contract
    /// violations, or [`Error::Planning`] when no valid contraction tree can be
    /// built.
    pub fn prepare_read_subscripts<'a, I>(inputs: I, subscripts: &EinsumSubscripts) -> Result<Self>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let subscripts = Subscripts::from(subscripts);
        Self::prepare_subscripts_internal(read_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Execute this plan on dtype-erased concrete tensor inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when inputs differ from the prepared
    /// rank, shape, or dtype contract, or [`Error::Tensor`] for a typed backend
    /// failure.
    pub fn execute<'a, I, B>(&self, inputs: I, backend: &mut B) -> Result<Tensor>
    where
        I: AsRef<[&'a Tensor]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        self.validate_inputs(&input_specs(inputs), PLAN_EXECUTE_OP)?;
        backend
            .with_backend_session(|exec| eager_einsum_exec(exec, inputs, &self.tree))
            .map_err(Error::from)
    }

    /// Execute this plan on typed concrete tensor inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when inputs differ from the prepared rank
    /// or shape contract, or [`Error::Tensor`] for a typed backend failure.
    pub fn execute_typed<'a, T, I, B>(&self, inputs: I, backend: &mut B) -> Result<TypedTensor<T>>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        self.validate_inputs(&typed_input_specs(inputs), PLAN_EXECUTE_OP)?;
        let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
        let result = backend
            .with_backend_session(|exec| eager_einsum_exec_read(exec, &reads, &self.tree))?;
        into_typed_result(result, PLAN_EXECUTE_OP)
    }

    /// Execute this plan on read-only tensor inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when inputs differ from the prepared
    /// rank, shape, or dtype contract, or [`Error::Tensor`] for a typed backend
    /// failure.
    pub fn execute_read<'a, I, B>(&self, inputs: I, backend: &mut B) -> Result<Tensor>
    where
        I: AsRef<[TensorRead<'a>]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        self.validate_inputs(&read_input_specs(inputs), PLAN_EXECUTE_OP)?;
        backend
            .with_backend_session(|exec| eager_einsum_exec_read(exec, inputs, &self.tree))
            .map_err(Error::from)
    }

    /// Execute this plan on dtype-erased concrete tensor inputs into caller-provided output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for input or output rank, shape, or dtype
    /// mismatches, or [`Error::Tensor`] for a typed backend failure.
    pub fn execute_into<'a, I, B>(
        &self,
        inputs: I,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[&'a Tensor]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        let specs = input_specs(inputs);
        self.validate_inputs(&specs, PLAN_EXECUTE_OP)?;
        validate_output(&self.inputs, &self.tree, &out, PLAN_EXECUTE_OP)?;
        let reads: Vec<_> = inputs
            .iter()
            .map(|tensor| TensorRead::from_tensor(tensor))
            .collect();
        backend
            .with_backend_session(|exec| eager_einsum_exec_read_into(exec, &reads, &self.tree, out))
            .map_err(Error::from)
    }

    /// Execute this plan on typed concrete tensor inputs into caller-provided output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for input or output rank or shape
    /// mismatches, or [`Error::Tensor`] for a typed backend failure.
    pub fn execute_typed_into<'a, 'out, T, I, B, O>(
        &self,
        inputs: I,
        backend: &mut B,
        out: O,
    ) -> Result<()>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
        B: TensorBackend,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let inputs = inputs.as_ref();
        let specs = typed_input_specs(inputs);
        self.validate_inputs(&specs, PLAN_EXECUTE_OP)?;
        let out = out.into().into_tensor_write();
        validate_output(&self.inputs, &self.tree, &out, PLAN_EXECUTE_OP)?;
        let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
        backend
            .with_backend_session(|exec| eager_einsum_exec_read_into(exec, &reads, &self.tree, out))
            .map_err(Error::from)
    }

    /// Execute this plan on read-only tensor inputs into caller-provided output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for input or output rank, shape, or dtype
    /// mismatches, or [`Error::Tensor`] for a typed backend failure.
    pub fn execute_read_into<'a, I, B>(
        &self,
        inputs: I,
        backend: &mut B,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[TensorRead<'a>]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        let specs = read_input_specs(inputs);
        self.validate_inputs(&specs, PLAN_EXECUTE_OP)?;
        validate_output(&self.inputs, &self.tree, &out, PLAN_EXECUTE_OP)?;
        backend
            .with_backend_session(|exec| eager_einsum_exec_read_into(exec, inputs, &self.tree, out))
            .map_err(Error::from)
    }

    /// Execute this plan on read-only inputs with scaled output accumulation.
    ///
    /// `accumulation` follows the dot-general contract:
    /// `out = alpha * einsum(inputs) + beta * out`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::{
    ///     DotGeneralAccumulation, DType, Tensor, TensorRead, TensorWrite,
    /// };
    ///
    /// let lhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// let mut out = Tensor::from_vec_col_major(vec![], vec![1.0_f64])?;
    /// let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "i,i->")?;
    /// let mut backend = CpuBackend::new();
    /// plan.execute_read_into_accum(
    ///     [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
    ///     &mut backend,
    ///     DotGeneralAccumulation::add_to(DType::F64)?,
    ///     TensorWrite::from_tensor(&mut out),
    /// )?;
    /// assert_eq!(out.as_slice::<f64>()?, &[7.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for input or output rank, shape, or dtype
    /// mismatches, [`Error::Numerical`] for an invalid accumulation, or
    /// [`Error::Tensor`] for a typed backend failure.
    pub fn execute_read_into_accum<'a, I, B>(
        &self,
        inputs: I,
        backend: &mut B,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[TensorRead<'a>]>,
        B: TensorBackend,
    {
        let inputs = inputs.as_ref();
        let specs = read_input_specs(inputs);
        self.validate_inputs(&specs, PLAN_EXECUTE_OP)?;
        validate_output(&self.inputs, &self.tree, &out, PLAN_EXECUTE_OP)?;
        backend
            .with_backend_session(|exec| {
                eager_einsum_exec_read_into_accum(exec, inputs, &self.tree, accumulation, out)
            })
            .map_err(Error::from)
    }

    fn prepare_subscripts_internal(
        inputs: Vec<ConcreteEinsumInputSpec>,
        subscripts: &Subscripts,
    ) -> Result<Self> {
        let shapes: Vec<&[usize]> = inputs.iter().map(|input| input.shape.as_slice()).collect();
        let tree = plan_subscripts(subscripts, &shapes)?;
        Ok(Self { tree, inputs })
    }

    fn validate_inputs(&self, actual: &[ConcreteEinsumInputSpec], op: &'static str) -> Result<()> {
        if actual.len() != self.inputs.len() {
            return Err(Error::invalid_argument(
                op,
                "inputs",
                format!(
                    "prepared einsum expects {} inputs, got {}",
                    self.inputs.len(),
                    actual.len()
                ),
            ));
        }

        for (expected, actual) in self.inputs.iter().zip(actual.iter()) {
            if expected.dtype != actual.dtype {
                return Err(Error::dtype_mismatch(op, expected.dtype, actual.dtype));
            }
            if expected.shape != actual.shape {
                return Err(Error::shape_mismatch(
                    op,
                    expected.shape.clone(),
                    actual.shape.clone(),
                ));
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ConcreteEinsumInputSpec {
    dtype: DType,
    shape: Vec<usize>,
}

fn parse_subscripts(subscripts: &str, _op: &'static str) -> Result<Subscripts> {
    Subscripts::parse(subscripts)
}

fn input_specs(inputs: &[&Tensor]) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

fn typed_input_specs<T: TensorScalar>(inputs: &[&TypedTensor<T>]) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: T::dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

fn typed_view_input_specs<T: TensorScalar>(
    inputs: &[TypedTensorView<'_, T>],
) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: T::dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

fn read_input_specs(inputs: &[TensorRead<'_>]) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

fn typed_view_einsum_subscripts<T: TensorScalar>(
    backend: &mut impl TensorBackend,
    inputs: &[TypedTensorView<'_, T>],
    subscripts: &Subscripts,
    op: &'static str,
) -> Result<TypedTensor<T>> {
    let reads: Vec<_> = inputs
        .iter()
        .cloned()
        .map(|view| TensorRead::from_view(T::tensor_view(view)))
        .collect();
    let result = eager_einsum_read_subscripts(backend, &reads, subscripts)?;
    into_typed_result(result, op)
}

fn tensor_einsum_into_subscripts(
    backend: &mut impl TensorBackend,
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    out: TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    let specs = input_specs(inputs);
    let plan = ConcreteEinsumPlan::prepare_subscripts_internal(specs.clone(), subscripts)?;
    validate_output(&specs, &plan.tree, &out, op)?;
    let reads: Vec<_> = inputs
        .iter()
        .map(|tensor| TensorRead::from_tensor(tensor))
        .collect();
    backend
        .with_backend_session(|exec| eager_einsum_exec_read_into(exec, &reads, &plan.tree, out))
        .map_err(Error::from)
}

fn typed_view_einsum_into_subscripts<T: TensorScalar>(
    backend: &mut impl TensorBackend,
    inputs: &[TypedTensorView<'_, T>],
    subscripts: &Subscripts,
    out: TypedTensorWrite<'_, T>,
    op: &'static str,
) -> Result<()> {
    let specs = typed_view_input_specs(inputs);
    let plan = ConcreteEinsumPlan::prepare_subscripts_internal(specs.clone(), subscripts)?;
    let out = out.into_tensor_write();
    validate_output(&specs, &plan.tree, &out, op)?;
    let reads: Vec<_> = inputs
        .iter()
        .cloned()
        .map(|view| TensorRead::from_view(T::tensor_view(view)))
        .collect();
    backend
        .with_backend_session(|exec| eager_einsum_exec_read_into(exec, &reads, &plan.tree, out))
        .map_err(Error::from)
}

fn typed_einsum_into_subscripts<T: TensorScalar>(
    backend: &mut impl TensorBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &Subscripts,
    out: TypedTensorWrite<'_, T>,
    op: &'static str,
) -> Result<()> {
    let specs = typed_input_specs(inputs);
    let plan = ConcreteEinsumPlan::prepare_subscripts_internal(specs.clone(), subscripts)?;
    let out = out.into_tensor_write();
    validate_output(&specs, &plan.tree, &out, op)?;
    let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
    backend
        .with_backend_session(|exec| eager_einsum_exec_read_into(exec, &reads, &plan.tree, out))
        .map_err(Error::from)
}

fn tensor_read_einsum_into_subscripts(
    backend: &mut impl TensorBackend,
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
    out: TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    let specs = read_input_specs(inputs);
    let plan = ConcreteEinsumPlan::prepare_subscripts_internal(specs.clone(), subscripts)?;
    validate_output(&specs, &plan.tree, &out, op)?;
    backend
        .with_backend_session(|exec| eager_einsum_exec_read_into(exec, inputs, &plan.tree, out))
        .map_err(Error::from)
}

fn validate_output(
    inputs: &[ConcreteEinsumInputSpec],
    tree: &ContractionTree,
    out: &TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    let expected = output_spec(inputs, tree, op)?;
    if out.dtype() != expected.dtype {
        return Err(Error::dtype_mismatch(op, expected.dtype, out.dtype()));
    }
    if out.shape() != expected.shape.as_slice() {
        return Err(Error::shape_mismatch(
            op,
            out.shape().to_vec(),
            expected.shape.clone(),
        ));
    }
    Ok(())
}

fn output_spec(
    inputs: &[ConcreteEinsumInputSpec],
    tree: &ContractionTree,
    op: &'static str,
) -> Result<ConcreteEinsumInputSpec> {
    let dtype = inputs
        .first()
        .ok_or_else(|| {
            Error::invalid_argument(op, "inputs", "einsum requires at least one input tensor")
        })?
        .dtype;
    for input in inputs {
        if input.dtype != dtype {
            return Err(Error::dtype_mismatch(op, dtype, input.dtype));
        }
    }

    let mut output_shape = Vec::with_capacity(tree.subscripts.output.len());
    for &label in &tree.subscripts.output {
        let mut found = None;
        for (input, labels) in inputs.iter().zip(tree.subscripts.inputs.iter()) {
            if labels.len() != input.shape.len() {
                return Err(Error::rank_mismatch(op, labels.len(), input.shape.len()));
            }
            if let Some(axis) = labels.iter().position(|candidate| *candidate == label) {
                found = Some(input.shape[axis]);
                break;
            }
        }
        let Some(extent) = found else {
            return Err(Error::invalid_argument(
                op,
                "output labels",
                format!("output label {label} is missing from inputs"),
            ));
        };
        output_shape.push(extent);
    }
    Ok(ConcreteEinsumInputSpec {
        dtype,
        shape: output_shape,
    })
}

fn typed_einsum_subscripts<T: TensorScalar>(
    backend: &mut impl TensorBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &Subscripts,
    op: &'static str,
) -> Result<TypedTensor<T>> {
    let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
    let result = eager_einsum_read_subscripts(backend, &reads, subscripts)?;
    into_typed_result(result, op)
}

pub(crate) fn into_typed_result<T: TensorScalar>(
    result: Tensor,
    op: &'static str,
) -> Result<TypedTensor<T>> {
    let actual = result.dtype();
    T::into_typed(result).map_err(|_| Error::dtype_mismatch(op, T::dtype(), actual))
}
