//! Public concrete tensor einsum extension API.

use smallvec::SmallVec;
use tenferro_tensor::{
    BackendSession, DType, DotGeneralAccumulation, Tensor, TensorRead, TensorScalar, TensorWrite,
    TypedTensor, TypedTensorView, TypedTensorWrite,
};

use crate::binary_dot::BinaryDotOperandOrder;
use crate::eager::{
    binary_dot_config_for_into, binary_dot_plan_for_shapes, eager_einsum_exec,
    eager_einsum_exec_read, eager_einsum_exec_read_into, eager_einsum_exec_read_into_accum,
    eager_einsum_read_subscripts_on_session, eager_einsum_subscripts_on_session,
    execute_binary_dot_read_into, plan_subscripts,
};
use crate::ellipsis::resolve_einsum_notation;
use crate::TensorDotAxes;
use crate::{
    parse_einsum_notation, ContractionTree, EinsumNotation, EinsumSubscripts, Error, Result,
    Subscripts,
};

const TENSOR_EINSUM_INTO_OP: &str = "TensorEinsumIntoExt::einsum_into";
const TENSOR_READ_EINSUM_INTO_OP: &str = "TensorReadEinsumIntoExt::einsum_read_into";
const TYPED_TENSOR_EINSUM_OP: &str = "TypedTensorEinsumExt::einsum";
const TYPED_TENSOR_EINSUM_INTO_OP: &str = "TypedTensorEinsumIntoExt::einsum_into";
const TYPED_TENSOR_READ_EINSUM_OP: &str = "TypedTensorReadEinsumExt::einsum_read";
const TYPED_TENSOR_READ_EINSUM_INTO_OP: &str = "TypedTensorReadEinsumIntoExt::einsum_read_into";
const PLAN_EXECUTE_OP: &str = "ConcreteEinsumPlan::execute";
const TYPED_TENSOR_TENSORDOT_OP: &str = "TypedTensorTensordotExt::tensordot";

/// Backend-explicit tensordot sugar for dtype-erased concrete tensors.
pub trait TensorTensordotExt {
    /// Contract this tensor with `rhs` over explicit axes or an axis count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument`, `RankMismatch`,
    /// `AxisOutOfBounds`, or `ShapeMismatch` for invalid axes or incompatible
    /// contracting dimensions, or [`Error::Tensor`] when backend execution
    /// fails.
    fn tensordot(
        &self,
        rhs: &Tensor,
        axes: TensorDotAxes<'_>,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor>;
}

impl TensorTensordotExt for Tensor {
    fn tensordot(
        &self,
        rhs: &Tensor,
        axes: TensorDotAxes<'_>,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let config =
            crate::tensordot::dot_general_config(axes, self.shape().len(), rhs.shape().len())?;
        crate::tensordot::validate_concrete_contract_dims(self.shape(), rhs.shape(), &config)?;
        session
            .dot_general_read(
                TensorRead::from_tensor(self),
                TensorRead::from_tensor(rhs),
                &config,
            )
            .map_err(Error::from)
    }
}

/// Backend-explicit tensordot sugar for typed concrete tensors.
pub trait TypedTensorTensordotExt<T: TensorScalar> {
    /// Contract this tensor with `rhs` while preserving its scalar type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument`, `RankMismatch`,
    /// `AxisOutOfBounds`, or `ShapeMismatch` for invalid axes or incompatible
    /// contracting dimensions, or [`Error::Tensor`] when backend execution
    /// fails.
    fn tensordot(
        &self,
        rhs: &TypedTensor<T>,
        axes: TensorDotAxes<'_>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>>;
}

impl<T: TensorScalar> TypedTensorTensordotExt<T> for TypedTensor<T> {
    fn tensordot(
        &self,
        rhs: &TypedTensor<T>,
        axes: TensorDotAxes<'_>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let config =
            crate::tensordot::dot_general_config(axes, self.shape().len(), rhs.shape().len())?;
        crate::tensordot::validate_concrete_contract_dims(self.shape(), rhs.shape(), &config)?;
        let result = session
            .dot_general_read(T::tensor_read(self), T::tensor_read(rhs), &config)
            .map_err(Error::from)?;
        into_typed_result(result, TYPED_TENSOR_TENSORDOT_OP)
    }
}

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
/// use tenferro_tensor::{BackendSessionHost, Tensor};
///
/// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
/// let mut backend = CpuBackend::new();
///
/// let out = backend.with_backend_session(|session| {
///     [&lhs, &rhs].einsum("ij,jk->ik", session)
/// })?;
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
    fn einsum(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<Tensor>;

    /// Execute an einsum from rank-unresolved string/programmatic notation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation or backend error when notation, shapes, or execution are invalid.
    fn einsum_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor>;

    /// Execute an einsum from parsed integer-label subscripts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with shape, rank, or dtype payloads for an
    /// invalid contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor>;
}

impl TensorEinsumExt for [&Tensor] {
    fn einsum(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<Tensor> {
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_notation(&notation, session)
    }

    fn einsum_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let subscripts = resolve_tensor_notation(self, notation)?;
        eager_einsum_subscripts_on_session(session, self, &subscripts).map_err(Error::from)
    }

    fn einsum_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let subscripts = Subscripts::from(subscripts);
        eager_einsum_subscripts_on_session(session, self, &subscripts).map_err(Error::from)
    }
}

impl<const N: usize> TensorEinsumExt for [&Tensor; N] {
    fn einsum(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<Tensor> {
        self.as_slice().einsum(subscripts, session)
    }

    fn einsum_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        self.as_slice().einsum_notation(notation, session)
    }

    fn einsum_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        self.as_slice().einsum_subscripts(subscripts, session)
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
    fn einsum_into(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    /// Execute an einsum from rank-unresolved notation into caller-provided output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation or backend error when notation, shapes, or output are invalid.
    fn einsum_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    /// Execute an einsum from parsed integer-label subscripts into caller-provided output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a shape, rank, or dtype payload when
    /// inputs or output do not match, or [`Error::Tensor`] for a typed backend
    /// failure.
    fn einsum_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()>;
}

impl TensorEinsumIntoExt for [&Tensor] {
    fn einsum_into(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        if let ([lhs, rhs], Some((a, b, c))) = (self, parse_fast_ascii_binary_labels(subscripts)) {
            let reads = [TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs)];
            if let Some((order, config)) = read_binary_dot_config_for_labels(&reads, a, b, c, &out)
            {
                return execute_binary_dot_config_read_into(session, &reads, order, &config, out);
            }
        }
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_into_notation(&notation, session, out)
    }

    fn einsum_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let subscripts = resolve_tensor_notation(self, notation)?;
        tensor_einsum_into_subscripts(session, self, &subscripts, out, TENSOR_EINSUM_INTO_OP)
    }

    fn einsum_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let subscripts = Subscripts::from(subscripts);
        tensor_einsum_into_subscripts(session, self, &subscripts, out, TENSOR_EINSUM_INTO_OP)
    }
}

impl<const N: usize> TensorEinsumIntoExt for [&Tensor; N] {
    fn einsum_into(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice().einsum_into(subscripts, session, out)
    }

    fn einsum_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice().einsum_into_notation(notation, session, out)
    }

    fn einsum_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice()
            .einsum_into_subscripts(subscripts, session, out)
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
/// use tenferro_tensor::{BackendSessionHost, TypedTensor};
///
/// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
/// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 4], vec![1.0; 12]).unwrap();
/// let mut backend = CpuBackend::new();
///
/// let out = backend.with_backend_session(|session| {
///     [&lhs, &rhs].einsum("ij,jk->ik", session)
/// })?;
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
    fn einsum(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<TypedTensor<T>>;

    /// Execute an einsum from rank-unresolved notation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation or backend error when notation, shapes, or execution are invalid.
    fn einsum_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>>;

    /// Execute an einsum from parsed integer-label subscripts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with shape or rank payloads for an invalid
    /// contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>>;
}

impl<T: TensorScalar> TypedTensorEinsumExt<T> for [&TypedTensor<T>] {
    fn einsum(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_notation(&notation, session)
    }

    fn einsum_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let subscripts = resolve_typed_notation(self, notation)?;
        typed_einsum_subscripts(session, self, &subscripts, TYPED_TENSOR_EINSUM_OP)
    }

    fn einsum_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let subscripts = Subscripts::from(subscripts);
        typed_einsum_subscripts(session, self, &subscripts, TYPED_TENSOR_EINSUM_OP)
    }
}

impl<T: TensorScalar, const N: usize> TypedTensorEinsumExt<T> for [&TypedTensor<T>; N] {
    fn einsum(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        self.as_slice().einsum(subscripts, session)
    }

    fn einsum_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_notation(notation, session)
    }

    fn einsum_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_subscripts(subscripts, session)
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
/// use tenferro_tensor::{BackendSessionHost, TypedTensor};
///
/// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
/// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
/// let mut backend = CpuBackend::new();
/// let result = backend.with_backend_session(|session| {
///     [lhs.as_view(), rhs.as_view()].einsum_read("i,i->", session)
/// })?;
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
    /// use tenferro_tensor::{BackendSessionHost, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0])?;
    /// let mut backend = CpuBackend::new();
    /// let result = backend.with_backend_session(|session| {
    ///     [input.as_view()].einsum_read("i->i", session)
    /// })?;
    /// assert_eq!(result.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with shape or rank payloads for incompatible
    /// views, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>>;

    /// Execute an einsum from rank-unresolved notation over typed views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation or backend error when notation, views, or execution are invalid.
    fn einsum_read_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>>;

    /// Execute an einsum from parsed integer-label subscripts over typed
    /// borrowed views.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::{EinsumSubscripts, TypedTensorReadEinsumExt};
    /// use tenferro_tensor::{BackendSessionHost, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0])?;
    /// let subscripts = EinsumSubscripts::new(&[&[0]], &[0]);
    /// let mut backend = CpuBackend::new();
    /// let result = backend.with_backend_session(|session| {
    ///     [input.as_view()].einsum_read_subscripts(&subscripts, session)
    /// })?;
    /// assert_eq!(result.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with shape or rank payloads for
    /// incompatible views, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>>;
}

impl<'a, T: TensorScalar> TypedTensorReadEinsumExt<T> for [TypedTensorView<'a, T>] {
    fn einsum_read(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_read_notation(&notation, session)
    }

    fn einsum_read_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let subscripts = resolve_view_notation(self, notation)?;
        typed_view_einsum_subscripts(session, self, &subscripts, TYPED_TENSOR_READ_EINSUM_OP)
    }

    fn einsum_read_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let subscripts = Subscripts::from(subscripts);
        typed_view_einsum_subscripts(session, self, &subscripts, TYPED_TENSOR_READ_EINSUM_OP)
    }
}

impl<'a, T: TensorScalar, const N: usize> TypedTensorReadEinsumExt<T>
    for [TypedTensorView<'a, T>; N]
{
    fn einsum_read(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_read(subscripts, session)
    }

    fn einsum_read_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_read_notation(notation, session)
    }

    fn einsum_read_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        self.as_slice().einsum_read_subscripts(subscripts, session)
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
    fn einsum_into<'out, O>(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>;

    /// Execute an einsum from rank-unresolved notation into typed output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation or backend error when notation, shapes, or output are invalid.
    fn einsum_into_notation<'out, O>(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>;

    /// Execute an einsum from parsed integer-label subscripts into caller-provided typed output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a shape, rank, or dtype payload when
    /// inputs or output do not match, or [`Error::Tensor`] for a typed backend
    /// failure.
    fn einsum_into_subscripts<'out, O>(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>;
}

impl<T: TensorScalar> TypedTensorEinsumIntoExt<T> for [&TypedTensor<T>] {
    fn einsum_into<'out, O>(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let out = out.into().into_tensor_write();
        if let ([lhs, rhs], Some((a, b, c))) = (self, parse_fast_ascii_binary_labels(subscripts)) {
            let reads = [T::tensor_read(lhs), T::tensor_read(rhs)];
            if let Some((order, config)) = read_binary_dot_config_for_labels(&reads, a, b, c, &out)
            {
                return execute_binary_dot_config_read_into(session, &reads, order, &config, out);
            }
        }
        let notation = parse_einsum_notation(subscripts)?;
        let subscripts = resolve_typed_notation(self, &notation)?;
        typed_einsum_into_subscripts(session, self, &subscripts, out, TYPED_TENSOR_EINSUM_INTO_OP)
    }

    fn einsum_into_notation<'out, O>(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let subscripts = resolve_typed_notation(self, notation)?;
        typed_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out.into().into_tensor_write(),
            TYPED_TENSOR_EINSUM_INTO_OP,
        )
    }

    fn einsum_into_subscripts<'out, O>(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let subscripts = Subscripts::from(subscripts);
        typed_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out.into().into_tensor_write(),
            TYPED_TENSOR_EINSUM_INTO_OP,
        )
    }
}

impl<T: TensorScalar, const N: usize> TypedTensorEinsumIntoExt<T> for [&TypedTensor<T>; N] {
    fn einsum_into<'out, O>(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice().einsum_into(subscripts, session, out)
    }

    fn einsum_into_notation<'out, O>(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice().einsum_into_notation(notation, session, out)
    }

    fn einsum_into_subscripts<'out, O>(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice()
            .einsum_into_subscripts(subscripts, session, out)
    }
}

/// Backend-explicit preallocated-output einsum methods for typed borrowed views.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::TypedTensorReadEinsumIntoExt;
/// use tenferro_tensor::{BackendSessionHost, TypedTensor};
///
/// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
/// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0])?;
/// let mut output = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0])?;
/// let mut backend = CpuBackend::new();
/// backend.with_backend_session(|session| {
///     [lhs.as_view(), rhs.as_view()].einsum_read_into("i,i->", session, &mut output)
/// })?;
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
    /// use tenferro_tensor::{BackendSessionHost, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0])?;
    /// let mut output = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0; 2])?;
    /// let mut backend = CpuBackend::new();
    /// backend.with_backend_session(|session| {
    ///     [input.as_view()].einsum_read_into("i->i", session, &mut output)
    /// })?;
    /// assert_eq!(output.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with a shape, rank, or dtype payload when inputs or
    /// output do not match, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read_into<'out, O>(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>;

    /// Execute an einsum from rank-unresolved notation over typed views into output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation or backend error when notation, views, or output are invalid.
    fn einsum_read_into_notation<'out, O>(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>;

    /// Execute an einsum from parsed integer-label subscripts over typed
    /// borrowed views into a caller-provided typed output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::{EinsumSubscripts, TypedTensorReadEinsumIntoExt};
    /// use tenferro_tensor::{BackendSessionHost, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0])?;
    /// let mut output = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0; 2])?;
    /// let subscripts = EinsumSubscripts::new(&[&[0]], &[0]);
    /// let mut backend = CpuBackend::new();
    /// backend.with_backend_session(|session| {
    ///     [input.as_view()].einsum_read_into_subscripts(&subscripts, session, &mut output)
    /// })?;
    /// assert_eq!(output.as_slice()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a shape, rank, or dtype payload when
    /// inputs or output do not match, or [`Error::Tensor`] for a typed backend
    /// failure.
    fn einsum_read_into_subscripts<'out, O>(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>;
}

impl<'a, T: TensorScalar> TypedTensorReadEinsumIntoExt<T> for [TypedTensorView<'a, T>] {
    fn einsum_read_into<'out, O>(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let out = out.into().into_tensor_write();
        if let Some((lhs, rhs, output)) = parse_fast_ascii_binary_labels(subscripts) {
            if let Some((order, config)) =
                typed_view_binary_dot_config(self, lhs, rhs, output, &out)
            {
                return execute_typed_view_binary_dot_into(session, self, order, &config, out);
            }
        }
        let notation = parse_einsum_notation(subscripts)?;
        let subscripts = resolve_view_notation(self, &notation)?;
        typed_view_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out,
            TYPED_TENSOR_READ_EINSUM_INTO_OP,
        )
    }

    fn einsum_read_into_notation<'out, O>(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let out = out.into().into_tensor_write();
        if let Some([lhs, rhs, output]) = borrowed_notation_labels(notation) {
            if let Some((order, config)) =
                typed_view_binary_dot_config(self, &lhs, &rhs, &output, &out)
            {
                return execute_typed_view_binary_dot_into(session, self, order, &config, out);
            }
        }
        let subscripts = resolve_view_notation(self, notation)?;
        typed_view_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out,
            TYPED_TENSOR_READ_EINSUM_INTO_OP,
        )
    }

    fn einsum_read_into_subscripts<'out, O>(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let out = out.into().into_tensor_write();
        if let [lhs, rhs] = subscripts.inputs.as_slice() {
            if let Some((order, config)) =
                typed_view_binary_dot_config(self, lhs, rhs, &subscripts.output, &out)
            {
                return execute_typed_view_binary_dot_into(session, self, order, &config, out);
            }
        }
        let subscripts = Subscripts::from(subscripts);
        typed_view_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out,
            TYPED_TENSOR_READ_EINSUM_INTO_OP,
        )
    }
}

impl<'a, T: TensorScalar, const N: usize> TypedTensorReadEinsumIntoExt<T>
    for [TypedTensorView<'a, T>; N]
{
    fn einsum_read_into<'out, O>(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice().einsum_read_into(subscripts, session, out)
    }

    fn einsum_read_into_notation<'out, O>(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice()
            .einsum_read_into_notation(notation, session, out)
    }

    fn einsum_read_into_subscripts<'out, O>(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        O: Into<TypedTensorWrite<'out, T>>,
    {
        self.as_slice()
            .einsum_read_into_subscripts(subscripts, session, out)
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
/// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorView};
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
/// let out = backend.with_backend_session(|session| inputs.einsum_read("ij,j->i", session))?;
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
    fn einsum_read(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<Tensor>;

    /// Execute an einsum from rank-unresolved notation over read-only inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation or backend error when notation, shapes, or execution are invalid.
    fn einsum_read_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor>;

    /// Execute an einsum from parsed integer-label subscripts over read-only
    /// tensor inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with shape, rank, or dtype payloads for an
    /// invalid contraction, or [`Error::Tensor`] for a typed backend failure.
    fn einsum_read_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor>;
}

impl<'a> TensorReadEinsumExt for [TensorRead<'a>] {
    fn einsum_read(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<Tensor> {
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_read_notation(&notation, session)
    }

    fn einsum_read_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let subscripts = resolve_read_notation(self, notation)?;
        eager_einsum_read_subscripts_on_session(session, self, &subscripts).map_err(Error::from)
    }

    fn einsum_read_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let subscripts = Subscripts::from(subscripts);
        eager_einsum_read_subscripts_on_session(session, self, &subscripts).map_err(Error::from)
    }
}

impl<'a, const N: usize> TensorReadEinsumExt for [TensorRead<'a>; N] {
    fn einsum_read(&self, subscripts: &str, session: &mut dyn BackendSession) -> Result<Tensor> {
        self.as_slice().einsum_read(subscripts, session)
    }

    fn einsum_read_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        self.as_slice().einsum_read_notation(notation, session)
    }

    fn einsum_read_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        self.as_slice().einsum_read_subscripts(subscripts, session)
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
    fn einsum_read_into(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    /// Execute an einsum from rank-unresolved notation over read-only inputs into output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation or backend error when notation, shapes, or output are invalid.
    fn einsum_read_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    /// Execute an einsum from parsed integer-label subscripts over read-only inputs into output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with a shape, rank, or dtype payload when
    /// inputs or output do not match, or [`Error::Tensor`] for a typed backend
    /// failure.
    fn einsum_read_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()>;
}

impl<'a> TensorReadEinsumIntoExt for [TensorRead<'a>] {
    fn einsum_read_into(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        if let Some((lhs, rhs, output)) = parse_fast_ascii_binary_labels(subscripts) {
            if let Some((order, config)) =
                read_binary_dot_config_for_labels(self, lhs, rhs, output, &out)
            {
                return execute_binary_dot_config_read_into(session, self, order, &config, out);
            }
        }
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_read_into_notation(&notation, session, out)
    }

    fn einsum_read_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        if let Some([lhs, rhs, output]) = borrowed_notation_labels(notation) {
            if let Some((order, config)) =
                read_binary_dot_config_for_labels(self, &lhs, &rhs, &output, &out)
            {
                return execute_binary_dot_config_read_into(session, self, order, &config, out);
            }
        }
        let subscripts = resolve_read_notation(self, notation)?;
        tensor_read_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out,
            TENSOR_READ_EINSUM_INTO_OP,
        )
    }

    fn einsum_read_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        if let [lhs, rhs] = subscripts.inputs.as_slice() {
            if let Some((order, config)) =
                read_binary_dot_config_for_labels(self, lhs, rhs, &subscripts.output, &out)
            {
                return execute_binary_dot_config_read_into(session, self, order, &config, out);
            }
        }
        let subscripts = Subscripts::from(subscripts);
        tensor_read_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out,
            TENSOR_READ_EINSUM_INTO_OP,
        )
    }
}

impl<'a, const N: usize> TensorReadEinsumIntoExt for [TensorRead<'a>; N] {
    fn einsum_read_into(
        &self,
        subscripts: &str,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice().einsum_read_into(subscripts, session, out)
    }

    fn einsum_read_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice()
            .einsum_read_into_notation(notation, session, out)
    }

    fn einsum_read_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        self.as_slice()
            .einsum_read_into_subscripts(subscripts, session, out)
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
/// use tenferro_tensor::{BackendSessionHost, Tensor};
///
/// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
/// let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
///
/// let mut backend = CpuBackend::new();
/// let out = backend
///     .with_backend_session(|session| plan.execute([&lhs, &rhs], session))?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
#[derive(Debug)]
pub struct ConcreteEinsumPlan {
    tree: ContractionTree,
    inputs: Vec<ConcreteEinsumInputSpec>,
    output_shape: Vec<usize>,
    binary_dot: Option<crate::binary_dot::BinaryDotPlan>,
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
        let notation = parse_einsum_notation(subscripts)?;
        Self::prepare_notation(inputs, &notation)
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

    /// Prepare a plan from rank-unresolved notation and concrete tensor inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed axis tokens,
    /// [`Error::Validation`] for rank, shape, or dtype violations, or
    /// [`Error::Planning`] when no contraction tree can be built.
    pub fn prepare_notation<'a, I>(inputs: I, notation: &EinsumNotation) -> Result<Self>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let inputs = inputs.as_ref();
        let subscripts = resolve_tensor_notation(inputs, notation)?;
        Self::prepare_subscripts_internal(input_specs(inputs), &subscripts)
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
        let notation = parse_einsum_notation(subscripts)?;
        Self::prepare_typed_notation(inputs, &notation)
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

    /// Prepare a plan from rank-unresolved notation and typed concrete inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed axis tokens,
    /// [`Error::Validation`] for rank or shape violations, or [`Error::Planning`]
    /// when no contraction tree can be built.
    pub fn prepare_typed_notation<'a, T, I>(inputs: I, notation: &EinsumNotation) -> Result<Self>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
    {
        let inputs = inputs.as_ref();
        let subscripts = resolve_typed_notation(inputs, notation)?;
        Self::prepare_subscripts_internal(typed_input_specs(inputs), &subscripts)
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
        let notation = parse_einsum_notation(subscripts)?;
        Self::prepare_read_notation(inputs, &notation)
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

    /// Prepare a plan from rank-unresolved notation and read-only inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed axis tokens,
    /// [`Error::Validation`] for rank, shape, or dtype violations, or
    /// [`Error::Planning`] when no contraction tree can be built.
    pub fn prepare_read_notation<'a, I>(inputs: I, notation: &EinsumNotation) -> Result<Self>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let inputs = inputs.as_ref();
        let subscripts = resolve_read_notation(inputs, notation)?;
        Self::prepare_subscripts_internal(read_input_specs(inputs), &subscripts)
    }

    /// Number of binary contraction steps in the prepared tree (diagnostics).
    pub(crate) fn step_count(&self) -> usize {
        self.tree.step_count()
    }

    /// Execute this plan on dtype-erased concrete tensor inputs inside a
    /// borrowed backend session.
    ///
    /// Validation and the contraction itself run in the caller's `session`;
    /// this method never enters a new backend session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::{BackendSessionHost, Tensor};
    ///
    /// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    /// let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
    ///
    /// let mut backend = CpuBackend::new();
    /// let out = backend
    ///     .with_backend_session(|session| plan.execute([&lhs, &rhs], session))?;
    /// assert_eq!(out.shape(), &[2, 4]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when inputs violate the prepared rank,
    /// shape, or input-count contract, [`Error::Tensor`] with a
    /// `tenferro_tensor::Error::Validation` `DTypeMismatch` payload when an
    /// input dtype differs from the prepared contract, or [`Error::Tensor`]
    /// for a typed backend failure.
    pub fn execute<'a, I>(&self, inputs: I, session: &mut dyn BackendSession) -> Result<Tensor>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let inputs = inputs.as_ref();
        self.validate_tensor_inputs(inputs, PLAN_EXECUTE_OP)?;
        eager_einsum_exec(session, inputs, &self.tree).map_err(Error::from)
    }

    /// Execute this plan on typed concrete tensor inputs inside a borrowed
    /// backend session.
    ///
    /// Validation and the contraction itself run in the caller's `session`;
    /// this method never enters a new backend session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::{BackendSessionHost, TypedTensor};
    ///
    /// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 4], vec![1.0; 12]).unwrap();
    /// let plan = ConcreteEinsumPlan::prepare_typed([&lhs, &rhs], "ij,jk->ik")?;
    ///
    /// let mut backend = CpuBackend::new();
    /// let out = backend
    ///     .with_backend_session(|session| plan.execute_typed([&lhs, &rhs], session))?;
    /// assert_eq!(out.shape(), &[2, 4]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when inputs violate the prepared rank,
    /// shape, or input-count contract, [`Error::Tensor`] with a
    /// `tenferro_tensor::Error::Validation` `DTypeMismatch` payload when the
    /// prepared dtype differs from `T` or the eager result dtype, or
    /// [`Error::Tensor`] for a typed backend failure.
    pub fn execute_typed<'a, T, I>(
        &self,
        inputs: I,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
    {
        let inputs = inputs.as_ref();
        self.validate_typed_inputs(inputs, PLAN_EXECUTE_OP)?;
        let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
        let result = eager_einsum_exec_read(session, &reads, &self.tree)?;
        into_typed_result(result, PLAN_EXECUTE_OP)
    }

    /// Execute this plan on read-only tensor inputs inside a borrowed backend
    /// session.
    ///
    /// Validation and the contraction itself run in the caller's `session`;
    /// this method never enters a new backend session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    ///
    /// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    /// let plan = ConcreteEinsumPlan::prepare_read(
    ///     [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
    ///     "ij,jk->ik",
    /// )?;
    ///
    /// let mut backend = CpuBackend::new();
    /// let reads = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    /// let out = backend
    ///     .with_backend_session(|session| plan.execute_read(reads, session))?;
    /// assert_eq!(out.shape(), &[2, 4]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when inputs violate the prepared rank,
    /// shape, or input-count contract, [`Error::Tensor`] with a
    /// `tenferro_tensor::Error::Validation` `DTypeMismatch` payload when an
    /// input dtype differs from the prepared contract, or [`Error::Tensor`]
    /// for a typed backend failure.
    pub fn execute_read<'a, I>(&self, inputs: I, session: &mut dyn BackendSession) -> Result<Tensor>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let inputs = inputs.as_ref();
        self.validate_read_inputs(inputs, PLAN_EXECUTE_OP)?;
        eager_einsum_exec_read(session, inputs, &self.tree).map_err(Error::from)
    }

    /// Execute this plan on dtype-erased concrete tensor inputs into
    /// caller-provided output inside a borrowed backend session.
    ///
    /// Validation and the contraction itself run in the caller's `session`;
    /// this method never enters a new backend session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::{BackendSessionHost, Tensor, TensorWrite};
    ///
    /// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    /// let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
    ///
    /// let mut backend = CpuBackend::new();
    /// let mut out = Tensor::from_vec_col_major(vec![2, 4], vec![0.0_f64; 8]).unwrap();
    /// backend.with_backend_session(|session| {
    ///     plan.execute_into(
    ///         [&lhs, &rhs],
    ///         session,
    ///         TensorWrite::from_tensor(&mut out),
    ///     )
    /// })?;
    /// assert_eq!(out.as_slice::<f64>()?, vec![3.0_f64; 8].as_slice());
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for input or output rank, shape, or
    /// input-count contract violations, [`Error::Tensor`] with a
    /// `tenferro_tensor::Error::Validation` `DTypeMismatch` payload for dtype
    /// mismatches, or [`Error::Tensor`] for a typed backend failure.
    pub fn execute_into<'a, I>(
        &self,
        inputs: I,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let inputs = inputs.as_ref();
        self.validate_tensor_inputs(inputs, PLAN_EXECUTE_OP)?;
        self.validate_cached_output(&out, PLAN_EXECUTE_OP)?;
        if let Some(binary_dot) = &self.binary_dot {
            if let [lhs, rhs] = inputs {
                let reads = [TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs)];
                return execute_binary_dot_read_into(session, &reads, binary_dot, out)
                    .map_err(Error::from);
            }
        }
        let reads: Vec<_> = inputs
            .iter()
            .map(|tensor| TensorRead::from_tensor(tensor))
            .collect();
        eager_einsum_exec_read_into(session, &reads, &self.tree, out).map_err(Error::from)
    }

    /// Execute this plan on typed concrete tensor inputs into caller-provided
    /// output inside a borrowed backend session.
    ///
    /// Validation and the contraction itself run in the caller's `session`;
    /// this method never enters a new backend session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::{BackendSessionHost, TypedTensor};
    ///
    /// let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3, 4], vec![1.0; 12]).unwrap();
    /// let plan = ConcreteEinsumPlan::prepare_typed([&lhs, &rhs], "ij,jk->ik")?;
    ///
    /// let mut backend = CpuBackend::new();
    /// let mut out = TypedTensor::<f64>::from_vec_col_major(vec![2, 4], vec![0.0; 8]).unwrap();
    /// backend.with_backend_session(|session| {
    ///     plan.execute_typed_into([&lhs, &rhs], session, &mut out)
    /// })?;
    /// assert_eq!(out.as_slice()?, vec![3.0_f64; 8].as_slice());
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for input or output rank, shape, or
    /// input-count contract violations, [`Error::Tensor`] with a
    /// `tenferro_tensor::Error::Validation` `DTypeMismatch` payload when the
    /// prepared dtype differs from `T` or the output dtype, or
    /// [`Error::Tensor`] for a typed backend failure.
    pub fn execute_typed_into<'a, 'out, T, I, O>(
        &self,
        inputs: I,
        session: &mut dyn BackendSession,
        out: O,
    ) -> Result<()>
    where
        T: TensorScalar,
        I: AsRef<[&'a TypedTensor<T>]>,
        O: Into<TypedTensorWrite<'out, T>>,
    {
        let inputs = inputs.as_ref();
        self.validate_typed_inputs(inputs, PLAN_EXECUTE_OP)?;
        let out = out.into().into_tensor_write();
        self.validate_cached_output(&out, PLAN_EXECUTE_OP)?;
        if let Some(binary_dot) = &self.binary_dot {
            if let [lhs, rhs] = inputs {
                let reads = [T::tensor_read(lhs), T::tensor_read(rhs)];
                return execute_binary_dot_read_into(session, &reads, binary_dot, out)
                    .map_err(Error::from);
            }
        }
        let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
        eager_einsum_exec_read_into(session, &reads, &self.tree, out).map_err(Error::from)
    }

    /// Execute this plan on read-only tensor inputs into caller-provided output
    /// inside a borrowed backend session.
    ///
    /// Validation and the contraction itself run in the caller's `session`;
    /// this method never enters a new backend session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorWrite};
    ///
    /// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    /// let plan = ConcreteEinsumPlan::prepare_read(
    ///     [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
    ///     "ij,jk->ik",
    /// )?;
    ///
    /// let mut backend = CpuBackend::new();
    /// let mut out = Tensor::from_vec_col_major(vec![2, 4], vec![0.0_f64; 8]).unwrap();
    /// let reads = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    /// backend.with_backend_session(|session| {
    ///     plan.execute_read_into(reads, session, TensorWrite::from_tensor(&mut out))
    /// })?;
    /// assert_eq!(out.as_slice::<f64>()?, vec![3.0_f64; 8].as_slice());
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for input or output rank, shape, or
    /// input-count contract violations, [`Error::Tensor`] with a
    /// `tenferro_tensor::Error::Validation` `DTypeMismatch` payload for dtype
    /// mismatches, or [`Error::Tensor`] for a typed backend failure.
    pub fn execute_read_into<'a, I>(
        &self,
        inputs: I,
        session: &mut dyn BackendSession,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let inputs = inputs.as_ref();
        self.validate_read_inputs(inputs, PLAN_EXECUTE_OP)?;
        self.validate_cached_output(&out, PLAN_EXECUTE_OP)?;
        if let Some(binary_dot) = &self.binary_dot {
            return execute_binary_dot_read_into(session, inputs, binary_dot, out)
                .map_err(Error::from);
        }
        eager_einsum_exec_read_into(session, inputs, &self.tree, out).map_err(Error::from)
    }

    /// Execute this plan on read-only inputs with scaled output accumulation
    /// inside a borrowed backend session.
    ///
    /// `accumulation` follows the dot-general contract:
    /// `out = alpha * einsum(inputs) + beta * out`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::{
    ///     BackendSessionHost, DotGeneralAccumulation, DType, Tensor, TensorRead, TensorWrite,
    /// };
    ///
    /// let lhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// let mut out = Tensor::from_vec_col_major(vec![], vec![1.0_f64])?;
    /// let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "i,i->")?;
    /// let mut backend = CpuBackend::new();
    /// backend.with_backend_session(|session| {
    ///     plan.execute_read_into_accum(
    ///         [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
    ///         session,
    ///         DotGeneralAccumulation::add_to(DType::F64)?,
    ///         TensorWrite::from_tensor(&mut out),
    ///     )
    /// })?;
    /// assert_eq!(out.as_slice::<f64>()?, &[7.0]);
    /// # Ok::<(), tenferro_einsum::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for input or output rank, shape, or
    /// input-count contract violations, [`Error::Tensor`] with a
    /// `tenferro_tensor::Error::Validation` `DTypeMismatch` payload for dtype
    /// mismatches, [`Error::Numerical`] for an invalid accumulation, or
    /// [`Error::Tensor`] for a typed backend failure.
    pub fn execute_read_into_accum<'a, I>(
        &self,
        inputs: I,
        session: &mut dyn BackendSession,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let inputs = inputs.as_ref();
        self.validate_read_inputs(inputs, PLAN_EXECUTE_OP)?;
        self.validate_cached_output(&out, PLAN_EXECUTE_OP)?;
        eager_einsum_exec_read_into_accum(session, inputs, &self.tree, accumulation, out)
            .map_err(Error::from)
    }

    fn prepare_subscripts_internal(
        inputs: Vec<ConcreteEinsumInputSpec>,
        subscripts: &Subscripts,
    ) -> Result<Self> {
        let shapes: Vec<&[usize]> = inputs.iter().map(|input| input.shape.as_slice()).collect();
        let binary_dot = binary_dot_plan_for_shapes(&shapes, subscripts);
        let tree = plan_subscripts(subscripts, &shapes)?;
        let output_shape = tree.output_shape();
        Ok(Self {
            tree,
            inputs,
            output_shape,
            binary_dot,
        })
    }

    fn validate_tensor_inputs(&self, actual: &[&Tensor], op: &'static str) -> Result<()> {
        self.validate_input_metadata(
            actual.iter().map(|tensor| (tensor.dtype(), tensor.shape())),
            op,
        )
    }

    fn validate_typed_inputs<T: TensorScalar>(
        &self,
        actual: &[&TypedTensor<T>],
        op: &'static str,
    ) -> Result<()> {
        self.validate_input_metadata(actual.iter().map(|tensor| (T::dtype(), tensor.shape())), op)
    }

    fn validate_read_inputs(&self, actual: &[TensorRead<'_>], op: &'static str) -> Result<()> {
        self.validate_input_metadata(
            actual.iter().map(|tensor| (tensor.dtype(), tensor.shape())),
            op,
        )
    }

    fn validate_input_metadata<'a>(
        &self,
        actual: impl ExactSizeIterator<Item = (DType, &'a [usize])>,
        op: &'static str,
    ) -> Result<()> {
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
        for (expected, (actual_dtype, actual_shape)) in self.inputs.iter().zip(actual) {
            if expected.dtype != actual_dtype {
                return Err(Error::dtype_mismatch(op, expected.dtype, actual_dtype));
            }
            if expected.shape != actual_shape {
                return Err(Error::shape_mismatch(
                    op,
                    expected.shape.clone(),
                    actual_shape.to_vec(),
                ));
            }
        }
        Ok(())
    }

    fn validate_cached_output(&self, out: &TensorWrite<'_>, op: &'static str) -> Result<()> {
        let dtype = self
            .inputs
            .first()
            .map(|input| input.dtype)
            .ok_or_else(|| {
                Error::invalid_argument(op, "inputs", "einsum requires at least one input tensor")
            })?;
        for input in &self.inputs[1..] {
            if input.dtype != dtype {
                return Err(Error::dtype_mismatch(op, dtype, input.dtype));
            }
        }
        if out.dtype() != dtype {
            return Err(Error::dtype_mismatch(op, dtype, out.dtype()));
        }
        if out.shape() != self.output_shape {
            return Err(Error::shape_mismatch(
                op,
                out.shape().to_vec(),
                self.output_shape.clone(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ConcreteEinsumInputSpec {
    dtype: DType,
    shape: Vec<usize>,
}

fn resolve_shapes(notation: &EinsumNotation, shapes: Vec<&[usize]>) -> Result<Subscripts> {
    resolve_einsum_notation(notation, &shapes)
}

fn resolve_tensor_notation(inputs: &[&Tensor], notation: &EinsumNotation) -> Result<Subscripts> {
    resolve_shapes(
        notation,
        inputs.iter().map(|tensor| tensor.shape()).collect(),
    )
}

fn resolve_typed_notation<T: TensorScalar>(
    inputs: &[&TypedTensor<T>],
    notation: &EinsumNotation,
) -> Result<Subscripts> {
    resolve_shapes(
        notation,
        inputs.iter().map(|tensor| tensor.shape()).collect(),
    )
}

fn resolve_view_notation<'a, T: TensorScalar>(
    inputs: &[TypedTensorView<'a, T>],
    notation: &EinsumNotation,
) -> Result<Subscripts> {
    resolve_shapes(notation, inputs.iter().map(|view| view.shape()).collect())
}

fn resolve_read_notation<'a>(
    inputs: &[TensorRead<'a>],
    notation: &EinsumNotation,
) -> Result<Subscripts> {
    resolve_shapes(notation, inputs.iter().map(|input| input.shape()).collect())
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
    session: &mut dyn BackendSession,
    inputs: &[TypedTensorView<'_, T>],
    subscripts: &Subscripts,
    op: &'static str,
) -> Result<TypedTensor<T>> {
    let reads: Vec<_> = inputs
        .iter()
        .cloned()
        .map(|view| TensorRead::from_view(T::tensor_view(view)))
        .collect();
    let plan =
        ConcreteEinsumPlan::prepare_subscripts_internal(read_input_specs(&reads), subscripts)?;
    let result = plan.execute_read(&reads, session)?;
    into_typed_result(result, op)
}

fn read_binary_dot_config_for_labels<L: Copy + PartialEq>(
    inputs: &[TensorRead<'_>],
    lhs_labels: &[L],
    rhs_labels: &[L],
    output_labels: &[L],
    out: &TensorWrite<'_>,
) -> Option<(BinaryDotOperandOrder, tenferro_tensor::DotGeneralConfig)> {
    if inputs.len() != 2
        || inputs[0].dtype() != inputs[1].dtype()
        || out.dtype() != inputs[0].dtype()
    {
        return None;
    }
    binary_dot_config_for_into(
        inputs[0].shape(),
        inputs[1].shape(),
        lhs_labels,
        rhs_labels,
        output_labels,
        out.shape(),
    )
}

fn read_binary_dot_config(
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
    out: &TensorWrite<'_>,
) -> Option<(BinaryDotOperandOrder, tenferro_tensor::DotGeneralConfig)> {
    let [lhs, rhs] = subscripts.inputs.as_slice() else {
        return None;
    };
    read_binary_dot_config_for_labels(inputs, lhs, rhs, &subscripts.output, out)
}

fn execute_binary_dot_config_read_into(
    session: &mut dyn BackendSession,
    inputs: &[TensorRead<'_>],
    order: BinaryDotOperandOrder,
    config: &tenferro_tensor::DotGeneralConfig,
    out: TensorWrite<'_>,
) -> Result<()> {
    let (lhs, rhs) = match order {
        BinaryDotOperandOrder::Original => (0, 1),
        BinaryDotOperandOrder::Swapped => (1, 0),
    };
    session
        .dot_general_read_into(inputs[lhs].clone(), inputs[rhs].clone(), config, out)
        .map_err(Error::from)
}

fn tensor_einsum_into_subscripts(
    session: &mut dyn BackendSession,
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    out: TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    if inputs.len() == 2 {
        let reads = [
            TensorRead::from_tensor(inputs[0]),
            TensorRead::from_tensor(inputs[1]),
        ];
        if let Some((order, config)) = read_binary_dot_config(&reads, subscripts, &out) {
            return execute_binary_dot_config_read_into(session, &reads, order, &config, out);
        }
    }
    let plan = ConcreteEinsumPlan::prepare_subscripts_internal(input_specs(inputs), subscripts)?;
    validate_output(&plan.inputs, &plan.tree, &out, op)?;
    plan.execute_into(inputs, session, out)
}

fn parse_fast_ascii_binary_labels(notation: &str) -> Option<(&[u8], &[u8], &[u8])> {
    let (terms, output) = notation.split_once("->")?;
    let (lhs, rhs) = terms.split_once(',')?;
    // Borrow labels directly. Separators or unsupported labels in any term send
    // the entire expression through the canonical parser, including its errors.
    [lhs, rhs, output]
        .iter()
        .all(|term| term.bytes().all(|byte| byte.is_ascii_alphabetic()))
        .then_some((lhs.as_bytes(), rhs.as_bytes(), output.as_bytes()))
}

fn borrowed_notation_labels(notation: &EinsumNotation) -> Option<[SmallVec<[u32; 8]>; 3]> {
    let [lhs, rhs] = notation.inputs.as_slice() else {
        return None;
    };
    let labels = |axes: &[crate::EinsumAxis]| {
        axes.iter()
            .map(|axis| match axis {
                crate::EinsumAxis::Label(label) => Some(*label),
                crate::EinsumAxis::Ellipsis => None,
            })
            .collect::<Option<SmallVec<[u32; 8]>>>()
    };
    Some([labels(lhs)?, labels(rhs)?, labels(&notation.output)?])
}

fn typed_view_binary_dot_config<T: TensorScalar, L: Copy + PartialEq>(
    inputs: &[TypedTensorView<'_, T>],
    lhs_labels: &[L],
    rhs_labels: &[L],
    output_labels: &[L],
    out: &TensorWrite<'_>,
) -> Option<(BinaryDotOperandOrder, tenferro_tensor::DotGeneralConfig)> {
    if inputs.len() != 2 || out.dtype() != T::dtype() {
        return None;
    }
    binary_dot_config_for_into(
        inputs[0].shape(),
        inputs[1].shape(),
        lhs_labels,
        rhs_labels,
        output_labels,
        out.shape(),
    )
}

fn execute_typed_view_binary_dot_into<T: TensorScalar>(
    session: &mut dyn BackendSession,
    inputs: &[TypedTensorView<'_, T>],
    order: BinaryDotOperandOrder,
    config: &tenferro_tensor::DotGeneralConfig,
    out: TensorWrite<'_>,
) -> Result<()> {
    let (lhs, rhs) = match order {
        BinaryDotOperandOrder::Original => (0, 1),
        BinaryDotOperandOrder::Swapped => (1, 0),
    };
    let lhs = TensorRead::from_view(T::tensor_view(inputs[lhs].clone()));
    let rhs = TensorRead::from_view(T::tensor_view(inputs[rhs].clone()));
    session
        .dot_general_read_into(lhs, rhs, config, out)
        .map_err(Error::from)
}

fn typed_view_einsum_into_subscripts<T: TensorScalar>(
    session: &mut dyn BackendSession,
    inputs: &[TypedTensorView<'_, T>],
    subscripts: &Subscripts,
    out: TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    let reads: Vec<_> = inputs
        .iter()
        .cloned()
        .map(|view| TensorRead::from_view(T::tensor_view(view)))
        .collect();
    let plan =
        ConcreteEinsumPlan::prepare_subscripts_internal(read_input_specs(&reads), subscripts)?;
    validate_output(&plan.inputs, &plan.tree, &out, op)?;
    plan.execute_read_into(&reads, session, out)
}

fn typed_einsum_into_subscripts<T: TensorScalar>(
    session: &mut dyn BackendSession,
    inputs: &[&TypedTensor<T>],
    subscripts: &Subscripts,
    out: TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    if inputs.len() == 2 {
        if let [lhs, rhs] = subscripts.inputs.as_slice() {
            if let Some((order, config)) = binary_dot_config_for_into(
                inputs[0].shape(),
                inputs[1].shape(),
                lhs,
                rhs,
                &subscripts.output,
                out.shape(),
            ) {
                if out.dtype() == T::dtype() {
                    let reads = [T::tensor_read(inputs[0]), T::tensor_read(inputs[1])];
                    return execute_binary_dot_config_read_into(
                        session, &reads, order, &config, out,
                    );
                }
            }
        }
    }
    let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
    let plan =
        ConcreteEinsumPlan::prepare_subscripts_internal(read_input_specs(&reads), subscripts)?;
    validate_output(&plan.inputs, &plan.tree, &out, op)?;
    plan.execute_read_into(&reads, session, out)
}

fn tensor_read_einsum_into_subscripts(
    session: &mut dyn BackendSession,
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
    out: TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    if let Some((order, config)) = read_binary_dot_config(inputs, subscripts, &out) {
        return execute_binary_dot_config_read_into(session, inputs, order, &config, out);
    }
    let plan =
        ConcreteEinsumPlan::prepare_subscripts_internal(read_input_specs(inputs), subscripts)?;
    validate_output(&plan.inputs, &plan.tree, &out, op)?;
    plan.execute_read_into(inputs, session, out)
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

    for (input, labels) in inputs.iter().zip(tree.subscripts.inputs.iter()) {
        if labels.len() != input.shape.len() {
            return Err(Error::rank_mismatch(op, labels.len(), input.shape.len()));
        }
    }
    let output_shape = tree.output_shape();
    if output_shape.len() != tree.subscripts.output.len() {
        return Err(Error::invalid_argument(
            op,
            "output labels",
            "an output label is missing from all inputs",
        ));
    }
    Ok(ConcreteEinsumInputSpec {
        dtype,
        shape: output_shape,
    })
}

fn typed_einsum_subscripts<T: TensorScalar>(
    session: &mut dyn BackendSession,
    inputs: &[&TypedTensor<T>],
    subscripts: &Subscripts,
    op: &'static str,
) -> Result<TypedTensor<T>> {
    let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
    let plan =
        ConcreteEinsumPlan::prepare_subscripts_internal(read_input_specs(&reads), subscripts)?;
    let result = plan.execute_read(&reads, session)?;
    into_typed_result(result, op)
}

pub(crate) fn into_typed_result<T: TensorScalar>(
    result: Tensor,
    op: &'static str,
) -> Result<TypedTensor<T>> {
    let actual = result.dtype();
    T::into_typed(result).map_err(|_| Error::dtype_mismatch(op, T::dtype(), actual))
}
