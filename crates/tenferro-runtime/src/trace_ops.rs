//! Typed operation surface over one [`TraceContext`](crate::TraceContext).
//!
//! Every standard operation and every traced constructor is written once here
//! against a `(TraceContext, TraceValue)` pair. [`TracedTensor`] methods and the
//! extension traits delegate to these helpers, so there is exactly one place that
//! appends an operation and exactly one place that derives a tensor's cached
//! metadata.

use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, IntoShapeVec, Tensor, TensorScalar, TensorValue};

use crate::checkpoint::RetainedValue;
use crate::error::{Error, ErrorPhase, Result};
use crate::program::{CoreSemanticOp, ProgramBuildError, ProgramInputSpec, ProgramValueMetadata};
use crate::sym_dim::SymDim;
use crate::trace::{TraceContext, TraceValue};
use crate::traced::TracedTensor;

/// Map a standard operation payload onto the closed core vocabulary.
///
/// # Errors
///
/// Returns [`Error::RuntimeState`] for the extension carrier, which must be
/// appended through [`append_extension`] instead.
pub(crate) fn core_op(op: &StdTensorOp) -> Result<CoreSemanticOp> {
    CoreSemanticOp::try_from(op).map_err(|source| {
        Error::runtime_state_source("semantic core operation", ErrorPhase::GraphBuild, source)
    })
}

/// Append one standard operation to the context.
pub(crate) fn append_core(
    ctx: &TraceContext,
    op: &StdTensorOp,
    inputs: &[TraceValue],
) -> Result<Box<[TraceValue]>> {
    let core = core_op(op)?;
    ctx.add_op(core, inputs)
        .map_err(|source| build_error("semantic standard operation", source))
}

/// Append one extension operation to the context.
pub(crate) fn append_extension(
    ctx: &TraceContext,
    op: Arc<dyn ExtensionOp>,
    inputs: &[TraceValue],
) -> Result<Box<[TraceValue]>> {
    ctx.add_extension(op, inputs)
        .map_err(|source| build_error("semantic extension operation", source))
}

/// Build a tensor for a value the context just appended.
///
/// `rank`/`dtype`/`hint` are the op-specific expectations; the returned cache is
/// derived from the context metadata and cross-checked against them, so the cache
/// can never disagree with the program.
pub(crate) fn finish_value(
    ctx: &TraceContext,
    value: TraceValue,
    rank: usize,
    dtype: DType,
    hint: Option<Vec<SymDim>>,
) -> Result<TracedTensor> {
    let metadata = ctx
        .value_metadata(value)
        .map_err(|source| build_error("semantic value metadata", source))?;
    debug_assert_eq!(
        rank,
        metadata.shape().len(),
        "cached rank must match the appended value"
    );
    debug_assert_eq!(
        dtype,
        metadata.dtype(),
        "cached dtype must match the appended value"
    );
    let derived = metadata_sym_shape(&metadata);
    Ok(TracedTensor {
        ctx: ctx.clone(),
        value,
        rank: metadata.shape().len(),
        dtype: metadata.dtype(),
        shape_hint: restamp_hint(derived.or(hint), symbolic_id(value.program_value())),
        data: None,
    })
}

/// Build a leaf tensor for a value the context just added as an input.
fn finish_leaf(
    ctx: &TraceContext,
    value: TraceValue,
    rank: usize,
    dtype: DType,
    hint: Option<Vec<SymDim>>,
    data: Option<Arc<RetainedValue>>,
) -> Result<TracedTensor> {
    let mut tensor = finish_value(ctx, value, rank, dtype, hint)?;
    tensor.data = data;
    Ok(tensor)
}

/// Build a tensor purely from the value's context metadata.
pub(crate) fn finish_derived(ctx: &TraceContext, value: TraceValue) -> Result<TracedTensor> {
    let metadata = ctx
        .value_metadata(value)
        .map_err(|source| build_error("semantic value metadata", source))?;
    let rank = metadata.shape().len();
    let dtype = metadata.dtype();
    let hint = metadata_sym_shape(&metadata);
    finish_value(ctx, value, rank, dtype, hint)
}

/// Build a tensor for a value the context just bound to concrete data.
pub(crate) fn finish_bound_value(
    ctx: &TraceContext,
    value: TraceValue,
    data: Arc<RetainedValue>,
) -> Result<TracedTensor> {
    let metadata = ctx
        .value_metadata(value)
        .map_err(|source| build_error("semantic value metadata", source))?;
    let rank = metadata.shape().len();
    let dtype = metadata.dtype();
    let hint = metadata_sym_shape(&metadata);
    let mut tensor = finish_value(ctx, value, rank, dtype, hint)?;
    tensor.data = Some(data);
    Ok(tensor)
}

/// Rebuild a symbolic extent from a program dimension expression.
///
/// Program extents name inputs by program input index; the traced layer names the
/// same axis by that index as its symbolic tensor id, so
/// [`SymDim::to_dim_expr`](tenferro_ops::SymDim::to_dim_expr) round-trips with the
/// identity tensor map.
/// Rebuild a symbolic extent from a program dimension expression.
///
/// Owner-scoped hook for the traced AD layer, which validates seed metadata
/// against program extents.
#[doc(hidden)]
pub fn dim_expr_to_sym_dim_public(expr: &DimExpr) -> Option<SymDim> {
    dim_expr_to_sym_dim(expr)
}

pub(crate) fn dim_expr_to_sym_dim(expr: &DimExpr) -> Option<SymDim> {
    Some(match expr {
        DimExpr::Const(value) => SymDim::from(*value),
        DimExpr::InputDim { input_idx, axis } => SymDim::tensor_axis(*input_idx as u64, *axis),
        DimExpr::Add(lhs, rhs) => dim_expr_to_sym_dim(lhs)? + dim_expr_to_sym_dim(rhs)?,
        DimExpr::Sub(lhs, rhs) => dim_expr_to_sym_dim(lhs)? - dim_expr_to_sym_dim(rhs)?,
        DimExpr::Mul(lhs, rhs) => dim_expr_to_sym_dim(lhs)? * dim_expr_to_sym_dim(rhs)?,
        DimExpr::FloorDiv(lhs, rhs) => dim_expr_to_sym_dim(lhs)? / dim_expr_to_sym_dim(rhs)?,
        DimExpr::Min(lhs, rhs) => dim_expr_to_sym_dim(lhs)?.min(dim_expr_to_sym_dim(rhs)?),
        DimExpr::Max(lhs, rhs) => dim_expr_to_sym_dim(lhs)?.max(dim_expr_to_sym_dim(rhs)?),
    })
}

/// Symbolic identity of one traced value.
///
/// A value's slot is unique and stable inside its context, and the owning
/// builder nonce keeps two contexts' identities apart, so a foreign
/// [`SymDim`] fails to convert instead of naming a local value.
pub(crate) fn symbolic_id(value: crate::program::ProgramValue) -> u64 {
    (value.owner.raw() << 32) | u64::from(value.slot)
}

/// Re-express a symbolic shape in terms of the value that owns it.
///
/// `sym_size`/`axis_sym_dim` must hand back expressions that a *consumer* of this
/// value can convert into its own input coordinates, so every non-constant axis
/// names this value's axis instead of the axes of the operation that produced it.
fn restamp_hint(hint: Option<Vec<SymDim>>, own_id: u64) -> Option<Vec<SymDim>> {
    hint.map(|shape| {
        shape
            .into_iter()
            .enumerate()
            .map(|(axis, dim)| {
                if dim.constant_value().is_some() {
                    dim
                } else {
                    SymDim::tensor_axis(own_id, axis)
                }
            })
            .collect()
    })
}

/// Derive the symbolic shape cache from value metadata.
fn metadata_sym_shape(metadata: &ProgramValueMetadata) -> Option<Vec<SymDim>> {
    metadata
        .shape()
        .iter()
        .map(|extent| match extent {
            ShapeExtent::Exact(expr) => dim_expr_to_sym_dim(expr),
            _ => None,
        })
        .collect()
}

pub(crate) fn build_error(op: &'static str, source: ProgramBuildError) -> Error {
    match source {
        ProgramBuildError::Metadata { source } => *source,
        other => Error::runtime_state_source(op, ErrorPhase::GraphBuild, other),
    }
}

fn concrete_spec(
    dtype: DType,
    shape: &[usize],
    identity: Option<&'static str>,
) -> ProgramInputSpec {
    let metadata = ProgramValueMetadata::new(dtype, shape.iter().map(|&size| DimExpr::Const(size)));
    let spec = ProgramInputSpec::from_metadata(metadata);
    match identity {
        Some(identity) => spec.with_scalar_identity(identity),
        None => spec,
    }
}

fn concrete_hint(shape: &[usize]) -> Option<Vec<SymDim>> {
    Some(shape.iter().map(|&size| SymDim::from(size)).collect())
}

fn symbolic_spec(dtype: DType, rank: usize, input_idx: usize) -> ProgramInputSpec {
    ProgramInputSpec::new(
        dtype,
        (0..rank).map(|axis| DimExpr::InputDim { input_idx, axis }),
    )
}

impl TraceContext {
    /// Add a placeholder input with a fixed concrete shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::TraceContext;
    /// use tenferro_tensor::DType;
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.input_concrete_shape(DType::F64, &[2, 3])?;
    /// assert_eq!(x.rank(), 2);
    /// assert!(x.is_concrete_shape());
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the trace cannot accept another input
    /// or a binding already exists for it.
    pub fn input_concrete_shape(&self, dtype: DType, shape: &[usize]) -> Result<TracedTensor> {
        self.input_concrete_shape_with_identity(dtype, shape, None)
    }

    /// Add a placeholder input of concrete shape declaring an external scalar.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the trace cannot accept another input
    /// or a binding already exists for it, and [`Error::RuntimeStateSource`]
    /// wrapping `ProgramBuildError::ExternalScalarWithoutIdentity` when an
    /// externally defined scalar dtype reaches the builder without the declared
    /// identity.
    pub fn input_concrete_shape_declaring_scalar(
        &self,
        dtype: DType,
        shape: &[usize],
        identity: &'static str,
    ) -> Result<TracedTensor> {
        self.input_concrete_shape_with_identity(dtype, shape, Some(identity))
    }

    fn input_concrete_shape_with_identity(
        &self,
        dtype: DType,
        shape: &[usize],
        identity: Option<&'static str>,
    ) -> Result<TracedTensor> {
        let spec = concrete_spec(dtype, shape, identity);
        let value = self
            .input(spec)
            .map_err(|source| build_error("trace input", source))?;
        finish_leaf(self, value, shape.len(), dtype, concrete_hint(shape), None)
    }

    /// Add a placeholder input with symbolic (rank-only) extents.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::TraceContext;
    /// use tenferro_tensor::DType;
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.input_symbolic_shape(DType::F64, 2)?;
    /// assert_eq!(x.rank(), 2);
    /// assert!(!x.is_concrete_shape());
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the trace cannot accept another input.
    pub fn input_symbolic_shape(&self, dtype: DType, rank: usize) -> Result<TracedTensor> {
        let index = self.input_count();
        let spec = symbolic_spec(dtype, rank, index);
        let value = self
            .input(spec)
            .map_err(|source| build_error("trace input", source))?;
        finish_leaf(self, value, rank, dtype, metadata_hint(self, value)?, None)
    }

    /// Add a constant leaf holding `tensor` with its concrete shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{Tensor, TraceContext};
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_tensor_concrete_shape(
    ///     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?,
    /// )?;
    /// assert!(x.attached_value().is_some());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a malformed tensor or
    /// [`Error::RuntimeState`] when the trace cannot bind the constant.
    pub fn from_tensor_concrete_shape(&self, tensor: Tensor) -> Result<TracedTensor> {
        self.tensor_concrete_shape_with_identity(tensor, None)
    }

    /// Add a constant leaf of concrete shape declaring an external scalar.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a malformed tensor, [`Error::RuntimeState`]
    /// when the trace cannot bind the constant, and [`Error::RuntimeStateSource`]
    /// wrapping `ProgramBuildError::ExternalScalarWithoutIdentity` when an
    /// externally defined scalar dtype reaches the builder without the declared
    /// identity.
    pub fn from_tensor_concrete_shape_declaring_scalar(
        &self,
        tensor: Tensor,
        identity: &'static str,
    ) -> Result<TracedTensor> {
        self.tensor_concrete_shape_with_identity(tensor, Some(identity))
    }

    fn tensor_concrete_shape_with_identity(
        &self,
        tensor: Tensor,
        identity: Option<&'static str>,
    ) -> Result<TracedTensor> {
        let dtype = tensor.dtype();
        let shape = tensor.shape().to_vec();
        let spec = concrete_spec(dtype, &shape, identity);
        let data = Arc::new(RetainedValue::from_tensor(tensor));
        let value = self
            .input_bound(spec, Arc::clone(&data))
            .map_err(|source| build_error("trace constant", source))?;
        finish_leaf(
            self,
            value,
            shape.len(),
            dtype,
            concrete_hint(&shape),
            Some(data),
        )
    }

    /// Add a constant leaf holding `tensor` but advertising symbolic extents.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the trace cannot bind the constant.
    pub fn from_tensor_symbolic_shape(&self, tensor: Tensor) -> Result<TracedTensor> {
        let data = Arc::new(RetainedValue::from_tensor(tensor));
        self.from_shared_tensor_value_symbolic_shape(data)
    }

    /// Add a constant leaf from a value, advertising symbolic extents.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the value cannot be retained or the
    /// trace cannot bind it.
    #[doc(hidden)]
    pub fn from_tensor_value_symbolic_shape(&self, value: TensorValue) -> Result<TracedTensor> {
        let data = Arc::new(RetainedValue::from_tensor_value(value)?);
        self.from_shared_tensor_value_symbolic_shape(data)
    }

    /// Add a constant leaf from an already-retained value with symbolic extents.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the trace cannot bind the value.
    #[doc(hidden)]
    pub fn from_shared_tensor_value_symbolic_shape(
        &self,
        data: Arc<RetainedValue>,
    ) -> Result<TracedTensor> {
        let rank = data.shape().len();
        let dtype = data.dtype();
        let index = self.input_count();
        let spec = symbolic_spec(dtype, rank, index);
        let value = self
            .input_bound(spec, Arc::clone(&data))
            .map_err(|source| build_error("trace constant", source))?;
        finish_leaf(
            self,
            value,
            rank,
            dtype,
            metadata_hint(self, value)?,
            Some(data),
        )
    }

    /// Add a constant leaf from compact column-major data.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] with a typed validation error when the
    /// shape and data length disagree.
    pub fn from_vec_col_major<T: TensorScalar>(
        &self,
        shape: impl IntoShapeVec,
        data: Vec<T>,
    ) -> Result<TracedTensor> {
        let tensor = Tensor::from_vec_col_major(shape, data)?;
        self.from_tensor_concrete_shape(tensor)
    }
}

fn metadata_hint(ctx: &TraceContext, value: TraceValue) -> Result<Option<Vec<SymDim>>> {
    let metadata = ctx
        .value_metadata(value)
        .map_err(|source| build_error("semantic value metadata", source))?;
    Ok(metadata_sym_shape(&metadata))
}
