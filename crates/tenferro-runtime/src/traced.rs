use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use computegraph::graph::{Graph, GraphBuilder};
use computegraph::types::{OperationRole, ValueKey, ValueRef};
use computegraph::LocalValueId;
use num_complex::{Complex32, Complex64};
use tenferro_ops::ad::context::GlobalMetadataScope;
use tenferro_ops::broadcast::{
    broadcast_error_to_validation, broadcast_in_dim_extent_error, broadcast_input_plan,
    broadcast_shape, broadcast_shapes, BroadcastError,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, Error as TensorError, GatherConfig, IntoShapeVec,
    PadConfig, ScatterConfig, ShapeMismatch, SliceConfig, Tensor, TensorScalar, ValidationError,
};

use super::error::{Error, ErrorPhase, Result};
use super::sym_dim::SymDim;
use crate::checkpoint::CheckpointNode;
use crate::metadata::{
    concrete_tensor_meta, register_scoped_graph_metadata, register_scoped_value_metadata,
    symbolic_input_meta, tensor_meta, MetadataScopeChain,
};
use crate::scalar_semantics::{bool_from_real_for_op, round_real_to_i32_for_op, round_real_to_i64};
use crate::shape_constraint::ConstraintScopeChain;

static NEXT_INPUT_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_TRACED_ID: AtomicU64 = AtomicU64::new(0);

pub type TracedTensorId = u64;

pub(crate) fn next_input_key() -> TensorInputKey {
    TensorInputKey::User {
        id: NEXT_INPUT_ID.fetch_add(1, Ordering::Relaxed),
    }
}

pub(crate) fn next_traced_id() -> TracedTensorId {
    NEXT_TRACED_ID.fetch_add(1, Ordering::Relaxed)
}

type TracedInputMap = HashMap<TensorInputKey, Arc<Tensor>>;

#[derive(Clone)]
pub struct TracedTensor {
    pub id: TracedTensorId,
    pub rank: usize,
    pub dtype: DType,
    pub(crate) graph: Arc<Graph<StdTensorOp>>,
    pub val: LocalValueId,
    pub(crate) data: Option<Arc<Tensor>>,
    pub(crate) shape_hint: Option<Vec<SymDim>>,
    pub(crate) inputs_map: Arc<TracedInputMap>,
    pub(crate) extra_roots: Vec<Arc<Graph<StdTensorOp>>>,
    pub(crate) checkpoint_chain: Option<Arc<CheckpointNode>>,
    pub(crate) metadata_scopes: MetadataScopeChain,
    pub(crate) constraint_scopes: ConstraintScopeChain,
}

impl fmt::Debug for TracedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracedTensor")
            .field("id", &self.id)
            .field("rank", &self.rank)
            .field("dtype", &self.dtype)
            .field("val", &self.val)
            .field("shape_hint", &self.shape_hint)
            .field("has_data", &self.data.is_some())
            .finish_non_exhaustive()
    }
}

pub(crate) fn merge_traced_inputs_map<'a>(
    inputs: impl IntoIterator<Item = &'a TracedTensor>,
) -> Arc<TracedInputMap> {
    let maps: Vec<_> = inputs
        .into_iter()
        .map(|input| &input.inputs_map)
        .filter(|map| !map.is_empty())
        .collect();
    match maps.as_slice() {
        [] => return Arc::new(HashMap::new()),
        [single] => return Arc::clone(*single),
        _ => {}
    }

    for &candidate in &maps {
        if input_map_matches_ordered_merge(candidate.as_ref(), &maps) {
            return Arc::clone(candidate);
        }
    }

    let mut merged = (**maps[0]).clone();
    for map in maps.iter().skip(1) {
        merged.extend(
            map.iter()
                .map(|(key, tensor)| (key.clone(), tensor.clone())),
        );
    }
    Arc::new(merged)
}

fn input_map_matches_ordered_merge(
    candidate: &TracedInputMap,
    maps: &[&Arc<TracedInputMap>],
) -> bool {
    for map in maps {
        for key in map.keys() {
            let Some(final_tensor) = maps.iter().rev().find_map(|source| source.get(key)) else {
                return false;
            };
            let Some(candidate_tensor) = candidate.get(key) else {
                return false;
            };
            if !Arc::ptr_eq(candidate_tensor, final_tensor) {
                return false;
            }
        }
    }
    true
}

pub(crate) fn try_concrete_shape(tensor: &TracedTensor) -> Option<Vec<usize>> {
    tensor
        .shape_hint
        .as_ref()?
        .iter()
        .map(SymDim::constant_value)
        .collect()
}

fn graph_validation(op: &'static str, source: impl Into<ValidationError>) -> Error {
    Error::validation(op, ErrorPhase::GraphBuild, source.into())
}

fn graph_invalid_argument(
    op: &'static str,
    argument: &'static str,
    message: impl Into<String>,
) -> Error {
    graph_validation(
        op,
        ValidationError::InvalidArgument {
            argument,
            message: message.into(),
        },
    )
}

fn graph_broadcast_error(op: &'static str, error: BroadcastError) -> Error {
    graph_validation(op, broadcast_error_to_validation(error))
}

fn graph_tensor_error(op: &'static str, error: TensorError) -> Error {
    match error {
        TensorError::Validation { source, .. } => graph_validation(op, source),
        other => Error::TensorRuntime(other),
    }
}

fn graph_error_with_context(op: &'static str, error: Error) -> Error {
    match error {
        Error::Validation { source, .. } => graph_validation(op, source),
        other => other,
    }
}

pub(crate) fn concrete_shape(tensor: &TracedTensor) -> Result<Vec<usize>> {
    tensor
        .shape_hint
        .as_ref()
        .ok_or_else(|| {
            graph_invalid_argument(
                "TracedTensor::concrete_shape",
                "shape",
                format!("missing shape hint for traced tensor {}", tensor.id),
            )
        })?
        .iter()
        .map(|dim| {
            dim.constant_value().ok_or_else(|| {
                graph_invalid_argument(
                    "TracedTensor::concrete_shape",
                    "shape",
                    format!("symbolic dimension in shape hint for tensor {}", tensor.id),
                )
            })
        })
        .collect()
}

/// Broadcast a traced tensor to `target_shape`.
///
/// Expanding singleton axes are first reshaped away so the existing
/// `BroadcastInDim` transpose rule reduces them correctly during VJP.
pub(crate) fn broadcast_to(tensor: &TracedTensor, target_shape: &[usize]) -> Result<TracedTensor> {
    let tensor_shape = concrete_shape(tensor)?;
    if tensor_shape == target_shape {
        return Ok(tensor.clone());
    }

    let plan = broadcast_input_plan(&tensor_shape, target_shape)
        .map_err(|err| graph_broadcast_error("broadcast_to", err))?;

    let source = if plan.source_shape == tensor_shape {
        tensor.clone()
    } else {
        tensor.reshape(&plan.source_shape)?
    };
    source.broadcast_in_dim(target_shape, &plan.dims)
}

/// Broadcast two tensors to a common shape.
pub(crate) fn broadcast_binary(
    a: &TracedTensor,
    b: &TracedTensor,
) -> Result<(TracedTensor, TracedTensor)> {
    if a.shape_hint == b.shape_hint && a.rank == b.rank {
        return Ok((a.clone(), b.clone()));
    }
    if (try_concrete_shape(a).is_none() || try_concrete_shape(b).is_none()) && a.rank == b.rank {
        return Ok((a.clone(), b.clone()));
    }
    let a_shape = concrete_shape(a)?;
    let b_shape = concrete_shape(b)?;
    let target = broadcast_shape(&a_shape, &b_shape)
        .map_err(|err| graph_broadcast_error("broadcast_binary", err))?;
    Ok((broadcast_to(a, &target)?, broadcast_to(b, &target)?))
}

pub(crate) fn broadcast_ternary(
    a: &TracedTensor,
    b: &TracedTensor,
    c: &TracedTensor,
) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    let a_shape = concrete_shape(a)?;
    let b_shape = concrete_shape(b)?;
    let c_shape = concrete_shape(c)?;
    let target = broadcast_shapes([a_shape.as_slice(), b_shape.as_slice(), c_shape.as_slice()])
        .map_err(|err| graph_broadcast_error("broadcast_ternary", err))?;
    Ok((
        broadcast_to(a, &target)?,
        broadcast_to(b, &target)?,
        broadcast_to(c, &target)?,
    ))
}

fn scale_with_constant(input: &TracedTensor, op: StdTensorOp) -> Result<TracedTensor> {
    let scalar = apply_nullary(op, 0, input.dtype, Some(vec![]))?;
    apply_binary(
        StdTensorOp::Mul,
        input,
        &scalar,
        input.rank,
        input.shape_hint.clone(),
    )
}

fn try_inferred_output_dtype(
    op: &StdTensorOp,
    inputs: &[DType],
    context: &'static str,
) -> Result<DType> {
    crate::shape_infer::infer_output_dtype_at(op, inputs, ErrorPhase::GraphBuild)
        .map_err(|err| graph_error_with_context(context, err))
}

fn checked_shape_product_for_graph_build(
    shape: &[usize],
    context: &'static str,
    _role: &'static str,
) -> Result<usize> {
    shape.iter().copied().try_fold(1usize, |acc, dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| graph_validation(context, ValidationError::IntegerOverflow))
    })
}

fn validate_concrete_reshape_shape(input: &TracedTensor, shape: &[usize]) -> Result<()> {
    let to = checked_shape_product_for_graph_build(shape, "TracedTensor::reshape", "target")?;
    let Some(input_shape) = try_concrete_shape(input) else {
        return Ok(());
    };
    let from =
        checked_shape_product_for_graph_build(&input_shape, "TracedTensor::reshape", "input")?;
    if from != to {
        return Err(graph_validation(
            "TracedTensor::reshape",
            ShapeMismatch::ReshapeElementCount { from, to },
        ));
    }
    Ok(())
}

fn traced_input_shape_exprs(input_idx: usize, tensor: &TracedTensor) -> Vec<DimExpr> {
    match tensor.shape_hint.as_ref() {
        Some(shape) => shape
            .iter()
            .enumerate()
            .map(|(axis, dim)| {
                dim.constant_value()
                    .map_or(DimExpr::InputDim { input_idx, axis }, DimExpr::Const)
            })
            .collect(),
        None => (0..tensor.rank)
            .map(|axis| DimExpr::InputDim { input_idx, axis })
            .collect(),
    }
}

fn traced_input_sym_shape(tensor: &TracedTensor) -> Vec<SymDim> {
    tensor.shape_hint.clone().unwrap_or_else(|| {
        (0..tensor.rank)
            .map(|axis| SymDim::tensor_axis(tensor.id, axis))
            .collect()
    })
}

pub(crate) fn infer_traced_single_output_shape(
    op_name: &'static str,
    op: &StdTensorOp,
    inputs: &[&TracedTensor],
) -> Result<(usize, Option<Vec<SymDim>>)> {
    let input_shape_exprs: Vec<Vec<DimExpr>> = inputs
        .iter()
        .enumerate()
        .map(|(input_idx, tensor)| traced_input_shape_exprs(input_idx, tensor))
        .collect();
    let input_shape_refs: Vec<&[DimExpr]> = input_shape_exprs.iter().map(Vec::as_slice).collect();
    let output_shapes = crate::shape_infer::infer_output_shapes(op, &input_shape_refs)
        .map_err(|err| graph_error_with_context(op_name, err))?;
    let output_shape = output_shapes.first().ok_or_else(|| {
        Error::Internal(format!("{op_name}: shape inference returned no outputs"))
    })?;
    if output_shapes.len() != 1 {
        return Err(Error::Internal(format!(
            "{op_name}: expected single-output shape inference, got {} outputs",
            output_shapes.len()
        )));
    }

    let input_sym_shapes: Vec<Vec<SymDim>> = inputs
        .iter()
        .map(|tensor| traced_input_sym_shape(tensor))
        .collect();
    let input_sym_refs: Vec<&[SymDim]> = input_sym_shapes.iter().map(Vec::as_slice).collect();
    let out_shape_hint = output_shape
        .iter()
        .map(|dim| SymDim::from_dim_expr(dim, &input_sym_refs))
        .collect();
    Ok((output_shape.len(), Some(out_shape_hint)))
}

pub(crate) fn register_metadata_or_runtime_state<E>(
    result: std::result::Result<GlobalMetadataScope, E>,
) -> Result<GlobalMetadataScope>
where
    E: StdError + Send + Sync + 'static,
{
    result.map_err(|err| Error::runtime_state_source("metadata", ErrorPhase::Compile, err))
}

fn reduction_output_meta(
    tensor: &TracedTensor,
    axes: &[usize],
    op: &'static str,
) -> Result<(usize, Option<Vec<SymDim>>)> {
    let mut seen = vec![false; tensor.rank];
    for &axis in axes {
        if axis >= tensor.rank {
            return Err(graph_validation(
                op,
                ValidationError::AxisOutOfBounds {
                    axis,
                    rank: tensor.rank,
                },
            ));
        }
        if seen[axis] {
            return Err(graph_validation(
                op,
                ValidationError::DuplicateAxis {
                    axis,
                    role: "reduction",
                },
            ));
        }
        seen[axis] = true;
    }

    let out_shape_hint = tensor.shape_hint.as_ref().map(|shape| {
        (0..shape.len())
            .filter(|d| !axes.contains(d))
            .map(|d| shape[d].clone())
            .collect()
    });
    Ok((tensor.rank - axes.len(), out_shape_hint))
}

fn validate_traced_axis(tensor: &TracedTensor, axis: usize, op: &'static str) -> Result<()> {
    if axis >= tensor.rank {
        return Err(graph_validation(
            op,
            ValidationError::AxisOutOfBounds {
                axis,
                rank: tensor.rank,
            },
        ));
    }
    Ok(())
}

fn validate_traced_axes(rank: usize, axes: &[usize], op: &'static str) -> Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(graph_validation(
                op,
                ValidationError::AxisOutOfBounds { axis, rank },
            ));
        }
        if seen[axis] {
            return Err(graph_validation(
                op,
                ValidationError::DuplicateAxis { axis, role: "axis" },
            ));
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_traced_insert_axis(rank: usize, axis: usize, op: &'static str) -> Result<()> {
    if axis > rank {
        return Err(graph_invalid_argument(
            op,
            "axis",
            format!("axis {axis} out of bounds for rank {rank} insertion"),
        ));
    }
    Ok(())
}

fn validate_traced_perm(rank: usize, perm: &[usize], op: &'static str) -> Result<()> {
    if perm.len() != rank {
        return Err(graph_validation(
            op,
            ValidationError::InvalidPermutationLength {
                expected: rank,
                actual: perm.len(),
            },
        ));
    }
    let mut seen = vec![false; rank];
    for &axis in perm {
        if axis >= rank {
            return Err(graph_validation(
                op,
                ValidationError::AxisOutOfBounds { axis, rank },
            ));
        }
        if seen[axis] {
            return Err(graph_validation(
                op,
                ValidationError::DuplicateAxis {
                    axis,
                    role: "permutation",
                },
            ));
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_broadcast_in_dim_args(
    input: &TracedTensor,
    output_shape: &[SymDim],
    dims: &[usize],
    op: &'static str,
) -> Result<()> {
    if dims.len() != input.rank {
        return Err(graph_validation(
            op,
            ValidationError::RankMismatch {
                expected: input.rank,
                actual: dims.len(),
            },
        ));
    }

    let concrete_input_shape: Option<Vec<usize>> = input
        .shape_hint
        .as_ref()
        .and_then(|shape| shape.iter().map(SymDim::constant_value).collect());
    let concrete_output_shape = output_shape
        .iter()
        .map(SymDim::constant_value)
        .collect::<Option<Vec<_>>>();
    if let (Some(input_shape), Some(output_shape)) = (
        concrete_input_shape.as_deref(),
        concrete_output_shape.as_deref(),
    ) {
        if let Some(error) = broadcast_in_dim_extent_error(input_shape, output_shape, dims) {
            return Err(graph_broadcast_error(op, error));
        }
    }

    let mut seen = vec![false; output_shape.len()];
    for &dim in dims {
        if dim >= output_shape.len() {
            return Err(graph_validation(
                op,
                ValidationError::AxisOutOfBounds {
                    axis: dim,
                    rank: output_shape.len(),
                },
            ));
        }
        if seen[dim] {
            return Err(graph_validation(
                op,
                ValidationError::DuplicateAxis {
                    axis: dim,
                    role: "broadcast",
                },
            ));
        }
        seen[dim] = true;
    }

    Ok(())
}

impl std::ops::Add for &TracedTensor {
    type Output = Result<TracedTensor>;

    fn add(self, rhs: &TracedTensor) -> Result<TracedTensor> {
        TracedTensor::add(self, rhs)
    }
}

impl std::ops::Sub for &TracedTensor {
    type Output = Result<TracedTensor>;

    fn sub(self, rhs: &TracedTensor) -> Result<TracedTensor> {
        TracedTensor::sub(self, rhs)
    }
}

impl std::ops::Mul for &TracedTensor {
    type Output = Result<TracedTensor>;

    fn mul(self, rhs: &TracedTensor) -> Result<TracedTensor> {
        TracedTensor::mul(self, rhs)
    }
}

impl std::ops::Mul<f64> for &TracedTensor {
    type Output = Result<TracedTensor>;

    fn mul(self, rhs: f64) -> Result<TracedTensor> {
        self.scale_real(rhs)
    }
}

impl std::ops::Mul<&TracedTensor> for f64 {
    type Output = Result<TracedTensor>;

    fn mul(self, rhs: &TracedTensor) -> Result<TracedTensor> {
        rhs.scale_real(self)
    }
}

impl std::ops::Neg for &TracedTensor {
    type Output = Result<TracedTensor>;

    fn neg(self) -> Self::Output {
        TracedTensor::neg(self)
    }
}

impl std::ops::Div for &TracedTensor {
    type Output = Result<TracedTensor>;

    fn div(self, rhs: &TracedTensor) -> Result<TracedTensor> {
        TracedTensor::div(self, rhs)
    }
}

impl std::ops::Rem for &TracedTensor {
    type Output = Result<TracedTensor>;

    fn rem(self, rhs: &TracedTensor) -> Result<TracedTensor> {
        TracedTensor::rem(self, rhs)
    }
}

impl TracedTensor {
    /// Return the graph that owns this traced tensor's current value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// let _graph = x.graph();
    /// ```
    pub fn graph(&self) -> &Arc<Graph<StdTensorOp>> {
        &self.graph
    }

    /// Return the concrete tensor data attached to this traced value, if any.
    ///
    /// Placeholder tensors created with `input_concrete_shape` or
    /// `input_symbolic_shape` have no attached data until execution bindings
    /// provide it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, TracedTensor};
    ///
    /// let concrete = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// assert!(concrete.attached_data().is_some());
    ///
    /// let placeholder = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    /// assert!(placeholder.attached_data().is_none());
    /// ```
    pub fn attached_data(&self) -> Option<&Arc<Tensor>> {
        self.data.as_ref()
    }

    /// Build a [`TracedTensor`] leaf from a concrete [`Tensor`], keeping its
    /// shape as a concrete `shape_hint`.
    ///
    /// This is the common constructor when you have concrete tensor data that
    /// you want to use both for graph building and for evaluation. The
    /// resulting tensor is treated as a concrete-shape leaf by downstream
    /// passes (binary einsum decomposition, build-time reshape folding, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{Tensor, TracedTensor};
    ///
    /// let a = TracedTensor::from_tensor_concrete_shape(
    ///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    /// )
    /// .unwrap();
    /// assert_eq!(a.rank, 2);
    /// assert!(a.is_concrete_shape());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when graph metadata registration
    /// cannot retain the concrete tensor's shape or dtype.
    pub fn from_tensor_concrete_shape(tensor: Tensor) -> Result<Self> {
        let shape = tensor.shape().to_vec();
        let rank = shape.len();
        let dtype = tensor.dtype();
        let key = next_input_key();
        let id = next_traced_id();
        let data = Arc::new(tensor);

        let mut builder = GraphBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let graph = Arc::new(builder.build());
        let metadata_scope = register_metadata_or_runtime_state(register_scoped_value_metadata(
            graph.values()[val].key.clone(),
            concrete_tensor_meta(dtype, &shape),
        ))?;

        let mut map = HashMap::new();
        map.insert(key, Arc::clone(&data));

        Ok(Self {
            id,
            rank,
            dtype,
            graph,
            val,
            data: Some(data),
            shape_hint: Some(shape.into_iter().map(SymDim::from).collect()),
            inputs_map: Arc::new(map),
            extra_roots: Vec::new(),
            checkpoint_chain: None,
            metadata_scopes: MetadataScopeChain::from_scope(metadata_scope),
            constraint_scopes: ConstraintScopeChain::empty(),
        })
    }

    /// Build a [`TracedTensor`] leaf from a concrete [`Tensor`] but advertise
    /// a symbolic shape during graph construction.
    ///
    /// The tensor data is still attached (so plain `eval` works without
    /// bindings), but graph passes see the leaf as shape-symbolic. This is
    /// useful for building a single traced program that should not bake in
    /// shape-specific optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{Tensor, TracedTensor};
    ///
    /// let t = TracedTensor::from_tensor_symbolic_shape(
    ///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    /// )
    /// .unwrap();
    /// assert_eq!(t.rank, 2);
    /// assert!(!t.is_concrete_shape());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when symbolic graph metadata
    /// registration is unavailable or its registry state is poisoned.
    pub fn from_tensor_symbolic_shape(tensor: Tensor) -> Result<Self> {
        let rank = tensor.shape().len();
        let dtype = tensor.dtype();
        let key = next_input_key();
        let id = next_traced_id();
        let data = Arc::new(tensor);

        let mut builder = GraphBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let graph = Arc::new(builder.build());
        let metadata_scope = register_metadata_or_runtime_state(register_scoped_value_metadata(
            graph.values()[val].key.clone(),
            symbolic_input_meta(dtype, id, rank),
        ))?;

        let mut map = HashMap::new();
        map.insert(key, Arc::clone(&data));

        Ok(Self {
            id,
            rank,
            dtype,
            graph,
            val,
            data: Some(data),
            shape_hint: None,
            inputs_map: Arc::new(map),
            extra_roots: Vec::new(),
            checkpoint_chain: None,
            metadata_scopes: MetadataScopeChain::from_scope(metadata_scope),
            constraint_scopes: ConstraintScopeChain::empty(),
        })
    }

    /// Build a data-less placeholder leaf with a fixed (concrete) shape.
    ///
    /// Must be bound via [`crate::GraphExecutor::run_with_inputs`] before evaluation.
    /// Use this when you know the exact shape of the input but want to build
    /// the graph once and feed different concrete tensors at execution time.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 3]).unwrap();
    /// assert_eq!(x.rank, 2);
    /// assert!(x.is_concrete_shape());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when graph metadata registration
    /// fails or the registry state is poisoned. `dtype` and `shape` are
    /// metadata values and are not revalidated by this constructor.
    pub fn input_concrete_shape(dtype: DType, shape: &[usize]) -> Result<Self> {
        let shape = shape.to_vec();
        let rank = shape.len();
        let key = next_input_key();
        let id = next_traced_id();

        let mut builder = GraphBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let graph = Arc::new(builder.build());
        let metadata_scope = register_metadata_or_runtime_state(register_scoped_value_metadata(
            graph.values()[val].key.clone(),
            concrete_tensor_meta(dtype, &shape),
        ))?;

        Ok(Self {
            id,
            rank,
            dtype,
            graph,
            val,
            data: None,
            shape_hint: Some(shape.into_iter().map(SymDim::from).collect()),
            inputs_map: Arc::new(HashMap::new()),
            extra_roots: Vec::new(),
            checkpoint_chain: None,
            metadata_scopes: MetadataScopeChain::from_scope(metadata_scope),
            constraint_scopes: ConstraintScopeChain::empty(),
        })
    }

    /// Build a data-less placeholder leaf with the given rank but fully
    /// symbolic shape (every dim is a distinct `SymDim::TensorAxis`).
    ///
    /// Must be bound via [`crate::GraphExecutor::run_with_inputs`] before
    /// evaluation. Use this to build shape-agnostic graphs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    /// assert_eq!(x.rank, 2);
    /// assert!(!x.is_concrete_shape());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when graph metadata registration
    /// fails or the registry state is poisoned. `rank` is recorded as the
    /// symbolic placeholder rank and is not otherwise rejected here.
    pub fn input_symbolic_shape(dtype: DType, rank: usize) -> Result<Self> {
        let key = next_input_key();
        let id = next_traced_id();

        let mut builder = GraphBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let graph = Arc::new(builder.build());
        let metadata_scope = register_metadata_or_runtime_state(register_scoped_value_metadata(
            graph.values()[val].key.clone(),
            symbolic_input_meta(dtype, id, rank),
        ))?;

        Ok(Self {
            id,
            rank,
            dtype,
            graph,
            val,
            data: None,
            shape_hint: None,
            inputs_map: Arc::new(HashMap::new()),
            extra_roots: Vec::new(),
            checkpoint_chain: None,
            metadata_scopes: MetadataScopeChain::from_scope(metadata_scope),
            constraint_scopes: ConstraintScopeChain::empty(),
        })
    }

    /// Build a concrete-shape [`TracedTensor`] leaf from column-major typed
    /// `Vec<T>` data.
    ///
    /// The data must already be in tenferro's physical column-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
    /// )?;
    /// assert_eq!(a.rank, 2);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] containing
    /// `ValidationError::ShapeDataLengthMismatch` when the shape product does
    /// not equal `data.len()`, or `ValidationError::IntegerOverflow` when the
    /// shape product cannot be represented by `usize`.
    pub fn from_vec_col_major<T: TensorScalar>(
        shape: impl IntoShapeVec,
        data: Vec<T>,
    ) -> Result<Self> {
        Self::from_tensor_concrete_shape(Tensor::from_vec_col_major(shape, data)?)
    }

    /// Return the tensor element dtype recorded for this traced value.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns `true` iff every dim of this tensor's `shape_hint` is a
    /// constant `SymDim` (i.e. the shape is fully known at graph-build time).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    /// assert!(a.is_concrete_shape());
    /// assert!(!b.is_concrete_shape());
    /// ```
    pub fn is_concrete_shape(&self) -> bool {
        try_concrete_shape(self).is_some()
    }

    /// Return the fully-concrete shape of this tensor, if every dim of
    /// its shape-hint is a constant `SymDim`. Returns `None` if any
    /// dimension is symbolic.
    ///
    /// This is the counterpart to [`Self::is_concrete_shape`] for callers
    /// that need to *use* the concrete shape (e.g. external composition
    /// wrappers building `broadcast_in_dim` payloads from known shapes).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// assert_eq!(a.try_concrete_shape(), Some(vec![2, 3]));
    ///
    /// let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    /// assert!(b.try_concrete_shape().is_none());
    /// ```
    pub fn try_concrete_shape(&self) -> Option<Vec<usize>> {
        try_concrete_shape(self)
    }

    /// Return the concrete tensor shape.
    ///
    /// Returns an error when a shape hint is missing or any dimension is
    /// symbolic. Composite traced ops that require concrete sizes should
    /// propagate this error instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument` when this tensor
    /// has no shape hint or any dimension is symbolic.
    pub fn concrete_shape(&self) -> Result<Vec<usize>> {
        concrete_shape(self)
    }

    /// If this `TracedTensor` is a leaf (single-node input graph),
    /// return its input key. Computed tensors return `None`.
    pub fn input_key(&self) -> Option<TensorInputKey> {
        match &self.graph.values()[self.val].key {
            ValueKey::Input(key) => Some(key.clone()),
            _ => None,
        }
    }

    /// Elementwise addition with NumPy-style broadcasting.
    ///
    /// Prefer using the `+` operator when it reads naturally.
    ///
    /// A longer expression such as `a + b + c` does not compose because the
    /// first `+` returns `Result<TracedTensor, Error>`, so the second `+`
    /// would receive a result rather than a tensor. Use `?` at each step or
    /// the explicit fallible method chain shown below when the operation
    /// sequence is more important than notation:
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::{Error, TracedTensor};
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// # let z = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    /// let y = x.add(&z);
    /// let y2 = &x + &z;
    /// # fn add_three(
    /// #     a: &TracedTensor,
    /// #     b: &TracedTensor,
    /// #     c: &TracedTensor,
    /// # ) -> Result<TracedTensor, Error> {
    /// let ab = (a + b)?;
    /// let sum = (&ab + c)?;
    /// let method_chain = a.add(b)?.add(c)?;
    /// let _ = method_chain;
    /// # Ok(sum)
    /// # }
    /// ```
    ///
    /// Tenferro prioritizes robust error handling over the conciseness of
    /// chained operator notation; the explicit fallible methods are the
    /// canonical form for longer sequences.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when operand shapes
    /// cannot be broadcast, or [`Error::RuntimeStateSource`] when graph
    /// metadata registration fails.
    ///
    /// # Deferred errors
    ///
    /// If symbolic dimensions prevent shape comparison during graph
    /// construction, the same `ShapeMismatch` can be reported during
    /// compilation or execution, with the corresponding [`ErrorPhase`].
    pub fn add(&self, other: &TracedTensor) -> Result<TracedTensor> {
        let (lhs, rhs) = broadcast_binary(self, other)?;
        apply_binary(
            StdTensorOp::Add,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise subtraction with NumPy-style broadcasting.
    ///
    /// Prefer using the `-` operator when it reads naturally.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when operand shapes
    /// cannot be broadcast, or [`Error::RuntimeStateSource`] when graph
    /// metadata registration fails.
    ///
    /// # Deferred errors
    ///
    /// If symbolic dimensions prevent shape comparison during graph
    /// construction, the same `ShapeMismatch` can be reported during
    /// compilation or execution, with the corresponding [`ErrorPhase`].
    pub fn sub(&self, other: &TracedTensor) -> Result<TracedTensor> {
        let (lhs, rhs) = broadcast_binary(self, other)?;
        apply_binary(
            StdTensorOp::Sub,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise multiplication with NumPy-style broadcasting.
    ///
    /// Prefer using the `*` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// # let z = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    /// let y = x.mul(&z);
    /// let y2 = &x * &z;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when operand shapes
    /// cannot be broadcast, or [`Error::RuntimeStateSource`] when graph
    /// metadata registration fails.
    ///
    /// # Deferred errors
    ///
    /// If symbolic ranks prevent shape comparison during graph construction,
    /// the same `ShapeMismatch` can be reported during compilation or
    /// execution, with the corresponding [`ErrorPhase`].
    pub fn mul(&self, other: &TracedTensor) -> Result<TracedTensor> {
        let (lhs, rhs) = broadcast_binary(self, other)?;
        apply_binary(
            StdTensorOp::Mul,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise division with NumPy-style broadcasting.
    ///
    /// Prefer using the `/` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// # let z = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    /// let y = x.div(&z);
    /// let y2 = &x / &z;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when operand shapes
    /// cannot be broadcast, or [`Error::RuntimeStateSource`] when graph
    /// metadata registration fails.
    ///
    /// # Deferred errors
    ///
    /// If symbolic ranks prevent shape comparison during graph construction,
    /// the same `ShapeMismatch` can be reported during compilation or
    /// execution, with the corresponding [`ErrorPhase`]. For integer inputs,
    /// a zero divisor is reported during execution as
    /// [`Error::TensorRuntime`] containing a
    /// [`tenferro_tensor::Error::Extension`] classified as
    /// `tenferro_tensor::ErrorKind::NumericalFailure` and retaining the typed
    /// backend source; floating-point and complex zero divisors follow their
    /// numeric semantics instead.
    pub fn div(&self, other: &TracedTensor) -> Result<TracedTensor> {
        let (lhs, rhs) = broadcast_binary(self, other)?;
        apply_binary(
            StdTensorOp::Div,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise remainder with NumPy-style broadcasting.
    ///
    /// Prefer using the `%` operator when it reads naturally.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when operand shapes
    /// cannot be broadcast, [`Error::Unsupported`] at
    /// [`ErrorPhase::GraphBuild`] when either operand has a complex dtype, or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    ///
    /// # Deferred errors
    ///
    /// If symbolic ranks prevent shape comparison during graph construction,
    /// the same `ShapeMismatch` can be reported during compilation or
    /// execution, with the corresponding [`ErrorPhase`]. For integer inputs,
    /// a zero divisor is reported during execution as
    /// [`Error::TensorRuntime`] containing a
    /// [`tenferro_tensor::Error::Extension`] classified as
    /// `tenferro_tensor::ErrorKind::NumericalFailure` and retaining the typed
    /// backend source; floating-point zero divisors follow their numeric
    /// semantics.
    pub fn rem(&self, other: &TracedTensor) -> Result<TracedTensor> {
        let (lhs, rhs) = broadcast_binary(self, other)?;
        apply_binary(
            StdTensorOp::Rem,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise comparison with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when the concrete
    /// operands cannot be broadcast, [`Error::Unsupported`] when ordered
    /// comparison rejects a complex dtype, or [`Error::RuntimeStateSource`]
    /// when result metadata cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// With same-rank symbolic operands, shape compatibility is retained as a
    /// graph constraint. A concrete mismatch is reported later as
    /// [`Error::TensorRuntime`] containing a typed validation source, with the
    /// failure phase identifying compilation or execution.
    pub fn compare(&self, other: &TracedTensor, dir: CompareDir) -> Result<TracedTensor> {
        apply_broadcast_binary_op(StdTensorOp::Compare(dir), self, other)
    }

    /// Elementwise maximum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when the concrete
    /// operands cannot be broadcast, [`Error::Unsupported`] when ordered
    /// maximum rejects a complex dtype, or [`Error::RuntimeStateSource`] when
    /// result metadata cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// With same-rank symbolic operands, the broadcast constraint may fail at
    /// compile or execution and is returned as [`Error::TensorRuntime`] with
    /// its typed validation source.
    pub fn maximum(&self, other: &TracedTensor) -> Result<TracedTensor> {
        apply_broadcast_binary_op(StdTensorOp::Maximum, self, other)
    }

    /// Elementwise minimum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when the concrete
    /// operands cannot be broadcast, [`Error::Unsupported`] when ordered
    /// minimum rejects a complex dtype, or [`Error::RuntimeStateSource`] when
    /// result metadata cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// With same-rank symbolic operands, the broadcast constraint may fail at
    /// compile or execution and is returned as [`Error::TensorRuntime`] with
    /// its typed validation source.
    pub fn minimum(&self, other: &TracedTensor) -> Result<TracedTensor> {
        apply_broadcast_binary_op(StdTensorOp::Minimum, self, other)
    }

    /// Select values from `on_true` or `on_false` using `condition`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument` when an operand
    /// lacks concrete shape metadata, or `ShapeMismatch` when the concrete
    /// condition and branches cannot share a broadcast shape. Dtype promotion
    /// failures are returned as [`Error::TensorRuntime`] with the typed
    /// `UnsupportedDTypeConversion` source; metadata failures retain
    /// [`Error::RuntimeStateSource`].
    pub fn where_select(
        condition: &TracedTensor,
        on_true: &TracedTensor,
        on_false: &TracedTensor,
    ) -> Result<TracedTensor> {
        apply_broadcast_ternary_op(StdTensorOp::Select, condition, on_true, on_false)
    }

    /// Alias for [`Self::where_select`].
    ///
    /// # Errors
    ///
    /// Returns the same concrete failures as [`Self::where_select`]:
    /// [`Error::Validation`] with `InvalidArgument`/`ShapeMismatch` for shape
    /// metadata or broadcasting, [`Error::TensorRuntime`] with
    /// `UnsupportedDTypeConversion` for failed promotion, and
    /// [`Error::RuntimeStateSource`] for metadata registration.
    pub fn select(
        condition: &TracedTensor,
        on_true: &TracedTensor,
        on_false: &TracedTensor,
    ) -> Result<TracedTensor> {
        Self::where_select(condition, on_true, on_false)
    }

    /// Clamp values elementwise between lower and upper bounds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument` when an operand
    /// lacks concrete shape metadata, `ShapeMismatch` when bounds cannot be
    /// broadcast with the input, [`Error::Unsupported`] for an ordered
    /// complex dtype, or [`Error::RuntimeStateSource`] when metadata cannot be
    /// registered.
    pub fn clamp(&self, lower: &TracedTensor, upper: &TracedTensor) -> Result<TracedTensor> {
        apply_broadcast_ternary_op(StdTensorOp::Clamp, self, lower, upper)
    }

    fn apply_same_shape_unary(&self, op: StdTensorOp) -> Result<TracedTensor> {
        apply_unary(op, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise negation.
    ///
    /// Prefer using the unary `-` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.neg().unwrap();
    /// let y2 = (-&x).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn neg(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Neg)
    }

    /// Elementwise complex conjugate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use num_complex::Complex64;
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(
    /// #     vec![2],
    /// #     vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    /// # )
    /// # .unwrap();
    /// let y = x.conj().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn conj(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Conj)
    }

    /// Elementwise absolute value.
    ///
    /// Complex inputs return real magnitudes (`C32 -> F32`, `C64 -> F64`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![-1.0_f64, 2.0]).unwrap();
    /// let y = x.abs().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn abs(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Abs)
    }

    /// Elementwise sign.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![-1.0_f64, 2.0]).unwrap();
    /// let y = x.sign().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn sign(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Sign)
    }

    /// Scale by a real scalar: `y = factor * x`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.scale_real(2.0)?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument` when an integer or
    /// boolean factor is non-finite or out of range for the input dtype, or
    /// [`Error::RuntimeStateSource`] when output metadata registration fails.
    pub fn scale_real(&self, factor: f64) -> Result<TracedTensor> {
        let op = match self.dtype {
            DType::F64 => StdTensorOp::constant(factor),
            DType::F32 => StdTensorOp::constant(factor as f32),
            DType::I32 => StdTensorOp::constant(round_real_to_i32_for_op("scale_real", factor)?),
            DType::I64 => StdTensorOp::constant(round_real_to_i64(factor)?),
            DType::Bool => StdTensorOp::constant(bool_from_real_for_op("scale_real", factor)?),
            DType::C64 => StdTensorOp::constant(Complex64::new(factor, 0.0)),
            DType::C32 => StdTensorOp::constant(Complex32::new(factor as f32, 0.0)),
        };
        scale_with_constant(self, op)
    }

    /// Scale by a complex scalar: `y = factor * x`.
    ///
    /// Only complex tensors support complex scaling. For a real scalar factor
    /// that should preserve the input dtype, prefer [`scale_real`](Self::scale_real).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(
    /// #     vec![2],
    /// #     vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    /// # )
    /// # .unwrap();
    /// let y = x.scale_complex(Complex64::new(0.0, 1.0)).unwrap(); // multiply by i
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument` when a complex
    /// factor is applied to a non-complex dtype, or
    /// [`Error::RuntimeStateSource`] when output metadata registration fails.
    pub fn scale_complex(&self, factor: Complex64) -> Result<TracedTensor> {
        match self.dtype {
            DType::C64 => scale_with_constant(self, StdTensorOp::constant(factor)),
            DType::C32 => scale_with_constant(
                self,
                StdTensorOp::constant(Complex32::new(factor.re as f32, factor.im as f32)),
            ),
            DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => {
                Err(graph_invalid_argument(
                    "scale_complex",
                    "dtype",
                    format!("requires complex tensor dtype, got {:?}", self.dtype),
                ))
            }
        }
    }

    /// Elementwise exponential.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.exp().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn exp(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Exp)
    }

    /// Elementwise natural logarithm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.log().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn log(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Log)
    }

    /// Elementwise sine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.sin().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn sin(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Sin)
    }

    /// Elementwise cosine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.cos().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn cos(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Cos)
    }

    /// Elementwise hyperbolic tangent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.tanh().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn tanh(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Tanh)
    }

    /// Elementwise square root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap();
    /// let y = x.sqrt().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn sqrt(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Sqrt)
    }

    /// Elementwise reciprocal square root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap();
    /// let y = x.rsqrt().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn rsqrt(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Rsqrt)
    }

    /// Elementwise power with NumPy-style broadcasting.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let base = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    /// # let exp = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 2.0]).unwrap();
    /// let y = base.pow(&exp);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch` when the concrete
    /// operands cannot be broadcast, or [`Error::RuntimeStateSource`] when
    /// result metadata cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// A symbolic broadcast mismatch or integer negative exponent is
    /// discovered at compile or execution and is returned as
    /// [`Error::TensorRuntime`] with a typed `ShapeMismatch` or
    /// `NegativeIntegerExponent` numerical source and the corresponding
    /// [`ErrorPhase`].
    pub fn pow(&self, other: &TracedTensor) -> Result<TracedTensor> {
        let (lhs, rhs) = broadcast_binary(self, other)?;
        apply_binary(
            StdTensorOp::Pow,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise `exp(x) - 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.expm1().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn expm1(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Expm1)
    }

    /// Elementwise `log(1 + x)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = x.log1p().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when the graph metadata registry
    /// is unavailable or poisoned while recording the unary result.
    pub fn log1p(&self) -> Result<TracedTensor> {
        self.apply_same_shape_unary(StdTensorOp::Log1p)
    }

    /// Convert the tensor to a different dtype using checked conversion.
    ///
    /// Use [`cast`](Self::cast) when a lossy dtype projection is intended.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::DType;
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    ///
    /// let y = x.convert(DType::C64)?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// requested pair is outside tenferro's checked dtype-promotion lattice,
    /// or [`Error::Validation`] when graph metadata rejects the conversion.
    /// Use [`cast`](Self::cast) for explicit lossy dtype projection.
    pub fn convert(&self, to: DType) -> Result<TracedTensor> {
        tenferro_tensor::validate::validate_convert_dtype("TracedTensor::convert", self.dtype, to)?;
        self.cast(to)
    }

    /// Cast the tensor to a different dtype using explicit dtype projection.
    ///
    /// `cast` may truncate, narrow precision, project complex values to their
    /// real component, or use boolean truthiness where the backend supports the
    /// requested projection.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::DType;
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.2_f64, -2.8]).unwrap();
    ///
    /// let y = x.cast(DType::I32).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] containing
    /// `UnsupportedDTypeConversion` when the requested input-to-target
    /// projection is not supported, or [`Error::RuntimeStateSource`] when
    /// converted-output metadata cannot be registered.
    pub fn cast(&self, to: DType) -> Result<TracedTensor> {
        if self.dtype == to {
            return Ok(self.clone());
        }

        apply_unary_with_dtype(
            StdTensorOp::Convert {
                from: self.dtype,
                to,
            },
            self,
            self.rank,
            self.shape_hint.clone(),
            to,
        )
    }

    /// Generalized tensor contraction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::{DotGeneralConfig, TracedTensor};
    /// # let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// # let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    /// # let config = DotGeneralConfig {
    /// #     lhs_contracting_dims: vec![1],
    /// #     rhs_contracting_dims: vec![0],
    /// #     lhs_batch_dims: vec![],
    /// #     rhs_batch_dims: vec![],
    /// # };
    /// let y = a.dot_general(&b, config)?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch`, `AxisOutOfBounds`,
    /// `DuplicateAxis`, or `AxisRoleConflict` when dimension numbers are
    /// invalid for the operand ranks, and [`Error::RuntimeStateSource`] when
    /// output metadata cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// Contracting or batch dimensions whose sizes are symbolic are checked
    /// when concrete inputs reach compilation or execution. A mismatch is
    /// returned as [`Error::TensorRuntime`] with a typed `ShapeMismatch`
    /// source and its corresponding [`ErrorPhase`].
    pub fn dot_general(
        &self,
        other: &TracedTensor,
        config: DotGeneralConfig,
    ) -> Result<TracedTensor> {
        config
            .validate_dims_with_ranks(self.rank, other.rank)
            .map_err(|err| graph_tensor_error("dot_general", err))?;
        let lhs_free: Vec<usize> = (0..self.rank)
            .filter(|d| {
                !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d)
            })
            .collect();
        let rhs_free: Vec<usize> = (0..other.rank)
            .filter(|d| {
                !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d)
            })
            .collect();
        let out_rank = config.lhs_batch_dims.len() + lhs_free.len() + rhs_free.len();
        let out_shape_hint = match (&self.shape_hint, &other.shape_hint) {
            (Some(lhs_shape), Some(rhs_shape)) => {
                let mut out_shape = Vec::with_capacity(out_rank);
                for &d in &lhs_free {
                    out_shape.push(lhs_shape[d].clone());
                }
                for &d in &rhs_free {
                    out_shape.push(rhs_shape[d].clone());
                }
                for &d in &config.lhs_batch_dims {
                    out_shape.push(lhs_shape[d].clone());
                }
                Some(out_shape)
            }
            _ => None,
        };

        apply_binary(
            StdTensorOp::DotGeneral { config },
            self,
            other,
            out_rank,
            out_shape_hint,
        )
    }

    /// Matrix multiplication for rank-2 tensors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch` when either operand is
    /// not rank 2, `ShapeMismatch::ContractedDimensions` when known matrix
    /// dimensions differ, or [`Error::RuntimeStateSource`] when output
    /// metadata cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// If either contracted dimension is symbolic, the mismatch is discovered
    /// at compilation or execution and returned as [`Error::TensorRuntime`]
    /// with its typed `ShapeMismatch` source.
    pub fn matmul(&self, other: &TracedTensor) -> Result<TracedTensor> {
        if self.rank != 2 {
            return Err(graph_validation(
                "TracedTensor::matmul",
                ValidationError::RankMismatch {
                    expected: 2,
                    actual: self.rank,
                },
            ));
        }
        if other.rank != 2 {
            return Err(graph_validation(
                "TracedTensor::matmul",
                ValidationError::RankMismatch {
                    expected: 2,
                    actual: other.rank,
                },
            ));
        }
        if let (Some(lhs_shape), Some(rhs_shape)) = (&self.shape_hint, &other.shape_hint) {
            if let (Some(lhs_cols), Some(rhs_rows)) =
                (lhs_shape[1].constant_value(), rhs_shape[0].constant_value())
            {
                if lhs_cols != rhs_rows {
                    return Err(graph_validation(
                        "TracedTensor::matmul",
                        ShapeMismatch::ContractedDimensions {
                            lhs_axis: 1,
                            lhs_size: lhs_cols,
                            rhs_axis: 0,
                            rhs_size: rhs_rows,
                        },
                    ));
                }
            }
        }
        self.dot_general(
            other,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
    }

    /// Sum over the given axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    /// let y = x.reduce_sum(&[0])?;
    /// let y2 = x.reduce_sum(&[0])?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when an axis is
    /// outside the input rank or `DuplicateAxis` when `axes` repeats an axis,
    /// or [`Error::RuntimeStateSource`] when output metadata cannot be
    /// registered.
    pub fn reduce_sum(&self, axes: &[usize]) -> Result<TracedTensor> {
        let (out_rank, out_shape_hint) =
            reduction_output_meta(self, axes, "TracedTensor::reduce_sum")?;
        apply_unary(
            StdTensorOp::ReduceSum {
                axes: axes.to_vec(),
            },
            self,
            out_rank,
            out_shape_hint,
        )
    }

    /// Reduce by taking the maximum along the given axes.
    ///
    /// Used by tropical (max-plus) compositions: a max-plus reduction over
    /// an axis is `ReduceMax` on that axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    /// let y = x.reduce_max(&[0])?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when an axis is
    /// outside the input rank or `DuplicateAxis` when `axes` repeats an axis,
    /// [`Error::Unsupported`] when a non-empty maximum reduction receives a
    /// complex dtype, or [`Error::RuntimeStateSource`] when output metadata
    /// cannot be registered.
    pub fn reduce_max(&self, axes: &[usize]) -> Result<TracedTensor> {
        let (out_rank, out_shape_hint) =
            reduction_output_meta(self, axes, "TracedTensor::reduce_max")?;
        try_apply_unary(
            StdTensorOp::ReduceMax {
                axes: axes.to_vec(),
            },
            self,
            out_rank,
            out_shape_hint,
            "TracedTensor::reduce_max",
        )
    }

    /// Reduce by taking the minimum along the given axes.
    ///
    /// Used by tropical (min-plus) compositions: a min-plus reduction over
    /// an axis is `ReduceMin` on that axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    /// let y = x.reduce_min(&[0])?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when an axis is
    /// outside the input rank or `DuplicateAxis` when `axes` repeats an axis,
    /// [`Error::Unsupported`] when a non-empty minimum reduction receives a
    /// complex dtype, or [`Error::RuntimeStateSource`] when output metadata
    /// cannot be registered.
    pub fn reduce_min(&self, axes: &[usize]) -> Result<TracedTensor> {
        let (out_rank, out_shape_hint) =
            reduction_output_meta(self, axes, "TracedTensor::reduce_min")?;
        try_apply_unary(
            StdTensorOp::ReduceMin {
                axes: axes.to_vec(),
            },
            self,
            out_rank,
            out_shape_hint,
            "TracedTensor::reduce_min",
        )
    }

    /// Reduce by taking the product along the given axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    /// let y = x.reduce_prod(&[0])?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when an axis is
    /// outside the input rank or `DuplicateAxis` when `axes` repeats an axis,
    /// or [`Error::RuntimeStateSource`] when output metadata cannot be
    /// registered.
    pub fn reduce_prod(&self, axes: &[usize]) -> Result<TracedTensor> {
        let (out_rank, out_shape_hint) =
            reduction_output_meta(self, axes, "TracedTensor::reduce_prod")?;
        apply_unary(
            StdTensorOp::ReduceProd {
                axes: axes.to_vec(),
            },
            self,
            out_rank,
            out_shape_hint,
        )
    }

    /// Reshape without changing element order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64; 4]).unwrap();
    /// let y = x.reshape(&[2, 2])?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `ShapeMismatch::ReshapeElementCount`
    /// when a concrete input has a different element count, or
    /// `IntegerOverflow` when the target shape product overflows `usize`.
    pub fn reshape(&self, shape: &[usize]) -> Result<TracedTensor> {
        validate_concrete_reshape_shape(self, shape)?;
        apply_unary_with_dtype(
            StdTensorOp::Reshape {
                to_shape: DimExpr::from_concrete(shape),
            },
            self,
            shape.len(),
            Some(shape.iter().copied().map(SymDim::from).collect()),
            self.dtype,
        )
    }

    /// Return a symbolic expression for the size of one axis, suitable as
    /// an `InputDim`-style reference when composing with
    /// [`TracedTensor::reshape_sym`].
    ///
    /// Semantics: if this tensor's `shape_hint` has a symbolic
    /// (non-constant) entry for `axis`, that entry is returned
    /// verbatim. Otherwise — including when `shape_hint[axis]` is a
    /// concrete `SymDim::Concrete(n)` — a
    /// `SymDim::tensor_axis(self.id, axis)` reference is returned so the
    /// resulting graph remains shape-polymorphic if the same graph is
    /// later evaluated against a differently-shaped binding.
    ///
    /// For a canonical "what is the size of this axis?" query that
    /// reports the concrete size when it is known, prefer
    /// [`Self::axis_sym_dim`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let rows = x.sym_size(0)?;
    /// let cols = x.sym_size(1)?;
    /// let y = x.reshape_sym(&[rows * cols]).unwrap();
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when `axis` is
    /// outside this tensor's rank.
    pub fn sym_size(&self, axis: usize) -> Result<SymDim> {
        validate_traced_axis(self, axis, "TracedTensor::sym_size")?;
        Ok(self
            .shape_hint
            .as_ref()
            .and_then(|shape| shape.get(axis))
            .filter(|dim| dim.constant_value().is_none())
            .cloned()
            .unwrap_or_else(|| SymDim::tensor_axis(self.id, axis)))
    }

    /// Return the canonical `SymDim` for `axis` — the concrete
    /// `SymDim::Concrete(n)` when the size is known, otherwise a symbolic
    /// expression identifying this tensor's axis.
    ///
    /// Unlike [`Self::sym_size`], this method does **not** rewrite
    /// concrete axes into `TensorAxis` references. It is the accessor
    /// external composition wrappers should use when building mixed
    /// concrete/symbolic target shapes for operations like
    /// [`Self::broadcast_in_dim_sym`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// // Concrete axis: reports the constant size.
    /// assert_eq!(a.axis_sym_dim(0).unwrap().constant_value(), Some(2));
    ///
    /// let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    /// // Fully symbolic leaf: reports a TensorAxis reference.
    /// assert!(b.axis_sym_dim(0).unwrap().constant_value().is_none());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when `axis` is
    /// outside this tensor's rank.
    pub fn axis_sym_dim(&self, axis: usize) -> Result<SymDim> {
        validate_traced_axis(self, axis, "TracedTensor::axis_sym_dim")?;
        match self.shape_hint.as_ref().and_then(|shape| shape.get(axis)) {
            Some(dim) => Ok(dim.clone()),
            None => Ok(SymDim::tensor_axis(self.id, axis)),
        }
    }

    /// Return the full symbolic shape of this tensor when a `shape_hint`
    /// is present.
    ///
    /// Returns `None` for fully-symbolic placeholders produced via
    /// [`Self::input_symbolic_shape`] (where `shape_hint` is intentionally
    /// absent). For those, build the shape axis-by-axis via
    /// [`Self::axis_sym_dim`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// assert!(a.sym_shape().is_some());
    /// assert_eq!(a.sym_shape().unwrap().len(), 2);
    ///
    /// let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    /// assert!(b.sym_shape().is_none());
    /// ```
    pub fn sym_shape(&self) -> Option<&[SymDim]> {
        self.shape_hint.as_deref()
    }

    /// Reshape using symbolic dimensions derived from traced tensor axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let rows = x.sym_size(0)?;
    /// let cols = x.sym_size(1)?;
    /// let y = x.reshape_sym(&[rows * cols]).unwrap();
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::SymbolicShapeConversion`] when a supplied symbolic
    /// dimension cannot be mapped to this graph, or [`Error::RuntimeStateSource`]
    /// when result metadata cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// Element-count compatibility for symbolic dimensions is checked when
    /// concrete inputs reach compilation or execution. A mismatch is returned
    /// as [`Error::TensorRuntime`] with a typed `ShapeMismatch` source.
    pub fn reshape_sym(&self, shape: &[SymDim]) -> Result<TracedTensor> {
        let tensor_map = [(self.id, 0usize)];
        let to_shape = shape
            .iter()
            .map(|dim| {
                dim.to_dim_expr(&tensor_map)
                    .map_err(|source| Error::SymbolicShapeConversion {
                        op: "TracedTensor::reshape_sym",
                        phase: ErrorPhase::GraphBuild,
                        source,
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let out_shape_hint = Some(shape.to_vec());
        apply_unary(
            StdTensorOp::Reshape { to_shape },
            self,
            shape.len(),
            out_shape_hint,
        )
    }

    /// Broadcast into a larger shape with explicit dimension placement.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    /// let y = x.broadcast_in_dim(&[2, 3], &[1])?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch` when `dims` does not
    /// have one entry per input axis, `AxisOutOfBounds` or `DuplicateAxis` for
    /// an invalid output mapping, or `InvalidArgument` when known dimensions
    /// cannot broadcast. [`Error::RuntimeStateSource`] reports failure to
    /// register the result metadata.
    pub fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Result<TracedTensor> {
        let out_shape_hint: Vec<SymDim> = shape.iter().copied().map(SymDim::from).collect();
        validate_broadcast_in_dim_args(
            self,
            &out_shape_hint,
            dims,
            "TracedTensor::broadcast_in_dim",
        )?;
        apply_unary(
            StdTensorOp::BroadcastInDim {
                shape: DimExpr::from_concrete(shape),
                dims: dims.to_vec(),
            },
            self,
            shape.len(),
            Some(out_shape_hint),
        )
    }

    /// Broadcast into a symbolic target shape with explicit dimension
    /// placement.
    ///
    /// Unlike [`Self::broadcast_in_dim`], each axis of `shape` is a
    /// [`SymDim`], so the target shape can mix concrete sizes (via
    /// `SymDim::from(n)`) with symbolic references to this tensor's axes
    /// (via [`Self::axis_sym_dim`]) or to axes of other traced tensors.
    ///
    /// When `shape` contains a `SymDim` that references a traced tensor
    /// other than `self`, the referenced tensor(s) must be supplied in
    /// `shape_refs`. They are wired into the built op as auxiliary
    /// shape-reference inputs — the op does not read their data, only
    /// their runtime shape. `shape_refs` must be listed in the same order
    /// in which their tensor IDs first appear when walking `shape` after
    /// any references to `self`. Usually the simplest correct thing is to
    /// pass each unique non-self reference tensor once.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    /// let m = a.axis_sym_dim(0)?;
    /// let k = a.axis_sym_dim(1)?;
    /// let n = b.axis_sym_dim(1)?;
    /// // Broadcast `a[m, k]` to `[m, k, n]`, placing `a`'s axes at 0, 1
    /// // and taking `n` from `b` as an auxiliary shape reference.
    /// let a_b = a.broadcast_in_dim_sym(&[m, k, n], &[0, 1], &[&b])?;
    /// assert_eq!(a_b.rank, 3);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch`, `AxisOutOfBounds`,
    /// `DuplicateAxis`, or `InvalidArgument` when the output mapping or shape
    /// references are invalid, [`Error::SymbolicShapeConversion`] for an
    /// unmappable symbolic dimension, or [`Error::RuntimeStateSource`] when
    /// metadata cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// If a symbolic output dimension is smaller than a non-unit input axis,
    /// the concrete broadcast check is deferred to compilation or execution
    /// and is returned as [`Error::TensorRuntime`] with a typed validation
    /// source.
    pub fn broadcast_in_dim_sym(
        &self,
        shape: &[SymDim],
        dims: &[usize],
        shape_refs: &[&TracedTensor],
    ) -> Result<TracedTensor> {
        validate_broadcast_in_dim_args(self, shape, dims, "TracedTensor::broadcast_in_dim_sym")?;

        // Build a dedup'd list of shape-reference tensors (first occurrence
        // wins) and index them starting at 1 — the primary input `self`
        // is at 0.
        let mut dedup_refs: Vec<&TracedTensor> = Vec::with_capacity(shape_refs.len());
        let mut tensor_map: Vec<(u64, usize)> = vec![(self.id, 0)];
        for &t in shape_refs {
            if !tensor_map.iter().any(|(id, _)| *id == t.id) {
                let idx = tensor_map.len();
                tensor_map.push((t.id, idx));
                dedup_refs.push(t);
            }
        }

        let to_shape: Vec<DimExpr> = shape
            .iter()
            .map(|dim| {
                dim.to_dim_expr(&tensor_map)
                    .map_err(|source| Error::SymbolicShapeConversion {
                        op: "broadcast_in_dim_sym",
                        phase: ErrorPhase::GraphBuild,
                        source,
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        // Trim auxiliary shape-reference inputs down to those actually
        // used by the generated `DimExpr`s. If the target shape resolved
        // to all constants (the concrete-shape case) the op is a plain
        // unary broadcast with no extra parents. Otherwise the op needs
        // a contiguous prefix of shape-ref inputs covering every
        // referenced `input_idx`.
        let max_used_idx = DimExpr::max_input_idx_all(&to_shape).unwrap_or(0);
        let used_refs: Vec<&TracedTensor> = dedup_refs.into_iter().take(max_used_idx).collect();

        let out_shape_hint = Some(shape.to_vec());
        apply_unary_with_shape_refs(
            StdTensorOp::BroadcastInDim {
                shape: to_shape,
                dims: dims.to_vec(),
            },
            self,
            &used_refs,
            shape.len(),
            out_shape_hint,
        )
    }

    /// Slice with explicit start, limit, and stride per axis.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch` when the start/limit/
    /// stride vectors do not match the input rank, `InvalidSliceStep` when a
    /// stride is zero, `InvalidSliceBounds` when a limit precedes its start,
    /// or [`Error::RuntimeStateSource`] when output metadata cannot be
    /// registered.
    pub fn slice(&self, config: SliceConfig) -> Result<TracedTensor> {
        let op = StdTensorOp::Slice(config);
        let (out_rank, out_shape_hint) =
            infer_traced_single_output_shape("TracedTensor::slice", &op, &[self])?;
        apply_unary(op, self, out_rank, out_shape_hint)
    }

    /// Pad with zeros using StableHLO-style edge and interior padding.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch` when padding vectors
    /// do not match the input rank, `InvalidArgument` for negative interior
    /// padding, or `IntegerOverflow` when the padded extent exceeds `usize`.
    /// [`Error::RuntimeStateSource`] is returned when output metadata cannot be
    /// registered.
    pub fn pad(&self, config: PadConfig) -> Result<TracedTensor> {
        let op = StdTensorOp::Pad(config);
        let (out_rank, out_shape_hint) =
            infer_traced_single_output_shape("TracedTensor::pad", &op, &[self])?;
        apply_unary(op, self, out_rank, out_shape_hint)
    }

    /// Reverse the order of elements along the requested axes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when an axis is
    /// outside the input rank or `DuplicateAxis` when `axes` repeats one, or
    /// [`Error::RuntimeStateSource`] when result metadata cannot be
    /// registered.
    pub fn reverse(&self, axes: &[usize]) -> Result<TracedTensor> {
        validate_traced_axes(self.rank, axes, "TracedTensor::reverse")?;
        apply_unary(
            StdTensorOp::Reverse {
                axes: axes.to_vec(),
            },
            self,
            self.rank,
            self.shape_hint.clone(),
        )
    }

    /// Gather slices from `self` using integer start indices.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch`, `AxisOutOfBounds`,
    /// `DuplicateAxis`, or `ShapeMismatch` when indices or the gather
    /// configuration is incompatible with the input, and
    /// [`Error::RuntimeStateSource`] when output metadata cannot be
    /// registered.
    ///
    /// # Deferred errors
    ///
    /// Runtime index values are checked after binding. An out-of-range index
    /// is returned as [`Error::TensorRuntime`] with the backend's typed
    /// validation source and [`ErrorPhase::Execution`].
    pub fn gather(&self, indices: &TracedTensor, config: GatherConfig) -> Result<TracedTensor> {
        let op = StdTensorOp::Gather(config);
        let (out_rank, out_shape_hint) =
            infer_traced_single_output_shape("TracedTensor::gather", &op, &[self, indices])?;
        apply_binary_preserve_input_dtypes(op, self, indices, out_rank, out_shape_hint, self.dtype)
    }

    /// Scatter updates into `self` using StableHLO scatter semantics.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch`, `AxisOutOfBounds`,
    /// `DuplicateAxis`, or `ShapeMismatch` when indices, updates, or the
    /// scatter configuration is incompatible, [`Error::TensorRuntime`] with
    /// `UnsupportedDTypeConversion` when dtype promotion cannot be
    /// represented, or [`Error::RuntimeStateSource`] when output metadata
    /// cannot be registered.
    ///
    /// # Deferred errors
    ///
    /// Runtime index/update values are checked after binding. An invalid
    /// index or update shape is returned as [`Error::TensorRuntime`] with its
    /// typed validation source and [`ErrorPhase::Execution`].
    pub fn scatter(
        &self,
        indices: &TracedTensor,
        updates: &TracedTensor,
        config: ScatterConfig,
    ) -> Result<TracedTensor> {
        let op = StdTensorOp::Scatter(config);
        let (out_rank, out_shape_hint) = infer_traced_single_output_shape(
            "TracedTensor::scatter",
            &op,
            &[self, indices, updates],
        )?;
        let out_dtype = crate::shape_infer::promote_dtype(self.dtype, updates.dtype);
        let operand = if self.dtype != out_dtype {
            self.cast(out_dtype)?
        } else {
            self.clone()
        };
        let updates = if updates.dtype != out_dtype {
            updates.cast(out_dtype)?
        } else {
            updates.clone()
        };
        apply_ternary_with_output_dtype(
            op,
            &operand,
            indices,
            &updates,
            out_rank,
            out_shape_hint,
            out_dtype,
        )
    }

    /// Slice using runtime start indices.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `RankMismatch`, `AxisOutOfBounds`,
    /// or `InvalidArgument` when `starts` or `sizes` has an incompatible rank
    /// or extent, and [`Error::RuntimeStateSource`] when output metadata cannot
    /// be registered.
    ///
    /// # Deferred errors
    ///
    /// Runtime start values are checked after binding. An out-of-range start
    /// is returned as [`Error::TensorRuntime`] with the backend's typed
    /// validation source and [`ErrorPhase::Execution`].
    pub fn dynamic_slice(&self, starts: &TracedTensor, sizes: &[usize]) -> Result<TracedTensor> {
        let op = StdTensorOp::DynamicSlice {
            slice_sizes: sizes.to_vec(),
        };
        let (out_rank, out_shape_hint) =
            infer_traced_single_output_shape("TracedTensor::dynamic_slice", &op, &[self, starts])?;
        apply_binary_preserve_input_dtypes(op, self, starts, out_rank, out_shape_hint, self.dtype)
    }

    /// Keep the lower triangle and zero the rest.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let matrix = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
    /// let lower = matrix.tril(0)?;
    /// assert_eq!(lower.rank, 2);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when traced output metadata
    /// registration is unavailable or inconsistent with the graph.
    pub fn tril(&self, k: i64) -> Result<TracedTensor> {
        apply_unary(
            StdTensorOp::Tril { k },
            self,
            self.rank,
            self.shape_hint.clone(),
        )
    }

    /// Keep the upper triangle and zero the rest.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let matrix = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
    /// let upper = matrix.triu(0)?;
    /// assert_eq!(upper.rank, 2);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when traced output metadata
    /// registration is unavailable or inconsistent with the graph.
    pub fn triu(&self, k: i64) -> Result<TracedTensor> {
        apply_unary(
            StdTensorOp::Triu { k },
            self,
            self.rank,
            self.shape_hint.clone(),
        )
    }

    /// Permute tensor axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let y = x.transpose(&[1, 0])?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidPermutationLength`,
    /// `AxisOutOfBounds`, or `DuplicateAxis` when `perm` is not a valid
    /// permutation of the tensor axes, or [`Error::RuntimeStateSource`] when
    /// output metadata registration fails.
    pub fn transpose(&self, perm: &[usize]) -> Result<TracedTensor> {
        validate_traced_perm(self.rank, perm, "TracedTensor::transpose")?;
        let out_shape_hint = self
            .shape_hint
            .as_ref()
            .map(|shape| perm.iter().map(|&p| shape[p].clone()).collect());
        apply_unary(
            StdTensorOp::Transpose {
                perm: perm.to_vec(),
            },
            self,
            self.rank,
            out_shape_hint,
        )
    }

    /// Extract the diagonal along two axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    /// let y = x.extract_diag(0, 1)?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when either axis
    /// is outside the input rank or `InvalidArgument` when `axis_a == axis_b`.
    pub fn extract_diag(&self, axis_a: usize, axis_b: usize) -> Result<TracedTensor> {
        validate_traced_axis(self, axis_a, "TracedTensor::extract_diag")?;
        validate_traced_axis(self, axis_b, "TracedTensor::extract_diag")?;
        if axis_a == axis_b {
            return Err(graph_invalid_argument(
                "TracedTensor::extract_diag",
                "axes",
                "diagonal axes must be distinct",
            ));
        }
        let op = StdTensorOp::ExtractDiag { axis_a, axis_b };
        let (out_rank, out_shape_hint) =
            infer_traced_single_output_shape("TracedTensor::extract_diag", &op, &[self])?;
        apply_unary(op, self, out_rank, out_shape_hint)
    }

    /// Embed a vector or lower-rank tensor along a diagonal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    /// let y = x.embed_diag(0, 1)?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when `axis_a` is
    /// outside the input rank or `InvalidArgument` when `axis_b` is not a
    /// valid insertion axis.
    pub fn embed_diag(&self, axis_a: usize, axis_b: usize) -> Result<TracedTensor> {
        validate_traced_axis(self, axis_a, "TracedTensor::embed_diag")?;
        validate_traced_insert_axis(self.rank, axis_b, "TracedTensor::embed_diag")?;
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            let mut out_shape = shape.clone();
            out_shape.insert(axis_b, shape[axis_a].clone());
            out_shape
        });
        apply_unary(
            StdTensorOp::EmbedDiag { axis_a, axis_b },
            self,
            self.rank + 1,
            out_shape_hint,
        )
    }

    /// Return the runtime size of one axis as a scalar `f64` tensor.
    ///
    /// The result is metadata-derived and therefore has no gradient.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let cols = x.shape_of(1)?;
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&cols).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    /// assert_eq!(out.shape(), &[] as &[usize]);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when `axis` is
    /// outside the input rank, or [`Error::RuntimeStateSource`] when scalar
    /// output metadata cannot be registered.
    pub fn shape_of(&self, axis: usize) -> Result<TracedTensor> {
        validate_traced_axis(self, axis, "TracedTensor::shape_of")?;
        apply_unary_with_dtype(
            StdTensorOp::ShapeOf { axis },
            self,
            0,
            Some(vec![]),
            DType::F64,
        )
    }

    /// Truncate this tensor along `axis` to the first `size` elements.
    ///
    /// `size` is read at runtime from a scalar traced tensor. Values are
    /// rounded to the nearest integer, clamped to `[0, self.shape[axis]]`,
    /// and the output keeps the same element dtype as the input.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let size = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    /// let y = x.dynamic_truncate(&size, 0)?;
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    /// assert_eq!(out.shape(), &[2]);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when `axis` is
    /// outside the input rank or `RankMismatch` when `size` is not scalar.
    ///
    /// # Deferred errors
    ///
    /// At execution, non-`f32`/`f64`/`i64` size dtypes return
    /// [`Error::TensorRuntime`] with `Unsupported`, non-finite size values
    /// return a typed `InvalidArgument`, and an empty scalar buffer returns a
    /// typed runtime-state source.
    pub fn dynamic_truncate(&self, size: &TracedTensor, axis: usize) -> Result<TracedTensor> {
        validate_traced_axis(self, axis, "TracedTensor::dynamic_truncate")?;
        if size.rank != 0 {
            return Err(graph_validation(
                "TracedTensor::dynamic_truncate",
                ValidationError::RankMismatch {
                    expected: 0,
                    actual: size.rank,
                },
            ));
        }
        apply_binary_preserve_input_dtypes(
            StdTensorOp::DynamicTruncate { axis },
            self,
            size,
            self.rank,
            None,
            self.dtype,
        )
    }

    /// Pad this tensor with zeros along `axis` to match `reference.shape[axis]`.
    ///
    /// If `reference` is smaller along that axis, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let reference = TracedTensor::from_vec_col_major(vec![4], vec![0.0_f64, 0.0, 0.0, 0.0]).unwrap();
    /// let y = x.pad_to_match(&reference, 0)?;
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    /// assert_eq!(out.shape(), &[4]);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when `axis` is
    /// outside either tensor's rank, or [`Error::RuntimeStateSource`] when
    /// output metadata cannot be registered.
    pub fn pad_to_match(&self, reference: &TracedTensor, axis: usize) -> Result<TracedTensor> {
        validate_traced_axis(self, axis, "TracedTensor::pad_to_match")?;
        validate_traced_axis(reference, axis, "TracedTensor::pad_to_match")?;
        let op = StdTensorOp::PadToMatch { axis };
        let (out_rank, out_shape_hint) = infer_traced_single_output_shape(
            "TracedTensor::pad_to_match",
            &op,
            &[self, reference],
        )?;
        apply_binary_preserve_input_dtypes(
            op,
            self,
            reference,
            out_rank,
            out_shape_hint,
            self.dtype,
        )
    }
}

pub(crate) fn apply_unary(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
) -> Result<TracedTensor> {
    let out_dtype = try_inferred_output_dtype(&op, &[input.dtype], "apply_unary")?;
    apply_unary_with_dtype(op, input, out_rank, out_shape_hint, out_dtype)
}

fn try_apply_unary(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    context: &'static str,
) -> Result<TracedTensor> {
    let out_dtype = try_inferred_output_dtype(&op, &[input.dtype], context)?;
    apply_unary_with_dtype(op, input, out_rank, out_shape_hint, out_dtype)
}

pub(crate) fn apply_unary_with_dtype(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    out_dtype: DType,
) -> Result<TracedTensor> {
    let mut builder = GraphBuilder::new();
    builder.add_parent(input.graph.clone());
    let input_ref = ValueRef::External(input.graph.values()[input.val].key.clone());
    let outputs = builder.add_operation(op, vec![input_ref], OperationRole::Primary);
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(graph.as_ref(), outputs[0], out_dtype, &out_shape_hint)?;

    Ok(TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: out_dtype,
        graph,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: input.inputs_map.clone(),
        extra_roots: input.extra_roots.clone(),
        checkpoint_chain: input.checkpoint_chain.clone(),
        metadata_scopes: MetadataScopeChain::with_new(metadata_scope, [&input.metadata_scopes]),
        constraint_scopes: input.constraint_scopes.clone(),
    })
}

/// Apply a unary-primary op that additionally references one or more
/// tensors for shape resolution only.
///
/// The primary `input` becomes op input 0; each tensor in `shape_refs`
/// becomes op input 1, 2, … in order. Used by
/// [`TracedTensor::broadcast_in_dim_sym`] when the target shape
/// references axes of tensors other than the primary input; the op
/// reads only their runtime shape, not their data.
pub(crate) fn apply_unary_with_shape_refs(
    op: StdTensorOp,
    input: &TracedTensor,
    shape_refs: &[&TracedTensor],
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
) -> Result<TracedTensor> {
    let mut builder = GraphBuilder::new();
    builder.add_parent(input.graph.clone());
    for t in shape_refs {
        builder.add_parent(t.graph.clone());
    }
    let mut op_inputs: Vec<ValueRef<StdTensorOp>> = Vec::with_capacity(1 + shape_refs.len());
    op_inputs.push(ValueRef::External(
        input.graph.values()[input.val].key.clone(),
    ));
    for t in shape_refs {
        op_inputs.push(ValueRef::External(t.graph.values()[t.val].key.clone()));
    }
    let outputs = builder.add_operation(op, op_inputs, OperationRole::Primary);
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(graph.as_ref(), outputs[0], input.dtype, &out_shape_hint)?;

    let inputs_map =
        merge_traced_inputs_map(std::iter::once(input).chain(shape_refs.iter().copied()));

    let mut extra_roots = input.extra_roots.clone();
    for t in shape_refs {
        extra_roots.extend(t.extra_roots.iter().cloned());
    }

    let mut checkpoint_chain = input.checkpoint_chain.clone();
    for t in shape_refs {
        checkpoint_chain =
            CheckpointNode::merge_chains(checkpoint_chain, t.checkpoint_chain.clone());
    }

    Ok(TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: input.dtype,
        graph,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map,
        extra_roots,
        checkpoint_chain,
        metadata_scopes: MetadataScopeChain::with_new(
            metadata_scope,
            std::iter::once(&input.metadata_scopes)
                .chain(shape_refs.iter().map(|tensor| &tensor.metadata_scopes)),
        ),
        constraint_scopes: ConstraintScopeChain::merge(
            std::iter::once(&input.constraint_scopes)
                .chain(shape_refs.iter().map(|tensor| &tensor.constraint_scopes)),
        ),
    })
}

pub(crate) fn apply_nullary(
    op: StdTensorOp,
    rank: usize,
    dtype: DType,
    shape_hint: Option<Vec<SymDim>>,
) -> Result<TracedTensor> {
    let mut builder = GraphBuilder::new();
    let outputs = builder.add_operation(op, vec![], OperationRole::Primary);
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(graph.as_ref(), outputs[0], dtype, &shape_hint)?;

    Ok(TracedTensor {
        id: next_traced_id(),
        rank,
        dtype,
        graph,
        val: outputs[0],
        data: None,
        shape_hint,
        inputs_map: Arc::new(HashMap::new()),
        extra_roots: Vec::new(),
        checkpoint_chain: None,
        metadata_scopes: MetadataScopeChain::from_scope(metadata_scope),
        constraint_scopes: ConstraintScopeChain::empty(),
    })
}

pub(crate) fn apply_binary(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
) -> Result<TracedTensor> {
    let input_dtype = crate::shape_infer::promote_dtype_for_binary_op(&op, lhs.dtype, rhs.dtype);
    let out_dtype = try_inferred_output_dtype(&op, &[lhs.dtype, rhs.dtype], "apply_binary")?;

    // Insert Convert ops when an input dtype differs from the primitive input dtype.
    let lhs = if lhs.dtype != input_dtype {
        lhs.cast(input_dtype)?
    } else {
        lhs.clone()
    };
    let rhs = if rhs.dtype != input_dtype {
        rhs.cast(input_dtype)?
    } else {
        rhs.clone()
    };

    apply_binary_with_output_dtype(op, &lhs, &rhs, out_rank, out_shape_hint, out_dtype)
}

fn try_apply_binary(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    context: &'static str,
) -> Result<TracedTensor> {
    let input_dtype = crate::shape_infer::promote_dtype_for_binary_op(&op, lhs.dtype, rhs.dtype);
    let out_dtype = try_inferred_output_dtype(&op, &[lhs.dtype, rhs.dtype], context)?;

    let lhs = if lhs.dtype != input_dtype {
        lhs.cast(input_dtype)?
    } else {
        lhs.clone()
    };
    let rhs = if rhs.dtype != input_dtype {
        rhs.cast(input_dtype)?
    } else {
        rhs.clone()
    };

    apply_binary_with_output_dtype(op, &lhs, &rhs, out_rank, out_shape_hint, out_dtype)
}

pub(crate) fn apply_binary_preserve_input_dtypes(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    out_dtype: DType,
) -> Result<TracedTensor> {
    apply_binary_with_output_dtype(op, lhs, rhs, out_rank, out_shape_hint, out_dtype)
}

pub(crate) fn apply_broadcast_binary_op(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
) -> Result<TracedTensor> {
    let (lhs, rhs) = broadcast_binary(lhs, rhs)?;
    try_apply_binary(
        op,
        &lhs,
        &rhs,
        lhs.rank,
        lhs.shape_hint.clone(),
        "broadcast_binary",
    )
}

pub(crate) fn apply_broadcast_ternary_op(
    op: StdTensorOp,
    first: &TracedTensor,
    second: &TracedTensor,
    third: &TracedTensor,
) -> Result<TracedTensor> {
    let (first, second, third) = broadcast_ternary(first, second, third)?;
    try_apply_ternary(
        op,
        &first,
        &second,
        &third,
        first.rank,
        first.shape_hint.clone(),
        "broadcast_ternary",
    )
}

fn try_apply_ternary(
    op: StdTensorOp,
    first: &TracedTensor,
    second: &TracedTensor,
    third: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    context: &'static str,
) -> Result<TracedTensor> {
    let out_dtype =
        try_inferred_output_dtype(&op, &[first.dtype, second.dtype, third.dtype], context)?;
    let (first, second, third) = match op {
        StdTensorOp::Select => {
            let value_dtype = crate::shape_infer::promote_dtype(second.dtype, third.dtype);
            let second = if second.dtype != value_dtype {
                second.cast(value_dtype)?
            } else {
                second.clone()
            };
            let third = if third.dtype != value_dtype {
                third.cast(value_dtype)?
            } else {
                third.clone()
            };
            (first.clone(), second, third)
        }
        _ => {
            let input_dtype =
                crate::shape_infer::promote_dtypes([first.dtype, second.dtype, third.dtype]);
            let first = if first.dtype != input_dtype {
                first.cast(input_dtype)?
            } else {
                first.clone()
            };
            let second = if second.dtype != input_dtype {
                second.cast(input_dtype)?
            } else {
                second.clone()
            };
            let third = if third.dtype != input_dtype {
                third.cast(input_dtype)?
            } else {
                third.clone()
            };
            (first, second, third)
        }
    };
    apply_ternary_with_output_dtype(
        op,
        &first,
        &second,
        &third,
        out_rank,
        out_shape_hint,
        out_dtype,
    )
}

fn apply_binary_with_output_dtype(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    out_dtype: DType,
) -> Result<TracedTensor> {
    let lhs_ref = ValueRef::External(lhs.graph.values()[lhs.val].key.clone());
    let rhs_ref = ValueRef::External(rhs.graph.values()[rhs.val].key.clone());

    let mut builder = GraphBuilder::new();
    builder.add_parent(lhs.graph.clone());
    builder.add_parent(rhs.graph.clone());
    let outputs = builder.add_operation(op, vec![lhs_ref, rhs_ref], OperationRole::Primary);
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(graph.as_ref(), outputs[0], out_dtype, &out_shape_hint)?;

    let mut extra_roots = lhs.extra_roots.clone();
    extra_roots.extend(rhs.extra_roots.iter().cloned());

    Ok(TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: out_dtype,
        graph,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: merge_traced_inputs_map([lhs, rhs]),
        extra_roots,
        checkpoint_chain: CheckpointNode::merge_chains(
            lhs.checkpoint_chain.clone(),
            rhs.checkpoint_chain.clone(),
        ),
        metadata_scopes: MetadataScopeChain::with_new(
            metadata_scope,
            [&lhs.metadata_scopes, &rhs.metadata_scopes],
        ),
        constraint_scopes: ConstraintScopeChain::merge([
            &lhs.constraint_scopes,
            &rhs.constraint_scopes,
        ]),
    })
}

fn apply_ternary_with_output_dtype(
    op: StdTensorOp,
    first: &TracedTensor,
    second: &TracedTensor,
    third: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    out_dtype: DType,
) -> Result<TracedTensor> {
    let first_ref = ValueRef::External(first.graph.values()[first.val].key.clone());
    let second_ref = ValueRef::External(second.graph.values()[second.val].key.clone());
    let third_ref = ValueRef::External(third.graph.values()[third.val].key.clone());

    let mut builder = GraphBuilder::new();
    builder.add_parent(first.graph.clone());
    builder.add_parent(second.graph.clone());
    builder.add_parent(third.graph.clone());
    let outputs = builder.add_operation(
        op,
        vec![first_ref, second_ref, third_ref],
        OperationRole::Primary,
    );
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(graph.as_ref(), outputs[0], out_dtype, &out_shape_hint)?;

    let mut extra_roots = first.extra_roots.clone();
    extra_roots.extend(second.extra_roots.iter().cloned());
    extra_roots.extend(third.extra_roots.iter().cloned());

    let checkpoint_chain = CheckpointNode::merge_chains(
        CheckpointNode::merge_chains(
            first.checkpoint_chain.clone(),
            second.checkpoint_chain.clone(),
        ),
        third.checkpoint_chain.clone(),
    );

    Ok(TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: out_dtype,
        graph,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: merge_traced_inputs_map([first, second, third]),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: MetadataScopeChain::with_new(
            metadata_scope,
            [
                &first.metadata_scopes,
                &second.metadata_scopes,
                &third.metadata_scopes,
            ],
        ),
        constraint_scopes: ConstraintScopeChain::merge([
            &first.constraint_scopes,
            &second.constraint_scopes,
            &third.constraint_scopes,
        ]),
    })
}

fn register_single_output_metadata(
    graph: &Graph<StdTensorOp>,
    output: LocalValueId,
    dtype: DType,
    shape_hint: &Option<Vec<SymDim>>,
) -> Result<GlobalMetadataScope> {
    if let Some(shape) = shape_hint {
        // Fresh graph output keys are generated in this builder, so metadata
        // registration failure would indicate a global metadata invariant bug.
        register_metadata_or_runtime_state(register_scoped_value_metadata(
            graph.values()[output].key.clone(),
            tensor_meta(dtype, shape.clone()),
        ))
    } else {
        // Fresh graph output keys are generated in this builder, so metadata
        // registration failure would indicate a global metadata invariant bug.
        register_metadata_or_runtime_state(register_scoped_graph_metadata(
            graph,
            std::iter::empty(),
        ))
    }
}

impl TracedTensor {
    pub(crate) fn resolve_roots(&self) -> Vec<Arc<Graph<StdTensorOp>>> {
        let mut roots = Vec::with_capacity(1 + self.extra_roots.len());
        roots.push(self.graph.clone());
        roots.extend(self.extra_roots.iter().cloned());
        roots
    }
}

#[cfg(test)]
mod tests;
