use std::ops::Range;
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueRef};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{
    GatherConfig, ShapeMismatch, ShapeVec, SliceConfig, Tensor, TypedTensor, ValidationError,
};

use crate::checkpoint::CheckpointNode;
use crate::error::{Error, ErrorPhase, Result};
use crate::metadata::{register_scoped_value_metadata, tensor_meta, MetadataScopeChain};
use crate::shape_constraint::ConstraintScopeChain;
use crate::shape_infer::promote_dtypes;
use crate::sym_dim::SymDim;
use crate::traced::{
    apply_binary_preserve_input_dtypes, infer_traced_single_output_shape, merge_traced_inputs_map,
    merge_traced_leaf_metas, next_traced_id, try_concrete_shape,
};
use crate::TracedTensor;

fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        rank.checked_sub(axis.unsigned_abs()).ok_or_else(|| {
            Error::validation(
                op,
                ErrorPhase::GraphBuild,
                ValidationError::AxisOutOfBounds {
                    axis: axis.unsigned_abs(),
                    rank,
                },
            )
        })?
    };
    if normalized >= rank {
        return Err(Error::validation(
            op,
            ErrorPhase::GraphBuild,
            ValidationError::AxisOutOfBounds {
                axis: axis.unsigned_abs(),
                rank,
            },
        ));
    }
    Ok(normalized)
}

fn normalize_insert_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let insert_rank = rank.checked_add(1).ok_or_else(|| {
        Error::validation(
            op,
            ErrorPhase::GraphBuild,
            ValidationError::AxisOutOfBounds {
                axis: axis.unsigned_abs(),
                rank,
            },
        )
    })?;
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        insert_rank
            .checked_sub(axis.unsigned_abs())
            .ok_or_else(|| {
                Error::validation(
                    op,
                    ErrorPhase::GraphBuild,
                    ValidationError::AxisOutOfBounds {
                        axis: axis.unsigned_abs(),
                        rank: insert_rank,
                    },
                )
            })?
    };
    if normalized > rank {
        return Err(Error::validation(
            op,
            ErrorPhase::GraphBuild,
            ValidationError::AxisOutOfBounds {
                axis: axis.unsigned_abs(),
                rank: insert_rank,
            },
        ));
    }
    Ok(normalized)
}

fn index_select_config(
    shape: &[usize],
    axis: isize,
    positions: &[usize],
) -> Result<(Tensor, GatherConfig, Vec<usize>)> {
    let axis = normalize_existing_axis("index_select", axis, shape.len())?;
    let axis_extent = shape[axis];
    for &position in positions {
        if position >= axis_extent {
            return Err(Error::validation(
                "index_select",
                ErrorPhase::GraphBuild,
                ValidationError::InvalidArgument {
                    argument: "positions",
                    message: format!(
                    "position {position} out of bounds for axis {axis} with extent {axis_extent}"
                    ),
                },
            ));
        }
    }

    let mut out_shape = shape.to_vec();
    out_shape[axis] = positions.len();

    let mut slice_sizes = shape.to_vec();
    slice_sizes[axis] = 1;

    let offset_dims = (0..shape.len()).filter(|&dim| dim != axis).collect();
    let index_data = positions
        .iter()
        .map(|&position| {
            i64::try_from(position).map_err(|_| {
                Error::validation(
                    "index_select",
                    ErrorPhase::GraphBuild,
                    ValidationError::InvalidArgument {
                        argument: "positions",
                        message: format!("position {position} cannot be represented as i64"),
                    },
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let indices = Tensor::I64(TypedTensor::from_vec_col_major(
        vec![positions.len(), 1],
        index_data,
    )?);

    let config = GatherConfig {
        offset_dims,
        collapsed_slice_dims: vec![axis],
        start_index_map: vec![axis],
        index_vector_dim: 1,
        slice_sizes,
    };

    Ok((indices, config, out_shape))
}

fn validate_stack_shapes(op: &'static str, shapes: &[&[usize]]) -> Result<()> {
    let Some(first) = shapes.first() else {
        return Err(Error::validation(
            op,
            ErrorPhase::GraphBuild,
            ValidationError::InvalidArgument {
                argument: "tensors",
                message: "stack requires at least one input".into(),
            },
        ));
    };
    for shape in shapes.iter().skip(1) {
        if *shape != *first {
            return Err(Error::validation(
                op,
                ErrorPhase::GraphBuild,
                ShapeMismatch::IncompatibleShapes {
                    lhs: ShapeVec::from_vec(first.to_vec()),
                    rhs: ShapeVec::from_vec(shape.to_vec()),
                }
                .into(),
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
enum AxisSelection {
    Slice {
        axis: usize,
        range: Range<usize>,
        step: usize,
    },
    Take {
        axis: usize,
        indices: Vec<usize>,
    },
}

fn concrete_shape_for_axis_slice(tensor: &TracedTensor, op: &'static str) -> Result<Vec<usize>> {
    try_concrete_shape(tensor).ok_or_else(|| {
        Error::validation(
            op,
            ErrorPhase::GraphBuild,
            ValidationError::InvalidArgument {
                argument: "shape",
                message: format!("{op} requires a concrete shape hint"),
            },
        )
    })
}

fn validate_axis_selection(
    op: &'static str,
    rank: usize,
    seen: &mut [bool],
    axis: usize,
) -> Result<()> {
    if axis >= rank {
        return Err(Error::validation(
            op,
            ErrorPhase::GraphBuild,
            ValidationError::AxisOutOfBounds { axis, rank },
        ));
    }
    if seen[axis] {
        return Err(Error::validation(
            op,
            ErrorPhase::GraphBuild,
            ValidationError::DuplicateAxis {
                axis,
                role: "selection",
            },
        ));
    }
    seen[axis] = true;
    Ok(())
}

fn apply_slice_axis_config(
    op: &'static str,
    shape: &[usize],
    selections: &[AxisSelection],
) -> Result<Option<SliceConfig>> {
    let mut starts = vec![0; shape.len()];
    let mut limits = shape.to_vec();
    let mut strides = vec![1; shape.len()];
    let mut has_slice = false;
    for selection in selections {
        let AxisSelection::Slice { axis, range, step } = selection else {
            continue;
        };
        if *step == 0 {
            return Err(Error::validation(
                op,
                ErrorPhase::GraphBuild,
                ValidationError::InvalidSliceStep { step: 0 },
            ));
        }
        let extent = shape[*axis];
        if range.start > range.end || range.end > extent {
            let start = isize::try_from(range.start).map_err(|_| {
                Error::validation(op, ErrorPhase::GraphBuild, ValidationError::IntegerOverflow)
            })?;
            let end = isize::try_from(range.end).map_err(|_| {
                Error::validation(op, ErrorPhase::GraphBuild, ValidationError::IntegerOverflow)
            })?;
            return Err(Error::validation(
                op,
                ErrorPhase::GraphBuild,
                ValidationError::InvalidSliceBounds {
                    start,
                    end,
                    axis_len: extent,
                },
            ));
        }
        starts[*axis] = range.start;
        limits[*axis] = range.end;
        strides[*axis] = *step;
        has_slice = true;
    }
    Ok(has_slice.then_some(SliceConfig {
        starts,
        limits,
        strides,
    }))
}

/// Rank-preserving traced tensor slicing builder.
///
/// Unspecified axes are kept whole. Range selections become one `Slice`
/// operation; host-known position selections become `Gather`/`index_select`
/// operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::TracedTensor;
///
/// let x = TracedTensor::from_vec_col_major(vec![3, 4], vec![0.0_f64; 12]).unwrap();
/// let y = x.slice_builder().axis(0, 0..2).axis_step(1, 0..4, 2).apply().unwrap();
/// assert_eq!(y.try_concrete_shape(), Some(vec![2, 2]));
/// ```
#[derive(Clone, Debug)]
pub struct TracedSliceBuilder<'a> {
    tensor: &'a TracedTensor,
    selections: Vec<AxisSelection>,
}

impl<'a> TracedSliceBuilder<'a> {
    fn new(tensor: &'a TracedTensor) -> Self {
        Self {
            tensor,
            selections: Vec::new(),
        }
    }

    /// Add an exclusive-end range selection for one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let y = x.slice_builder().axis(0, 1..3).apply().unwrap();
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// ```
    pub fn axis(mut self, axis: usize, range: Range<usize>) -> Self {
        self.selections.push(AxisSelection::Slice {
            axis,
            range,
            step: 1,
        });
        self
    }

    /// Add an exclusive-end strided range selection for one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    /// let y = x.slice_builder().axis_step(0, 0..5, 2).apply().unwrap();
    /// assert_eq!(y.try_concrete_shape(), Some(vec![3]));
    /// ```
    pub fn axis_step(mut self, axis: usize, range: Range<usize>, step: usize) -> Self {
        self.selections
            .push(AxisSelection::Slice { axis, range, step });
        self
    }

    /// Add a host-known position selection for one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    /// let y = x.slice_builder().take_axis(0, &[2, 0]).apply().unwrap();
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// ```
    pub fn take_axis(mut self, axis: usize, indices: &[usize]) -> Self {
        self.selections.push(AxisSelection::Take {
            axis,
            indices: indices.to_vec(),
        });
        self
    }

    /// Build and apply the requested slice/take operations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let y = x.slice_builder().axis(0, 1..4).apply().unwrap();
    /// assert_eq!(y.try_concrete_shape(), Some(vec![3]));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` or
    /// `DuplicateAxis` when selections are invalid, and propagates the
    /// underlying [`Error::Validation`] from a slice/take graph operation
    /// that cannot be built.
    pub fn apply(self) -> Result<TracedTensor> {
        let shape = concrete_shape_for_axis_slice(self.tensor, "slice_builder")?;
        let mut seen = vec![false; shape.len()];
        for selection in &self.selections {
            let axis = match selection {
                AxisSelection::Slice { axis, .. } | AxisSelection::Take { axis, .. } => *axis,
            };
            validate_axis_selection("slice_builder", shape.len(), &mut seen, axis)?;
        }

        let mut output = self.tensor.clone();
        if let Some(config) = apply_slice_axis_config("slice_builder", &shape, &self.selections)? {
            output = output.slice(config)?;
        }
        for selection in self.selections {
            if let AxisSelection::Take { axis, indices } = selection {
                output = output.take_axis(axis, &indices)?;
            }
        }
        Ok(output)
    }
}

impl TracedTensor {
    /// Slice one axis with an exclusive-end range, keeping all other axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let y = x.slice_axis(0, 1..3).unwrap();
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when `axis` is
    /// outside the concrete rank, or `InvalidArgument` when `range` is
    /// outside the selected axis extent.
    pub fn slice_axis(&self, axis: usize, range: Range<usize>) -> Result<Self> {
        self.slice_builder().axis(axis, range).apply()
    }

    /// Start a rank-preserving slicing builder for this tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    /// let y = x.slice_builder().axis(0, 0..2).apply().unwrap();
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// ```
    pub fn slice_builder(&self) -> TracedSliceBuilder<'_> {
        TracedSliceBuilder::new(self)
    }

    /// Select entries from one axis using host-known indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();
    /// let y = x.take_axis(0, &[2, 0]).unwrap();
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds` when `axis` is
    /// outside the concrete rank, or `InvalidArgument` when an index list
    /// cannot be applied to the selected axis.
    pub fn take_axis(&self, axis: usize, indices: &[usize]) -> Result<Self> {
        let axis = isize::try_from(axis).map_err(|_| {
            Error::validation(
                "take_axis",
                ErrorPhase::GraphBuild,
                ValidationError::InvalidArgument {
                    argument: "axis",
                    message: format!("axis {axis} cannot be represented as isize"),
                },
            )
        })?;
        self.index_select(axis, indices)
    }

    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, Runtime, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::from_tensor_concrete_shape(
    ///     Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap(),
    /// )
    /// .unwrap();
    /// let y = x.index_select(-1, &[2, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let backend = CpuBackend::new();
    /// let mut builder = Runtime::builder();
    /// builder
    ///     .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
    ///     .unwrap();
    /// let runtime = builder.build().unwrap();
    /// let outputs = runtime.run_compiled(&program, &[]).unwrap();
    /// let out = &outputs[0];
    ///
    /// assert_eq!(
    ///     out.as_slice::<f64>().unwrap(),
    ///     &[30.0, 10.0],
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument` when the tensor
    /// shape is not concrete, `AxisOutOfBounds` when `axis` is outside its
    /// rank, or `InvalidArgument` when a position is outside the selected
    /// axis extent.
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> Result<Self> {
        let shape = try_concrete_shape(self).ok_or_else(|| {
            Error::validation(
                "index_select",
                ErrorPhase::GraphBuild,
                ValidationError::InvalidArgument {
                    argument: "shape",
                    message: "index_select requires a concrete shape hint".into(),
                },
            )
        })?;
        let (indices_tensor, config, out_shape) = index_select_config(&shape, axis, positions)?;
        let indices = TracedTensor::from_tensor_concrete_shape(indices_tensor)?;
        apply_binary_preserve_input_dtypes(
            StdTensorOp::Gather(config),
            self,
            &indices,
            out_shape.len(),
            Some(out_shape.into_iter().map(SymDim::from).collect()),
            self.dtype,
        )
    }

    /// Stack tensors along a newly inserted axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, Runtime, Tensor, TracedTensor};
    ///
    /// let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap()).unwrap();
    /// let b = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap()).unwrap();
    /// let stacked = TracedTensor::stack(&[&a, &b], -1).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&stacked).unwrap();
    /// let backend = CpuBackend::new();
    /// let mut builder = Runtime::builder();
    /// builder
    ///     .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
    ///     .unwrap();
    /// let runtime = builder.build().unwrap();
    /// let outputs = runtime.run_compiled(&program, &[]).unwrap();
    /// let out = &outputs[0];
    ///
    /// assert_eq!(
    ///     out.as_slice::<f64>().unwrap(),
    ///     &[1.0, 2.0],
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument` for an empty input
    /// list, `ShapeMismatch` for incompatible input shapes, or
    /// `AxisOutOfBounds` when `dim` is outside the output rank.
    pub fn stack(tensors: &[&Self], dim: isize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| {
            Error::validation(
                "stack",
                ErrorPhase::GraphBuild,
                ValidationError::InvalidArgument {
                    argument: "tensors",
                    message: "stack requires at least one input".into(),
                },
            )
        })?;
        let first_shape = try_concrete_shape(first).ok_or_else(|| {
            Error::validation(
                "stack",
                ErrorPhase::GraphBuild,
                ValidationError::InvalidArgument {
                    argument: "shape",
                    message: "stack requires concrete shape hints".into(),
                },
            )
        })?;
        let mut shapes = Vec::with_capacity(tensors.len());
        shapes.push(first_shape.as_slice());
        let mut owned_shapes = Vec::with_capacity(tensors.len().saturating_sub(1));
        for tensor in tensors.iter().copied().skip(1) {
            owned_shapes.push(try_concrete_shape(tensor).ok_or_else(|| {
                Error::validation(
                    "stack",
                    ErrorPhase::GraphBuild,
                    ValidationError::InvalidArgument {
                        argument: "shape",
                        message: "stack requires concrete shape hints".into(),
                    },
                )
            })?);
        }
        shapes.extend(owned_shapes.iter().map(Vec::as_slice));
        validate_stack_shapes("stack", &shapes)?;

        let axis = normalize_insert_axis("stack", dim, first.rank)?;
        let mut expanded_shape = first_shape;
        expanded_shape.insert(axis, 1);
        let mut out_shape = expanded_shape.clone();
        out_shape[axis] = tensors.len();
        let expanded = tensors
            .iter()
            .map(|tensor| tensor.reshape(&expanded_shape))
            .collect::<Result<Vec<_>>>()?;
        let refs = expanded.iter().collect::<Vec<_>>();
        apply_nary_concatenate(
            &refs,
            axis,
            out_shape.into_iter().map(SymDim::from).collect(),
        )
    }

    /// Concatenate tensors along one existing axis.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument` for an empty input
    /// list, `RankMismatch`/`ShapeMismatch` for incompatible input shapes, or
    /// `AxisOutOfBounds` when `axis` is outside the input rank.
    pub fn concatenate(tensors: &[&Self], axis: usize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| {
            Error::validation(
                "concatenate",
                ErrorPhase::GraphBuild,
                ValidationError::InvalidArgument {
                    argument: "tensors",
                    message: "concatenate requires at least one input".into(),
                },
            )
        })?;
        if axis >= first.rank {
            return Err(Error::validation(
                "concatenate",
                ErrorPhase::GraphBuild,
                ValidationError::AxisOutOfBounds {
                    axis,
                    rank: first.rank,
                },
            ));
        }
        for tensor in tensors.iter().copied().skip(1) {
            if tensor.rank != first.rank {
                return Err(Error::validation(
                    "concatenate",
                    ErrorPhase::GraphBuild,
                    ValidationError::RankMismatch {
                        expected: first.rank,
                        actual: tensor.rank,
                    },
                ));
            }
        }

        let op = StdTensorOp::Concatenate {
            axis,
            input_count: tensors.len(),
        };
        let (_, out_shape_hint) =
            infer_traced_single_output_shape("TracedTensor::concatenate", &op, tensors)?;
        let out_shape = out_shape_hint.ok_or_else(|| {
            Error::Internal("concatenate shape inference returned no shape hint".into())
        })?;
        apply_nary_concatenate(tensors, axis, out_shape)
    }
}

fn apply_nary_concatenate(
    tensors: &[&TracedTensor],
    axis: usize,
    out_shape: Vec<SymDim>,
) -> Result<TracedTensor> {
    let out_dtype = promote_dtypes(tensors.iter().map(|tensor| tensor.dtype));
    let tensors = tensors
        .iter()
        .map(|tensor| {
            if tensor.dtype != out_dtype {
                tensor.cast(out_dtype)
            } else {
                Ok((*tensor).clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let mut builder = GraphBuilder::new();
    for tensor in &tensors {
        builder.add_parent(tensor.graph.clone());
    }
    let input_refs = tensors
        .iter()
        .map(|tensor| ValueRef::External(tensor.graph.values()[tensor.val].key.clone()))
        .collect::<Vec<_>>();
    let outputs = builder.add_operation(
        StdTensorOp::Concatenate {
            axis,
            input_count: tensors.len(),
        },
        input_refs,
        OperationRole::Primary,
    );
    builder.set_outputs(outputs.clone());
    let graph = Arc::new(builder.build());
    // Callers route through shape inference before graph construction.
    let metadata_scope =
        super::traced::register_metadata_or_runtime_state(register_scoped_value_metadata(
            graph.values()[outputs[0]].key.clone(),
            tensor_meta(out_dtype, out_shape.clone()),
        ))?;

    let inputs_map = merge_traced_inputs_map(tensors.iter());
    let leaf_metas = merge_traced_leaf_metas(tensors.iter());
    let mut extra_roots = Vec::new();
    let mut checkpoint_chain = None;
    for tensor in &tensors {
        extra_roots.extend(tensor.extra_roots.iter().cloned());
        checkpoint_chain =
            CheckpointNode::merge_chains(checkpoint_chain, tensor.checkpoint_chain.clone());
    }
    Ok(TracedTensor {
        id: next_traced_id(),
        rank: out_shape.len(),
        dtype: out_dtype,
        graph,
        val: outputs[0],
        data: None,
        shape_hint: Some(out_shape),
        inputs_map,
        leaf_metas,
        extra_roots,
        checkpoint_chain,
        metadata_scopes: MetadataScopeChain::with_new(
            metadata_scope,
            tensors.iter().map(|tensor| &tensor.metadata_scopes),
        ),
        constraint_scopes: ConstraintScopeChain::merge(
            tensors.iter().map(|tensor| &tensor.constraint_scopes),
        ),
    })
}

#[cfg(test)]
mod tests;
