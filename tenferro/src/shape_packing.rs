use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{OpMode, ValRef};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{GatherConfig, Tensor, TypedTensor};

use crate::checkpoint::CheckpointNode;
use crate::eager::EagerTensor;
use crate::error::{Error, Result};
use crate::metadata::{metadata_scopes_with_new, register_scoped_fragment_metadata};
use crate::shape_infer::promote_dtypes;
use crate::sym_dim::SymDim;
use crate::traced::{apply_binary_preserve_input_dtypes, next_traced_id, try_concrete_shape};
use crate::TracedTensor;

fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 { rank as isize + axis } else { axis };
    if normalized < 0 || normalized >= rank as isize {
        return Err(tenferro_tensor::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank,
        }
        .into());
    }
    Ok(normalized as usize)
}

fn normalize_insert_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 {
        rank as isize + 1 + axis
    } else {
        axis
    };
    if normalized < 0 || normalized > rank as isize {
        return Err(tenferro_tensor::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank: rank + 1,
        }
        .into());
    }
    Ok(normalized as usize)
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
            return Err(tenferro_tensor::Error::InvalidConfig {
                op: "index_select",
                message: format!(
                    "position {position} out of bounds for axis {axis} with extent {axis_extent}"
                ),
            }
            .into());
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
            i64::try_from(position).map_err(|_| tenferro_tensor::Error::InvalidConfig {
                op: "index_select",
                message: format!("position {position} cannot be represented as i64"),
            })
        })
        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
    let indices = Tensor::I64(TypedTensor::from_vec(vec![positions.len(), 1], index_data));

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
        return Err(tenferro_tensor::Error::InvalidConfig {
            op,
            message: "stack requires at least one input".into(),
        }
        .into());
    };
    for shape in shapes.iter().skip(1) {
        if *shape != *first {
            return Err(tenferro_tensor::Error::ShapeMismatch {
                op,
                lhs: first.to_vec(),
                rhs: shape.to_vec(),
            }
            .into());
        }
    }
    Ok(())
}

impl EagerTensor {
    /// Select entries from one axis using host-known indices.
    ///
    /// The index list is primal metadata: gradients flow to `self`, including
    /// accumulation for repeated indices, but not to the selected positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]),
    ///     ctx,
    /// );
    /// let y = x.take_axis(0, &[2, 0]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[30.0, 10.0]);
    /// ```
    pub fn take_axis(&self, axis: usize, indices: &[usize]) -> Result<Self> {
        let axis = isize::try_from(axis).map_err(|_| {
            Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
                op: "take_axis",
                message: format!("axis {axis} cannot be represented as isize"),
            })
        })?;
        self.index_select(axis, indices)
    }

    /// Select matrix rows using host-known row indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
    ///     ctx,
    /// );
    /// let y = x.take_rows(&[1]).unwrap();
    ///
    /// assert_eq!(y.data().shape(), &[1, 2]);
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// ```
    pub fn take_rows(&self, rows: &[usize]) -> Result<Self> {
        self.take_axis(0, rows)
    }

    /// Select matrix columns using host-known column indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
    ///     ctx,
    /// );
    /// let y = x.take_cols(&[1]).unwrap();
    ///
    /// assert_eq!(y.data().shape(), &[2, 1]);
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[3.0, 4.0]);
    /// ```
    pub fn take_cols(&self, cols: &[usize]) -> Result<Self> {
        self.take_axis(1, cols)
    }

    /// Select a matrix block using host-known row and column indices.
    ///
    /// This is a convenience wrapper over row selection followed by column
    /// selection. The row and column lists, plus the approximation rank implied
    /// by their lengths, are fixed primal metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
    ///     ctx,
    /// );
    /// let y = x.take_block(&[1], &[0]).unwrap();
    ///
    /// assert_eq!(y.data().shape(), &[1, 1]);
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[2.0]);
    /// ```
    pub fn take_block(&self, rows: &[usize], cols: &[usize]) -> Result<Self> {
        self.take_rows(rows)?.take_cols(cols)
    }

    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]),
    ///     ctx,
    /// );
    /// let y = x.index_select(-1, &[2, 0]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[30.0, 10.0]);
    /// ```
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> Result<Self> {
        let (indices, config, _) = index_select_config(self.data().shape(), axis, positions)?;
        let indices = self.ctx.constant_from(indices);
        self.gather(&indices, config)
    }

    /// Stack tensors along a newly inserted axis.
    ///
    /// The returned tensor uses the context of the first input, matching
    /// [`Self::concatenate`]. All inputs must belong to that same context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![], vec![1.0_f64]), ctx.clone());
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec(vec![], vec![2.0_f64]), ctx);
    /// let out = EagerTensor::stack(&[&a, &b], -1).unwrap();
    ///
    /// assert_eq!(out.data().shape(), &[2]);
    /// assert_eq!(out.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn stack(tensors: &[&Self], dim: isize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| {
            Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
                op: "stack",
                message: "stack requires at least one input".into(),
            })
        })?;
        let shapes = tensors
            .iter()
            .map(|tensor| tensor.data().shape())
            .collect::<Vec<_>>();
        validate_stack_shapes("stack", &shapes)?;

        let axis = normalize_insert_axis("stack", dim, first.data().shape().len())?;
        let mut expanded_shape = first.data().shape().to_vec();
        expanded_shape.insert(axis, 1);

        let expanded = tensors
            .iter()
            .map(|tensor| tensor.reshape(&expanded_shape))
            .collect::<Result<Vec<_>>>()?;
        let refs = expanded.iter().collect::<Vec<_>>();
        Self::concatenate(&refs, axis)
    }
}

impl TracedTensor {
    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::from_tensor_concrete_shape(
    ///     Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]),
    /// );
    /// let y = x.index_select(-1, &[2, 0]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    ///
    /// assert_eq!(
    ///     out.as_slice::<f64>().unwrap(),
    ///     &[30.0, 10.0],
    /// );
    /// ```
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> Result<Self> {
        let shape = try_concrete_shape(self).ok_or_else(|| {
            Error::Internal("index_select currently requires a concrete shape hint".into())
        })?;
        let (indices_tensor, config, out_shape) = index_select_config(&shape, axis, positions)?;
        let indices = TracedTensor::from_tensor_concrete_shape(indices_tensor);
        Ok(apply_binary_preserve_input_dtypes(
            StdTensorOp::Gather(config),
            self,
            &indices,
            out_shape.len(),
            Some(out_shape.into_iter().map(SymDim::from).collect()),
            self.dtype,
        ))
    }

    /// Stack tensors along a newly inserted axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
    ///
    /// let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec(vec![], vec![1.0_f64]));
    /// let b = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec(vec![], vec![2.0_f64]));
    /// let stacked = TracedTensor::stack(&[&a, &b], -1).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&stacked).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    ///
    /// assert_eq!(
    ///     out.as_slice::<f64>().unwrap(),
    ///     &[1.0, 2.0],
    /// );
    /// ```
    pub fn stack(tensors: &[&Self], dim: isize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| {
            Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
                op: "stack",
                message: "stack requires at least one input".into(),
            })
        })?;
        let first_shape = try_concrete_shape(first).ok_or_else(|| {
            Error::Internal("stack currently requires concrete shape hints".into())
        })?;
        let mut shapes = Vec::with_capacity(tensors.len());
        shapes.push(first_shape.as_slice());
        let mut owned_shapes = Vec::with_capacity(tensors.len().saturating_sub(1));
        for tensor in tensors.iter().copied().skip(1) {
            owned_shapes.push(try_concrete_shape(tensor).ok_or_else(|| {
                Error::Internal("stack currently requires concrete shape hints".into())
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
            .collect::<Vec<_>>();
        let refs = expanded.iter().collect::<Vec<_>>();
        Ok(apply_nary_concatenate(&refs, axis, out_shape))
    }
}

fn apply_nary_concatenate(
    tensors: &[&TracedTensor],
    axis: usize,
    out_shape: Vec<usize>,
) -> TracedTensor {
    let out_dtype = promote_dtypes(tensors.iter().map(|tensor| tensor.dtype));
    let tensors = tensors
        .iter()
        .map(|tensor| {
            if tensor.dtype != out_dtype {
                tensor.convert(out_dtype)
            } else {
                (*tensor).clone()
            }
        })
        .collect::<Vec<_>>();

    let mut builder = FragmentBuilder::new();
    for tensor in &tensors {
        builder.add_parent(tensor.fragment.clone());
    }
    let input_refs = tensors
        .iter()
        .map(|tensor| ValRef::External(tensor.fragment.vals()[tensor.val].key.clone()))
        .collect::<Vec<_>>();
    let outputs = builder.add_op(
        StdTensorOp::Concatenate {
            axis,
            n_inputs: tensors.len(),
        },
        input_refs,
        OpMode::Primal,
    );
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    let metadata_scope = register_scoped_fragment_metadata(fragment.as_ref(), std::iter::empty());

    let mut inputs_map = HashMap::new();
    let mut extra_roots = Vec::new();
    let mut checkpoint_chain = None;
    for tensor in &tensors {
        inputs_map.extend(
            tensor
                .inputs_map
                .iter()
                .map(|(k, v)| (k.clone(), v.clone())),
        );
        extra_roots.extend(tensor.extra_roots.iter().cloned());
        checkpoint_chain =
            CheckpointNode::merge_chains(checkpoint_chain, tensor.checkpoint_chain.clone());
    }
    let inherited_scopes = tensors
        .iter()
        .map(|tensor| tensor.metadata_scopes.as_slice())
        .collect::<Vec<_>>();

    TracedTensor {
        id: next_traced_id(),
        rank: out_shape.len(),
        dtype: out_dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: Some(out_shape.into_iter().map(SymDim::from).collect()),
        inputs_map: Arc::new(inputs_map),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: metadata_scopes_with_new(metadata_scope, inherited_scopes),
    }
}
