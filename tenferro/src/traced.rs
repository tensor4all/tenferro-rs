use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use computegraph::fragment::{Fragment, FragmentBuilder};
#[cfg(feature = "autodiff")]
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::LocalValId;
use num_complex::{Complex32, Complex64};
use tenferro_ops::broadcast::{broadcast_input_plan, broadcast_shape, broadcast_shapes};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
#[cfg(feature = "autodiff")]
use tenferro_ops::ShapeGuardContext;
#[cfg(feature = "autodiff")]
use tenferro_tensor::TensorBackend;
#[cfg(feature = "autodiff")]
use tenferro_tensor::TypedTensor;
use tenferro_tensor::{CompareDir, DType, DotGeneralConfig, Tensor, TensorScalar};
#[cfg(feature = "autodiff")]
use tidu::{try_differentiate, try_transpose};

use super::error::{Error, Result};
#[cfg(feature = "autodiff")]
use super::graph::{GraphCompiler, GraphExecutor};
use super::sym_dim::SymDim;
use crate::checkpoint::CheckpointNode;
use crate::metadata::{
    concrete_tensor_meta, metadata_scopes_for_scope, metadata_scopes_with_new, push_metadata_scope,
    register_scoped_fragment_metadata, register_scoped_value_metadata, symbolic_input_meta,
    tensor_meta, MetadataScope,
};
#[cfg(feature = "autodiff")]
use crate::metadata::{registered_meta, tensor_meta_from_tensor};
use crate::scalar_semantics::round_real_to_i64;

static NEXT_INPUT_ID: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "autodiff")]
static NEXT_DIFF_PASS_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_TRACED_ID: AtomicU64 = AtomicU64::new(0);

pub type TracedTensorId = u64;

pub(crate) fn next_input_key() -> TensorInputKey {
    TensorInputKey::User {
        id: NEXT_INPUT_ID.fetch_add(1, Ordering::Relaxed),
    }
}

#[cfg(feature = "autodiff")]
fn next_pass_id() -> u64 {
    NEXT_DIFF_PASS_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn next_traced_id() -> TracedTensorId {
    NEXT_TRACED_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone)]
pub struct TracedTensor {
    pub id: TracedTensorId,
    pub rank: usize,
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: LocalValId,
    pub data: Option<Arc<Tensor>>,
    pub(crate) shape_hint: Option<Vec<SymDim>>,
    pub(crate) inputs_map: Arc<HashMap<TensorInputKey, Arc<Tensor>>>,
    pub(crate) extra_roots: Vec<Arc<Fragment<StdTensorOp>>>,
    pub(crate) checkpoint_chain: Option<Arc<CheckpointNode>>,
    pub(crate) metadata_scopes: Vec<Arc<MetadataScope>>,
}

pub(crate) fn try_concrete_shape(tensor: &TracedTensor) -> Option<Vec<usize>> {
    tensor
        .shape_hint
        .as_ref()?
        .iter()
        .map(SymDim::constant_value)
        .collect()
}

pub(crate) fn concrete_shape(tensor: &TracedTensor) -> Vec<usize> {
    tensor
        .shape_hint
        .as_ref()
        .unwrap_or_else(|| panic!("missing shape hint for traced tensor {}", tensor.id))
        .iter()
        .map(|dim| {
            dim.constant_value().unwrap_or_else(|| {
                panic!("symbolic dimension in shape hint for tensor {}", tensor.id)
            })
        })
        .collect()
}

#[cfg(feature = "autodiff")]
fn error_shape_hint(tensor: &TracedTensor) -> Vec<usize> {
    try_concrete_shape(tensor).unwrap_or_else(|| vec![0; tensor.rank])
}

/// Broadcast a traced tensor to `target_shape`.
///
/// Expanding singleton axes are first reshaped away so the existing
/// `BroadcastInDim` transpose rule reduces them correctly during VJP.
pub(crate) fn broadcast_to(tensor: &TracedTensor, target_shape: &[usize]) -> TracedTensor {
    let tensor_shape = concrete_shape(tensor);
    if tensor_shape == target_shape {
        return tensor.clone();
    }

    let plan =
        broadcast_input_plan(&tensor_shape, target_shape).unwrap_or_else(|err| panic!("{err}"));

    let source = if plan.source_shape == tensor_shape {
        tensor.clone()
    } else {
        tensor.reshape(&plan.source_shape)
    };
    source.broadcast_in_dim(target_shape, &plan.dims)
}

/// Broadcast two tensors to a common shape.
pub(crate) fn broadcast_binary(a: &TracedTensor, b: &TracedTensor) -> (TracedTensor, TracedTensor) {
    if a.shape_hint == b.shape_hint && a.rank == b.rank {
        return (a.clone(), b.clone());
    }
    let a_shape = concrete_shape(a);
    let b_shape = concrete_shape(b);
    let target = broadcast_shape(&a_shape, &b_shape).unwrap_or_else(|_| {
        panic!(
            "incompatible shapes for broadcast: {:?} and {:?}",
            a_shape, b_shape
        )
    });
    (broadcast_to(a, &target), broadcast_to(b, &target))
}

pub(crate) fn broadcast_ternary(
    a: &TracedTensor,
    b: &TracedTensor,
    c: &TracedTensor,
) -> (TracedTensor, TracedTensor, TracedTensor) {
    let a_shape = concrete_shape(a);
    let b_shape = concrete_shape(b);
    let c_shape = concrete_shape(c);
    let target = broadcast_shapes([a_shape.as_slice(), b_shape.as_slice(), c_shape.as_slice()])
        .unwrap_or_else(|err| panic!("{err}"));
    (
        broadcast_to(a, &target),
        broadcast_to(b, &target),
        broadcast_to(c, &target),
    )
}

fn scale_with_constant(input: &TracedTensor, op: StdTensorOp) -> TracedTensor {
    let scalar = apply_nullary(op, 0, input.dtype, Some(vec![]));
    let input_shape = concrete_shape(input);
    let factor = broadcast_to(&scalar, &input_shape);
    apply_binary(
        StdTensorOp::Mul,
        input,
        &factor,
        input.rank,
        input.shape_hint.clone(),
    )
}

impl std::ops::Add for &TracedTensor {
    type Output = TracedTensor;

    fn add(self, rhs: &TracedTensor) -> TracedTensor {
        TracedTensor::add(self, rhs)
    }
}

impl std::ops::Mul for &TracedTensor {
    type Output = TracedTensor;

    fn mul(self, rhs: &TracedTensor) -> TracedTensor {
        TracedTensor::mul(self, rhs)
    }
}

impl std::ops::Mul<f64> for &TracedTensor {
    type Output = TracedTensor;

    fn mul(self, rhs: f64) -> TracedTensor {
        self.scale_real(rhs)
    }
}

impl std::ops::Mul<&TracedTensor> for f64 {
    type Output = TracedTensor;

    fn mul(self, rhs: &TracedTensor) -> TracedTensor {
        rhs.scale_real(self)
    }
}

impl std::ops::Neg for &TracedTensor {
    type Output = TracedTensor;

    fn neg(self) -> TracedTensor {
        TracedTensor::neg(self)
    }
}

impl std::ops::Div for &TracedTensor {
    type Output = TracedTensor;

    fn div(self, rhs: &TracedTensor) -> TracedTensor {
        TracedTensor::div(self, rhs)
    }
}

impl TracedTensor {
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
    /// use tenferro::{Tensor, TracedTensor};
    ///
    /// let a = TracedTensor::from_tensor_concrete_shape(
    ///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
    /// );
    /// assert_eq!(a.rank, 2);
    /// assert!(a.is_concrete_shape());
    /// ```
    pub fn from_tensor_concrete_shape(tensor: Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        let rank = shape.len();
        let dtype = tensor.dtype();
        let key = next_input_key();
        let id = next_traced_id();
        let data = Arc::new(tensor);

        let mut builder = FragmentBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let fragment = Arc::new(builder.build());
        let metadata_scope = register_scoped_value_metadata(
            fragment.vals()[val].key.clone(),
            concrete_tensor_meta(dtype, &shape),
        );

        let mut map = HashMap::new();
        map.insert(key, Arc::clone(&data));

        Self {
            id,
            rank,
            dtype,
            fragment,
            val,
            data: Some(data),
            shape_hint: Some(shape.into_iter().map(SymDim::from).collect()),
            inputs_map: Arc::new(map),
            extra_roots: Vec::new(),
            checkpoint_chain: None,
            metadata_scopes: metadata_scopes_for_scope(metadata_scope),
        }
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
    /// use tenferro::{Tensor, TracedTensor};
    ///
    /// let t = TracedTensor::from_tensor_symbolic_shape(
    ///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
    /// );
    /// assert_eq!(t.rank, 2);
    /// assert!(!t.is_concrete_shape());
    /// ```
    pub fn from_tensor_symbolic_shape(tensor: Tensor) -> Self {
        let rank = tensor.shape().len();
        let dtype = tensor.dtype();
        let key = next_input_key();
        let id = next_traced_id();
        let data = Arc::new(tensor);

        let mut builder = FragmentBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let fragment = Arc::new(builder.build());
        let metadata_scope = register_scoped_value_metadata(
            fragment.vals()[val].key.clone(),
            symbolic_input_meta(dtype, id, rank),
        );

        let mut map = HashMap::new();
        map.insert(key, Arc::clone(&data));

        Self {
            id,
            rank,
            dtype,
            fragment,
            val,
            data: Some(data),
            shape_hint: None,
            inputs_map: Arc::new(map),
            extra_roots: Vec::new(),
            checkpoint_chain: None,
            metadata_scopes: metadata_scopes_for_scope(metadata_scope),
        }
    }

    /// Build a data-less placeholder leaf with a fixed (concrete) shape.
    ///
    /// Must be bound via [`GraphExecutor::run_with_inputs`] before evaluation.
    /// Use this when you know the exact shape of the input but want to build
    /// the graph once and feed different concrete tensors at execution time.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro::TracedTensor;
    ///
    /// let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 3]);
    /// assert_eq!(x.rank, 2);
    /// assert!(x.is_concrete_shape());
    /// ```
    pub fn input_concrete_shape(dtype: DType, shape: &[usize]) -> Self {
        let shape = shape.to_vec();
        let rank = shape.len();
        let key = next_input_key();
        let id = next_traced_id();

        let mut builder = FragmentBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let fragment = Arc::new(builder.build());
        let metadata_scope = register_scoped_value_metadata(
            fragment.vals()[val].key.clone(),
            concrete_tensor_meta(dtype, &shape),
        );

        Self {
            id,
            rank,
            dtype,
            fragment,
            val,
            data: None,
            shape_hint: Some(shape.into_iter().map(SymDim::from).collect()),
            inputs_map: Arc::new(HashMap::new()),
            extra_roots: Vec::new(),
            checkpoint_chain: None,
            metadata_scopes: metadata_scopes_for_scope(metadata_scope),
        }
    }

    /// Build a data-less placeholder leaf with the given rank but fully
    /// symbolic shape (every dim is a distinct `SymDim::TensorAxis`).
    ///
    /// Must be bound via [`GraphExecutor::run_with_inputs`] before
    /// evaluation. Use this to build shape-agnostic graphs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro::TracedTensor;
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 2);
    /// assert_eq!(x.rank, 2);
    /// assert!(!x.is_concrete_shape());
    /// ```
    pub fn input_symbolic_shape(dtype: DType, rank: usize) -> Self {
        let key = next_input_key();
        let id = next_traced_id();

        let mut builder = FragmentBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let fragment = Arc::new(builder.build());
        let metadata_scope = register_scoped_value_metadata(
            fragment.vals()[val].key.clone(),
            symbolic_input_meta(dtype, id, rank),
        );

        Self {
            id,
            rank,
            dtype,
            fragment,
            val,
            data: None,
            shape_hint: None,
            inputs_map: Arc::new(HashMap::new()),
            extra_roots: Vec::new(),
            checkpoint_chain: None,
            metadata_scopes: metadata_scopes_for_scope(metadata_scope),
        }
    }

    /// Build a concrete-shape [`TracedTensor`] leaf from column-major typed
    /// `Vec<T>` data.
    ///
    /// The data must already be in tenferro's physical column-major order.
    /// For row-major data, use [`TracedTensor::from_vec_row_major`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
    /// );
    /// assert_eq!(a.rank, 2);
    /// ```
    pub fn from_vec_col_major<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
        Self::from_tensor_concrete_shape(Tensor::from_vec_col_major(shape, data))
    }

    /// Build a concrete-shape [`TracedTensor`] leaf from row-major typed
    /// `Vec<T>` data.
    ///
    /// The data is converted into tenferro's physical column-major storage
    /// order before it is attached to the traced leaf. For data that is
    /// already in physical column-major order, use
    /// [`TracedTensor::from_vec_col_major`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_row_major(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// );
    /// assert_eq!(a.rank, 2);
    /// ```
    pub fn from_vec_row_major<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
        Self::from_tensor_concrete_shape(Tensor::from_vec_row_major(shape, data))
    }

    /// Returns `true` iff every dim of this tensor's `shape_hint` is a
    /// constant `SymDim` (i.e. the shape is fully known at graph-build time).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DType;
    /// use tenferro::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
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
    /// use tenferro::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// assert_eq!(a.try_concrete_shape(), Some(vec![2, 3]));
    ///
    /// let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    /// assert!(b.try_concrete_shape().is_none());
    /// ```
    pub fn try_concrete_shape(&self) -> Option<Vec<usize>> {
        try_concrete_shape(self)
    }

    /// Return the concrete tensor shape, panicking if any dimension is symbolic.
    ///
    /// This mirrors the existing traced frontend behavior for composite ops
    /// that require concrete ranks and sizes at graph construction time.
    pub fn concrete_shape(&self) -> Vec<usize> {
        concrete_shape(self)
    }

    /// If this `TracedTensor` is a leaf (single-node input fragment),
    /// return its input key. Computed tensors return `None`.
    pub fn input_key(&self) -> Option<TensorInputKey> {
        match &self.fragment.vals()[self.val].key {
            GlobalValKey::Input(key) => Some(key.clone()),
            _ => None,
        }
    }

    #[cfg(feature = "autodiff")]
    pub fn grad(&self, wrt: &TracedTensor) -> Result<TracedTensor> {
        if self.rank != 0 {
            return Err(Error::NonScalarGrad {
                shape: error_shape_hint(self),
            });
        }

        let ones = ones_tensor(self.dtype, vec![]);
        let seed = TracedTensor::from_tensor_concrete_shape(ones);
        self.try_vjp_result(wrt, &seed)?.ok_or_else(|| {
            Error::Internal(format!(
                "grad output is inactive for {:?}",
                leaf_input_key(wrt)
            ))
        })
    }

    /// Like [`grad`](Self::grad) but returns `None` when the scalar output does
    /// not depend on `wrt`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// # let loss = x.scale_real(2.0);
    /// let maybe_dx = loss.try_grad(&x).unwrap();
    /// ```
    #[cfg(feature = "autodiff")]
    pub fn try_grad(&self, wrt: &TracedTensor) -> Result<Option<TracedTensor>> {
        if self.rank != 0 {
            return Err(Error::NonScalarGrad {
                shape: error_shape_hint(self),
            });
        }

        let ones = ones_tensor(self.dtype, vec![]);
        let seed = TracedTensor::from_tensor_concrete_shape(ones);
        self.try_vjp_result(wrt, &seed)
    }

    /// Evaluate this tensor and replace its graph with a concrete leaf.
    ///
    /// This keeps downstream forward evaluation rooted at the concrete value
    /// while retaining the original fragment chain for later reverse-mode AD.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let mut compiler = GraphCompiler::new();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let mut y = &x * &x;
    /// y.checkpoint(&mut compiler, &mut executor).unwrap();
    ///
    /// let program = compiler.compile(&y).unwrap();
    /// assert_eq!(executor.run(&program).unwrap().shape(), &[] as &[usize]);
    /// ```
    #[cfg(feature = "autodiff")]
    pub fn checkpoint<B: TensorBackend>(
        &mut self,
        compiler: &mut GraphCompiler,
        executor: &mut GraphExecutor<B>,
    ) -> Result<()> {
        let data = if let Some(data) = &self.data {
            data.clone()
        } else {
            let program = compiler.compile(self)?;
            Arc::new(executor.run(&program)?)
        };
        let concrete_shape_hint = Some(data.shape().iter().copied().map(SymDim::from).collect());

        let old_fragment = self.fragment.clone();
        let old_output_key = old_fragment.vals()[self.val].key.clone();
        let old_inputs = (*self.inputs_map).clone();
        let concrete_meta = tensor_meta_from_tensor(data.as_ref());

        let new_key = next_input_key();
        let mut builder = FragmentBuilder::new();
        let leaf_val = builder.add_input(new_key.clone());
        builder.set_outputs(vec![leaf_val]);
        let new_fragment = Arc::new(builder.build());
        let new_metadata_scope = register_scoped_value_metadata(
            new_fragment.vals()[leaf_val].key.clone(),
            concrete_meta.clone(),
        );
        // Dynamic shape ops may have conservative static metadata on their
        // graph output. A checkpoint has evaluated the concrete tensor, so AD
        // alias resolution should see the runtime shape on both sides.
        let old_output_metadata_scope =
            register_scoped_value_metadata(old_output_key.clone(), concrete_meta);

        let node = CheckpointNode {
            fragment: old_fragment,
            alias_key: new_key.clone(),
            alias_target: old_output_key,
            old_inputs,
            prev: self.checkpoint_chain.take(),
        };

        self.fragment = new_fragment;
        self.val = leaf_val;
        self.extra_roots.clear();
        self.data = Some(data.clone());
        self.shape_hint = concrete_shape_hint;
        self.checkpoint_chain = Some(Arc::new(node));
        push_metadata_scope(&mut self.metadata_scopes, Arc::new(new_metadata_scope));
        push_metadata_scope(
            &mut self.metadata_scopes,
            Arc::new(old_output_metadata_scope),
        );

        let mut merged = HashMap::new();
        if let Some(chain) = &self.checkpoint_chain {
            merged.extend(chain.collect_inputs());
        }
        merged.insert(new_key, data);
        self.inputs_map = Arc::new(merged);

        Ok(())
    }

    #[cfg(feature = "autodiff")]
    pub fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor {
        self.try_jvp(wrt, tangent)
            .unwrap_or_else(|| panic!("jvp output is inactive for {:?}", leaf_input_key(wrt)))
    }

    /// Like [`jvp`](Self::jvp) but returns `None` when the output does not
    /// depend on `wrt` (i.e. the tangent is structurally zero).
    #[cfg(feature = "autodiff")]
    pub fn try_jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Option<TracedTensor> {
        self.try_jvp_result(wrt, tangent)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible variant of [`try_jvp`](Self::try_jvp).
    ///
    /// This returns an error when a primitive or extension cannot emit its
    /// linearization rule.
    #[cfg(feature = "autodiff")]
    pub fn try_jvp_result(
        &self,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        let wrt_input_key = leaf_input_key(wrt);
        let output_key = self.fragment.vals()[self.val].key.clone();
        let aliases = self
            .checkpoint_chain
            .as_ref()
            .map(|chain| chain.collect_aliases())
            .unwrap_or_default();
        let checkpoint_fragments = self
            .checkpoint_chain
            .as_ref()
            .map(|chain| chain.collect_fragments())
            .unwrap_or_default();
        let mut roots = self.resolve_roots();
        roots.extend(checkpoint_fragments.iter().cloned());
        let view = resolve(roots);
        let mut ad_ctx = ShapeGuardContext::with_global_metadata();
        let linear = try_differentiate(
            &view,
            std::slice::from_ref(&output_key),
            std::slice::from_ref(&wrt_input_key),
            next_pass_id(),
            &mut ad_ctx,
            &aliases,
        )?;
        let Some(tangent_output) = linear.tangent_outputs[0] else {
            return Ok(None);
        };
        let tangent_input_key = linear_input_key(&linear.fragment, linear.tangent_inputs[0].1);
        let metadata_scope = register_scoped_fragment_metadata(
            &linear.fragment,
            vec![(
                GlobalValKey::Input(tangent_input_key.clone()),
                tensor_meta_from_tensor(
                    tangent
                        .data
                        .as_ref()
                        .unwrap_or_else(|| panic!("jvp tangent must have concrete tensor data"))
                        .as_ref(),
                ),
            )],
        );

        let mut inputs_map = (*self.inputs_map).clone();
        if let Some(chain) = &self.checkpoint_chain {
            inputs_map.extend(chain.collect_inputs());
        }
        inputs_map.insert(
            tangent_input_key,
            tangent
                .data
                .clone()
                .unwrap_or_else(|| panic!("jvp tangent must have concrete tensor data")),
        );

        let mut extra_roots = vec![self.fragment.clone()];
        extra_roots.extend(checkpoint_fragments);
        extra_roots.extend(self.extra_roots.iter().cloned());

        Ok(Some(TracedTensor {
            id: next_traced_id(),
            rank: self.rank,
            dtype: self.dtype,
            fragment: Arc::new(linear.fragment),
            val: tangent_output,
            data: None,
            shape_hint: self.shape_hint.clone(),
            inputs_map: Arc::new(inputs_map),
            extra_roots,
            checkpoint_chain: self.checkpoint_chain.clone(),
            metadata_scopes: metadata_scopes_with_new(
                metadata_scope,
                [
                    self.metadata_scopes.as_slice(),
                    wrt.metadata_scopes.as_slice(),
                    tangent.metadata_scopes.as_slice(),
                ],
            ),
        }))
    }

    #[cfg(feature = "autodiff")]
    pub fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> TracedTensor {
        match self.try_vjp_result(wrt, cotangent) {
            Ok(Some(vjp)) => vjp,
            Ok(None) => panic!("vjp output is inactive for {:?}", leaf_input_key(wrt)),
            Err(err) => panic!("{err}"),
        }
    }

    /// Fallible reverse-mode product helper.
    ///
    /// This returns `Ok(None)` when the cotangent for `wrt` is structurally
    /// inactive, and returns an error when a primitive or extension is missing
    /// the required AD rule.
    #[cfg(feature = "autodiff")]
    pub fn try_vjp_result(
        &self,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        let wrt_input_key = leaf_input_key(wrt);
        let output_key = self.fragment.vals()[self.val].key.clone();
        let aliases = self
            .checkpoint_chain
            .as_ref()
            .map(|chain| chain.collect_aliases())
            .unwrap_or_default();
        let checkpoint_fragments = self
            .checkpoint_chain
            .as_ref()
            .map(|chain| chain.collect_fragments())
            .unwrap_or_default();
        let mut roots = self.resolve_roots();
        roots.extend(checkpoint_fragments.iter().cloned());
        let view = resolve(roots);
        let mut ad_ctx = ShapeGuardContext::with_global_metadata();
        let linear = try_differentiate(
            &view,
            std::slice::from_ref(&output_key),
            std::slice::from_ref(&wrt_input_key),
            next_pass_id(),
            &mut ad_ctx,
            &aliases,
        )?;
        if linear.tangent_outputs[0].is_none() {
            return Ok(None);
        }
        let linear_seed_key = linear_input_key(&linear.fragment, linear.tangent_inputs[0].1);
        let linear_metadata_scope = register_scoped_fragment_metadata(
            &linear.fragment,
            vec![(
                GlobalValKey::Input(linear_seed_key),
                registered_meta(&wrt.fragment.vals()[wrt.val].key),
            )],
        );
        ad_ctx.refresh_global_metadata();
        let transposed = try_transpose(&linear, &mut ad_ctx)?;
        let cotangent_input_key =
            linear_input_key(&transposed.fragment, transposed.tangent_inputs[0].1);
        let transposed_metadata_scope = register_scoped_fragment_metadata(
            &transposed.fragment,
            vec![(
                GlobalValKey::Input(cotangent_input_key.clone()),
                tensor_meta_from_tensor(
                    cotangent
                        .data
                        .as_ref()
                        .unwrap_or_else(|| panic!("vjp cotangent must have concrete tensor data"))
                        .as_ref(),
                ),
            )],
        );
        let linear_fragment = Arc::new(linear.fragment);
        let Some(cotangent_output) = transposed.tangent_outputs[0] else {
            return Ok(None);
        };

        let mut inputs_map = (*self.inputs_map).clone();
        if let Some(chain) = &self.checkpoint_chain {
            inputs_map.extend(chain.collect_inputs());
        }
        inputs_map.insert(
            cotangent_input_key.clone(),
            cotangent
                .data
                .clone()
                .unwrap_or_else(|| panic!("vjp cotangent must have concrete tensor data")),
        );
        // Inactive tangent keys are intentionally absent from `inputs_map`.
        // Graph execution resolves them through deferred-zero synthesis keyed
        // on the primal binding, avoiding dense zero tensors during VJP graph
        // construction.

        let mut extra_roots = vec![self.fragment.clone(), linear_fragment];
        extra_roots.extend(checkpoint_fragments);
        extra_roots.extend(self.extra_roots.iter().cloned());

        Ok(Some(TracedTensor {
            id: next_traced_id(),
            rank: wrt.rank,
            dtype: wrt.dtype,
            fragment: Arc::new(transposed.fragment),
            val: cotangent_output,
            data: None,
            shape_hint: wrt.shape_hint.clone(),
            inputs_map: Arc::new(inputs_map),
            extra_roots,
            checkpoint_chain: self.checkpoint_chain.clone(),
            metadata_scopes: {
                let mut scopes = metadata_scopes_with_new(
                    linear_metadata_scope,
                    [
                        self.metadata_scopes.as_slice(),
                        wrt.metadata_scopes.as_slice(),
                        cotangent.metadata_scopes.as_slice(),
                    ],
                );
                push_metadata_scope(&mut scopes, Arc::new(transposed_metadata_scope));
                scopes
            },
        }))
    }

    /// Elementwise addition with NumPy-style broadcasting.
    ///
    /// Prefer using the `+` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// # let z = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
    /// let y = x.add(&z);
    /// let y2 = &x + &z;
    /// ```
    pub fn add(&self, other: &TracedTensor) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
        apply_binary(
            StdTensorOp::Add,
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
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// # let z = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
    /// let y = x.mul(&z);
    /// let y2 = &x * &z;
    /// ```
    pub fn mul(&self, other: &TracedTensor) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
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
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// # let z = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
    /// let y = x.div(&z);
    /// let y2 = &x / &z;
    /// ```
    pub fn div(&self, other: &TracedTensor) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
        apply_binary(
            StdTensorOp::Div,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise comparison with NumPy-style broadcasting.
    pub fn compare(&self, other: &TracedTensor, dir: CompareDir) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
        apply_binary(
            StdTensorOp::Compare(dir),
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise negation.
    ///
    /// Prefer using the unary `-` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.neg();
    /// let y2 = -&x;
    /// ```
    pub fn neg(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Neg, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise complex conjugate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use num_complex::Complex64;
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(
    /// #     vec![2],
    /// #     vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    /// # );
    /// let y = x.conj();
    /// ```
    pub fn conj(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Conj, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise absolute value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![-1.0_f64, 2.0]);
    /// let y = x.abs();
    /// ```
    pub fn abs(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Abs, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise sign.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![-1.0_f64, 2.0]);
    /// let y = x.sign();
    /// ```
    pub fn sign(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sign, self, self.rank, self.shape_hint.clone())
    }

    /// Scale by a real scalar: `y = factor * x`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.scale_real(2.0);
    /// ```
    pub fn scale_real(&self, factor: f64) -> TracedTensor {
        let op = match self.dtype {
            DType::F64 => StdTensorOp::constant_f64(factor),
            DType::F32 => StdTensorOp::constant_f32(factor as f32),
            DType::I32 => StdTensorOp::constant_i32(round_real_to_i64(factor) as i32),
            DType::I64 => StdTensorOp::constant_i64(round_real_to_i64(factor)),
            DType::Bool => StdTensorOp::constant_bool(factor != 0.0),
            DType::C64 => StdTensorOp::constant_c64(Complex64::new(factor, 0.0)),
            DType::C32 => StdTensorOp::constant_c32(Complex32::new(factor as f32, 0.0)),
        };
        scale_with_constant(self, op)
    }

    /// Scale by a complex scalar: `y = factor * x`.
    ///
    /// This currently supports complex tensors only. For real scaling, prefer
    /// [`scale_real`](Self::scale_real).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(
    /// #     vec![2],
    /// #     vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    /// # );
    /// let y = x.scale_complex(Complex64::new(0.0, 1.0)); // multiply by i
    /// ```
    pub fn scale_complex(&self, factor: Complex64) -> TracedTensor {
        match self.dtype {
            DType::C64 => scale_with_constant(self, StdTensorOp::constant_c64(factor)),
            DType::C32 => scale_with_constant(
                self,
                StdTensorOp::constant_c32(Complex32::new(factor.re as f32, factor.im as f32)),
            ),
            DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => {
                panic!(
                    "scale_complex only supports complex tensors; use scale_real for real tensors"
                )
            }
        }
    }

    /// Elementwise exponential.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.exp();
    /// ```
    pub fn exp(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Exp, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise natural logarithm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.log();
    /// ```
    pub fn log(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Log, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise sine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.sin();
    /// ```
    pub fn sin(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sin, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise cosine.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.cos();
    /// ```
    pub fn cos(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Cos, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise hyperbolic tangent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.tanh();
    /// ```
    pub fn tanh(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Tanh, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise square root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]);
    /// let y = x.sqrt();
    /// ```
    pub fn sqrt(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sqrt, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise reciprocal square root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]);
    /// let y = x.rsqrt();
    /// ```
    pub fn rsqrt(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Rsqrt, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise power with NumPy-style broadcasting.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let base = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]);
    /// # let exp = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 2.0]);
    /// let y = base.pow(&exp);
    /// ```
    pub fn pow(&self, other: &TracedTensor) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
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
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.expm1();
    /// ```
    pub fn expm1(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Expm1, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise `log(1 + x)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let y = x.log1p();
    /// ```
    pub fn log1p(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Log1p, self, self.rank, self.shape_hint.clone())
    }

    /// Convert the tensor to a different dtype.
    ///
    /// For real-to-complex conversions this embeds the real values as
    /// `x + 0i`. For complex-to-real conversions this extracts the real part.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::DType;
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    ///
    /// let y = x.convert(DType::C64);
    /// ```
    pub fn convert(&self, to: DType) -> TracedTensor {
        if self.dtype == to {
            return self.clone();
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
    /// # use tenferro::{DotGeneralConfig, TracedTensor};
    /// # let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// # let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
    /// # let config = DotGeneralConfig {
    /// #     lhs_contracting_dims: vec![1],
    /// #     rhs_contracting_dims: vec![0],
    /// #     lhs_batch_dims: vec![],
    /// #     rhs_batch_dims: vec![],
    /// # };
    /// let y = a.dot_general(&b, config);
    /// ```
    pub fn dot_general(&self, other: &TracedTensor, config: DotGeneralConfig) -> TracedTensor {
        config
            .validate_dims_with_ranks(self.rank, other.rank)
            .expect("DotGeneral config dimension validation failed");
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

    /// Sum over the given axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    /// let y = x.reduce_sum(&[0]);
    /// let y2 = x.sum(&[0]);
    /// ```
    pub fn reduce_sum(&self, axes: &[usize]) -> TracedTensor {
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            (0..shape.len())
                .filter(|d| !axes.contains(d))
                .map(|d| shape[d].clone())
                .collect()
        });
        apply_unary(
            StdTensorOp::ReduceSum {
                axes: axes.to_vec(),
            },
            self,
            self.rank - axes.len(),
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
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    /// let y = x.reduce_max(&[0]);
    /// ```
    pub fn reduce_max(&self, axes: &[usize]) -> TracedTensor {
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            (0..shape.len())
                .filter(|d| !axes.contains(d))
                .map(|d| shape[d].clone())
                .collect()
        });
        apply_unary(
            StdTensorOp::ReduceMax {
                axes: axes.to_vec(),
            },
            self,
            self.rank - axes.len(),
            out_shape_hint,
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
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    /// let y = x.reduce_min(&[0]);
    /// ```
    pub fn reduce_min(&self, axes: &[usize]) -> TracedTensor {
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            (0..shape.len())
                .filter(|d| !axes.contains(d))
                .map(|d| shape[d].clone())
                .collect()
        });
        apply_unary(
            StdTensorOp::ReduceMin {
                axes: axes.to_vec(),
            },
            self,
            self.rank - axes.len(),
            out_shape_hint,
        )
    }

    /// Reduce by taking the product along the given axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    /// let y = x.reduce_prod(&[0]);
    /// ```
    pub fn reduce_prod(&self, axes: &[usize]) -> TracedTensor {
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            (0..shape.len())
                .filter(|d| !axes.contains(d))
                .map(|d| shape[d].clone())
                .collect()
        });
        apply_unary(
            StdTensorOp::ReduceProd {
                axes: axes.to_vec(),
            },
            self,
            self.rank - axes.len(),
            out_shape_hint,
        )
    }

    /// Reshape without changing element order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64; 4]);
    /// let y = x.reshape(&[2, 2]);
    /// ```
    pub fn reshape(&self, shape: &[usize]) -> TracedTensor {
        apply_unary(
            StdTensorOp::Reshape {
                to_shape: DimExpr::from_concrete(shape),
            },
            self,
            shape.len(),
            Some(shape.iter().copied().map(SymDim::from).collect()),
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
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// let rows = x.sym_size(0);
    /// let cols = x.sym_size(1);
    /// let y = x.reshape_sym(&[rows * cols]).unwrap();
    /// ```
    pub fn sym_size(&self, axis: usize) -> SymDim {
        assert!(
            axis < self.rank,
            "axis {axis} out of bounds for rank {}",
            self.rank
        );
        self.shape_hint
            .as_ref()
            .and_then(|shape| shape.get(axis))
            .filter(|dim| dim.constant_value().is_none())
            .cloned()
            .unwrap_or_else(|| SymDim::tensor_axis(self.id, axis))
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
    /// use tenferro::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// // Concrete axis: reports the constant size.
    /// assert_eq!(a.axis_sym_dim(0).constant_value(), Some(2));
    ///
    /// let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    /// // Fully symbolic leaf: reports a TensorAxis reference.
    /// assert!(b.axis_sym_dim(0).constant_value().is_none());
    /// ```
    pub fn axis_sym_dim(&self, axis: usize) -> SymDim {
        assert!(
            axis < self.rank,
            "axis {axis} out of bounds for rank {}",
            self.rank
        );
        match self.shape_hint.as_ref().and_then(|shape| shape.get(axis)) {
            Some(dim) => dim.clone(),
            None => SymDim::tensor_axis(self.id, axis),
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
    /// use tenferro::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// assert!(a.sym_shape().is_some());
    /// assert_eq!(a.sym_shape().unwrap().len(), 2);
    ///
    /// let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
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
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// let rows = x.sym_size(0);
    /// let cols = x.sym_size(1);
    /// let y = x.reshape_sym(&[rows * cols]).unwrap();
    /// ```
    pub fn reshape_sym(&self, shape: &[SymDim]) -> Result<TracedTensor> {
        let tensor_map = [(self.id, 0usize)];
        let to_shape = shape
            .iter()
            .map(|dim| dim.to_dim_expr(&tensor_map).map_err(Error::Internal))
            .collect::<Result<Vec<_>>>()?;
        let out_shape_hint = Some(shape.to_vec());
        Ok(apply_unary(
            StdTensorOp::Reshape { to_shape },
            self,
            shape.len(),
            out_shape_hint,
        ))
    }

    /// Broadcast into a larger shape with explicit dimension placement.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]);
    /// let y = x.broadcast_in_dim(&[2, 3], &[1]);
    /// let y2 = x.broadcast(&[2, 3], &[1]);
    /// ```
    pub fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> TracedTensor {
        apply_unary(
            StdTensorOp::BroadcastInDim {
                shape: DimExpr::from_concrete(shape),
                dims: dims.to_vec(),
            },
            self,
            shape.len(),
            Some(shape.iter().copied().map(SymDim::from).collect()),
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
    /// # Panics
    ///
    /// Panics if a `SymDim` in `shape` references a traced tensor that is
    /// neither `self` nor any tensor listed in `shape_refs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::TracedTensor;
    ///
    /// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
    /// let m = a.axis_sym_dim(0);
    /// let k = a.axis_sym_dim(1);
    /// let n = b.axis_sym_dim(1);
    /// // Broadcast `a[m, k]` to `[m, k, n]`, placing `a`'s axes at 0, 1
    /// // and taking `n` from `b` as an auxiliary shape reference.
    /// let a_b = a.broadcast_in_dim_sym(&[m, k, n], &[0, 1], &[&b]);
    /// assert_eq!(a_b.rank, 3);
    /// ```
    pub fn broadcast_in_dim_sym(
        &self,
        shape: &[SymDim],
        dims: &[usize],
        shape_refs: &[&TracedTensor],
    ) -> TracedTensor {
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
                dim.to_dim_expr(&tensor_map).unwrap_or_else(|err| {
                    panic!(
                        "broadcast_in_dim_sym: unresolved symbolic dimension: {}; \
                         pass every referenced tensor via `shape_refs`",
                        err
                    )
                })
            })
            .collect();

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

    /// Permute tensor axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    /// let y = x.transpose(&[1, 0]);
    /// ```
    pub fn transpose(&self, perm: &[usize]) -> TracedTensor {
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
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    /// let y = x.extract_diag(0, 1);
    /// ```
    pub fn extract_diag(&self, axis_a: usize, axis_b: usize) -> TracedTensor {
        assert!(
            axis_a < self.rank && axis_b < self.rank && axis_a != axis_b,
            "extract_diag: invalid axes"
        );
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            shape
                .iter()
                .enumerate()
                .filter_map(|(axis, dim)| (axis != axis_b).then_some(dim.clone()))
                .collect()
        });
        apply_unary(
            StdTensorOp::ExtractDiag { axis_a, axis_b },
            self,
            self.rank - 1,
            out_shape_hint,
        )
    }

    /// Embed a vector or lower-rank tensor along a diagonal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]);
    /// let y = x.embed_diag(0, 1);
    /// ```
    pub fn embed_diag(&self, axis_a: usize, axis_b: usize) -> TracedTensor {
        assert!(
            axis_a < self.rank && axis_b <= self.rank,
            "embed_diag: invalid axes"
        );
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

    /// Alias for [`Self::reduce_sum`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    /// let y = x.sum(&[0]);
    /// ```
    pub fn sum(&self, axes: &[usize]) -> TracedTensor {
        self.reduce_sum(axes)
    }

    /// Alias for [`Self::broadcast_in_dim`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro::TracedTensor;
    /// # let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]);
    /// let y = x.broadcast(&[2, 3], &[1]);
    /// ```
    pub fn broadcast(&self, shape: &[usize], dims: &[usize]) -> TracedTensor {
        self.broadcast_in_dim(shape, dims)
    }

    /// Return the runtime size of one axis as a scalar `f64` tensor.
    ///
    /// The result is metadata-derived and therefore has no gradient.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let cols = x.shape_of(1);
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&cols).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    /// assert_eq!(out.shape(), &[] as &[usize]);
    /// ```
    pub fn shape_of(&self, axis: usize) -> TracedTensor {
        assert!(
            axis < self.rank,
            "axis {axis} out of bounds for rank {}",
            self.rank
        );
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
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]);
    /// let size = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]);
    /// let y = x.dynamic_truncate(&size, 0);
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    /// assert_eq!(out.shape(), &[2]);
    /// ```
    pub fn dynamic_truncate(&self, size: &TracedTensor, axis: usize) -> TracedTensor {
        assert!(
            axis < self.rank,
            "axis {axis} out of bounds for rank {}",
            self.rank
        );
        assert!(
            size.rank == 0,
            "dynamic_truncate size must be a scalar tensor, got rank {}",
            size.rank
        );
        apply_binary(
            StdTensorOp::DynamicTruncate { axis },
            self,
            size,
            self.rank,
            None,
        )
    }

    /// Pad this tensor with zeros along `axis` to match `reference.shape[axis]`.
    ///
    /// If `reference` is smaller along that axis, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let reference = TracedTensor::from_vec_col_major(vec![4], vec![0.0_f64, 0.0, 0.0, 0.0]);
    /// let y = x.pad_to_match(&reference, 0);
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    /// assert_eq!(out.shape(), &[4]);
    /// ```
    pub fn pad_to_match(&self, reference: &TracedTensor, axis: usize) -> TracedTensor {
        assert!(
            axis < self.rank,
            "axis {axis} out of bounds for rank {}",
            self.rank
        );
        assert!(
            axis < reference.rank,
            "reference axis {axis} out of bounds for rank {}",
            reference.rank
        );
        apply_binary(
            StdTensorOp::PadToMatch { axis },
            self,
            reference,
            self.rank,
            reference.shape_hint.clone(),
        )
    }
}

pub(crate) fn apply_unary(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
) -> TracedTensor {
    apply_unary_with_dtype(op, input, out_rank, out_shape_hint, input.dtype)
}

pub(crate) fn apply_unary_with_dtype(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    out_dtype: DType,
) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    builder.add_parent(input.fragment.clone());
    let input_ref = ValRef::External(input.fragment.vals()[input.val].key.clone());
    let outputs = builder.add_op(op, vec![input_ref], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(fragment.as_ref(), outputs[0], out_dtype, &out_shape_hint);

    TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: out_dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: input.inputs_map.clone(),
        extra_roots: input.extra_roots.clone(),
        checkpoint_chain: input.checkpoint_chain.clone(),
        metadata_scopes: metadata_scopes_with_new(
            metadata_scope,
            [input.metadata_scopes.as_slice()],
        ),
    }
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
) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    builder.add_parent(input.fragment.clone());
    for t in shape_refs {
        builder.add_parent(t.fragment.clone());
    }
    let mut op_inputs: Vec<ValRef<StdTensorOp>> = Vec::with_capacity(1 + shape_refs.len());
    op_inputs.push(ValRef::External(
        input.fragment.vals()[input.val].key.clone(),
    ));
    for t in shape_refs {
        op_inputs.push(ValRef::External(t.fragment.vals()[t.val].key.clone()));
    }
    let outputs = builder.add_op(op, op_inputs, OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    let metadata_scope = register_single_output_metadata(
        fragment.as_ref(),
        outputs[0],
        input.dtype,
        &out_shape_hint,
    );

    let mut merged = (*input.inputs_map).clone();
    for t in shape_refs {
        merged.extend(t.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    let mut extra_roots = input.extra_roots.clone();
    for t in shape_refs {
        extra_roots.extend(t.extra_roots.iter().cloned());
    }

    let mut checkpoint_chain = input.checkpoint_chain.clone();
    for t in shape_refs {
        checkpoint_chain =
            CheckpointNode::merge_chains(checkpoint_chain, t.checkpoint_chain.clone());
    }

    TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: input.dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: Arc::new(merged),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: {
            let mut scopes =
                metadata_scopes_with_new(metadata_scope, [input.metadata_scopes.as_slice()]);
            for t in shape_refs {
                for scope in &t.metadata_scopes {
                    push_metadata_scope(&mut scopes, Arc::clone(scope));
                }
            }
            scopes
        },
    }
}

pub(crate) fn apply_nullary(
    op: StdTensorOp,
    rank: usize,
    dtype: DType,
    shape_hint: Option<Vec<SymDim>>,
) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    let outputs = builder.add_op(op, vec![], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(fragment.as_ref(), outputs[0], dtype, &shape_hint);

    TracedTensor {
        id: next_traced_id(),
        rank,
        dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint,
        inputs_map: Arc::new(HashMap::new()),
        extra_roots: Vec::new(),
        checkpoint_chain: None,
        metadata_scopes: metadata_scopes_for_scope(metadata_scope),
    }
}

pub(crate) fn apply_binary(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
) -> TracedTensor {
    let input_dtype = crate::shape_infer::promote_dtype_for_binary_op(&op, lhs.dtype, rhs.dtype);
    let out_dtype = crate::shape_infer::infer_output_dtype(&op, &[lhs.dtype, rhs.dtype]);

    // Insert Convert ops when an input dtype differs from the primitive input dtype.
    let lhs = if lhs.dtype != input_dtype {
        lhs.convert(input_dtype)
    } else {
        lhs.clone()
    };
    let rhs = if rhs.dtype != input_dtype {
        rhs.convert(input_dtype)
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
) -> TracedTensor {
    apply_binary_with_output_dtype(op, lhs, rhs, out_rank, out_shape_hint, out_dtype)
}

pub(crate) fn apply_broadcast_binary_op(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
) -> TracedTensor {
    let (lhs, rhs) = broadcast_binary(lhs, rhs);
    apply_binary(op, &lhs, &rhs, lhs.rank, lhs.shape_hint.clone())
}

pub(crate) fn apply_broadcast_ternary_op(
    op: StdTensorOp,
    first: &TracedTensor,
    second: &TracedTensor,
    third: &TracedTensor,
) -> TracedTensor {
    let (first, second, third) = broadcast_ternary(first, second, third);
    apply_ternary(
        op,
        &first,
        &second,
        &third,
        first.rank,
        first.shape_hint.clone(),
    )
}

pub(crate) fn apply_ternary(
    op: StdTensorOp,
    first: &TracedTensor,
    second: &TracedTensor,
    third: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
) -> TracedTensor {
    let out_dtype =
        crate::shape_infer::infer_output_dtype(&op, &[first.dtype, second.dtype, third.dtype]);
    let (first, second, third) = match op {
        StdTensorOp::Select => {
            let value_dtype = crate::shape_infer::promote_dtype(second.dtype, third.dtype);
            let second = if second.dtype != value_dtype {
                second.convert(value_dtype)
            } else {
                second.clone()
            };
            let third = if third.dtype != value_dtype {
                third.convert(value_dtype)
            } else {
                third.clone()
            };
            (first.clone(), second, third)
        }
        _ => {
            let input_dtype =
                crate::shape_infer::promote_dtypes([first.dtype, second.dtype, third.dtype]);
            let first = if first.dtype != input_dtype {
                first.convert(input_dtype)
            } else {
                first.clone()
            };
            let second = if second.dtype != input_dtype {
                second.convert(input_dtype)
            } else {
                second.clone()
            };
            let third = if third.dtype != input_dtype {
                third.convert(input_dtype)
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
) -> TracedTensor {
    let lhs_ref = ValRef::External(lhs.fragment.vals()[lhs.val].key.clone());
    let rhs_ref = ValRef::External(rhs.fragment.vals()[rhs.val].key.clone());

    let mut builder = FragmentBuilder::new();
    builder.add_parent(lhs.fragment.clone());
    builder.add_parent(rhs.fragment.clone());
    let outputs = builder.add_op(op, vec![lhs_ref, rhs_ref], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(fragment.as_ref(), outputs[0], out_dtype, &out_shape_hint);

    let mut merged = (*lhs.inputs_map).clone();
    merged.extend(rhs.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));
    let mut extra_roots = lhs.extra_roots.clone();
    extra_roots.extend(rhs.extra_roots.iter().cloned());

    TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: out_dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: Arc::new(merged),
        extra_roots,
        checkpoint_chain: CheckpointNode::merge_chains(
            lhs.checkpoint_chain.clone(),
            rhs.checkpoint_chain.clone(),
        ),
        metadata_scopes: metadata_scopes_with_new(
            metadata_scope,
            [
                lhs.metadata_scopes.as_slice(),
                rhs.metadata_scopes.as_slice(),
            ],
        ),
    }
}

fn apply_ternary_with_output_dtype(
    op: StdTensorOp,
    first: &TracedTensor,
    second: &TracedTensor,
    third: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    out_dtype: DType,
) -> TracedTensor {
    let first_ref = ValRef::External(first.fragment.vals()[first.val].key.clone());
    let second_ref = ValRef::External(second.fragment.vals()[second.val].key.clone());
    let third_ref = ValRef::External(third.fragment.vals()[third.val].key.clone());

    let mut builder = FragmentBuilder::new();
    builder.add_parent(first.fragment.clone());
    builder.add_parent(second.fragment.clone());
    builder.add_parent(third.fragment.clone());
    let outputs = builder.add_op(op, vec![first_ref, second_ref, third_ref], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    let metadata_scope =
        register_single_output_metadata(fragment.as_ref(), outputs[0], out_dtype, &out_shape_hint);

    let mut merged = (*first.inputs_map).clone();
    merged.extend(
        second
            .inputs_map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone())),
    );
    merged.extend(third.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));

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

    TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: out_dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: Arc::new(merged),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: metadata_scopes_with_new(
            metadata_scope,
            [
                first.metadata_scopes.as_slice(),
                second.metadata_scopes.as_slice(),
                third.metadata_scopes.as_slice(),
            ],
        ),
    }
}

fn register_single_output_metadata(
    fragment: &Fragment<StdTensorOp>,
    output: LocalValId,
    dtype: DType,
    shape_hint: &Option<Vec<SymDim>>,
) -> MetadataScope {
    if let Some(shape) = shape_hint {
        register_scoped_value_metadata(
            fragment.vals()[output].key.clone(),
            tensor_meta(dtype, shape.clone()),
        )
    } else {
        register_scoped_fragment_metadata(fragment, std::iter::empty())
    }
}

impl TracedTensor {
    pub(crate) fn resolve_roots(&self) -> Vec<Arc<Fragment<StdTensorOp>>> {
        let mut roots = Vec::with_capacity(1 + self.extra_roots.len());
        roots.push(self.fragment.clone());
        roots.extend(self.extra_roots.iter().cloned());
        roots
    }
}

#[cfg(feature = "autodiff")]
fn leaf_input_key(tt: &TracedTensor) -> TensorInputKey {
    match &tt.fragment.vals()[tt.val].key {
        GlobalValKey::Input(key) => key.clone(),
        other => panic!("expected traced leaf input, got {:?}", other),
    }
}

#[cfg(feature = "autodiff")]
fn linear_input_key(fragment: &Fragment<StdTensorOp>, local_id: LocalValId) -> TensorInputKey {
    match &fragment.vals()[local_id].key {
        GlobalValKey::Input(key) => key.clone(),
        other => panic!("expected linear fragment input, got {:?}", other),
    }
}

#[cfg(feature = "autodiff")]
fn ones_tensor(dtype: DType, shape: Vec<usize>) -> Tensor {
    match dtype {
        DType::F32 => Tensor::F32(TypedTensor::ones(shape)),
        DType::F64 => Tensor::F64(TypedTensor::ones(shape)),
        DType::I32 => Tensor::I32(TypedTensor::ones(shape)),
        DType::I64 => Tensor::I64(TypedTensor::ones(shape)),
        DType::Bool => {
            let len = shape.iter().product();
            Tensor::Bool(TypedTensor::from_vec_col_major(shape, vec![true; len]))
        }
        DType::C32 => Tensor::C32(TypedTensor::ones(shape)),
        DType::C64 => Tensor::C64(TypedTensor::ones(shape)),
    }
}
