use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::GlobalValKey;
use lru::LruCache;
use tenferro_einsum::ContractionTree;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::{DType, Tensor};

use super::cache::{
    compile_cache_stats, compute_cache_key, einsum_parse_cache_stats, nary_einsum_cache_stats,
    CacheKey, EinsumCacheKey, EinsumParseCache, GraphCompilerCacheStats, NaryEinsumCache,
    ParsedEinsum, DEFAULT_COMPILE_CACHE_CAPACITY, DEFAULT_EINSUM_CACHE_CAPACITY,
};
use super::program::{GraphProgram, GraphProgramInput};
use crate::compiler::compile_std_to_exec;
use crate::einsum_subscripts::parse_einsum_subscripts;
use crate::error::{Error, Result};
use crate::exec::ExecProgram;
use crate::traced::{try_concrete_shape, TracedTensor};

#[derive(Clone)]
struct InputDescriptor {
    key: TensorInputKey,
    dtype: DType,
    shape: Vec<usize>,
    default_tensor: Option<Arc<Tensor>>,
}

/// Compiler for traced tensor graphs.
///
/// A graph compiler lowers one or more [`TracedTensor`] outputs to a reusable
/// [`GraphProgram`] without requiring a backend.
///
/// # Examples
///
/// ```
/// use tenferro::{GraphCompiler, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let y = &x + &x;
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// assert_eq!(program.output_count(), 1);
/// ```
pub struct GraphCompiler {
    compile_cache: LruCache<CacheKey, ExecProgram>,
    static_einsum_cache: NaryEinsumCache,
    einsum_parse_cache: EinsumParseCache,
}

impl GraphCompiler {
    /// Create a compiler with bounded default caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            compile_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPILE_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
            ),
            static_einsum_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_EINSUM_CACHE_CAPACITY)
                    .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
            ),
            einsum_parse_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_EINSUM_CACHE_CAPACITY)
                    .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
            ),
        }
    }

    /// Create a compiler with an explicit static einsum cache capacity.
    ///
    /// The capacity applies to both the static contraction-plan cache and the
    /// parsed-subscript cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::with_einsum_cache_capacity(
    ///     NonZeroUsize::new(16).unwrap(),
    /// );
    /// assert_eq!(compiler.einsum_cache_capacity().get(), 16);
    /// ```
    pub fn with_einsum_cache_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            compile_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPILE_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
            ),
            static_einsum_cache: LruCache::new(capacity),
            einsum_parse_cache: LruCache::new(capacity),
        }
    }

    /// Compile one traced output into a graph program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]);
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x.neg()).unwrap();
    /// assert_eq!(program.input_count(), 1);
    /// ```
    pub fn compile(&mut self, output: &TracedTensor) -> Result<GraphProgram> {
        self.compile_many(&[output])
    }

    /// Compile multiple traced outputs into one graph program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]);
    /// let y = x.neg();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile_many(&[&x, &y]).unwrap();
    /// assert_eq!(program.output_count(), 2);
    /// ```
    pub fn compile_many(&mut self, outputs: &[&TracedTensor]) -> Result<GraphProgram> {
        let mut all_inputs = HashMap::new();
        for output in outputs {
            all_inputs.extend(
                output
                    .inputs_map
                    .iter()
                    .map(|(key, tensor)| (key.clone(), tensor.clone())),
            );
        }
        self.compile_many_with_descriptors(outputs, &HashMap::new(), &all_inputs)
    }

    /// Compile one traced output with concrete placeholder specs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{DType, GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler
    ///     .compile_with_input_specs(&x.neg(), &[(&x, DType::F64, &[3])])
    ///     .unwrap();
    /// assert_eq!(program.input_specs()[0].shape(), &[3]);
    /// ```
    pub fn compile_with_input_specs(
        &mut self,
        output: &TracedTensor,
        bindings: &[(&TracedTensor, DType, &[usize])],
    ) -> Result<GraphProgram> {
        let mut binding_specs = HashMap::new();
        for (index, (placeholder, dtype, shape)) in bindings.iter().enumerate() {
            validate_placeholder_spec(index, placeholder, *dtype, shape)?;
            let key = placeholder.input_key().ok_or(Error::UnexpectedBinding {
                binding_index: index,
            })?;
            if binding_specs
                .insert(
                    key.clone(),
                    InputDescriptor {
                        key: key.clone(),
                        dtype: *dtype,
                        shape: (*shape).to_vec(),
                        default_tensor: None,
                    },
                )
                .is_some()
            {
                return Err(Error::DuplicateBinding {
                    input_key: format!("{:?}", key),
                });
            }
        }

        self.compile_many_with_descriptors(&[output], &binding_specs, output.inputs_map.as_ref())
    }

    /// Number of compiled programs currently retained.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn compile_cache_len(&self) -> usize {
        self.compile_cache.len()
    }

    /// Number of cached static einsum contraction trees currently retained.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.einsum_cache_len(), 0);
    /// ```
    pub fn einsum_cache_len(&self) -> usize {
        self.static_einsum_cache.len()
    }

    /// Current capacity of the static einsum caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert!(compiler.einsum_cache_capacity().get() > 0);
    /// ```
    pub fn einsum_cache_capacity(&self) -> NonZeroUsize {
        self.static_einsum_cache.cap()
    }

    /// Resize the static einsum contraction-plan and parse caches.
    ///
    /// Shrinking below the current length evicts least-recently-used entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.set_einsum_cache_capacity(NonZeroUsize::new(8).unwrap());
    /// assert_eq!(compiler.einsum_cache_capacity().get(), 8);
    /// ```
    pub fn set_einsum_cache_capacity(&mut self, capacity: NonZeroUsize) {
        self.static_einsum_cache.resize(capacity);
        self.einsum_parse_cache.resize(capacity);
    }

    /// Current compiled-program cache capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert!(compiler.compile_cache_capacity().get() > 0);
    /// ```
    pub fn compile_cache_capacity(&self) -> NonZeroUsize {
        self.compile_cache.cap()
    }

    /// Resize the compiled-program cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.set_compile_cache_capacity(NonZeroUsize::new(2).unwrap());
    /// assert_eq!(compiler.compile_cache_capacity().get(), 2);
    /// ```
    pub fn set_compile_cache_capacity(&mut self, capacity: NonZeroUsize) {
        self.compile_cache.resize(capacity);
    }

    /// Clear the compiled-program cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_compile_cache();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn clear_compile_cache(&mut self) {
        self.compile_cache.clear();
    }

    /// Clear parsed and planned einsum caches owned by the compiler.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_einsum_caches();
    /// assert_eq!(compiler.cache_stats().static_einsum_plans.entries, 0);
    /// ```
    pub fn clear_einsum_caches(&mut self) {
        self.static_einsum_cache.clear();
        self.einsum_parse_cache.clear();
    }

    /// Clear every cache owned by the compiler.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_caches();
    /// assert_eq!(compiler.cache_stats().compile.entries, 0);
    /// ```
    pub fn clear_caches(&mut self) {
        self.clear_compile_cache();
        self.clear_einsum_caches();
    }

    /// Return cache-entry and retained-byte stats.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// let stats = compiler.cache_stats();
    /// assert_eq!(stats.compile.entries, 0);
    /// ```
    pub fn cache_stats(&self) -> GraphCompilerCacheStats {
        GraphCompilerCacheStats {
            compile: compile_cache_stats(&self.compile_cache),
            static_einsum_plans: nary_einsum_cache_stats(&self.static_einsum_cache),
            einsum_parse: einsum_parse_cache_stats(&self.einsum_parse_cache),
        }
    }

    pub(crate) fn cached_subscripts(&mut self, subscripts: &str) -> Result<Arc<ParsedEinsum>> {
        if let Some(cached) = self.einsum_parse_cache.get(subscripts) {
            return Ok(Arc::clone(cached));
        }

        let parsed = parse_einsum_subscripts(subscripts)?;
        let cached = Arc::new(ParsedEinsum { subscripts: parsed });
        self.einsum_parse_cache
            .put(subscripts.to_owned(), Arc::clone(&cached));
        Ok(cached)
    }

    pub(crate) fn cached_static_einsum_tree(
        &mut self,
        key: EinsumCacheKey,
        build: impl FnOnce() -> Result<ContractionTree>,
    ) -> Result<Arc<ContractionTree>> {
        if let Some(cached) = self.static_einsum_cache.get(&key) {
            return Ok(Arc::clone(cached));
        }

        let tree = Arc::new(build()?);
        self.static_einsum_cache.put(key, Arc::clone(&tree));
        Ok(tree)
    }

    fn compile_many_with_descriptors(
        &mut self,
        outputs: &[&TracedTensor],
        binding_specs: &HashMap<TensorInputKey, InputDescriptor>,
        default_inputs: &HashMap<TensorInputKey, Arc<Tensor>>,
    ) -> Result<GraphProgram> {
        let mut roots = Vec::new();
        let mut output_keys = Vec::with_capacity(outputs.len());
        for output in outputs {
            roots.extend(output.resolve_roots());
            output_keys.push(output.fragment.vals()[output.val].key.clone());
        }

        let view = resolve(roots);
        let graph = materialize_merge(&view, &output_keys);
        let compiled = compile(&graph);

        let mut descriptors = Vec::with_capacity(graph.inputs.len());
        let mut input_dtypes = Vec::with_capacity(graph.inputs.len());
        let mut input_shapes = Vec::with_capacity(graph.inputs.len());
        for key in &graph.inputs {
            let GlobalValKey::Input(input_key) = key else {
                return Err(Error::Internal(
                    "expected Input key in graph inputs".to_string(),
                ));
            };
            let descriptor = descriptor_for_input(input_key, binding_specs, default_inputs)?;
            input_dtypes.push(descriptor.dtype);
            input_shapes.push(DimExpr::from_concrete(&descriptor.shape));
            descriptors.push(GraphProgramInput::new(
                descriptor.key,
                descriptor.dtype,
                descriptor.shape.clone(),
                DimExpr::from_concrete(&descriptor.shape),
                descriptor.default_tensor,
            ));
        }

        let exec = compile_std_to_exec(&compiled, &input_dtypes, &input_shapes);
        let exec = self.get_or_compile(exec);
        Ok(GraphProgram::new(exec, descriptors))
    }

    fn get_or_compile(&mut self, exec: ExecProgram) -> ExecProgram {
        let key = compute_cache_key(&exec);
        if let Some(cached) = self.compile_cache.get(&key) {
            return cached.clone();
        }
        self.compile_cache.put(key, exec.clone());
        exec
    }
}

impl Default for GraphCompiler {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_placeholder_spec(
    index: usize,
    placeholder: &TracedTensor,
    dtype: DType,
    shape: &[usize],
) -> Result<()> {
    if placeholder.data.is_some() {
        return Err(Error::UnexpectedBinding {
            binding_index: index,
        });
    }
    let _ = placeholder.input_key().ok_or(Error::UnexpectedBinding {
        binding_index: index,
    })?;

    if placeholder.dtype != dtype {
        return Err(Error::PlaceholderDtypeMismatch {
            expected: placeholder.dtype,
            actual: dtype,
        });
    }
    validate_placeholder_shape(placeholder, shape)
}

fn validate_placeholder_shape(placeholder: &TracedTensor, shape: &[usize]) -> Result<()> {
    match try_concrete_shape(placeholder) {
        Some(expected_shape) => {
            if expected_shape.as_slice() != shape {
                return Err(Error::PlaceholderShapeMismatch {
                    expected: expected_shape,
                    actual: shape.to_vec(),
                });
            }
        }
        None => {
            if placeholder.rank != shape.len() {
                return Err(Error::PlaceholderRankMismatch {
                    expected: placeholder.rank,
                    actual: shape.len(),
                });
            }
        }
    }
    Ok(())
}

fn descriptor_for_input(
    key: &TensorInputKey,
    binding_specs: &HashMap<TensorInputKey, InputDescriptor>,
    default_inputs: &HashMap<TensorInputKey, Arc<Tensor>>,
) -> Result<InputDescriptor> {
    if let Some(tensor) = default_inputs.get(key) {
        return Ok(InputDescriptor {
            key: key.clone(),
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            default_tensor: Some(tensor.clone()),
        });
    }
    if let Some(spec) = binding_specs.get(key) {
        return Ok(spec.clone());
    }
    if !matches!(key, TensorInputKey::User { .. }) {
        let root = tangent_primal_root(key);
        if let Some(spec) = binding_specs.get(root) {
            return Ok(InputDescriptor {
                key: key.clone(),
                dtype: spec.dtype,
                shape: spec.shape.clone(),
                default_tensor: spec
                    .default_tensor
                    .as_ref()
                    .map(|tensor| Arc::new(zeros_tensor(tensor.dtype(), tensor.shape().to_vec()))),
            });
        }
    }
    Err(Error::UnboundPlaceholder {
        input_key: format!("{:?}", key),
    })
}

fn tangent_primal_root(key: &TensorInputKey) -> &TensorInputKey {
    match key {
        TensorInputKey::User { .. } => key,
        TensorInputKey::Tangent { of, .. } => tangent_primal_root(of),
    }
}

fn zeros_tensor(dtype: DType, shape: Vec<usize>) -> Tensor {
    match dtype {
        DType::F32 => Tensor::F32(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::F64 => Tensor::F64(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::I64 => Tensor::I64(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::C32 => Tensor::C32(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::C64 => Tensor::C64(tenferro_tensor::TypedTensor::zeros(shape)),
    }
}
