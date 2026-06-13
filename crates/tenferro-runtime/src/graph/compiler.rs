use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::ValueKey;
use lru::LruCache;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::{DType, Tensor};

use super::cache::{
    compile_cache_stats, compute_cache_key, CacheKey, GraphCompilerCacheStats,
    DEFAULT_COMPILE_CACHE_CAPACITY,
};
use super::program::{GraphProgram, GraphProgramInput};
use crate::compiler::{compile_std_to_exec_with_options, CompilerOptions};
use crate::error::{Error, Result};
use crate::exec::ExecProgram;
use crate::extension_cache::{ExtensionCacheSelector, ExtensionCacheStore};
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
/// use tenferro_runtime::{GraphCompiler, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let y = &x + &x;
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// assert_eq!(program.output_count(), 1);
/// ```
pub struct GraphCompiler {
    compile_cache: LruCache<CacheKey, ExecProgram>,
    extension_cache: ExtensionCacheStore,
    compiler_options: CompilerOptions,
}

impl fmt::Debug for GraphCompiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphCompiler")
            .field("cache_stats", &self.cache_stats())
            .field("compile_cache_capacity", &self.compile_cache_capacity())
            .field("compiler_options", &self.compiler_options)
            .field("extension_cache", &self.extension_cache)
            .finish_non_exhaustive()
    }
}

impl GraphCompiler {
    /// Create a compiler with bounded default caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            compile_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPILE_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
            ),
            extension_cache: ExtensionCacheStore::new(),
            compiler_options: CompilerOptions::default(),
        }
    }

    /// Create a compiler with explicit lowering and optimizer options.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CompilerOptions, OptimizerConfig};
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::with_compiler_options(CompilerOptions {
    ///     optimizer: OptimizerConfig {
    ///         dot_decomposer: true,
    ///         ..OptimizerConfig::default()
    ///     },
    /// });
    /// assert!(compiler.compiler_options().optimizer.dot_decomposer);
    /// ```
    pub fn with_compiler_options(compiler_options: CompilerOptions) -> Self {
        Self {
            compile_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPILE_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
            ),
            extension_cache: ExtensionCacheStore::new(),
            compiler_options,
        }
    }

    /// Compile one traced output into a graph program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
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
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
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
    /// use tenferro_runtime::{DType, GraphCompiler, TracedTensor};
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
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn compile_cache_len(&self) -> usize {
        self.compile_cache.len()
    }

    /// Current compiled-program cache capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
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
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.set_compile_cache_capacity(NonZeroUsize::new(2).unwrap());
    /// assert_eq!(compiler.compile_cache_capacity().get(), 2);
    /// ```
    pub fn set_compile_cache_capacity(&mut self, capacity: NonZeroUsize) {
        self.compile_cache.resize(capacity);
    }

    /// Return the compiler options used for future graph lowerings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::CompilerOptions;
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert_eq!(compiler.compiler_options(), CompilerOptions::default());
    /// ```
    pub fn compiler_options(&self) -> CompilerOptions {
        self.compiler_options
    }

    /// Replace compiler options and clear compiled graph cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CompilerOptions, OptimizerConfig};
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// let options = CompilerOptions {
    ///     optimizer: OptimizerConfig {
    ///         dot_decomposer: true,
    ///         ..OptimizerConfig::default()
    ///     },
    /// };
    /// compiler.set_compiler_options(options);
    /// assert_eq!(compiler.compiler_options(), options);
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn set_compiler_options(&mut self, compiler_options: CompilerOptions) {
        if self.compiler_options == compiler_options {
            return;
        }
        self.compiler_options = compiler_options;
        self.clear_compile_cache();
    }

    /// Clear the compiled-program cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_compile_cache();
    /// assert_eq!(compiler.compile_cache_len(), 0);
    /// ```
    pub fn clear_compile_cache(&mut self) {
        self.compile_cache.clear();
    }

    /// Clear generic extension compile-time cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_extension_caches();
    /// assert_eq!(compiler.cache_stats().extensions.entries, 0);
    /// ```
    pub fn clear_extension_caches(&mut self) {
        self.extension_cache.clear();
    }

    /// Clear every cache owned by the compiler.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.clear_caches();
    /// assert_eq!(compiler.cache_stats().compile.entries, 0);
    /// ```
    pub fn clear_caches(&mut self) {
        self.clear_compile_cache();
        self.clear_extension_caches();
    }

    /// Return cache-entry and retained-byte stats.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// let stats = compiler.cache_stats();
    /// assert_eq!(stats.compile.entries, 0);
    /// ```
    pub fn cache_stats(&self) -> GraphCompilerCacheStats {
        GraphCompilerCacheStats {
            compile: compile_cache_stats(&self.compile_cache),
            extensions: self.extension_cache.stats(ExtensionCacheSelector::All),
        }
    }

    /// Borrow generic extension compile-time cache storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let compiler = GraphCompiler::new();
    /// assert!(compiler.extension_caches().is_empty());
    /// ```
    pub fn extension_caches(&self) -> &ExtensionCacheStore {
        &self.extension_cache
    }

    /// Mutably borrow generic extension compile-time cache storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::GraphCompiler;
    ///
    /// let mut compiler = GraphCompiler::new();
    /// compiler.extension_caches_mut().clear();
    /// ```
    pub fn extension_caches_mut(&mut self) -> &mut ExtensionCacheStore {
        &mut self.extension_cache
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
            output_keys.push(output.graph.values()[output.val].key.clone());
        }

        let view = resolve(roots);
        let graph = materialize_merge(&view, &output_keys);
        let compiled = compile(&graph);

        let mut descriptors = Vec::with_capacity(graph.inputs.len());
        let mut input_dtypes = Vec::with_capacity(graph.inputs.len());
        let mut input_shapes = Vec::with_capacity(graph.inputs.len());
        for key in &graph.inputs {
            let ValueKey::Input(input_key) = key else {
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

        let exec = compile_std_to_exec_with_options(
            &compiled,
            &input_dtypes,
            &input_shapes,
            self.compiler_options,
        )?;
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
        if let Some(tensor) = default_inputs.get(root) {
            return Ok(InputDescriptor {
                key: key.clone(),
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                default_tensor: Some(Arc::new(zeros_tensor(
                    tensor.dtype(),
                    tensor.shape().to_vec(),
                ))),
            });
        }
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
    key.primal_root()
}

fn zeros_tensor(dtype: DType, shape: Vec<usize>) -> Tensor {
    match dtype {
        DType::F32 => Tensor::F32(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::F64 => Tensor::F64(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::I32 => Tensor::I32(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::I64 => Tensor::I64(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::Bool => {
            let len = shape.iter().product();
            Tensor::Bool(tenferro_tensor::TypedTensor::from_vec_col_major(
                shape,
                vec![false; len],
            ))
        }
        DType::C32 => Tensor::C32(tenferro_tensor::TypedTensor::zeros(shape)),
        DType::C64 => Tensor::C64(tenferro_tensor::TypedTensor::zeros(shape)),
    }
}
