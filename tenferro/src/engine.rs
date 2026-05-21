use std::mem::{size_of, size_of_val};
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;

use super::exec::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_einsum::{ContractionTree, Subscripts};
use tenferro_ops::std_tensor_op::EinsumSubscripts;
use tenferro_tensor::{
    buffer_pool::BufferPoolStats,
    cpu::{CpuBackend, CpuContext},
    CacheStats, RuntimeCacheControl, Tensor, TensorBackend,
};

/// Parsed einsum notation retained separately from shape-specific plans.
pub(crate) struct ParsedEinsum {
    pub(crate) subscripts: EinsumSubscripts,
}

/// Key used for the N-ary einsum cache: `(integer_subscripts, shapes)`.
pub(crate) type EinsumCacheKey = (EinsumSubscripts, Vec<Vec<usize>>);

/// LRU cache of optimized contraction trees keyed by einsum subscripts + input shapes.
pub(crate) type NaryEinsumCache = LruCache<EinsumCacheKey, Arc<ContractionTree>>;

/// LRU cache of parsed einsum subscripts keyed only by notation.
pub(crate) type EinsumParseCache = LruCache<String, Arc<ParsedEinsum>>;

/// Default capacity for `Engine::einsum_cache`.
///
/// Each `ContractionTree` is typically a few KB; 256 entries ≈ under 1 MB.
pub const DEFAULT_EINSUM_CACHE_CAPACITY: usize = 256;

/// Default capacity for compiled execution programs retained by an [`Engine`].
///
/// Each entry is a compiled `ExecProgram` plus its structural cache key. Program
/// size depends on graph size, so the default is bounded conservatively.
pub const DEFAULT_COMPILE_CACHE_CAPACITY: usize = 256;

/// Stats for every cache owned by an [`Engine`].
///
/// `retained_bytes` fields are logical payload estimates, not process RSS.
///
/// # Examples
///
/// ```
/// use tenferro::{engine::EngineCacheStats, CacheStats};
///
/// let stats = EngineCacheStats {
///     compile: CacheStats::empty(),
///     einsum_plans: CacheStats::empty(),
///     einsum_parse: CacheStats::empty(),
///     backend: CacheStats::empty(),
/// };
/// assert_eq!(stats.compile.entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineCacheStats {
    /// Compiled execution-program cache.
    pub compile: CacheStats,
    /// N-ary einsum contraction-plan cache.
    pub einsum_plans: CacheStats,
    /// Parsed einsum-subscript cache.
    pub einsum_parse: CacheStats,
    /// Backend-specific runtime analysis cache.
    pub backend: CacheStats,
}

/// Cache and resource-pool stats for a CPU-backed [`Engine`].
///
/// The CPU buffer pool is reported with cache-style accounting: entries are
/// retained buffers, and retained bytes are retained vector capacity.
/// `thread_pools` reports the process-wide CPU thread-pool handle cache.
///
/// # Examples
///
/// ```
/// use tenferro::{engine::{CpuEngineCacheStats, EngineCacheStats}, CacheStats};
///
/// let stats = CpuEngineCacheStats {
///     engine: EngineCacheStats::default(),
///     buffer_pool: CacheStats::empty(),
///     thread_pools: CacheStats::empty(),
/// };
/// assert_eq!(stats.buffer_pool.retained_bytes, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuEngineCacheStats {
    /// Engine-owned caches.
    pub engine: EngineCacheStats,
    /// CPU backend buffer pool.
    pub buffer_pool: CacheStats,
    /// Process-wide CPU thread-pool handle cache.
    pub thread_pools: CacheStats,
}

/// Cache key derived from the compiled graph topology.
///
/// Uses the number and order of instructions, their op variants, and
/// slot counts as a cheap proxy for structural identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CacheKey {
    /// Number of instructions, input slots, output slots, and total slots.
    shape: (usize, usize, usize, usize),
    /// A hash of the instruction ops (using Debug representation for simplicity).
    op_hash: u64,
}

fn compute_cache_key(exec: &ExecProgram) -> CacheKey {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for inst in &exec.instructions {
        format!("{:?}", inst.op).hash(&mut hasher);
        inst.input_slots.hash(&mut hasher);
        inst.output_slots.hash(&mut hasher);
    }
    exec.input_slots.hash(&mut hasher);
    exec.output_slots.hash(&mut hasher);

    CacheKey {
        shape: (
            exec.instructions.len(),
            exec.input_slots.len(),
            exec.output_slots.len(),
            exec.n_slots,
        ),
        op_hash: hasher.finish(),
    }
}

fn vec_retained_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * size_of::<T>()
}

fn vec_of_vec_retained_bytes<T>(values: &[Vec<T>]) -> usize {
    values.iter().map(vec_retained_bytes).sum()
}

fn einsum_subscripts_retained_bytes(subscripts: &EinsumSubscripts) -> usize {
    vec_of_vec_retained_bytes(&subscripts.inputs) + vec_retained_bytes(&subscripts.output)
}

fn exec_op_retained_bytes(op: &ExecOp) -> usize {
    match op {
        ExecOp::Constant { bytes, .. } => vec_retained_bytes(bytes),
        ExecOp::NaryEinsum { subscripts } => einsum_subscripts_retained_bytes(subscripts),
        ExecOp::Extension(extension) => size_of_val(extension),
        _ => 0,
    }
}

fn exec_instruction_retained_bytes(inst: &ExecInstruction) -> usize {
    size_of::<ExecInstruction>()
        + exec_op_retained_bytes(&inst.op)
        + vec_retained_bytes(&inst.input_slots)
        + vec_retained_bytes(&inst.output_slots)
        + vec_of_vec_retained_bytes(&inst.output_shapes)
        + vec_of_vec_retained_bytes(&inst.output_extents)
        + vec_retained_bytes(&inst.last_use)
}

fn exec_program_retained_bytes(program: &ExecProgram) -> usize {
    size_of::<ExecProgram>()
        + vec_retained_bytes(&program.instructions)
        + program
            .instructions
            .iter()
            .map(exec_instruction_retained_bytes)
            .sum::<usize>()
        + vec_retained_bytes(&program.input_slots)
        + vec_retained_bytes(&program.output_slots)
}

fn compile_cache_stats(cache: &LruCache<CacheKey, ExecProgram>) -> CacheStats {
    CacheStats {
        entries: cache.len(),
        retained_bytes: cache
            .iter()
            .map(|(_, program)| size_of::<CacheKey>() + exec_program_retained_bytes(program))
            .sum(),
    }
}

fn einsum_cache_key_retained_bytes(key: &EinsumCacheKey) -> usize {
    einsum_subscripts_retained_bytes(&key.0) + vec_of_vec_retained_bytes(&key.1)
}

fn nary_einsum_cache_stats(cache: &NaryEinsumCache) -> CacheStats {
    CacheStats {
        entries: cache.len(),
        retained_bytes: cache
            .iter()
            .map(|(key, tree)| {
                einsum_cache_key_retained_bytes(key)
                    + size_of::<Arc<ContractionTree>>()
                    + tree.retained_bytes_for_cache_stats()
            })
            .sum(),
    }
}

fn parsed_einsum_retained_bytes(parsed: &ParsedEinsum) -> usize {
    size_of::<ParsedEinsum>() + einsum_subscripts_retained_bytes(&parsed.subscripts)
}

fn einsum_parse_cache_stats(cache: &EinsumParseCache) -> CacheStats {
    CacheStats {
        entries: cache.len(),
        retained_bytes: cache
            .iter()
            .map(|(notation, parsed)| {
                notation.capacity()
                    + size_of::<Arc<ParsedEinsum>>()
                    + parsed_einsum_retained_bytes(parsed)
            })
            .sum(),
    }
}

/// Execution engine holding the backend and compile caches.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::cpu::CpuBackend;
/// use tenferro::engine::Engine;
///
/// let mut engine = Engine::new(CpuBackend::new());
/// ```
pub struct Engine<B: TensorBackend> {
    pub(crate) backend: B,
    pub(crate) backend_cache: B::RuntimeCache,
    pub(crate) compile_cache: LruCache<CacheKey, ExecProgram>,
    pub(crate) einsum_cache: NaryEinsumCache,
    pub(crate) einsum_parse_cache: EinsumParseCache,
    pub(crate) slot_workspace: Vec<Option<Tensor>>,
}

impl<B: TensorBackend> Engine<B> {
    /// Create a new engine with the given backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuBackend;
    /// use tenferro::engine::Engine;
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// ```
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            backend_cache: B::RuntimeCache::default(),
            compile_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPILE_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
            ),
            einsum_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_EINSUM_CACHE_CAPACITY)
                    .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
            ),
            einsum_parse_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_EINSUM_CACHE_CAPACITY)
                    .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
            ),
            slot_workspace: Vec::new(),
        }
    }

    /// Borrow the backend used by this engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// let _backend = engine.backend();
    /// ```
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Number of cached einsum contraction trees currently retained by the engine.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// assert_eq!(engine.einsum_cache_len(), 0);
    /// ```
    pub fn einsum_cache_len(&self) -> usize {
        self.einsum_cache.len()
    }

    /// Construct a new engine with an explicit `einsum_cache` capacity.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::num::NonZeroUsize;
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::with_einsum_cache_capacity(
    ///     CpuBackend::new(),
    ///     NonZeroUsize::new(64).unwrap(),
    /// );
    /// ```
    pub fn with_einsum_cache_capacity(backend: B, capacity: NonZeroUsize) -> Self {
        Self {
            backend,
            backend_cache: B::RuntimeCache::default(),
            compile_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPILE_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
            ),
            einsum_cache: LruCache::new(capacity),
            einsum_parse_cache: LruCache::new(capacity),
            slot_workspace: Vec::new(),
        }
    }

    /// Current capacity of the einsum contraction-tree cache.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// assert_eq!(engine.einsum_cache_capacity().get(), tenferro::engine::DEFAULT_EINSUM_CACHE_CAPACITY);
    /// ```
    pub fn einsum_cache_capacity(&self) -> NonZeroUsize {
        self.einsum_cache.cap()
    }

    /// Resize the einsum contraction-tree cache.
    ///
    /// Shrinking below the current length evicts least-recently-used entries.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::num::NonZeroUsize;
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.set_einsum_cache_capacity(NonZeroUsize::new(32).unwrap());
    /// ```
    pub fn set_einsum_cache_capacity(&mut self, capacity: NonZeroUsize) {
        self.einsum_cache.resize(capacity);
        self.einsum_parse_cache.resize(capacity);
    }

    /// Number of compiled execution programs currently retained by this engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// assert_eq!(engine.compile_cache_len(), 0);
    /// ```
    pub fn compile_cache_len(&self) -> usize {
        self.compile_cache.len()
    }

    /// Current capacity of the compiled execution-program cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{engine::DEFAULT_COMPILE_CACHE_CAPACITY, CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// assert_eq!(engine.compile_cache_capacity().get(), DEFAULT_COMPILE_CACHE_CAPACITY);
    /// ```
    pub fn compile_cache_capacity(&self) -> NonZeroUsize {
        self.compile_cache.cap()
    }

    /// Resize the compiled execution-program cache.
    ///
    /// Shrinking below the current length evicts least-recently-used programs.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.set_compile_cache_capacity(NonZeroUsize::new(8).unwrap());
    /// assert_eq!(engine.compile_cache_capacity().get(), 8);
    /// ```
    pub fn set_compile_cache_capacity(&mut self, capacity: NonZeroUsize) {
        self.compile_cache.resize(capacity);
    }

    /// Clear the compiled execution-program cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.clear_compile_cache();
    /// assert_eq!(engine.compile_cache_len(), 0);
    /// ```
    pub fn clear_compile_cache(&mut self) {
        self.compile_cache.clear();
    }

    /// Clear cached parsed and planned einsum entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.clear_einsum_caches();
    /// assert_eq!(engine.einsum_cache_len(), 0);
    /// ```
    pub fn clear_einsum_caches(&mut self) {
        self.einsum_cache.clear();
        self.einsum_parse_cache.clear();
    }

    /// Clear the backend-specific runtime cache owned by this engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.clear_backend_cache();
    /// assert_eq!(engine.cache_stats().backend.entries, 0);
    /// ```
    pub fn clear_backend_cache(&mut self) {
        self.backend_cache.clear();
    }

    /// Clear every cache owned directly by this engine.
    ///
    /// For CPU engines, use [`Engine::<CpuBackend>::clear_all_caches`] when the
    /// CPU buffer pool should be cleared together with engine-owned caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.clear_caches();
    /// assert_eq!(engine.cache_stats().compile.entries, 0);
    /// ```
    pub fn clear_caches(&mut self) {
        self.clear_compile_cache();
        self.clear_einsum_caches();
        self.clear_backend_cache();
    }

    /// Return stats for every cache owned directly by this engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// let stats = engine.cache_stats();
    /// assert_eq!(stats.compile.entries, 0);
    /// assert_eq!(stats.einsum_plans.entries, 0);
    /// ```
    pub fn cache_stats(&self) -> EngineCacheStats {
        EngineCacheStats {
            compile: compile_cache_stats(&self.compile_cache),
            einsum_plans: nary_einsum_cache_stats(&self.einsum_cache),
            einsum_parse: einsum_parse_cache_stats(&self.einsum_parse_cache),
            backend: self.backend_cache.stats(),
        }
    }

    /// Returns `true` if the einsum cache contains a tree for `key`.
    ///
    /// Does not modify LRU recency.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// let key = ("ij,jk->ik".to_string(), vec![vec![2, 3], vec![3, 4]]);
    /// assert!(!engine.einsum_cache_contains(&key));
    /// ```
    pub fn einsum_cache_contains(&self, key: &(String, Vec<Vec<usize>>)) -> bool {
        let subscripts = Subscripts::parse(&key.0)
            .map(|subscripts| EinsumSubscripts {
                inputs: subscripts.inputs,
                output: subscripts.output,
            })
            .expect("invalid einsum_cache_contains key");
        let key = (subscripts, key.1.clone());
        self.einsum_cache.contains(&key)
    }

    /// Returns `true` if the einsum cache contains a tree for integer labels.
    pub fn einsum_cache_contains_subscripts(
        &self,
        key: &(EinsumSubscripts, Vec<Vec<usize>>),
    ) -> bool {
        self.einsum_cache.contains(key)
    }

    /// Look up a cached ExecProgram, or cache and return the given one.
    ///
    /// Returns a clone of the cached program to avoid borrow conflicts
    /// with `self.backend`.
    pub(crate) fn get_or_compile(&mut self, exec: ExecProgram) -> ExecProgram {
        let key = compute_cache_key(&exec);
        if let Some(cached) = self.compile_cache.get(&key) {
            return cached.clone();
        }
        self.compile_cache.put(key, exec.clone());
        exec
    }

    /// Evaluate an `ExecProgram` through this engine, reusing the persistent
    /// `einsum_cache` for any `NaryEinsum` ops encountered in the program.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{CpuBackend, Engine};
    /// use tenferro::exec::ExecProgram;
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// // let outputs = engine.eval_exec_ir(&program, inputs)?;
    /// ```
    pub fn eval_exec_ir(
        &mut self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> crate::error::Result<Vec<Tensor>> {
        crate::segment::eval_exec_segmented_with_cache_and_workspace(
            &mut self.backend,
            program,
            inputs,
            &mut self.einsum_cache,
            &mut self.slot_workspace,
            &mut self.backend_cache,
        )
    }

    /// Evaluate an `ExecProgram` without consuming the caller's input tensors.
    ///
    /// This keeps the public ownership choice explicit. Today this clones the
    /// inputs before entering the owned execution path, so it preserves caller
    /// tensors but still pays host clone cost for host-backed inputs.
    pub fn eval_exec_ir_non_consuming(
        &mut self,
        program: &ExecProgram,
        inputs: &[Tensor],
    ) -> crate::error::Result<Vec<Tensor>> {
        self.eval_exec_ir(program, inputs.to_vec())
    }
}

impl Engine<CpuBackend> {
    /// Number of reusable typed host buffers currently retained by the CPU backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// assert_eq!(engine.buffer_pool_len(), 0);
    /// ```
    pub fn buffer_pool_len(&self) -> usize {
        self.backend.buffer_pool_len()
    }

    /// Snapshot reusable typed host buffers currently retained by the CPU backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// let stats = engine.buffer_pool_stats();
    /// assert_eq!(stats.buffers, 0);
    /// assert_eq!(stats.capacity_bytes, 0);
    /// ```
    pub fn buffer_pool_stats(&self) -> BufferPoolStats {
        self.backend.buffer_pool_stats()
    }

    /// Reset all reusable typed host buffers retained by the CPU backend.
    ///
    /// This clears tenferro's explicit buffer pool. The process allocator may
    /// still keep released pages mapped, so operating-system RSS is not a
    /// precise measure of whether the pool is empty.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.reset_buffer_pool();
    /// assert_eq!(engine.buffer_pool_len(), 0);
    /// ```
    pub fn reset_buffer_pool(&mut self) {
        self.backend.reset_buffer_pool();
    }

    /// Return stats for engine caches and the CPU buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// let stats = engine.cpu_cache_stats();
    /// assert_eq!(stats.engine.compile.entries, 0);
    /// assert_eq!(stats.buffer_pool.entries, 0);
    /// ```
    pub fn cpu_cache_stats(&self) -> CpuEngineCacheStats {
        CpuEngineCacheStats {
            engine: self.cache_stats(),
            buffer_pool: self.backend.buffer_pool_cache_stats(),
            thread_pools: CpuContext::shared_pool_cache_stats(),
        }
    }

    /// Clear engine-owned caches and the CPU backend buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.clear_all_caches();
    /// assert_eq!(engine.cpu_cache_stats().buffer_pool.entries, 0);
    /// ```
    pub fn clear_all_caches(&mut self) {
        self.clear_caches();
        self.reset_buffer_pool();
        CpuContext::clear_shared_pool_cache();
    }

    /// Return the CPU GEMM analysis-cache slot capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// assert!(engine.gemm_analysis_cache_capacity() > 0);
    /// ```
    pub fn gemm_analysis_cache_capacity(&self) -> usize {
        self.backend_cache.capacity()
    }

    /// Resize the CPU GEMM analysis-cache slot capacity.
    ///
    /// Shrinking truncates cached slots beyond the new limit. A capacity of zero
    /// disables retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.set_gemm_analysis_cache_capacity(0);
    /// assert_eq!(engine.gemm_analysis_cache_capacity(), 0);
    /// ```
    pub fn set_gemm_analysis_cache_capacity(&mut self, capacity: usize) {
        self.backend_cache.set_capacity(capacity);
    }

    /// Return the CPU buffer-pool retention limit in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let engine = Engine::new(CpuBackend::new());
    /// assert!(engine.buffer_pool_limit_bytes() > 0);
    /// ```
    pub fn buffer_pool_limit_bytes(&self) -> usize {
        self.backend.buffer_pool_limit_bytes()
    }

    /// Update the CPU buffer-pool retention limit in bytes.
    ///
    /// A limit of zero disables buffer retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine};
    ///
    /// let mut engine = Engine::new(CpuBackend::new());
    /// engine.set_buffer_pool_limit_bytes(0);
    /// assert_eq!(engine.buffer_pool_limit_bytes(), 0);
    /// ```
    pub fn set_buffer_pool_limit_bytes(&mut self, max_retained_capacity_bytes: usize) {
        self.backend
            .set_buffer_pool_limit_bytes(max_retained_capacity_bytes);
    }
}
