use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;

use super::exec::ExecProgram;
use tenferro_einsum::ContractionTree;
use tenferro_tensor::{cpu::CpuBackend, TensorBackend};

/// Key used for the N-ary einsum cache: `(subscripts, shapes)`.
pub(crate) type EinsumCacheKey = (String, Vec<Vec<usize>>);

/// LRU cache of optimized contraction trees keyed by einsum subscripts + input shapes.
pub(crate) type NaryEinsumCache = LruCache<EinsumCacheKey, Arc<ContractionTree>>;

/// Default capacity for `Engine::einsum_cache`.
///
/// Each `ContractionTree` is typically a few KB; 256 entries ≈ under 1 MB.
pub const DEFAULT_EINSUM_CACHE_CAPACITY: usize = 256;

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
    pub(crate) compile_cache: HashMap<CacheKey, ExecProgram>,
    pub(crate) einsum_cache: NaryEinsumCache,
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
            compile_cache: HashMap::new(),
            einsum_cache: LruCache::new(
                NonZeroUsize::new(DEFAULT_EINSUM_CACHE_CAPACITY)
                    .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
            ),
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
            compile_cache: HashMap::new(),
            einsum_cache: LruCache::new(capacity),
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
        self.einsum_cache.contains(key)
    }

    /// Look up a cached ExecProgram, or cache and return the given one.
    ///
    /// Returns a clone of the cached program to avoid borrow conflicts
    /// with `self.backend`.
    pub(crate) fn get_or_compile(&mut self, exec: ExecProgram) -> ExecProgram {
        let key = compute_cache_key(&exec);
        self.compile_cache.entry(key).or_insert(exec).clone()
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
        inputs: Vec<tenferro_tensor::Tensor>,
    ) -> crate::error::Result<Vec<tenferro_tensor::Tensor>> {
        crate::segment::eval_exec_segmented_with_cache(
            &mut self.backend,
            program,
            inputs,
            &mut self.einsum_cache,
        )
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
}
