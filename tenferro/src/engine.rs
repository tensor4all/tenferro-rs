use std::collections::HashMap;

use super::buffer_pool::BufferPool;
use super::exec::ExecProgram;
use tenferro_tensor::TensorBackend;

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

/// Execution engine holding the backend, compile cache, and buffer pool.
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
    pub(crate) buffer_pool: BufferPool,
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
            buffer_pool: BufferPool::new(),
        }
    }

    /// Number of reusable host buffers currently retained by the engine.
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
        self.buffer_pool.len()
    }

    /// Look up a cached ExecProgram, or cache and return the given one.
    ///
    /// Returns a clone of the cached program to avoid borrow conflicts
    /// with `self.backend`.
    pub(crate) fn get_or_compile(&mut self, exec: ExecProgram) -> ExecProgram {
        let key = compute_cache_key(&exec);
        self.compile_cache.entry(key).or_insert(exec).clone()
    }
}
