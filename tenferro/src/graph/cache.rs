use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use lru::LruCache;
use tenferro_einsum::ContractionTree;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::EinsumSubscripts;
use tenferro_tensor::CacheStats;

use crate::exec::{ExecInstruction, ExecOp, ExecProgram};

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

/// Default capacity for graph-compiler static einsum caches.
pub(crate) const DEFAULT_EINSUM_CACHE_CAPACITY: usize = 256;

/// Default capacity for compiled graph programs retained by a [`GraphCompiler`](super::GraphCompiler).
#[allow(dead_code)]
pub const DEFAULT_GRAPH_COMPILE_CACHE_CAPACITY: usize = 256;

/// Internal alias matching the existing engine cache helper name.
pub(crate) const DEFAULT_COMPILE_CACHE_CAPACITY: usize = DEFAULT_GRAPH_COMPILE_CACHE_CAPACITY;

/// Stats for caches owned by a [`GraphCompiler`](super::GraphCompiler).
///
/// `retained_bytes` fields are logical payload estimates, not process RSS.
///
/// # Examples
///
/// ```
/// use tenferro::{CacheStats, GraphCompilerCacheStats};
///
/// let stats = GraphCompilerCacheStats {
///     compile: CacheStats::empty(),
///     static_einsum_plans: CacheStats::empty(),
///     einsum_parse: CacheStats::empty(),
/// };
/// assert_eq!(stats.compile.entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphCompilerCacheStats {
    /// Compiled execution-program cache.
    pub compile: CacheStats,
    /// Static N-ary einsum contraction-plan cache.
    pub static_einsum_plans: CacheStats,
    /// Parsed einsum-subscript cache.
    pub einsum_parse: CacheStats,
}

/// Stats for runtime caches owned by a future graph executor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct GraphExecutorCacheStats {
    /// Runtime N-ary einsum contraction-plan cache.
    pub runtime_einsum_plans: CacheStats,
    /// Backend-specific runtime analysis cache.
    pub backend: CacheStats,
}

/// Stats for CPU graph-executor runtime caches and resource pools.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct CpuGraphExecutorCacheStats {
    /// Executor-owned runtime caches.
    pub executor: GraphExecutorCacheStats,
    /// CPU backend buffer pool.
    pub buffer_pool: CacheStats,
}

/// Cache key derived from compiled graph topology and execution metadata.
#[derive(Clone, Debug)]
pub(crate) struct CacheKey {
    fingerprint: String,
    extensions: Vec<Arc<dyn ExtensionOp>>,
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
            && self.extensions.len() == other.extensions.len()
            && self
                .extensions
                .iter()
                .zip(&other.extensions)
                .all(|(lhs, rhs)| {
                    lhs.family_id() == rhs.family_id() && lhs.payload_eq(rhs.as_ref())
                })
    }
}

impl Eq for CacheKey {}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
    }
}

pub(crate) fn compute_cache_key(exec: &ExecProgram) -> CacheKey {
    let mut fingerprint = String::new();
    let mut extensions = Vec::new();
    write_exec_program_fingerprint(exec, &mut fingerprint, &mut extensions);
    CacheKey {
        fingerprint,
        extensions,
    }
}

fn cache_key_retained_bytes(key: &CacheKey) -> usize {
    size_of::<CacheKey>()
        + key.fingerprint.capacity()
        + key.extensions.capacity() * size_of::<Arc<dyn ExtensionOp>>()
}

fn write_exec_program_fingerprint(
    exec: &ExecProgram,
    out: &mut String,
    extensions: &mut Vec<Arc<dyn ExtensionOp>>,
) {
    let _ = write!(
        out,
        "program inputs={:?}; outputs={:?}; n_slots={}; instructions=[",
        exec.input_slots, exec.output_slots, exec.n_slots
    );
    for inst in &exec.instructions {
        write_exec_instruction_fingerprint(inst, out, extensions);
    }
    out.push(']');
}

fn write_exec_instruction_fingerprint(
    inst: &ExecInstruction,
    out: &mut String,
    extensions: &mut Vec<Arc<dyn ExtensionOp>>,
) {
    let _ = write!(
        out,
        "inst inputs={:?}; outputs={:?}; dtype={:?}; shapes={:?}; extents={:?}; last_use={:?}; op=",
        inst.input_slots,
        inst.output_slots,
        inst.dtype,
        inst.output_shapes,
        inst.output_extents,
        inst.last_use
    );
    write_exec_op_fingerprint(&inst.op, out, extensions);
    out.push(';');
}

fn write_exec_op_fingerprint(
    op: &ExecOp,
    out: &mut String,
    extensions: &mut Vec<Arc<dyn ExtensionOp>>,
) {
    match op {
        ExecOp::Extension(extension) => {
            let payload_hash = extension_payload_hash(extension.as_ref());
            let _ = write!(
                out,
                "Extension(family_id={:?}, payload_hash={payload_hash})",
                extension.family_id()
            );
            extensions.push(Arc::clone(extension));
        }
        other => {
            let _ = write!(out, "{other:?}");
        }
    }
}

fn extension_payload_hash(extension: &dyn ExtensionOp) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    extension.payload_hash(&mut DynHasherProxy::new(&mut hasher));
    hasher.finish()
}

struct DynHasherProxy<'a, H: Hasher + ?Sized> {
    inner: &'a mut H,
}

impl<'a, H: Hasher + ?Sized> DynHasherProxy<'a, H> {
    fn new(inner: &'a mut H) -> Self {
        Self { inner }
    }
}

impl<H: Hasher + ?Sized> Hasher for DynHasherProxy<'_, H> {
    fn finish(&self) -> u64 {
        self.inner.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.inner.write(bytes);
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

pub(crate) fn compile_cache_stats(cache: &LruCache<CacheKey, ExecProgram>) -> CacheStats {
    CacheStats {
        entries: cache.len(),
        retained_bytes: cache
            .iter()
            .map(|(key, program)| {
                cache_key_retained_bytes(key) + exec_program_retained_bytes(program)
            })
            .sum(),
    }
}

fn einsum_cache_key_retained_bytes(key: &EinsumCacheKey) -> usize {
    einsum_subscripts_retained_bytes(&key.0) + vec_of_vec_retained_bytes(&key.1)
}

pub(crate) fn nary_einsum_cache_stats(cache: &NaryEinsumCache) -> CacheStats {
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

pub(crate) fn einsum_parse_cache_stats(cache: &EinsumParseCache) -> CacheStats {
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
