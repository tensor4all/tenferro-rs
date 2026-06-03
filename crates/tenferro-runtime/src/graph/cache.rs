use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use lru::LruCache;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_tensor::CacheStats;

use crate::exec::{ExecInstruction, ExecOp, ExecProgram};

/// Default capacity for compiled graph programs retained by a [`GraphCompiler`](super::GraphCompiler).
// Public constant kept as the documented default; the crate-local alias below
// is what current implementation paths consume.
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
/// use tenferro_runtime::{CacheStats, GraphCompilerCacheStats};
///
/// let stats = GraphCompilerCacheStats {
///     compile: CacheStats::empty(),
///     extensions: CacheStats::empty(),
/// };
/// assert_eq!(stats.compile.entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphCompilerCacheStats {
    /// Compiled execution-program cache.
    pub compile: CacheStats,
    /// Generic extension compile-time caches.
    pub extensions: CacheStats,
}

/// Stats for runtime caches owned by a graph executor.
///
/// `retained_bytes` fields are logical payload estimates, not process RSS.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{CacheStats, GraphExecutorCacheStats};
///
/// let stats = GraphExecutorCacheStats {
///     extensions: CacheStats::empty(),
///     backend: CacheStats::empty(),
/// };
/// assert_eq!(stats.extensions.entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphExecutorCacheStats {
    /// Generic extension runtime caches.
    pub extensions: CacheStats,
    /// Backend-specific runtime analysis cache.
    pub backend: CacheStats,
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

fn exec_op_retained_bytes(op: &ExecOp) -> usize {
    match op {
        ExecOp::Constant { bytes, .. } => vec_retained_bytes(bytes),
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
