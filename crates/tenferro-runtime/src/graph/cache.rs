use std::hash::{Hash, Hasher};
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use lru::LruCache;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_tensor::CacheStats;

use crate::exec::{ExecInstruction, ExecOp, ExecOutputExtents, ExecOutputShapes, ExecProgram};

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
    fingerprint: ExecProgramKey,
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
    let mut extensions = Vec::new();
    let fingerprint = exec_program_key(exec, &mut extensions);
    CacheKey {
        fingerprint,
        extensions,
    }
}

fn cache_key_retained_bytes(key: &CacheKey) -> usize {
    saturating_sum([
        size_of::<CacheKey>(),
        exec_program_key_retained_bytes(&key.fingerprint),
        key.extensions
            .capacity()
            .saturating_mul(size_of::<Arc<dyn ExtensionOp>>()),
    ])
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ExecProgramKey {
    instructions: Vec<ExecInstructionKey>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    n_slots: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ExecInstructionKey {
    op: ExecOpKey,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    dtype: tenferro_tensor::DType,
    output_shapes: ExecOutputShapes,
    output_extents: ExecOutputExtents,
    last_use: Vec<bool>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum ExecOpKey {
    Transpose {
        perm: Vec<usize>,
    },
    Reshape {
        shape: Vec<tenferro_ops::dim_expr::DimExpr>,
    },
    BroadcastInDim {
        shape: Vec<tenferro_ops::dim_expr::DimExpr>,
        dims: Vec<usize>,
    },
    Convert {
        to: tenferro_tensor::DType,
    },
    Constant {
        dtype: tenferro_tensor::DType,
        bytes: Vec<u8>,
    },
    DotGeneral(tenferro_tensor::DotGeneralConfig),
    DotGeneralWithConj {
        config: tenferro_tensor::DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    },
    ReduceSum {
        axes: Vec<usize>,
    },
    ExtractDiag {
        axis_a: usize,
        axis_b: usize,
    },
    EmbedDiag {
        axis_a: usize,
        axis_b: usize,
    },
    Tril {
        k: i64,
    },
    Triu {
        k: i64,
    },
    Add,
    Multiply,
    Negate,
    Conj,
    Divide,
    Abs,
    Sign,
    Maximum,
    Minimum,
    Compare(tenferro_tensor::CompareDir),
    Select,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
    Gather(tenferro_tensor::GatherConfig),
    GatherDynamicSliceSizes {
        offset_dims: Vec<usize>,
        collapsed_slice_dims: Vec<usize>,
        start_index_map: Vec<usize>,
        index_vector_dim: usize,
        slice_sizes: Vec<tenferro_ops::dim_expr::DimExpr>,
    },
    Scatter(tenferro_tensor::ScatterConfig),
    Slice(tenferro_tensor::SliceConfig),
    DynamicSlice {
        slice_sizes: Vec<usize>,
    },
    DynamicUpdateSlice,
    Pad(tenferro_tensor::PadConfig),
    Concatenate {
        axis: usize,
    },
    Reverse {
        axes: Vec<usize>,
    },
    ShapeOf {
        axis: usize,
    },
    DynamicTruncate {
        axis: usize,
    },
    PadToMatch {
        axis: usize,
    },
    ReduceProd {
        axes: Vec<usize>,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    Extension {
        family_id: &'static str,
        payload_hash: u64,
    },
}

fn exec_program_key(
    exec: &ExecProgram,
    extensions: &mut Vec<Arc<dyn ExtensionOp>>,
) -> ExecProgramKey {
    ExecProgramKey {
        instructions: exec
            .instructions
            .iter()
            .map(|inst| exec_instruction_key(inst, extensions))
            .collect(),
        input_slots: exec.input_slots.clone(),
        output_slots: exec.output_slots.clone(),
        n_slots: exec.n_slots,
    }
}

fn exec_instruction_key(
    inst: &ExecInstruction,
    extensions: &mut Vec<Arc<dyn ExtensionOp>>,
) -> ExecInstructionKey {
    ExecInstructionKey {
        op: exec_op_key(&inst.op, extensions),
        input_slots: inst.input_slots.clone(),
        output_slots: inst.output_slots.clone(),
        dtype: inst.dtype,
        output_shapes: inst.output_shapes.clone(),
        output_extents: inst.output_extents.clone(),
        last_use: inst.last_use.clone(),
    }
}

fn exec_op_key(op: &ExecOp, extensions: &mut Vec<Arc<dyn ExtensionOp>>) -> ExecOpKey {
    match op {
        ExecOp::Transpose { perm } => ExecOpKey::Transpose { perm: perm.clone() },
        ExecOp::Reshape { shape } => ExecOpKey::Reshape {
            shape: shape.clone(),
        },
        ExecOp::BroadcastInDim { shape, dims } => ExecOpKey::BroadcastInDim {
            shape: shape.clone(),
            dims: dims.clone(),
        },
        ExecOp::Convert { to } => ExecOpKey::Convert { to: *to },
        ExecOp::Constant { dtype, bytes } => ExecOpKey::Constant {
            dtype: *dtype,
            bytes: bytes.clone(),
        },
        ExecOp::DotGeneral(config) => ExecOpKey::DotGeneral(config.clone()),
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => ExecOpKey::DotGeneralWithConj {
            config: config.clone(),
            lhs_conj: *lhs_conj,
            rhs_conj: *rhs_conj,
        },
        ExecOp::ReduceSum { axes } => ExecOpKey::ReduceSum { axes: axes.clone() },
        ExecOp::ExtractDiag { axis_a, axis_b } => ExecOpKey::ExtractDiag {
            axis_a: *axis_a,
            axis_b: *axis_b,
        },
        ExecOp::EmbedDiag { axis_a, axis_b } => ExecOpKey::EmbedDiag {
            axis_a: *axis_a,
            axis_b: *axis_b,
        },
        ExecOp::Tril { k } => ExecOpKey::Tril { k: *k },
        ExecOp::Triu { k } => ExecOpKey::Triu { k: *k },
        ExecOp::Add => ExecOpKey::Add,
        ExecOp::Multiply => ExecOpKey::Multiply,
        ExecOp::Negate => ExecOpKey::Negate,
        ExecOp::Conj => ExecOpKey::Conj,
        ExecOp::Divide => ExecOpKey::Divide,
        ExecOp::Abs => ExecOpKey::Abs,
        ExecOp::Sign => ExecOpKey::Sign,
        ExecOp::Maximum => ExecOpKey::Maximum,
        ExecOp::Minimum => ExecOpKey::Minimum,
        ExecOp::Compare(dir) => ExecOpKey::Compare(dir.clone()),
        ExecOp::Select => ExecOpKey::Select,
        ExecOp::Clamp => ExecOpKey::Clamp,
        ExecOp::Exp => ExecOpKey::Exp,
        ExecOp::Log => ExecOpKey::Log,
        ExecOp::Sin => ExecOpKey::Sin,
        ExecOp::Cos => ExecOpKey::Cos,
        ExecOp::Tanh => ExecOpKey::Tanh,
        ExecOp::Sqrt => ExecOpKey::Sqrt,
        ExecOp::Rsqrt => ExecOpKey::Rsqrt,
        ExecOp::Pow => ExecOpKey::Pow,
        ExecOp::Expm1 => ExecOpKey::Expm1,
        ExecOp::Log1p => ExecOpKey::Log1p,
        ExecOp::Gather(config) => ExecOpKey::Gather(config.clone()),
        ExecOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
        } => ExecOpKey::GatherDynamicSliceSizes {
            offset_dims: offset_dims.clone(),
            collapsed_slice_dims: collapsed_slice_dims.clone(),
            start_index_map: start_index_map.clone(),
            index_vector_dim: *index_vector_dim,
            slice_sizes: slice_sizes.clone(),
        },
        ExecOp::Scatter(config) => ExecOpKey::Scatter(config.clone()),
        ExecOp::Slice(config) => ExecOpKey::Slice(config.clone()),
        ExecOp::DynamicSlice { slice_sizes } => ExecOpKey::DynamicSlice {
            slice_sizes: slice_sizes.clone(),
        },
        ExecOp::DynamicUpdateSlice => ExecOpKey::DynamicUpdateSlice,
        ExecOp::Pad(config) => ExecOpKey::Pad(config.clone()),
        ExecOp::Concatenate { axis } => ExecOpKey::Concatenate { axis: *axis },
        ExecOp::Reverse { axes } => ExecOpKey::Reverse { axes: axes.clone() },
        ExecOp::ShapeOf { axis } => ExecOpKey::ShapeOf { axis: *axis },
        ExecOp::DynamicTruncate { axis } => ExecOpKey::DynamicTruncate { axis: *axis },
        ExecOp::PadToMatch { axis } => ExecOpKey::PadToMatch { axis: *axis },
        ExecOp::ReduceProd { axes } => ExecOpKey::ReduceProd { axes: axes.clone() },
        ExecOp::ReduceMax { axes } => ExecOpKey::ReduceMax { axes: axes.clone() },
        ExecOp::ReduceMin { axes } => ExecOpKey::ReduceMin { axes: axes.clone() },
        ExecOp::Extension(extension) => {
            let key = ExecOpKey::Extension {
                family_id: extension.family_id(),
                payload_hash: extension_payload_hash(extension.as_ref()),
            };
            extensions.push(Arc::clone(extension));
            key
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
    values.capacity().saturating_mul(size_of::<T>())
}

fn vec_of_vec_retained_bytes<T>(values: &[Vec<T>]) -> usize {
    saturating_sum(values.iter().map(vec_retained_bytes))
}

fn exec_program_key_retained_bytes(key: &ExecProgramKey) -> usize {
    saturating_sum([
        size_of::<ExecProgramKey>(),
        vec_retained_bytes(&key.instructions),
        saturating_sum(
            key.instructions
                .iter()
                .map(exec_instruction_key_retained_bytes),
        ),
        vec_retained_bytes(&key.input_slots),
        vec_retained_bytes(&key.output_slots),
    ])
}

fn exec_instruction_key_retained_bytes(key: &ExecInstructionKey) -> usize {
    saturating_sum([
        size_of::<ExecInstructionKey>(),
        exec_op_key_retained_bytes(&key.op),
        vec_retained_bytes(&key.input_slots),
        vec_retained_bytes(&key.output_slots),
        vec_of_vec_retained_bytes(&key.output_shapes),
        vec_of_vec_retained_bytes(&key.output_extents),
        vec_retained_bytes(&key.last_use),
    ])
}

fn exec_op_key_retained_bytes(key: &ExecOpKey) -> usize {
    saturating_sum([
        size_of::<ExecOpKey>(),
        match key {
            ExecOpKey::Transpose { perm } => vec_retained_bytes(perm),
            ExecOpKey::Reshape { shape } => vec_retained_bytes(shape),
            ExecOpKey::BroadcastInDim { shape, dims } => {
                saturating_sum([vec_retained_bytes(shape), vec_retained_bytes(dims)])
            }
            ExecOpKey::Constant { bytes, .. } => vec_retained_bytes(bytes),
            ExecOpKey::DotGeneral(config) => dot_general_config_retained_bytes(config),
            ExecOpKey::DotGeneralWithConj { config, .. } => {
                dot_general_config_retained_bytes(config)
            }
            ExecOpKey::ReduceSum { axes }
            | ExecOpKey::Reverse { axes }
            | ExecOpKey::ReduceProd { axes }
            | ExecOpKey::ReduceMax { axes }
            | ExecOpKey::ReduceMin { axes } => vec_retained_bytes(axes),
            ExecOpKey::Gather(config) => gather_config_retained_bytes(config),
            ExecOpKey::GatherDynamicSliceSizes {
                offset_dims,
                collapsed_slice_dims,
                start_index_map,
                slice_sizes,
                ..
            } => saturating_sum([
                vec_retained_bytes(offset_dims),
                vec_retained_bytes(collapsed_slice_dims),
                vec_retained_bytes(start_index_map),
                vec_retained_bytes(slice_sizes),
            ]),
            ExecOpKey::Scatter(config) => scatter_config_retained_bytes(config),
            ExecOpKey::Slice(config) => slice_config_retained_bytes(config),
            ExecOpKey::DynamicSlice { slice_sizes } => vec_retained_bytes(slice_sizes),
            ExecOpKey::Pad(config) => pad_config_retained_bytes(config),
            ExecOpKey::Convert { .. }
            | ExecOpKey::ExtractDiag { .. }
            | ExecOpKey::EmbedDiag { .. }
            | ExecOpKey::Tril { .. }
            | ExecOpKey::Triu { .. }
            | ExecOpKey::Add
            | ExecOpKey::Multiply
            | ExecOpKey::Negate
            | ExecOpKey::Conj
            | ExecOpKey::Divide
            | ExecOpKey::Abs
            | ExecOpKey::Sign
            | ExecOpKey::Maximum
            | ExecOpKey::Minimum
            | ExecOpKey::Compare(_)
            | ExecOpKey::Select
            | ExecOpKey::Clamp
            | ExecOpKey::Exp
            | ExecOpKey::Log
            | ExecOpKey::Sin
            | ExecOpKey::Cos
            | ExecOpKey::Tanh
            | ExecOpKey::Sqrt
            | ExecOpKey::Rsqrt
            | ExecOpKey::Pow
            | ExecOpKey::Expm1
            | ExecOpKey::Log1p
            | ExecOpKey::DynamicUpdateSlice
            | ExecOpKey::Concatenate { .. }
            | ExecOpKey::ShapeOf { .. }
            | ExecOpKey::DynamicTruncate { .. }
            | ExecOpKey::PadToMatch { .. }
            | ExecOpKey::Extension { .. } => 0,
        },
    ])
}

fn dot_general_config_retained_bytes(config: &tenferro_tensor::DotGeneralConfig) -> usize {
    saturating_sum([
        vec_retained_bytes(&config.lhs_contracting_dims),
        vec_retained_bytes(&config.rhs_contracting_dims),
        vec_retained_bytes(&config.lhs_batch_dims),
        vec_retained_bytes(&config.rhs_batch_dims),
    ])
}

fn gather_config_retained_bytes(config: &tenferro_tensor::GatherConfig) -> usize {
    saturating_sum([
        vec_retained_bytes(&config.offset_dims),
        vec_retained_bytes(&config.collapsed_slice_dims),
        vec_retained_bytes(&config.start_index_map),
        vec_retained_bytes(&config.slice_sizes),
    ])
}

fn scatter_config_retained_bytes(config: &tenferro_tensor::ScatterConfig) -> usize {
    saturating_sum([
        vec_retained_bytes(&config.update_window_dims),
        vec_retained_bytes(&config.inserted_window_dims),
        vec_retained_bytes(&config.scatter_dims_to_operand_dims),
    ])
}

fn slice_config_retained_bytes(config: &tenferro_tensor::SliceConfig) -> usize {
    saturating_sum([
        vec_retained_bytes(&config.starts),
        vec_retained_bytes(&config.limits),
        vec_retained_bytes(&config.strides),
    ])
}

fn pad_config_retained_bytes(config: &tenferro_tensor::PadConfig) -> usize {
    saturating_sum([
        vec_retained_bytes(&config.edge_padding_low),
        vec_retained_bytes(&config.edge_padding_high),
        vec_retained_bytes(&config.interior_padding),
    ])
}

fn exec_op_retained_bytes(op: &ExecOp) -> usize {
    match op {
        ExecOp::Constant { bytes, .. } => vec_retained_bytes(bytes),
        ExecOp::Extension(extension) => size_of_val(extension),
        _ => 0,
    }
}

fn exec_instruction_retained_bytes(inst: &ExecInstruction) -> usize {
    saturating_sum([
        size_of::<ExecInstruction>(),
        exec_op_retained_bytes(&inst.op),
        vec_retained_bytes(&inst.input_slots),
        vec_retained_bytes(&inst.output_slots),
        vec_of_vec_retained_bytes(&inst.output_shapes),
        vec_of_vec_retained_bytes(&inst.output_extents),
        vec_retained_bytes(&inst.last_use),
    ])
}

fn exec_program_retained_bytes(program: &ExecProgram) -> usize {
    saturating_sum([
        size_of::<ExecProgram>(),
        vec_retained_bytes(&program.instructions),
        saturating_sum(
            program
                .instructions
                .iter()
                .map(exec_instruction_retained_bytes),
        ),
        vec_retained_bytes(&program.input_slots),
        vec_retained_bytes(&program.output_slots),
    ])
}

pub(crate) fn compile_cache_stats(cache: &LruCache<CacheKey, ExecProgram>) -> CacheStats {
    CacheStats {
        entries: cache.len(),
        retained_bytes: cache
            .iter()
            .map(|(key, program)| {
                saturating_sum([
                    cache_key_retained_bytes(key),
                    exec_program_retained_bytes(program),
                ])
            })
            .fold(0usize, usize::saturating_add),
    }
}

fn saturating_sum(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().fold(0usize, usize::saturating_add)
}

#[cfg(test)]
mod tests;
