//! AD context for guard-based shape resolution and value metadata queries.
//!
//! During AD graph construction, linalg rules such as SVD, QR, and LU need
//! concrete matrix dimensions to choose between structurally different
//! subgraphs. `ShapeGuardContext` records those dimension comparisons as guards
//! so cached AD graphs can later be invalidated when the observed shape
//! relationship changes.

use std::cmp::Ordering;
use std::collections::HashMap;
#[cfg(feature = "autodiff")]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use computegraph::graph::Graph;
use computegraph::types::{ValueKey, ValueRef};
use tenferro_tensor::DType;

use crate::dim_expr::{DimExpr, DimExprEvalError};
#[cfg(feature = "autodiff")]
use crate::ext_op::{
    ExtensionLinearTransposeRule, ExtensionLinearizeRule, ExtensionPrimalVjpRule, ExtensionRuleSet,
};
use crate::shape_extent::ShapeExtent;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

type MetadataMap = HashMap<ValueKey<StdTensorOp>, TensorMeta>;

type GlobalMetadataMap = HashMap<ValueKey<StdTensorOp>, GlobalMetadataEntry>;

#[derive(Clone, Debug)]
struct GlobalMetadataEntry {
    meta: TensorMeta,
    scoped_refs: usize,
}

/// Error returned when the process-global AD metadata registry is unavailable.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MetadataRegistryError {
    /// A previous panic poisoned the global metadata mutex.
    #[error("AD global metadata registry lock poisoned")]
    LockPoisoned,
}

/// Error returned when shape-guard metadata cannot be resolved.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ShapeGuardError {
    /// A local graph value was queried before a graph was attached.
    #[error("cannot resolve local value {local_id} without an attached graph")]
    LocalWithoutAttachedGraph {
        /// Graph-local value id.
        local_id: usize,
    },
    /// A local graph value id is outside the attached graph's value table.
    #[error("local value {local_id} is out of bounds for the attached graph")]
    LocalOutOfBounds {
        /// Graph-local value id.
        local_id: usize,
    },
    /// No metadata was registered for the resolved value key.
    #[error("missing TensorMeta for {key:?}")]
    MissingMetadata {
        /// Resolved value key.
        key: ValueKey<StdTensorOp>,
    },
    /// Metadata exists, but at least one axis is only bounded or unknown.
    #[error("TensorMeta for {key:?} does not have an exact shape; query extents instead")]
    NonExactShape {
        /// Resolved value key.
        key: ValueKey<StdTensorOp>,
    },
}

/// Result type used by shape-guard metadata queries.
pub type ShapeGuardResult<T> = Result<T, ShapeGuardError>;

#[cfg(feature = "autodiff")]
impl From<ShapeGuardError> for tidu::ADRuleError {
    fn from(err: ShapeGuardError) -> Self {
        tidu::ADRuleError::invalid_input(
            "tenferro.shape_guard",
            tidu::ADRuleKind::Jvp,
            err.to_string(),
        )
    }
}

/// Global metadata registry.
///
/// Stored as `Mutex<MetadataMap>` directly: writes insert in place (O(1)),
/// and reads lock briefly for targeted key lookups. `ShapeGuardContext::metadata_of`
/// reaches into the registry lazily via [`lookup_global_metadata`] and caches the
/// result into the context's local map.
///
/// Earlier designs either cloned the whole map up-front into each AD
/// `ShapeGuardContext` or kept the map in an `Arc` and cloned on every write.
/// Both variants were quadratic across the monotonically growing registry and
/// dominated oracle_replay runtime.
static GLOBAL_METADATA: OnceLock<Mutex<GlobalMetadataMap>> = OnceLock::new();

fn global_metadata_registry() -> &'static Mutex<GlobalMetadataMap> {
    GLOBAL_METADATA.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lifetime token for graph-scoped global metadata.
///
/// Dropping the last frontend owner of a traced graph drops this scope and
/// releases the metadata keys that were registered for that graph graph.
#[doc(hidden)]
#[derive(Debug)]
pub struct GlobalMetadataScope {
    keys: Vec<ValueKey<StdTensorOp>>,
}

impl Drop for GlobalMetadataScope {
    fn drop(&mut self) {
        release_scoped_global_metadata(&self.keys);
    }
}

/// Per-value tensor metadata used by AD rules.
///
/// Shape information is stored as per-axis [`ShapeExtent`] values. Callers must
/// explicitly choose whether they need an exact shape or only a known bound.
///
/// # Examples
///
/// ```
/// use tenferro_ops::{SymDim, TensorMeta};
/// use tenferro_tensor::DType;
///
/// let meta = TensorMeta::exact(DType::F64, vec![SymDim::from(2usize), SymDim::from(3usize)]);
/// assert_eq!(meta.rank(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    /// Element dtype of the tensor value.
    pub dtype: DType,
    /// Per-axis shape guarantees.
    pub extents: Vec<ShapeExtent<SymDim>>,
}

impl TensorMeta {
    /// Construct metadata whose every axis is exact.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::{SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let meta = TensorMeta::exact(DType::F64, vec![SymDim::from(4usize)]);
    /// assert_eq!(meta.exact_shape(), Some(vec![SymDim::from(4usize)]));
    /// ```
    pub fn exact(dtype: DType, shape: Vec<SymDim>) -> Self {
        let extents = shape.iter().cloned().map(ShapeExtent::exact).collect();
        Self { dtype, extents }
    }

    /// Construct metadata from per-axis extents.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::{ShapeExtent, SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let meta = TensorMeta::with_extents(
    ///     DType::F64,
    ///     vec![ShapeExtent::upper_bound(SymDim::from(8usize))],
    /// );
    /// assert_eq!(meta.exact_shape(), None);
    /// ```
    pub fn with_extents(dtype: DType, extents: Vec<ShapeExtent<SymDim>>) -> Self {
        Self { dtype, extents }
    }

    /// Return the tensor rank known by this metadata record.
    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    /// Return the per-axis shape guarantees.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::{SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let meta = TensorMeta::exact(DType::F64, vec![SymDim::from(4usize)]);
    /// assert_eq!(meta.extents().len(), 1);
    /// ```
    pub fn extents(&self) -> &[ShapeExtent<SymDim>] {
        &self.extents
    }

    /// Return the shape only when every axis is exact.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::{ShapeExtent, SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let meta = TensorMeta::with_extents(
    ///     DType::F64,
    ///     vec![ShapeExtent::upper_bound(SymDim::from(8usize))],
    /// );
    /// assert_eq!(meta.exact_shape(), None);
    /// ```
    pub fn exact_shape(&self) -> Option<Vec<SymDim>> {
        self.extents
            .iter()
            .map(|extent| extent.as_exact().cloned())
            .collect()
    }

    /// Return one known bound per axis when every axis has a bound.
    ///
    /// This is intentionally separate from [`TensorMeta::exact_shape`]: a bound
    /// is not proof of the runtime size.
    pub fn bound_shape(&self) -> Option<Vec<SymDim>> {
        self.extents
            .iter()
            .map(|extent| extent.bound_expr().cloned())
            .collect()
    }
}

/// A recorded dimension comparison made during AD graph construction.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
/// use tenferro_ops::ShapeGuard;
///
/// let guard = ShapeGuard {
///     dim_a: 5,
///     dim_b: 3,
///     ordering: Ordering::Greater,
/// };
///
/// assert_eq!(guard.ordering, Ordering::Greater);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeGuard {
    /// First dimension value, such as `m`.
    pub dim_a: usize,
    /// Second dimension value, such as `n`.
    pub dim_b: usize,
    /// The observed ordering `dim_a.cmp(&dim_b)`.
    pub ordering: Ordering,
}

/// AD context providing dimension resolution, guard recording, and value metadata.
///
/// # Examples
///
/// ```
/// use tenferro_ops::ShapeGuardContext;
///
/// let ctx = ShapeGuardContext::default();
/// assert!(ctx.guards().is_empty());
/// ```
#[derive(Clone, Debug, Default)]
pub struct ShapeGuardContext {
    guards: Vec<ShapeGuard>,
    metadata: MetadataMap,
    use_global_registry: bool,
    local_keys: Option<Vec<ValueKey<StdTensorOp>>>,
    #[cfg(feature = "autodiff")]
    extension_rules: Option<ExtensionRuleSet>,
    #[cfg(feature = "autodiff")]
    active_value_keys: Option<std::sync::Arc<std::collections::HashSet<ValueKey<StdTensorOp>>>>,
    #[cfg(feature = "autodiff")]
    transpose_primal_outputs: Option<Vec<ValueKey<StdTensorOp>>>,
    #[cfg(feature = "autodiff")]
    transpose_primal_outputs_used: bool,
}

impl ShapeGuardContext {
    /// Create a context backed by the global metadata registry.
    ///
    /// Instead of cloning the entire global registry up-front (which used
    /// to be O(N) per AD pass and quadratic across oracle_replay), the
    /// context keeps a flag and lazily fetches entries from the shared
    /// [`lookup_global_metadata`] on first miss, caching into its local
    /// `metadata` map for subsequent reads within the same pass.
    ///
    /// # Examples
    ///
    /// ```
    /// let ctx = tenferro_ops::ShapeGuardContext::with_global_metadata();
    /// assert!(ctx.guards().is_empty());
    /// ```
    pub fn with_global_metadata() -> Self {
        Self {
            use_global_registry: true,
            ..Self::default()
        }
    }

    #[doc(hidden)]
    /// Keep global-registry lookup enabled after a pass boundary.
    ///
    /// This is intentionally a no-op for cached entries: global metadata is
    /// already read lazily on cache misses, and clearing the local cache would
    /// also discard metadata inserted directly into this context.
    pub fn refresh_global_metadata(&mut self) {
        self.use_global_registry = true;
    }

    /// Use an explicit extension AD rule set for this context.
    ///
    /// Extension AD lookup is context-owned: a context without an attached rule
    /// set has no extension AD rules.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::{ExtensionRuleSet, ShapeGuardContext};
    ///
    /// let _ctx = ShapeGuardContext::default().with_extension_rules(ExtensionRuleSet::new());
    /// ```
    #[cfg(feature = "autodiff")]
    pub fn with_extension_rules(mut self, rules: ExtensionRuleSet) -> Self {
        self.extension_rules = Some(rules);
        self
    }

    #[cfg(feature = "autodiff")]
    pub fn with_linearize_active_values(
        mut self,
        keys: std::sync::Arc<std::collections::HashSet<ValueKey<StdTensorOp>>>,
    ) -> Self {
        self.active_value_keys = Some(keys);
        self
    }

    /// Whether a primal value lies on a path from the current linearize targets.
    ///
    /// When no active set was attached, every value is treated as active so
    /// existing callers keep the conservative full JVP graphs.
    #[cfg(feature = "autodiff")]
    pub fn is_value_active_in_linearize(&self, key: &ValueKey<StdTensorOp>) -> bool {
        self.active_value_keys
            .as_ref()
            .is_none_or(|set| set.contains(key))
    }

    /// Primal output keys for the operation currently being transposed.
    ///
    /// Primary-mode extension transpose rules such as `Eigh` use these to reuse
    /// forward eigenvectors instead of recomputing a decomposition.
    #[cfg(feature = "autodiff")]
    pub fn set_transpose_primal_outputs(&mut self, keys: Option<Vec<ValueKey<StdTensorOp>>>) {
        self.transpose_primal_outputs = keys;
        self.transpose_primal_outputs_used = false;
    }

    /// Return the current primal outputs and mark them as consumed by this rule.
    #[cfg(feature = "autodiff")]
    pub fn transpose_primal_outputs(&mut self) -> Option<&[ValueKey<StdTensorOp>]> {
        if self.transpose_primal_outputs.is_some() {
            self.transpose_primal_outputs_used = true;
        }
        self.transpose_primal_outputs.as_deref()
    }

    #[cfg(feature = "autodiff")]
    pub fn transpose_primal_outputs_were_used(&self) -> bool {
        self.transpose_primal_outputs_used
    }

    /// Look up an extension linearize rule using this context's ownership policy.
    ///
    /// Contexts without an explicit rule set have no extension AD rules.
    #[doc(hidden)]
    #[cfg(feature = "autodiff")]
    pub(crate) fn extension_linearize_rule_for(
        &self,
        family_id: &str,
    ) -> Option<Arc<dyn ExtensionLinearizeRule>> {
        self.extension_rules
            .as_ref()
            .and_then(|rules| rules.lookup_linearize(family_id))
    }

    /// Look up an extension linear-transpose rule using this context's
    /// ownership policy.
    #[doc(hidden)]
    #[cfg(feature = "autodiff")]
    pub(crate) fn extension_linear_transpose_rule_for(
        &self,
        family_id: &str,
    ) -> Option<Arc<dyn ExtensionLinearTransposeRule>> {
        self.extension_rules
            .as_ref()
            .and_then(|rules| rules.lookup_linear_transpose(family_id))
    }

    /// Look up an extension direct primal-VJP rule using this context's
    /// ownership policy.
    #[doc(hidden)]
    #[cfg(feature = "autodiff")]
    pub(crate) fn extension_primal_vjp_rule_for(
        &self,
        family_id: &str,
    ) -> Option<Arc<dyn ExtensionPrimalVjpRule>> {
        self.extension_rules
            .as_ref()
            .and_then(|rules| rules.lookup_primal_vjp(family_id))
    }

    /// Returns the guards recorded so far.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ShapeGuardContext;
    ///
    /// let ctx = ShapeGuardContext::default();
    /// assert_eq!(ctx.guards(), &[]);
    /// ```
    pub fn guards(&self) -> &[ShapeGuard] {
        &self.guards
    }

    /// Clears all recorded guards.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ShapeGuardContext;
    ///
    /// let mut ctx = ShapeGuardContext::default();
    /// ctx.clear_guards();
    /// assert!(ctx.guards().is_empty());
    /// ```
    pub fn clear_guards(&mut self) {
        self.guards.clear();
    }

    /// Return the shape metadata for a value reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use computegraph::types::{ValueKey, ValueRef};
    /// use tenferro_ops::input_key::TensorInputKey;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    /// use tenferro_ops::{ShapeGuardContext, SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 1 });
    /// let value = ValueRef::External(key.clone());
    /// let mut ctx = ShapeGuardContext::default();
    /// ctx.insert_metadata(key, TensorMeta::exact(DType::F64, vec![SymDim::from(4usize)]));
    ///
    /// let shape = ctx.shape_of(&value).unwrap();
    /// assert_eq!(shape, &[SymDim::from(4usize)]);
    /// ```
    pub fn shape_of(&mut self, val: &ValueRef<StdTensorOp>) -> ShapeGuardResult<Vec<SymDim>> {
        let key = self.resolve_key(val)?.clone();
        self.ensure_metadata_loaded(&key);
        let meta = self
            .metadata
            .get(&key)
            .ok_or_else(|| ShapeGuardError::MissingMetadata { key: key.clone() })?;
        meta.exact_shape()
            .ok_or(ShapeGuardError::NonExactShape { key })
    }

    /// Return the rank for a value reference without requiring exact extents.
    ///
    /// Use this when an AD rule only needs axis count or needs to build
    /// runtime-shape references. Calling [`ShapeGuardContext::shape_of`] in those
    /// cases would reject valid values such as `DynamicTruncate` outputs whose
    /// runtime extent is known only as an upper bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use computegraph::types::{ValueKey, ValueRef};
    /// use tenferro_ops::input_key::TensorInputKey;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    /// use tenferro_ops::{ShapeExtent, ShapeGuardContext, SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 1 });
    /// let value = ValueRef::External(key.clone());
    /// let mut ctx = ShapeGuardContext::default();
    /// ctx.insert_metadata(
    ///     key,
    ///     TensorMeta::with_extents(DType::F64, vec![ShapeExtent::upper_bound(SymDim::from(8usize))]),
    /// );
    ///
    /// assert_eq!(ctx.rank_of(&value).unwrap(), 1);
    /// ```
    pub fn rank_of(&mut self, val: &ValueRef<StdTensorOp>) -> ShapeGuardResult<usize> {
        self.metadata_of(val).map(TensorMeta::rank)
    }

    /// Return per-axis shape guarantees for a value reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use computegraph::types::{ValueKey, ValueRef};
    /// use tenferro_ops::input_key::TensorInputKey;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    /// use tenferro_ops::{ShapeExtent, ShapeGuardContext, SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 1 });
    /// let value = ValueRef::External(key.clone());
    /// let mut ctx = ShapeGuardContext::default();
    /// ctx.insert_metadata(
    ///     key,
    ///     TensorMeta::with_extents(DType::F64, vec![ShapeExtent::upper_bound(SymDim::from(8usize))]),
    /// );
    ///
    /// let extents = ctx.extents_of(&value).unwrap();
    /// assert_eq!(extents[0], ShapeExtent::upper_bound(SymDim::from(8usize)));
    /// ```
    pub fn extents_of(
        &mut self,
        val: &ValueRef<StdTensorOp>,
    ) -> ShapeGuardResult<&[ShapeExtent<SymDim>]> {
        self.metadata_of(val).map(TensorMeta::extents)
    }

    /// Return the exact shape for a value reference, if all axes are exact.
    ///
    /// # Examples
    ///
    /// ```
    /// use computegraph::types::{ValueKey, ValueRef};
    /// use tenferro_ops::input_key::TensorInputKey;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    /// use tenferro_ops::{ShapeExtent, ShapeGuardContext, SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 1 });
    /// let value = ValueRef::External(key.clone());
    /// let mut ctx = ShapeGuardContext::default();
    /// ctx.insert_metadata(
    ///     key,
    ///     TensorMeta::with_extents(DType::F64, vec![ShapeExtent::upper_bound(SymDim::from(8usize))]),
    /// );
    ///
    /// let maybe_shape = ctx.exact_shape_of(&value).unwrap();
    /// assert_eq!(maybe_shape, None);
    /// ```
    pub fn exact_shape_of(
        &mut self,
        val: &ValueRef<StdTensorOp>,
    ) -> ShapeGuardResult<Option<Vec<SymDim>>> {
        self.metadata_of(val).map(TensorMeta::exact_shape)
    }

    #[doc(hidden)]
    pub fn shape_if_available(&mut self, val: &ValueRef<StdTensorOp>) -> Option<Vec<SymDim>> {
        self.metadata_if_available(val)
            .and_then(TensorMeta::exact_shape)
    }

    /// Return the dtype metadata for a value reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use computegraph::types::{ValueKey, ValueRef};
    /// use tenferro_ops::input_key::TensorInputKey;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    /// use tenferro_ops::{ShapeGuardContext, SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 1 });
    /// let value = ValueRef::External(key.clone());
    /// let mut ctx = ShapeGuardContext::default();
    /// ctx.insert_metadata(key, TensorMeta::exact(DType::F64, vec![SymDim::from(4usize)]));
    ///
    /// let dtype = ctx.dtype_of(&value).unwrap();
    /// assert_eq!(dtype, DType::F64);
    /// ```
    pub fn dtype_of(&mut self, val: &ValueRef<StdTensorOp>) -> ShapeGuardResult<DType> {
        self.metadata_of(val).map(|meta| meta.dtype)
    }

    /// Return the complete metadata record for a value reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use computegraph::types::{ValueKey, ValueRef};
    /// use tenferro_ops::input_key::TensorInputKey;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    /// use tenferro_ops::{ShapeGuardContext, SymDim, TensorMeta};
    /// use tenferro_tensor::DType;
    ///
    /// let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 1 });
    /// let value = ValueRef::External(key.clone());
    /// let mut ctx = ShapeGuardContext::default();
    /// ctx.insert_metadata(key, TensorMeta::exact(DType::F64, vec![SymDim::from(4usize)]));
    ///
    /// let meta = ctx.metadata_of(&value).unwrap();
    /// assert_eq!(meta.dtype, DType::F64);
    /// ```
    pub fn metadata_of(&mut self, val: &ValueRef<StdTensorOp>) -> ShapeGuardResult<&TensorMeta> {
        let key = self.resolve_key(val)?.clone();
        self.ensure_metadata_loaded(&key);
        self.metadata
            .get(&key)
            .ok_or(ShapeGuardError::MissingMetadata { key })
    }

    #[doc(hidden)]
    pub fn metadata_if_available(&mut self, val: &ValueRef<StdTensorOp>) -> Option<&TensorMeta> {
        let key = self.resolve_key_if_available(val)?.clone();
        self.ensure_metadata_loaded(&key);
        self.metadata.get(&key)
    }

    #[doc(hidden)]
    pub fn attach_graph(&mut self, graph: &Graph<StdTensorOp>) {
        self.local_keys = Some(graph.values().iter().map(|node| node.key.clone()).collect());
    }

    #[doc(hidden)]
    pub fn insert_metadata(&mut self, key: ValueKey<StdTensorOp>, meta: TensorMeta) {
        self.metadata.insert(key, meta);
    }

    #[doc(hidden)]
    pub fn extend_metadata<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
    {
        self.metadata.extend(entries);
    }

    fn resolve_key_if_available<'a>(
        &'a self,
        val: &'a ValueRef<StdTensorOp>,
    ) -> Option<&'a ValueKey<StdTensorOp>> {
        match val {
            ValueRef::External(key) => Some(key),
            ValueRef::Local(local_id) => self
                .local_keys
                .as_ref()
                .and_then(|keys| keys.get(*local_id)),
        }
    }

    fn resolve_key<'a>(
        &'a self,
        val: &'a ValueRef<StdTensorOp>,
    ) -> ShapeGuardResult<&'a ValueKey<StdTensorOp>> {
        match val {
            ValueRef::External(key) => Ok(key),
            ValueRef::Local(local_id) if self.local_keys.is_none() => {
                Err(ShapeGuardError::LocalWithoutAttachedGraph {
                    local_id: *local_id,
                })
            }
            ValueRef::Local(local_id) => self
                .local_keys
                .as_ref()
                .and_then(|keys| keys.get(*local_id))
                .ok_or(ShapeGuardError::LocalOutOfBounds {
                    local_id: *local_id,
                }),
        }
    }

    fn ensure_metadata_loaded(&mut self, key: &ValueKey<StdTensorOp>) {
        if !self.metadata.contains_key(key) && self.use_global_registry {
            if let Ok(Some(meta)) = lookup_global_metadata(key) {
                self.metadata.insert(key.clone(), meta);
            }
        }
    }
}

/// Look up a single metadata entry from the global registry.
///
/// Locks the registry briefly for a single `HashMap::get` + clone.
///
/// # Examples
///
/// ```
/// use computegraph::types::ValueKey;
/// use tenferro_ops::ad::context::lookup_global_metadata;
/// use tenferro_ops::input_key::TensorInputKey;
/// use tenferro_ops::std_tensor_op::StdTensorOp;
///
/// let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 99 });
/// let meta = lookup_global_metadata(&key).unwrap();
/// assert!(meta.is_none());
/// ```
pub fn lookup_global_metadata(
    key: &ValueKey<StdTensorOp>,
) -> Result<Option<TensorMeta>, MetadataRegistryError> {
    let guard = global_metadata_registry()
        .lock()
        .map_err(|_| MetadataRegistryError::LockPoisoned)?;
    Ok(guard.get(key).map(|entry| entry.meta.clone()))
}

#[doc(hidden)]
pub fn register_scoped_global_metadata_batch<I>(
    entries: I,
) -> Result<GlobalMetadataScope, MetadataRegistryError>
where
    I: IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
{
    let mut guard = global_metadata_registry()
        .lock()
        .map_err(|_| MetadataRegistryError::LockPoisoned)?;
    let mut keys = Vec::new();
    for (key, meta) in entries {
        let entry = guard.entry(key.clone()).or_insert(GlobalMetadataEntry {
            meta: meta.clone(),
            scoped_refs: 0,
        });
        entry.meta = meta;
        entry.scoped_refs += 1;
        keys.push(key);
    }
    Ok(GlobalMetadataScope { keys })
}

fn release_scoped_global_metadata(keys: &[ValueKey<StdTensorOp>]) {
    let Ok(mut guard) = global_metadata_registry().lock() else {
        // Drop cannot return an error. Failing closed here avoids reading or
        // mutating data from a poisoned registry at the cost of leaking entries
        // until process exit.
        return;
    };
    for key in keys {
        let should_remove = if let Some(entry) = guard.get_mut(key) {
            entry.scoped_refs = entry.scoped_refs.saturating_sub(1);
            entry.scoped_refs == 0
        } else {
            false
        };
        if should_remove {
            guard.remove(key);
        }
    }
}

/// Resolve a [`DimExpr`] to a concrete `usize`.
#[doc(hidden)]
pub fn resolve_dim(dim: &DimExpr) -> Result<usize, DimExprEvalError> {
    dim.eval(&[])
}

/// Resolve matrix dimensions and record their ordering as a guard.
#[doc(hidden)]
pub fn resolve_and_guard(
    m: &DimExpr,
    n: &DimExpr,
    ctx: &mut ShapeGuardContext,
) -> Result<(usize, usize), DimExprEvalError> {
    let m_size = resolve_dim(m)?;
    let n_size = resolve_dim(n)?;
    ctx.guards.push(ShapeGuard {
        dim_a: m_size,
        dim_b: n_size,
        ordering: m_size.cmp(&n_size),
    });
    Ok((m_size, n_size))
}
