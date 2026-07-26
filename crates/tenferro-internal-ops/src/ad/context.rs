//! AD context for guard-based shape resolution and value metadata queries.
//!
//! During AD graph construction, linalg rules such as SVD, QR, and LU need
//! concrete matrix dimensions to choose between structurally different
//! subgraphs. `ShapeGuardContext` records those dimension comparisons as guards
//! so cached AD graphs can later be invalidated when the observed shape
//! relationship changes.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
#[cfg(feature = "autodiff")]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use computegraph::graph::Graph;
use computegraph::types::{ValueKey, ValueRef};
use tenferro_tensor::DType;

#[cfg(feature = "autodiff")]
use crate::ad::{ADRuleError, ADRuleKind, ExtensionAdDispatcher};
use crate::dim_expr::{DimExpr, DimExprEvalError};
use crate::shape_extent::ShapeExtent;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

type MetadataMap = HashMap<ValueKey<StdTensorOp>, TensorMeta>;

type GlobalMetadataMap = HashMap<ValueKey<StdTensorOp>, GlobalMetadataEntry>;

#[derive(Clone, Debug)]
struct GlobalMetadataEntry {
    stack: Vec<GlobalMetadataRegistration>,
}

#[derive(Clone, Debug)]
struct GlobalMetadataRegistration {
    token: u64,
    meta: TensorMeta,
}

#[derive(Clone, Debug)]
struct ScopedGlobalMetadataRegistration {
    key: ValueKey<StdTensorOp>,
    token: u64,
}

/// Error returned when the process-global AD metadata registry is unavailable.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MetadataRegistryError {
    /// A previous panic poisoned the global metadata mutex.
    #[error("AD global metadata registry lock poisoned")]
    LockPoisoned,
}

/// Error returned when shape-guard metadata cannot be resolved.
///
/// # Examples
///
/// ```
/// use tenferro_ops::ShapeGuardError;
///
/// let error = ShapeGuardError::LocalWithoutAttachedGraph { local_id: 0 };
/// assert!(error.to_string().contains("attached graph"));
/// ```
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
///
/// The error side preserves a [`ShapeGuardFailure`] wrapper so an AD callback
/// can retain the original [`ShapeGuardError`] even when a foreign callback
/// protocol accepts only a rendered message.
///
/// # Examples
///
/// ```
/// use tenferro_ops::ShapeGuardResult;
///
/// let result: ShapeGuardResult<()> = Ok(());
/// assert!(result.is_ok());
/// ```
pub type ShapeGuardResult<T> = Result<T, ShapeGuardFailure>;

#[cfg(feature = "autodiff")]
impl From<ShapeGuardFailure> for ADRuleError {
    fn from(err: ShapeGuardFailure) -> Self {
        err.record_for_ad_boundary();
        ADRuleError::invalid_input(
            "tenferro.shape_guard",
            ADRuleKind::Jvp,
            err.typed_source().to_string(),
        )
    }
}

/// Error returned by a shape-guard metadata query.
///
/// The public [`ShapeGuardError`] remains the typed source. The private side
/// channel is shared with the owning [`ShapeGuardContext`] so an external
/// message-only AD callback can report the same typed source at the runtime
/// boundary without changing the callback protocol.
///
/// # Examples
///
/// ```
/// use computegraph::types::{ValueKey, ValueRef};
/// use tenferro_ops::input_key::TensorInputKey;
/// use tenferro_ops::std_tensor_op::StdTensorOp;
/// use tenferro_ops::{ShapeGuardContext, ShapeGuardError};
///
/// let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 8 });
/// let value = ValueRef::External(key);
/// let mut ctx = ShapeGuardContext::default();
/// let failure = ctx.shape_of(&value).unwrap_err();
/// assert!(matches!(
///     failure.typed_source(),
///     ShapeGuardError::MissingMetadata { .. }
/// ));
/// ```
#[derive(Clone, Debug)]
pub struct ShapeGuardFailure {
    source: ShapeGuardError,
    #[cfg(feature = "autodiff")]
    deferred: Arc<Mutex<Option<ShapeGuardError>>>,
}

impl ShapeGuardFailure {
    #[cfg(feature = "autodiff")]
    fn new(source: ShapeGuardError, deferred: Arc<Mutex<Option<ShapeGuardError>>>) -> Self {
        Self { source, deferred }
    }

    #[cfg(not(feature = "autodiff"))]
    fn new(source: ShapeGuardError) -> Self {
        Self { source }
    }

    /// Return the original typed shape-guard failure.
    pub fn typed_source(&self) -> &ShapeGuardError {
        &self.source
    }

    /// Consume this boundary error and return its original typed failure.
    pub fn into_typed_source(self) -> ShapeGuardError {
        self.source
    }

    #[cfg(feature = "autodiff")]
    fn record_for_ad_boundary(&self) {
        if let Ok(mut deferred) = self.deferred.lock() {
            if deferred.is_none() {
                *deferred = Some(self.source.clone());
            }
        }
    }
}

impl std::fmt::Display for ShapeGuardFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.source.fmt(f)
    }
}

impl std::error::Error for ShapeGuardFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl PartialEq for ShapeGuardFailure {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl Eq for ShapeGuardFailure {}

/// Global metadata registry.
///
/// Stored as a tokenized stack per value key: duplicate scoped registrations
/// shadow older metadata while they are live, and dropping scopes in any order
/// removes only the matching token. `ShapeGuardContext::metadata_of` reaches into
/// the registry lazily via [`lookup_global_metadata`] and caches the result into
/// the context's local map.
///
/// Earlier designs either cloned the whole map up-front into each AD
/// `ShapeGuardContext` or kept the map in an `Arc` and cloned on every write.
/// Both variants were quadratic across the monotonically growing registry and
/// dominated oracle_replay runtime.
static GLOBAL_METADATA: OnceLock<Mutex<GlobalMetadataMap>> = OnceLock::new();
static NEXT_GLOBAL_METADATA_TOKEN: AtomicU64 = AtomicU64::new(0);

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
    registrations: Vec<ScopedGlobalMetadataRegistration>,
}

impl Drop for GlobalMetadataScope {
    fn drop(&mut self) {
        release_scoped_global_metadata(&self.registrations);
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
    shape_sources: HashMap<u64, ValueKey<StdTensorOp>>,
    use_global_registry: bool,
    local_keys: Option<Vec<ValueKey<StdTensorOp>>>,
    #[cfg(feature = "autodiff")]
    deferred_shape_error: Arc<Mutex<Option<ShapeGuardError>>>,
    #[cfg(feature = "autodiff")]
    extension_ad_dispatcher: Option<Arc<dyn ExtensionAdDispatcher>>,
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

    #[doc(hidden)]
    pub fn insert_shape_source(&mut self, tensor_id: u64, key: ValueKey<StdTensorOp>) {
        self.shape_sources.entry(tensor_id).or_insert(key);
    }

    #[doc(hidden)]
    pub fn shape_source(&self, tensor_id: u64) -> Option<&ValueKey<StdTensorOp>> {
        self.shape_sources.get(&tensor_id)
    }

    #[doc(hidden)]
    #[cfg(feature = "autodiff")]
    pub fn with_extension_ad_dispatcher(
        mut self,
        dispatcher: Arc<dyn ExtensionAdDispatcher>,
    ) -> Self {
        self.extension_ad_dispatcher = Some(dispatcher);
        self
    }

    #[doc(hidden)]
    #[cfg(feature = "autodiff")]
    pub(crate) fn extension_ad_dispatcher(&self) -> Option<Arc<dyn ExtensionAdDispatcher>> {
        self.extension_ad_dispatcher.as_ref().map(Arc::clone)
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

    /// Take the first typed shape-guard failure recorded while crossing an AD
    /// callback boundary.
    ///
    /// AD callbacks expose only a message-bearing error. AD
    /// frontends call this after the callback returns and attach the typed
    /// value to their public runtime error.
    #[doc(hidden)]
    #[cfg(feature = "autodiff")]
    pub fn take_deferred_shape_error(&mut self) -> Option<ShapeGuardError> {
        self.deferred_shape_error
            .lock()
            .ok()
            .and_then(|mut error| error.take())
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
    ///
    /// # Errors
    ///
    /// Returns [`ShapeGuardError`] when the value cannot be resolved, metadata
    /// is missing, or the metadata does not describe an exact shape.
    pub fn shape_of(&mut self, val: &ValueRef<StdTensorOp>) -> ShapeGuardResult<Vec<SymDim>> {
        let key = self.resolve_key(val)?.clone();
        self.ensure_metadata_loaded(&key);
        let meta = self.metadata.get(&key).ok_or_else(|| {
            self.shape_guard_failure(ShapeGuardError::MissingMetadata { key: key.clone() })
        })?;
        meta.exact_shape()
            .ok_or_else(|| self.shape_guard_failure(ShapeGuardError::NonExactShape { key }))
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
    ///
    /// # Errors
    ///
    /// Returns [`ShapeGuardError`] when the value cannot be resolved or its
    /// metadata is unavailable.
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
    ///
    /// # Errors
    ///
    /// Returns [`ShapeGuardError`] when the value cannot be resolved or its
    /// metadata is unavailable.
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
    ///
    /// # Errors
    ///
    /// Returns [`ShapeGuardError`] when the value cannot be resolved or its
    /// metadata is unavailable.
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
    ///
    /// # Errors
    ///
    /// Returns [`ShapeGuardError`] when the value cannot be resolved or its
    /// metadata is unavailable.
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
    ///
    /// # Errors
    ///
    /// Returns [`ShapeGuardError`] when the value cannot be resolved or its
    /// metadata is unavailable.
    pub fn metadata_of(&mut self, val: &ValueRef<StdTensorOp>) -> ShapeGuardResult<&TensorMeta> {
        let key = self.resolve_key(val)?.clone();
        self.ensure_metadata_loaded(&key);
        self.metadata
            .get(&key)
            .ok_or_else(|| self.shape_guard_failure(ShapeGuardError::MissingMetadata { key }))
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
            ValueRef::Local(local_id) if self.local_keys.is_none() => Err(self
                .shape_guard_failure(ShapeGuardError::LocalWithoutAttachedGraph {
                    local_id: *local_id,
                })),
            ValueRef::Local(local_id) => self
                .local_keys
                .as_ref()
                .and_then(|keys| keys.get(*local_id))
                .ok_or_else(|| {
                    self.shape_guard_failure(ShapeGuardError::LocalOutOfBounds {
                        local_id: *local_id,
                    })
                }),
        }
    }

    #[cfg(feature = "autodiff")]
    fn shape_guard_failure(&self, source: ShapeGuardError) -> ShapeGuardFailure {
        ShapeGuardFailure::new(source, Arc::clone(&self.deferred_shape_error))
    }

    #[cfg(not(feature = "autodiff"))]
    fn shape_guard_failure(&self, source: ShapeGuardError) -> ShapeGuardFailure {
        ShapeGuardFailure::new(source)
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
///
/// # Errors
///
/// Returns [`MetadataRegistryError::LockPoisoned`] when the global metadata
/// registry lock is poisoned.
pub fn lookup_global_metadata(
    key: &ValueKey<StdTensorOp>,
) -> Result<Option<TensorMeta>, MetadataRegistryError> {
    let guard = global_metadata_registry()
        .lock()
        .map_err(|_| MetadataRegistryError::LockPoisoned)?;
    Ok(guard
        .get(key)
        .and_then(|entry| entry.stack.last())
        .map(|registration| registration.meta.clone()))
}

#[doc(hidden)]
///
/// # Errors
///
/// Returns [`MetadataRegistryError::LockPoisoned`] when the global metadata
/// registry lock is poisoned.
pub fn register_scoped_global_metadata_batch<I>(
    entries: I,
) -> Result<GlobalMetadataScope, MetadataRegistryError>
where
    I: IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
{
    let mut guard = global_metadata_registry()
        .lock()
        .map_err(|_| MetadataRegistryError::LockPoisoned)?;
    let mut registrations = Vec::new();
    for (key, meta) in entries {
        let token = NEXT_GLOBAL_METADATA_TOKEN.fetch_add(1, AtomicOrdering::Relaxed);
        let entry = guard
            .entry(key.clone())
            .or_insert_with(|| GlobalMetadataEntry { stack: Vec::new() });
        entry.stack.push(GlobalMetadataRegistration { token, meta });
        registrations.push(ScopedGlobalMetadataRegistration { key, token });
    }
    Ok(GlobalMetadataScope { registrations })
}

fn release_scoped_global_metadata(registrations: &[ScopedGlobalMetadataRegistration]) {
    let Ok(mut guard) = global_metadata_registry().lock() else {
        // Drop cannot return an error. Failing closed here avoids reading or
        // mutating data from a poisoned registry at the cost of leaking entries
        // until process exit.
        return;
    };
    for registration in registrations {
        let should_remove = if let Some(entry) = guard.get_mut(&registration.key) {
            if let Some(position) = entry
                .stack
                .iter()
                .rposition(|candidate| candidate.token == registration.token)
            {
                entry.stack.remove(position);
            }
            entry.stack.is_empty()
        } else {
            false
        };
        if should_remove {
            guard.remove(&registration.key);
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
