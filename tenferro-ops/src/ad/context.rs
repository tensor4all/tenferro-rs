//! AD context for guard-based shape resolution and value metadata queries.
//!
//! During AD graph construction, linalg rules such as SVD, QR, and LU need
//! concrete matrix dimensions to choose between structurally different
//! subgraphs. `ShapeGuardContext` records those dimension comparisons as guards
//! so cached AD graphs can later be invalidated when the observed shape
//! relationship changes.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use computegraph::fragment::Fragment;
use computegraph::types::{GlobalValKey, ValRef};
use tenferro_tensor::DType;

use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

type MetadataMap = HashMap<GlobalValKey<StdTensorOp>, TensorMeta>;

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
static GLOBAL_METADATA: OnceLock<Mutex<MetadataMap>> = OnceLock::new();

fn global_metadata_registry() -> &'static Mutex<MetadataMap> {
    GLOBAL_METADATA.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Per-value tensor metadata used by AD rules.
///
/// `shape` is expressed in graph-global [`SymDim`] terms rather than op-local
/// [`DimExpr`] references.
///
/// # Examples
///
/// ```ignore
/// use tenferro_ops::{SymDim, TensorMeta};
/// use tenferro_tensor::DType;
///
/// let meta = TensorMeta {
///     dtype: DType::F64,
///     shape: vec![SymDim::from(2usize), SymDim::from(3usize)],
/// };
/// assert_eq!(meta.shape.len(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    pub dtype: DType,
    pub shape: Vec<SymDim>,
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
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ShapeGuardContext {
    guards: Vec<ShapeGuard>,
    metadata: MetadataMap,
    use_global_registry: bool,
    local_keys: Option<Vec<GlobalValKey<StdTensorOp>>>,
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
    /// ```ignore
    /// let ctx = tenferro_ops::ShapeGuardContext::with_global_metadata();
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
    /// ```ignore
    /// let shape = ctx.shape_of(&value);
    /// ```
    pub fn shape_of(&mut self, val: &ValRef<StdTensorOp>) -> &[SymDim] {
        &self.metadata_of(val).shape
    }

    #[doc(hidden)]
    pub fn try_shape_of(&mut self, val: &ValRef<StdTensorOp>) -> Option<&[SymDim]> {
        self.try_metadata_of(val).map(|meta| meta.shape.as_slice())
    }

    /// Return the dtype metadata for a value reference.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let dtype = ctx.dtype_of(&value);
    /// ```
    pub fn dtype_of(&mut self, val: &ValRef<StdTensorOp>) -> DType {
        self.metadata_of(val).dtype
    }

    /// Return the complete metadata record for a value reference.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let meta = ctx.metadata_of(&value);
    /// ```
    pub fn metadata_of(&mut self, val: &ValRef<StdTensorOp>) -> &TensorMeta {
        let key = self.resolve_key(val).clone();
        if !self.metadata.contains_key(&key) && self.use_global_registry {
            if let Some(meta) = lookup_global_metadata(&key) {
                self.metadata.insert(key.clone(), meta);
            }
        }
        self.metadata
            .get(&key)
            .unwrap_or_else(|| panic!("ShapeGuardContext: missing TensorMeta for {:?}", key))
    }

    #[doc(hidden)]
    pub fn try_metadata_of(&mut self, val: &ValRef<StdTensorOp>) -> Option<&TensorMeta> {
        let key = self.resolve_key(val).clone();
        if !self.metadata.contains_key(&key) && self.use_global_registry {
            if let Some(meta) = lookup_global_metadata(&key) {
                self.metadata.insert(key.clone(), meta);
            }
        }
        self.metadata.get(&key)
    }

    #[doc(hidden)]
    pub fn attach_fragment(&mut self, fragment: &Fragment<StdTensorOp>) {
        self.local_keys = Some(
            fragment
                .vals()
                .iter()
                .map(|node| node.key.clone())
                .collect(),
        );
    }

    #[doc(hidden)]
    pub fn insert_metadata(&mut self, key: GlobalValKey<StdTensorOp>, meta: TensorMeta) {
        self.metadata.insert(key, meta);
    }

    #[doc(hidden)]
    pub fn extend_metadata<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (GlobalValKey<StdTensorOp>, TensorMeta)>,
    {
        self.metadata.extend(entries);
    }

    fn resolve_key<'a>(&'a self, val: &'a ValRef<StdTensorOp>) -> &'a GlobalValKey<StdTensorOp> {
        match val {
            ValRef::External(key) => key,
            ValRef::Local(local_id) => self
                .local_keys
                .as_ref()
                .unwrap_or_else(|| {
                    panic!(
                        "ShapeGuardContext: cannot resolve local value {local_id} without an attached fragment"
                    )
                })
                .get(*local_id)
                .unwrap_or_else(|| {
                    panic!("ShapeGuardContext: local value {local_id} is out of bounds")
                }),
        }
    }
}

#[doc(hidden)]
pub fn register_global_metadata(key: GlobalValKey<StdTensorOp>, meta: TensorMeta) {
    let mut guard = global_metadata_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.insert(key, meta);
}

/// Look up a single metadata entry from the global registry.
///
/// Locks the registry briefly for a single `HashMap::get` + clone.
///
/// # Examples
///
/// ```ignore
/// let meta = tenferro_ops::ad::context::lookup_global_metadata(&key);
/// ```
pub fn lookup_global_metadata(key: &GlobalValKey<StdTensorOp>) -> Option<TensorMeta> {
    let guard = global_metadata_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.get(key).cloned()
}

#[doc(hidden)]
pub fn register_global_metadata_batch<I>(entries: I)
where
    I: IntoIterator<Item = (GlobalValKey<StdTensorOp>, TensorMeta)>,
{
    let mut guard = global_metadata_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.extend(entries);
}

/// Resolve a [`DimExpr`] to a concrete `usize`.
///
/// Currently this evaluates the expression without any input shapes, which
/// works for `DimExpr::Const` and expressions composed entirely from constants.
/// `DimExpr::InputDim` references will panic, which is currently a programming
/// invariant enforced by the linalg AD callers.
pub(crate) fn resolve_dim(dim: &DimExpr) -> usize {
    dim.eval(&[])
}

/// Resolve matrix dimensions and record their ordering as a guard.
pub(crate) fn resolve_and_guard(
    m: &DimExpr,
    n: &DimExpr,
    ctx: &mut ShapeGuardContext,
) -> (usize, usize) {
    let m_size = resolve_dim(m);
    let n_size = resolve_dim(n);
    ctx.guards.push(ShapeGuard {
        dim_a: m_size,
        dim_b: n_size,
        ordering: m_size.cmp(&n_size),
    });
    (m_size, n_size)
}
