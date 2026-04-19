//! AD context for guard-based shape resolution and value metadata queries.
//!
//! During AD graph construction, linalg rules such as SVD, QR, and LU need
//! concrete matrix dimensions to choose between structurally different
//! subgraphs. `ShapeGuardContext` records those dimension comparisons as guards
//! so cached AD graphs can later be invalidated when the observed shape
//! relationship changes.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use computegraph::fragment::Fragment;
use computegraph::types::{GlobalValKey, ValRef};
use tenferro_tensor::DType;

use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

type MetadataMap = HashMap<GlobalValKey<StdTensorOp>, TensorMeta>;

static GLOBAL_METADATA: OnceLock<Mutex<MetadataMap>> = OnceLock::new();

fn global_metadata_registry() -> &'static Mutex<MetadataMap> {
    GLOBAL_METADATA.get_or_init(|| Mutex::new(HashMap::new()))
}

fn global_metadata_snapshot() -> Arc<MetadataMap> {
    Arc::new(
        global_metadata_registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone(),
    )
}

#[doc(hidden)]
pub fn snapshot_global_metadata() -> Arc<MetadataMap> {
    global_metadata_snapshot()
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeGuardContext {
    guards: Vec<ShapeGuard>,
    metadata: MetadataMap,
    global_metadata: Arc<MetadataMap>,
    local_keys: Option<Vec<GlobalValKey<StdTensorOp>>>,
}

impl Default for ShapeGuardContext {
    fn default() -> Self {
        Self {
            guards: Vec::new(),
            metadata: HashMap::new(),
            global_metadata: Arc::new(HashMap::new()),
            local_keys: None,
        }
    }
}

impl ShapeGuardContext {
    /// Create a context backed by the current global metadata snapshot.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let ctx = tenferro_ops::ShapeGuardContext::with_global_metadata();
    /// ```
    pub fn with_global_metadata() -> Self {
        Self {
            global_metadata: global_metadata_snapshot(),
            ..Self::default()
        }
    }

    #[doc(hidden)]
    pub fn refresh_global_metadata(&mut self) {
        self.global_metadata = global_metadata_snapshot();
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
    pub fn shape_of(&self, val: &ValRef<StdTensorOp>) -> &[SymDim] {
        &self.metadata_of(val).shape
    }

    /// Return the dtype metadata for a value reference.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let dtype = ctx.dtype_of(&value);
    /// ```
    pub fn dtype_of(&self, val: &ValRef<StdTensorOp>) -> DType {
        self.metadata_of(val).dtype
    }

    /// Return the complete metadata record for a value reference.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let meta = ctx.metadata_of(&value);
    /// ```
    pub fn metadata_of(&self, val: &ValRef<StdTensorOp>) -> &TensorMeta {
        let key = self.resolve_key(val);
        self.metadata
            .get(key)
            .or_else(|| self.global_metadata.get(key))
            .unwrap_or_else(|| panic!("ShapeGuardContext: missing TensorMeta for {:?}", key))
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
    global_metadata_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key, meta);
}

#[doc(hidden)]
pub fn register_global_metadata_batch<I>(entries: I)
where
    I: IntoIterator<Item = (GlobalValKey<StdTensorOp>, TensorMeta)>,
{
    global_metadata_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .extend(entries);
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
