use std::sync::Arc;

use super::identity::validate_identifier;
use super::{EngineId, HardwareClassId, IdentityError, IdentityKind, PlacementConstraintError};

/// Selects the planning and execution reproducibility policy.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::Determinism;
///
/// assert_eq!(Determinism::Fast, Determinism::Fast);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Determinism {
    /// Prefer fast backend choices when multiple valid plans exist.
    Fast,
    /// Prefer reproducible backend choices when multiple valid plans exist.
    Reproducible,
}

/// Validated namespaced storage-class identity.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::StorageClass;
///
/// assert_eq!(
///     StorageClass::new("tenferro.storage.host")?.as_str(),
///     "tenferro.storage.host"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StorageClass(Arc<str>);

impl StorageClass {
    /// Validate a lowercase ASCII namespaced storage-class identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::StorageClass;
    ///
    /// assert!(StorageClass::new("tenferro.storage.host").is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError`] with [`IdentityKind::StorageClass`] when
    /// `value` does not match the lowercase ASCII namespaced identifier grammar.
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError> {
        validate_identifier(value.into(), IdentityKind::StorageClass).map(Self)
    }

    /// Borrow the validated identifier text.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::StorageClass;
    ///
    /// assert_eq!(StorageClass::new("tenferro.storage.host")?.as_str(), "tenferro.storage.host");
    /// # Ok(())
    /// # }
    /// ```
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn runtime_created(value: impl Into<Arc<str>>) -> Self {
        Self(value.into())
    }
}

/// Immutable execution endpoint used to identify a transfer route.
///
/// An endpoint combines the logical runtime engine with the storage class that
/// the engine exposes. Equality, ordering, and hashing are based on both
/// values, so two engines that expose the same storage class remain distinct
/// route endpoints.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{EngineId, StorageClass, TransferEndpoint};
///
/// let endpoint = TransferEndpoint::new(
///     EngineId::new("tenferro.cpu")?,
///     StorageClass::new("tenferro.storage.host")?,
/// );
/// assert_eq!(endpoint.engine_id().as_str(), "tenferro.cpu");
/// assert_eq!(endpoint.storage_class().as_str(), "tenferro.storage.host");
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TransferEndpoint {
    engine_id: EngineId,
    storage_class: StorageClass,
}

impl TransferEndpoint {
    /// Construct an endpoint from an engine and one of its storage classes.
    ///
    /// The runtime validates that the engine is registered and supports the
    /// storage class when a transfer-provider candidate is frozen.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{EngineId, StorageClass, TransferEndpoint};
    ///
    /// let endpoint = TransferEndpoint::new(
    ///     EngineId::new("tenferro.cpu")?,
    ///     StorageClass::new("tenferro.storage.host")?,
    /// );
    /// assert_eq!(endpoint.engine_id().as_str(), "tenferro.cpu");
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(engine_id: EngineId, storage_class: StorageClass) -> Self {
        Self {
            engine_id,
            storage_class,
        }
    }

    /// Return the logical engine that owns this endpoint.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{EngineId, StorageClass, TransferEndpoint};
    ///
    /// let engine = EngineId::new("tenferro.cpu")?;
    /// let endpoint = TransferEndpoint::new(
    ///     engine.clone(),
    ///     StorageClass::new("tenferro.storage.host")?,
    /// );
    /// assert_eq!(endpoint.engine_id(), &engine);
    /// # Ok(())
    /// # }
    /// ```
    pub fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    /// Return the storage class exposed at this endpoint.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{EngineId, StorageClass, TransferEndpoint};
    ///
    /// let storage = StorageClass::new("tenferro.storage.host")?;
    /// let endpoint = TransferEndpoint::new(EngineId::new("tenferro.cpu")?, storage.clone());
    /// assert_eq!(endpoint.storage_class(), &storage);
    /// # Ok(())
    /// # }
    /// ```
    pub fn storage_class(&self) -> &StorageClass {
        &self.storage_class
    }
}

/// Validated namespaced layout-class identity.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::LayoutClass;
///
/// assert_eq!(
///     LayoutClass::new("tenferro.layout.col-major")?.as_str(),
///     "tenferro.layout.col-major"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LayoutClass(Arc<str>);

impl LayoutClass {
    /// Validate a lowercase ASCII namespaced layout-class identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::LayoutClass;
    ///
    /// assert!(LayoutClass::new("tenferro.layout.col-major").is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError`] with [`IdentityKind::LayoutClass`] when
    /// `value` does not match the lowercase ASCII namespaced identifier grammar.
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError> {
        validate_identifier(value.into(), IdentityKind::LayoutClass).map(Self)
    }

    /// Borrow the validated identifier text.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::LayoutClass;
    ///
    /// assert_eq!(LayoutClass::new("tenferro.layout.col-major")?.as_str(), "tenferro.layout.col-major");
    /// # Ok(())
    /// # }
    /// ```
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn runtime_created(value: impl Into<Arc<str>>) -> Self {
        Self(value.into())
    }
}

/// Requested program placement constraints.
///
/// An empty engine list means any engine is allowed.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::ProgramPlacementConstraint;
///
/// assert!(ProgramPlacementConstraint::any().allowed_engines().is_empty());
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProgramPlacementConstraint {
    allowed_engines: Arc<[EngineId]>,
    storage_class: Option<StorageClass>,
}

impl ProgramPlacementConstraint {
    /// Return an unconstrained placement request.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ProgramPlacementConstraint;
    ///
    /// let any = ProgramPlacementConstraint::any();
    /// assert!(any.allowed_engines().is_empty());
    /// assert!(any.storage_class().is_none());
    /// ```
    pub fn any() -> Self {
        Self {
            allowed_engines: Arc::from(Vec::<EngineId>::new()),
            storage_class: None,
        }
    }

    /// Build a placement constraint from engine preferences and storage class.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{EngineId, ProgramPlacementConstraint, StorageClass};
    ///
    /// let engine = EngineId::new("tenferro.cpu")?;
    /// let storage = StorageClass::new("tenferro.storage.host")?;
    /// let constraint = ProgramPlacementConstraint::new(vec![engine.clone()], Some(storage.clone()))?;
    /// assert_eq!(constraint.allowed_engines(), &[engine]);
    /// assert_eq!(constraint.storage_class(), Some(&storage));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`PlacementConstraintError::DuplicateEngine`] when the same
    /// engine appears more than once.
    pub fn new(
        allowed_engines: impl Into<Arc<[EngineId]>>,
        storage_class: Option<StorageClass>,
    ) -> Result<Self, PlacementConstraintError> {
        let allowed_engines = allowed_engines.into();
        for duplicate_index in 0..allowed_engines.len() {
            if let Some(first_index) = (0..duplicate_index).find(|&first_index| {
                allowed_engines[first_index] == allowed_engines[duplicate_index]
            }) {
                return Err(PlacementConstraintError::DuplicateEngine {
                    engine_id: allowed_engines[duplicate_index].clone(),
                    first_index,
                    duplicate_index,
                });
            }
        }
        Ok(Self {
            allowed_engines,
            storage_class,
        })
    }

    /// Return the allowed engines in preference order.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{EngineId, ProgramPlacementConstraint};
    ///
    /// let engine = EngineId::new("tenferro.cpu")?;
    /// let constraint = ProgramPlacementConstraint::new(vec![engine.clone()], None)?;
    /// assert_eq!(constraint.allowed_engines(), &[engine]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn allowed_engines(&self) -> &[EngineId] {
        &self.allowed_engines
    }

    /// Return the requested storage class, if one was supplied.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{ProgramPlacementConstraint, StorageClass};
    ///
    /// let storage = StorageClass::new("tenferro.storage.host")?;
    /// let constraint = ProgramPlacementConstraint::new(Vec::new(), Some(storage.clone()))?;
    /// assert_eq!(constraint.storage_class(), Some(&storage));
    /// # Ok(())
    /// # }
    /// ```
    pub fn storage_class(&self) -> Option<&StorageClass> {
        self.storage_class.as_ref()
    }
}

/// Runtime-resolved program placement.
///
/// A0 exposes no public constructor for this runtime-created value; B0 supplies
/// its runtime creation path.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::ResolvedProgramPlacement;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<ResolvedProgramPlacement>();
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResolvedProgramPlacement {
    engine_id: EngineId,
    storage_class: StorageClass,
}

impl ResolvedProgramPlacement {
    // INVARIANT: A0 module-local tests create resolved placements until B0 owns runtime resolution.
    #[allow(dead_code)]
    pub(crate) fn new(engine_id: EngineId, storage_class: StorageClass) -> Self {
        Self {
            engine_id,
            storage_class,
        }
    }

    /// Return the resolved execution engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::ResolvedProgramPlacement;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<ResolvedProgramPlacement>();
    /// ```
    pub fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    /// Return the resolved storage class.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::ResolvedProgramPlacement;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<ResolvedProgramPlacement>();
    /// ```
    pub fn storage_class(&self) -> &StorageClass {
        &self.storage_class
    }
}

/// Controls how a caller reacts to a preparation already in flight.
///
/// The default is [`CacheInFlightBehavior::Wait`].
///
/// # Examples
///
/// ```
/// use tenferro_runtime::CacheInFlightBehavior;
///
/// assert_eq!(CacheInFlightBehavior::default(), CacheInFlightBehavior::Wait);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CacheInFlightBehavior {
    /// Wait for the in-flight preparation.
    #[default]
    Wait,
    /// Refuse to join the in-flight preparation.
    Refuse,
}

/// Process-wide execution planning policy.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{Determinism, ExecutionPolicy};
///
/// let policy = ExecutionPolicy::new(Determinism::Fast, Some(1024), 7);
/// assert_eq!(policy.hard_workspace_limit_bytes(), Some(1024));
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionPolicy {
    determinism: Determinism,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: u64,
}

impl ExecutionPolicy {
    /// Build an execution policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{Determinism, ExecutionPolicy};
    ///
    /// let policy = ExecutionPolicy::new(Determinism::Reproducible, None, 0);
    /// assert_eq!(policy.determinism(), Determinism::Reproducible);
    /// ```
    pub fn new(
        determinism: Determinism,
        hard_workspace_limit_bytes: Option<usize>,
        planning_seed: u64,
    ) -> Self {
        Self {
            determinism,
            hard_workspace_limit_bytes,
            planning_seed,
        }
    }

    /// Return the determinism mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{Determinism, ExecutionPolicy};
    ///
    /// assert_eq!(
    ///     ExecutionPolicy::new(Determinism::Fast, None, 0).determinism(),
    ///     Determinism::Fast
    /// );
    /// ```
    pub fn determinism(&self) -> Determinism {
        self.determinism
    }

    /// Return the hard workspace limit in bytes, if one is configured.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{Determinism, ExecutionPolicy};
    ///
    /// assert_eq!(
    ///     ExecutionPolicy::new(Determinism::Fast, Some(8), 0).hard_workspace_limit_bytes(),
    ///     Some(8)
    /// );
    /// ```
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize> {
        self.hard_workspace_limit_bytes
    }

    /// Return the planning seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{Determinism, ExecutionPolicy};
    ///
    /// assert_eq!(ExecutionPolicy::new(Determinism::Fast, None, 42).planning_seed(), 42);
    /// ```
    pub fn planning_seed(&self) -> u64 {
        self.planning_seed
    }
}

/// Per-prepare request options before runtime normalization.
///
/// Raw options intentionally do not implement `Hash`; normalized keys are
/// created only after placement and planning resolution.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::PrepareOptions;
///
/// assert!(PrepareOptions::new().planning_seed().is_none());
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrepareOptions {
    placement: Option<ProgramPlacementConstraint>,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: Option<u64>,
    cache_in_flight: CacheInFlightBehavior,
}

impl PrepareOptions {
    /// Return default prepare options.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CacheInFlightBehavior, PrepareOptions};
    ///
    /// let options = PrepareOptions::new();
    /// assert_eq!(options.cache_in_flight(), CacheInFlightBehavior::Wait);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the placement constraint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{PrepareOptions, ProgramPlacementConstraint};
    ///
    /// let placement = ProgramPlacementConstraint::any();
    /// let options = PrepareOptions::new().with_placement(placement.clone());
    /// assert_eq!(options.placement(), Some(&placement));
    /// ```
    pub fn with_placement(mut self, placement: ProgramPlacementConstraint) -> Self {
        self.placement = Some(placement);
        self
    }

    /// Set or reset the per-call workspace override.
    ///
    /// Passing `None` resets the field to inherit the policy value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::PrepareOptions;
    ///
    /// let options = PrepareOptions::new().with_hard_workspace_limit_bytes(Some(0));
    /// assert_eq!(options.hard_workspace_limit_bytes(), Some(0));
    /// assert_eq!(options.with_hard_workspace_limit_bytes(None).hard_workspace_limit_bytes(), None);
    /// ```
    pub fn with_hard_workspace_limit_bytes(mut self, limit: Option<usize>) -> Self {
        self.hard_workspace_limit_bytes = limit;
        self
    }

    /// Set the per-call planning seed override.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::PrepareOptions;
    ///
    /// assert_eq!(PrepareOptions::new().with_planning_seed(3).planning_seed(), Some(3));
    /// ```
    pub fn with_planning_seed(mut self, seed: u64) -> Self {
        self.planning_seed = Some(seed);
        self
    }

    /// Set the in-flight cache behavior for this request.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CacheInFlightBehavior, PrepareOptions};
    ///
    /// let options = PrepareOptions::new().with_cache_in_flight(CacheInFlightBehavior::Refuse);
    /// assert_eq!(options.cache_in_flight(), CacheInFlightBehavior::Refuse);
    /// ```
    pub fn with_cache_in_flight(mut self, behavior: CacheInFlightBehavior) -> Self {
        self.cache_in_flight = behavior;
        self
    }

    /// Return the raw placement constraint, if supplied.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{PrepareOptions, ProgramPlacementConstraint};
    ///
    /// let placement = ProgramPlacementConstraint::any();
    /// let options = PrepareOptions::new().with_placement(placement.clone());
    /// assert_eq!(options.placement(), Some(&placement));
    /// ```
    pub fn placement(&self) -> Option<&ProgramPlacementConstraint> {
        self.placement.as_ref()
    }

    /// Return the raw per-call workspace override.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::PrepareOptions;
    ///
    /// assert_eq!(
    ///     PrepareOptions::new().with_hard_workspace_limit_bytes(Some(16)).hard_workspace_limit_bytes(),
    ///     Some(16)
    /// );
    /// ```
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize> {
        self.hard_workspace_limit_bytes
    }

    /// Return the raw per-call planning seed override.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::PrepareOptions;
    ///
    /// assert_eq!(PrepareOptions::new().with_planning_seed(9).planning_seed(), Some(9));
    /// ```
    pub fn planning_seed(&self) -> Option<u64> {
        self.planning_seed
    }

    /// Return the request's in-flight cache behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CacheInFlightBehavior, PrepareOptions};
    ///
    /// assert_eq!(PrepareOptions::new().cache_in_flight(), CacheInFlightBehavior::Wait);
    /// ```
    pub fn cache_in_flight(&self) -> CacheInFlightBehavior {
        self.cache_in_flight
    }
}

/// Normalized prepare-options cache key.
///
/// A0 exposes no public constructor for this runtime-created value; B0 supplies
/// its runtime creation path.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::PrepareOptionsKey;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<PrepareOptionsKey>();
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PrepareOptionsKey {
    resolved_placement: ResolvedProgramPlacement,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: u64,
}

impl PrepareOptionsKey {
    // INVARIANT: A0 module-local tests create normalized keys until B0 owns prepare-key resolution.
    #[allow(dead_code)]
    pub(crate) fn from_resolved(
        resolved_placement: ResolvedProgramPlacement,
        hard_workspace_limit_bytes: Option<usize>,
        planning_seed: u64,
    ) -> Self {
        Self {
            resolved_placement,
            hard_workspace_limit_bytes,
            planning_seed,
        }
    }

    /// Return the resolved placement carried by this key.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PrepareOptionsKey;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PrepareOptionsKey>();
    /// ```
    pub fn resolved_placement(&self) -> &ResolvedProgramPlacement {
        &self.resolved_placement
    }

    /// Return the resolved hard workspace limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PrepareOptionsKey;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PrepareOptionsKey>();
    /// ```
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize> {
        self.hard_workspace_limit_bytes
    }

    /// Return the resolved planning seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PrepareOptionsKey;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PrepareOptionsKey>();
    /// ```
    pub fn planning_seed(&self) -> u64 {
        self.planning_seed
    }
}

/// Fully resolved planning configuration.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{
///     Determinism, ExecutionPolicy, HardwareClassId, PrepareOptions, ResolvedPlanningConfig,
/// };
///
/// let policy = ExecutionPolicy::new(Determinism::Fast, Some(32), 5);
/// let config = ResolvedPlanningConfig::resolve(
///     &policy,
///     &PrepareOptions::new(),
///     HardwareClassId::new("tenferro.cpu")?,
/// );
/// assert_eq!(config.hard_workspace_limit_bytes(), Some(32));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResolvedPlanningConfig {
    determinism: Determinism,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: u64,
    hardware_class: HardwareClassId,
}

impl ResolvedPlanningConfig {
    /// Resolve raw policy and per-call options into concrete planning config.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{
    ///     Determinism, ExecutionPolicy, HardwareClassId, PrepareOptions, ResolvedPlanningConfig,
    /// };
    ///
    /// let policy = ExecutionPolicy::new(Determinism::Fast, Some(32), 5);
    /// let options = PrepareOptions::new().with_hard_workspace_limit_bytes(Some(0));
    /// let config = ResolvedPlanningConfig::resolve(
    ///     &policy,
    ///     &options,
    ///     HardwareClassId::new("tenferro.cpu")?,
    /// );
    /// assert_eq!(config.hard_workspace_limit_bytes(), Some(0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn resolve(
        policy: &ExecutionPolicy,
        options: &PrepareOptions,
        hardware_class: HardwareClassId,
    ) -> Self {
        Self {
            determinism: policy.determinism(),
            hard_workspace_limit_bytes: options
                .hard_workspace_limit_bytes()
                .or(policy.hard_workspace_limit_bytes()),
            planning_seed: options.planning_seed().unwrap_or(policy.planning_seed()),
            hardware_class,
        }
    }

    /// Return the resolved determinism mode.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{
    ///     Determinism, ExecutionPolicy, HardwareClassId, PrepareOptions, ResolvedPlanningConfig,
    /// };
    ///
    /// let config = ResolvedPlanningConfig::resolve(
    ///     &ExecutionPolicy::new(Determinism::Reproducible, None, 0),
    ///     &PrepareOptions::new(),
    ///     HardwareClassId::new("tenferro.cpu")?,
    /// );
    /// assert_eq!(config.determinism(), Determinism::Reproducible);
    /// # Ok(())
    /// # }
    /// ```
    pub fn determinism(&self) -> Determinism {
        self.determinism
    }

    /// Return the resolved hard workspace limit.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{
    ///     Determinism, ExecutionPolicy, HardwareClassId, PrepareOptions, ResolvedPlanningConfig,
    /// };
    ///
    /// let config = ResolvedPlanningConfig::resolve(
    ///     &ExecutionPolicy::new(Determinism::Fast, Some(7), 0),
    ///     &PrepareOptions::new(),
    ///     HardwareClassId::new("tenferro.cpu")?,
    /// );
    /// assert_eq!(config.hard_workspace_limit_bytes(), Some(7));
    /// # Ok(())
    /// # }
    /// ```
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize> {
        self.hard_workspace_limit_bytes
    }

    /// Return the resolved planning seed.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{
    ///     Determinism, ExecutionPolicy, HardwareClassId, PrepareOptions, ResolvedPlanningConfig,
    /// };
    ///
    /// let config = ResolvedPlanningConfig::resolve(
    ///     &ExecutionPolicy::new(Determinism::Fast, None, 19),
    ///     &PrepareOptions::new(),
    ///     HardwareClassId::new("tenferro.cpu")?,
    /// );
    /// assert_eq!(config.planning_seed(), 19);
    /// # Ok(())
    /// # }
    /// ```
    pub fn planning_seed(&self) -> u64 {
        self.planning_seed
    }

    /// Return the resolved hardware class.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{
    ///     Determinism, ExecutionPolicy, HardwareClassId, PrepareOptions, ResolvedPlanningConfig,
    /// };
    ///
    /// let hardware = HardwareClassId::new("tenferro.cpu")?;
    /// let config = ResolvedPlanningConfig::resolve(
    ///     &ExecutionPolicy::new(Determinism::Fast, None, 0),
    ///     &PrepareOptions::new(),
    ///     hardware.clone(),
    /// );
    /// assert_eq!(config.hardware_class(), &hardware);
    /// # Ok(())
    /// # }
    /// ```
    pub fn hardware_class(&self) -> &HardwareClassId {
        &self.hardware_class
    }
}

/// Normalized planning cache key.
///
/// A0 exposes no public constructor for this runtime-created value; B0 supplies
/// its runtime creation path.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::ResolvedPlanningKey;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<ResolvedPlanningKey>();
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResolvedPlanningKey {
    determinism: Determinism,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: u64,
    hardware_class: HardwareClassId,
}

impl ResolvedPlanningKey {
    // INVARIANT: A0 module-local tests create normalized keys until B0 owns planning-key resolution.
    #[allow(dead_code)]
    pub(crate) fn from_config(config: &ResolvedPlanningConfig) -> Self {
        Self {
            determinism: config.determinism(),
            hard_workspace_limit_bytes: config.hard_workspace_limit_bytes(),
            planning_seed: config.planning_seed(),
            hardware_class: config.hardware_class().clone(),
        }
    }

    /// Return the key's determinism component.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::ResolvedPlanningKey;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<ResolvedPlanningKey>();
    /// ```
    pub fn determinism(&self) -> Determinism {
        self.determinism
    }

    /// Return the key's workspace component.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::ResolvedPlanningKey;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<ResolvedPlanningKey>();
    /// ```
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize> {
        self.hard_workspace_limit_bytes
    }

    /// Return the key's planning seed component.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::ResolvedPlanningKey;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<ResolvedPlanningKey>();
    /// ```
    pub fn planning_seed(&self) -> u64 {
        self.planning_seed
    }

    /// Return the key's hardware-class component.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::ResolvedPlanningKey;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<ResolvedPlanningKey>();
    /// ```
    pub fn hardware_class(&self) -> &HardwareClassId {
        &self.hardware_class
    }
}
