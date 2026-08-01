use std::any::TypeId;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::Arc;

use super::{IdentityError, IdentityKind};

/// Opaque runtime identity, created only inside the runtime in A0.
///
/// B0 supplies the public creation path.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::RuntimeId;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<RuntimeId>();
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeId(NonZeroU64);

impl RuntimeId {
    // INVARIANT: A0 exposes crate-private construction for runtime-owned callers; B0 is its first production owner.
    #[allow(dead_code)]
    pub(crate) const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    // INVARIANT: A0 module-local tests are the only current consumer; B0 runtime ownership will read this opaque value.
    #[allow(dead_code)]
    pub(crate) const fn get(self) -> NonZeroU64 {
        self.0
    }
}

/// Opaque runtime epoch, created only inside the runtime in A0.
///
/// B0 supplies the public creation path.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::RuntimeEpoch;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<RuntimeEpoch>();
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeEpoch(NonZeroU64);

impl RuntimeEpoch {
    // INVARIANT: A0 defines epoch-one construction for the later B0 runtime owner without a public constructor.
    #[allow(dead_code)]
    pub(crate) const fn one() -> Self {
        Self(NonZeroU64::MIN)
    }

    // INVARIANT: A0 module-local tests cover crate-private epoch construction until B0 creates epochs in production.
    #[allow(dead_code)]
    pub(crate) const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    // INVARIANT: A0 module-local tests cover opaque access until B0 owns epoch state transitions.
    #[allow(dead_code)]
    pub(crate) const fn get(self) -> NonZeroU64 {
        self.0
    }

    // INVARIANT: A0 module-local tests cover overflow termination until B0 maps it to a runtime reconfiguration error.
    #[allow(dead_code)]
    pub(crate) fn checked_next(self) -> Option<Self> {
        self.0
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(Self)
    }
}

/// Validated namespaced identity of an execution engine.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::EngineId;
///
/// assert_eq!(EngineId::new("tenferro.cpu.v1")?.as_str(), "tenferro.cpu.v1");
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EngineId(Arc<str>);

impl EngineId {
    /// Validate a lowercase ASCII namespaced engine identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::EngineId;
    ///
    /// assert!(EngineId::new("tenferro.cpu").is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError`] with [`IdentityKind::Engine`] when `value`
    /// does not match the lowercase ASCII namespaced identifier grammar.
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError> {
        validate_identifier(value.into(), IdentityKind::Engine).map(Self)
    }

    /// Borrow the validated identifier text.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::EngineId;
    ///
    /// assert_eq!(EngineId::new("tenferro.cpu")?.as_str(), "tenferro.cpu");
    /// # Ok(())
    /// # }
    /// ```
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Validated namespaced identity of a runtime provider.
///
/// A provider ID identifies the implementation namespace in the runtime
/// control plane. It is deliberately distinct from tensor placement metadata.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::ProviderId;
///
/// assert_eq!(ProviderId::new("tenferro.cpu")?.as_str(), "tenferro.cpu");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderId(Arc<str>);

impl ProviderId {
    /// Validate a lowercase ASCII namespaced provider identifier.
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError`] with [`IdentityKind::Provider`] when `value`
    /// does not match the lowercase ASCII namespaced identifier grammar.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ProviderId;
    ///
    /// assert!(ProviderId::new("tenferro.cuda").is_ok());
    /// ```
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError> {
        validate_identifier(value.into(), IdentityKind::Provider).map(Self)
    }

    /// Borrow the validated provider identifier text.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ProviderId;
    ///
    /// assert_eq!(ProviderId::new("tenferro.cuda")?.as_str(), "tenferro.cuda");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable runtime-control-plane binding of a provider to one physical target.
///
/// The target identity is opaque to the runtime and is canonicalized by the
/// provider. It is not [`tenferro_tensor::DeviceId`] or tensor placement data.
/// Because this value appears in structured diagnostics and debug output, A0.1
/// accepts nonempty ASCII graphic text (no controls, whitespace, or Unicode
/// confusables). Providers that need byte-level or Unicode identities must
/// first canonicalize them into an escaped diagnostic-safe ASCII form.
/// Equality, ordering, and hashing include both the provider and target.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{ProviderDeviceIdentity, ProviderId};
///
/// let identity = ProviderDeviceIdentity::new(ProviderId::new("tenferro.cuda")?, "device-0")?;
/// assert_eq!(identity.provider_id().as_str(), "tenferro.cuda");
/// assert_eq!(identity.target_identity(), "device-0");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderDeviceIdentity {
    provider_id: ProviderId,
    target_identity: Arc<str>,
}

impl ProviderDeviceIdentity {
    /// Construct a provider binding from a validated provider and opaque target.
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError`] with [`IdentityKind::ProviderTarget`] when the
    /// provider-canonical target identity is empty or is not ASCII graphic
    /// text. Target text is deliberately diagnostic-safe; opaque
    /// provider-specific byte or Unicode identities must be encoded by the
    /// provider before construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{ProviderDeviceIdentity, ProviderId};
    ///
    /// let identity = ProviderDeviceIdentity::new(
    ///     ProviderId::new("tenferro.cpu")?,
    ///     "host-0",
    /// )?;
    /// assert_eq!(identity.target_identity(), "host-0");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        provider_id: ProviderId,
        target_identity: impl Into<Arc<str>>,
    ) -> Result<Self, IdentityError> {
        let target_identity = target_identity.into();
        if target_identity.is_empty()
            || !target_identity
                .chars()
                .all(|character| character.is_ascii_graphic())
        {
            return Err(IdentityError::malformed(IdentityKind::ProviderTarget));
        }
        Ok(Self {
            provider_id,
            target_identity,
        })
    }

    /// Return the validated provider namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{ProviderDeviceIdentity, ProviderId};
    ///
    /// let provider = ProviderId::new("tenferro.cpu")?;
    /// let identity = ProviderDeviceIdentity::new(provider.clone(), "host-0")?;
    /// assert_eq!(identity.provider_id(), &provider);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    /// Return the provider-canonical opaque target identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{ProviderDeviceIdentity, ProviderId};
    ///
    /// let identity = ProviderDeviceIdentity::new(ProviderId::new("tenferro.cpu")?, "host-0")?;
    /// assert_eq!(identity.target_identity(), "host-0");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn target_identity(&self) -> &str {
        &self.target_identity
    }
}

/// Validated namespaced identity of a hardware class.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::HardwareClassId;
///
/// assert_eq!(HardwareClassId::new("tenferro.cpu.host")?.as_str(), "tenferro.cpu.host");
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HardwareClassId(Arc<str>);

impl HardwareClassId {
    /// Validate a lowercase ASCII namespaced hardware-class identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::HardwareClassId;
    ///
    /// assert!(HardwareClassId::new("tenferro.cpu").is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError`] with [`IdentityKind::HardwareClass`] when
    /// `value` does not match the lowercase ASCII namespaced identifier grammar.
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError> {
        validate_identifier(value.into(), IdentityKind::HardwareClass).map(Self)
    }

    /// Borrow the validated identifier text.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::HardwareClassId;
    ///
    /// assert_eq!(HardwareClassId::new("tenferro.cpu")?.as_str(), "tenferro.cpu");
    /// # Ok(())
    /// # }
    /// ```
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque runtime-local registration identity, created only inside the runtime in A0.
///
/// B0 supplies the public creation path. Its debug output exposes the useful
/// ordinal but never the private issuer.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::RegistrationIdentity;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<RegistrationIdentity>();
/// ```
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RegistrationIdentity {
    issuer: NonZeroU64,
    ordinal: NonZeroU64,
}

impl RegistrationIdentity {
    // INVARIANT: A0 module-local tests require crate-private construction; B0 owns all production registration issuance.
    #[allow(dead_code)]
    pub(crate) const fn new(issuer: NonZeroU64, ordinal: NonZeroU64) -> Self {
        Self { issuer, ordinal }
    }

    /// Return this registration's runtime-local ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::RegistrationIdentity;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<RegistrationIdentity>();
    /// ```
    pub fn ordinal(self) -> NonZeroU64 {
        self.ordinal
    }
}

impl fmt::Debug for RegistrationIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RegistrationIdentity")
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

/// Type identity of an execution context without retaining a context value.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::ExecutionContextIdentity;
///
/// assert_eq!(ExecutionContextIdentity::of::<u64>().type_name(), std::any::type_name::<u64>());
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionContextIdentity {
    type_id: TypeId,
    type_name: &'static str,
}

impl ExecutionContextIdentity {
    /// Return the identity of a `'static` context type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExecutionContextIdentity;
    ///
    /// assert_eq!(ExecutionContextIdentity::of::<u64>(), ExecutionContextIdentity::of::<u64>());
    /// ```
    pub fn of<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
        }
    }

    /// Return the diagnostic name of the context type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExecutionContextIdentity;
    ///
    /// assert_eq!(ExecutionContextIdentity::of::<u64>().type_name(), std::any::type_name::<u64>());
    /// ```
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
}

pub(super) fn validate_identifier(
    value: Arc<str>,
    kind: IdentityKind,
) -> Result<Arc<str>, IdentityError> {
    let valid = value.is_ascii()
        && value.split('.').count() >= 2
        && value.split('.').all(valid_identifier_component);
    valid
        .then_some(value)
        .ok_or_else(|| IdentityError::malformed(kind))
}

fn valid_identifier_component(component: &str) -> bool {
    let bytes = component.as_bytes();
    match bytes {
        [single] => single.is_ascii_lowercase() || single.is_ascii_digit(),
        [first, middle @ .., last] => {
            (first.is_ascii_lowercase() || first.is_ascii_digit())
                && (last.is_ascii_lowercase() || last.is_ascii_digit())
                && middle.iter().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'-' | b'_')
                })
        }
        [] => false,
    }
}
