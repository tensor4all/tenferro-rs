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
/// use tenferro_runtime::EngineId;
///
/// assert_eq!(EngineId::new("tenferro.cpu.v1").unwrap().as_str(), "tenferro.cpu.v1");
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
    /// use tenferro_runtime::EngineId;
    ///
    /// assert_eq!(EngineId::new("tenferro.cpu").unwrap().as_str(), "tenferro.cpu");
    /// ```
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Validated namespaced identity of a hardware class.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::HardwareClassId;
///
/// assert_eq!(HardwareClassId::new("tenferro.cpu.host").unwrap().as_str(), "tenferro.cpu.host");
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
    /// use tenferro_runtime::HardwareClassId;
    ///
    /// assert_eq!(HardwareClassId::new("tenferro.cpu").unwrap().as_str(), "tenferro.cpu");
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
    /// Return the identity of a `Send + Sync + 'static` context type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExecutionContextIdentity;
    ///
    /// assert_eq!(ExecutionContextIdentity::of::<u64>(), ExecutionContextIdentity::of::<u64>());
    /// ```
    pub fn of<T: Send + Sync + 'static>() -> Self {
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
