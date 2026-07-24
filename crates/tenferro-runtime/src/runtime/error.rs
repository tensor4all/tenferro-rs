/// Classifies a malformed runtime identifier without retaining its input.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::EngineId;
///
/// let error = EngineId::new("not namespaced").unwrap_err();
/// assert_eq!(error.kind(), tenferro_runtime::IdentityKind::Engine);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("malformed {kind:?} identifier")]
pub struct IdentityError {
    kind: IdentityKind,
}

impl IdentityError {
    /// Return the kind of identifier that failed validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, IdentityKind};
    ///
    /// assert_eq!(EngineId::new("engine").unwrap_err().kind(), IdentityKind::Engine);
    /// ```
    pub fn kind(&self) -> IdentityKind {
        self.kind
    }

    pub(super) const fn malformed(kind: IdentityKind) -> Self {
        Self { kind }
    }
}

/// Identifies the runtime namespace validated by an opaque identifier.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::IdentityKind;
///
/// assert_eq!(IdentityKind::Engine, IdentityKind::Engine);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum IdentityKind {
    /// An execution engine identifier.
    Engine,
    /// A hardware-class identifier.
    HardwareClass,
    /// A storage-class identifier.
    StorageClass,
    /// A layout-class identifier.
    LayoutClass,
}
