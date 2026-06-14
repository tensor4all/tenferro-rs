use sha2::{Digest, Sha256};

/// StableHLO MLIR module text plus a deterministic fingerprint.
///
/// # Examples
///
/// ```
/// use tenferro_xla::StableHloModule;
///
/// let module = StableHloModule::new("module {}".to_string());
/// assert_eq!(module.as_str(), "module {}");
/// assert_eq!(module.fingerprint().as_bytes().len(), 32);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StableHloModule {
    text: String,
    fingerprint: StableHloModuleFingerprint,
}

impl StableHloModule {
    /// Create a module wrapper and fingerprint its text.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::StableHloModule;
    ///
    /// let module = StableHloModule::new("module {}".to_string());
    /// assert!(module.fingerprint().to_hex().len() == 64);
    /// ```
    pub fn new(text: String) -> Self {
        let fingerprint = StableHloModuleFingerprint::from_text(&text);
        Self { text, fingerprint }
    }

    /// Borrow the StableHLO MLIR text.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::StableHloModule;
    ///
    /// let module = StableHloModule::new("module {}".to_string());
    /// assert!(module.as_str().starts_with("module"));
    /// ```
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Return the module fingerprint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::StableHloModule;
    ///
    /// let module = StableHloModule::new("module {}".to_string());
    /// let _fingerprint = module.fingerprint();
    /// ```
    pub fn fingerprint(&self) -> StableHloModuleFingerprint {
        self.fingerprint
    }
}

/// SHA-256 fingerprint of StableHLO module text.
///
/// # Examples
///
/// ```
/// use tenferro_xla::StableHloModuleFingerprint;
///
/// let fingerprint = StableHloModuleFingerprint::from_text("module {}");
/// assert_eq!(fingerprint.as_bytes().len(), 32);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StableHloModuleFingerprint([u8; 32]);

impl StableHloModuleFingerprint {
    /// Hash StableHLO text into a fingerprint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::StableHloModuleFingerprint;
    ///
    /// let a = StableHloModuleFingerprint::from_text("module {}");
    /// let b = StableHloModuleFingerprint::from_text("module {}");
    /// assert_eq!(a, b);
    /// ```
    pub fn from_text(text: &str) -> Self {
        let digest = Sha256::digest(text.as_bytes());
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        Self(bytes)
    }

    /// Borrow the raw fingerprint bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::StableHloModuleFingerprint;
    ///
    /// let fingerprint = StableHloModuleFingerprint::from_text("module {}");
    /// assert_eq!(fingerprint.as_bytes().len(), 32);
    /// ```
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Return the lowercase hexadecimal fingerprint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::StableHloModuleFingerprint;
    ///
    /// let hex = StableHloModuleFingerprint::from_text("module {}").to_hex();
    /// assert_eq!(hex.len(), 64);
    /// ```
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
