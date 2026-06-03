#[cfg(feature = "autodiff")]
use tidu::{ADKey, DiffPassId};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum TensorInputKey {
    User {
        id: u64,
    },
    #[cfg(feature = "autodiff")]
    Tangent {
        of: Box<TensorInputKey>,
        pass: DiffPassId,
    },
}

impl TensorInputKey {
    /// Returns `true` when this key names an AD tangent input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::input_key::TensorInputKey;
    ///
    /// let key = TensorInputKey::User { id: 0 };
    /// assert!(!key.is_tangent());
    /// ```
    pub fn is_tangent(&self) -> bool {
        match self {
            TensorInputKey::User { .. } => false,
            #[cfg(feature = "autodiff")]
            TensorInputKey::Tangent { .. } => true,
        }
    }

    /// Returns the user input key that owns this input's concrete primal data.
    ///
    /// For non-AD keys this returns `self`; for tangent keys it recursively
    /// follows the `of` chain to the original user input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::input_key::TensorInputKey;
    ///
    /// let key = TensorInputKey::User { id: 0 };
    /// assert_eq!(key.primal_root(), &key);
    /// ```
    pub fn primal_root(&self) -> &Self {
        match self {
            TensorInputKey::User { .. } => self,
            #[cfg(feature = "autodiff")]
            TensorInputKey::Tangent { of, .. } => of.primal_root(),
        }
    }
}

#[cfg(feature = "autodiff")]
impl ADKey for TensorInputKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        TensorInputKey::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}
