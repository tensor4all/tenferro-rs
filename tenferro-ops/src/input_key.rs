#[cfg(feature = "autodiff")]
use chainrules_core::{ADKey, DiffPassId};

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

#[cfg(feature = "autodiff")]
impl ADKey for TensorInputKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        TensorInputKey::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}
