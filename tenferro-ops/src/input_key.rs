use chainrules_core::{ADKey, DiffPassId};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum TensorInputKey {
    User { id: u64 },
    Tangent { of: Box<TensorInputKey>, pass: DiffPassId },
}

impl ADKey for TensorInputKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        TensorInputKey::Tangent {
            of: Box::new(self.clone()),
            pass,
        }
    }
}
