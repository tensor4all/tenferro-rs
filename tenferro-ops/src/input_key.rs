use chainrules_core::{ADKey, DiffPassId};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TensorInputKey {
    pub id: u64,
}

impl ADKey for TensorInputKey {
    fn tangent_of(&self, _pass: DiffPassId) -> Self {
        todo!()
    }
}
