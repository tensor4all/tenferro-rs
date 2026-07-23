use std::fmt;
use std::sync::Arc;

use tenferro_tensor::Tensor;

use super::value::ProgramBuilderNonce;
use super::{BindingKey, ProgramValue};

pub(crate) struct PendingBinding {
    pub(crate) key: BindingKey,
    pub(crate) input: ProgramValue,
    pub(crate) tensor: Arc<Tensor>,
}

struct ProgramBinding {
    key: BindingKey,
    tensor: Arc<Tensor>,
}

/// Immutable tensor defaults and large constants kept outside semantic structure.
#[derive(Clone)]
pub struct ProgramBindings {
    owner: ProgramBuilderNonce,
    entries: Arc<[ProgramBinding]>,
}

impl ProgramBindings {
    pub(crate) fn freeze(owner: ProgramBuilderNonce, pending: Vec<PendingBinding>) -> Self {
        let mut entries: Vec<_> = pending
            .into_iter()
            .map(|binding| ProgramBinding {
                key: binding.key,
                tensor: binding.tensor,
            })
            .collect();
        entries.sort_unstable_by_key(|entry| entry.key.slot);
        Self {
            owner,
            entries: entries.into(),
        }
    }

    /// Return the number of tensor bindings.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether no tensor bindings are present.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow a tensor by opaque binding key.
    pub fn get(&self, key: BindingKey) -> Option<&Tensor> {
        if key.owner != self.owner {
            return None;
        }
        self.entries
            .binary_search_by_key(&key.slot, |entry| entry.key.slot)
            .ok()
            .and_then(|index| self.entries.get(index))
            .filter(|entry| entry.key == key)
            .map(|entry| entry.tensor.as_ref())
    }

    /// Iterate over ordered binding keys and borrowed tensors.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (BindingKey, &Tensor)> + '_ {
        self.entries
            .iter()
            .map(|entry| (entry.key, entry.tensor.as_ref()))
    }
}

impl fmt::Debug for ProgramBindings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProgramBindings")
            .field("len", &self.entries.len())
            .finish()
    }
}
