use chainrules::Tape;

use crate::core::DynTensor;

/// Public reverse-mode tape wrapper for [`DynAdTensor`](crate::DynAdTensor).
///
/// `DynTape` hides the internal `Tape<DynTensor>` payload type from normal
/// dyadtensor users. Reverse-mode leaves are created through
/// [`DynAdTensor::new_reverse_leaf`](crate::DynAdTensor::new_reverse_leaf).
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{DynAdTensor, DynTape};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tape = DynTape::new();
/// let x = DynAdTensor::new_reverse_leaf(
///     Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
///     &tape,
/// )
/// .unwrap();
/// assert_eq!(x.tape_id(), Some(tape.id() as u64));
/// ```
#[derive(Clone, Default)]
pub struct DynTape(Tape<DynTensor>);

impl DynTape {
    /// Creates a new empty reverse-mode tape.
    pub fn new() -> Self {
        Self(Tape::new())
    }

    /// Returns a stable process-local identifier for this tape.
    pub fn id(&self) -> usize {
        self.0.id()
    }

    /// Returns `true` when `self` and `other` refer to the same tape.
    pub fn same_tape(&self, other: &Self) -> bool {
        self.0.same_tape(&other.0)
    }

    pub(crate) fn as_inner(&self) -> &Tape<DynTensor> {
        &self.0
    }
}
