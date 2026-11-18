//! Internal support surface used by `tenferro-ad`.

pub use crate::checkpoint::RetainedValue;
pub use crate::metadata::tensor_meta_from_tensor;

/// Allocate one fresh traced leaf key.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::ad_support::next_input_key;
///
/// let first = next_input_key();
/// let second = next_input_key();
/// assert_ne!(first, second);
/// ```
pub fn next_input_key() -> tenferro_ops::input_key::TensorInputKey {
    crate::traced::next_input_key()
}

#[cfg(test)]
mod tests;
