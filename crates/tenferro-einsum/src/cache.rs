use std::mem::size_of;

use crate::EinsumSubscripts;

/// Stable family identifier for the standard tenferro einsum extension.
pub const EINSUM_EXTENSION_FAMILY_ID: &str = "tenferro.einsum.v1";

/// Compiler-side subscript parse cache name.
pub(crate) const EINSUM_PARSE_CACHE: &str = "parse";
/// Executor-side runtime contraction-plan cache name.
pub(crate) const EINSUM_RUNTIME_PLANS_CACHE: &str = "runtime_plans";
/// EagerTensor expanded standard-op program cache name.
#[cfg(feature = "autodiff")]
pub(crate) const EINSUM_EAGER_EXPANDED_PROGRAMS_CACHE: &str = "eager_expanded_programs";

/// Parsed einsum notation retained by parse caches.
pub(crate) struct ParsedEinsum {
    /// Canonical parsed subscripts.
    pub(crate) subscripts: EinsumSubscripts,
}

/// Return the retained-byte estimate for canonical subscripts.
#[must_use]
pub(crate) fn einsum_subscripts_retained_bytes(subscripts: &EinsumSubscripts) -> usize {
    saturating_sum([
        vec_of_vec_retained_bytes(&subscripts.inputs),
        vec_retained_bytes(&subscripts.output),
    ])
}

pub(crate) fn vec_retained_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity().saturating_mul(size_of::<T>())
}

pub(crate) fn vec_of_vec_retained_bytes<T>(values: &[Vec<T>]) -> usize {
    saturating_sum(values.iter().map(vec_retained_bytes))
}

pub(crate) fn saturating_sum(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().fold(0usize, usize::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::saturating_sum;

    #[test]
    fn retained_byte_sums_saturate() {
        assert_eq!(saturating_sum([usize::MAX, 1]), usize::MAX);
        assert_eq!(saturating_sum([usize::MAX - 4, 2, 8]), usize::MAX);
    }
}
