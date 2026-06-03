use std::mem::size_of;

use crate::EinsumSubscripts;

/// Stable family identifier for the standard tenferro einsum extension.
pub const EINSUM_EXTENSION_FAMILY_ID: &str = "tenferro.einsum.v1";

/// Compiler-side static contraction-plan cache name.
pub(crate) const EINSUM_STATIC_PLANS_CACHE: &str = "static_plans";
/// Compiler-side subscript parse cache name.
pub(crate) const EINSUM_PARSE_CACHE: &str = "parse";
/// Executor-side runtime contraction-plan cache name.
pub(crate) const EINSUM_RUNTIME_PLANS_CACHE: &str = "runtime_plans";

/// Parsed einsum notation retained by parse caches.
pub(crate) struct ParsedEinsum {
    /// Canonical parsed subscripts.
    pub(crate) subscripts: EinsumSubscripts,
}

/// Return the retained-byte estimate for canonical subscripts.
#[must_use]
pub(crate) fn einsum_subscripts_retained_bytes(subscripts: &EinsumSubscripts) -> usize {
    vec_of_vec_retained_bytes(&subscripts.inputs) + vec_retained_bytes(&subscripts.output)
}

fn vec_retained_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * size_of::<T>()
}

fn vec_of_vec_retained_bytes<T>(values: &[Vec<T>]) -> usize {
    values.iter().map(vec_retained_bytes).sum()
}
