//! Signed public axes (#2552 §3.2): every public axis argument is `isize`,
//! negative values count from the end, and normalization happens here once.

use tenferro_tensor::{Error, Result, ValidationError, INLINE_RANK};
use tinyvec::TinyVec;

/// Normalized axis list.
pub(crate) type Axes = TinyVec<[usize; INLINE_RANK]>;

/// Normalize one signed axis of `op` against `rank`.
pub(crate) fn axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    tenferro_tensor::normalize_axis(axis, rank).map_err(|e| Error::validation(op, e))
}

/// Normalize a signed axis list of `op` against `rank`, rejecting duplicates.
///
/// Every entry is range checked before any repeat is reported, which is the
/// order the eager and traced surfaces normalize in.
pub(crate) fn axes(op: &'static str, axes: &[isize], rank: usize) -> Result<Axes> {
    let normalized = normalize_all(op, axes, rank)?;
    reject_duplicates(op, "axes", &normalized)?;
    Ok(normalized)
}

/// Normalize an optional signed axis list; `None` stays `None`.
pub(crate) fn opt_axes(
    op: &'static str,
    list: Option<&[isize]>,
    rank: usize,
) -> Result<Option<Axes>> {
    list.map(|list| axes(op, list, rank)).transpose()
}

/// Normalize a signed permutation of `op` against `rank`.
///
/// The order matches the eager and traced surfaces: every entry is range
/// checked, then the length is checked, then a repeat is reported as
/// `DuplicateAxis { role: "permutation" }`.
pub(crate) fn perm(op: &'static str, perm: &[isize], rank: usize) -> Result<Axes> {
    let normalized = normalize_all(op, perm, rank)?;
    if normalized.len() != rank {
        return Err(Error::validation(
            op,
            ValidationError::InvalidPermutationLength {
                expected: rank,
                actual: normalized.len(),
            },
        ));
    }
    reject_duplicates(op, "permutation", &normalized)?;
    Ok(normalized)
}

fn normalize_all(op: &'static str, axes: &[isize], rank: usize) -> Result<Axes> {
    axes.iter().map(|&entry| axis(op, entry, rank)).collect()
}

/// Report the first repeated axis with `role`; a linear scan over an
/// already-normalized list keeps this allocation-free.
fn reject_duplicates(op: &'static str, role: &'static str, axes: &[usize]) -> Result<()> {
    for (position, &axis) in axes.iter().enumerate() {
        if axes[..position].contains(&axis) {
            return Err(Error::validation(
                op,
                ValidationError::DuplicateAxis { axis, role },
            ));
        }
    }
    Ok(())
}
