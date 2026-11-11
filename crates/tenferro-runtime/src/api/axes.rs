//! Signed public axes (#2552 §3.2): every public axis argument is `isize`,
//! negative values count from the end, and normalization happens here once.

use tenferro_tensor::{Error, Result, INLINE_RANK};
use tinyvec::TinyVec;

/// Normalized axis list.
pub(crate) type Axes = TinyVec<[usize; INLINE_RANK]>;

/// Normalize one signed axis of `op` against `rank`.
pub(crate) fn axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    tenferro_tensor::normalize_axis(axis, rank).map_err(|e| Error::validation(op, e))
}

/// Normalize a signed axis list of `op` against `rank`, rejecting duplicates.
pub(crate) fn axes(op: &'static str, axes: &[isize], rank: usize) -> Result<Axes> {
    tenferro_tensor::normalize_axes(axes, rank).map_err(|e| Error::validation(op, e))
}

/// Normalize an optional signed axis list; `None` stays `None`.
pub(crate) fn opt_axes(
    op: &'static str,
    list: Option<&[isize]>,
    rank: usize,
) -> Result<Option<Axes>> {
    list.map(|list| axes(op, list, rank)).transpose()
}
