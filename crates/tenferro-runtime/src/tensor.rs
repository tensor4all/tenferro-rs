//! Axis-list helper shared by the core grammar bodies.

use std::borrow::Cow;

/// Resolve a reduction-family axis argument: `None` selects every axis.
///
/// An explicit axis list stays borrowed; only `None` builds a list.
pub(crate) fn all_axes_if_none(rank: usize, axes: Option<&[usize]>) -> Cow<'_, [usize]> {
    match axes {
        Some(axes) => Cow::Borrowed(axes),
        None => Cow::Owned((0..rank).collect()),
    }
}
