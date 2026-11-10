//! Prototype 5: family error causes through the three layers (#2552 rule 5).
//!
//! The grammar keeps the family-specific error (for example
//! `tenferro_linalg::Error`) reachable from the layer's outer error through
//! `std::error::Error::source`. [`find_cause`] is the reader side of that claim.

use std::error::Error as StdError;

/// Walk `err` and its `source()` chain and return the first value of type `E`.
///
/// The walk starts at `err` itself, so a top-level `E` is found too.
pub fn find_cause<'a, E: StdError + 'static>(err: &'a (dyn StdError + 'static)) -> Option<&'a E> {
    let mut cur: Option<&'a (dyn StdError + 'static)> = Some(err);
    while let Some(e) = cur {
        if let Some(found) = e.downcast_ref::<E>() {
            return Some(found);
        }
        cur = e.source();
    }
    None
}

/// Type names along the `source()` chain, outermost first (diagnostic helper).
pub fn chain_type_names(err: &(dyn StdError + 'static)) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Option<&(dyn StdError + 'static)> = Some(err);
    while let Some(e) = cur {
        out.push(format!("{e:?}").chars().take(60).collect::<String>());
        cur = e.source();
    }
    out
}
