//! Splitting one mutable buffer into the two disjoint slices that a mutable
//! multi-slice view hands to its kernels.
//!
//! The contracts below state, for each returned slice, the exact initial and
//! final contents of its requested inclusive range, so the kernels that write
//! through them write exactly the elements the layout names. They are
//! discharged against `creusot-std`'s specification of `[T]::split_at_mut_checked`
//! (`creusot-std/src/std/slice.rs`), which is Creusot's own model of the
//! standard library; the exclusive-borrow guarantee itself is Rust's.
//!
//! # Standalone compilation
//!
//! `ext/proofs/split-ranges` compiles this file on its own (`#[path = ...]`) so
//! that a prover run does not need the whole `tenferro-tensor` crate, whose
//! derives are outside Creusot's standard-library model. Keep this module
//! self-contained: no `crate::` items, no types with derived equality, and only
//! the `creusot` cfg for the specifications.

/// Split `data` into the inclusive index ranges `before` and `after`, where
/// `before` lies strictly before `after`.
///
/// The checked splits cannot return `None` once the caller's ranges are inside
/// `data`, so the proved postcondition rules that out.
#[cfg_attr(creusot, creusot_std::macros::requires(before_start@ <= before_end@))]
#[cfg_attr(creusot, creusot_std::macros::requires(before_end@ < after_start@))]
#[cfg_attr(
    creusot,
    creusot_std::macros::requires(after_start@ <= after_end@ && after_end@ < data@.len())
)]
#[cfg_attr(
    creusot,
    creusot_std::macros::ensures(match result {
        Some((left, right)) => left@ == data@[before_start@ .. before_end@ + 1]
            && right@ == data@[after_start@ .. after_end@ + 1]
            && (^left)@ == (^data)@[before_start@ .. before_end@ + 1]
            && (^right)@ == (^data)@[after_start@ .. after_end@ + 1],
        None => false,
    })
)]
pub(crate) fn split_ordered_ranges_mut<T>(
    data: &mut [T],
    before_start: usize,
    before_end: usize,
    after_start: usize,
    after_end: usize,
) -> Option<(&mut [T], &mut [T])> {
    let (_, remainder) = data.split_at_mut_checked(before_start)?;
    let (before_slice, remainder) =
        remainder.split_at_mut_checked(before_end - before_start + 1)?;
    let (_, remainder) = remainder.split_at_mut_checked(after_start - before_end - 1)?;
    let (after_slice, _) = remainder.split_at_mut_checked(after_end - after_start + 1)?;
    Some((before_slice, after_slice))
}

/// Split `data` into the two inclusive ranges `first` and `second`.
///
/// Returns `None` when the ranges overlap, which is the only case the caller
/// turns into an unsupported slice pair.
#[cfg_attr(
    creusot,
    creusot_std::macros::requires(first.0@ <= first.1@ && first.1@ < data@.len())
)]
#[cfg_attr(
    creusot,
    creusot_std::macros::requires(second.0@ <= second.1@ && second.1@ < data@.len())
)]
#[cfg_attr(
    creusot,
    creusot_std::macros::ensures(match result {
        Some((first_slice, second_slice)) => first_slice@ == data@[first.0@ .. first.1@ + 1]
            && second_slice@ == data@[second.0@ .. second.1@ + 1]
            && (^first_slice)@ == (^data)@[first.0@ .. first.1@ + 1]
            && (^second_slice)@ == (^data)@[second.0@ .. second.1@ + 1]
            && (first.1@ < second.0@ || second.1@ < first.0@),
        None => true,
    })
)]
pub(crate) fn split_two_mut_ranges<T>(
    data: &mut [T],
    first: (usize, usize),
    second: (usize, usize),
) -> Option<(&mut [T], &mut [T])> {
    if first.1 < second.0 {
        return split_ordered_ranges_mut(data, first.0, first.1, second.0, second.1);
    }
    if second.1 < first.0 {
        return split_ordered_ranges_mut(data, second.0, second.1, first.0, first.1)
            .map(|(second_slice, first_slice)| (first_slice, second_slice));
    }
    None
}
