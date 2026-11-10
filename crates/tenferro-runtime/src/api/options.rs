//! Options types of the core grammar.

/// Options of `add`. There is no setting yet; the type exists so that `add`
/// has an `add_with`/`add_into` form like every operation with a destination.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::AddOptions;
///
/// assert_eq!(AddOptions::default(), AddOptions::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct AddOptions {}
