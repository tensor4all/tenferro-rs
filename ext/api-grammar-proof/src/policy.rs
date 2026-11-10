//! Shared execution-constraint type used by every prototype.

/// What one invocation may do to its input layout (tenferro-rs#2552 ruling).
///
/// `Allow` lets the provider normalize a same-placement, same-dtype input once.
/// `Forbid` asserts that no implicit input-layout materialization happens; the
/// call is refused with a typed error before any side effect otherwise. Neither
/// value authorizes a placement change, a dtype change, or different mathematics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterializationPolicy {
    /// Provider may normalize the input layout once (default).
    #[default]
    Allow,
    /// Refuse any implicit input-layout materialization.
    Forbid,
}
