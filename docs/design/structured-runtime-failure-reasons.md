# Structured Runtime Failure Reasons

## Status

Accepted design for issue #1712. Implementation requires an independent design-review verdict before code changes.

## Goal

Downstream callers inspect common runtime preparation/execution failures through one stable borrowed reason view, without formatted-message parsing or source-chain downcasts. Existing `Error` variants, `kind`, `phase`, display text, and `StdError::source` remain intact.

## Public API

Add to `tenferro-runtime`:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum RuntimeFailureReasonRef<'a> {
    MissingExtension { family: &'a str },
    NoInputIngress { input_index: usize, placement: &'a Placement },
    UnsupportedOperation { operation: &'a str },
    Other,
}

impl Error {
    pub fn reason(&self) -> RuntimeFailureReasonRef<'_>;
}
```

The enum is borrowed and non-exhaustive. `Other` covers every unknown/currently unclassified failure, allowing future reason variants without changing owned errors.

## Source classification

Add a precise owned `PrepareError::MissingExtension { family_id: &'static str }` variant at the existing compiled-preparation point where `missing_extension_family` is already known. This replaces only the ambiguous `PrepareError::Unsupported { UnsupportedReason::Operation }` emitted at that point; it does not replace top-level runtime errors or alter their source chain. Genuine provider operation rejection remains `PrepareError::Unsupported` and maps to `UnsupportedOperation`.

`Error::reason`:

1. For `Error::WithSuppressed`, classify the primary recursively and never let the suppressed secondary override it.
2. Classify direct top-level unsupported/extension/tensor-runtime variants when they carry a stable operation/family.
3. Walk the standard source chain once and downcast to `PrepareError`:
   - `MissingExtension` → `MissingExtension`;
   - `NoInputIngress` → borrowed input index and placement;
   - `Unsupported { reason: UnsupportedReason::Operation }` → `UnsupportedOperation`.
4. Return `Other` for all remaining errors and unknown future variants.

Traversal is deterministic, bounded by the acyclic standard error source chain, allocation-free, and performs no string parsing. The original typed source remains available.

## Public examples and migration

Rustdoc demonstrates exact matching with a constructed typed error and `Other`. Runtime integration examples/tests replace variant-set/message searches and nested `source().source().downcast_ref` calls with `reason()`.

No application must exhaustively match the non-exhaustive enum; examples include a wildcard/`Other` branch.

## Tests

- A compiled extension with no installed module reports exact `MissingExtension { family }`.
- A no-input-ingress failure reports exact input index and borrowed placement with no downcast.
- A genuine unsupported operation reports `UnsupportedOperation` and is not mislabeled missing extension.
- `kind`, `phase`, display, and every original `source()` link are unchanged.
- `WithSuppressed` returns the primary reason even when the suppressed error has another actionable reason.
- Nested runtime wrappers classify through one or more source layers.
- Unknown validation/internal/future-like failures return `Other`.

## Non-goals

- No replacement giant owned error enum.
- No removal or flattening of existing variants/source chains.
- No required display-text changes.
- No string parsing or duplicated owned source payload.
- No speculative reason variants without a current reachable failure.

## Verification

Run runtime error/preparation/execution unit and integration tests, doctests, clippy, modified-file coverage review, and combined PR gates.
