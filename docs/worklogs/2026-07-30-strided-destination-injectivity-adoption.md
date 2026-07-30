# Strided Destination Injectivity Adoption

## Summary

Advanced every workspace `strided-*` dependency from `4c19952f` to merged
strided-rs PR #183, commit `4be1c8a82c0eaf78ee5a9f42ce4b7ac72416e86a`.
The upstream fix rejects non-injective map, zip, and multiply destinations
before any write, including identity fast paths and erased replay.

## Context And Decision

- Upstream issue #181 and PR #183 were reviewed for destination aliasing,
  erased one-shot validation, allocation behavior, and bounded-layout policy.
- This PR only adopts the merged dependency and updates the existing source
  contract to keep all five workspace strided packages on one revision.
- No tenferro source, behavior, public API, or execution policy changes are
  intended. The safety semantics are supplied by the merged upstream package.

## Verification

- Updated the workspace manifest and Cargo lock resolution.
- Ran formatting, the focused dependency/build test, `check-pr-fast.sh`, and
  `scripts/repository-rules-review.py` against `origin/main`.
- Confirmed every resolved strided package in `Cargo.lock` uses the merged
  `strided-rs` revision.

## Remaining Risk

This adoption does not independently change tenferro call paths. Upstream
behavior and allocation tests remain the authority for the new destination
injectivity contract; tenferro CI provides downstream compilation and tests.
