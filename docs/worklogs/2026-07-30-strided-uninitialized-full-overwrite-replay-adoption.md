# Strided Uninitialized Full-Overwrite Replay Adoption

## Summary

Advanced every workspace `strided-*` dependency from `4be1c8a8` to merged
tensor4all/strided-rs#184, commit
`6885f52edb5fa2348fea47413bc43561e436ef63`.

## Context And Decision

- This adopts the upstream uninitialized full-overwrite replay behavior from
  strided-rs#184, following the prior merged strided adoption pattern.
- The adoption depends on the merged upstream strided change and does not
  change tenferro implementation or call paths.
- No tenferro call path changes are included yet; follow-up work remains
  separate from this pin update and does not start #1516A.

## Verification

- Updated the workspace manifest and exact-revision source contract.
- Resolved workspace metadata against the merged strided revision, then ran
  formatting, the focused source-contract test, and the focused fast PR check.

## Remaining Risk

The new replay behavior is supplied by the merged upstream dependency. This
PR does not independently exercise or alter tenferro callers beyond compiling
and testing them against the adopted revision.
