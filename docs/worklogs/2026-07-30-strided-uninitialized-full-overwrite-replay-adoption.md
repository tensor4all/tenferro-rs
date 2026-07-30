# Strided Uninitialized Full-Overwrite Replay Adoption

## Summary

Advanced every workspace `strided-*` dependency from `4be1c8a8` to merge
commit `6885f52edb5fa2348fea47413bc43561e436ef63` from
tensor4all/strided-rs#185, which implemented and closed issue
tensor4all/strided-rs#184.

## Context And Decision

- This adopts the upstream uninitialized full-overwrite replay behavior
  specified by strided-rs#184 and implemented by strided-rs#185, following the
  prior merged strided adoption pattern.
- The adoption depends on merged upstream PR strided-rs#185 and does not change
  tenferro implementation or call paths.
- No tenferro call path changes are included yet; follow-up work remains
  separate from this pin update and does not start #1516A.

## Verification

- Updated the workspace manifest and exact-revision source contract.
- Resolved workspace metadata against the strided-rs#185 merge commit, then
  ran formatting, the focused source-contract test, and the focused fast PR
  check.

## Remaining Risk

The new replay behavior is supplied by merged upstream PR strided-rs#185. This
PR does not independently exercise or alter tenferro callers beyond compiling
and testing them against the adopted revision.
