# strided Policy Threshold Pin

## Summary

Updated tenferro-rs to consume `tensor4all/strided-rs@18d74ffacccefee8c476379c21a99eae532b917b`,
the merge commit for strided-rs PR #164. That strided-rs change makes large
erased indexed/reduction/static-indexing replay paths honor
`ExecContext::max_threads(n)` above the shared `MINTHREADLENGTH` threshold.

## Context

- strided-rs #161 identified that erased replay could silently degrade
  `ExecContext::MaxThreads(n)` to serial execution.
- tenferro-rs #1501 already removed production hot-path hardcoding of
  `ExecContext::serial()` by routing CPU backend execution through
  `CpuExecutionContext::strided_exec_context()`.
- strided-rs #164 then added the initial threshold-backed parallel replay for
  axis reductions, gather, dynamic slice, dynamic update overwrite, and pad.

## Decision

Pin all workspace `strided-*` dependencies to strided-rs #164 and update the
CI source-contract expectation to the same revision. Do not change tenferro
call sites in this PR: production backend dispatch already passes the explicit
CPU-derived strided execution context, and remaining `ExecContext::serial()`
uses are direct helper tests or the intentional one-thread fallback inside
`CpuExecutionContext::strided_exec_context()`.

## Verification

- `python3 -m unittest scripts.ci.tests.test_build_artifact_contracts -v`
