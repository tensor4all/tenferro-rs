# Issue 1761: binary einsum skips contraction-order search

Parent: tensor4all/tenferro-rs#1758 (Phase 0). Extracted from the mixed branch
`archive/1760-einsum-probes-20260906`; the crate-private component probes that
motivated this change stay on that archive branch and are not merged.

## Change

`ContractionTree::optimize_with_options` builds the tree directly through
`from_pairs(subscripts, shapes, &[(0, 1)])` when there are exactly two operands.
Options are still validated, `from_pairs` still performs shape validation and
step-plan construction, and three or more operands still go through TreeSA.
No rank, dtype, spelling or surface is specialized, no cache is added, and
prepared-execution checks are unchanged.

## Evidence

- Diagnostic at 0457a2ed (Apple M5 Max, release, faer, explicit 1 thread,
  F64 2x2): automatic tree optimization 4.94 us versus tree from a fixed pair
  1.98 us; the ordinary one-shot string einsum was 8.10 us and reused-plan
  execution 2.23 us. Separate medians, not an additive profile.
- Behavioral seam: test-only thread-local counters at both general optimizer
  entries (omeco and self-greedy). Public `optimize`, `optimize_with_options`
  and `ConcreteEinsumPlan::prepare` for a binary contraction leave both at
  zero; N-ary planning and an explicit fallback call are positive controls.
  Removing only the shortcut makes the new test fail with counts (1, 0).
- Label/shape-equivalence and rejection tests for the binary path pass both
  before and after the shortcut.

The end-to-end paired base/candidate number for the ordinary public call is
recorded in the PR body under the parent's measurement standard (same host,
sequential, CoV <= 10% valid).
