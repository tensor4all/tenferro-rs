# PyTorch Dense CPU Parity Audit

This document audits tenferro's dense tensor coverage against the subset of
PyTorch relevant to the current workspace design effort:

- dense tensor primal execution
- VJP / JVP support
- oracle-backed HVP coverage where `tensor-ad-oracles` publishes a family
- layer cleanliness and CPU/GPU-generic abstraction boundaries

It is intentionally family-first rather than a literal one-row-per-PyTorch-op
inventory.

## Scope

This audit covers dense tensor functionality only. Sparse tensors, random
factories, FFT, sorting, indexing-heavy APIs, and neural-network higher-level
surfaces are out of scope.

## Audit Method

The audit groups APIs by tenferro family and then maps relevant PyTorch dense
CPU operations into those families. Coverage is tracked separately for:

- primal execution
- VJP
- JVP
- oracle-backed HVP
- CPU/GPU-generic abstraction cleanliness
- layer cleanliness

## Coverage Matrix

This section will record the family-first parity matrix.

## PyTorch-to-tenferro Mapping

This section will group PyTorch dense CPU APIs by the tenferro family that
should own them.

## Layer Findings

This section will record abstraction and layering issues discovered while
auditing parity gaps.

## Follow-up Backlog

This section will record the issue-ready backlog implied by the audit.
