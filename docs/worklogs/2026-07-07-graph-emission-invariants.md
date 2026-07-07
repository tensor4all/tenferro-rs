# Graph Emission Invariants Work Log

## Summary

This session implemented the HANDOFF v2 structural-prevention item for AD graph
emission. The immediate bug class was dtype-polymorphic constants written as
analytic identities, especially linalg identity construction through
`Exp(anchor - anchor)`.

## Context Read

- `HANDOFF.md` v2 section F from the issue #1321 handoff
- `REPOSITORY_RULES.md` AD rule-source and work-log sections
- `ai/contribution-workflows/bugfix-pr.md`
- `ai/contribution-workflows/repository-remediation.md`
- `docs/architecture/ad-pipeline.md`
- `docs/spec/ad-contract.md`
- `crates/tenferro-internal-ops/src/ad/{support.rs,zeros.rs}`
- `crates/tenferro-linalg/src/ad/rules/support.rs`

## Decisions

- Expose semantic AD rule helpers through `tenferro_ops::ad::support` instead
  of adding required methods to `PrimitiveRuleBuilder`. This keeps existing
  builder implementations stable while giving extension and operation-family
  rules a public rule-support path for constants, zero-like, one-like, and
  identity-matrix emission.
- Rewrite linalg one-like and identity helpers to use semantic constants
  instead of analytic `Exp` shortcuts.
- Add structural tests that fail when identity helper emission uses analytic
  operations for constants.
- Record the durable policy in active architecture/spec docs and point
  repository rules at machine-checkable gates.
- Add the capability-landing sweep step to the remediation workflow so new
  canonical helpers trigger a search for obsolete workaround patterns.

## Verification

Verification performed during implementation:

- `cargo test -p tenferro-linalg --features autodiff,cpu-faer identity_matrix_fixed_uses_semantic_constant_not_analytic_shortcut -- --nocapture`
- `cargo test -p tenferro-internal-ops --features autodiff identity_matrix_helper_emits_semantic_constant_and_remaps_shape_source -- --nocapture`

Broader verification is recorded in the PR body.

## Residual Risks

The structural tests cover the helper path and the linalg identity construction
that regressed. They do not yet enumerate every possible extension rule fixture.
Future AD-rule capability landings should add or reuse structural gates for the
specific invariant they introduce.
