# Canonical case contract refinement

Base6858475b; parent#1758, benchmark#95 integration. Design:
[public-boundary-case-contract-refinement](../design/public-boundary-case-contract-refinement.md).

## Gate before implementation

DeepSeek V4 Flash design review: **Correct-to-merge** before Luna implementation.
No blocking findings. Parent verified the two non-blocking clarifications: all90
ordinary concrete contracts describe full calls, execution is existing phase
vocabulary, eager is an existing surface, and the fixed v1 schema identifier is
separate from the content freshness digest. The per-case filter must be validated
before expansion, reject empty rather than treating it as absent, and preserve
coverage for every discovered operation.

No runtime/numerical/API change, no timing evidence and no parent completion claim.
## Implementation and parent verification

Luna implemented the validated per-case filter, phase corrections and one genuine
eager-add route. Parent independently checked the full script diff and compared
exports: all180 original IDs and fields preserved except the intended90 phase
corrections; exactly `core.add.ordinary.eager` added.

Passed: inventory checker (181 cases/six families), mutation tests, 19 change-policy
and36 run-profile tests, doc snippets, diff checks and the complete focused fast gate.
The stale-snapshot message during mutation tests is an expected negative assertion,
not a failure. An extra attempted Rust test used nonexistent target `numpy_api`
instead of `integration`; no Rust-test execution is claimed. No Rust code changed;
the new reference was traced to its actual EagerTensor.add test body.

Flash full-diff review, committed-head rules check and hosted CI/merge remain pending.
