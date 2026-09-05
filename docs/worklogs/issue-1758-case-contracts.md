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

## Einsum route-contract correction

Applied the Flash-approved data-only correction from
[`einsum-route-contract-correction`](../design/einsum-route-contract-correction.md):
184 exported cases now retain the prior 181 IDs and add only
`einsum.einsum.ordinary.eager`, `einsum.einsum.prepare.concrete`, and
`einsum.einsum.prepared.concrete`. Concrete ownership now records public plan
preparation, input-spec revalidation, and the caller-owned backend session;
eager ownership records direct `dot_general`, optional whole-program,
expanded standard-operation, and extension-fallback branches. Existing einsum
anchors were replaced with numerical/public-contract tests, including complex
and mismatch evidence. No Rust, dependency, API, timer, or benchmark-producer
code changed; benchmark statuses remain pending/follow-up.

The initial mutation run exposed stale fixed-count and eager-operation assertions.
Flash approved the design amendment before changing the existing Python test:
it now checks184 rows, the three exact new IDs/surfaces/phases, and add/einsum eager
operations. Parent independently verified no old ID was removed and all other
families remained byte-for-byte equivalent as parsed data. The inventory checker,
mutation script and CI-only fast gate passed. The stale-snapshot diagnostic is
intentional mutation-test output, not a gate failure. Final Flash full-six-file
review returned Correct-to-merge after correcting concrete admission to
not-applicable/caller-borrowed and Eager wrapping to tracked/untracked EagerTensor.
An earlier mutation-test finding referenced nonexistent code and was explicitly
retracted against the actual rejection helper and tests; no bypass was added.
Exact-head rules check and hosted CI/merge remain pending for this correction.
