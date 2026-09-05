# Public-boundary case contract refinement

Status: DeepSeek V4 Flash design **Correct-to-merge**, before implementation.
Parent #1758; integration prerequisite for benchmark #95.
Base: `6858475b7bd8156e8e78abe55c2a8958d6deca21`.
No Rust/runtime/public API/dependency change or performance claim.

## Observed mismatch

The merged inventory export describes `core.add.ordinary.concrete` as phase
`validation`, but its setup/workflow explicitly includes the complete public call
through dispatch and result wrapping. The benchmark's ordinary execution timer
cannot honestly bind to a validation-only phase. The export also lacks an ordinary
EagerTensor add contract; aliasing that route to concrete/traced would be wrong.
Existing case status is pending, not evidence of actual benchmark execution.

## Minimal change

1. Correct all existing `.ordinary.concrete` templates to phase `execution`,
   retaining IDs and their full-call workflow boundary. No existing measurement is
   relabeled: there are no accepted timings behind these pending definitions.
2. Add exactly one new canonical route contract, `core.add.ordinary.eager`, under
   `core.elementwise`, phase execution and surface eager. Scope: ordinary forward
   EagerTensor add, including its API-owned admission and result/optional AD
   recording; input/leaf construction and backward are outside. Scenario-level
   no-AD/active-AD settings remain explicit in the benchmark, not inferred here.
3. Support an optional per-case `operations` list restricting a template to a
   nonempty, duplicate-free subset of the selector's already-authoritative operations.
   Existing unfiltered templates still expand across the whole selector. This is a
   representative-case filter, not another operation registry or source of membership.
4. Validate each case/filter before expansion. Reject wrong types, empty/duplicate
   filters, unknown operations and selector-external operations. Filtering must not
   allow a discovered operation to lose all case contracts. Preserve global expanded
   ID uniqueness and all existing owner/source/staleness checks.
5. Regenerate Markdown and benchmark JSON through the existing checker. Export shape
   and schema identifier stay v1: existing fields/IDs remain, corrected metadata and
   one additive route are not a new wire format. No mirrored hand-maintained registry.

Parent confirmed all90 existing ordinary-concrete rows have the same complete-call
workflow, not a validation-only one. `execution` is already present in v1 cases and
`eager` is already a supported inventory surface. The export schema ID is a fixed
format identifier, distinct from the generated freshness digest; the latter changes
with the overlay. No exported field structure changes in this amendment.

## Source-backed regression linkage

The new eager case references the real test
`crates/tenferro-ad/tests/integration/numpy_api.rs::eager_add_uses_numpy_broadcasting_for_rank_padding_and_singletons`.
It constructs EagerTensor operands, calls `lhs.add(&rhs)`, checks [3,4] output shape
and all12 values. This is not EagerBackend/TensorElementwise masquerading as an eager
API test. Existing eager owners remain source-backed. AD-specific benchmark checks
are separate; this reference alone is not an AD coverage claim.

## Cross-repository interpretation

The exported ID identifies a canonical route/phase contract, not a concrete size,
layout or thread scenario. Benchmark instance IDs remain distinct, with actual
per-instance setup inclusions/exclusions and API tier. Fresh/shared concrete calls
may link the same ordinary surface but must not be compared as equivalent work when
shapes/call counts/setup differ. Eager scenarios must link the new eager contract.
Private components and compiled/prepared scenarios need their own accurate links;
the existing pending prepared/traced definitions are not blanket aliases for them.
This bounded amendment does not claim their integration or complete family coverage.

## Acceptance

- Existing180 case IDs retained; exactly one eager-add route added at this base.
- Ordinary concrete phases reflect full execution, not pure validation.
- Mutation tests cover bad filters, malformed filtered cases, uncovered operations,
  duplicate IDs and stale outputs. Positive test proves only add receives the new
  eager row while existing unfiltered expansion remains unchanged.
- Real eager test linkage resolves; no runtime/numerical/AD implementation changes.
- Checker, mutation tests, CI-helper tests, docs snippets, full relevant fast gate and
  committed-head deterministic rules review pass; Flash full-diff review then required
  hosted CI before merge. No benchmark timing required/claimed by this data amendment.
- Worklog records design approval before Luna edits, exact diff verdict and commands.
