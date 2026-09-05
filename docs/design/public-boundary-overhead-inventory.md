# Public boundary overhead inventory

Status: design approved by DeepSeek V4 Flash (round 1); implementation in the assigned worktree pending parent full-diff review.
Non-blocking review pins are resolved below.

Issue: https://github.com/tensor4all/tenferro-rs/issues/1759
Parent: https://github.com/tensor4all/tenferro-rs/issues/1758
Inventory baseline: `0457a2ed0aeea21b14f4297f7f4731e09b3a0507`.

## Outcome and non-goals

Maintain an evidence-backed map from every current operation family and public
surface to validation, metadata preparation, optional planning, execution
admission/session/provider, and result/AD wrapping. This change inventories and
checks those responsibilities; it does not optimize them or claim timings.
Do not add public APIs, new dependencies, caches, a universal execution pipeline,
or a second operation registry. Numerical behavior and existing user changes
remain unchanged. Later measured optimizations belong to #1760–1762.

## Existing authorities

Use the core descriptors in `crates/tenferro-core-ops/src/catalog.rs`, backend
capabilities in `crates/tenferro-tensor/src/capability.rs`, primitive AD registry
and support in `crates/tenferro-internal-ops/src/ad/`, and extension-owned operation
kinds, capabilities and AD support (including
`crates/tenferro-linalg/src/ad/support.rs`). These remain authorities for operation
membership and semantic support. Audit einsum, linalg, FFT, sparse and tropical,
and host/dynamic/metadata routes, not merely the first CPU numerical examples.
Inspect existing manifest tests and source-contract scripts before adding checks.
Use a new sibling `scripts/check-public-boundary-inventory.py` as the inventory
entry point. Reuse existing extraction utilities where applicable, but do not copy
or extend `scripts/check-operation-categories.py`'s hardcoded operation lists.
Discover and check `ext/sparse` and `ext/tropical` authorities explicitly, despite
their exclusion from the root Cargo workspace. No new Rust AD-support manifests
are needed for this inventory.

## Maintained overlay and generated inventory

Add a small machine-readable overlay under existing `docs/` infrastructure and
one standard-library Python checker/generator under `scripts/`. It references
canonical operation identifiers or source-derived categories, not copied enum
bodies. Membership is derived from the existing authorities. Explicit override
selectors are allowed for different routes within one family; reject unknown,
duplicate, overlapping and uncovered selectors. Newly introduced families,
operations and renamed identifiers must fail until their disposition is reviewed.
Prefer reusing an existing extractor/checker; any source extractor added here
must detect unsupported input syntax rather than silently yielding no operations.

For each operation resolve concrete, typed, borrowed/read, output-reuse, eager
and traced surfaces. Missing surfaces get explicit unsupported or follow-up
reasons; no omitted column implies support. Each route records:

- responsibility owners with repository-relative source and symbol references;
- metadata facts reused and the required shape/dtype/layout/config/placement
  compatibility, mutation validity, owner and lifetime contract;
- source-observed allocation sites, payload access/materialization/transfer
  boundaries, clearly distinguished from measured allocation counts;
- disposition: measured (published exact evidence), alias-with-evidence (named
  route and forwarding/test evidence), unsupported (contract reason), or explicit
  follow-up (owning issue, question and required evidence);
- representative component/public-workflow case contracts and existing regression
  tests, including applicable values-only, materialization/transfer, output reuse
  and nested-session behavior.

Semantic equivalence does not establish performance equivalence. Aliases may
share numerical coverage but retain distinct boundary timing cases. Source-only
allocation observations never become measured timings or measured counts. All
unmeasured routes must remain explicit follow-ups, not implicitly successful.
GPU/XLA/multi-device limitations are stated separately from CPU dispositions.

Generate a deterministic human-readable snapshot identifying the exact baseline
revision and the overlay/source inputs. `git rev-parse HEAD origin/main` confirmed
both as the stated baseline before implementation. Keep baseline identity distinct
from later inventory commits: unrelated commits must not stale the snapshot just
because HEAD changed. Validate source-derived membership against the current tree
and record the resolved generation revision/provenance without a self-referential
commit-hash requirement. Operation/family additions deliberately require reviewed
dispositions in the same PR, which the PR/worklog must state. Check it for staleness in CI. The maintained
overlay is the only hand-maintained overhead map; generated tables are not edited
by hand or copied into unrelated guides.

## Change selection and benchmark seam

Allow the checker to select affected representative case IDs from changed owner
paths, conservatively selecting all cases for unknown relevant source changes.
Tie stable case IDs to explicit operation, phase, surface and setup-boundary
contracts. Pending benchmark #95 implementations must be reported as pending,
not runnable or measured. A machine-readable export supplies benchmark #95 and
#96 without copying library implementation. The later benchmark integration must
validate these IDs/contracts against its executable cases before the parent closes.

Keep validation and metadata probes distinct. Preserve operation-specific planning:
binary contraction still needs pair metadata even when no contraction-order search
is required. Prepared repetition validates compatible inputs; integer-label equations
are structured input, not a blanket promise of no preparation.

## Canonical API roles

Ordinary allocating/read/output APIs are the default and should become efficient
at their existing shared owners. Existing prepared APIs serve compatible repetition
with explicit preparation/setup cost. Integer-label/programmatic equations express
structured equations. Eager and traced routes retain their execution and AD semantics.
No API removal is proposed here; an actual redundancy requires a later explicit
source-backed decision and synchronized caller/docs migration, not another fast spelling.

## Verification and acceptance

1. Exhaustive family/operation/surface disposition at the recorded baseline; verify
   owner references, case contracts and regression-test references exist.
2. Checker mutation tests reject operation addition/removal/rename, missing family,
   unknown or overlapping selectors, invalid disposition/evidence, missing owner,
   stale generated snapshot and malformed authority extraction. A source edit in a
   shared owner selects all associated representative cases; unknown source changes
   select conservatively. Test that semantic aliases keep boundary cases distinct.
3. Run checker/generator consistency and its focused Python tests. Integrate with
   the existing CI helper/policy lane and test that integration, rather than inventing
   another workflow. Run the applicable local `scripts/check-pr-fast.sh` gate and
   committed-head deterministic `scripts/repository-rules-review.py` with
   `--dry-run --llm-skipped-reason "local deterministic review"`.
4. Reuse existing applicable Rust regressions; add no numerical tests merely for
   a documentation/script-only diff. Any Rust change requires a reviewed design
   amendment, focused tests and coverage review, public doctests and CI-parity lint.
5. Record exact reviewed design/diff revisions, commands and limitations in a
   curated `docs/worklogs/` entry. DeepSeek V4 Flash must approve this design before
   Luna implements, then review the entire final diff. Fix and re-review findings.
6. No benchmark collection is needed to claim this inventory's explicit follow-up
   dispositions. Actual performance acceptance remains unresolved in the parent
   until validated ordinary end-to-end evidence and the final cross-phase audit.

## Alternatives rejected

A second Rust operation enum/AD registry duplicates semantic authorities. A single
mandatory pipeline forces unnecessary setup onto simple operations. Static claims
that a suspected lock/allocation dominates replace evidence with guesses. Broad
Cartesian-product benchmarking is unnecessary; the derived responsibility map
selects representative cases while documenting exceptions explicitly.
