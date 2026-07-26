# API and Convention Freeze

## Purpose

This document records the clean-break release posture for tenferro-rs API and
internal-structure cleanup before the first release that treats public APIs as
durable contracts.

The goal is not to preserve compatibility with pre-release callers. The goal is
to make the release surface coherent, small, documented, and aligned with crate
ownership. Compatibility aliases, old names, and public implementation details
should be removed instead of carried forward when they make the release API less
clear.

This document is implementation-facing design intent. Normative operation
semantics, trait signatures, tensor layout rules, backend behavior, AD
contracts, and extension contracts belong in `docs/spec/`. User workflows and
examples belong in `docs/guides/` and rustdoc.

## Release Stance

Before the release freeze closes, breaking changes are allowed when they make
the API or internal structure cleaner. A change is clean when it reduces the
public surface, moves behavior to the owning crate, removes duplicated
semantics, or makes device, layout, dtype, backend, AD, cache, or extension
boundaries explicit.

The release branch should avoid:

- compatibility shims and old-name aliases;
- public helper APIs that exist only for tests, benchmarks, internal planning,
  lowering, cache plumbing, backend glue, or sibling-crate reach-through;
- broad facade paths that hide direct crate ownership;
- user-facing docs that restate internal implementation details instead of
  linking to the owning spec or design document;
- examples that compile but do not verify meaningful behavior.

The release branch should prefer:

- fewer public items with stronger documentation and examples;
- direct crate imports and explicit feature flags;
- explicit CPU/GPU transfer, layout, dtype, and materialization boundaries;
- spec-backed behavior and design-backed ownership decisions;
- automated checks that make drift visible before merge.

## Source Of Truth

Each durable fact should have one owner.

| Location | Owns |
| --- | --- |
| `docs/spec/` | Normative contracts: op semantics, trait signatures, backend behavior, AD contracts, tensor layout rules, extension identity and dispatch. |
| `docs/design/` | Current implementation design: ownership boundaries, architecture choices, migration rationale, performance design, and review intent. |
| `docs/worklogs/` | Session-level rationale for completed nontrivial work. These explain why a change was made in one PR, not the long-term rule by themselves. |
| `docs/plans/` | Historical planning records. They are not updated to match the current release design. |
| `docs/guides/` and `README.md` | User workflows and public capability summaries. They should not expose internal graph, IR, cache, or backend plumbing details. |
| rustdoc | Exact public item contracts and minimal examples that compile and run as doctests. |

When two documents conflict, the more specific normative document wins for the
behavior it owns. If the conflict is between current design documents, the
release cleanup must either resolve the conflict or mark one document as
historical before relying on it.

API convention rules that should remain stable after the release cleanup live
in `docs/spec/api-conventions.md`. This design document records the cleanup
posture, triage workflow, and migration rationale.

## Crate Ownership

The release API should make crate ownership obvious.

| Crate | Release ownership target |
| --- | --- |
| `tenferro-tensor-core` | Host-only dtype tags, scalar traits, rank/layout metadata, compact host tensors, and metadata-only host views. It must not depend on execution backends, graph execution, AD, or GPU runtimes. |
| `tenferro-tensor` | Runtime dense tensor values, runtime views, placement metadata, backend traits, and backend-parametric tensor contracts. It does not own CPU or GPU kernel implementations. |
| `tenferro-cpu` | CPU backend implementations, CPU kernels, CPU execution context, provider selection, thread policy, and CPU resource pools. |
| `tenferro-gpu` | GPU backend implementations, device buffers, explicit upload/download helpers, CubeCL/CUDA integration, GPU launch metadata, and GPU resource ownership. |
| `tenferro-core-ops` | Internal primitive operation catalog metadata shared by lower layers. |
| `tenferro-internal-ops` | Internal graph operation vocabulary and graph-level AD rule implementations. |
| `tenferro-runtime` | Backend-agnostic runtime helpers, traced tensor construction, graph compilation and execution, extension registration, and extension cache storage. |
| `tenferro-ad` | Eager AD surfaces, eager tensors, AD contexts, traced AD helper traits, and user-facing differentiation APIs. |
| `tenferro-einsum` | Einsum public API, subscript parsing, contraction planning, concrete/traced/autodiff-eager wrappers, extension runtime, extension-owned caches, and optional AD rule ownership. |
| `tenferro-linalg` | Linalg public API, extension payloads, linalg backend hooks, CPU linalg kernels, optional CUDA linalg bridge code, eager/traced wrappers, and optional AD rule ownership. |
| `tenferro-fft` | FFT public API, extension runtime registration, concrete/traced wrappers, optional AD rule ownership, and FFT-specific execution behavior. |

If an implementation needs a cross-crate bridge, the bridge must be narrower
than the internal subsystem it exposes. `#[doc(hidden)] pub` is not a substitute
for an ownership contract; it is still public and should be treated as release
surface unless a design document names the bridge and explains why it exists.

## Public API Strata

Every exported item should fall into one release stratum.

| Stratum | Meaning | Release requirement |
| --- | --- | --- |
| Release API | Intended for downstream users. | Public rustdoc contract, runnable example, accepted owning crate, and guide coverage when it teaches a workflow. |
| Owner-scoped bridge | Required only because Rust crate boundaries separate owning implementation units. | Narrow API, explicit design rationale, no user-facing guide promotion, and tests at the owning boundary. |
| Experimental or unsupported | Visible only because a feature is intentionally incomplete or stubbed. | Explicit docs that say what is unsupported and what error is returned. Avoid exporting if a private item is sufficient. |
| Internal | Used for tests, planning, lowering, dispatch, cache plumbing, provider selection, or backend glue. | `pub(crate)` or private. Move tests or helpers to the owning crate instead of widening visibility. |

The default classification is internal. An item becomes release API only when
there is a clear downstream use case, the owning crate is correct, the name
matches release conventions, and the behavior can be documented without
describing private implementation machinery.

## API Convention Doctrine

Release APIs should follow these naming and shape rules.

- There is no root `tenferro` facade crate. Users import direct crates such as
  `tenferro_tensor`, `tenferro_cpu`, `tenferro_runtime`, `tenferro_ad`,
  `tenferro_einsum`, `tenferro_linalg`, and `tenferro_fft`.
- Standard operation families remain first-class crates, not modules under a
  broad facade.
- Tensor operation names are public vocabulary. Owned compact tensor inputs use
  unsuffixed operation names.
- `_read` is reserved for APIs that explicitly accept borrowed read-oriented
  inputs.
- `_view` is reserved for metadata-only layout operations. Any operation that
  allocates, canonicalizes, executes kernels, transfers data, or otherwise
  materializes data must not use `_view`.
- Flat-buffer constructors, exports, examples, and FFI contracts must say
  whether data is column-major. Constructor names should encode non-obvious
  layout expectations.
- Scalar constructors should prefer generic `TensorScalar`-bounded entry points
  over per-dtype constructors.
- Unary single-output traced ops are methods on `TracedTensor`.
- Binary single-output ops use operator overloads when the operator is natural;
  otherwise they use methods.
- Multi-output linalg and decomposition ops are tensor extension-trait methods.
- Einsum is owned by `tenferro-einsum` through crate-root extension traits:
  concrete owned input slices/arrays use `TensorEinsumExt` and
  `TypedTensorEinsumExt`, while borrowed inputs use `TensorReadEinsumExt` and
  `TypedTensorReadEinsumExt`; repeated concrete executions use
  `ConcreteEinsumPlan`; traced graph construction uses
  `TraceContextEinsumExt`; autodiff eager inputs use `EagerEinsumExt`.
- Traced tensor methods do not use a `traced_` prefix.
- User-facing backend features are named for concrete backend families, such as
  `cuda` and `rocm`. Public extension crates should not expose a vague `gpu`
  feature.
- Optional operation-specific AD support belongs behind `autodiff` in the
  owning operation crate.
- Result-returning public backend boundaries must return explicit errors for
  unsupported operation, dtype, or placement combinations. They must not fall
  back to CPU or transfer tensor payloads implicitly.

## API Inconsistency Detection

The freeze audit must detect API inconsistencies as first-class findings, not
only missing docs or overly broad visibility.

An API inconsistency exists when two public surfaces expose the same concept but
diverge without a documented reason in one or more of these dimensions:

- name, suffix, module path, or crate path;
- method vs free-function shape;
- owned input vs borrowed/read input shape;
- typed vs dynamic dtype behavior;
- eager vs traced behavior;
- CPU vs GPU placement behavior;
- error type, error variant, or panic/`Result` boundary;
- feature flag, optional dependency, or backend availability;
- layout convention for flat buffers;
- AD support, unsupported behavior, or fallback behavior;
- rustdoc examples and guide snippets.

The audit should compare APIs by concept, not by exact spelling. For example,
`reshape`, `reshape_view`, traced reshape, eager reshape, dynamic tensor
reshape, backend reshape, and guide examples belong to one concept family even
when they live in different crates. A difference is acceptable only when the
owning spec, design doc, or rustdoc contract explains the reason.

A missing counterpart can also be an inconsistency. If a guide, rustdoc page,
feature flag, trait, or operation matrix implies that a concept is available on
typed, dynamic, eager, traced, CPU, GPU, or extension surfaces, the audit must
either find the corresponding API or record the absence as unsupported with an
explicit owner and error contract.

The first API consistency matrix should cover these surface pairs:

| Surface pair | Detection question |
| --- | --- |
| typed tensor vs dynamic tensor | Do constructors, layout assumptions, accessors, and error behavior expose the same concept with compatible names and contracts? |
| owned tensor ops vs borrowed/read ops | Are `_read` and unsuffixed names used only for their documented input ownership model? |
| metadata views vs materializing ops | Are `_view` names limited to metadata-only behavior across core, runtime, docs, and examples? |
| eager vs traced APIs | Do same-concept operations have intentional naming, dtype, shape, AD, and error differences? |
| runtime extension APIs vs operation crates | Does each extension operation use the owning operation crate as the public import path and register runtime behavior explicitly? |
| CPU backend vs GPU backend | Do unsupported dtype/op/placement combinations return explicit errors with consistent user guidance? |
| rustdoc vs guides vs README | Do examples use the same public crate paths, names, layout conventions, and supported capability claims? |

The output of inconsistency detection should be a ledger entry for each root
cause, not one issue per spelling mismatch. Each entry should record the
concept family, affected APIs, owner crate, expected convention, current
divergence, and whether the fix is a rename, hide/remove, docs clarification,
or design decision.

## Internal Structure Doctrine

Internal cleanup should follow the same ownership rules as the public API.

- Fix behavior at the owning abstraction. Avoid downstream patches around a
  lower-layer gap.
- Remove duplicate compatibility paths after the release replacement is in
  place.
- Prefer one typed generic implementation with outer dtype dispatch over
  hand-copied per-dtype bodies.
- Use macros or small descriptors for wrapper families only when the generated
  wrappers are genuinely same-shaped and docs, feature gates, and error
  behavior remain obvious at the call site.
- Split files by behavior, abstraction, feature, ownership, or public/private
  API boundary. Do not split solely to lower line count.
- Keep cache ownership explicit and top-level. Caches need bounds, clear paths,
  stats, and documented ownership.
- Keep CPU provider selection inside `tenferro-cpu` and GPU transfer/device
  behavior inside `tenferro-gpu`.
- Keep AD rule semantics in graph-level `linearize` and `transpose_rule`
  surfaces unless a separate accepted design changes that model.
- Keep tests with the crate that owns the behavior. Do not widen public API
  only so another crate can test a private helper.

## Documentation Normalization

The release cleanup should normalize documentation before declaring the API
frozen.

1. Classify active `docs/design/` files as current design, current migration
   note, or historical context. Historical material should move out of the
   active design path or clearly say that it is historical.
2. Ensure each `docs/spec/` page owns a distinct normative area and that other
   docs link to it instead of restating the same contract.
3. Ensure `README.md` and `docs/guides/` describe only supported public
   workflows through direct public crates.
4. Ensure rustdoc examples compile, run, and assert meaningful behavior.
5. Ensure GPU, AD, linalg, einsum, and FFT docs distinguish supported,
   unsupported, and feature-gated behavior with explicit error expectations.

## Audit And Remediation Workflow

The clean-break freeze should proceed in this order:

1. Accept this doctrine as the release cleanup baseline.
2. Generate or otherwise derive a public API inventory for every workspace
   crate that exports user-facing or bridge APIs.
3. Build concept-family consistency matrices for typed, dynamic, eager,
   traced, backend, extension, rustdoc, guide, and README surfaces.
4. Classify each public item as release API, owner-scoped bridge,
   experimental/unsupported, or internal.
5. Classify each inconsistency or visibility finding as `Auto Fix`,
   `Verify First`, `Design Gate`, or
   `Out Of Scope`.
6. Resolve design-gated decisions before editing code that depends on them.
7. Apply remediation in coherent batches grouped by ownership and verification
   path.
8. Update the owning `docs/spec/`, `docs/design/`, guides, rustdoc, and tests
   in the same batch as the code change.
9. Run release gates before declaring the batch complete.

Public API removal, crate-boundary changes, feature flag changes, AD semantics
changes, and backend support-surface changes are allowed during this freeze,
but each one needs an explicit owner and verification path.

## Enforcement Targets

The release branch should converge on automated checks for:

- public API inventory diffs;
- API concept-family consistency matrices;
- release-freeze API convention findings via
  `python3 scripts/check-api-consistency.py --fail-on-findings`;
- forbidden facade paths and internal crate names in user guides;
- `_view`, `_read`, feature flag, and traced method naming conventions;
- missing rustdoc examples on release APIs;
- doctests and guide snippets that compile and run where hardware allows;
- docs/spec/design link integrity and source-of-truth ownership;
- unsupported GPU, AD, linalg, einsum, and FFT behavior returning explicit
  errors rather than silent fallbacks.

These checks do not replace maintainer review. They make the release doctrine
cheap to enforce after the cleanup is complete.
