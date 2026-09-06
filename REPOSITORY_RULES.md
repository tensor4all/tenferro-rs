# Repository Rules

Tenferro-specific rules, applied on top of the shared tensor4all rules from
`tensor4all-agent-rules` (see the shared `rules/index.md`). Keep this file
minimal.

Performance, layout, cache, threading, and backend contracts live in
`PERFORMANCE_TIPS.md`; read it before performance-sensitive work and reviews.

`scripts/repository-rules-review.py` treats each `##` heading in this file and
in `PERFORMANCE_TIPS.md` as a routing unit. Sections that should reach the
diff-scoped review bot must be listed in that script's `ALWAYS_SECTIONS` or
`SECTION_TRIGGERS`; human/process protocols go in `HUMAN_ONLY_SECTIONS`.

## Public Surface Drift

- `README`, rustdoc, and examples must not claim capabilities beyond the current public surface.
- When the public API changes, check `README`, rustdoc, and examples for stale names, stale capability claims, and deleted paths before considering the work complete.

## Naming Style

- Prefer `tensor4all` over `Tensor4all` in prose, documentation, issue text,
  comments, and contributor notes unless quoting an external name or preserving
  an existing proper noun.

## Public Surface Discipline

- Keep the public API small. Prefer `pub(crate)` for types, functions, traits,
  modules, fields, and helper constructors unless external users are expected
  to call them directly.
- Public types should implement `Debug`. Prefer `derive(Debug)` when cheap,
  useful, and free of unstable internals. Hand-write a summary for graph,
  runtime, cache, tensor, backend, or FFI wrapper types when derived output
  would materialize data, dump large buffers, leak representation details, or
  constrain future changes.
- Do not make implementation details public because another module needs
  them; first consider moving the code or adding a narrower crate-private
  helper.
- Public APIs should be selected deliberately and documented as user-facing
  contracts. Items mainly for tests, benchmarks, internal planning, execution
  dispatch, lowering, caching, or backend glue should normally be private or
  `pub(crate)`.
- `#[doc(hidden)] pub` is not a substitute for privacy. Use it only for
  supported macro output, required trait contracts, or documented extension
  contracts.
- Do not keep low-level execution helpers, dispatch entrypoints, cache
  plumbing, or internal IR evaluators public for tests, parity checks,
  convenience, or sibling-crate reach-through. When another crate needs
  access, expose the narrowest owner-scoped API that preserves runtime, cache,
  backend, and extension-dispatch invariants.
- Before adding or keeping a `pub` item, ask whether it is useful outside this
  repository and whether tenferro will support its semantics as a public
  contract. If unclear, keep it `pub(crate)` and expose a smaller high-level
  API.
- Tensor operation names are public vocabulary: unsuffixed names take owned
  compact tensor inputs; `_read` marks APIs that accept borrowed views or
  `TensorRead`-style references.
- Public shape parameters use `impl Into<R::Shape>` when the rank type owns the
  representation and `impl IntoShapeVec` at dynamic-rank boundaries, so
  arrays, vectors, slices, and `ShapeVec` values are accepted consistently.
  Axis lists, dimension mappings, and other borrowed collections use
  `impl AsRef<[T]>` unless ownership or iteration semantics require otherwise.
- Metadata-only layout/view operations use a `_view` suffix
  (`transpose_view`, `slice_view`, `reshape_view`). Never use `_view` for
  operations that allocate, canonicalize, execute kernels, or transfer data.

## Public Boundary Safety Audits

- User-reachable tensor, runtime, eager, traced, CPU, GPU, and extension APIs
  must validate input-derived shape, axis, dtype, padding, slice, gather,
  scatter, and linalg config before no-op shortcuts, allocation, launch
  planning, or FFI calls. Review fast paths and zero-size returns like the
  main path.
- Shape products, byte lengths, strides, offsets, launch sizes, padding extents,
  and FFI dimensions use checked arithmetic before conversion to `usize`, `i32`,
  `u32`, pointer offsets, or allocation sizes. Audits should search for
  `shape.iter().product`, `* size_of`, `as usize`, `as i32`, `as u32`, `stride
  *=`, and unchecked `+`/`*` on shape-derived values.
- Pointer-offset loops over batches, matrix blocks, tensor strides, or packed
  FFI pointer arrays check both the per-item stride product and the
  `batch * stride` offset. Source-contract tests suit GPU/FFI paths that not
  every CI machine can exercise.
- Publicly reachable library paths must not turn invalid user input into
  `panic`, `unwrap`, `expect`, unchecked indexing, poisoned-lock unwraps, or
  debug-only assertions. Keep truly internal invariants close to their proof;
  otherwise return a typed error.
- When a bug exposes a public API design mismatch, fix the canonical API
  contract. API compatibility is not a goal unless the task requires it. Do
  not preserve a panicking or lossy API by adding a parallel `try_*` escape;
  make the canonical operation return a typed `Result`. Keep `try_*` names
  only when they are the intended canonical Rust API.
- Do not keep public infallible accessors or constructors for operations that
  can fail through materialization, metadata registration, backend/device
  transfer, validation, or lock state. Replace them with `Result`-returning
  methods and update callers, docs, and tests directly; no deprecated
  panicking shims unless maintainers require a compatibility window.
- Public dtype conversion, promotion, and explicit lossy cast semantics are
  specified under `docs/spec/` before implementation. Keep checked `convert`
  separate from explicit `cast`, and keep CPU/GPU/eager/traced behavior
  aligned unless the owning spec names a backend limitation and its typed
  error.
- Public integer tensor arithmetic is a CPU/CUDA parity contract, not a
  debug-build Rust overflow contract. Supported `I32` and `I64` add, sub, mul,
  neg, abs, pow, `reduce_sum`, and `reduce_prod` use explicit two's-complement
  wrapping semantics in CPU code and matching CUDA kernels; no bare `+`, `-`,
  `*`, unary negation, or unchecked integer folds on user data unless the
  surrounding helper proves wrapping semantics. Integer div/rem/pow domain
  failures (division by zero, negative exponents) return typed errors, and
  CUDA support needs CPU-vs-CUDA edge-case tests before the capability
  descriptor marks it supported.
- AD cotangent seed helpers use dtype-aware constructors such as shared
  zero/one helpers, not analytic operations like `exp`, `log`, or `sin` as
  constant shortcuts. Seeds are dtype plumbing and must work for all supported
  dtypes or return an intentional typed limitation.
- If the cleanest fix reshapes tenferro AD transform or rule APIs, prefer that
  cleanup and optimize for the long-term contract. Do not hide a tenferro bug
  behind lossy error categories or local compatibility shims.
- Public cache, runtime, extension, and AD registry locks must not silently
  ignore poison by reporting empty/default state. Return a typed poison error
  where a `Result` is possible; where a non-`Result` API must remain for an
  approved compatibility or trait constraint, document the reason nearby and
  make failures visible.
- Do not add public errors, helpers, validation branches, or synthetic test
  seams for states unreachable through supported APIs. Validate real user and
  backend inputs, rely on proven internal invariants, and add failure
  contracts when the input or state is introduced. This never permits panics
  or unchecked handling of reachable failures.
- Public traced/eager helpers validate rank and axis counts before computing
  output ranks or indexing shape arrays. Symbolic-shape traced values must not
  pass through concrete-shape helpers unless the API explicitly returns a
  symbolic-shape error.
- Repeated public-boundary validation lives in shared helpers or fallible
  metadata/validation functions when sibling surfaces need the same rank,
  axis, dtype, shape, padding, or linalg checks. Do not duplicate hand-written
  checks across owned/read, eager/traced, CPU/GPU, or extension paths.
- Public validation APIs return crate error types, not `String` or `&str`.
  Translate into another layer's error type explicitly, preserving the
  original in the `source()` chain; render as text only at the final display,
  logging, serialization, or message-only external-protocol boundary.
- Validation helpers for operation configs return typed prepared metadata when
  downstream code would otherwise repeat indexing, rank-minus-one,
  shape-product, or dimension-role calculations. Pass validated
  `DotGeneral`, gather/scatter, slice, pad, linalg, or batch-layout metadata
  to backend code instead of calling `validate_*` and recomputing unchecked
  offsets.
- Do not build runtime operation configs from raw rank arithmetic such as
  `shape().len() - 1` or `rank - 1` unless a preceding checked helper proves
  the rank in the same expression or validated metadata type. Common
  rank-derived configs such as rank-2 matmul use shared helpers.
- Source-contract tests should guard high-risk patterns that have regressed
  before: raw `shape.iter().product`, unchecked rank-minus-one, shape inference
  indexing before config validation, and allocation helpers bypassing shared
  checked shape-product functions.

## Invariant Markers

- Use one canonical marker, `// INVARIANT: <why this is valid, bounded, or
  intentional>`, for non-obvious invariants kept for performance or by design:
  rank-bounded quadratic loops, checked-by-construction arithmetic,
  ownership-required copies, semantic zero-fill, constant-bounded scans, and
  similar audit false-positive hotspots. Keep `// SAFETY:` for unsafe blocks.
- The marker states the invariant concretely enough to re-verify: name the
  validation site, the bound, or the owning contract, not just the intent.
- `#[allow(...)]` suppressions for repository-mandated lints carry an adjacent
  `// INVARIANT:` line. Allowlist and ratchet-baseline entries in audit
  tooling carry a rationale string; an entry without a reason is a defect.
- Audit tooling and prompts, human or AI, must not flag a site governed by an
  `// INVARIANT:` marker as a violation; they check whether the stated
  invariant still holds and report only when it does not.
- Rejecting an audit finding as a false positive is complete only when the
  marker (or a source-contract test) has landed at the site, per the
  false-positive ledger rule in Work Logs And Design Records.
- Parallel operation surfaces keep validation and promotion semantics in parity
  across owned/read, eager/traced, CPU/GPU, and extension wrapper paths. A bug
  in one surface should trigger an audit of the others before the fix is
  complete.
- Every non-trivial binary under `docs/tutorial-code/src/bin/` should be covered
  by a runnable tutorial test unless explicitly compile-only or hardware-gated.
  Markdown snippets copied from those binaries stay synced with the source.

## Wrapper DRY And Codegen

- Do not assume tensor, eager, traced, and extension wrapper surfaces are
  isomorphic; they often differ in dtype constraints, feature gates, error
  ownership, shape metadata, AD behavior, or rustdoc examples.
- `macro_rules!`, build-time codegen, and small descriptors are acceptable for
  genuinely same-shaped wrapper families when the invocation is easier to
  review than the duplicated code.
- Avoid opportunistic macro/codegen refactors that hide public semantics,
  erase clear rustdoc, obscure feature gates, move errors to the wrong crate,
  or force unlike wrappers into a generic parameter bag.
- When macro/codegen produces public wrappers, keep generated names, docs,
  feature gating, and error behavior obvious at the call site, and add focused
  tests or doctests that fail if one generated wrapper stops forwarding.

## Work Logs And Design Records

- Nontrivial refactors, cleanup streams, AI-assisted implementation, and PRs
  with explicit design tradeoffs leave a curated work log under
  `docs/worklogs/` recording the session summary, code and documents read,
  reference implementations considered, decisions, alternatives rejected or
  deferred, verification performed, and remaining risks.
- Work logs are reviewer-facing decision records, not raw transcripts or
  implementation plans: concise enough to review, specific enough to explain
  why an abstraction, split, macro, descriptor, public API choice, or deferral
  was selected.
- PR bodies for such work link the `docs/worklogs/` file. Reviewers should read
  linked work logs before challenging scope, abstraction, or design intent.
- When a PR establishes or changes durable design intent, update the relevant
  `docs/design/` document in the same PR. Work logs hold session-level
  rationale; design docs hold decisions future implementation and review
  should continue to follow.
- When a bug report or audit finding is a false positive because of an
  intentional invariant, record the evidence in the issue or PR ledger and add
  a nearby `// INVARIANT:` comment (see Invariant Markers), rustdoc note, or
  source-contract test when the invariant is not obvious from the code, so
  later humans and agents do not rediscover the same non-bug.
- Before adding an audit or repository rule, inventory nearby rules and merge,
  tighten, or relocate overlapping guidance. Prefer one sharper general rule
  over many narrow bullets.

## Final Cross-Phase Multi-Agent Audit

Human/process protocol. This section is intentionally not routed to the
diff-scoped review bot.

Repository-scale, multi-phase implementation programs require one final audit
after every phase and its task-local reviews are complete, but before the
umbrella issue or implementation branch is declared ready for integration.

- Audit one exact candidate commit. Every report must name that commit, and an
  auditor must not audit a lane whose implementation or task-local review it
  performed. The lanes may run in batches when agent concurrency is limited.
- Assign a distinct independent auditor to each required lane:
  1. **Specification and architecture:** accepted issues, phase acceptance
     criteria, eager/graph semantic parity, extension lowering, and migration
     compatibility.
  2. **Rust safety and resource lifecycle:** aliasing, unsafe boundaries,
     lifetimes, permits, locks, buffers, caches, identifiers, and cleanup on
     success, error, cancellation, and unwind.
  3. **Performance and parallelism:** current-main baseline, eager fast path,
     allocations and request/container overhead, nested fan-out, provider
     worker ownership, thread-count and placement control, and CPU/GPU
     synchronization.
  4. **Public API and documentation:** facade boundaries, operation-family
     traits, typed errors, feature combinations, runnable examples, online
     parallelism documentation, and source/checker consistency.
  5. **CPU and NUMA:** managed and external domains, strict versus advisory
     placement, resource arbitration, faer/BLAS/strided behavior, multiple
     sockets, re-entry, fairness, and failure recovery.
  6. **GPU, XLA, and multi-GPU:** context/stream/event ownership,
     backend-neutral artifacts, compiler and prepared-operation caches, device
     placement, independent devices, and cross-device failure handling.
- After all lane reports, a separate integration auditor must check
  cross-phase invariants, duplicated or contradictory findings, and the
  closure evidence.
- Each lane report must record the candidate commit; relevant feature,
  toolchain, and hardware configuration; inspected files, public contracts,
  and issue acceptance criteria; fresh commands and complete result
  classifications; findings classified as `Critical`, `Important`, or
  `Minor`; and explicit limitations or skipped hardware paths. Performance
  results must be classified as `PASS`, `FAIL`, or `INCONCLUSIVE`. Do not infer
  a pass from an implementer's earlier run. Source scanners and mutation tests
  support, but do not replace, call-path review and runtime tests.
- Apply the existing [Public Boundary Safety Audits](#public-boundary-safety-audits),
  [Unsafe Code Boundary](#unsafe-code-boundary),
  [Performance-Sensitive Safety Contracts](PERFORMANCE_TIPS.md#performance-sensitive-safety-contracts),
  [Materialization And Copies](PERFORMANCE_TIPS.md#materialization-and-copies),
  [Performance-Gated Experiment Protocol](PERFORMANCE_TIPS.md#performance-gated-experiment-protocol),
  [Cache Ownership](PERFORMANCE_TIPS.md#cache-ownership),
  [CPU Threading Contract](PERFORMANCE_TIPS.md#cpu-threading-contract),
  [GPU Backend Contract](#gpu-backend-contract),
  [Documentation Policy](#documentation-policy), and
  [Work Logs And Design Records](#work-logs-and-design-records) in the relevant
  lanes instead of restating their detailed checklists here.
- The final audit passes only when every `Critical` and `Important` finding is
  fixed and independently re-reviewed; every `Minor` finding is fixed or has a
  written rationale and accepted tracking issue; every required performance
  gate is `PASS`; and the integration auditor reports no unresolved
  cross-phase contradiction. `INCONCLUSIVE` blocks promotion until a valid
  rerun or explicit accepted scope decision is recorded.
- Environment-limited CPU, GPU, XLA, or multi-device paths must retain
  reproducible diagnostics and an identified verification owner. The final
  worklog must link every lane report, the integration report, the exact
  candidate commit, and the final verification commands.
- This gate supplements rather than replaces task-local TDD, specification
  review, code-quality review, CI, and required performance gates.
- Auditing is read-only: audit agents must not modify the candidate while
  reviewing it. A finding fix creates a new exact candidate revision. Before
  the audit can pass, every lane report must be refreshed to name and validate
  that final revision: each auditor reviews the intervening diff, every
  affected lane reruns its relevant evidence, and an unaffected lane may carry
  earlier runtime evidence forward only with a recorded diff-impact rationale.
  The separate integration auditor runs last against the same final revision.

## External Contribution Intake

Human/process protocol. This section is intentionally not routed to the
diff-scoped review bot.

- PR creation is restricted to collaborators. External requests, reproducers,
  proposed regression tests, benchmark reports, and prototype branches should be
  collected in issues first.
- Collaborator bug-fix PRs may be reviewed as merge candidates when they fix
  behavior already intended by current docs, specs, or tests.
- New features start as feature request issues. A new-feature implementation PR
  is not the source of truth unless maintainers accepted the linked issue and
  agreed implementation should start; PRs opened before that may be closed and
  redirected to an issue, with prototype code linked from the issue as reference
  material.
- Accepted feature issues, specs, and repository rules are the source of truth
  for implementation. External prototype code may inform implementation only
  when its license and provenance are compatible with this repository.
- When maintainers take over a prototype branch, preserve the contributor's
  commits where practical and add new commits on top. When maintainer or
  AI-assisted implementation is otherwise based on external prototype code,
  preserve copyright notices, license obligations, attribution, and links to
  the original prototype or issue discussion.

## CI Cost Discipline

Human/process protocol. This section is intentionally not routed to the
diff-scoped review bot.

- Expensive CI lanes, especially GPU or larger-runner jobs, are gated behind
  cheaper repository-policy and non-GPU checks. Do not trigger hardware-backed
  runners directly on PR updates when an earlier review, lint, docs, or CPU
  test gate can reject the PR first.

## Publication Order And Publish-Safety

Maintainer and release-workflow protocol; not routed to the review bot. The
normative cross-repository publication rules live in `tensor4all-agent-rules`
(`rules/common/repository.md`); this section records only tenferro-specific
constraints.

- A publishable workspace crate must not have a versioned normal, build, or
  dev dependency on a publishable crate later in the canonical publication
  order (the Phase 3 list in `ai/contribution-workflows/release-publish.md`).
  Cargo resolves versioned dependencies during `cargo package` even with
  `--no-verify`, so a forward reference fails before any point of no return.
- Cross-layer tests may use path-only (unversioned) dev-dependencies on
  publishable crates: dev-dependencies are stripped from published manifests
  and never registry-resolve for consumers, so an unversioned path dev-edge is
  safe even when it points forward (`tenferro-xla` -> `tenferro-einsum`) or
  closes a cycle (`tenferro-runtime` -> `tenferro-cpu`). A declared dev
  version must still match the workspace version.
- The complete dependency graph among publishable crates, including dev and
  build edges, must not contain a publication cycle over versioned edges (the
  only edges that registry-resolve at package time).
- Published crates remain packageable from their manifests without temporary
  local `[patch.crates-io]` bootstrap configuration.
- Release validation is change-aware: a helper-or-workflow-only diff runs the
  focused `ci-config` lane; a publication-metadata-only diff runs metadata,
  publish-layout, and archive/dry-run checks; a semantic manifest diff runs
  affected tests plus the applicable CI tier; Rust source or ambiguous diffs
  run full validation. `scripts/release-validation-policy.py` classifies the
  lane from old/new manifest content. Before skipping a rerun on the strength
  of passed CI, verify every required check run for the exact release commit
  via `gh api repos/tensor4all/tenferro-rs/commits/<SHA>/check-runs --paginate --slurp`
  and require `head_sha == <SHA>`, `status == "completed"`, and
  `conclusion == "success"` per required check; anything else fails closed and
  reruns the applicable tier.
- Publication is human-only: agents generate a guarded handoff script with
  `scripts/release-publish.py --generate-script PATH` and never type the
  publication confirmation. The generated script re-runs the fail-closed
  preflight, requires one exact lowercase `y` at a TTY, and invokes the helper
  with `--execute` only after verifying SHA-256 pins of the helper and the
  canonical workflow; it is written outside the release worktree (mode 0700,
  `bash -n` clean, restart-safe).
- `scripts/check-publish-layout.py` enforces the canonical order, forward
  dependency, and publication-cycle rules; a violating manifest or release
  change fails the check before tagging.

## Standard Extension Boundary

- Standard operation families (`tenferro-einsum`, `tenferro-linalg`,
  `tenferro-fft`, and future peers) are first-class crates, not modules of a
  `tenferro` facade.
- There is intentionally no root `tenferro` facade crate. Do not add facade
  paths such as `tenferro::einsum`, `tenferro::linalg`, or `tenferro::fft`.
- Users import operation crates directly, bring their extension traits into
  scope, and register runtimes explicitly when graph execution reaches an
  extension family: for example `tenferro_einsum::TraceContextEinsumExt` /
  `tenferro_einsum::TracedTensorEinsumExt` plus
  `Runtime::builder().install_extension_module(...)` after registering the
  target engine.
- Extension runtime dispatch fails explicitly when a runtime owner exists but
  the family is not registered. Never silently fall back from a
  registered-runtime path to an `ExtensionOp::eager_execute` implementation,
  CPU backend, reference path, or freshly constructed backend. Public eager
  wrappers for standard extensions must either register their runtime before
  dispatch, following the crate's established pattern, or expose the
  missing-runtime error. Add regression tests for missing-runtime behavior
  when changing dispatch.
- Hidden backend hooks for internal optimized operations (prepared solves,
  values-only decompositions, cache-aware kernels, device-specific fast paths)
  must not silently fall back to a slower public operation, full decomposition,
  freshly constructed backend, CPU transfer, or reference implementation. The
  trait default should return an explicit unsupported or backend error; each
  supporting backend must override the hook. Add source-contract or behavior
  tests when introducing such hooks.
- Optional capabilities are feature boundaries, not new crates: put
  operation-specific AD support behind an `autodiff` feature in the owning
  crate, not in `*-ad` companion crates.
- User-facing operation crates expose backend features as `cuda`, `rocm`, and
  `webgpu`. Do not document or require a public `gpu` feature; use internal
  `cfg(any(feature = "cuda", feature = "rocm", feature = "webgpu"))` checks or
  internal helper features.
- Extension crates depend on the runtime, tensor, AD, or GPU crate they need;
  dependency flow must not require a facade crate to depend back on them.
- Extension AD rules are owned by an explicit `tenferro_ad::AdContext` or rule
  set. No process-global registration shims; a legacy bridge exists only when
  explicitly approved by the task or maintainer decision.

## Oracle Gate

- No AD rule implementation in the mainline without a corresponding oracle family.
- Prefer oracle families with both Torch reference data and finite-difference checks; finite-difference-only is acceptable when no Torch reference exists.
- If no oracle exists yet, add it to `tensor-ad-oracles` before treating the rule as a supported mainline AD rule.

## Rule Source Of Truth

- `SemanticProgram -> SemanticProgram` transforms in `tenferro-ad` are the
  semantic AD boundary. `tenferro-internal-ops` owns the primitive rule
  vocabulary used to implement them.
- Semantic transforms use the validation-preserving program builder rather
  than exposing primitive graph keys or mutating a frozen program.
- Do not model tensor primitive AD by implementing
  `chainrules_core::ReverseRule<StdTensorOp>` or `ForwardRule<StdTensorOp>`;
  those are value/tape-level interfaces, not the semantic transform surface.
- Avoid ChainRules-style `frule`/`rrule` terminology for new tenferro AD APIs.
  Use `linearize` for JVP graph emission and `transpose_rule` for transposed
  linear graph emission. Design any future externally supplied pullback API
  separately from the primitive rule contract.
- Reverse-mode support is not always a direct `transpose_rule` arm; some ops
  apply `linearize` and then transpose the emitted linear graph. Before filing
  or closing an AD support issue, check the machine-readable manifest in
  `tenferro-internal-ops/src/ad/support.rs`; `SupportedViaLinearize` means a
  missing direct `transpose_rule` arm is intentional.
- AD graph-emission rules distinguish rank, exact extents, conservative
  extents, and runtime shape sources. Do not call exact-shape helpers when the
  rule only needs rank or can emit runtime `DimExpr::InputDim` references.
  Exact shapes are required only for concrete op payloads that cannot
  represent runtime dimensions; handle non-exact metadata conservatively, never
  reinterpreting bounds as sizes.
- AD graph-emission invariants are enforced by API shape or structural tests,
  not prose. Constant, zero, one, and identity construction in AD rules should
  use the semantic helpers in `tenferro_ops::ad::support`; tests such as
  `identity_matrix_helper_emits_semantic_constant_and_remaps_shape_source` and
  `identity_matrix_fixed_uses_semantic_constant_not_analytic_shortcut` guard
  against analytic constant shortcuts.
- Reference JAX (`jax/_src/lax/lax.py`, `jax/_src/lax/linalg.py`) when
  implementing new AD rules.

## AD Rule Coverage

- Every `linearize` / `transpose_rule` implementation has a finite-difference
  integration test verifying numerical correctness.
- For linalg ops, prefer oracle families with both Torch reference data and
  finite-difference checks when available in `third_party/tensor-ad-oracles/`.
- **Numerical coverage is the real guarantee for AD rules, not line
  coverage.** An AD rule is covered when its differentiable outputs match a
  finite-difference and/or Torch oracle and its per-output support status is
  asserted against the machine-readable manifest, regardless of llvm-cov line
  percentage. For linalg this is enforced by:
  - the finite-difference sweep in
    `crates/tenferro-linalg/tests/integration/traced_ad_explicit.rs` (cholesky,
    qr, eig, eigh, lu, full_piv_lu, solve, triangular_solve, svd/eig/eigh
    values),
  - the manifest assertions in
    `crates/tenferro-linalg/tests/integration/ad_support_manifest.rs` against
    `crates/tenferro-linalg/src/ad/support.rs` (`all_linalg_ad_support()`
    covers every dispatch arm and output status), and
  - the oracle support table in `docs/oracle/tensor-ad-oracles-support.md`.
- Consequently `crates/tenferro-linalg/src/ad/rules/*.rs` carry intentionally
  below-default per-file thresholds in `coverage-thresholds.json`. Their
  uncovered lines are dtype-guard arms (real to complex casts, integer-dtype
  early returns) and F32/error branches the oracles do not exercise. Do not
  add line-padding tests, and do not read the low coverage as missing AD
  validation. When adding or changing a linalg AD rule, extend the
  finite-difference sweep and the manifest first; raise the threshold only
  when real new numerical tests warrant it.

## No Ad Hoc Fixes

- Do not add ad hoc fixes that violate DRY, KISS, or layering.
- Do not introduce compatibility shims, duplicated logic, or downstream reach-through into lower layers when the correct fix belongs in an existing seam or high-level API.

## Unsafe Code Boundary

`unsafe` is confined to FFI bindings and backend leaf code and is near-absent
from the algorithmic layers. Do not read the raw count as a red flag without
checking where it lives.

- **Count it correctly.** Plain `grep -c unsafe` over-counts ~30%+ (comments,
  docs, string literals, tests, `target/`). Use
  `python3 scripts/count-unsafe.py` for the production figure with a
  per-crate / per-category breakdown. As of this writing the real total is
  ~460, essentially all FFI/backend: cuSOLVER/cuTENSOR/cuBLAS and LAPACK
  bindings, the batched raw-pointer setup feeding them, SIMD elementwise
  kernels, thread affinity, and buffer pools (`tenferro-linalg`,
  `tenferro-cpu`, `tenferro-gpu`).
- **Keep the higher layers unsafe-free.** `tenferro-runtime`,
  `tenferro-einsum`, `tenferro-ad`, and the graph/AD logic in
  `tenferro-tensor` are ~zero `unsafe` by design. A needed kernel or FFI call
  belongs behind the backend/FFI seam, not in graph or rule code.
- **Document each block.** Keep the validation invariant next to the `unsafe`
  block (`// SAFETY:`) and test the boundary conditions. New FFI binding files
  should follow the existing `cusolver.rs` / `lapack_linalg/*` patterns.

## File Organization

Keep source files small and focused, but do not split solely to reduce line
count. ~1000 lines is a soft review trigger, not a mechanical limit; a coherent
single concern may stay larger until a clear behavior, abstraction, feature, or
ownership boundary exists.

A split must make the code easier to reason about. Prefer boundaries such as
parsing, plan computation, execution, dispatch, public API, AD rules, operation
families, backend glue, validation, or cache ownership. No `part1` / `part2`
splits, and no tiny files that scatter one concept across many modules.

Line count decides where to inspect first; responsibility, change frequency,
public/private API boundaries, and human navigation decide whether and how to
split.

## Unit Test Organization

Keep production source files focused on production code. No inline
`#[cfg(test)]` blocks in normal modules unless the file is a genuinely tiny
leaf module and the test is trivially small. Prefer module-local test
directories such as `src/<module>/tests/*.rs`, leaving only
`#[cfg(test)] mod tests;` in the source file. Reserve crate-root `tests/` for
integration tests. Do not use `include!` to inject test files.

A developer reading `src/**` should not scroll through large unit-test blocks;
split larger extracted suites by concern rather than keeping one monolithic
module.

Tests follow implementation ownership.

- Public facade crates should prefer integration tests for user-visible
  behavior.
- Private implementation details must be tested in the owning crate, typically
  an internal crate, not through a public facade.
- If a crate sets `[lib] test = false`, do not add `src/**/tests`, inline
  `#[cfg(test)] mod tests`, or other crate-local unit-test entrypoints.
- If a private helper in a facade crate needs direct unit testing, move it into
  the owning internal crate instead of re-enabling facade-crate lib tests.
- Repository contract tests enforce this rule and must stay green in CI.

## Device Transfer And Backend Buffer Errors

- tenferro follows the PyTorch convention: no implicit CPU-GPU transfer.
  Tensors must already live on the device the backend operation requires.
- Canonicalization is allowed only within the existing placement. Host views
  may copy into host compact tensors; GPU backend views may canonicalize or
  copy back on the GPU. Neither may download or upload payloads unless the API
  is explicitly a transfer API.
- User code explicitly uploads CPU tensors before CUDA execution and downloads
  CUDA tensors before CPU-only execution or host value inspection.
- A GPU backend op receiving a CPU tensor returns a runtime-state error saying
  the op expected a GPU tensor and pointing to `upload_tensor()`.
- A `Result`-returning CPU backend op receiving a backend/GPU buffer returns a
  runtime-state error at the CPU backend boundary, saying to download the
  tensor to host first.
- Direct host-inspection APIs such as `TypedTensor::host_data()` and
  `TypedTensor::host_data_mut()` return `Result` and report backend buffers as
  runtime-state failures. Any infallible host-inspection API returning a
  borrowed slice must document an explicit panic boundary before exposure.
- Execution pipeline internals may handle placement for documented cases:
  `Constant` ops may auto-upload through `upload_host_tensor()`, and
  host-dependent ops such as `ShapeOf` or `DynamicTruncate` may read metadata
  or download single scalar values as the backend contract documents.
- Unsupported CUDA op or dtype combinations return an explicit error, never a
  silent CPU fallback.

## CPU Kernel Implementation

- No naive CPU loop fallbacks. CPU tensor kernels use optimized
  implementations unless listed as an exception below.
- CPU affine-strided copy, permutation, broadcast, map, zip-map, fused
  elementwise replay, sum/product reductions, gather, additive scatter, and
  fixed-window dynamic slice/update delegate to `strided-rs` when their
  semantics fit an existing strided plan. Tenferro owns tensor semantics,
  validation, dtype dispatch, placement checks, error translation, execution
  contexts, and reusable output storage; it does not duplicate the
  tensor-sized traversal.
- Required CPU implementations by operation category:

  | Category | Required implementation |
  |---|---|
  | Elementwise (`add`, `mul`, `neg`, `exp`, fused eager replay, ...) | `strided-kernel` (`map_into`, `zip_map2_into`, `ErasedFusedPlan`, etc.) |
  | Reduction (`reduce_sum`, `reduce_prod`) | `strided-kernel` (`ErasedReducePlan::compile_axes`) |
  | Structural (`transpose`, `broadcast`, `extract_diag`) | `strided-kernel` (`permute` + `copy_into`, `broadcast`, `diagonal_view`) |
  | Gather | `strided-kernel` (`ErasedGatherPlan`) |
  | Additive scatter | `strided-kernel` (`ErasedScatterPlan`) |
  | Fixed-window dynamic slice/update | `strided-kernel` (`ErasedDynamicSlicePlan`, `ErasedDynamicUpdateSlicePlan`) |
  | GEMM (`dot_general`) | faer (`cpu-faer`) or BLAS (`cpu-blas`) |
  | Linalg (`svd`, `qr`, `cholesky`, `eigh`, `solve`) | faer (`cpu-faer`) or LAPACK (`cpu-blas`) |

- The ownership and overlap boundary is:

  | Operation or responsibility | `strided-rs` owner | tenferro owner |
  |---|---|---|
  | Affine-strided copy and permutation | Bulk `copy_into` traversal and serial/parallel kernel selection | Shape, stride, offset, reachable-range, dtype, placement, and destination validation; backend-scoped allocation and error mapping |
  | Broadcast | Zero-stride broadcast views and bulk copy/map traversal | Broadcast dimension semantics, output shape, placement, and allocation |
  | Unary map, binary zip-map, and fused elementwise replay | Affine iteration, tiling, static specialization, and erased replay execution | Operation semantics, dtype dispatch/promotion, capability checks, and errors |
  | Sum/product reductions | Axis and multi-axis strided reduction replay | Axis validation, identities, dtype policy, output wrapping, and max/min NaN policy exceptions |
  | Gather | Indexed-read traversal and erased replay dispatch | Gather semantics, index validation/normalization, dtype dispatch, output allocation, and error translation |
  | Additive scatter and fixed-window dynamic slice/update | Indexed replay traversal for matching erased plans | Index validation/normalization, clamp semantics, dtype dispatch, output allocation, and error translation |
  | Other indirect indexing | No ownership until a suitable general primitive exists | Indirect-index semantics and current dedicated kernels |
  | Einsum/dot-general | Reusable strided primitives may be consumed where they fit | Planning, optimized preparation, provider integration, and benchmark accountability |

<!-- TENFERRO_CPU_STRIDED_OWNERSHIP_CONTRACT_BEGIN -->

  ```text
  schema = tenferro.cpu-strided-ownership.v1
  affine-kernel-owner = strided-rs
  affine-kernels = copy,permutation,broadcast,map,zip-map,fused-elementwise,sum-product-reduction,gather,additive-scatter,dynamic-slice,dynamic-update-slice
  einsum-owner = tenferro:benchmark-backed-exception
  execution-entry = CpuBackend
  execution-resources = persistent-BufferPool,uninitialized-full-overwrite,CpuContext-Rayon,nested-execution,serial-parallel-threshold
  noncompliant = context-free-strided-call,throwaway-pool,ambient-global-Rayon
  resource-classification = memory-reuse-and-thread-policy:execution-not-metadata
  ```

<!-- TENFERRO_CPU_STRIDED_OWNERSHIP_CONTRACT_END -->

- Ownership priority is lower-layer first: when a CPU operation is metadata
  preparation plus an existing `strided-rs` primitive, tenferro delegates the
  bulk traversal. If a generally useful primitive is missing, add it to
  `strided-rs` first. A tenferro-owned traversal is allowed only for semantics
  outside the affine-strided model or for an explicitly approved,
  benchmark-backed exception. Einsum is the benchmark-backed exception; new
  exceptions require an accepted issue, comparative benchmark evidence, and a
  recorded ownership rationale.
- Calling a `strided-rs` primitive is necessary but not sufficient. Public
  tensor-sized CPU work enters through `CpuBackend` and its configured
  execution scope: persistent `BufferPool`, fully-overwritten uninitialized
  output allocation, configured `CpuContext` Rayon pool, nested-execution
  safety, and serial/parallel threshold policy. A context-free `strided-rs`
  call, a throwaway pool, or Rayon's ambient global pool is non-compliant.
  Memory reuse and thread policy are execution resources, not tensor metadata;
  backend-neutral tensor/view types expose metadata-only layout transforms and
  do not own data-moving convenience methods.
- Exceptions with dedicated implementations: `reshape` (metadata-only),
  `embed_diagonal`, index-dependent triangular masks (`tril`/`triu`), max/min
  reductions until strided exposes matching NaN semantics, and indexing ops
  without a matching strided plan such as static slice, pad, concatenate, and
  reverse.
- CPU provider features are additive. At least one of `cpu-faer` or `cpu-blas`
  is enabled; both together must compile. `CpuBackend` owns runtime provider
  selection: `CpuBackend::new()` picks the default compiled provider
  (BLAS/LAPACK when `cpu-blas` is compiled, otherwise `cpu-faer`); explicit
  constructors or application configuration select another compiled provider.
- Tensor-sized CPU kernels run through the repository CPU threading policy.
  `strided-kernel` is compiled with its `parallel` feature for elementwise,
  reduction, and structural materialization kernels. CPU contraction providers
  receive their policy from the owning `CpuExecutionContext`; Faer uses
  `Par::Seq` for one-thread contexts and explicit `Par::rayon(n)` for bounded
  multi-thread contexts.
- A tensor-sized CPU operation that remains a dedicated sequential loop
  because no strided-kernel/backend-native parallel primitive fits keeps a
  nearby source comment naming that rationale. Undocumented serial loops must
  not become the default fallback.

## Tensor Core Data Model

- `tenferro-tensor-core` owns backend-independent host tensor metadata and
  contiguous host storage: `DType`, `TensorScalar`, `HostTensor<T>`, dynamic
  `Tensor`, host/dynamic views, `TensorRef`, `ShapeVec`, `StrideVec`,
  `SliceSpec`, and metadata-only `reshape_view`, `transpose_view`, and
  `slice_view`.
- It must not depend on CUDA, GPU backends, backend buffers, provider
  selection, or execution backend traits. Its owned `Tensor` must not grow
  inherent `TensorBackend` execution helpers.
- Core views and layouts validate bounds eagerly with checked arithmetic.
  `TensorLayout` metadata views may use signed strides and negative slice
  steps when reachable-range validation succeeds; zero step remains invalid.
  Do not implement `PartialEq` for views.
- `ShapeVec` and `StrideVec` use `SmallVec` with inline rank capacity 8.

## GPU Backend Contract

- Before touching CubeCL/GPU backend code, read
  [`docs/design/gpu-backend-design.md`](docs/design/gpu-backend-design.md).
  It is the developer-facing source for CubeCL kernel ownership, runtime
  shape/stride metadata conventions, launch configuration rules, and device
  transfer behavior; any change to those conventions updates it in the same
  PR.
- CUDA operation paths whose accepted implementation is an NVIDIA vendor
  library must fail with typed load or provider errors when the required
  NVIDIA library is unavailable or lacks support. They must not silently fall
  back to native CubeCL kernels. Native CubeCL kernels are separate provider
  implementations for operations or dtypes not covered by the accepted
  vendor-library path, not fallback tiers for missing CUDA libraries.
- CUDA `eig` (non-symmetric eigenvalue decomposition, LAPACK `dgeev`) is not
  provided by cuSOLVER. `CudaBackend::eig` returns `Unsupported`; users
  download to CPU and use `CpuBackend::eig`.

## Structured Error Classification

- Validation facts use the shared typed vocabulary for shape, rank, axis,
  dtype, configuration, and invalid arguments. Eager and traced paths report
  the same validation kind and payload for the same known input; `ErrorPhase`
  is a separate graph-build, compile, or execution axis.
- Unsupported operations and operation-specific unsupported dtypes use
  `Unsupported`. A crate owning a richer reason retains a typed local source;
  `UnsupportedDTypeConversion` is reserved for an actual from-dtype to
  to-dtype conversion.
- Singularities, non-convergence, division by zero, and other numeric-domain
  failures use `NumericalFailure` and retain the owning crate's typed source.
- Typed backend/kernel errors use a source-preserving backend wrapper. The
  text-only `BackendFailure` category is reserved for vendor/backend status
  text with no typed source or more specific category.
- Typed file, stream, serialization, and dynamic-library failures use `Io` and
  retain their source. Missing, uninitialized, poisoned, or otherwise invalid
  executor, cache, device, and buffer state uses `RuntimeState`, retaining a
  typed source when one exists. Impossible internal invariants remain
  `Internal`. None of these are catch-alls for known input validation.
- Public error conversion preserves the `source()` chain across crate
  boundaries. Converting a typed error to `String` is permitted only for
  display/logging, vendor/FFI arguments, final serialization, or an explicit
  message-only external protocol boundary.
- When a documented alternative API, explicit conversion, feature, or
  supported-value set provides one reliable remediation, append it to the
  diagnosis as `<what failed>; <what to do>`. Do not invent a remedy when no
  universal next action exists.

## Documentation Policy

### Source of Truth

- **Source code** is the source of truth for internal design (op catalog, backend contract, AD rules, compilation pipeline).
- **Online docs** are primarily user-facing: how to use the explicit runtime,
  AD, tensor, GPU, and standard operation crates.
- The online **Internals section** is the exception for implementation-oriented
  readers: it may publish curated architecture, specification, and active
  design notes when those pages are linked from rendered docs. Historical
  `docs/plans/` records stay offline unless a maintainer decides otherwise.
- **AGENTS.md** is the entry point for developers and AI agents, with pointers to source code locations.
- Do NOT duplicate source-code-level information in online docs; if it can be learned from the source, put a pointer.
- Development assumes AI agentic coding. Machine-readable sources (code + doc comments) are authoritative.

### User-Facing Docs

- User docs target PyTorch/JAX users of public direct crates such as
  `tenferro-runtime`, `tenferro-ad`, `tenferro-gpu`, `tenferro-einsum`,
  `tenferro-linalg`, and `tenferro-fft`.
- Imports use public direct crates such as `tenferro_runtime`, `tenferro_ad`,
  `tenferro_gpu`, and standard extension crates. Do not reference internal
  crates (`tenferro-internal-ops`, `computegraph`, etc.) in user-facing docs.
- Do NOT expose internal jargon (Graph, StableHLO, ExecOp, ValueRef, etc.) in user-facing pages.
- Provide PyTorch/JAX equivalents when introducing tenferro concepts.

### User-Facing Code Snippet Consistency

- Non-trivial user-facing snippets have an executable source of truth: include
  a checked example, test, or doctest instead of hand-copying code into
  Markdown.
- If a Markdown page must contain a copied snippet, add an automated sync or
  extraction check that fails when it drifts from the executable source.
- Guide code demonstrating a workflow compiles in CI. When runtime execution
  needs special hardware or external libraries, CI still compile-checks the
  example with the required feature flags, and the guide documents the command
  that runs it on a correctly configured machine.
- CPU and GPU workflow examples should include meaningful assertions on shapes,
  dtypes, or values whenever the result is deterministic; avoid print-only
  examples unless the output itself is the documented behavior.
- GPU quickstart examples use the same executable source as the guide,
  explicitly show upload/download boundaries, and assert downloaded CPU
  results for at least one supported operation.

### Diagram Consistency

- Diagrams in docs and `README.md` (SVG under `docs/assets/`, ASCII diagrams
  in design, architecture, and spec pages) are part of the documented surface:
  crate names, layer assignments, dependency directions, and public entry
  points must match the current implementation.
- A PR that adds, removes, renames, or re-layers a crate, or changes a
  diagrammed extension or backend boundary, updates the affected diagrams in
  the same PR.
- Stale diagrams are worse than missing diagrams; delete rather than leave
  inaccurate.
- Diagrams show boundaries and dependencies, not internal implementation
  detail the source code owns.

### Doc Examples

- Every public type, trait, and function includes minimal but sufficient usage
  examples in its doc comments (`/// # Examples`). `#[doc(hidden)]` items are
  exempt.
- Doc examples must NOT use `ignore` or `no_run`; every example compiles AND
  runs as a doctest. If it cannot, refactor it until it can.
- An example demonstrates real usage. Path-only or assignment-only examples
  (for example `let _method = Type::method;`) are not acceptable.
- Use `compile_fail` only for examples that intentionally demonstrate compile errors.
- When a canonical public API changes from infallible or panicking to
  `Result`-returning, update its examples, tutorials, and guides to propagate
  errors with `?` or handle the documented error explicitly, not by appending
  `unwrap()` or `expect()`. Tests may keep an intentional assertion boundary,
  and an example may unwrap only a locally proven invariant whose proof is
  explicit.
- Examples calling CPU or GPU backend operations should bind the backend to a
  local variable and reuse it, instead of chaining `Backend::new().op(...)`
  beyond a single trivial construction example.

### Public Result Error Documentation Gate

- Every public function, inherent method, and public-trait method returning a
  `Result` documents a `# Errors` section naming the concrete error variants
  or failure conditions the caller can observe; a generic "returns an error on
  failure" sentence is insufficient.
- Documentation for traced or symbolic APIs describes validation deferred to
  compile or execution and names the applicable `ErrorPhase`
  when that is part of the contract. Intentional panics use `# Panics`;
  deferred symbolic checks use `# Deferred errors`.
- `scripts/check-public-error-docs.py` audits the full Rust workspace and the
  Rust files changed by a PR and is a required CI gate. For traced APIs, prose
  promising validation at compile/execution or after input binding also goes
  under `# Deferred errors`. Keep the audit enabled without `clippy` or
  source-level allowlists; add the documentation at the API source.

## PR Content Hygiene

- Do not include AI-generated analysis, task, or verification reports as
  standalone files in PRs. Durable session records belong in
  `docs/worklogs/`; everything else stays out of the repository.
- Do not commit new top-level directories or dot-directories without an
  explicit maintainer decision recorded in the PR.
- User-facing guides must not pin commit hashes in setup instructions (for
  example `git checkout <hash>`); reference branches, tags, or released
  versions.

## Generic Over Scalar Type

- Use generic constructors with sealed traits instead of per-type functions.
- Bad: `TracedTensor::from_f64(...)`, `TracedTensor::from_f32(...)`, etc.
- Good: `TracedTensor::new<T: TensorScalar>(shape, data)`; type inference selects the variant.
- For dtype-polymorphic tensor operations, prefer one typed generic
  implementation plus outer dtype dispatch over per-dtype copies.
- If Rust generics cannot express the shared structure cleanly, use a local
  macro for the repetitive dispatch or variant plumbing.
- Sealed traits (`TensorScalar`, `PoolScalar`, etc.) restrict APIs to the
  supported dtype set.

## Public API Convention

- **AD core ops**: `EagerTensor` and `TracedTensor` use methods as the
  canonical surface for single-output operations (`x.exp()`,
  `x.reshape(shape)`, `a.dot_general(&b, config)`), operator overloads where
  they read naturally (`&a + &b`, `&a * &b`), and associated functions for
  core operations with no natural receiver (`EagerTensor::where_select(...)`,
  `TracedTensor::concatenate(...)`).
- **Non-AD concrete ops**: `Tensor` and dynamic-rank `TypedTensor<T>` use
  crate-root session extension traits (`TensorSessionOpsExt`,
  `TypedTensorSessionOpsExt`, and `TypedTensorMaskSessionOpsExt`) whose
  methods run inside a caller-provided `BackendSession` (entered via
  `TensorBackend::with_backend_session`). Private helper modules are fine, but
  public `tensor` / `typed_tensor` module free functions are not part of the
  release API.
- **Extension families**: extension crates cannot add inherent methods to
  external tensor types, so their canonical tensor-facing surface is extension
  traits (`TracedTensorLinalgExt`, `EagerEinsumExt`, `TraceContextEinsumExt`,
  `TracedTensorEinsumExt`, `TracedTensorFftExt`) re-exported at the crate
  root. Do not expose public `traced_tensor` / `eager_tensor` module free
  functions for standard operation families.
- **No compatibility shims for operation-surface style changes**: when API
  compatibility is not explicitly required, remove old module functions
  instead of keeping wrappers beside the canonical surface.
- No `traced_` prefix on methods. `TracedTensor` methods are inherently traced.

### Output And Write Surface Vocabulary

- Operation suffixes describe ownership and output-update semantics; never use
  a suffix only to avoid a naming conflict.
- Unsuffixed operation names allocate and return a fresh result tensor.
- `_read` means one or more inputs are `TensorRead`/borrowed views; the output
  is still backend-allocated unless another suffix names an output.
- Bare `_into` overwrites a caller-provided output. The operation validates
  dtype and shape, does not resize the destination, and does not read the
  previous output value as part of the semantic update.
- `_read_into` is the main backend hook shape for borrowed inputs plus an
  overwritten `TensorWrite` output.
- `_in_place` is a destructive update of an input tensor, distinct from
  `_into`. Do not implement in-place elementwise variants by passing aliased
  input/output views into generic `*_into` kernels unless the kernel's
  aliasing contract is proven and tested.
- `_add_to` is read-modify-write accumulation, `out += op(...)`. Do not hide
  accumulation behind a bare `_into` method.
- Dot/GEMM read-modify-write uses the existing `_into_accum` vocabulary with a
  `DotGeneralAccumulation` argument; do not introduce `_linear_into` as a
  second spelling.
- Typed-face preallocated methods may take `&mut TypedTensor<T>` when the dtype
  is statically known. Erased-face methods should take `TensorWrite<'_>` so
  dynamic dtype and view outputs are validated at the backend boundary.
