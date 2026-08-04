# Repository Rules

These are tenferro-specific rules. Apply them on top of the shared tensor4all
rules from `tensor4all-agent-rules`.

`scripts/repository-rules-review.py` treats each `##` heading below as a
routing unit. Sections that should reach the diff-scoped review bot must be
listed in that script's `ALWAYS_SECTIONS` or `SECTION_TRIGGERS`; human/process
protocols must be listed in `HUMAN_ONLY_SECTIONS` instead.

## Public Surface Drift

- `README`, rustdoc, and examples must not claim capabilities beyond the current public surface.
- When the public API changes, check for stale names, stale capability claims, and deleted paths in `README`, rustdoc, and examples before considering the work complete.

## Naming Style

- Prefer `tensor4all` over `Tensor4all` in project prose, documentation,
  issue text, comments, and contributor notes unless quoting an external name
  or preserving an existing proper noun.

## Public Surface Discipline

- Keep the public API intentionally small. Prefer `pub(crate)` for types,
  functions, traits, modules, fields, and helper constructors unless external
  users are expected to call them directly.
- Public types should implement `Debug`. Prefer `derive(Debug)` when the output
  is cheap, useful, and does not expose unstable internals. Use a hand-written
  summary for graph, runtime, cache, tensor, backend, or FFI wrapper types when
  derived output would materialize data, dump large buffers, leak internal
  representation details, or make future implementation changes harder.
- Do not make implementation details public just because another module needs
  them. First consider whether the code belongs in the same crate/module, or
  whether a narrower crate-private helper is the right abstraction.
- Public APIs should be selected deliberately and documented as user-facing
  contracts. If an item is primarily for tests, benchmarks, internal planning,
  execution dispatch, lowering, caching, or backend glue, it should normally be
  private or `pub(crate)`.
- `#[doc(hidden)] pub` is not a substitute for privacy. Use hidden public items
  only for explicitly supported macro output, required trait contracts, or
  documented extension contracts; otherwise make the item private or
  `pub(crate)`.
- Do not keep low-level execution helpers, dispatch entrypoints, cache plumbing,
  or internal IR evaluators public only for tests, parity checks, convenience,
  or sibling-crate reach-through. When another crate genuinely needs access,
  expose the narrowest owner-scoped API that preserves the intended runtime,
  cache, backend, and extension-dispatch invariants.
- Before adding or keeping a `pub` item, ask whether it is useful outside this
  repository and whether tenferro is prepared to support its semantics as a
  public contract. If the answer is unclear, keep it `pub(crate)` and expose a
  smaller high-level API instead.
- Tensor operation names are public vocabulary. Use unsuffixed operation names
  for owned compact tensor inputs, and add a `_read` suffix only for APIs that
  explicitly accept borrowed views or `TensorRead`-style input references.
- Public shape parameters use `impl Into<R::Shape>` when the rank type owns the
  representation and `impl IntoShapeVec` at dynamic-rank boundaries, so arrays,
  vectors, slices, and `ShapeVec` values are accepted consistently. Axis lists,
  dimension mappings, and other borrowed parameter collections use
  `impl AsRef<[T]>` unless ownership or iteration semantics require a different
  contract.
- Metadata-only layout/view operations must use a `_view` suffix, for example
  `transpose_view`, `slice_view`, or `reshape_view`. Do not use `_view` for
  operations that allocate, canonicalize, execute kernels, or transfer data.

## Public Boundary Safety Audits

- User-reachable tensor, runtime, eager, traced, CPU, GPU, and extension APIs
  must validate input-derived shape, axis, dtype, padding, slice, gather,
  scatter, and linalg config before no-op shortcuts, allocation, launch planning,
  or FFI calls. Review fast paths and zero-size returns with the same scrutiny as
  the main path.
- Shape products, byte lengths, strides, offsets, launch sizes, padding extents,
  and FFI dimensions must use checked arithmetic before conversion to `usize`,
  `i32`, `u32`, pointer offsets, or allocation sizes. Audits should search for
  `shape.iter().product`, `* size_of`, `as usize`, `as i32`, `as u32`,
  `stride *=`, and unchecked `+`/`*` on shape-derived values.
- Pointer-offset loops over batches, matrix blocks, tensor strides, or packed
  FFI pointer arrays must check both the per-item stride product and the
  `batch * stride` offset. Source-contract tests are appropriate for GPU/FFI
  paths that cannot be exercised on every CI machine.
- Publicly reachable library paths must not turn invalid user input into
  `panic`, `unwrap`, `expect`, unchecked indexing, poisoned-lock unwraps, or
  debug-only assertions. If an invariant is truly internal, keep it close to the
  proof; otherwise return a typed error.
- When a bug exposes a public API design mismatch, prefer the cleanest
  root-cause design fix by changing the canonical API contract. API
  compatibility is not a goal unless the task explicitly requires it. Do not
  preserve a panicking or lossy public API by adding a parallel `try_*`
  compatibility escape; normally make the canonical operation return a typed
  `Result`. Keep `try_*` names only when they are the intended canonical Rust
  API, not a workaround for avoiding a cleaner API change.
- Do not keep public infallible accessors or constructors for operations that
  can fail through materialization, metadata registration, backend/device
  transfer, validation, or lock state. Replace the canonical API with a
  `Result`-returning method and update callers, docs, and tests directly; do
  not leave deprecated panicking shims behind unless maintainers explicitly
  require a compatibility window.
- Public dtype conversion, promotion, and explicit lossy cast semantics must be
  specified under `docs/spec/` before implementation. Keep checked `convert`
  separate from explicit `cast`, and keep CPU/GPU/eager/traced behavior aligned
  unless the owning spec names a backend limitation and its typed error.
- Public integer tensor arithmetic is a CPU/CUDA parity contract, not a
  debug-build Rust overflow contract. Supported `I32` and `I64` add, sub, mul,
  neg, abs, pow, `reduce_sum`, and `reduce_prod` paths must use explicit
  two's-complement wrapping semantics in CPU code and matching CUDA kernels.
  Do not use bare `+`, `-`, `*`, unary negation, or unchecked integer folds on
  user data in these paths unless the surrounding helper proves wrapping
  semantics. Integer div/rem/pow domain failures such as division by zero or
  negative exponents must return typed errors, and CUDA support must include
  CPU-vs-CUDA edge-case tests before the capability descriptor marks it
  supported.
- AD cotangent seed helpers must use dtype-aware tensor constructors such as
  shared zero/one helpers, not backend analytic operations like `exp`, `log`,
  or `sin` as a shortcut for constants. Seed construction is dtype plumbing,
  not math dispatch; it must work consistently for all supported tensor dtypes
  or return an intentional typed limitation.
- If the cleanest fix requires reshaping tenferro AD transform or rule APIs,
  make that API cleanup the preferred repair path and optimize for the
  long-term clean contract first. Do not hide a tenferro bug behind lossy error
  categories or local compatibility shims when a clearer AD contract is the
  root-cause fix.
- Public cache, runtime, extension, and AD registry locks must not silently
  ignore poison by reporting empty/default state. If the method can return a
  `Result`, return a typed poison error. If a non-`Result` API must remain
  because of an explicitly approved compatibility or trait constraint, document
  that reason near the wrapper and make failures visible rather than fabricating
  success.
- Do not add public errors, helpers, validation branches, or synthetic test
  seams for states unreachable through current supported APIs. Validate real
  user and backend inputs, rely on proven internal invariants, and add future
  failure contracts only when the corresponding input or state is introduced.
  This does not permit panics or unchecked handling of reachable failures.
- Public traced/eager helpers must validate rank and axis counts before
  computing output ranks or indexing shape arrays. Symbolic-shape traced values
  must not be forced through concrete-shape helpers unless the API explicitly
  returns a symbolic-shape error.
- Repeated public-boundary input validation must live in shared helpers or
  fallible metadata/validation functions when sibling operation surfaces need
  the same rank, axis, dtype, shape, padding, or linalg checks. Do not duplicate
  hand-written checks across owned/read, eager/traced, CPU/GPU, or extension
  paths when a helper can enforce validation before fast paths and reduce public
  panic risk.
- Public validation APIs must return crate error types, not `String` or `&str`
  errors. If a caller needs to translate into another layer's error type, it
  should do so explicitly while preserving the original typed validation error
  in the `source()` chain; render it as text only at the final display,
  logging, serialization, or message-only external-protocol boundary.
- Validation helpers for operation configs should return typed prepared
  metadata when downstream code will otherwise repeat indexing, rank-minus-one,
  shape-product, or dimension-role calculations. Prefer passing a validated
  `DotGeneral`, gather/scatter, slice, pad, linalg, or batch-layout metadata
  value to backend code over separately calling `validate_*` and then
  recomputing unchecked offsets.
- Do not construct runtime operation configs from raw rank arithmetic such as
  `shape().len() - 1` or `rank - 1` unless a preceding checked helper proves the
  rank is large enough in the same expression or validated metadata type. Common
  rank-derived configs such as rank-2 matmul must use shared helpers.
- Source-contract tests should guard high-risk public-boundary patterns that
  have previously regressed, including raw `shape.iter().product`, unchecked
  rank-minus-one, shape inference indexing before config validation, and
  allocation helpers that bypass shared checked shape-product functions.

## Invariant Markers

- Use one canonical marker, `// INVARIANT: <why this is valid, bounded, or
  intentional>`, for source comments that record a non-obvious invariant kept
  for performance or by design: rank-bounded quadratic loops,
  checked-by-construction arithmetic, ownership-required copies, semantic
  zero-fill, constant-bounded scans, and similar audit false-positive
  hotspots. Keep `// SAFETY:` for unsafe-block justifications.
- The marker must state the invariant concretely enough that a later reader
  can re-verify it — name the validation site, the bound, or the owning
  contract — not just assert intent.
- `#[allow(...)]` suppressions for repository-mandated lints must carry an
  adjacent `// INVARIANT:` line. Allowlist and ratchet-baseline entries in
  audit tooling must carry a rationale string; an entry without a reason is a
  defect.
- Audit tooling and audit prompts, human or AI, must not flag a site governed
  by an `// INVARIANT:` marker as a violation. They must instead check whether
  the stated invariant still holds and report only when it does not.
- Rejecting an audit finding as a false positive is complete only when the
  marker (or a source-contract test) has landed at the site, per the
  false-positive ledger rule in Work Logs And Design Records.
- Parallel operation surfaces must keep validation and promotion semantics in
  parity across owned/read, eager/traced, CPU/GPU, and extension wrapper paths.
  A bug in one surface should trigger an audit of the corresponding surfaces
  before the fix is considered complete.
- Every non-trivial binary under `docs/tutorial-code/src/bin/` should be covered
  by a runnable tutorial test unless the example is explicitly compile-only or
  hardware-gated. Markdown snippets copied from those binaries must stay synced
  with the executable source.

## Wrapper DRY And Codegen

- Do not assume tensor, eager, traced, and extension wrapper surfaces are fully
  isomorphic. Public wrapper layers often differ in dtype constraints, feature
  gates, error ownership, shape metadata, AD behavior, or rustdoc examples.
- Do not categorically avoid `macro_rules!`, build-time code generation, or
  small descriptors for wrapper families. They are acceptable when the wrapper
  family is genuinely same-shaped and the invocation remains easier to review
  than the duplicated hand-written code.
- Avoid opportunistic macro/codegen refactors that hide public semantics,
  erase clear rustdoc, obscure feature gates, move errors to the wrong crate,
  or force unlike wrappers into a generic parameter bag.
- When macro/codegen is used for public wrappers, keep generated public names,
  docs, feature gating, and error behavior obvious at the call site, and add
  focused tests or doctests that would fail if one generated wrapper stops
  forwarding correctly.

## Work Logs And Design Records

- Nontrivial refactors, cleanup streams, AI-assisted implementation, and PRs
  that make explicit design tradeoffs must leave a curated work log under
  `docs/worklogs/`. The work log should record the session summary, code and
  documents read, reference implementations considered, decisions made,
  alternatives rejected or deferred, verification performed, and remaining
  risks.
- Work logs are not raw transcripts and are not implementation plans. They are
  reviewer-facing decision records for the completed work. Keep them concise
  enough to review, but specific enough that a later reviewer can understand
  why an abstraction, split, macro, descriptor, public API choice, or deferral
  was selected.
- PR bodies for work that requires a work log must link the relevant
  `docs/worklogs/` file. Reviewers should read linked work logs before
  challenging scope, abstraction choices, or design intent.
- When a PR establishes or changes durable design intent, update the
  appropriate document under `docs/design/` in the same PR. Use work logs for
  session-level rationale and design docs for decisions future implementation
  and review should continue to follow.
- When a bug report or audit finding is a false positive because of an
  intentional invariant, record the evidence in the issue or PR ledger and add
  a nearby `// INVARIANT:` source comment (see Invariant Markers), rustdoc
  note, or source-contract test when that
  invariant is not obvious from the code. Do not just skip the finding; leave
  enough context that later humans and AI agents do not rediscover the same
  non-bug as suspicious.
- Before adding a new audit or repository rule, inventory nearby existing rules
  and merge, tighten, or relocate overlapping guidance when possible. Prefer
  one sharper general rule over many narrow bullets that future agents must
  reconcile.

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
  [Performance-Sensitive Safety Contracts](#performance-sensitive-safety-contracts),
  [Materialization And Copies](#materialization-and-copies),
  [Performance-Gated Experiment Protocol](#performance-gated-experiment-protocol),
  [Cache Ownership](#cache-ownership),
  [CPU Threading Contract](#cpu-threading-contract),
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

- Pull request creation is currently restricted to collaborators. External
  requests, reproducers, proposed regression tests, benchmark reports, and
  prototype branches should be collected in issues first.
- Bug-fix PRs from collaborators may be reviewed as merge candidates when they
  fix behavior that is already intended by current docs, specs, or tests.
- New features must start as feature request issues. Do not treat a
  new-feature implementation PR as the source of truth unless maintainers
  already accepted the linked issue and agreed that implementation should
  start.
- New-feature PRs opened before an accepted issue may be closed and redirected
  to an issue. Prototype code should be linked from the issue as reference
  material.
- Accepted feature issues, specs, and repository rules are the source of truth
  for implementation. External prototype code may inform implementation only
  when its license and provenance are compatible with this repository.
- When maintainers take over a prototype branch, preserve the contributor's
  original commits where practical and add new commits on top. When maintainer
  or AI-assisted implementation is otherwise based on external prototype code,
  preserve appropriate copyright notices, license obligations, attribution, and
  links to the original prototype or issue discussion.

## CI Cost Discipline

Human/process protocol. This section is intentionally not routed to the
diff-scoped review bot.

- Expensive CI lanes, especially GPU or larger-runner jobs, must be gated behind
  cheaper repository-policy and non-GPU checks. Do not trigger hardware-backed
  runners directly on PR updates when an earlier review, lint, docs, or CPU test
  gate can reject the PR first.

## Standard Extension Boundary

- Standard operation families (`tenferro-einsum`, `tenferro-linalg`,
  `tenferro-fft`, and future peers) are first-class crates, not modules of a
  broad `tenferro` facade.
- The workspace intentionally has no root `tenferro` facade crate. Do not add
  operation-family facade paths such as `tenferro::einsum`,
  `tenferro::linalg`, or `tenferro::fft`; users import operation crates
  directly.
- Users import operation crates directly, bring their extension traits into
  scope, and register runtimes explicitly when graph execution reaches an
  extension family; for example the direct
  `tenferro_einsum::TraceContextEinsumExt` /
  `tenferro_einsum::TracedTensorEinsumExt` operation surfaces plus
  `Runtime::builder().install_extension_module(...)` after registering the
  target engine.
- Extension runtime dispatch must fail explicitly when a runtime owner is
  available but the extension family is not registered. Do not silently fall
  back from a registered-runtime execution path to an `ExtensionOp::eager_execute`
  implementation, CPU backend, reference path, or freshly constructed backend.
  Public eager wrappers for standard extensions must either register their
  runtime before dispatch, following the crate's established pattern, or expose
  the missing-runtime error. Add regression tests for missing-runtime behavior
  when changing extension dispatch.
- Hidden backend hooks for internal optimized operations, such as prepared
  solves, values-only decompositions, cache-aware kernels, or device-specific
  fast paths, must not silently fall back to a slower public operation, full
  decomposition, freshly constructed backend, CPU transfer, or reference
  implementation. The trait default should return an explicit unsupported or
  backend error, and each backend that supports the path must override the hook
  directly. Add source-contract or behavior tests when introducing such hooks.
- Optional capabilities are feature boundaries, not new operation-family
  crates. Put operation-specific AD support behind an `autodiff` feature in the
  owning operation crate instead of adding `*-ad` companion crates.
- User-facing operation crates expose backend features as `cuda` and `rocm`.
  Do not document or require a public `gpu` feature on those crates; use
  internal `cfg(any(feature = "cuda", feature = "rocm"))` checks or internal
  helper features when needed.
- Extension crates depend on the runtime, tensor, AD, or GPU crate they need;
  dependency flow must not require a facade crate to depend back on them.
- Extension AD rules should be owned by an explicit `tenferro_ad::AdContext`
  or rule set. Do not add process-global registration shims; a legacy bridge
  may exist only when explicitly approved by the task or maintainer decision.

## Oracle Gate

- Do not add or keep an AD rule implementation in the mainline without a
  corresponding oracle family.
- Prefer oracle families with both Torch reference data and finite-difference checks.
- If a Torch reference is not available, a finite-difference-only oracle is acceptable.
- If no corresponding oracle exists yet, add it to `tensor-ad-oracles` before treating the rule as a supported mainline AD rule.

## Rule Source Of Truth

- `SemanticProgram -> SemanticProgram` transforms in `tenferro-ad` are the
  current semantic AD boundary. `tenferro-internal-ops` still owns the
  primitive rule vocabulary used to implement those transforms.
- The semantic transforms use the validation-preserving program builder rather
  than exposing primitive graph keys or mutating a frozen program.
- Do not model tensor primitive AD by implementing
  `chainrules_core::ReverseRule<StdTensorOp>` or `ForwardRule<StdTensorOp>`;
  those traits are value/tape-level interfaces and are not the standard
  semantic transform surface in this repository.
- Avoid introducing ChainRules-style `frule`/`rrule` terminology for new
  tenferro AD APIs. Use `linearize` for JVP graph emission and
  `transpose_rule` for transposed linear graph emission. A future
  externally supplied pullback API must be designed separately instead of
  being mixed into the current primitive rule contract.
- Reverse-mode support is not always a direct `transpose_rule` arm on the
  primal op. Some ops are supported by first applying `linearize` and then
  transposing the emitted linear primitive graph. Before filing or closing an
  AD support issue, check the machine-readable AD support manifest in
  `tenferro-internal-ops/src/ad/support.rs`; `SupportedViaLinearize` means a missing
  direct `transpose_rule` arm is intentional.
- AD graph-emission rules must distinguish rank, exact extents, conservative
  extents, and runtime shape sources. Do not call exact-shape helpers when the
  rule only needs rank or can emit runtime `DimExpr::InputDim` references.
  Exact-shape requirements are appropriate only when constructing a concrete
  op payload that cannot represent runtime dimensions; handle non-exact
  metadata conservatively instead of reinterpreting bounds as sizes.
- AD graph-emission invariants must be enforced by API shape or structural
  tests, not prose alone. Constant, zero, one, and identity construction in AD
  rules should use the semantic helpers in `tenferro_ops::ad::support`; tests
  such as `identity_matrix_helper_emits_semantic_constant_and_remaps_shape_source`
  and `identity_matrix_fixed_uses_semantic_constant_not_analytic_shortcut`
  guard against analytic constant shortcuts reappearing.
- Reference JAX's implementations (`jax/_src/lax/lax.py`, `jax/_src/lax/linalg.py`)
  when implementing new AD rules.

## AD Rule Coverage

- Every `linearize` / `transpose_rule` implementation must have a corresponding
  finite-difference integration test that verifies numerical correctness.
- For linalg ops, prefer oracle families with both Torch reference data and
  finite-difference checks when available in `third_party/tensor-ad-oracles/`.
- **Numerical coverage is the real guarantee for AD rules, not line coverage.**
  An AD rule is "covered" when its differentiable outputs match a
  finite-difference and/or Torch oracle, and its per-output support status is
  asserted against the machine-readable manifest — not when llvm-cov reports a
  high line percentage. For linalg this is enforced by:
  - the finite-difference sweep in
    `crates/tenferro-linalg/tests/integration/traced_ad_explicit.rs` (cholesky, qr, eig,
    eigh, lu, full_piv_lu, solve, triangular_solve, svd/eig/eigh values),
  - the manifest assertions in
    `crates/tenferro-linalg/tests/integration/ad_support_manifest.rs` against
    `crates/tenferro-linalg/src/ad/support.rs`
    (`all_linalg_ad_support()` covers every dispatch arm and output status), and
  - the oracle support table in `docs/oracle/tensor-ad-oracles-support.md`.
- Consequently the `crates/tenferro-linalg/src/ad/rules/*.rs` files carry
  intentionally below-default per-file thresholds in `coverage-thresholds.json`.
  Their uncovered lines are dtype-guard arms (real→complex casts, integer-dtype
  early returns) and F32/error branches that the numerical oracles do not
  exercise. Do not "fix" the low line coverage by adding line-padding tests, and
  do not read it as missing AD validation. If you add or change a linalg AD rule,
  extend the finite-difference sweep and the manifest first; raise the threshold
  only if real new numerical tests warrant it.

## No Ad Hoc Fixes

- Do not add ad hoc fixes that violate DRY, KISS, or layering.
- Do not introduce compatibility shims, duplicated logic, or downstream reach-through into lower layers when the correct fix belongs in an existing seam or high-level API.

## Unsafe Code Boundary

`unsafe` is intentionally confined to FFI bindings and backend leaf code, and is
near-absent from the algorithmic layers. Reviewers must not read the raw count
as a red flag without first checking *where* it lives.

- **Count it correctly.** A plain `grep -c unsafe` over-counts ~30%+: it picks
  up `// SAFETY:` comments, doc comments, string literals, test code, and
  generated code under `target/`. Use `python3 scripts/count-unsafe.py` for the
  real production figure with a per-crate / per-category breakdown. As of this
  writing the real total is ~460, of which essentially all is FFI/backend:
  cuSOLVER/cuTENSOR/cuBLAS and LAPACK bindings, the batched raw-pointer setup
  that feeds those calls, SIMD elementwise kernels, thread affinity, and buffer
  pools (`tenferro-linalg`, `tenferro-cpu`, `tenferro-gpu`).
- **Keep the higher layers unsafe-free.** `tenferro-runtime`, `tenferro-einsum`,
  `tenferro-ad`, and the graph/AD logic in `tenferro-tensor` are ~zero `unsafe`
  by design. Do not introduce `unsafe` there; if a kernel or FFI call is needed,
  it belongs behind the backend/FFI seam, not in graph or rule code.
- **Document each block.** As in the slicing rules above, keep the validation
  invariant next to the `unsafe` block (a `// SAFETY:` note) and test the
  boundary conditions. New FFI binding files should follow the existing
  `cusolver.rs` / `lapack_linalg/*` patterns.

## File Organization

Keep source files small and focused, but do not split files solely to reduce
line count. Treat ~1000 lines as a soft review trigger, not as a mechanical
limit. A file that remains one coherent concern may stay above that size until
there is a clear behavior, abstraction, feature, or ownership boundary to
extract.

When splitting a large file, the split must make the code easier to reason
about. Prefer boundaries such as parsing, plan computation, execution,
dispatch, public API, AD rules, operation families, backend glue, validation,
or cache ownership. Avoid arbitrary `part1` / `part2` splits and avoid moving
code into tiny files that force readers to chase one concept across many
modules.

Use line count to decide where to inspect first. Use responsibility, change
frequency, public/private API boundaries, and human navigation to decide
whether and how to split.

## Unit Test Organization

For Rust modules, keep production source files focused on production code.
Do not keep inline `#[cfg(test)]` blocks in normal modules unless the file is a
genuinely tiny leaf module and the test is trivially small. Prefer
module-local test directories such as `src/<module>/tests/*.rs` and leave only
`#[cfg(test)] mod tests;` in the source file. Reserve crate-root `tests/` for
integration tests. Do not use `include!` to inject test files into modules.

When splitting tests, optimize for keeping AI and human reading context clean:
a developer reading `src/**` should not need to scroll through large unit-test
blocks to understand the implementation. Prefer splitting larger extracted test
suites by concern rather than keeping one monolithic test module.

Tests follow implementation ownership.

- Public facade crates should prefer integration tests for user-visible
  behavior.
- Private implementation details must be tested in the crate that owns the
  implementation, typically an internal crate, not through a public facade
  crate.
- If a crate sets `[lib] test = false`, do not add `src/**/tests`, inline
  `#[cfg(test)] mod tests`, or other crate-local unit-test entrypoints to that
  crate.
- If a private helper in a facade crate needs direct unit testing, move that
  helper into the owning internal crate instead of re-enabling facade-crate lib
  tests.
- This rule is enforced by repository contract tests and must stay green in CI.

## Performance-Sensitive Safety Contracts

- Potentially dangerous operations that are intentionally kept for performance,
  such as raw scratch-buffer acquisition, unchecked indexing after validation,
  raw pointer arithmetic, or backend-native view construction, must carry a
  nearby one-line comment explaining the invariant that makes the operation
  valid, using the `// INVARIANT:` marker (see Invariant Markers). This is
  required even when the operation is technically correct, so
  later agents and reviewers do not "fix" a false positive by adding hidden
  copies, repeated checks, or unconditional initialization to a hot path.
- Buffer-pool and scratch-buffer APIs must distinguish full-overwrite callers
  from read-before-write callers. Do not fix stale or uninitialized reads by
  adding unconditional zero-fill to shared hot-path acquisition. Instead expose
  an explicit zeroed/initialized acquisition path, keep raw acquisition unsafe
  and documented for full-overwrite kernels only, and add regression coverage
  for both contracts.
- While a backend still exposes a cloneable owner-like `Arc` buffer, an
  in-place write requires a local uniqueness proof such as `Arc::get_mut` or
  exclusive runtime ownership for the whole operation. This rule applies to
  that legacy owner representation, not to lifetime-only root retention.
  Under the final storage contract, an `Arc<RootResource>` may be shared by
  disjoint claims, read-only retained records, or retirement records; its
  strong count is neither write authority nor a reason to reject an otherwise
  valid write. Write authority comes only from provenance-carrying
  `StorageMut` derived from an exclusive Rust borrow (or a fresh output), with
  checked span and injectivity proofs. Add mutation-after-handoff and disjoint-
  claim tests where an optimization preserves aliases or views that a previous
  implementation copied.

## Complexity Budget

- Do not introduce accidental `O(n^2)` behavior in graph construction,
  metadata propagation, key hashing/equality, compilation, or execution
  scheduling. If a quadratic algorithm is intentional, document why the input
  size is bounded or why the tradeoff is acceptable with an `// INVARIANT:`
  marker (see Invariant Markers).
- Avoid repeatedly cloning, hashing, formatting, or scanning whole graph
  histories, metadata scope lists, tensor input maps, or structural keys inside
  per-node/per-op loops. Prefer stable IDs, interning, cached fingerprints with
  exact equality checks, or persistent/shared data structures.
- When optimizing compiler or graph-build overhead, measure scaling across
  increasing graph sizes, not only one fixed benchmark case.

## Materialization And Copies

- Production tensor paths must not silently materialize dense temporary buffers
  whose memory or time scales with an unconstrained product of tensor
  dimensions. Dense/reference implementations are allowed only when the API is
  explicitly named and documented as dense, reference, or debug behavior.
- Avoid dense copy-in/copy-out around operations that can consume strided
  views, borrowed slices, backend-native buffers, or metadata-only layout
  changes.
- Do not introduce hidden CPU-GPU transfers. Follow the public device-transfer
  policy: callers explicitly upload/download tensors, while execution pipeline
  internals may move constants or scalar metadata only where the backend
  contract documents that behavior.
- When a copy or materialization is required by an output contiguity contract,
  backend limitation, or external ABI boundary, make that boundary explicit in
  the implementation and cover it with tests.

## Device Transfer And Backend Buffer Errors

- tenferro follows the PyTorch convention: no implicit CPU-GPU transfer.
  Tensors must already live on the device required by the backend operation.
- Canonicalization is allowed only within the existing placement. Host views may
  copy into host compact tensors; GPU backend views may canonicalize or copy
  back on the GPU. These boundaries must not download or upload tensor payloads
  unless the API is explicitly a transfer API.
- User code must explicitly upload CPU tensors before CUDA backend execution
  and explicitly download CUDA tensors before CPU-only execution or host value
  inspection.
- A GPU backend op receiving a CPU tensor must return a runtime-state error
  with a diagnostic that says the op expected a GPU tensor and points users to
  `upload_tensor()`.
- A `Result`-returning CPU backend op receiving a backend/GPU buffer must
  return a runtime-state error where the buffer is detected at the CPU backend
  boundary. The diagnostic should say to download the tensor to host before
  CPU execution.
- Direct host-inspection APIs such as `TypedTensor::host_data()` and
  `TypedTensor::host_data_mut()` return `Result` and must report backend
  buffers as runtime-state failures. Any infallible host-inspection API that
  returns a borrowed slice instead of `Result` must document an explicit panic
  boundary before it is exposed.
- Execution pipeline internals may handle placement for documented cases:
  `Constant` ops may auto-upload through `upload_host_tensor()`, and
  host-dependent ops such as `ShapeOf` or `DynamicTruncate` may read metadata
  or download single scalar values as documented by the backend contract.
- Unsupported CUDA op or dtype combinations must return an explicit error, not
  silently fall back to CPU execution.

## Dense Layout And Linear Algebra

- tenferro uses column-major (Fortran order) dense storage: the leftmost
  dimension has the smallest stride and varies fastest in memory.
- Owned runtime tensors remain compact column-major only. Arbitrary strides,
  offsets, transposes, slices, and reverse views live on
  `TypedTensorView`/`TypedTensorViewMut` or metadata-only layout values until an
  explicit same-placement canonicalization boundary is reached.
- Public flat-buffer constructors, exports, examples, FFI contracts, and docs
  must state or preserve column-major semantics.
- Do not add row-major compatibility shims or hidden row-major round-trips in
  library code. If external data is naturally row-shaped, convert privately at
  that explicit boundary.
- For batched GEMM-style layouts, put contraction/compute dimensions on the
  left and batch dimensions on the right. In column-major storage, this keeps
  each batch slice contiguous for the GEMM kernel.
- Use existing tensor/backend abstractions for GEMM, einsum, and dense linear
  algebra. Do not reimplement linalg kernels locally in downstream layers when
  the CPU/GPU backend already owns the operation.

## Range Checks And Slicing

- Public indexing and slicing APIs must validate rank, bounds, steps, output
  shape, and empty/singleton boundary behavior at the API boundary or planning
  boundary.
- Tensor layout metadata and runtime typed views may use signed strides and
  negative slice steps when reachable-range validation proves every logical
  element maps inside the backing allocation. Zero step remains invalid. Do not
  reject negative strides solely because they are negative. Narrower adapter
  APIs may document stricter compatibility limits, but those limits must be
  explicit at the API boundary.
- After validation, hot loops and kernels should not repeat the same range
  checks per element. Carry validated shape/stride/offset metadata into the
  inner implementation instead.
- Prefer safe Rust patterns that help LLVM eliminate bounds checks: iterate
  over slices directly, slice once before the loop, use `chunks_exact` when the
  chunk size divides the length, and add explicit pre-loop assertions for
  validated index ranges. Use unchecked indexing only as a last resort.
- Prefer metadata-only slices, strided views, or backend-native slice kernels
  over dense copies. If a slice must allocate, document and test the reason.
- Slice, reshape, transpose, gather, scatter, pad, concatenate, and reverse
  implementations must preserve column-major shape/stride/offset semantics.
- If unchecked indexing or unsafe pointer access is used after validation, keep
  the validation invariant close to the unsafe block and add tests for full
  range, empty or singleton slices, lower and upper boundaries, out-of-range
  errors, rank mismatch, and non-contiguous slices.

## CPU Kernel Implementation

- No naive CPU loop fallbacks. CPU tensor kernels must use optimized
  implementations unless the operation is explicitly listed as an exception
  below.
- CPU affine-strided copy, permutation, broadcast, map, zip-map, fused
  elementwise replay, sum/product reductions, gather, additive scatter, and
  fixed-window dynamic slice/update delegate to `strided-rs` when their
  semantics fit an existing strided plan. Tenferro owns tensor semantics,
  validation, dtype dispatch, placement checks, error translation, execution
  contexts, and reusable output storage; it does not duplicate the tensor-sized
  traversal.
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

- Ownership priority is lower-layer first: when a CPU operation can be
  expressed as metadata preparation followed by an existing `strided-rs`
  primitive, tenferro must delegate the bulk traversal. If a generally useful
  primitive is missing, add it to `strided-rs` first and then consume it from
  tenferro. A tenferro-owned traversal is allowed only for semantics outside
  the affine-strided primitive model or for an explicitly approved,
  benchmark-backed exception. Einsum is the benchmark-backed tenferro exception;
  new exceptions require an accepted issue, comparative benchmark evidence,
  and a recorded ownership rationale.
- Calling a `strided-rs` primitive is necessary but not sufficient. Public
  tensor-sized CPU work must enter through `CpuBackend` and its configured
  execution scope. High-performance materialization needs the backend's
  persistent `BufferPool`, fully-overwritten uninitialized output allocation,
  configured `CpuContext` Rayon pool, nested-execution safety, and
  serial/parallel threshold policy. A context-free `strided-rs` call, a
  throwaway pool, or Rayon's ambient global Rayon pool is non-compliant.
  Memory reuse and thread policy are execution resources, not tensor metadata;
  backend-neutral tensor/view types therefore expose metadata-only layout
  transforms and do not own data-moving convenience methods.
- Exceptions with dedicated implementations are `reshape` (metadata-only),
  `embed_diagonal`, index-dependent triangular masks (`tril`/`triu`), max/min
  reductions until strided exposes matching NaN semantics, and indexing ops
  without a matching strided plan such as static slice, pad, concatenate, and
  reverse.
- CPU provider features are additive. At least one of `cpu-faer` or `cpu-blas`
  must be enabled; enabling both is valid and must compile. `CpuBackend` owns
  the runtime provider selection. `CpuBackend::new()` selects the default
  compiled provider (BLAS/LAPACK when `cpu-blas` is compiled, otherwise
  `cpu-faer`), and explicit constructors or application configuration select a
  different compiled provider when needed.
- Tensor-sized CPU kernels must run through the repository CPU threading
  policy. `strided-kernel` must be compiled with its `parallel` feature for
  elementwise, reduction, and structural materialization kernels. CPU
  contraction providers must receive their policy from the owning
  `CpuExecutionContext`; Faer uses `Par::Seq` for one-thread contexts and
  explicit `Par::rayon(n)` for bounded multi-thread contexts.
- If a tensor-sized CPU operation remains a dedicated sequential loop because
  no strided-kernel/backend-native parallel primitive fits the indexing pattern
  yet, keep a nearby source comment naming that rationale. Do not let
  undocumented serial loops become the default fallback pattern.

## Tensor Core Data Model

- `tenferro-tensor-core` owns backend-independent host tensor metadata and
  contiguous host storage: `DType`, `TensorScalar`, `HostTensor<T>`,
  dynamic `Tensor`, host/dynamic views, `TensorRef`, `ShapeVec`, `StrideVec`,
  `SliceSpec`, and metadata-only `reshape_view`, `transpose_view`, and
  `slice_view`.
- `tenferro-tensor-core` must not depend on CUDA, GPU backends, backend buffers,
  provider selection, or execution backend traits. Its owned `Tensor` must not
  grow inherent `TensorBackend` execution helpers.
- Core views and layouts must validate bounds eagerly using checked arithmetic.
  `TensorLayout` metadata views may use signed strides and negative slice steps
  when reachable-range validation succeeds; zero step remains invalid. Do not
  implement `PartialEq` for views.
- `ShapeVec` and `StrideVec` use `SmallVec` with inline rank capacity 8.

## Faer Integration

- Prefer zero-copy `faer::MatRef` / `faer::MatMut` views over packing data into
  temporary dense matrices. Validate shape, bounds, alignment, and aliasing
  before constructing unsafe raw faer views.
- Feed faer column-major-friendly layouts whenever possible. For faer dense
  linalg, row stride `1` is the preferred contiguous layout; generic-stride
  inputs that trigger faer performance warnings must be justified or converted
  deliberately at an explicit boundary.
- Pass faer parallelism through `CpuContext` only. Do not choose `Par::Rayon`
  or thread counts independently inside operation helpers.
- For linalg scratch space, compute scratch requirements before execution and
  allocate reusable scratch once for the operation. Avoid repeated `Vec`
  allocation in decomposition, solve, or batched inner loops.

## Performance Anti-Patterns

- Do not hand-copy near-identical dtype-specific operation bodies. Prefer
  generic helpers, sealed traits, or macros, and keep any unavoidable dtype
  dispatch isolated at the outer boundary.
- Do not allocate dense buffers when strided or backend-native access is
  available.
- Do not zero-initialize buffers that will be fully overwritten.
- Avoid per-element index multiplication in hot loops; use incremental pointer
  offsets or precomputed strides.
- Do not allocate `Vec` or other heap buffers inside hot loops. Pre-allocate
  and reuse scratch buffers.
- Do not call `Backend::plan()` or equivalent planning APIs inside execution
  loops. Pre-compute plans before the loop and pass them in.

## Performance-Sensitive Tests And Benchmarks

- Small reference tests may materialize dense tensors, but should materialize
  each full result once and compare the whole result. Avoid per-element
  re-contraction, re-evaluation, or repeated graph execution as the comparison
  mechanism.
- Long regression tests should be sized so accidental dense materialization,
  accidental `O(n^2)` graph work, or unexpected copies fail quickly while the
  intended algorithm remains cheap.
- For approximate equality, report a useful residual such as an absolute max
  error or relative norm error so performance-related failures remain
  diagnosable.
- Use release-mode benchmarks for performance claims. Prefer Criterion-style
  benchmarks for microbenchmarks, wrap benchmark inputs and outputs with
  `std::hint::black_box` where needed, and pin relevant thread counts when
  comparing CPU behavior.
- Benchmark scaling across representative tensor sizes, shapes, layouts, and
  thread counts. A single fixed-size speedup is not enough evidence for a
  performance-sensitive change.

## Performance-Gated Experiment Protocol

Human/process protocol. This section is intentionally not routed to the
diff-scoped review bot.

- Performance candidates found by static/source audit alone must pass a
  need-before-implementation gate before code changes start. First measure the
  candidate path's share of an end-to-end workload, or another
  issue-predeclared representative workload. If the path does not occupy a
  meaningful share under the predeclared threshold, record the measurement or
  argument in the issue and close or defer it without implementation. A
  microbenchmark of the targeted helper is useful after the need is established,
  but by itself it proves only effect, not that the optimization is worth its
  semantic, aliasing, cache, or maintenance risk.
- Before running a candidate, record the baseline commit, candidate commit,
  benchmark source, build profile, hardware and affinity configuration,
  provider/thread settings, complete case list, comparison statistic,
  acceptance threshold, repetition policy, and host-noise observables and
  thresholds. Candidate results must not influence these choices.
- Run the complete baseline/candidate suite as one paired experiment. Do not
  selectively retry, omit, replace, or promote individual favorable cases.
- If a predeclared host-noise or validity gate fails, classify the entire paired
  experiment as `INCONCLUSIVE`. Reconsideration requires a complete paired
  rerun under the same predeclared protocol, not a selective retry.
- Record every measured case, confidence interval, validity observation, and
  regression in the worklog. A negative or inconclusive primary result is
  evidence and must not be rewritten as success because secondary cases
  improved.
- Promote a performance-gated change only when its predeclared primary gate and
  all required non-regression/correctness gates pass. Do not relax thresholds,
  redefine the primary metric, or add post-hoc exclusions after seeing the
  candidate.

## Cache Ownership

- Long-lived runtime/compiler caches must be owned by `Engine` or another
  explicit top-level runtime object, not hidden in thread-local/global state or
  buried in backend internals.
- Every cache must have a bounded default, a user-facing way to configure that
  bound, and a user-facing way to clear it.
- Every cache must expose user-facing introspection for the number of retained
  entries and retained bytes. Report retained bytes as the cache's owned/logical
  payload estimate, not operating-system RSS or allocator arena usage.
- Top-level runtime objects that own multiple caches must provide aggregate
  clear and aggregate stats APIs in addition to cache-specific controls.
- Backend resource pools such as buffer pools may live on the backend, but they
  still need explicit limit/clear controls, stats APIs, and documentation.
- Backends that own resource pools or runtime contexts, including
  `CpuBackend`'s buffer pool and `Arc<CpuContext>` and CUDA backends' runtime
  client/context, are construct-once-and-reuse values. Examples should bind the
  backend once and reuse it across related operations; they must not present
  per-call `CpuBackend::new()` or GPU backend construction chained directly
  into an operation as the normal idiom. A genuinely standalone single-op
  example does not need to invent an unrelated long-lived backend variable.
- Do not add a new cache without documenting its owner, lifetime, default
  capacity, memory behavior, entry/byte accounting, and
  clear/configuration/stats path.

## CPU Threading Contract

- For faer-backed CPU ops, `CpuContext` is the single source of truth for thread-pool policy.
- Do not derive faer parallelism independently inside individual ops or helper functions.
- Execute faer-backed work only inside `ctx.install(...)` so the owned rayon context is preserved.
- Use `Par::Seq` for one-thread contexts and explicit `Par::rayon(n)` from the
  configured `CpuContext` degree for multi-thread contexts. Do not derive the
  policy from an ambient Rayon pool during plan or session setup.
- Tensor-sized strided CPU kernels that are not provider-owned must also run
  inside `CpuContext::install(...)`, so Rayon-backed `strided-kernel` work uses
  the backend's owned pool. BLAS/LAPACK provider-owned threading remains
  controlled by provider variables such as `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`.

## GPU Backend Contract

- Before touching CubeCL/GPU backend code, read
  [`docs/design/gpu-backend-design.md`](docs/design/gpu-backend-design.md).
- That document is the developer-facing source for CubeCL kernel ownership,
  runtime shape/stride metadata conventions, launch configuration rules, and
  device transfer behavior. Any change to those conventions must update that
  document in the same PR.
- CUDA operation paths whose accepted implementation is an NVIDIA vendor
  library must fail with typed load or provider errors when the required
  NVIDIA library is unavailable or lacks support. They must not silently fall
  back to native CubeCL kernels. Native CubeCL kernels are separate provider
  implementations for operations or dtypes not covered by the accepted
  vendor-library path, not fallback tiers for missing CUDA libraries.
- CUDA `eig` (non-symmetric eigenvalue decomposition, LAPACK `dgeev`) is not
  provided by cuSOLVER. `CudaBackend::eig` returns `Unsupported`; users
  must explicitly download to CPU and compute via `CpuBackend::eig`.

## Structured Error Classification

- Validation facts use the shared typed vocabulary for shape, rank, axis, dtype,
  configuration, and invalid arguments. Eager and traced paths must report the
  same validation kind and payload for the same known input; `ErrorPhase` is a
  separate graph-build, compile, or execution axis.
- Unsupported operations and operation-specific unsupported dtypes use the
  `Unsupported` category. A crate that owns a richer reason retains a typed
  local source; `UnsupportedDTypeConversion` is reserved for an actual
  from-dtype to to-dtype conversion.
- Singularities, non-convergence, division by zero, and other numeric-domain
  failures use `NumericalFailure` and retain the owning crate's typed source.
- Typed backend/kernel errors use a source-preserving backend wrapper. The
  text-only `BackendFailure` category is reserved for vendor/backend status
  text when no typed source or more specific category exists.
- Typed file, stream, serialization, and dynamic-library failures use `Io` and
  retain their source. Missing, uninitialized, poisoned, or otherwise invalid
  executor, cache, device, and buffer state uses `RuntimeState`, retaining a
  typed source when one exists. Impossible internal invariants remain
  `Internal`; these categories must not be used as catch-alls for known input
  validation.
- Public error conversion must preserve the `source()` chain across crate
  boundaries. Converting a typed error to `String` is permitted only for
  display/logging, vendor/FFI arguments, final serialization, or an explicit
  message-only external protocol boundary.

## Documentation Policy

### Source of Truth

- **Source code** is the source of truth for internal design (op catalog, backend contract, AD rules, compilation pipeline).
- **Online docs** are primarily user-facing — how to use the explicit runtime,
  AD, tensor, GPU, and standard operation crates.
- The online **Internals section** is the exception for implementation-oriented
  readers: it may publish curated architecture, specification, and active
  design notes when those pages are linked from rendered docs. Historical
  `docs/plans/` records stay offline unless a maintainer explicitly decides
  otherwise.
- **AGENTS.md** is the entry point for developers and AI agents. It contains pointers to source code locations.
- Do NOT duplicate source-code-level information in online docs. If it can be learned by reading the source, put a pointer instead of a copy.
- Development assumes AI agentic coding. Keep machine-readable sources (code + doc comments) authoritative.

### User-Facing Docs

- User docs target PyTorch/JAX users who interact with public direct crates
  such as `tenferro-runtime`, `tenferro-ad`, `tenferro-gpu`,
  `tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft`.
- Imports must use public direct crates such as `tenferro_runtime`,
  `tenferro_ad`, `tenferro_gpu`, and standard extension crates. Do not
  reference internal crates (`tenferro-internal-ops`, `computegraph`, etc.) in
  user-facing docs.
- Do NOT expose internal jargon (Graph, StableHLO, ExecOp, ValueRef, etc.) in user-facing pages.
- Provide PyTorch/JAX equivalents when introducing tenferro concepts.

### User-Facing Code Snippet Consistency

- Non-trivial user-facing code snippets must have an executable source of truth:
  prefer including a checked example, test, or doctest instead of copying code
  into Markdown by hand.
- If a Markdown page must contain a copied snippet, add an automated sync or
  extraction check that fails when the snippet drifts from the executable
  source.
- Guide code that demonstrates a workflow should compile in CI. When runtime
  execution requires special hardware or external libraries, CI must still
  compile-check the example with the required feature flags, and the guide must
  document the command that runs the example on a correctly configured machine.
- CPU and GPU workflow examples should include meaningful assertions on shapes,
  dtypes, or values whenever the result is deterministic. Avoid examples that
  only print output unless the output itself is the documented behavior.
- GPU quickstart examples must use the same executable source as the guide,
  explicitly show upload/download boundaries, and assert downloaded CPU results
  for at least one supported operation.

### Diagram Consistency

- Diagrams in docs and `README.md` (SVG under `docs/assets/`, ASCII diagrams
  in design, architecture, and spec pages) are part of the documented surface:
  crate names, layer assignments, dependency directions, and public entry
  points shown in a diagram must match the current implementation.
- A PR that adds, removes, renames, or re-layers a crate, or changes an
  extension or backend boundary shown in a diagram, must update the affected
  diagrams in the same PR.
- Stale diagrams are worse than missing diagrams. Delete a diagram rather than
  leave it inaccurate.
- Diagrams follow the same source-of-truth rule as text: show boundaries and
  dependencies, not internal implementation detail that the source code owns.

### Doc Examples

- Every public type, trait, and function must include minimal but sufficient
  usage examples in its doc comments (`/// # Examples`). `#[doc(hidden)]`
  items are exempt.
- Doc examples (`/// # Examples`) must NOT use `ignore` or `no_run` attributes.
- Every example must compile AND run as a doctest.
- An example must demonstrate real usage. Examples consisting only of path or
  assignment statements (for example `let _method = Type::method;`) satisfy
  the doctest gate without documenting anything and are not acceptable.
- Use `compile_fail` only for examples that intentionally demonstrate compile errors.
- If an example cannot run as a doctest, refactor it until it can.
- When a canonical public API changes from infallible or panicking to
  `Result`-returning, update its user-facing examples, tutorials, and guides to
  propagate recoverable errors with `?` or handle the documented error
  explicitly. Do not preserve the old call shape by appending `unwrap()` or
  `expect()`. Tests may still use an intentional assertion boundary, and an
  example may unwrap only a locally proven invariant whose proof is explicit.
- Examples that call CPU or GPU backend operations should bind the backend to a
  local variable and reuse it for related operations instead of chaining
  `Backend::new().op(...)` beyond a single trivial construction example.

### Public Result Error Documentation Gate

- Every public function, inherent method, and public-trait method returning a
  `Result` must document a `# Errors` section. The section must name the
  concrete error variants or failure conditions the caller can observe; a
  generic “returns an error on failure” sentence is insufficient.
- Documentation for traced or symbolic APIs must describe validation deferred
  to compile or execution and identify the applicable `ErrorPhase` when that
  behavior is part of the contract. Intentional panics use `# Panics`; deferred
  symbolic checks use `# Deferred errors`.
- `scripts/check-public-error-docs.py` audits the full Rust workspace and the
  Rust files changed by a PR and is a required CI gate. For traced APIs, prose
  that promises validation at compile/execution or after input binding must
  also be under `# Deferred errors`. Keep the audit enabled without `clippy`
  or source-level allowlists; add the concrete documentation at the API source
  instead.

## PR Content Hygiene

- Do not include AI-generated analysis, task, or verification reports as
  standalone files in PRs. Durable session records belong in
  `docs/worklogs/`; everything else stays out of the repository.
- Do not commit new top-level directories or dot-directories without an
  explicit maintainer decision recorded in the PR.
- User-facing guides must not pin commit hashes in setup instructions
  (for example `git checkout <hash>`); reference branches, tags, or released
  versions instead.

## Generic Over Scalar Type

- Use generic constructors with sealed traits instead of per-type functions.
- Bad: `TracedTensor::from_f64(...)`, `TracedTensor::from_f32(...)`, etc.
- Good: `TracedTensor::new<T: TensorScalar>(shape, data)` — type inference selects the variant.
- For dtype-polymorphic tensor operations, prefer one typed generic
  implementation plus outer dtype dispatch over per-dtype copy-pasted
  implementations.
- If Rust generics cannot express the shared structure cleanly, use a local
  macro to generate the repetitive dispatch or variant plumbing.
- Sealed traits (`TensorScalar`, `PoolScalar`, etc.) restrict APIs to the
  currently supported dtype set.

## Public API Convention

- **AD core ops**: `EagerTensor` and `TracedTensor` use methods as the
  canonical surface for single-output operations (`x.exp()`, `x.reshape(shape)`,
  `a.dot_general(&b, config)`). Use operator overloads where they read
  naturally (`&a + &b`, `&a * &b`) and associated functions for core operations
  with no natural receiver (`EagerTensor::where_select(...)`,
  `TracedTensor::concatenate(...)`).
- **Non-AD concrete ops**: `Tensor` and dynamic-rank `TypedTensor<T>` use
  crate-root extension-trait methods with an explicit backend (`TensorOpsExt`,
  `TypedTensorOpsExt`, and `TypedTensorMaskOpsExt`). The implementation may use
  private helper modules, but public `tensor` / `typed_tensor` module free
  functions are not part of the release API.
- **Extension families**: extension crates cannot add inherent methods to
  external tensor types, so their canonical tensor-facing surface is extension
  traits (`TracedTensorLinalgExt`, `EagerEinsumExt`,
  `TraceContextEinsumExt`, `TracedTensorEinsumExt`,
  `TracedTensorFftExt`) re-exported at the crate root. Do not expose public
  `traced_tensor` / `eager_tensor` module free functions for standard
  operation families.
- **No compatibility shims for operation-surface style changes**: when API
  compatibility is not explicitly required, remove old module functions instead
  of keeping wrappers beside the canonical method/associated-function or
  extension-trait surface.
- No `traced_` prefix on methods. `TracedTensor` methods are inherently traced.

### Output And Write Surface Vocabulary

- Operation suffixes describe ownership and output-update semantics. Do not
  use a suffix only to avoid a naming conflict.
- Unsuffixed operation names allocate and return a fresh result tensor.
- `_read` means one or more inputs are `TensorRead`/borrowed views; the output
  is still backend-allocated unless another suffix also names an output.
- Bare `_into` means overwrite a caller-provided output. The operation must
  validate dtype and shape, must not resize the destination, and must not read
  the previous output value as part of the semantic update.
- `_read_into` is the main backend hook shape for borrowed inputs plus an
  overwritten `TensorWrite` output.
- `_in_place` means destructive update of an input tensor and is distinct from
  `_into`. Do not implement in-place elementwise variants by passing aliased
  input/output views into generic `*_into` kernels unless the kernel's aliasing
  contract has been proven and tested.
- `_add_to` means read-modify-write accumulation, `out += op(...)`. Do not hide
  accumulation behind a bare `_into` method.
- Dot/GEMM read-modify-write uses the existing `_into_accum` vocabulary with a
  `DotGeneralAccumulation` argument. This is the dot-specific accumulation
  surface; do not introduce `_linear_into` as a second spelling.
- Typed-face preallocated methods may take `&mut TypedTensor<T>` when the dtype
  is statically known. Erased-face methods should take `TensorWrite<'_>` so
  dynamic dtype and view outputs are validated at the backend boundary.
