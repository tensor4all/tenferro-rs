# Issue 1256 AD VJP Default Pipeline

## Session Summary

Started the #1256 implementation by making traced VJP prefer
`linearize + linear_transpose` over direct primary-graph transpose, while
retaining the primary transpose walker as the fallback for rules that cannot
yet flow through the generic linear graph path.

The same branch also added targeted active-output pruning for linalg
decomposition linearizers where the active-output metadata already existed but
was not fully used: `Svd` now skips the vector tangent chain when only singular
values are consumed, and `Lu` now emits only requested factor tangent branches.

Follow-up work extended that active-output pruning to `Eigh`, `Eig`, and `Qr`.
`Eigh` now skips the eigenvalue tangent when only eigenvectors are consumed,
`Eig` now skips eigenvalue-only tangent work when the eigenvalue output is
inactive, and all three rules return no tangent graph when all outputs are
inactive. `Qr` now returns only the requested factor tangent and avoids the
final inactive branch emission.

The final slice added a traced AD transform optimizer, aval-carrying symbolic
zero instantiation, and regression coverage for zero propagation,
canonicalization, multi-output pruning, and cotangent accumulation. The
optimizer runs before materialization on traced JVP outputs and traced generic
VJP transpose outputs. It does reachable-output DCE plus local algebraic
canonicalization for AD-heavy identity patterns, leaving backend/layout
rewrites to the runtime compiler.

## Context Read

- Issue #1256 and its acceptance criteria around VJP generation, active output
  pruning, graph-size regression checks, cache ownership, and linalg-heavy AD
  cases.
- Shared tensor4all common, Rust, performance, numerical, docs, and test rules.
- `REPOSITORY_RULES.md`, especially the AD source-of-truth and oracle coverage
  requirements.
- `docs/architecture/ad-pipeline.md` for the intended
  `linearize -> linear_transpose -> materialize_merge` model.
- Existing traced AD implementation in `crates/tenferro-ad/src/traced.rs` and
  primary transpose fallback in `traced/primal_transpose.rs`.
- Existing linalg AD rules in `crates/tenferro-linalg/src/ad/rules/mod.rs`,
  including the prior `Eigh` active-output pruning pattern.
- `crates/tenferro-ad/src/traced/optimizer.rs` for the materialize-pre traced
  AD graph optimizer added in this work.
- `crates/tenferro-internal-ops/src/ad/zeros.rs` for symbolic zero
  instantiation helpers.

## Decisions

- Prefer the generic linearized VJP path first. If `linearize` reports the
  target as inactive, return `None`; if either `linearize` or
  `linear_transpose` fails, try the primary transpose fallback and report the
  original generic-path error only when the fallback cannot produce a usable
  cotangent graph.
- Keep direct primary transpose rules as an escape hatch rather than deleting
  them. Several extension rules use primal outputs directly and this preserves
  compatibility while the generic path becomes the default.
- Use the existing `linearize_active_value_keys` analysis as the explicit
  used-output pruning pass for traced AD. Linalg multi-output rules now check
  `ctx.is_value_active_in_linearize` before emitting expensive tangent branches.
- Keep QR/Eigh/Eig multi-output pruning at the rule-emission level. The
  existing active-output metadata is enough to avoid the known dead
  decomposition branches before any graph optimizer pass runs.
- Add a small traced AD graph optimizer for backend-independent cleanup after
  transform graph construction. It is intentionally stateless and
  metadata-only: DCE, double-neg/conj cancellation, identity convert/transpose,
  scalar add-zero, and scalar mul-one. Runtime compiler passes remain
  responsible for layout, dot, backend, and execution-IR cleanup.
- Carry zero abstract values at forced-instantiation boundaries with
  `SymbolicZero { dtype, rank, anchor }`. The tidu API still uses `None` for
  absent tangent/cotangent flow; tenferro instantiates zeros only when a
  primitive needs an actual zero input.
- Do not add a persistent traced AD optimizer cache. The optimizer has no
  current partial-result cache, uses only per-invocation scratch maps, and is
  covered by existing cache-owner documentation. The eager Tier-1 AD transform
  cache remains owned by `EagerRuntime`.

## Verification

- Red/green test for VJP path ordering:
  `cargo test -p tenferro-ad --test extension_op traced_vjp_prefers_linearize_transpose_over_primary_transpose -- --nocapture`
- Red/green linalg active-output pruning tests:
  `cargo test -p tenferro-linalg --features autodiff prune -- --nocapture`
- QR/Eigh/Eig follow-up RED/GREEN tests:
  `cargo test -p tenferro-linalg --features autodiff "linearize_prunes" -- --nocapture`
  `cargo test -p tenferro-linalg --features autodiff eig_linearize_prunes_unsupported_inactive_eigenvalue_output -- --nocapture`
  `cargo test -p tenferro-linalg --features autodiff one_input_linalg_jvps_prune_when_all_outputs_are_inactive -- --nocapture`
- AD graph optimizer RED/GREEN tests:
  `cargo test -p tenferro-ad optimizer_canonicalizes_ad_identity_chains -- --nocapture`
  `cargo test -p tenferro-ad --test ad_optimizer -- --nocapture`
- Symbolic zero RED/GREEN tests:
  `cargo test -p tenferro-internal-ops symbolic_zero_carries_aval_until_instantiated -- --nocapture`
  `cargo test -p tenferro-internal-ops ad::tests:: -- --nocapture`
- Primary-transpose fallback coverage after generic VJP became the default:
  `cargo test -p tenferro-ad --test extension_op traced_vjp_ -- --nocapture`
- Full touched-crate and linalg AD checks:
  `cargo test -p tenferro-ad`
  `cargo test -p tenferro-linalg --features autodiff`
- Repository checklist checks:
  `cargo fmt --all --check`
  `git diff --check`
  `cargo clippy --workspace --all-targets -- -D warnings`
  `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
  `cargo test --workspace --release`
  `cargo llvm-cov --workspace --release --json --output-path coverage.json`
  `python3 scripts/check-coverage.py coverage.json`
  `cargo doc --workspace --no-deps`
  `python3 scripts/check-docs-site.py`
  `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --worktree --output-json /tmp/repository-rules-review-worktree.json`

## Residual Risks

- The traced AD graph optimizer is intentionally conservative. It folds only
  identities that do not need shape reasoning beyond scalar constant facts and
  local unary provenance. Broader algebraic rewrites should be added with
  explicit metadata legality checks.
- The primary transpose fallback remains necessary for extension rules whose
  generic linearized path is incomplete.
- No persistent traced partial-result AD optimizer cache is introduced because
  the current pass is stateless and linear in the reachable transform graph.
  Future cache work must stay under one explicit owner and use structure and
  metadata keys only.
