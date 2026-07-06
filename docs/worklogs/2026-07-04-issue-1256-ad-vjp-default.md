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
- Keep the QR/Eigh/Eig follow-up at the rule-emission level rather than adding a
  new whole-graph optimizer pass. The existing active-output metadata is enough
  to avoid the known dead decomposition branches without changing the broader
  AD transform contract.
- Document cache ownership as unchanged: no persistent AD optimizer cache is
  introduced here. Transformed graphs live with the returned traced tensor and
  existing compiler/runtime/extension caches keep their current owners.

## Verification

- Red/green test for VJP path ordering:
  `cargo test -p tenferro-ad --test extension_op traced_vjp_prefers_linearize_transpose_over_primary_transpose -- --nocapture`
- Red/green linalg active-output pruning tests:
  `cargo test -p tenferro-linalg --features autodiff prune -- --nocapture`
- QR/Eigh/Eig follow-up RED/GREEN tests:
  `cargo test -p tenferro-linalg --features autodiff "linearize_prunes" -- --nocapture`
  `cargo test -p tenferro-linalg --features autodiff eig_linearize_prunes_unsupported_inactive_eigenvalue_output -- --nocapture`
  `cargo test -p tenferro-linalg --features autodiff one_input_linalg_jvps_prune_when_all_outputs_are_inactive -- --nocapture`
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

- This is an initial #1256 slice. It makes generic VJP the default and prunes
  high-impact decomposition linearizers, but it does not add a broad AD
  algebraic canonicalizer, aval-carrying symbolic zero type, or persistent AD
  graph optimizer cache.
- The primary transpose fallback remains necessary for extension rules whose
  generic linearized path is incomplete.
