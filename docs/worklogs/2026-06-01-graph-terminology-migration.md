# Graph terminology migration work log

Companion PRs:

- computegraph-rs: <https://github.com/tensor4all/computegraph-rs/pull/4>
- tidu-rs: <https://github.com/tensor4all/tidu-rs/pull/31>

Date: 2026-06-01

## Session summary

This change updates tenferro to the breaking graph terminology API merged in
`computegraph-rs` and `tidu-rs`.

The companion upstream PRs renamed the old graph-building vocabulary to the
current API:

- `Fragment` / `FragmentBuilder` became `Graph` / `GraphBuilder`
- value and operation identifiers use `Value*` / `Operation*` names
- operation roles use `OperationRole::{Primary, Linearized}`
- builder APIs use `add_operation`, `values`, `operations`, and
  `resolve_value`
- the old `OpEmitter` abstraction was removed

This tenferro PR pins both upstream commits, migrates active source and current
documentation, and deliberately avoids compatibility aliases.

It also fixes issue #964 by making the `tenferro-linalg/autodiff` AD helpers
accept the same `PrimitiveRuleBuilder` trait-object boundary used by the
extension AD rule entrypoints.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `AGENTS.md` | Confirm repository workflow and verification expectations. | Kept the change on a feature branch and added this work log. |
| `REPOSITORY_RULES.md` | Review terminology, docs, and PR gate requirements. | Excluded historical `docs/plans` material and planned the full local gate before PR. |
| local `computegraph-rs` checkout | Confirm the final renamed API surface. | Migrated field, type, and method names to the merged computegraph API. |
| local `tidu-rs` checkout | Confirm `PrimitiveBuilder`, `LinearizedGraph`, and eager backward APIs. | Added a tenferro-side `PrimitiveRuleBuilder` bridge and removed dependence on `OpEmitter`. |
| `tenferro-ad`, `tenferro-runtime`, `tenferro-internal-ops` | Locate graph metadata, eager backward, traced AD, and rule-builder boundaries. | Renamed graph metadata helpers and eager transpose execution to builder terminology. |
| extension crates (`tenferro-einsum`, `tenferro-linalg`, `tenferro-fft`, `ext/tropical`) | Check optional `autodiff` imports and rule signatures. | Kept feature-gated AD imports and updated rule parameters to `PrimitiveRuleBuilder`. |
| issue #964 | Confirm the reported `tenferro-linalg/autodiff` failure mode. | Converted remaining linalg AD helper signatures from sized `impl PrimitiveRuleBuilder` to `dyn PrimitiveRuleBuilder`. |

## Decisions made

- **No compatibility aliases.** The public and internal surface now uses the
  new graph/value/operation terminology directly.
- **Use `PrimitiveRuleBuilder` for tenferro AD rules.** It bridges
  `computegraph::GraphBuilder` and `tidu::PrimitiveBuilder` so rule code can
  call `add_operation` without depending on the removed `OpEmitter` API.
- **Rename eager transpose execution to builder terminology.**
  `EagerEmitter` became `EagerPrimitiveBuilder`, and the module moved from
  `eager_emitter.rs` to `eager_builder.rs`.
- **Rename graph metadata helpers.** Scoped metadata functions now refer to
  graphs rather than fragments.
- **Rename current Primitive AD architecture doc.** The old
  `docs/architecture/chainrules.md` page moved to
  `docs/architecture/primitive-ad.md`; historical/reference documents remain
  unchanged.
- **Update current docs only.** Historical design notes under `docs/plans` and
  generated/historical material remain untouched.

## Rejected or deferred alternatives

- **No alias module for old computegraph names.** The user explicitly allowed a
  destructive migration and no compatibility requirement exists.
- **No local copy of computegraph/tidu abstractions.** tenferro depends on the
  merged upstream commits instead of vendoring or shadowing the APIs.
- **No broad AD semantic rewrite.** The migration is terminology/API shape:
  graph-level `linearize` and `transpose_rule` semantics stay the same.

## Verification performed

- `cargo check --workspace --all-targets`
- `cargo check -p tenferro-linalg --features autodiff`
- `cargo fmt --all --check`
- `cargo test --workspace --release`
- `cargo test -p tenferro-linalg --features autodiff --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-doc-snippets.py`
- `python3 scripts/check-docs-site.py`
- `cargo clippy --workspace --all-targets --release -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `CUDARC_CUDA_VERSION=12080 cargo test --no-run --package tenferro-gpu --package tenferro-ad --package tenferro-linalg --features cuda --release`
- `git diff --check`
- old terminology scan across active source and current docs

## Remaining risk

- CI still needs to validate the branch in GitHub's environment, but the local
  release-mode workspace, coverage, docs, formatting, and lint gates passed.
