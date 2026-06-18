# tidu eager API boundary migration work log

Companion PR: <https://github.com/tensor4all/tidu-rs/pull/28>

Date: 2026-05-31

## Session summary

This change migrates tenferro's eager reverse-mode implementation to the
refined `tidu` eager API boundary.

The companion `tidu-rs` PR moved generic eager recording and backward support
under `tidu::eager`, moved the transpose emitter helper under `tidu::emit`,
and made trace node/edge internals private. This tenferro PR then:

- pins `tidu` to the merged commit containing the new eager boundary
- stores opaque `tidu::eager::Trace` handles instead of `GradNode` internals
- records eager operations through `tidu::eager::Recorder`
- runs backward through `tidu::eager::backward`
- executes transpose through `tidu::emit::linear_transpose_with_builder`

No public tenferro eager tensor API is changed.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `AGENTS.md` | Confirm repository workflow, verification, and work-log requirements. | Added this work log and kept dependency changes in manifests. |
| `REPOSITORY_RULES.md` | Review public surface and abstraction boundary rules. | Kept `tidu` internals out of tenferro's public API and avoided adding compatibility shims. |
| `tenferro-ad/src/eager.rs` | Locate eager tensor trace storage, operation recording, and backward entry point. | Replaced root-level `tidu` eager internals with `tidu::eager` handles and APIs. |
| `tenferro-ad/src/eager/backward.rs` | Locate concrete execution hooks for eager backward. | Implemented `BackwardExecutor` and routed transpose execution through `tidu::emit`. |
| `tenferro-ad/src/eager_ops.rs` and `tenferro-ad/src/extension.rs` | Locate consumers of recorded eager output metadata. | Passed opaque traces through result construction without exposing node layout. |
| `tidu-rs` PR #28 | Confirm the merged commit and final exported API names. | Pinned manifests to `6551e2de955c83a1c8c84b3bca0b037954559617`. |

## Decisions made

- **tenferro treats eager traces as opaque.** `EagerTensor` keeps
  `Option<tidu::eager::Trace<StdTensorOp>>`, and tenferro no longer imports or
  depends on trace node/edge types.
- **Recording is owned by `tidu::eager::Recorder`.** tenferro still supplies
  stable tensor input keys, but the graph layout and saved-forward bookkeeping
  stay behind the tidu boundary.
- **Backward execution is a downstream hook trait.**
  `TenferroBackwardCallbacks` implements `tidu::eager::BackwardExecutor`,
  preserving tenferro's concrete backend execution and extension executor
  integration while moving traversal ownership into tidu.
- **No tenferro public API expansion.** The migration adapts existing internals
  instead of exposing tidu eager types from user-facing constructors or methods.

## Rejected or deferred alternatives

- **No compatibility layer for removed tidu root exports.** The root-level
  eager symbols were intentionally retired in tidu; tenferro now imports the
  narrower modules directly.
- **No local copy of eager trace logic.** Keeping traversal and saved-forward
  layout in tenferro would recreate the boundary leak this refactor removes.
- **No broader eager tensor redesign.** This PR keeps current eager semantics,
  gradient accumulation, extension handling, and metadata registration behavior.

## Verification performed

- `cargo fmt --all --check`
- `cargo check -p tenferro-ad`
- `cargo test -p tenferro-ad eager --lib`
- `cargo test -p tenferro-ad --lib`
- `cargo clippy -p tenferro-ad --all-targets -- -D warnings`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo nextest run --workspace --release --no-fail-fast`
- `cargo test --doc --workspace --release`
- `cargo doc --workspace --no-deps`
- `git diff --check`

## Remaining risk

- CI remains authoritative for merge gating, but the local verification covered
  the same workspace release tests, doctests, docs build, and clippy checks used
  by the current non-GPU gates.
