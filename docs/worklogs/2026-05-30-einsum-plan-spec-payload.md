# Einsum plan-spec payload work log

Date: 2026-05-30

## Session summary

This work made einsum contraction planning part of extension payload identity
instead of treating only subscripts and shapes as semantically relevant. The
change lets traced and runtime execution distinguish different automatic
optimizer options, explicit JAX-style paths, parenthesized trees, and
precomputed concrete trees.

The implementation also carries the planning policy through traced AD. VJP
construction inherits automatic and left-to-right policies directly. Explicit
paths and fixed-pair plans are remapped to the VJP operand list so gradient
contractions do not accidentally reuse a primal positional path over different
operands.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `AGENTS.md` and `REPOSITORY_RULES.md` | Load tenferro workflow, docs, cache, and worklog requirements. | Kept the change in an isolated worktree from `origin/main`, added this worklog, and updated durable docs. |
| `tenferro-einsum/src/optimize.rs` | Inspect public strategy API and path conversion helpers. | Added crate-private `EinsumPlanSpec` and reused JAX path-to-fixed-pair conversion for validation and VJP remapping. |
| `tenferro-einsum/src/traced.rs` | Inspect symbolic/concrete traced lowering and static plan caching. | Stored plan specs in extension payloads and included plan-spec hashes in static-tree cache identity. |
| `tenferro-einsum/src/extension.rs` | Inspect payload identity, runtime execution, and extension AD rules. | Added plan-spec payload hash/equality, runtime cache-key separation, runtime plan resolution, and VJP plan inheritance. |
| `tenferro-einsum/src/eager_tensor.rs` | Check eager extension construction and eager AD behavior. | Added concrete output shape hints so eager backward can use the same VJP planning path. |
| `docs/design/einsum.md` and `docs/guides/einsum.md` | Check current user and design statements. | Documented symbolic `Path`, concrete-only `Tree`, cache identity, and VJP remapping. |

## Decisions made

- **Keep the extension family at `tenferro.einsum.v1`.** The repository has not
  shipped a serialized payload format for this family. The change is still an
  in-family payload refinement, and keeping `v1` preserves the current
  extension family name.
- **Represent planning policy with a crate-private plan spec.** Public
  `EinsumOptimize` remains the user API. Internally, `EinsumPlanSpec` preserves
  the shape-independent policy that should participate in payload identity and
  cache keys.
- **Reject shape-dependent `Tree` for symbolic traced inputs.** A
  `ContractionTree` contains concrete planning state. Symbolic traced calls can
  use `Path` or parenthesized notation instead.
- **Exclude resolved static trees from payload identity.** The static tree is
  an execution hint derived from the plan spec and concrete shapes. The plan
  spec is the semantic payload field.
- **Include plan specs in static and runtime cache keys.** Subscripts and
  concrete shapes alone are not enough when two calls request different
  optimizer options or explicit paths.
- **Derive VJP-specific fixed-pair plans for explicit paths.** A primal
  JAX-style path is positional over the primal operand list. VJP operands are
  `[cotangent, non-active primal operands...]`, so direct cloning would refer
  to the wrong tensors for non-first active inputs.

## Rejected or deferred alternatives

- **No public plan-spec API.** The current public surface can express the needed
  policies through `EinsumOptimize`; exposing an additional plan-spec type would
  add API weight without a current caller need.
- **No exact `ctx.shape_of(Local(ct))` dependency in transpose rules.** The
  current `tidu::linear_transpose` path creates cotangent seed locals before their
  metadata is registered. The einsum AD rule instead uses the extension output
  shape hint that traced and eager construction now attach.
- **No broad cache redesign.** The change only extends einsum-specific cache
  identity with a plan-spec hash and keeps existing extension cache ownership.
- **No GPU-specific changes.** The planning identity change is backend-neutral;
  CUDA/CubeCL support continues through existing execution paths.

## Verification performed

- `cargo build -p tenferro-einsum --features autodiff`
- `cargo test -p tenferro-einsum --features autodiff`
- `cargo test -p tenferro-einsum --features autodiff extension::tests::vjp_einsum_op --lib`
- `cargo test -p tenferro-einsum --features autodiff --test traced_ad_migration symbolic_grad_einsum_with_explicit_path`
- `cargo test -p tenferro-einsum --features autodiff --test eager_tensor eager_tensor_einsum_backward_populates_input_grads`
- `cargo test -p tenferro-einsum --test traced_correctness einsum_symbolic_explicit_path_matches_static_execution`
- `cargo test -p tenferro-einsum --doc`
- `cargo clippy -p tenferro-einsum --features autodiff --all-targets -- -D warnings`
- `cargo test -p tenferro-tensor --release tensor_buffer_refs_cover_backend_metadata`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `cargo fmt --all --check`
- `git diff --check`

## Remaining risk

- The VJP plan remapping preserves the primal tree structure where possible,
  but automatic optimizer plans for VJP still resolve independently from VJP
  shapes. That is intentional for `Auto`.
