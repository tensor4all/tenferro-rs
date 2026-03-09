# AD Correctness Batch Design

## Scope

This batch fixes four public AD/runtime correctness bugs that do not require GPU runtime support:

- `#413` `chainrules` / `tenferro-einsum` HVP examples are stale and `TrackedTensor` HVP through einsum loses the tangent seed.
- `#414` `chainrules-scalarops` `sqrt_frule` / `sqrt_rrule` silently clamp the singular derivative at zero to `0`.
- `#419` `tenferro-burn` exposes an N-ary autodiff `einsum` surface but panics for arities above `2`.
- `#420` `tenferro-linalg` solve AD rules mis-handle square matrix right-hand sides and truncate them as if `nrhs = 1`.

The batch does not change GPU behavior, DLPack runtime ownership, or tropical C-API behavior.

## Goals

- Make the public AD surfaces behave consistently with the current API and math contract.
- Replace silent wrong answers or panic-on-valid-input paths with working behavior.
- Add regression tests at the crate that owns each bug.
- Keep the fixes local to the owning crates instead of adding cross-crate policy layers.

## Approach

### `#413` HVP and stale docs

`tracked_einsum` currently records forward tangents from `Tensor::fw_grad`, which matches the internal forward-composition test but not the public `Tape::leaf_with_tangent` HVP route. The fix is to align tracked einsum tangent capture with the tracked-value AD metadata used by `chainrules`, so HVP sees the tangent direction recorded on the tape leaf. At the same time, refresh the stale `tracked_einsum` / `dual_einsum` examples in `extern/chainrules` and `docs/design/autodiff.md` so copy-paste usage matches the current context-taking APIs.

### `#414` singular `sqrt(0)` derivative

The scalar helper should follow the derivative formula instead of replacing the singular branch with zero. The fix is to remove the zero-clamping branch in `sqrt_frule` / `sqrt_rrule` and add regression tests that assert the zero case surfaces a non-finite derivative rather than erasing the signal.

### `#419` Burn N-ary autodiff

The forward Burn path already accepts `Vec<FloatTensor<_>>`, so the autodiff wrapper should stop panicking on three or more operands. The fix is to generalize the backward rule construction from unary/binary special cases to an N-ary contraction path that delegates gradient computation through the existing tenferro einsum AD machinery and then maps the cotangents back into Burn tensors.

### `#420` multi-RHS solve AD

The solve AD rules should infer RHS shape using the same matrix/vector logic as the forward tensor backend, not open-coded `dims()[1] != n` checks. The fix is to centralize or reuse the RHS interpretation for the AD rules, then add square-matrix-RHS regression tests for `solve_rrule`, `solve_frule`, `solve_triangular_rrule`, and `solve_triangular_frule`.

## Files Likely Touched

- `extern/chainrules/src/lib.rs`
- `extern/chainrules-scalarops/src/lib.rs`
- `extern/chainrules-scalarops/tests/scalarops_tests.rs`
- `docs/design/autodiff.md`
- `tenferro-einsum/src/ad.rs`
- `tenferro-einsum/tests/einsum_tests.rs`
- `extension/tenferro-burn/src/backward.rs`
- `extension/tenferro-burn/src/tests/mod.rs`
- `tenferro-linalg/src/lib.rs`
- `tenferro-linalg/tests/linalg_tests.rs`

## Testing Strategy

- Add one regression test per bug at the owning crate.
- Prefer red/green targeted tests first:
  - `cargo test -p tenferro-einsum ...`
  - `cargo test -p chainrules-scalarops ...`
  - `cargo test -p tenferro-burn ...`
  - `cargo test -p tenferro-linalg ...`
- After the targeted fixes, run the repository-required verification suite before PR creation.

## Risks

- `#413` and `#419` both touch AD plumbing, so the main risk is accidentally fixing one path while breaking an existing unary/binary path.
- `#420` touches numerically sensitive tests; shape regressions must be checked independently from floating-point closeness.
- `#414` changes behavior at a singular point; tests should assert "non-finite" rather than over-specifying one exact IEEE payload where unnecessary.
