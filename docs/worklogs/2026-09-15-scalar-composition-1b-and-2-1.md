# Work log: scalar composition stage 1b, stage 2.1, and the erasure prototype

Continuation of `docs/worklogs/2026-09-15-scalar-composition-stage1a.md`. Same
branch (`agent/scalar-composition-stage1`), same draft PR (#1800). The design
record is `docs/design/scalar-composition.md`.

## Context read

- `crates/tenferro-internal-cpu-kernels/src/elementwise.rs` — `replay_binary`,
  `replay_scalar_left`, `replay_scalar_right`, `typed_*_with_pool`, the per-op
  `dispatch!` and `dispatch_real_complex_scalar!` macros.
- `crates/tenferro-tensor-core/src/lib.rs` — `DType`, `TensorScalar`,
  `HostTensor`, `Tensor`, and the `impl_scalar!` table.
- `crates/tenferro-tensor/src/types.rs` — the runtime `Tensor` enum,
  `TypedTensor`, `OwnedTensorGroup`, `AllocationGroup`.
- `crates/tenferro-cpu-basic/src/buffer_pool.rs` — sealed `PoolScalar`.
- Issues #1787, #1785, #1789; `REPOSITORY_RULES.md`.

## Decisions

### Stage 1b — the seam is the pool bound, not the erased dispatch

The reusable numerical step is `strided-kernel`'s `zip_map2_into` and `reduce`,
reachable through `replay_binary` / `replay_scalar_*`. Those helpers required the
sealed `PoolScalar` on the element type even though the wrapped calls need only
`Copy`, so the numerical body was restricted to the seven presets for no
numerical reason. Removing that bound and adding caller-destination entry points
(`scalar_binary_into`, `scalar_fold`) gives an external scalar the same body,
while `PooledUninitOutput` keeps its bound because it genuinely allocates through
the sealed pool. The allocation seat differs; the numerical body does not.

The dispatch macros were then declared once
(`dispatch_read_same_variant`, `dispatch_read_real_complex_scalar`,
`dispatch_read_presets`) instead of once per operation.

### `Scalar` / `ScalarArithmetic` / `ad_admission`

`Scalar` carries storage properties plus the algebra (`ScalarDomain::Field` or
`NonField`). `ScalarArithmetic` carries the ordinary operations under the
scalar's own semantics, so integers keep wrapping exactly as the existing integer
kernels do, complex scalars stay fields, and `bool` stays storable without
arithmetic. `ad_admission` is a query that always rejects explicitly, so an
unsupported scalar or order can never become a silent zero gradient, and it is
not wired into any existing differentiation path.

### Stage 2.1 — the set is the value enum, not a wrapper

`define_scalar_set!` emits the tag, the value enum, and the `ScalarSet`
implementation from one member list, so `DType` and `DefaultScalars` are declared
once and `Tensor` becomes the alias `pub type Tensor = DefaultScalars`. This is
the shape #1785's comment asked for, and it avoids the wrapper struct that comment
warned against.

### Stage 2 end state — a tag plus an erased payload

The runtime value type is a seven-variant enum holding `TypedTensor` payloads. A
fixed variant list cannot carry a member tenferro does not declare, and
parameterizing the payload keeps the list fixed, so the end state is a tag plus
one erased payload with the variants removed last. The host prototype
(`ErasedHostTensor`) recovers the payload by its actual Rust type and never by a
tag, a size, or an alignment. `TypedTensor`, `Tensor`, and `TensorView` measure as
`Send + Sync + 'static`, so the erased shape is not blocked by a pool-backed
payload.

## Verification

- `cargo check --workspace --all-targets`: 0 errors, 0 warnings.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo test`: `tenferro-tensor-core` 97 unit + 110 doc, `tenferro-cpu` 538 +
  218 + supporting targets, `tenferro-internal-cpu-kernels` 25 + 2 doc,
  `tenferro-df64-proof` 11 + 6 doc, all pass. `cargo test -p tenferro-cpu` is the
  behaviour check for the rewritten dispatch.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test '...'`: passed on the
  pushed head.
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD
  --dry-run --llm-skipped-reason "local deterministic review"`: pass.
- Measured line accounting: the mechanical simplification removes 254 lines (68
  in 1a, 186 in `elementwise.rs`); the stage total is positive because the open
  contract, entry points, `ad_admission`, the set machinery, the erased prototype,
  and the proof crate are new code.

## Residual risk and open decisions

- The runtime representation choice is recorded but not made: one boxing
  allocation per erased tensor for every member, or a fast path for the preset
  members with erasure only for external ones. The measured 1-thread allocation
  and dispatch cost of both options is the missing input, and releasing a payload
  back to its originating owner is #1789's contract.
- `use Tensor::F64` imports, variant glob imports, dyn compatibility, and build
  cost are still unmeasured for the alias.
- The `trybuild` storage UI fixtures report 10 of 14 mismatches in this worktree
  both with and without these changes (path-prefix artifact).
- Stage 2's module conversion and the C API, XLA, and serialization boundaries are
  untouched.
