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

### Coverage of the changed files

Measured with `cargo llvm-cov -p tenferro-tensor-core -p tenferro-internal-cpu-kernels -p tenferro-df64-proof --profile ci --json`:

| File | Line coverage |
| --- | --- |
| `crates/tenferro-tensor-core/src/erased.rs` | 100% |
| `crates/tenferro-tensor-core/src/scalar_set.rs` | 100% |
| `crates/tenferro-tensor-core/src/scalar.rs` | 87.5% |
| `crates/tenferro-internal-cpu-kernels/src/scalar_ops.rs` | 88.5% |
| `ext/df64-proof/src/lib.rs` | 100% |

The repository enforces the default 80% per-file threshold from
`coverage-thresholds.json`, and every changed file clears it. The remaining
uncovered regions are the `map_err` closures that map a strided failure into
`crate::Error`: a host tensor with column-major strides cannot make
`StridedView::new` or `reduce` fail, so those branches are unreachable rather
than untested. Reaching the 90% soft target by padding them would add tests that
assert nothing, so they are left as measured.

### Workspace suite against the baseline

`cargo test --workspace --no-fail-fast` on this branch: 5163 passed, 3 failed.
The same command on a pristine `origin/main` worktree at `bca2d54a`: 5105 passed,
the same 3 failed. The three are `trybuild` fixture renderings whose committed
`.stderr` files were produced by a different compiler or path prefix
(`eager_backend_capability_contract::eager_backend_capability_boundary`,
`session_contract::execution_session_capability_cannot_project_or_escape_owner_borrow`,
`storage_ui_compile_contracts`). The branch therefore adds 58 passing tests and
introduces no new failure; the three are pre-existing environment artifacts, and
`cargo test --workspace` stops at the first of them without `--no-fail-fast`.

## Stage 2 progress landed after this list

- **Promotion lattice derived, not hand-written.** Each member declares its
  arithmetic kind, rank within the kind, and component width; `define_scalar_set!`
  derives `promote` from those facts. The hand-written table in
  `tenferro-tensor/src/validate/mod.rs` is deleted and `promote_dtype` delegates to
  the set. The derived lattice was checked against the recorded table for all 49
  pairs, and the external set in `ext/df64-proof` promotes within its own lattice.
- **Shared dispatch macros.** `same_variant_pair!` and `same_variant_unary!` live
  in `tenferro-internal-cpu-kernels/src/dispatch.rs`, are re-exported through
  `tenferro-cpu`, and take the fallback expression as a parameter, so a crate that
  converts a call site pays no macro-definition cost. `tenferro-linalg` drops its
  local copy and converts eight sites for 112 deleted lines net.
- **The mechanically convertible set is exhausted.** A workspace scan for matches
  whose whole body is a same-variant dispatch finds two remaining sites, both in
  the householder-QR factor import path, where the dispatched value is a
  `CompactQrResult` rather than a `Tensor`.
- **The remaining large clusters are not whole matches.** The owned-tensor
  elementwise operations carry six same-variant arms followed by four
  scalar-mixing arms and a fallback, so a whole-match macro cannot take them.

## The measurement that decides the next step

Adding the variant in a disposable worktree and compiling the workspace measured
what admitting an external member costs: an erased variant on `Tensor` needs 18
explicit arms, all inside `tenferro-tensor`, and an external variant on `DType`
needs 59 across about thirty files (7 in tests, 40 in production statement
position, 12 in production value position where the enclosing function must start
returning a result). Making `dtype()` fallible instead would touch 605 call sites.
So the hybrid shape is roughly 77 arms against the 2267 production pattern sites
that removing the seven variants rewrites, and it is the cheaper path.

Both new variants change public types, so under this repository's rules the change
needs maintainer acceptance before a feature pull request carries it. That, and
#1789's decision that an external payload is caller-owned rather than pool-owned,
are the only remaining inputs.

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
