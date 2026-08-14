# Session-Oriented Concrete APIs (issue #1673)

Status: design (pre-implementation review target).

## Overview

Add a session-explicit execution surface for concrete `Tensor`/`TypedTensor`
operations (including extension-crate operations) so a caller can reuse one
borrowed `BackendSession` across a sequence of operations instead of paying a
session entry per operation. The tensor value types remain context-free and
are never owned by or lifetime-bound to a session.

The one-shot API (`a.add(&b, &mut backend)`) stays available and becomes a
thin wrapper around the session implementation where practical.

## Motivation (measured)

Session entry for managed CPU execution is ~3 µs/op (release, pinned, 1
worker; issue #1673 data comment). For trivial unary/binary ops (~3 µs total)
it is ~100% of the cost; for a 10-op trivial chain the current API pays 10
session entries (~30 µs) where one entry (~3 µs + 10 op bodies) suffices.
For compute-heavy ops (solve, ~28 µs after #1675) the session share is ~10%,
so the surface targets short chains of cheap ops; it is not a GPU
small-op-launch fix.

**Gate (from the issue):** with `T_one_shot` = 10-op trivial chain via the
current one-shot API (10 session entries) and `T_one_session` = the same
chain via one session entry, require `T_one_shot / T_one_session >= 2`
(predicted ~6). The baseline is explicit: current one-shot API, 10 session
entries.

## Verified execution model (corrects the 2026-08-14 design review)

The design review assumed `with_backend_session` runs the closure
synchronously on the calling thread and therefore proposed dropping the
`Send` bounds. That is **false for CPU**:

- `CpuBackend::with_backend_session` → `run_backend_session_cached` →
  `enter_managed_session`/`enter` → `install_scoped`
  (`crates/tenferro-cpu/src/provider.rs`) → `CpuContext::install`
  (`docs/design/exec-session.md`, "Backend Mapping / CPU") →
  `rayon::ThreadPool::install` — a **real thread handoff to a Rayon worker**,
  even for 1-thread pools (verified empirically: caller thread id differs
  from the closure thread id).
- The default (non-overriding) backend path (`default_backend_session`,
  `crates/tenferro-tensor/src/backend.rs`) runs the closure on the calling
  thread.

**Decision: keep the `Send` bounds** (`R: Send`, `f: ... + Send`) on
`with_backend_session` / `with_backend_session_cached`. They are a soundness
requirement for the CPU managed session (a non-`Send` closure would execute
on another thread). This is an inherent property of the current CPU managed
session, not a migration accident: downstream users of the session-explicit
surface must capture only `Send` state, which the one-shot API already
requires of every closure it takes.

Dropping `Send` would require redesigning CPU session execution to run
closures on the calling thread with per-op pool entry — a performance/
semantics change to the hottest segmented-execution path. That redesign is
out of scope and tracked separately if ever desired; it must NOT be assumed
as a prerequisite.

## Nested-entry prohibition: mechanism, not convention

Operations receiving a `BackendSession` must never call `with_backend_session`
internally. Enforcement by backend family:

- **CPU (release)**: `inherited_or_new_execution_owner()`
  (`crates/tenferro-cpu/src/arbiter.rs`) already panics with
  `BACKEND_REENTRY_PANIC` when a CPU session is entered while
  `EXECUTION_OWNER` is set on the thread (or an owned Rayon scope has an
  active owner). Covers the one-shot-inside-session case, including the
  pinned one-worker managed pool handoff.
- **Default adapter (debug)**: `default_backend_session` sets a thread-local
  in-session flag (Drop-guard restored, so panics in `f` still restore) and
  `debug_assert!`s it was unset at entry (implemented in PR A). Covers
  non-overriding backends.
- **CUDA / WebGPU (not yet wired)**: both override `with_backend_session` and
  call `f` directly, so neither the portable guard nor the CPU panic applies.
  Dedicated enforcement for those overrides is tracked as follow-up; the
  acceptance criterion below is scoped to CPU + default-adapter backends
  until then.

## Generic spelling

Already settled by the current signatures: entries hand out
`&mut dyn BackendSession` (no `S: BackendSession + ?Sized` generic on the
trait methods). The `_in`-surface methods take `session: &mut dyn
BackendSession` directly.

## Cache access parity

`with_backend_session_cached` is the runtime-cache-aware entry (used by the
scheduler/eager paths); `with_backend_session` is the canonical user entry
and the one-shot concrete helpers delegate to it. The session-explicit
surface uses the plain entry; extension plans (`EinsumPlan` etc.) own their
long-lived caches. The canonical user entry must not silently become the slow
entry: `with_backend_session` may forward to the cached path when a cache is
available, but the public contract is the plain entry.

## Surface design (prototype plan)

Layering (same vocabulary, no duplicate validation):

```
one-shot API: &mut TensorBackend ──enter one session──▶ session implementation
session API:  &mut dyn BackendSession ─────────────────▶ session implementation
```

1. `TensorSessionOpsExt` / `TypedTensorSessionOpsExt` with `*_in` methods
   (naming: `_in` preferred; exact names decided in the prototype).
   Implemented by calling the session's own op traits directly; broadcasting,
   dtype promotion, validation, and typed errors identical to the one-shot
   path (shared helpers — no second vocabulary).
2. One-shot methods delegate to the `_in` implementations where practical.
3. Concrete-only extensions (einsum plans first: `EinsumPlan::prepare` +
   `execute_in_session`) compose standard session ops; no
   `ExtensionOp`/`SemanticProgram`/runtime registration required at that
   tier.
4. Runtime-integrated extensions keep the semantic `ExtensionOp` + AD rules;
   runtime preparation produces an executor that runs on the borrowed
   `BackendSession` like the concrete path.

## Non-goals

- No session/runtime handle inside `Tensor`/`TypedTensor`.
- No change to the `EagerTensor` ownership/AD model.
- No unbounded or globally owned sessions; closure-scoped borrowing stays the
  default.
- No second operation vocabulary or duplicated validation.
- GPU small-op launch overhead is a separate concern (fusion/static
  execution), not addressed by session reuse.
- The one-shot API is not broken for aesthetic consistency.

## Acceptance criteria

- One session entry covers a mixed sequence of standard and extension
  concrete operations.
- Standard broadcasting, dtype promotion, validation, placement, and typed
  errors identical to the one-shot path.
- One-shot methods delegate to the same implementation where practical.
- `Tensor`/`TypedTensor` remain context-free.
- Concrete-only extension authors need no runtime/graph/AD machinery.
- Nested-entry prohibition enforced per backend family (CPU release panic +
  default-adapter debug assert; CUDA/WebGPU enforcement tracked as
  follow-up).
- `Send` bounds preserved and documented as a soundness requirement.
- The 10-op trivial-chain gate passes (≥2x, predicted ~6x).
- No material regression for large operations.

## Prototype specification (PR B: the `_in` surface)

Scope decision for the first implementation PR. The design-review gate rules
apply: this section is the pre-implementation design document; the
implementation must not start until it has a reviewer-gpt verdict.

### Measured current state of the one-shot path

A binary op already pays more than one session entry today
(`crates/tenferro-runtime/src/tensor.rs`):

- `add` → `broadcast_binary` → `broadcast_to` per operand (each
  `broadcast_to` can enter 0-2 sessions: `reshape` + `broadcast_in_dim`, or
  zero when shapes already match via `duplicate`) → then
  `with_backend_session(|exec| exec.add(...))` for the op itself.
  Worst case: **5 sessions per binary op** (both operands need
  reshape+broadcast = 2 each + 1 final add, e.g. `[1,2] + [2,1]` → `[2,2]`);
  the common one-sided reshape+broadcast case is 3; equal-shape case is 1.
- `unary_fn` (exp etc.) and `reduce_sum`: 1 session each.

So the session-explicit surface must own the broadcast step too, or a binary
op still pays 2 entries inside one session.

### Surface (prototype op set)

`crates/tenferro-runtime/src/lib.rs`, next to `TensorOpsExt`:

```rust
pub trait TensorSessionOpsExt {
    fn add_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor>;
    fn mul_in(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor>;
    fn exp_in(&self, session: &mut dyn BackendSession) -> Result<Tensor>;
    fn reduce_sum_in(&self, axes: &[usize], session: &mut dyn BackendSession) -> Result<Tensor>;
}
```

and `TypedTensorSessionOpsExt<T: TensorScalar>` with the same four methods
returning `TypedTensor<T>`. Exact op set is deliberately small:
`add_in`/`mul_in` (binary + broadcast, the multi-session flagship),
`exp_in` (unary), `reduce_sum_in` (reduction).

**Naming is transitional**: the `_in` suffix exists only because the
session-explicit form coexists with the one-shot `TensorOpsExt` during the
migration — same receiver type (`Tensor`) + same method name on two in-scope
traits is an ambiguous-method-resolution error in Rust (and a prelude glob
would break every call site). The **final** canonical names drop the suffix:
once the one-shot API is migrated away (a release-boundary breaking change),
the session-explicit methods become the plain `add`/`exp`/`reduce_sum`. The
`_in` names must not be treated as the permanent public spelling.

### Validation and broadcast sharing

- Broadcast plan computation (`broadcast_shapes`, `broadcast_input_plan`) is
  pure and shared. The error mapping (`broadcast_error`) is **not** shared:
  the dynamic surface uses `broadcast_error_to_validation`
  (`crates/tenferro-runtime/src/tensor.rs`) and the typed surface has its own
  manual mapping (`typed_tensor.rs`). Each `_in` surface reuses its existing
  local mapping; no third mapping is added.
- **Dynamic `Tensor` helper (owned)**: session-level
  `broadcast_to_in(input, target_shape, session)` using
  `session.reshape` + `session.broadcast_in_dim` with the same plan logic as
  the existing `broadcast_to`, **preserving the `duplicate()` path** for
  equal shapes and broadcast sources (owned copy semantics unchanged).
  `add_in`/`mul_in` = `broadcast_to_in` both operands + `session.add`/
  `session.mul`, all inside the one borrowed session.
- **Typed `TypedTensor<T>` helper (borrowed, read-based)**: the typed one-shot
  path keeps inputs borrowed and dispatches to the `*_read` methods
  (`typed_tensor.rs`); the typed `_in` surface mirrors that with
  `reshape_read`/`broadcast_in_dim_read`/`add_read`/`mul_read` on the session
  and `into_typed_result` for the output (dtype fixed by `T`, no promotion
  logic). Do NOT funnel the typed path through the owned dynamic helper —
  that would introduce copies or change allocation semantics.
- No second validation vocabulary; no new error kinds. One-shot/session
  parity tests cover values and structured errors for both surfaces.

### One-shot delegation

One-shot `add`/`mul`/`exp`/`reduce_sum` become
`backend.with_backend_session(|s| ..._in(...))` where practical. This drops
the worst-case binary-op session count from 5 to 1 — a side benefit that must
be measured as a no-regression check, not assumed. `dot_general`/`matmul`
are excluded from this PR (cache-ownership decision pending) and keep their
current path.

### Explicitly out of scope for PR B

- `dot_general`/`matmul` (deferred until cache ownership/parity with
  `with_backend_session_cached` / `SessionCachedDot` is decided; the current
  one-shot matmul already uses the plain entry, so this is purely a
  cache-ownership decision).
- `convert`/`cast` and the rest of the op vocabulary (follow-up PRs).
- `EinsumPlan::execute_in_session` and all extension-crate tiers (PR C).
- GPU session surfaces; CUDA/WebGPU nested-entry enforcement.
- Removing `Send` bounds (soundness requirement, see above).

### Public API and tests (explicit before merge)

Both public traits (`TensorSessionOpsExt`, `TypedTensorSessionOpsExt`) get
runnable doc examples (AGENTS.md requirement — compile and run, no `ignore`).
Integration tests cover, per surface: equal-shape op, real broadcast, invalid
broadcast (structured error parity with the one-shot path), dtype/error
parity, typed output dtype validation (`into_typed_result`), and a test that
an `_in` chain executes inside exactly one session entry.

### Performance gate (measured, not assumed)

New criterion bench in `crates/tenferro-runtime/benches/` (criterion is
already a workspace dependency), with an explicit `[[bench]]` target and
`harness = false` in `Cargo.toml` (mirroring `elementwise_fusion.rs`):

- **Exact chain (10 ops)**: three repetitions of `add → exp → mul` (9 ops)
  followed by a final `reduce_sum([0])` (10th). `reduce_sum` is final so no
  intermediate op changes shape.
- **No-broadcast arm**: 1×8 f64 constant-filled operands (broadcast is the
  `duplicate` path).
- **Broadcast arm**: 1×1 f64 operands against 1×8 (real reshape+broadcast).
- **One-shot arm**: the chain via `TensorOpsExt` (each op enters its own
  session — 10 entries). **Session arm**: the same chain via
  `TensorSessionOpsExt` inside one `with_backend_session` (1 entry).
- Validate one result outside the timed region (finite, correct shape);
  `black_box` inputs and outputs inside the iter.
- **Protocol**: record base-commit (pre-delegation) and post-change medians;
  gate `T_one_shot / T_one_session >= 2` (predicted ~6); also a
  representative large-op one-shot before/after pair for the no-regression
  criterion. Report pinned (idle core), matching the session floor
  methodology; the gate is evaluated on the pinned numbers.

## Prototype specification (PR C: `ConcreteEinsumPlan::execute_in_session`)

The first extension-crate example of the session-explicit concrete surface
(design tier 1: concrete-only extensions composed from standard primitives —
no `ExtensionOp`, `SemanticProgram`, runtime registration, or AD rules).

### Key finding: the one-shot einsum already enters a session

`ConcreteEinsumPlan::execute` etc. (`crates/tenferro-einsum/src/concrete.rs`,
public plan type is **`ConcreteEinsumPlan`**) already do
`backend.with_backend_session(|exec| eager_einsum_exec(exec, inputs,
&self.tree))`; the eager core functions
(`crates/tenferro-einsum/src/eager.rs::eager_einsum_exec*`) all take
`&mut dyn BackendSession` already. `execute_*_in_session` is a thin
addition: plan validation (pure — `validate_inputs`/`input_specs` need no
backend) + the same core call on the caller's borrowed session — **no new
session entry**. One-shot methods delegate to the `_in_session` variants.

### Surface (complete signatures)

`ConcreteEinsumPlan` gains a `_in_session` mirror of every one-shot execute
method (all mechanical: validate + call the eager core on the borrowed
session). Signatures mirror the one-shot forms with `backend: &mut B` →
`session: &mut dyn BackendSession`:

```rust
pub fn execute_in_session<'a, I>(
    &self, inputs: I, session: &mut dyn BackendSession,
) -> Result<Tensor>
where I: AsRef<[&'a Tensor]>;

pub fn execute_typed_in_session<'a, T: TensorScalar, I>(
    &self, inputs: I, session: &mut dyn BackendSession,
) -> Result<TypedTensor<T>>
where I: AsRef<[&'a TypedTensor<T>]>;

pub fn execute_read_in_session<'a, I>(
    &self, inputs: I, session: &mut dyn BackendSession,
) -> Result<Tensor>
where I: AsRef<[TensorRead<'a>]>;

pub fn execute_into_in_session<'a, I>(
    &self, inputs: I, session: &mut dyn BackendSession, out: TensorWrite<'_>,
) -> Result<()>
where I: AsRef<[&'a Tensor]>;

pub fn execute_typed_into_in_session<'a, 'out, T: TensorScalar, I, O>(
    &self, inputs: I, session: &mut dyn BackendSession, out: O,
) -> Result<()>
where I: AsRef<[&'a TypedTensor<T>]>, O: Into<TypedTensorWrite<'out, T>>;

pub fn execute_read_into_in_session<'a, I>(
    &self, inputs: I, session: &mut dyn BackendSession, out: TensorWrite<'_>,
) -> Result<()>
where I: AsRef<[TensorRead<'a>]>;

pub fn execute_read_into_accum_in_session<'a, I>(
    &self, inputs: I, session: &mut dyn BackendSession,
    accumulation: DotGeneralAccumulation, out: TensorWrite<'_>,
) -> Result<()>
where I: AsRef<[TensorRead<'a>]>;
```

One-shot `execute*` become
`backend.with_backend_session(|s| self.execute*_in_session(inputs, s))`.

**Validation ordering change (accepted)**: the one-shot path currently
validates before entering the session, so invalid calls are entry-free. After
delegation, validation happens inside the session entry — an invalid call
pays one entry (~3 µs) before returning the identical error. Accepted (error
paths are exceptional); covered by an error-parity test asserting the same
structured error both ways.

**Naming**: `_in_session` matches the existing prepared-operation vocabulary
(`crates/tenferro-runtime/src/runtime/capability.rs`:
`prepared_operation.execute_in_session`); the runtime concrete noun ops use
`_in` (PR B). Both are transitional; final canonical names drop the suffix.

### Mixed chain (acceptance item)

```rust
backend.with_backend_session(|session| -> tenferro_einsum::Result<_> {
    let x = plan.execute_in_session(&[&a, &b], session)?;   // einsum Result
    let x = x.exp_in(session)?;                              // converts via From
    Ok(x.reduce_sum_in(&[0], session)?)
})
```
One entry covers standard ops + a prepared extension plan.

### Bench (crates/tenferro-einsum/benches/, `[[bench]] harness=false`)

Exact workload: a prepared plan for `"ij,jk->ik"` on 8×8 f64 column-major
inputs, executed 10 times per iter (10 calls). Backend: one-worker
`CpuBackend::with_threads(1)` for all arms. Correctness: validate one result
outside the timed region (shape `[8,8]`, finite). `black_box` inputs and
outputs inside the iter.

- **Arm A (one-shot)**: 10 × `plan.execute([a, b], backend)` — 10 session
  entries.
- **Arm B (in-session)**: the same 10 calls inside one
  `with_backend_session(|s| ... execute_in_session(...))` — 1 entry.
- **Arm C (mixed)**: `execute_in_session` + `exp_in` + `reduce_sum_in` ×10
  inside one session vs the one-shot equivalents.
- **Reported**: pinned (idle core) medians for A/B/C; the A/B ratio and the
  mixed-chain ratio; a baseline/post-delegation pair for the one-shot arms.
  The PR-B trivial-chain ≥2 gate is a separate acceptance; for einsum the
  recorded evidence is the measured ratios and the delegation baseline, not a
  predicted number.

### Acceptance (explicit before merge)

- Runnable doctests (no `ignore`/`no_run`) for all seven `_in_session`
  methods.
- Parity tests: `_in_session` results == one-shot results (values) for
  execute/execute_read/execute_typed; structured-error parity for invalid
  inputs (both orderings); typed dtype conversion; `execute_into`/`accum`
  output validation; and a session-counting test proving `_in_session` adds
  no nested entry (one entry for a mixed einsum + `_in` chain).

### Out of scope for PR C

- Native-kernel extension capabilities (FFT, decompositions, sparse — design
  tier 2) and runtime-integrated extension execution on the borrowed session
  (tier 3).
- Top-level `einsum()`-trait `_in_session` variants (the prepared plan is the
  recommended repeated-execution API; trait variants are follow-up if the
  plan-level surface proves out).
- Any change to the semantic/graph/AD einsum paths.
