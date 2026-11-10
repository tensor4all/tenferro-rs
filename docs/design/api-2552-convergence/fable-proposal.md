設計文書は依頼どおり英語で書きました。資料から確認できない事実は `NEED:` と明記しています。コンパイルやテストは実行していません。

---

# Design: one per-call `ExecutionRequest` carries the materialization policy

## 1. The decision

Every public concrete or eager entry in linalg, einsum and fft that today takes an execution parameter (`session: &mut dyn Engine`, or `exec: &mut Exec<'_>` in the six `*_with_exec` functions) takes an `ExecutionRequest` in that same position instead. The request is a by-value, non-`Clone` borrow of the resources plus a `MaterializationPolicy` that defaults to `Allow`. Nothing else changes: names, argument order, input types, options and result types stay as they are.

```rust
// In the crate that defines `Engine`.  NEED: which crate that is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterializationPolicy { #[default] Allow, Forbid }

pub struct ExecutionRequest<'a, R: ?Sized = dyn Engine + 'a> {
    resources: &'a mut R,
    materialization: MaterializationPolicy,
}
impl<'a> ExecutionRequest<'a> {                      // R = dyn Engine
    pub fn new(engine: &'a mut dyn Engine) -> Self;  // Allow
}
impl<'a, R: ?Sized> ExecutionRequest<'a, R> {
    pub fn over(resources: &'a mut R) -> Self;       // Allow; for non-Engine resources
    pub fn with_materialization(self, policy: MaterializationPolicy) -> Self;
    pub fn materialization(&self) -> MaterializationPolicy;
    pub fn resources(&mut self) -> &mut R;
    pub fn reborrow(&mut self) -> ExecutionRequest<'_, R>;
}

// tenferro_tensor::Error (already #[non_exhaustive]) gains one variant:
InputMaterializationForbidden { op: &'static str, input: usize }
```

The three entries from the material become:

```rust
// linalg, with_exec.rs:63 — only the first parameter's type changes
pub fn svd_with_exec<T: LinalgScalar>(
    exec: ExecutionRequest<'_, Exec<'_>>, buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>, full: bool,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T::RealScalar>, TypedTensor<T>)>;

// einsum, concrete.rs:137
fn einsum(&self, subscripts: &str, session: ExecutionRequest<'_>) -> Result<Tensor>;

// fft, lib.rs:263
pub fn fft(&mut self, input: &Tensor, n: Option<usize>, axis: isize, norm: FftNorm,
           session: ExecutionRequest<'_>) -> tenferro_tensor::Result<Tensor>;
```

Caller code:

```rust
a.einsum("ij,jk->ik", ExecutionRequest::new(&mut engine))?;                      // Allow
a.einsum("ij,jk->ik", ExecutionRequest::new(&mut engine)
        .with_materialization(MaterializationPolicy::Forbid))?;                  // assertion
svd_with_exec(ExecutionRequest::over(exec), buffers, &view, false)?;             // Allow
```

**Semantics.** The policy is a gate, never a selector. The provider chooses the same path under both values; `Forbid` only turns "this path copies an input to rearrange it" into `InputMaterializationForbidden`, raised before any side effect.

A copy counts as input materialization iff it would not happen had the same operand arrived in the layout the provider consumes directly, at the same placement and dtype. Copies made for every layout are workspace and are never refused. Placement, dtype and no-provider refusals are checked first and are identical under both values.

## 2. Where the policy lives

It lives on the request value, which exists for exactly one call. It is not on `Engine`: the trait gains no method or state. It is not in any `*Options` struct. Because the request holds `&mut` and is consumed by the call, a policy cannot be left behind for the next call, and the borrow checker enforces this.

**Why not the alternatives the owner asked to compare:**

- **(b) A separate `policy` argument.** It adds a parameter to every signature, which the issue's point 6 rejects. Rust has no defaults, so every caller would write `Allow` by rote. Helpers would thread two values where they now thread one.
- **Keeping `&mut engine` via `impl Into<ExecutionRequest>`.** This fails on the smallest case: `fn f(s: &mut dyn Engine) { a.svd(s)?; b.svd(s)?; }` stops compiling, because a generic parameter does not reborrow and `s` is moved. The concrete parameter type keeps both reborrow and `&mut CpuEngine → &mut dyn Engine` coercion, at the cost of writing `ExecutionRequest::new(..)`.
- **(c) An existing mechanism.** The only ones in the material are `Engine` and `Exec`/`BufferPool`. Putting the policy on either is the persistent state the owner ruled out. An `Engine`-implementing wrapper is the same thing by another route.

**Against R1–R7** (I have only their one-line statements):

| Rule | Status |
|---|---|
| R1 one fact, one expression | Holds. One policy type, one carrier, no `svd_no_copy` twin. `new` is separate from `over` only because `new(..).with_materialization(..)` would otherwise infer `R = CpuEngine`. |
| R2 `_into` is the primitive | Holds. The allocating form forwards the request untouched, so sugar and primitive cannot disagree on policy. |
| R3 never canonicalize implicitly | **Broken; amend it.** |
| R4 erasure only at the FFI edge | Not worsened. `dyn Engine` is the erasure already in today's signatures, and `R` lets static resources stay static. |
| R5 no foreign type in a public API | Not worsened, not fixed. `Exec` stays in the same six signatures; the new type names no foreign type. |
| R6 transitions not guarded by a comment | Holds. `Forbid` is a typed runtime refusal. |
| R7 push into the type | Holds. "Per invocation" is expressed by ownership. |

**Proposed R3 text:** "Never change placement or dtype implicitly. Same-placement, same-dtype input-layout normalization is done only by the provider, only under the invocation's `MaterializationPolicy` (default `Allow`); `Forbid` asserts its absence."

R3 should be amended for two reasons. Per the audit it is already false on main for every `dyn Engine` route. Its escape hatch (`buffers.normalize(view)`) either copies unconditionally or makes callers branch on provider knowledge.

`NEED:` the list of six withdrawn proposals, to confirm none is a per-call request type.

## 3. One convention for all three families

Each cell is today → proposed.

| Row | linalg | einsum | fft |
|---|---|---|---|
| **Eager entry** | `svd_with_exec(exec, buffers, view, full)` free fn, plus `TensorLinalgExt` methods with `session` last → unchanged | `a.einsum(subscripts, session)` trait method → unchanged | `FftExecutor::fft(&mut self, input, n, axis, norm, session)` → unchanged |
| **Traced entry** | `TraceContextLinalgExt::svd(&self, a: TraceValue) -> TracedTensor` → unchanged | `einsum(&mut self, inputs: &[TraceValue], subscripts: &str) -> Result<TraceValue>` → unchanged | `TraceContextFftExt::fft(&self, ..) -> TracedTensor` → unchanged |
| **Exec parameter** | `&mut Exec<'_>` (six fns) and `&mut dyn Engine` (traits) → `ExecutionRequest<'_, Exec<'_>>` and `ExecutionRequest<'_>` | `&mut dyn Engine` → `ExecutionRequest<'_>` | `&mut dyn Engine` → `ExecutionRequest<'_>` |
| **Buffer ownership** | Explicit `&mut BufferPool` (six fns), engine-owned (traits) → unchanged | Engine-owned → unchanged | Engine-owned; plan cache in `&mut self` → unchanged |
| **Options** | Positional `full: bool`, or `SvdOptions` etc. → unchanged | `&str`, or `einsum_with(.., optimize)` → unchanged | Loose `n, axis, norm` → unchanged |
| **Materialization policy** | Fixed per entry: `*_with_exec` refuses negative strides, trait routes gather → read from the request | Always gathers → read from the request | Always gathers → read from the request |

- **Traced:** the graph and its entries never see the policy.
- **Options:** no options type may gain a policy field.
- **FFT:** `&mut self` on `FftExecutor` is plan-cache ownership, not the exec parameter; the exec parameter is `session`.

The convention is one sentence: *the parameter that borrows execution resources is an `ExecutionRequest` over those resources, and it is the only place a call states what may happen to its input layout.* The remaining disagreements (input type, options, buffers, names) are untouched and listed in section 5.

**Traced execution.** The same substitution applies where a traced graph is run on an engine. `NEED:` that entry's name and signature, and whether `PrepareOptions` exists. Because the policy is a gate and not a selector, one physical plan serves both values and the cache key does not change. The plan records at prepare time whether it contains an input-materialization step; executing it with a `Forbid` request refuses if it does. This satisfies the owner's "validated before reuse".

## 4. Extensions follow the same convention

A family plugs in with three obligations and no trait to implement:

1. Its public entries take `ExecutionRequest` where they would take resources.
2. At the single point where it is about to gather an input (linalg's `with_owned_layout` is the existing example), it checks `request.materialization()`. On `Forbid` it returns `InputMaterializationForbidden { op, input }`.
3. Checks run in this order, all before any side effect: provider, dtype and placement (existing errors, policy not consulted), then the layout gate, then the zero-size shortcut (the #2505 ordering).

For FFT, `FftExecutor::fft(.., session: ExecutionRequest<'_>)` is the whole API change. The audit says the CPU path copies each lane into per-job buffers. If that happens for every layout, it is workspace under the criterion in section 1, and `Forbid` never refuses on its account.

`NEED:` whether the CPU FFT compacts the whole input before lane iteration for some layouts. If it does, that copy is the gated one.

A provider that cannot take the layout at all has two cases:

- **A same-placement gather would make it consumable.** `Allow` gathers once; `Forbid` refuses. CUDA is the live case: per the audit every non-compact view is gathered on device, so `Forbid` refuses them all. This is a documented per-provider outcome.
- **No gather helps** (wrong placement, dtype, or no provider). The existing typed refusal applies under both values, and no transfer is ever attempted.

## 5. Out of scope

- **Collapsing owned / `_read` / typed receivers** (audit C2): an independent decision about input types.
- **Renames**, including the `_with_exec` suffix and `*_values_*` (C1): mechanical, separate PR.
- **Options structs replacing `bool`s and loose arguments** (C3): changes semantic options, which this change must not touch.
- **Argument order** (C4), **result structs** (C5), **error type per layer** (C6, ledger F), **axis type** (C7), **layer parity** (C8): each is a separate convention.
- **Plan shape and `FftExecutor`** (C10), and **folding `(Exec, BufferPool)` into an engine**: this decides buffer ownership.
- **Removing `Exec` from public signatures** (ledger E): this change neither adds to it nor fixes it.
- **Widening trait-route admission to match `*_with_exec`** (audit row 1): a provider improvement. `Forbid` tests flip from refusal to success when it lands.
- **Materialization counter and stats**: needs a decision on where stats live. `Forbid` already tests "count is zero".
- **`_into` destination layouts**: the policy governs inputs only; `out` handling is unchanged.
- **Ledger A, C, D, G**: unrelated to execution parameters.

## 6. Migration and proof

**Callers change:**

- `&mut engine` → `ExecutionRequest::new(&mut engine)`.
- A `session: &mut dyn Engine` variable → `ExecutionRequest::new(session)`.
- A helper that passes a request to two callees → `req.reborrow()`.
- `svd_with_exec(exec, ..)` → `svd_with_exec(ExecutionRequest::over(exec), ..)`.

**One behaviour change:** the six `*_with_exec` functions now normalize a negative-stride input under the default, where they refused before. `Forbid` reproduces the old refusal.

**Gate:** the R3 amendment, `docs/spec/api-conventions.md` and the shared rule "No hidden … dense-layout conversions" in `tensor4all-agent-rules` land as linked PRs with this change.

**Assertions.** All fail to compile today because the type does not exist; the second also fails behaviourally today.

1. **Forbid refuses before side effects.** A negative-stride input under `Forbid` returns `InputMaterializationForbidden { input: 0, .. }`, and `FftExecutor::cache_stats()` is equal before and after the refused `fft` call.
2. **Allow equals explicit compaction.** `svd_with_exec` on a negative-stride view under the default returns `Ok`, equal to the result on a compacted copy.
3. **Gate, not selector.** For every layout where `Forbid` returns `Ok`, the outputs are bit-identical to `Allow`.
4. **Forbid does not refuse direct paths.** `svd_with_exec` on a transposed rank-3 batch, and the trait route on a transposed rank-2 matrix with `native`, both return `Ok` under `Forbid`.
5. **Policy does not persist.** After a refused `Forbid` call, the same engine and input under `new(..)` return `Ok`.
6. **Other refusals are policy-independent.** An unsupported dtype or placement returns the same error variant under both values.
7. **Empty tensors follow the layout rule.** An empty negative-stride input under `Forbid` refuses exactly like the non-empty case.
8. **Einsum reports the operand.** `input` equals the index of the operand that would be copied.
9. **Traced plans are validated on reuse.** One prepared plan runs under `Allow` and is refused under `Forbid`, with an unchanged cache key. (`NEED:` the prepare/run API.)
10. **Source audit.** No public function in the three families has a `&mut dyn Engine` parameter, no `*Options` struct has a `MaterializationPolicy` field, and `Engine` has no item mentioning it.
