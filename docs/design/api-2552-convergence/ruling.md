# Ruling on tenferro-rs#2552: the materialization policy (adjudication, after review)

Two independent designs were written from the same material, then each read the other. They crossed:
Fable 5.2 conceded the carrier to GPT Astra's **plain argument**, Astra conceded it to Fable's
**`ExecutionRequest` wrapper**. Both converged on the semantics and split on one question — how the
policy travels. This decides it, states the definition both left implicit, and marks what a review
of this ruling found still open.

## What both designs agree on (adopted)

1. One enum, in `tenferro_tensor`: `MaterializationPolicy { #[default] Allow, Forbid }`.
2. The policy is **per invocation**: not on `Engine` (no state, no setter), not in any `*Options`.
3. `Forbid` never changes placement, dtype, backend, or the requested mathematics.
4. One concrete error variant on the existing non-exhaustive `tenferro_tensor::Error`:
   `InputMaterializationForbidden { op: &'static str, input: usize }`, `input` being the zero-based
   operand index, remediation in the display text.
5. **Validate every operand before normalizing any**, then refuse before side effects. Without this
   rule an einsum can gather operand 0 and only then discover it must refuse on operand 2.
6. Normalized storage must outlive an asynchronous consumer through the existing retention path, not
   through the lifetime of the policy value.
7. Output allocation, intrinsic workspace, and kernel packing that accepts the original strides stay
   allowed under `Forbid`.
8. **R3 is amended**: "Never change placement or dtype implicitly. Same-placement, same-dtype
   input-layout normalization happens only by the provider, only under the invocation's
   `MaterializationPolicy` (default `Allow`); `Forbid` asserts its absence."
9. Traced execution carries the policy at the call that submits work; graph construction is
   unchanged, and the physical-plan cache key does not include the policy.

## The definition both proposals circled but did not state the same way

A copy is **input materialization** if and only if it would not happen had the same operand arrived
in the layout the selected provider consumes directly, at the same placement and dtype. Copies made
for every layout are intrinsic workspace and are never refused. Consequences, all of which follow
from that one sentence: renaming a temporary to "workspace" exempts nothing; an FFT lane buffer that
a directly supported input also needs is workspace, while an additional whole-input gather to
satisfy admission is materialization; a dense permutation that exists only to make an einsum operand
admissible is materialization; kernel packing that accepts the original strides is neither.

Which of today's copies fall on which side is **not** established by this document — see the open
items below.

## The one question decided: how the policy travels

| | plain argument (Astra) | `ExecutionRequest<'a, R>` (Fable) |
|---|---|---|
| signature churn | one added parameter per executing entry | the execution parameter changes type |
| call site | `svd_with_exec(exec, buffers, view, false, Default::default())` | `svd_with_exec(ExecutionRequest::over(exec), buffers, view, false)` |
| helper holding `&mut dyn Engine`, calling twice | unchanged | `reborrow()` on all but the last call |
| new public items | enum + error variant | enum + error variant + a six-item struct impl |
| a second execution constraint later | another parameter | absorbed by the wrapper |

**Decided: the plain argument.** Reasons, with their limits stated.

1. **Its costs land now; the wrapper's benefit is a hope.** The wrapper's case is that a second
   execution constraint will arrive and want the same carrier. Nobody has named that constraint. The
   wrapper also has a present benefit — call sites that keep passing `&mut engine` unchanged — and
   that is a real ergonomic advantage, not a speculative one; it is outweighed here because a new
   public type with a lifetime and a generic parameter, plus reborrow at every multi-call helper, is
   more surface than one `Copy` enum in one more position. Migration in either direction changes
   downstream signatures, so this decision is not free later; it is simply the smaller thing to
   carry today.
2. **A lifetime pitfall exists in the naive wrapper, not in the submitted one.** A four-file check
   with `rustc 1.96.0` shows the naive form failing for a plain `&mut dyn Engine`, and the fix is the
   default the submitted wrapper already carries. The failing form is exactly this — note the
   `impl<'a> Req<'a, dyn Engine>` that names the object type without a lifetime:

   ```rust
   pub struct Req<'a, R: ?Sized>(&'a mut R);
   impl<'a> Req<'a, dyn Engine> {
       pub fn new(e: &'a mut dyn Engine) -> Self { Req(e) }
   }
   ```

   and the compiler answers

   ```
   error: lifetime may not live long enough
     | impl<'a> Req<'a, dyn Engine> { pub fn new(e: &'a mut dyn Engine) -> Self { Req(e) } }
     |                                   -                       ^^^^^^ returning this value requires
     |                                   |                              that `'a` must outlive `'static`
     = note: requirement occurs because of the type `Req<'_, dyn Engine>`, which makes the generic
             argument `dyn Engine` invariant
   ```

   Writing `R: ?Sized = dyn Engine + 'a` fixes it, which is what the submitted wrapper does; see the
   appendix for the four files and all four outcomes. The same helper compiles untouched with the plain
   argument: `fn helper(s: &mut dyn Engine, m: MaterializationPolicy) { einsum(.., s, m); einsum(.., s, m); }`.
   The sources, the rustc invocation, and both diagnostics are attached to this issue as a comment.
3. **Neither carrier keeps a policy from being dropped, and no lint does either.** A wrapper can log
   the policy, branch on it once, or forward it to one of two nested calls and still satisfy
   `unused_variables`. What establishes the contract is a test that drives an adapter under
   `Forbid` and observes the refusal, at every adapter that forwards execution.

## What "policy-independent execution" means, and what it does not

Under both values the provider selects the same dispatch and the same kernel. That is a deliberate
restriction: it can refuse a call under `Forbid` even where a different, strided kernel would have
consumed the input directly. The alternative — switching kernels to satisfy `Forbid` — creates a
second numerical path, which this change forbids anyway.

It does **not** mean bit-identical output. Parallel reductions and vendor kernels can differ between
two runs of the same path, and two different paths can agree bit for bit. So the tests are: policy
does not change dispatch (asserted directly, per operation), plus each operation's existing
numerical comparison contract, unchanged.

## Proof

No public counter and no obligatory hot-path instrumentation. Instead: test-only instrumentation at
the normalization boundary (a test build records that a normalization happened and for which
operand), and tests that assert refusal under `Forbid` and at most one normalization per operand
under `Allow`. "`Forbid` returned `Ok`, therefore nothing was copied" is not a proof — an
implementation that copies silently and returns `Ok` satisfies it.

## Out of scope (not folded into this change)

- Repairing the three entry shapes (linalg `*_with_exec`, einsum trait methods, `FftExecutor`) and
  establishing `_into` primacy. Claiming it here would be false.
- New `_into` operations, output-allocation policy, `*Options` structs, argument-order cleanup.
- Moving FFT plan caches into engines; insulating foreign types; cargo features and CUDA gates.
- Any public profiling surface.

## Still open — name these before implementation

1. `NEED`: the traced/preparation entry signatures, and where a run receives its constraint.
2. `NEED`: the einsum scratch owner and its admission boundary; the FFT lane-buffer owner and
   retention path. Both decide which side of the copy definition today's copies fall on.
3. **Composite execution**: a graph can execute early nodes before discovering that a later
   intermediate needs normalization. Whether the policy and its atomic admission cover the whole
   run or each operation, how actual runtime layouts invalidate a fact recorded at preparation, and
   where an eager backward pass receives its constraint — all undecided here.
4. The landing order: R3's amendment, the shared rule in `tensor4all-agent-rules`, and the
   implementation have to move together, or the repository spends a day contradicting its own rules.

---

## Appendix: the four-file check (rustc 1.96.0, macOS arm64)

Each file is compiled with
`rustc --edition 2021 --crate-type lib -o /tmp/<name>.rlib /tmp/<name>.rs`.

### `t1.rs`
```rust
#![deny(unused_variables)]
#[derive(Clone, Copy, Default)]
pub enum MaterializationPolicy { #[default] Allow, Forbid }

pub struct Svd;
pub trait Engine {}

// 主張 A: 位置引数にすると「sugar が policy を落とす」のをコンパイラが止めるか
pub fn svd_into(_a: &[f64], _s: &mut dyn Engine, _m: MaterializationPolicy) -> Result<Svd, ()> { Ok(Svd) }

pub fn svd(a: &[f64], s: &mut dyn Engine, materialization: MaterializationPolicy) -> Result<Svd, ()> {
    svd_into(a, s, Default::default())   // ← 呼び出し元の policy を落としている
}
fn main() {}
```

### `t2.rs`
```rust
pub trait Engine {}
pub struct Exec<'a>(&'a ());
pub struct Req<'a, R: ?Sized>(&'a mut R);
impl<'a> Req<'a, dyn Engine> { pub fn new(e: &'a mut dyn Engine) -> Self { Req(e) } }
impl<'a> From<&'a mut dyn Engine> for Req<'a, dyn Engine> { fn from(e: &'a mut dyn Engine) -> Self { Req(e) } }

// 汎用 Into で受ける版（Fable が「壊れる」と言った形）
pub fn einsum_generic<'a, I: Into<Req<'a, dyn Engine>>>(_i: I) {}
// 具体型で受ける版
pub fn einsum_concrete<'a>(_i: Req<'a, dyn Engine>) {}

pub fn helper(s: &mut dyn Engine) {
    einsum_concrete(Req::from(&mut *s));   // 具体型なら再borrowで 2 回呼べる
    einsum_concrete(Req::from(&mut *s));
}
fn main() {}
```

### `t4.rs`
```rust
pub trait Engine {}
pub struct Req<'a, R: ?Sized>(&'a mut R);
impl<'a> Req<'a, dyn Engine + 'a> { pub fn new(e: &'a mut (dyn Engine + 'a)) -> Self { Req(e) } }
pub fn einsum_concrete<'a>(_i: Req<'a, dyn Engine + 'a>) {}

pub fn helper(s: &mut dyn Engine) {
    einsum_concrete(Req::new(&mut *s));   // 1 回目（再borrow）
    einsum_concrete(Req::new(&mut *s));   // 2 回目
}
fn main() {}
```

### `t6.rs`
```rust
pub trait Engine {}
#[derive(Clone, Copy)] pub enum MaterializationPolicy { Allow, Forbid }
pub fn einsum(_s: &str, _e: &mut dyn Engine, _m: MaterializationPolicy) {}
pub fn helper(s: &mut dyn Engine, m: MaterializationPolicy) {
    einsum("ij->i", s, m);
    einsum("ij->i", s, m);
}
fn main() {}
```

Observed outcomes:

| file | what it asks | outcome |
|---|---|---|
| `t1` | does a lint catch a sugar that drops the policy? | `error: unused variable: \`materialization\`` under `#![deny(unused_variables)]` — yes, and only for the unused case |
| `t2` | the wrapper as a reader would naively write it, used from a helper | `error: lifetime may not live long enough` (invariant `dyn Engine`, default object lifetime `'static`) |
| `t4` | the same wrapper with `R: ?Sized = dyn Engine + 'a` | compiles; the helper calls it twice |
| `t6` | the plain argument, `Copy` enum, same helper | compiles; the helper calls it twice with no `reborrow` |

`t3` (the same naive wrapper behind a generic `Into`) fails for the same lifetime reason as `t2`
before the move error matters, which is why the wrapper does not rescue the reborrow problem.
