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
2. **A lifetime pitfall exists in the uncorrected wrapper, not in the submitted one.** A four-file
   check with `rustc 1.96.0` shows `struct ExecutionRequest<'a, R: ?Sized>(&'a mut R)` with
   `impl<'a> ExecutionRequest<'a>` failing for a plain `&mut dyn Engine` (`dyn Engine` is invariant
   in `R`, the default object lifetime is `'static`), and `R: ?Sized = dyn Engine + 'a` fixing it —
   which the submitted wrapper already carries. The same helper compiles untouched with the plain
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
