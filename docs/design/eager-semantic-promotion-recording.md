# Eager Semantic Promotion Recording

## Status

Implementation design for [#1698](https://github.com/tensor4all/tenferro-rs/issues/1698).
This fixes a mismatch between accepted eager primal execution and its deferred
semantic AD carrier. It adds no public API and changes no dtype-promotion
policy.

## Problem

Eager primal execution promotes compatible mixed-dtype inputs immediately
before backend dispatch. Deferred semantic recording currently appends the
original unpromoted inputs to a raw graph. A mixed F64/C64 `Add`, for example,
executes as C64/C64 but is recorded as F64/C64.

Most direct derivatives do not need to replay the primal operation. A residual-
using downstream rule does: an untracked F64 identity plus a tracked C64 matrix,
followed by complex `Eigh`, fails during backward when the imported primal Add
reaches the same-dtype CPU kernel:

```text
add: dtype mismatch: expected F64, actual C64
```

Normal traced construction already inserts explicit `Convert` operations before
promoting binary operations. The raw eager carrier bypasses that path.

## Design

Record the same explicit input promotion that eager execution performs.

1. Add one crate-private promotion-plan helper beside eager execution. Given a
   `StdTensorOp` and input dtypes, it returns the dtype each backend operand
   receives.
2. Use that helper in both owned/read eager execution promotion sites and
   deferred semantic recording, so execution and recording cannot define
   separate promotion vocabularies.
3. In `record_semantic_eager_outputs`, after collecting lazy constants and
   before operation-specific shape canonicalization, append semantic `Convert`
   operations only where the semantic input dtype differs from its execution
   target dtype.
4. Keep the #1692 concatenate exact-shape step after dtype promotion. Both
   transformations exist only in the deferred semantic graph and add no eager
   tensor copy, transfer, materialization, or execution kernel.

The shared plan covers every current eager execution promotion family:

- binary Add/Sub/Mul/Div/Rem/Pow/Maximum/Minimum/Compare/DotGeneral;
- Select value operands, leaving the predicate unchanged;
- all Clamp and Concatenate operands;
- Scatter operand/updates, leaving indices unchanged; and
- DynamicUpdateSlice operand/update, leaving start indices unchanged.

All other inputs retain their original dtype. Arity validation remains owned by
existing operation/execution validation.

## AD contract

The `Convert` nodes make primal replay and derivative metadata identical to the
actual eager computation. Reverse mode projects complex cotangents back to real
tracked leaves under tenferro's real-inner-product convention; inactive real
constants remain inactive and receive no stored cotangent. Forward mode casts
active tangents to the promoted execution dtype.

This fix does not change backend dtype support, insert implicit device
transfers, or make an otherwise invalid dtype pair valid.

## Rejected alternatives

- **Downstream casts in tensor4all.** They hide an accepted tenferro eager
  operation's incorrect AD carrier and duplicate promotion policy.
- **Patch linalg cotangent accumulation.** The linalg seed/local mapping is
  correct; the first invalid operation is the unpromoted primal Add replayed by
  the derivative program.
- **Promote concrete EagerTensor inputs before forward execution.** This repeats
  cast kernels/materialization already owned by eager execution.
- **Fix only Add.** The same raw-recording mismatch exists for every eager
  promotion family listed above.

## Verification

- Unit-test the shared promotion plan for binary, predicate-preserving Select,
  Clamp/Concatenate, index-preserving Scatter/DynamicUpdateSlice, and an
  unchanged unary operation.
- Add exact mixed F64/C64 and F32/C32 eager Add JVP/VJP tests in both operand
  orders, including no gradient storage for inactive operands.
- Keep the end-to-end mixed F64/C64 Add -> complex Eigh eigenvector backward
  regression.
- Add at least one promoted non-Add semantic replay test to prevent a one-op
  fix; a mixed Concatenate or DotGeneral path is sufficient.
- Verify the #1692 mixed-stack matrix remains green.
- Run the focused autodiff integration target, formatting, clippy, and the
  repository local PR gate. Record coverage review and both reviewer-gpt gates
  in a curated worklog.

No tolerance change is permitted. Planning is `O(input_count)` metadata work;
only inputs whose dtype differs gain a semantic Convert node.
