# Einsum/Prims Follow-up Fix Design

**Date:** 2026-02-23
**Scope:** `tenferro-prims`, `tenferro-einsum`, `chainrules`
**Goal:** Fix three correctness/safety gaps found in post-#144 review.

---

## 1. Fix `UnaryOp::Conj` in `execute_elementwise_unary`

**Problem:** `execute_elementwise_unary`'s `UnaryOp::Conj` branch treats conjugation as identity for all types, including complex. While `resolve_conj()` handles real conjugation correctly via a separate path, the explicit unary op path is broken for `Complex64`/`Complex32`.

**Fix:** In the `Conj` match arm of `execute_elementwise_unary`, call `.conj()` on each element. For real types this is identity; for complex types it negates the imaginary part.

**Files:** `tenferro-prims/src/lib.rs`, `tenferro-prims/tests/prims_tests.rs`

**Tests:** Add explicit `Complex64`/`Complex32` tests that run `ElementwiseUnary::Conj` through `plan()`+`execute()` and verify imaginary parts are negated.

---

## 2. Harden `plan()` Validation

**Problem:** Three descriptors have insufficient validation in `CpuBackend::build_plan()`:

| Descriptor | Gap |
|---|---|
| `ElementwiseUnary` | No rank check, no shape equality |
| `ElementwiseMul` | No validation beyond shape count |
| `MakeContiguous` | No rank check, no shape equality |

Same-rank but different-dimension inputs pass `plan()` silently, then panic in `execute()`.

**Fix:**
- `ElementwiseUnary`: Add `validate_rank` + `validate_shape_eq` (input == output).
- `ElementwiseMul`: Add `validate_rank` for all 3 + `validate_shape_eq` (A == B == C).
- `MakeContiguous`: Add `validate_rank` + `validate_shape_eq` (input == output).
- Use existing error variants (`RankMismatch`, `ShapeMismatch`).

**Files:** `tenferro-prims/src/lib.rs`, `tenferro-prims/tests/prims_tests.rs`

**Tests:** Negative tests with same-rank but different dimensions for each descriptor. Verify `plan()` returns `Err`.

---

## 3. Reject Mixed-Tape Operands in `tracked_einsum`

**Problem:** `tracked_einsum` uses `find_map` to grab the first tape from any `requires_grad` operand. If operands come from different tapes, the second tape is silently ignored, corrupting pullback.

**Fix:**
- Add `Tape::same_tape(&self, other: &Tape<V>) -> bool` using `Rc::ptr_eq` on the inner `Rc<RefCell<TapeInner<V>>>`.
- In `tracked_einsum`, after finding the first tape, verify all other `requires_grad` operands share the same tape. Return AD error if not.

**Files:** `extern/chainrules/src/lib.rs`, `tenferro-einsum/src/lib.rs`, `tenferro-einsum/tests/einsum_tests.rs`

**Tests:** Create two `Tape` instances, create leaf tensors on each, pass both to `tracked_einsum`, expect error. Keep existing single-tape AD tests unchanged.

---

## Commit Strategy

1. `fix(prims): make UnaryOp::Conj true conjugation for complex types`
2. `fix(prims): strengthen plan validation for ElementwiseUnary/Mul/MakeContiguous`
3. `fix(einsum): reject mixed-tape operands in tracked_einsum`

Each commit includes its tests.
