# Two-operand einsum shortcut (tenferro-rs#2324, item A)

## Decisions
- `einsum`, `einsum_subscripts`, the typed and read variants, and
  `ConcreteEinsumPlan::execute*` skip `ContractionTree` planning for two
  operands when the labelled path would end in exactly one `dot_general` with
  nothing before or after it, and issue that `dot_general` directly. This is
  the same `DotGeneralConfig` the tree path builds, so the plan cache and the
  kernel are shared and the bits are unchanged.
- The shortcut is taken only for: unique labels, at least one contracted
  label, every non-output label present in both operands, output order equal to
  lhs-free, rhs-free, batch, equal operand dtypes, and equal sizes on
  contracted/batch labels.
- Declined, falling back to the tree path unchanged: repeated labels
  (diagonals), a label summed in one operand only, outer products (no
  contracted label), other output orders (the tree path transposes), size-1
  broadcasting of contracted/batch axes, and mixed dtypes (promotion). The
  einsum surface has no conjugation flag, so conj never reaches this path.
- A plan whose exact-output form needs the operands swapped (for example
  `ij,jk->ki`) is declined too: the tree path runs the original-order
  `dot_general` and then transposes, so running the swapped product would be
  faster but is a different kernel call whose bits are not guaranteed equal.
- Planned follow-ups left out: caching `ConcreteEinsumPlan` per
  (labels, shapes, dtype) for 1-2 operands (item B) and a swapped-order
  shortcut (needs a bitwise argument).

## Verification conclusions and constraints
- Tests compare the shortcut against the tree path with `to_bits` for f64 and
  c64 at rank 2 and 3, batch labels, scalar results, and assert which cases
  are routed to the shortcut.
- A/B timings are in the PR description (single thread, one pinned core).
