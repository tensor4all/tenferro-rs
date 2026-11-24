# tenferro-linalg crate rules

Template and policy: [`README.md`](README.md).

### Scale before you multiply when an intermediate can leave the type's range

- **Applies to:** any expression that multiplies, squares, or sums squares of
  user-derived values before a normalization or scaling step. In this crate:
  eigenvector normalization in `src/eig_normalize.rs` (already scaled, see
  below), and the pivot searches that compare `norm_sqr()` values in
  `src/extension.rs` and `src/extension/gauge.rs`, where two entries above
  about `1.3e154` (`f64`) both square to `inf` and compare equal. `det` is
  built from `slogdet` (`src/eager_composites.rs`), the log-domain form that
  avoids a raw diagonal product; keep it that way. `grep -rn 'norm_sqr\|re \* re + im \* im\|powi(2)'`
  finds the next site. A range check on argument **ratios** does not cover
  this: equal huge arguments have ratio 1 and still overflow.
- **Check:** a test at a magnitude where the naive intermediate overflows and
  the true result is finite (`f32` around `1e30`, `f64` around `2^520`), and
  that asserts a finite value against a reference, for example the result at
  `1` rescaled by the known homogeneity degree. The in-crate pattern to copy is
  `src/eig_normalize.rs`: the column is divided by its largest component before
  squaring ("Scaling by the largest component first keeps every squared term
  in [0, 1]"), and its test `a_huge_f32_column_does_not_square_to_infinity`
  is the check.
  Review question: "which intermediate here is the largest, and can it exceed
  `MAX` when every input is in range?"
- **Incident:**
  [tenferro-sci-rs#25](https://github.com/tensor4all/tenferro-sci-rs/issues/25)
  -- `elliprj(a, a, a, -a)` at `a = 2^520` returns `NaN`. In `rj_principal`,
  [`generic/carlson.rs:349`](https://github.com/tensor4all/tenferro-sci-rs/blob/a2f179ba74669707fb4fba4c04db3c01a4614329/crates/specials/src/generic/carlson.rs#L349)
  forms `x * z` (`2^1040`) for the `rc` argument before `rj_scaled` applies its
  `ldexp` scaling, and the entry point only rejects argument ratios wider than
  `RJ_RATIO_BITS = 600`. The same shape is in `rj_cpv_dispatch`,
  [`xsf/carlson.rs:617`](https://github.com/tensor4all/tenferro-sci-rs/blob/a2f179ba74669707fb4fba4c04db3c01a4614329/crates/specials/src/xsf/carlson.rs#L617)
  (`let xy = x * y;`). It was found by review, not by a test.

### Before publishing a ratio between two arms, check that both time the same work

- **Applies to:** `benches/` that time derivatives and any number from them
  quoted in a PR, an issue, or `tenferro-benchmark`. In `lu_ad_breakdown.rs`,
  the `lu_jvp_sum_lu_outputs` arm times `s.jvp(..)` of the LU loss plus
  `consume_f64` on the result, and `lu_forward_unpack` times `s.lu(..)` plus
  `consume_many` on four outputs; both consumers enter a serial eager session
  and `duplicate_value` each output inside the timed closure. The arms are
  components of one breakdown with different consumption work, not two
  implementations to divide.
- **Check:** before quoting "X is N times slower", list what each arm does
  inside the timed closure, call by call, and confirm the lists differ only in
  the thing being compared. A rebuilt return value, a second forward, a
  setup step, or a different thread count inside one arm makes the ratio a
  comparison of workloads, not of implementations. The shared rule for what a
  table may compare is `rules/benchmark/tables.md` in `tensor4all-agent-rules`.
- **Incident:**
  [tenferro-benchmark#517](https://github.com/tensor4all/tenferro-benchmark/pull/517#issuecomment-7130131155)
  -- the eager SVD/eigh VJP cells were published as about 1.95x slower at
  256x256 (4T/8T). The correction: the new harness "builds the return value
  after the VJP", so its measured region contains the same forward a second
  time, and "the ratio compares one forward against forward+VJP+forward, not
  two ways of computing the same quantity."
