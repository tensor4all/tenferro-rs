# Review decision records

> Names in this record predate the 0.12 call grammar (#2552); see `docs/spec/generated/api-ops.md` for the current surface.

tenferro keeps three different kinds of long-lived engineering notes:

- `docs/plans/`: historical implementation plans and prior discussions. These
  files can become stale and should not be updated to match current code.
- `docs/worklogs/`: curated records of completed nontrivial work. These explain
  what the implementer read, what reference code informed the change, which
  design was chosen, which alternatives were rejected or deferred, and what
  risks remain.
- `docs/design/`: durable design intent that future implementation and review
  should continue to follow.

## PR workflow

Nontrivial refactors, cleanup streams, AI-assisted implementation, and PRs that
make explicit tradeoffs should add or update a work log. The PR body should
link that work log so reviewers can evaluate the diff against the actual design
context instead of inferring intent from the changed files alone.

If the PR establishes a design rule that should outlive the session, the same
PR should also update `docs/design/`. A work log may explain why a decision was
made in one session, but design docs are the source reviewers should use when
the decision applies to future code.

## Review workflow

Before challenging a nontrivial abstraction, module split, macro/codegen choice,
public API boundary, or deferral, reviewers should read the linked work log and
any linked design docs. A review can still disagree with the decision, but it
should address the recorded rationale directly.

This keeps review feedback aligned with repository intent and avoids repeatedly
re-litigating decisions that were already made explicit.

## Recorded decisions

Short API-contract decisions that have no larger design document of their own.

### Unused placeholder in `compile_with_input_specs` (#1968)

`GraphCompiler::compile_with_input_specs` rejects a declared placeholder that
the output does not depend on (`Validation` / `InvalidArgument`, phase
`Compile`). Before, such a placeholder was silently dropped from the program
inputs, so a tensor passed for it at run time was either rejected as an extra
input or, when the input counts happened to match, bound to a retained constant
input in its place. Keeping unused placeholders as ignored program inputs was
rejected because it needs a semantic input that no operation reads; rejecting
the declaration is the fail-fast contract. Callers drop the binding, as the
kdv-pinn `pde::tests` do for the constant third derivative of `x^3`.

### Concrete session surface parity (#1858)

- **Receiver forms.** `TensorSessionOpsExt` exposes the indexing (`gather`,
  `scatter`, `slice`, `dynamic_slice`, `pad`, `concatenate`, `reverse`),
  reduction (`reduce_max`, `reduce_min`, `reduce_prod`, `reduce_sum_squares`),
  structural (`broadcast_in_dim`, `tril`, `triu`, `extract_diag`, `embed_diag`),
  dot (`dot_general`, `dot_general_with_conj`) and scaling (`scale_real`,
  `scale_complex`) operations as thin forwarders over the borrowed session,
  with the argument order and config types of the eager method of the same
  name and the session last. `concatenate` has no receiver, as on the eager
  session. Scaling shares `tenferro_runtime::scale` with the eager surface, so
  factor rounding and rejection are identical.
- **Typed narrowing.** `TypedTensorSessionOpsExt` gains the reductions, the dot
  family and scaling, whose backend hooks read borrowed views. The indexing,
  padding, concatenation and triangular/diagonal hooks take owned erased
  `&Tensor` inputs, so a typed forwarder would have to copy its receiver first
  (hidden materialization). Those stay on `Tensor`; `Tensor::from_typed` moves a
  typed tensor there without a copy. Typed bool-mask selection remains
  `TypedTensorMaskSessionOpsExt::where_select`.
- **Eager in-place FFT (option B2).** `fft_in_place` / `ifft_in_place` keep
  their consuming signature and `EagerFftInPlaceError`, and live on the
  separate `EagerTensorFftExt` trait, while ordinary eager FFTs are
  `&self`-style methods on `EagerSessionFftExt`. `From<EagerFftInPlaceError>
  for tenferro_ad::Error` drops a rejected input and keeps its error, so `?`
  works; callers that need the input back match on `Rejected`.
- **No concrete in-place FFT.** The consuming eager form exists to guarantee
  that no implicit copy happens under the eager ownership rules. A concrete
  `fft_in_place` would need its own in-place hook in the FFT backend SPI and
  the CPU lane kernel; it is not added until a concrete use case needs it.
