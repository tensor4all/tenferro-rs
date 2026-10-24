# Downstream host scalars: indexed column-major views (#1903, #1787)

## Decisions

- `ColMajorView` / `ColMajorViewMut` take the `[usize; N]` index spelling, and
  `TypedTensor<T, Rank<N>>` takes it too. The checked forms (`get`, `get_mut`)
  stay the non-panicking entry points, and the panic message names the shape, so
  a dense index loop in downstream code no longer needs `?` or `unwrap` on
  every element read. This is #1903's item 3.
- Both representations take `[usize; N]`. `Dynamic` is the type
  `from_vec_col_major` returns, and `Host` is what `into_host` narrows to, so a
  downstream caller reaches either.
- `Host` needs four spelling impls, not two. Adding `Index<[usize; N]>` next to
  the pre-existing `Index<&[usize]>` makes rustc stop inferring
  `tensor[&[0, 1]]`'s integer literals as `usize` — the five existing call sites
  in `tests/representation_boundaries.rs` stop compiling with "cannot be indexed
  by `&[{integer}; 2]`" — because neither impl is then the unique match for a
  by-reference array. Adding `Index<&[usize; N]>` (and its `IndexMut` twin,
  delegating to the by-value impl) gives that spelling an exact match and the
  slice spelling keeps its own. Verified by removing them again: the five sites
  break.
- The owned column-major view accessors no longer require `T: TensorScalar`. The
  bound was never load-bearing: the bodies only need `assert_col_major_contiguous`,
  `layout`, `as_slice` and `host_data_mut`, all of which are available without it,
  and deleting it leaves `TensorScalar` unimported in `col_major.rs`. That is what
  makes #1903's independent-use case real — a downstream crate storing its own
  scalar reaches the validated views with no tenferro session, no extension
  registration and no numerical or AD backend. `TypedTensorView::host_col_major_view`
  was already unbounded, so the two families now agree.
- The independent-use case is a test, not a doc example:
  `crates/tenferro-tensor/tests/downstream_host_operation.rs` runs one generic
  column-major matmul over the views for `f64` and for a plain `Weight(i32)`
  struct that is not a tenferro scalar, and asserts the same source lines serve
  both. The same test reaches those views from a `Host`-marked owner through
  `into_host`, so restoring the old `T: TensorScalar` bound on either accessor
  would fail it.

## Rejected alternatives

- Adding `Index<&[usize]>` for the default representation as well, to make the two
  spellings symmetric across markers. It is a second public indexing path for a
  case no caller has asked for, and the asymmetry it would remove is between a
  pre-existing impl and this one, not introduced here.
- Leaving the array spelling off the `Host` marker so only the default
  representation has it. That is one spelling per representation and no extra
  impls, but `into_host` is an ordinary narrowing step, and a caller who takes
  it would lose `t[[i, j]]` for no reason the caller can see.
- Replacing the pre-existing `Index<&[usize]>` (`R: TensorRank`) with a
  static-rank-only `&[usize; N]` impl instead of adding one. It would drop slice
  indexing for a `Host` tensor whose rank is only known dynamically.

## Residual risks

- `Host` carries `Index<&[usize; N]>`/`IndexMut<&[usize; N]>` purely to keep the
  by-reference array literal resolving. They are not a capability of their own:
  they delegate to the by-value impl. Deleting them is both a compile break for
  an inferred `&[i, j]` literal and an API removal for any caller that bounds on
  `Index<&[usize; N]>`. A reader may still mistake them for a third supported
  spelling.
- Widening the bound is additive and cannot break a caller, but it does mean a
  future per-scalar validation added to those bodies would have to be written
  without `TensorScalar`. Nothing in the current bodies wants one.
- Not measured: whether the `[usize; N]` path compiles to the same code as
  `get` (it delegates, so it should) and whether a downstream caller's index
  loop is measurably faster. Both are downstream claims (#1903), not this
  change's.

## Verification

- `cargo test -p tenferro-tensor`: 331 lib tests, 24 integration binaries and 575
  doctests pass. `tests/ui/storage/*` trybuild cases report path-normalization
  mismatches in this environment; they reproduce unchanged on a clean
  `origin/main` worktree, so they are not attributable to this change.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-tensor'`
