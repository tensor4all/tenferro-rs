# AD through an FFT in the middle of a chain (#2647, #2648)

## Decisions

- `SemanticProgramBuilder::resolved_extent` (`crates/tenferro-runtime/src/program/builder.rs`)
  folds an extent that names no input to `DimExpr::Const` by evaluating it. A bound program can
  leave `8 / 2 + 1` unfolded, and a core `slice` VJP needs a literal extent. An unevaluable
  expression (zero divisor, overflow) keeps its form instead of being reported, because it is
  still a valid declaration.
- The FFT adjoint length restore (`crates/tenferro-fft/src/lib.rs`) truncates a zero-padded tail
  with one static core `Slice` to the input extents. `DynamicTruncate` + `PadToMatch` publish
  `max(n, input_extent)` even though their runtime value is already the input extent, and an
  upstream op consumes that declared extent. The pad direction keeps the dynamic pair:
  `max(n, m) = m` there, and `DynamicTruncate` already clamps. The restore tracks the adjoint's
  literal shape across axes so that a multi-axis `fftn` with mixed pad and truncate axes slices
  with the correct non-target limits.
- `PadToMatch` semantics are unchanged: its rustdoc declares a larger reference a no-op, so
  truncating there would be a public behaviour change outside a bug-fix change. A trailing
  `Reshape` to the primal shape was rejected as well: the CPU reshape can materialize a copy.
- A staged `Slice` bakes the extent resolved at transform time, which is the assumption the AD
  transform already makes elsewhere (`bounded_usize_shape`); a program whose symbolic input
  extents were only bound at trace time therefore carries those literal lengths in its gradient.

## Verification conclusions and constraints

- Both regressions fail with their own hunk reverted: the slice-VJP unsupported-metadata error
  and the upstream-`mul` shape mismatch. The finite-difference checks use a quadratic loss, so
  the central difference is rounding-limited; the tolerances are `1e-8`.
- Remaining limitation: when the primal input extent is symbolic, a static slice cannot express
  it and the truncation direction still publishes `max(n, extent)`. The runtime value is already
  the input extent and the consuming core rule truncates to the primal shape, so nothing new
  fails, but the declaration is not repaired there.
- The real multi-axis transform is covered numerically only; `Mul` tolerates a `Max` extent, so
  the declared-extent repair on that path is pinned by the one-dimensional `c2c`/`rfft` cases.
- Independent review of the finished diff by `ollama-cloud/glm-5.3-flash:high`: `VERDICT:
  APPROVE`, no blockers; its findings are reflected in the decisions above.
