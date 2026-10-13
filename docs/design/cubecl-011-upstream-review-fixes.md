# Review of the arithmetic fix series (gpt-6.1-sol) and the corrections


## Review of the first PR candidate (gpt-6.1-sol) and the corrections

The arithmetic fix series was reviewed read-only by `openai/gpt-6.1-sol`. Verdict:
do not submit unchanged. Findings, all verified against the source, and what was
done:

| Finding | Action |
| --- | --- |
| **BLOCKER** `SizedType for BoolType` does not make bool buffers portable: WGSL prints `array<bool>`, which is not a host-shareable storage type, and the SPIR-V path has no physical bool buffer either | **held** — see below |
| **BLOCKER** the generic `log1p` polynomial regresses f64 (quartic truncation is millions of ULPs at `x = 1/128`) and is inaccurate around the threshold for f16/bf16 | **held** — see below |
| IMPORTANT bool *vectors*: IR size is `element × lanes` (4 bytes) while LLVM `<4 x i1>` is packed | part of the held bool work |
| IMPORTANT the fold removal was incomplete: `FSubOp` `x-x -> +0`, `FRemOp` `x%1.0 -> x` | **fixed** in the fold commit |
| IMPORTANT the FMA guard missed complex *vectors* (a vector's own type is not a complex scalar) | **fixed**: the predicate now asks the operand's `scalar_ty`; the subject says "real operands" |
| IMPORTANT no focused tests accompany the series | **open** — each PR needs one |
| MINOR the `sqrt(eps)` wording overstates the loss | folded into the held log1p work |
| Evidence wording overstated per-commit attribution | **fixed**: the messages now say the stack was measured and that the change was not measured in isolation |
| Integer simplifiers with the same class of error (`SRemOp`/`URemOp` `x % 1`, `SMulHiOp`/`UMulHiOp` `1 * x`, integer zero-divisor folds) | recorded as a separate contract item, not mixed into this PR |

The corrected candidate is `tensor4all/cubecl` `feat/tenferro-upstream-arith-fixes`
@ `5941a32b9`: two commits on upstream `main` `6bba5fdf`, +29/−59, `cargo check -p
cubecl-ir -p cubecl-opt` and `cargo fmt --all --check` clean. Before approval it
still needs its tests and a re-review.

### Held work and the shape it needs

- **bool storage contract**: define which representations are supported — either a
  backend-specific mapping of bool buffers (u32 arrays plus load/store conversion
  for WGSL/SPIR-V) or an explicit rejection on those backends. The CUDA/C++ and
  CPU/LLVM scalar paths are correct with a sized `BoolType`, so the fork line can
  keep the current patch while upstream settles the contract.
- **log1p**: native where the target has it, with a dtype-appropriate fallback
  (and the doc wording corrected). The quartic/`1e-2` polyfill stays fork-only.

### Why `log1p` needs its own PR: the mechanism, and the ready-made pattern

- `cubecl-cpp`'s lowering for `Log1pOp` does not call the target's `log1p`: it
  inlines the *frontend* polyfill (`cubecl-cpp/src/shared/unary.rs:3,315` import
  `cubecl_core::frontend::polyfills::log1p`, which is `(1 + x).ln()`), while a
  native C++ call is printed by the `shared_op!` mechanism other ops use, e.g.
  `shared_op_with_out!(FRemOp, |op, ctx| format!("fmod({lhs}, {rhs})"))`
  (`cubecl-cpp/src/shared/binary.rs:111`). The fix is therefore to give
  `Log1pOp` a printing lowering (`log1p(...)`, with the half marker as `FRemOp`
  carries) and keep the polyfill only for targets without it (Metal, as `ErfOp`
  already does).
- `cubecl-llvm` already has both paths: `lower_unary_math_arith!(... $gate)` in
  `crates/cubecl-llvm/src/shared/polyfill/math.rs:38-56` emits the *native* libm
  call unless the gate selects the polyfill, and `ExpOp`/`LogOp` use a gate
  (`is_cpu_narrow_float_line`, lines 62-70) while `Log1pOp` has none, so it always
  inlines the polyfill. Giving `Log1pOp` the same shape is the minimal change.
- WGSL has no `log1p`, so a polyfill stays there with a documented accuracy limit.
