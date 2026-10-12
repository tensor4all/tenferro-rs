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
