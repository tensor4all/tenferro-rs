# IEEE-unsound constant folds in CubeCL 0.11's math dialect

Found while fixing the A100 failures; not yet reported upstream.

## The folds

`crates/cubecl-ir/src/dialect/math.rs` gives the float arithmetic ops
`const_eval!` rules that fire when *either* operand is a compile-time constant.
Several of them are invalid under IEEE 754:

| Line | Rule | Counterexample |
|---|---|---|
| 390-397 | `0 * x -> 0`, `x * 0 -> 0` | `-4.0 * 0.0` is `-0.0`; `NaN * 0.0` is `NaN`. Both collapse to `+0.0`. |
| 256-265 | `0.0 + x -> x`, `x + 0.0 -> x` | `+0.0 + -0.0` is `+0.0`, not `-0.0`. |
| 505-517 | `0 / x -> 0` | `0.0 / 0.0` is `NaN`; `0.0 / -0.0` is `NaN`; `0.0 / inf` is `0.0`. |
| 636-646 | `0 % x -> 0` | `0.0 % 0.0` is `NaN`; `-0.0 % 2.0` is `-0.0`. |
| 687-697 | (same for `math.f_mod_floor`) | as above |

The multiply rule is the one that bit tenferro. Its `custom` branch takes the
constant operand and returns a `+0.0` attribute of the same type, so
`x * 0.0f32` — an idiom whose whole purpose is to carry `x`'s sign into a zero —
became a plain `+0.0`. Reproduced on the A100 with a three-lane probe kernel:

```
x=-4.0  rem=0x00000000 x*0=0x00000000 branch=0x00000000
```

`x * 0.0` should be `0x80000000`.

The rules are not sign-aware on purpose (the add rule shows the intent), so this
is a class of defect rather than one typo. Note the `simplify!` rules next to
them are narrower (`* 1.0`, `+ 0.0`, `% 1.0`), and the LLVM path explicitly
lowers `FRemOp` with fast math off (`cubecl-llvm/src/shared/to_llvm/math.rs:404`).

## Tenferro sites that the folds change

- `kernels/elementwise.rs`, the two float remainder kernels: `x * 0.0` (fixed by
  using `%`).
- `kernels/elementwise.rs`, `scalar_real_complex_binary`: the component formulae
  deliberately keep the zero cross terms ("they determine NaN, infinity,
  overflow, and signed zero"), and the folds change them:
  - `zero + im` with `im = -0.0` must be `+0.0`; the fold returns `im`.
  - `zero * im` with `im = inf` must be `NaN`; the fold returns `+0.0`. Same for
    `im * zero`, `zero * re`, `re * zero` — the fold silently removes NaN from
    a mixed complex multiply.
  - `zero - im` and `norm_sqr = scalar*scalar + zero*zero` happen to agree.
  Because a runtime `+0.0` cannot be derived from an arbitrary runtime float
  (`x - x` is NaN for `x = inf`), tenferro cannot express these formulae with a
  non-constant zero either; either the folds are fixed upstream or each site is
  rewritten as an explicit compare-and-select, which is the local workaround the
  migration rules forbid.

## What tenferro changed

`crates/tenferro-gpu/src/kernels/elementwise.rs`: the two float remainder
kernels used the textbook formula plus `if remainder == 0 { x * 0.0 }` to restore
the dividend's sign. They now use `%`, i.e. `math.f_rem`, which lowers to `fmod`
(`frem`), so the sign comes out of the operation instead of a fold-sensitive
trick.

## Upstream disposition (open)

Worth reporting: the folds change results for signed zeros and NaN, and a
correctly-written kernel can be silently changed by them. Options are (a) an
upstream bug report plus a fix (skip the fold unless the constant is a signed
`+0` *and* the other operand is known not to be `-0`/NaN, or drop the rule), or
(b) nothing, if upstream considers these folds intentional fast-math. Either way
tenferro no longer depends on them. Needs the user's approval like any other
`tracel-ai/*` report or PR.
