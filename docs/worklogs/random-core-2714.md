# Random numbers, step 1: `cbrng` (#2714)

The crate was introduced as `tenferro-random-core` and renamed to `cbrng` in #2733;
this record uses the new name.

The design is tensor4all-meta `docs/design/random.md`; this records the choices
it left open. The frozen sample layout itself is documented in the crate
(`crates/cbrng/src/lib.rs`), which every backend follows.

## Decisions

- **Word stream instead of "block `i / words_per_sample`".** A key's word
  stream is its blocks at counters 0, 1, 2, ... concatenated, and element `i`
  reads a fixed slice of it (`w i .. w i + w` with `w` words per sample). The
  definition is the same for a 4-word (Philox) and a 2-word (Threefry) block,
  so a sample that needs more words than one block holds (an `f64` normal
  pair under Threefry) is still defined without a per-algorithm rule.
- **Two words per `f64` uniform, first word high; one per `f32`.** The mantissa
  takes the top 52 or 23 bits, as the design's table says. Integers read two
  words as one 64-bit value, first word high, for `I32` and `I64` alike.
- **Real normals come in Box–Muller pairs.** Elements `2d` and `2d + 1` share
  draw `d` (cosine, sine), so no random bits are discarded; the complex normal
  uses one draw per element with radius `sqrt(-log(1 - a))`, so each part is
  N(0, 1/2). `1 - a` in `(0, 1]` avoids `log(0)`; the radius is written
  `sqrt(0 - 2 log)` so that `a = 0` gives `+0`, not `-0`.
- **`uniform_range` is clamped below `hi`.** `lo + (hi - lo) u` can round up to
  `hi`; the result is then the largest value below `hi`, which keeps the
  documented `[lo, hi)` (JAX only clamps at `lo`).
- **Threefry follows JAX exactly, including its shared counter space.** With
  partitionable Threefry (JAX's default since 0.5.0), `split(k)[i]`,
  `fold_in(k, i)` and the sample block `i` are the same words. Matching JAX's
  values was a design requirement, so the consequence is documented rather
  than changed; Philox keeps separate key domains (counter word 3 = 1 for
  split, 2 for fold-in) as the design decided.
- **`log` and `sincospi` are deterministic, not correctly rounded.** Both are
  fixed sequences of IEEE operations that are correctly rounded on every
  backend (`+ - * /`, `sqrt`, explicit `fma`, exact rounding to an integer, bit
  manipulation), which is what bit-identical samples need. A correctly rounded
  `log` would need multi-precision rare-case handling for no gain in the
  contract. `log` uses fdlibm's reduction (`s = f / (2 + f)`) with Taylor
  coefficients `2 / (2j + 1)`; `sincospi` reduces exactly (`x = n/2 + r`) and
  carries `pi r` in two parts. All constants are derived by
  `reference/math_constants.py`, nothing is copied from another library.

## Verification conclusions and constraints

- Block functions match the Random123 `kat_vectors` for both algorithms, and
  the Threefry block, seed, split and fold-in match JAX 0.12.1
  (`reference/jax_threefry.py`).
- Every exact sample (bits, uniforms, ranges, integers) and the Philox split
  and fold-in match an independent Python implementation of the layout
  (`reference/samples.py`); normal samples are frozen bits within 1 ulp of the
  mpmath value of the same Box–Muller draw.
- `log` and `sincospi` are within 1 ulp of mpmath at all fixed points. Two
  sweeps outside the default suite found no error above 1 ulp (the design
  target is 2 ulp): about 140,000 points against mpmath
  (`reference/accuracy_sweep.py`; maxima 0.76 ulp `log`, 0.98 `sinpi`, 0.93
  `cospi` in f64), and all 2^23 `f32` inputs Box–Muller can produce against an
  `f64` reference (the ignored test `f32_functions_over_every_box_muller_input`;
  maxima 0.81 ulp `log`, 0.99 `sinpi` and `cospi`).
- Bit identity with CUDA is not tested here: the CUDA block functions and
  templates are step 2 (t4a-cuda-kernels) and the CPU-against-CUDA test step 4.
  They must evaluate the same operation sequence with `--fmad=false`, no
  flush-to-zero, and IEEE division and square root.
