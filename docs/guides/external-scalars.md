---
title: "Adding an external scalar"
---

This guide runs the external-scalar path end to end: a scalar type tenferro does not declare,
carried through storage, views, arithmetic, conversions, an external QR factorization, first-order
AD, and two consumer roles. Every claim below names the test that carries it, so the guide can be
checked rather than believed.

The code lives in three unpublished crates:

| Crate | Role |
| --- | --- |
| `ext/df64-proof` | The contribution: the scalar, its arithmetic, QR, extension operations, and first-order AD rules. |
| `ext/scalar-consumer-algorithm` | The algorithm role. States the capabilities it needs and names no scalar, provider, or dtype. |
| `ext/scalar-consumer-application` | The application role. Binds standard support, or standard support plus the contribution, and declares its own set. |

## Run it

```bash
# The contribution: storage, views, arithmetic, QR, AD, boundaries, and the two measurements.
cargo test -j 16 -p tenferro-df64-proof

# Both consumer roles, including the four configurations and the second containing set.
cargo test -j 16 -p tenferro-scalar-consumer-algorithm -p tenferro-scalar-consumer-application

# Object-level evidence: the contribution's instantiations are parameterized by the scalar and
# the operation, and no set type appears in them.
python3 scripts/check-scalar-composition-kernel-sharing.py \
  --report /tmp/kernel-sharing.md \
  --against-package tenferro-scalar-consumer-application \
  --against-target contribution_reuse --against-sets 2 \
  --set-type-names ExtendedSet,ExtendedTag,ApplicationSet,ApplicationTag

# The cost of the session and dispatch layer around the bodies, one worker thread, both profiles.
cargo test -j 16 --release -p tenferro-df64-proof --test dispatch_overhead -- --nocapture
```

## Dependencies and provenance

The proof crate depends on no third-party scalar library. Its `Df64` is a local two-`f64`
expansion, which is enough to carry `2^-80` through a subtraction that an `f64` round trip
destroys (`tests/composition.rs`). A production adapter would use
[xprec-rs](https://github.com/tuwien-cms/xprec-rs) (MIT) in the same place; that dependency is
deliberately absent here, and the workspace-wide `half 2.7.1` entry exists only because the GPU
stack uses it. The design note records the version and licence of the production-shaped crate.

## Numerical domain and contract

Real, nonempty input with `m >= n` and full column rank. The factorization is reduced and
non-pivoted, and it is computed by modified Gram-Schmidt with one re-orthogonalization pass, which
is what keeps a column that is close to dependent orthogonal to working precision. `R` carries a
positive diagonal because the diagonal is the square root of a sum of squares, and a defensive
flip keeps a column and its diagonal entry consistent should a norm ever come back negative.

The near-singular treatment is exact rather than thresholded. A column whose computed norm is
exactly zero has no unit vector and is refused with a typed error; there is no tolerance that
declares a column near-singular, so a nearly dependent column proceeds and is handled by the
extended scalar's precision together with the re-orthogonalization pass. Rank-deficient input is
outside the supported domain rather than approximated.

`tests/extension_qr.rs` checks reconstruction and orthogonality below `1e-30`, the sign
convention, the 3-4-5 case, and the adjoint against an analytic reference, so the tolerances are
the extended scalar's own rather than an `f64` threshold. `tests/connected_qr_ad.rs` checks the
connected graph's orientation case, `dL/dA = [[6], [8]]` for `A = [[3], [4]]`.

Everything outside the domain is a typed error rather than a wrong answer:
`tests/extension_boundaries.rs` covers a rank-1 input, a wide matrix with more columns than rows,
and a zero column, and `the_factorization_derivatives_refuse_a_singular_factor` covers the
derivatives' own requirement that the triangular factor be invertible.

Outside the domain the answer is a typed error, never a silently wrong one: complex scalars, wide
or rank-deficient input, higher-order AD, and the backend operations the contribution does not
implement all refuse. `docs/design/scalar-composition.md` §5.21 lists every boundary, whether it
is present, rejected by design, or missing, with the test that shows it.

## What is inherited, and what the contribution adds

Inherited from tenferro: the storage model and its lifetime rules, the typed-dispatch entry points,
sessions and their admission, the AD engine and its rule roles, canonical promotion and conversion
policy, and the standard numerical kernels, which are not modified. The contribution adds its
scalar's arithmetic, its QR, its directed conversions, and its first-order rule families.

The object-level check above is the evidence for the boundary: in a program that uses two sets
containing the contribution, every contribution instantiation is parameterized by the
contribution's scalar and operation, and no set type appears in the parameters at all.

## Selection and conversion

The application registers the CPU engine and installs the contribution's extension module, then
selects the support the algorithm states. Conversions are directed and explicit, with the four AD
combinations the `ad-contract` specification fixes; `tests/connected_conversion_ad.rs` shows that
narrowing does not recover the discarded low component while widening returns the expected `f64`
values (`[6, 8]` in the loss example). Conversion is a pair capability, not a promotion rule: no
multi-hop path is created.

## Resource ownership

One CPU owner per backend, with the thread budget stated at construction. The contribution's
scratch goes through the runtime's accounted extension cache rather than a private pool, which is
what makes it visible to `Runtime::cache_stats` and releasable by `Runtime::clear_caches`;
`tests/scratch_allocation.rs` records the allocation counts and the retained bytes, and
`tests/retention_controls.rs` shows that clearing releases the retention without touching a value
that was live across it. Two owners that share a compatible CPU domain may hand a value over, which
`ext/scalar-consumer-application/tests/configurations.rs` exercises with disjoint CPU slices under
distinct engine identities, and the typed-rejection path is asserted in case the contract tightens.

## Derivative order and helper closure

First order is supported, and forward-only use needs no AD registration at all. This is a
contribution-owned rule set registered into the active `AdContext`, not a claim about mainline
`tensor-ad-oracles` support: the owning issue makes adding an oracle family a condition for
claiming mainline AD, and nothing here does. The rules emit
programs and own no resources; `tests/extension_ad.rs`, `tests/connected_conversion_ad.rs`, and
`tests/connected_qr_ad.rs` run JVP and VJP through the connected graphs. The first-order helpers
the rules emit are terminal: `Df64QrVjp` and `Df64QrJvp` are runtime-registered execution
helpers with no further AD rule, so differentiation *through* them is unsupported by declaration
rather than by accident, and it fails with a typed error instead of a zero gradient
(`tests/ad_rule_boundaries.rs`). Differentiation after the forward session exits is covered by
`ext/scalar-consumer-application/tests/later_backward.rs`.

## Standard bfloat16

`ext/bf16-proof` carries the standard `half::bf16` type through the same boundary, and its tests
run with:

```bash
cargo test -j 16 -p tenferro-bf16-proof
```

The representation is `half::bf16` itself; the crate wraps it only because a foreign type cannot
implement tenferro's local scalar traits, which is the same coherence wrapper #1785 accepts for
the extended scalar. Because the payload is caller-owned, this needs no pooled storage: a
*standard* member of the default set would need a new variant in the pool's per-member resource pin,
which is #1789's boundary and stays untouched here.

The contract is stated rather than implied, which is what #1785 asks for:

- **Storage** is bfloat16, so a stored value is the nearest bfloat16 to what was written.
- **A single operation** is computed in `f32` and rounded back once.
- **A reduction** accumulates in `f32` and rounds once at the end, and
  `reduction::sum_in_f32_accumulation` is that contract.

The tests measure the difference between that promise and the weaker one instead of asserting only
that a sum is close. Summing three hundred stored ones gives `300` through the promised
accumulation, while the shared fold — which applies the element type's own addition and therefore
rounds at every step — stalls at `256`, where bfloat16 spacing above one is two. That contrast is
what #1785 asks for when it requires tests to detect unintended per-step rounding.

Conversions are directed and explicit. Widening is exact, because every bfloat16 is an `f32`.
Narrowing rounds to nearest with ties to even: `1 + 2^-8` is the midpoint above one and stores as
`1`, while `1 + 2^-6` is exact. The top of the range is not preserved, and the test says so:
`f32::MAX` narrows to infinity, because its significand is all ones and the rounding carries past
the largest finite bfloat16. The bottom is, because the exponent ranges agree.

## A contraction in the extended scalar

#1793's example is an ordinary einsum: `einsum("ik,kj->ij", A, B)`. The contribution owns that
contraction, so it runs through the runtime's extension module with the extended scalar's own
accumulation:

```rust,ignore
use tenferro_df64_proof::extension::Df64Einsum;
use tenferro_runtime::extension::apply;

let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2])?;   // "ik,kj->ij"
let output = apply(std::sync::Arc::new(op), &[&lhs, &rhs])?;
```

`ext/df64-proof/tests/einsum.rs` reproduces #1793's table exactly — `A = [[1, 2], [3, 4]]` and
`B = [[5, 6], [7, 8]]` contract to `[[19, 22], [43, 50]]` — and runs the precision row the issue
names: contracting the row `[1, 1]` with the column `[1, 2^-80]` keeps `2^-80` after subtracting
one, which an `f64` accumulator cannot. The contrast is asserted in the same test.

Any two-input pattern is accepted, not only the matrix one: a **batched** contraction such as
`bij,bjk->bik`, an **outer product** with no shared label, and a pattern whose label one input names
and the output omits, which the notation sums. Each has its own test with hand-checkable values, and
the summation walks the output index space and accumulates the contracted one in the extended
scalar, so the low component survives a contraction whose accumulation would otherwise round it
away.

What is refused is refused with a typed error rather than approximated: a label that repeats inside
one input, because that is a trace; an output label that no input names; inputs that disagree on the
extent of a shared label; and differentiating the contraction, which fails with the family's own
message that it has no Linearize rule for the operation. That capability matches the reference
consumer in `ext/tropical`, which also refuses diagonal extraction, pre-reduction, and N-ary
contractions — #1787 calls this deliverable "#1793's einsum/tropical example", so matching that
example is the bar, and the refusals are typed on both sides.

## Unsupported cases

The AD contract admits first-order field arithmetic: an order other than one returns
`UnsupportedAdOrder`, a scalar that is not a field returns `NonFieldScalar`, and an operation
outside the contributed rule set returns `AdRuleUnavailable`, each covered in
`crates/tenferro-tensor-core/src/scalar/tests.rs` and cross-crate in
`ext/df64-proof/tests/ad_rule_boundaries.rs`. The ordinary einsum surface rejects an externally
defined dtype with a typed error. The core backend's elementwise and reduction operations refuse a
caller-owned payload, and a reduction over no axes returns the caller's value unchanged because it
is the identity for every scalar.
