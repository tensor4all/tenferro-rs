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

Real, nonempty input with `m >= n` and full column rank. The factorization is non-pivoted and
reduced, with a positive diagonal in `R`, and `tests/extension_qr.rs` checks reconstruction and
orthogonality far below the working precision, the sign convention, the 3-4-5 case, and the
adjoint against an analytic reference. `tests/connected_qr_ad.rs` checks the connected graph's
orientation case, `dL/dA = [[6], [8]]` for `A = [[3], [4]]`.

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

First order is supported, and forward-only use needs no AD registration at all. The rules emit
programs and own no resources; `tests/extension_ad.rs`, `tests/connected_conversion_ad.rs`, and
`tests/connected_qr_ad.rs` run JVP and VJP through the connected graphs. The first-order helpers
the rules emit are terminal: `Df64QrVjp` and `Df64QrJvp` are runtime-registered execution
helpers with no further AD rule, so differentiation *through* them is unsupported by declaration
rather than by accident, and it fails with a typed error instead of a zero gradient
(`tests/ad_rule_boundaries.rs`). Differentiation after the forward session exits is covered by
`ext/scalar-consumer-application/tests/later_backward.rs`.

## Unsupported cases

The AD contract admits first-order field arithmetic: an order other than one returns
`UnsupportedAdOrder`, a scalar that is not a field returns `NonFieldScalar`, and an operation
outside the contributed rule set returns `AdRuleUnavailable`, each covered in
`crates/tenferro-tensor-core/src/scalar/tests.rs` and cross-crate in
`ext/df64-proof/tests/ad_rule_boundaries.rs`. The ordinary einsum surface rejects an externally
defined dtype with a typed error. The core backend's elementwise and reduction operations refuse a
caller-owned payload, and a reduction over no axes returns the caller's value unchanged because it
is the identity for every scalar.
