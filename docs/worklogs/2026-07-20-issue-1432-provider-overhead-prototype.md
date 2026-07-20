# Issue #1432 provider-boundary overhead prototype

## Purpose

Measure the cost of the work-in-progress CPU provider boundary before changing
the public API. The prototype stays in a benchmark target and does not add a
provider abstraction to library code.

## Measurement plan

- Compare borrowing the existing `DotGeneralConfig` vectors with constructing
  `SmallVec<[usize; 8]>` per call, borrowing a prepared `SmallVec`, and copying
  into fixed inline arrays.
- Compare the current `HashSet`-based validation with allocation-free linear
  scan and `u64` bitset validation for ranks 2, 3, 4, 8, 9, 16, and 32.
- Isolate direct calls, dynamic provider dispatch, decorator fallback, an
  atomic thread-budget read, string-keyed registry lookup, and resolved-slot
  lookup.
- Measure complete eager `dot_general` calls for 1x1, 2x2, and 8x8 matrices on
  a one-thread CPU backend.
- Report allocation count, allocated bytes, and representation size separately
  from Criterion timing.

## Controls

Run in release mode with external numerical-library and Rayon thread counts set
to one. Inputs and results cross `criterion::black_box`. Preparation is kept
outside steady-state measurements except in explicitly named `per_call` cases.

## Decision boundary

This prototype may reject per-call representation conversions or string lookup.
It does not by itself establish the final provider trait, request type, cache
ownership, or resource-arbitration API.

## Results

Measured on an AMD EPYC 7713P with Rust 1.96.0 at `85855e27`, using the
release profile and one-thread environment controls. Criterion `--quick` was
used for the full matrix; dispatch was repeated with 1 s warm-up, 3 s
measurement, and 100 samples.

Representation sizes were 80 bytes for the borrowed request, 336 bytes for the
four-`SmallVec` request, and 1072 bytes for the four fixed `[usize; 32]` arrays.

| Case | Rank 4 | Rank 16 | Rank 32 | Per-call allocation at rank 32 |
|---|---:|---:|---:|---:|
| Borrow existing `Vec` as slices | 6.44 ns | 7.06 ns | 7.00 ns | 0 |
| Construct four `SmallVec<[usize; 8]>` | 25.44 ns | 29.93 ns | 72.46 ns | 4 / 512 B |
| Borrow prepared `SmallVec` | 7.31 ns | 6.75 ns | 6.18 ns | 0 |
| Copy into four fixed arrays | 53.07 ns | 59.30 ns | 59.94 ns | 0 |

The representative configuration splits axes evenly between contracting and
batch roles. Therefore rank 32 exceeds the inline capacity in all four
`SmallVec`s, while rank 16 fits exactly.

| Validation | Rank 4 | Rank 16 | Rank 32 | Per-call allocation at rank 32 |
|---|---:|---:|---:|---:|
| Current `HashSet` implementation | 307.03 ns | 1.555 us | 2.867 us | 16 / 2416 B |
| Allocation-free linear scan | 21.08 ns | 116.46 ns | 325.49 ns | 0 |
| Four `u64` role masks | 10.18 ns | 28.29 ns | 47.08 ns | 0 |

The bitset is only a validation accelerator: it cannot replace ordered axis
lists because contracting-axis correspondence is order-sensitive. A production
validator also needs a fallback for ranks above 64.

| Dispatch case | Time |
|---|---:|
| Direct function | 5.71 ns |
| Concrete provider | 5.66 ns |
| Dynamic provider | 5.69 ns |
| Decorator, handled | 13.22 ns |
| Decorator, delegated | 10.60 ns |
| Dynamic provider plus relaxed atomic budget read | 6.20 ns |
| String-keyed `HashMap` lookup plus dispatch | 31.47 ns |
| Resolved slot lookup plus dispatch | 6.31 ns |

The complete eager one-thread `dot_general` measured 13.67 us for 1x1,
15.47 us for 2x2, and 16.45 us for 8x8. Each call performed approximately 37
allocations, with 839 B, 862 B, and 1342 B allocated respectively.

## Prototype conclusion

- Keep steady-state provider requests borrowed and slice-based. Constructing
  `SmallVec` or fixed-array forms per call is a regression; a prepared owner can
  use either representation internally without exposing it in the trait.
- Dynamic dispatch is not a meaningful bottleneck in this experiment.
- Resolve extension names to slots during preparation; do not hash strings on
  every execution.
- Optimize or cache validation before attributing tiny-operation overhead to
  the provider boundary. A hybrid bitset/fallback validator merits a separate
  correctness-preserving change.
- Passing a precomputed thread budget is effectively free here. The full
  resource arbiter still needs an implementation-level benchmark once its
  ownership and synchronization contract exists.
