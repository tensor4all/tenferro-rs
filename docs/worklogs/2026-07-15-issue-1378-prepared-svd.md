# Issue 1378: Prepared Compact SVD

## Goal

Implement the accepted backend-neutral prepared factorization lifecycle for
compact SVD, with Faer as the first provider and no owned-output fallback.

## Decisions

- Keep `PreparedSvd`, `SvdWorkspace`, and `SvdOutputWrites` public while
  sealing provider dispatch.
- Bind plans and workspaces to retained CPU coordinator/context identity; raw
  pointer identity was rejected because allocator address reuse creates an ABA
  failure mode.
- Validate every input/output byte region before writes and conservatively
  reject all six possible overlap pairs.
- Write compact `U` directly, stage Faer's `V`, and conjugate-transpose it into
  caller-owned `Vt`. Signed-stride inputs use a workspace pack buffer when they
  cannot be borrowed directly.
- Reuse the existing owned-path gauge implementation in place so prepared and
  owned semantics cannot drift.
- Represent unsupported provider/dtype/layout/binding as structured capability
  errors. String-only backend failures were rejected because callers cannot
  reliably branch on diagnostic text.

## Verification Method

Focused correctness tests compare prepared and owned output semantics and
check reconstruction plus both `U` and `Vt` unitarity. Contract tests cover
empty dimensions, signed strides, compact output subviews, unchanged sentinel
outputs on validation failure, all alias pairs, backend/plan/workspace mismatch,
and independent workspaces.

The allocator test counts the complete release-mode warm call, including
validation, packing, provider execution, `Vt` conversion, and gauge. Retaining
`CpuSet` storage and active-request capacity removed the two one-thread
coordination allocations: contiguous/strided F64 and contiguous C64 now measure
zero. A two-thread probe still measured six allocations, exactly matching an
empty `CpuBackend::install`, while `CpuContext::install` measured zero. That
multi-thread execution-owner broadcast cost remains under design audit; the
public zero-allocation contract is not weakened to one thread by this record.

The Criterion benchmark uses persistent one- and two-thread Faer backends per
path. Preparation, workspace allocation, and destination allocation are outside
the prepared timer. Owned `svd_read` includes result allocation and destruction;
prepared timing includes destination overwrite. Shapes cover square, tall, and
wide matrices from 2-wide through 64-square. A separate one-thread 8x4 case
compares the same positive-stride borrowed view on both paths.

On the local arm64 macOS host, a release `--quick` run measured these ratios
using the reported point estimates:

| Shape / input | Threads | Owned | Prepared | Owned / prepared |
| --- | ---: | ---: | ---: | ---: |
| 2x2 compact | 1 | 1.162 us | 0.530 us | 2.19x |
| 4x2 compact | 1 | 1.278 us | 0.648 us | 1.97x |
| 2x4 compact | 1 | 1.306 us | 0.648 us | 2.02x |
| 4x4 compact | 1 | 2.325 us | 1.672 us | 1.39x |
| 8x4 positive stride | 1 | 2.523 us | 1.845 us | 1.37x |
| 16x16 compact | 1 | 17.385 us | 16.693 us | 1.04x |
| 2x2 compact | 2 | 20.013 us | 17.719 us | 1.13x |
| 16x16 compact | 2 | 39.648 us | 38.481 us | 1.03x |
| 64x64 compact | 2 | 395.64 us | 394.50 us | 1.00x |

Two-thread absolute times expose the current per-call execution-owner broadcast
cost. The prepared path still avoids owned output work, but the independent CPU
execution prerequisite is required before claiming multi-thread warm allocation
or small-matrix latency is complete.
