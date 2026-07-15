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

The Criterion benchmark uses one persistent one-thread Faer backend per path.
Preparation, workspace allocation, and destination allocation are outside the
prepared timer. Owned `svd_read` includes result allocation and destruction;
prepared timing includes destination overwrite. Shapes cover square, tall, and
wide matrices from 4-wide through 64-square.
