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
- Report provider-resource bytes without conflating the plan and workspace:
  Faer's plan retains no heap buffer, workspace required bytes are logical
  prepared sizes, and workspace retained bytes include actual vector capacity.
  Shared context storage and inline object size are deliberately excluded.
- Add an operation-neutral `PreparedFactorizationSession` rather than an
  SVD-specific session. Standalone execution is a one-leaf adapter, while
  repeated leaves reuse one CPU permit, execution owner, and pool entry.
- Keep the leaf path statically dispatched. Reusing tensor `BackendSession`
  would introduce an unrelated trait object and buffer-pool loan; the prepared
  session instead uses `CpuBackend::install` once and a private operation enum.

## Session Amendment

The original complete-call allocation contract exposed a multi-worker CPU
coordination cost before it exposed a provider allocation. The amendment keeps
the backend arbitration and reentry semantics intact while moving that cost to
one explicit, operation-neutral session entry. A prepared leaf validates the
session/backend, plan, workspace, input, destinations, and alias regions before
writes, then invokes Faer without another install. Ordinary errors remain
inside the session and panic unwinding releases the outer execution guard.

The session callback uses a higher-ranked lifetime, so the opaque non-`Clone`
handle cannot escape. One workspace remains exclusive to one lane; independent
sessions and workspaces are the concurrency mechanism. Provider-specific
threading and bindings remain private and can be replaced without changing the
public lifecycle.

Input and destination descriptors are caller-owned setup. Dynamic-rank
`TensorRead` and `TensorWrite` view construction retains shape and stride
metadata and may allocate before the leaf is entered. The allocation gate
therefore preconstructs those descriptors rather than adding a second rank-2
descriptor API that would duplicate the tensor view contract.

Directly counting one CPU entry and zero leaf reentries would require exposing
or test-instrumenting private backend coordination. This change does not add a
public counter solely for tests. CPU reentry tests and the leaf allocation gate
provide indirect evidence; integration callers should retain a structural
executor-entry counter when adopting sessions so this invariant is gated at the
call site as well.

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
probe motivated amortizing backend entry through the session. It did not prove
that provider scheduler activity inside later leaves would remain globally
allocation-free.

The session allocator amendment measures at least 128 F64 and C64 leaves for
contiguous and non-compact strided inputs. Session construction and warm-up are
excluded. The small one-worker cases retain strict global-zero gates as measured
evidence, not as an all-shape provider guarantee. Two- and four-worker runs
retain repeated-leaf correctness and benchmark coverage, but deliberately have
no allocation upper-bound assertion.

That distinction follows a release-mode repetition study: 23 of 30 multiworker
runs failed the proposed global-zero assertion even after leaf warm-up. Splitting
the four input cases showed the allocation bursts moving between measured
windows; a representative two-worker failure reported `[25, 0, 0, 0]` for F64
contiguous, F64 strided, C64 contiguous, and C64 strided. Re-running the same
binary also alternated between zero and nonzero counts. A sequential-provider
diagnostic made the attribution narrower: these moving bursts were dominated by
managed Rayon activity from the outer session entry completing after the
callback allocation counter became active. Separate Faer and Spindle internal
allocation sources exist, but this experiment did not prove that they produced
these particular moving counts.

An independent 64x64 F64 source/root diagnostic primed and settled each path,
then measured 47 provider allocations per call for one-worker sequential Faer.
Its original two- and four-worker figures were discarded: `Par::rayon(0)` had
captured the ambient global degree when the plan was prepared outside the
managed pool, so those figures did not prove either configured worker count.
After the CPU-context fix passed the configured degree explicitly, five release
runs each confirmed actual degrees 1, 2, and 4. The allocation counts per call
were 47 for one worker, 72 for two workers, and a median 112 for four workers;
the four-worker totals over eight leaves were 896, 903, 896, 896, and 897. The
sequential source includes the `did_pack_lhs` vectors in `gemm-common` 0.19.0
`src/gemm.rs` (the sequential allocation at line 490 and per-worker storage at
line 654). Faer 0.24.4 reaches Spindle 0.2.6 `for_each` through
`src/utils/mod.rs::thread::join_raw`, including calls from
`src/linalg/svd/bidiag.rs`; Spindle's `for_each_imp` reserves iterator-split
storage at `src/lib.rs` line 660 and otherwise delegates work to Rayon when no
Spindle root is installed. This evidence does not attribute Faer's calls to
Spindle `with_lock` or its root-manager allocation.
Consequently the accepted boundary is that
tenferro-controlled output, pack, and prepared-workspace storage does not grow
after warm-up. Provider-internal numerical and scheduler allocations remain
allowed and are recorded by benchmark. No public capability enum is added until
another provider demonstrates the abstraction it must express.

External merge remains blocked until the accepted prepared-factorization issues
approve this operation-neutral session amendment. The local implementation and
verification do not change that issue-acceptance gate.

The Criterion benchmark uses persistent one- and two-thread Faer backends per
path. Preparation, workspace allocation, and destination allocation are outside
the prepared timer. Owned `svd_read` includes result allocation and destruction;
prepared timing includes destination overwrite. Shapes cover square, tall, and
wide matrices from 2-wide through 64-square. A separate one-thread 8x4 case
compares the same positive-stride borrowed view on both paths.

On the local arm64 macOS host, a release `--quick` run at this branch's HEAD
measured these two-worker ratios using the reported point estimates:

| Shape / input | Threads | Owned | Prepared | Owned / prepared |
| --- | ---: | ---: | ---: | ---: |
| 2x2 compact | 2 | 6.339 us | 5.737 us | 1.10x |
| 4x4 compact | 2 | 7.442 us | 6.880 us | 1.08x |
| 16x16 compact | 2 | 25.944 us | 25.533 us | 1.02x |
| 64x64 compact | 2 | 435.47 us | 429.45 us | 1.01x |

The CPU execution-scope prerequisite removes the former roughly 18 us small
matrix entry penalty. The standalone prepared adapter still enters one session
per call; callers with multiple leaves use the explicit session API to amortize
that remaining entry boundary.
