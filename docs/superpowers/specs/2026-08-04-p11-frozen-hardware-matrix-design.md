# P11 Frozen CPU/CUDA/WebGPU/Apple-Metal Hardware Matrix Design

Date: 2026-08-04

Status: design complete; execution waits for a P13-A candidate

Authority: #1555, #1568, `docs/design/storage-ownership-contracts.md`
(G1/G3/G5/G7), and `scripts/storage-ownership-contracts.toml`

## Scope

P11 executes the final adapter-free product tree on CPU and real CUDA, WebGPU,
and Apple/Metal hardware. It creates evidence only; it does not fix product
code, change APIs, relax tests, or add provider recovery machinery. Every lane
names the same P13-A candidate commit.

If a lane exposes a product defect, work returns to the owning phase, a new
candidate is frozen, and affected P11/P12/P13 evidence is rerun. Evidence-only
commits may add the matrix report and promote ledger state, but may not modify
the candidate product tree.

## Candidate identity

The candidate is the exact 40-hex commit recorded by
`docs/design/storage-contract-freeze.md`. It already contains:

- final code and public documentation;
- all provider and semantic tests;
- all P10/P11/P12/P13 checker scripts;
- no migration scaffolding.

P11 jobs check out that commit directly. A tracked report path plus candidate
commit and ordinary CI/job URL or repository-relative log path identify the
evidence. No content digest, nonce, challenge, or attestation is added.

## Required lanes

### CPU reference

CPU executes the complete semantic reference matrix for f32/f64/i32/i64/bool,
C32/C64 where supported, and empty, scalar, compact, positive-stride,
transpose, and reverse-stride layouts. It covers:

- owner/view/view-mut and allocation-free static-rank reborrows;
- prepared contiguous and strided host access;
- N-way disjoint mutable split and structural extraction;
- sealed reinterpretation pairs;
- detached submission and synchronous scoped read-only execution;
- eager/traced AD and checkpointing with reason-classified counters.

### CUDA multi-device

CUDA requires at least two visible devices. The lane creates distinct
caller-selected engines, allocation domains, event domains, queues, caches,
and workspaces for each device. It proves:

- concurrent independent graphs on both devices;
- correct same-device input/output and foreign-device/domain rejection;
- no fallback or host staging on wrong endpoint;
- prepared-once counts independent of tensor size;
- immediate owner/handle drop after enqueue and both teardown orders;
- completed, retired-failed, detached-handle, provider-panic, and
  completion-unproven behavior;
- explicit duplicate/upload/download accounting;
- CUDA eager/traced forward/backward and checkpointing as the designated real
  asynchronous AD lane, with zero retention-attributable copies/allocations.

### WebGPU

A real non-Metal WebGPU adapter, where the supported CI hardware provides one,
executes:

- device-local map rejection;
- explicit upload/download and transfer counters;
- device write followed by host download/readback;
- event retirement and immediate-drop behavior;
- zero-copy reinterpretation and prepared-once counts;
- no hidden CPU fallback or host round trip.

### Apple/Metal

A real Apple host with Metal and host-visible primary memory executes:

- one allocation owner/domain/key across CPU and Metal endpoints;
- CPU→Metal→CPU ordering and read-after-device-write;
- host/GPU access exclusion;
- zero transfer bytes and unchanged allocation identity for endpoint switches
  and reinterpretation;
- explicit upload/download counters;
- managed-state preservation;
- real FFT and linalg paths;
- retirement and immediate-drop behavior.

WebGPU and Metal are distinct report lanes even when they share provider code.

## Cross-lane scenarios

Every relevant provider records:

- pre-admission rejection returning unchanged owners;
- identity, repeated, metadata-only, and duplicate outputs retaining one owner;
- completed and retired-failed owner return only after retirement;
- handle drop as detach;
- completion-unproven diagnostics with no owner and retained private record;
- wrong endpoint/domain typed errors without fallback;
- explicit duplicate/transfer destination identities;
- resolution counts independent of element count;
- numerical values or residuals, not shape-only success.

P11 also verifies P10 performance and codegen report references. It does not
rerun an incompatible benchmark and relabel it comparable.

## Required-mode behavior

Hardware test binaries accept ordinary local mode and required mode. Ordinary
mode may produce a structured skip containing provider, detected environment,
and reason. Required mode converts absence, skip, insufficient CUDA device
count, unsupported feature, or unexecuted case into failure.

The matrix runner uses one variable:

```text
TENFERRO_STORAGE_REQUIRED_LANES=cpu,cuda2,webgpu,metal,cuda-ad
```

The value is a comma-separated set from the fixed lane vocabulary above.
Unknown, duplicate, or empty required entries are errors. P11 final evidence
uses the full value shown. Provider-specific CI jobs may set their subset, but
the assembled report must contain passing evidence for every required lane.

A lane may report `unsupported` only during ordinary development. The final
P11 report accepts only `pass`; `skip`, `unsupported`, `inconclusive`, and
`not-run` block activation.

## Report contract

Artifact and command:

```text
docs/testing/storage-hardware-matrix.md
python3 scripts/check-storage-hardware-matrix.py \
  --report docs/testing/storage-hardware-matrix.md
```

The Markdown report contains exactly one fenced JSON record with schema
`tenferro.storage-hardware-matrix.v1`. Its required top-level fields are
`schema`, `candidate_commit`, `freeze_report`, `required_lanes`, and `lanes`.
`candidate_commit` is exactly 40 lowercase hexadecimal characters;
`freeze_report` is exactly `docs/design/storage-contract-freeze.md`; and
`required_lanes` is exactly `cpu`, `cuda2`, `webgpu`, `metal`, and `cuda-ad` in
that order.

Each lane object records `id`, `status`, exact command argv, OS/architecture,
compiler/toolchain, provider/runtime versions, detected devices, feature flags,
executed test count, numerical/counter observations, and an ordinary CI URL or
tracked repository-relative evidence path. CUDA records both devices; Metal
records Apple hardware/OS; WebGPU records adapter/backend. Every field contains
a measured concrete value.

The checker:

1. parses the freeze report and requires one candidate commit;
2. requires every lane to name that commit and every required status to be
   `pass`;
3. validates exact command arrays and required environment facts;
4. requires the scenario/counter fields owned by each lane;
5. verifies P10 report paths and candidate compatibility: no source file named
   by either P10 hot-path report changed between its measured commit and the
   P13-A candidate;
6. rejects product-tree changes between the candidate and evidence HEAD except
   the P13 evidence allowlist;
7. verifies the ledger row remains nonterminal until P13-B.

The checker trusts ordinary tracked evidence and hosted CI results. It does not
attempt to defend against a malicious runner or query remote hardware during a
local verification.

## Failure ownership

- semantic/ownership failure returns to P3/P9 or the common storage phase;
- CUDA failure returns to P7;
- WebGPU/Metal failure returns to P8;
- API/hot-path/performance failure returns to P10;
- docs-driven misuse discovered while running examples returns to P12;
- checker/report defect returns to its design owner before a new candidate.

Any fix creates a new candidate. Only lanes affected by a purely provider-local
fix need rerun, but P13-B independently decides whether cross-lane evidence was
also invalidated. Candidate identity is never patched in place.

## Proportional-safety boundary

P11 tests supported runtime behavior: construction rejection, Rust borrowing,
prepared access, numerical correctness, event retirement, and explicit
transfer accounting. It does not test malicious providers, global quarantine,
mutex poison recovery, destructor panic recovery, cryptographic receipts, or
repeated/per-element validation. Source scans remain supplemental to runtime,
compile-fail, numerical, and hardware observations.

## Exit

P11 is complete when all required lanes pass on the same candidate, the report
passes its checker, performance/codegen references are conclusive, and no
product file changed. The `p11-hardware` ledger row may then activate. P13-B
remains responsible for independent closure.
