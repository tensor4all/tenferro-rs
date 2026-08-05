# P13 Adapter-Free Freeze and Independent Closure Design

Date: 2026-08-04

Status: design complete; P13-A waits for P8/P9 reconciliation and P10

Authority: #1555, #1567, `docs/design/storage-ownership-contracts.md`
(G1-G7), and `scripts/storage-ownership-contracts.toml`

## Scope

P13 has two gates:

- **P13-A** removes every migration/legacy path, verifies the complete product
  tree, and freezes one exact candidate commit.
- **P13-B** runs an independent closure audit after P11 hardware and P12
  documentation evidence name that same candidate.

P13 adds evidence and ledger promotions only after the candidate. It does not
repair product behavior inside an evidence commit. Any product/API/docs/checker
fix creates a new candidate and invalidates affected evidence.

## Candidate and evidence-commit model

A tracked report cannot contain the hash of the same commit that contains the
report. The design therefore distinguishes the product candidate from its
trusted evidence-only descendants:

1. Commit **C** contains final code, final public documentation, all tests and
   checker scripts, and no migration scaffolding.
2. A P13-A evidence commit adds
   `docs/design/storage-contract-freeze.md`, naming C, and promotes only the
   `p13-freeze` state.
3. P11 and P12 run checkouts of C. Evidence-only commits add their reports and
   promote their rows; every report names C.
4. P13-B audits C and the evidence set. Its evidence-only commit adds
   `docs/worklogs/storage-redesign-closure.md` and promotes `p13-closure`.

The allowed post-C diff is closed and path-specific:

- `docs/design/storage-contract-freeze.md`;
- `docs/testing/storage-hardware-matrix.md`;
- `docs/testing/storage-documentation-audit.md`;
- `docs/worklogs/storage-redesign-closure.md`;
- tagged `state` promotions only in
  `scripts/storage-ownership-contracts.toml`;
- the active-ID fixture in `scripts/test-storage-ownership-contracts-v2.py`,
  limited to adding exactly the IDs promoted by the same evidence commit.

Product docs, Rust/Python implementation, benchmark/codegen reports, checker
logic, and all other tests must already be in C and are not on this allowlist.
The active-ID fixture exception may change only its expected set literal; it
cannot change executable validation logic. An allowed path may contain evidence
only and cannot redefine a contract or API.

Exact Git commit and tracked repository-relative paths are sufficient. No
manifest digest, file checksum, nonce, challenge, or attestation is introduced.

## P13-A product removal inventory

The candidate physically lacks:

- public/legacy `Buffer<T>`, `BackendBuffer<T>`, typed backend owners, and
  `StorageBuffer::Backend` ownership bridges;
- `TensorOwnedView`, shallow-clone owners, COW, mutable owner projections, and
  pair-only split implementations;
- parallel `AllocationGroup::tensor_owners` or provider-buffer fields;
- `Arc<Tensor>` runtime/AD ownership, materialized owner caches, and old
  borrow-and-clone submission;
- `ExecutionResult { inputs, outputs }`, `Completed(Vec<Tensor>)`, extracted
  flags, and temporary runtime/AD adapters;
- provider-specific migration bridges, optional allocation domains, fixed
  engine IDs, transfer defaults, flat/deprecated provider exports, and safe
  unleased raw-handle escapes;
- direct CUDA/WebGPU launch paths that bypass prepared device payloads;
- duplicate host/device preparation or dispatch/storage paths;
- global generation/tombstone/liveness/retirement registries, COW,
  quarantine/poison/retry/cancellation protocols, repeated validation, and
  attestation machinery;
- the prior handoff artifact and every inbound reference.

The candidate has exactly one owner/view/view-mut family, one
root/span/allocation-group model, one prepared-access hierarchy, and one
detached plus eligible synchronous-scoped runtime model.

## P13-A prerequisite evidence

Before C is selected, all P0-P10 obligations are active and passing. The
candidate also contains the complete P12 product docs and P11/P12/P13 checkers,
even though their evidence rows remain deferred. Required non-hardware checks
include:

- formatting, workspace check/test/clippy, extensions, doctests, and docs;
- all storage ledger/unit/compile-fail/provider tests runnable without required
  hardware;
- public API/error/category/docs-site checks;
- P10 structural, performance, and codegen evidence, all conclusive;
- token/compile/runtime ownership inventory;
- clean tracked worktree and exact base/candidate provenance.

Hardware-specific tests may produce structured local skips at this gate; P11
required mode owns actual execution.

## Freeze report

Artifact and command:

```text
docs/design/storage-contract-freeze.md
python3 scripts/check-storage-contract-freeze.py \
  --report docs/design/storage-contract-freeze.md
```

The report contains exactly one fenced JSON record with schema
`tenferro.storage-contract-freeze.v1` and concrete fields:

- candidate commit C and base commit;
- ledger schema/revision and active obligations through P10;
- final public API/provider namespace summary;
- one-owner/prepared/submission model paths;
- complete removed-scaffolding inventory with proof kind;
- non-hardware command executions and tracked evidence paths;
- P10 benchmark/codegen measured commits and compatibility status;
- required future P11/P12/P13 report paths;
- `terminal: false`.

`check-storage-contract-freeze.py` verifies:

1. candidate/base commits exist and candidate is an ancestor of evidence HEAD;
2. all product files/checker scripts required by P11/P12/P13 exist in C;
3. the diff from C to HEAD is inside the evidence allowlist;
4. worktree/index are clean;
5. ledger graph/state is valid and all through-P10 obligations are active;
6. `storage_public_api`, storage compile contracts, and provider/runtime tests
   prove the legacy ownership paths are absent;
7. `storage_public_api.rs` uses the workspace `syn` dependency to parse the
   fixed owner/group/runtime/provider source manifest and reject named
   owner/clone/raw/adapter declarations and call paths; the Python checker
   invokes that test, while lexical scans remain supplemental;
8. obsolete handoff files/references are absent;
9. every recorded command/evidence path is concrete and tracked;
10. no performance result is inconclusive.

The checker does not hash tracked files or defend against a malicious checkout.

## Evidence invalidation

A change after C is classified as:

- **evidence-only allowlisted change** — C remains valid;
- **product, public docs, test, checker, benchmark, or contract change** — C is
  invalid and P13-A restarts;
- **hardware-environment-only rerun** — C remains valid if no repository change
  occurs.

When a new candidate is required, the freeze report, affected hardware/docs
reports, and closure report are regenerated. Old reports remain in Git history,
not as current accepted artifacts.

## P13-B independent audit

P13-B is performed by reviewers who did not author the implementation slice
being judged. It audits seven lanes:

1. architecture and dependency direction;
2. Rust ownership, aliasing, and unsafe/provider boundaries;
3. asynchronous event/resource lifecycle;
4. public API, explicit copies/transfers, and provider parity;
5. storage hot-path structure, performance, and static-rank codegen;
6. CPU/CUDA/WebGPU/Metal and multi-device integration;
7. documentation usability and AD/checkpoint retention.

Each finding has severity `Critical`, `Important`, or `Suggestion`, a concrete
contract clause, file/evidence references, and disposition. Any unresolved
Critical or Important finding blocks closure. A performance result marked
inconclusive or a required hardware lane not executed is an Important blocker.

The audit confirms:

- one move-only owner per physical span and Rust-only write authority;
- aliases represented by group descriptors;
- construction-time validation retained into one prepared path;
- no repeated/per-element provider/storage validation;
- proven retirement before owner return and ownerless permanent retention when
  completion is unproven;
- explicit duplicate/upload/download and zero-byte Apple endpoint switching;
- final provider namespaces and no safe raw escape;
- zero AD-retention copy/allocation events;
- P11 and P12 name candidate C and contain no required skip;
- source-blind docs allow a downstream user to build the required example.

## Closure report and checker

Artifact and command:

```text
docs/worklogs/storage-redesign-closure.md
python3 scripts/check-storage-redesign-closure.py \
  --report docs/worklogs/storage-redesign-closure.md
```

The report contains exactly one fenced JSON record with schema
`tenferro.storage-redesign-closure.v1` and concrete fields:

- candidate C and freeze/P11/P12 report paths;
- audit reviewer identities/independence statements;
- seven lane outcomes and evidence references;
- all findings and dispositions;
- rerun/invalidation decisions;
- final active obligation list;
- `terminal: true` only when no blocker remains.

The checker:

1. validates C against the freeze report;
2. requires P11 and P12 reports to name C and pass their own checkers;
3. enforces the evidence-only diff allowlist;
4. requires every ledger obligation active and passing at evidence HEAD;
5. rejects unresolved Critical/Important findings, inconclusive performance,
   and required hardware skips;
6. requires the obsolete handoff and all migration paths absent in C;
7. confirms closure evidence paths are tracked and worktree/index clean;
8. emits nonterminal status on every incomplete condition.

A normal independent review is sufficient. The checker does not add generic
unsafe-count targets, forged-review detection, cryptographic provenance, or
malicious-maintainer defenses.

## State summary

| Stage | Product changes allowed | Evidence state | Terminal |
|---|---:|---|---:|
| before C | yes, through owning phase | rows through P10 active | no |
| P13-A evidence | no | freeze report, `p13-freeze` active | no |
| P11/P12 evidence | no | hardware/docs rows active on C | no |
| P13-B audit | no | closure report and all rows active | yes if blocker-free |
| finding requires fix | new candidate required | affected evidence invalid | no |

## Proportional-safety boundary and exit

P13 proves supported scientific-computing contracts using compile/runtime,
numerical, benchmark, hardware, documentation, and independent-review
evidence. It does not add quarantine/corruption recovery, destructor panic
recovery, repeated identity validation, checksums for tracked files,
nonce/attestation, or malicious-runner assumptions.

#1555 may close only after P13-B passes, all obligations are active and passing,
all reports name the same candidate C, the evidence HEAD differs from C only by
the closed allowlist, and no Critical/Important finding remains.
