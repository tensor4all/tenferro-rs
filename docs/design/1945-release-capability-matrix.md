# #1945 release and adoption capability matrix

What #1945 has actually placed in the hands of a downstream consumer of this repository, at the pinned
revisions recorded in §1. This is the U5 record: capability-specific release statements, the guidance
that names each capability, and the adoption gates that follow from them.

## 1. Pinned revisions this matrix describes

| Repository | Revision |
| --- | --- |
| `tensor4all/tenferro-rs` | `main` at `7f3094ae8` (#2093); later upstream commits are tests, docs and examples only |
| `tensor4all/cubecl` (fork) | `main` at `1811309791f675125379b326bf748f2ec1e29f2a` (#27, #28, #29) |
| `tensor4all/cubek` (fork) | `release/t4a-0.2.2` at `507c91f5a70e284595755b72598f60c684d1132b` |
| `tensor4all/tensor4all-rs` | `main` at `11f29ff0` (tenferro pin `ff94aeded…`), with the B1 frontend surface |

"Released" below means merged with its gates green, not "planned".

## 2. Capability matrix

| Capability | State | Evidence | Downstream milestone |
| --- | --- | --- | --- |
| **Held CPU session (U1)** — caller-owned concrete session, ownership/entry proof, no hidden session entry | released | #2053, held-session tests | tensor4all-rs CPU backend/coexistence milestone (#866, #873) |
| **Concrete non-AD fast path (U2-concrete)** — public session entry, per-op submission, phase lease, paired measurements | released | #2054, #2055, #2056, guide #2057 (`docs/guides/session-entry-cost.md`) | same milestone; the explicit frontend's dispatch route |
| **Held CUDA session (U3)** — held reservation, typed reentry/contention, logical `StreamId` capture, no per-op flush, `Drop` does not flush | released | #2059, 7 A100 tests, 3 compile-fail fixtures, `docs/design/held-cuda-session-1945-u3.md`, guide #2071 | tensor4all-rs CUDA/async milestone |
| **Owning pending transfers (U4 items 2–3)** — `download_pending` / `upload_pending`, admission that returns the package on refusal and retains on unproven enqueue, bounded retirement queue, no wait in `Drop` | released | #2061, #2065, #2073, #2076, `docs/design/{asynchronous-transfer,owning-pending-transfers}-1945-u4.md`, transfer bench #2066 | CUDA/async milestone; the only route that moves a value without host export/import |
| **Source publication (U4 item 1, source half)** — a produced value published with its producer frontier; the session, backend handle and source tensor may go away; the publication owns the value, its allocation address and readiness | released | #2101, `crates/tenferro-gpu/tests/source_publication.rs` (7 A100 tests), 2 compile-fail fixtures, `docs/design/source-publication-1945-u4.md` (revision 5) | CUDA/async milestone |
| **Cubecl-side boundary changes** — `check_errors` publication boundary (#27), bounded staging retention on the write path (#28), write authority required for server writes (#29) | released in the fork | fork PRs, `docs/design/cubecl-fork-exit-1945.md` | prerequisite for the routes above; not yet proposed upstream |
| **Optional held AD adapter (U2-AD)** | **deferred, not solved** | upstream design record D12; our measurement puts session entry at ≈5% of a 13.5 µs op, so the measured-signal precondition is unmet | must not be advertised; B3 depends on it |
| **Consumer half of the transfer contract** — registry-bound consumer copy of a published value, dependency admission in place of the scheduler's host bridge | **specified (revision 2), not implemented** (`docs/design/consumer-read-handoff-1945-u4-2.md`): one owner-scoped registry binding, a registry-provenanced completion of the recorded frontier, admission on the destination run, and an owned consuming copy on the consumer's stream | design only; revision 1 was rejected by its pre-implementation review and revision 2 is pending re-review | must not be advertised until implemented and measured |
| **Direct-required / allow-staging route policy, P2P capability registration** | not implemented | route inventory only | no frontend redesign required to add them |
| **Producer/consumer overlap** | measured | #2106, the `transfer_overlap` group in `crates/tenferro-gpu/benches/transfer_paths.rs` (controlled schedule) | measured claim only |
| **Cross-stream readiness, host-wait-free publication, fault-injected terminal paths** | not established | every transfer/publication test uses the ambient captured stream; the host bridge is still the only cross-domain route | claims must stay bounded until measured |
| **Differentiated (AD) transfer** | not implemented | #1945 transfer contract item 6 | separate work; must not be implied by `detach` |

## 3. Guidance that names the capability set

| Document | Names |
| --- | --- |
| `docs/guides/session-entry-cost.md` | scoped vs held CPU entry, what holding a session across a stage costs, when to hold |
| `docs/guides/cuda-sessions-and-transfers.md` | held CUDA sessions, the pending transfer routes, what is not claimed (no fake async, no hidden waits) |
| `docs/design/held-cuda-session-1945-u3.md` | what the held CUDA session does and does not submit |
| `docs/design/{asynchronous-transfer,owning-pending-transfers}-1945-u4.md` | admission, refusal, retention and teardown contracts for transfers |
| `docs/design/source-publication-1945-u4.md` | the publication contract and every claim it declines to make |
| `docs/design/cubecl-fork-exit-1945.md` | which substrate changes the fork carries, which stay upstream, and the own-backend contingency |

## 4. Adoption gates

1. **CPU milestone.** A consumer that only needs held concrete CPU execution can adopt at the revisions
   in §1 without waiting for CUDA, transfer or AD work: the compatibility (eager) frontend and the
   explicit frontend build against the same revision, with no signature, storage or ABI rewrite and no
   second dependency version.
2. **CUDA/async milestone.** Adoption includes the held CUDA session and the owning pending transfers.
   Source publication joins it when #2101 merges.
3. **Per pin update.** Both downstream frontends (compatibility/eager and explicit) are checked against
   the new revision, together with the CUDA test partition and the held-session / pending-transfer /
   publication tests.
4. **Not a gate.** Universal application migration and removal of the global defaults are not required
   for any gate above, and no gate claims that held CPU sessions alone complete the CPU/GPU architecture.

## 5. What this matrix refuses to claim

- No capability marked deferred, unmeasured or not implemented may be described as released, and none is
  reachable through the public API as if it were.
- The pending routes add a route; they do not reroute the blocking transfer APIs, which stay blocking and
  keep their documented readiness.
- The fork's changes are released *in the fork*. Until they are proposed and accepted upstream, they are
  not part of cubecl's own story, and #1945's residue there is described in
  `docs/design/cubecl-fork-exit-1945.md` rather than claimed as fixed.
