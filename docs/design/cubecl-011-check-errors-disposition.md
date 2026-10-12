# Disposition: the fork's `Client::check_errors` against upstream 0.11

Status: awaiting a decision. This is the last compile error of the tenferro-gpu
0.11 migration (564 -> 1).

## What the fork had

`tensor4all/cubecl` adds `Client::check_errors(&self) -> Result<(), ServerError>`
(fork PR #27, the soundness set #27/#28/#29). It is client-wide:

```rust
pub fn check_errors(&self) -> Result<(), ServerError> {
    let stream_id = self.stream_id();
    self.device.submit_blocking(move |server| server.check_errors(stream_id))…
}
```

and the server side is `flush_errors(stream_id)` collected into
`ServerError::ServerUnhealthy`. tenferro calls it at four publication boundaries
in `cubecl/transfer.rs` (279, 788, 1115, 1638) and maps any error to
`Rejected`/`publication_rejected` with the value and buffer returned unchanged.

## What upstream 0.11 actually has

The capability is present, in a **buffer-scoped** and explicitly-specified form —
this is not an API that 0.11 lacks:

- `Client::check<'a>(&self, handles) -> Result<(), ServerError>`
  (`cubecl-runtime/src/client.rs:1285`) — "Instant means enqueue-time failures
  only … `ServerError::Several` naming every failure these buffers carry, each
  once however many carry it. The bytes are gone, so there is nothing to retry:
  this is the answer, not a hint."
- `Client::sync_buffers<'a>(&self, handles)` (`client.rs:1414`) — the barrier plus
  that same claim check. `Client::sync()` is `sync_buffers([])`.
- Server side: `CudaServer::check` is `self.streams.ensure_written(handles)`
  (`cubecl-cuda/src/compute/server.rs:222`), i.e. exactly "the claim check a read
  would have made".
- Upstream's own docs state the model: "A launch failure is not this sync's to
  report: it lives on the buffers the launch never wrote and surfaces on any
  read, `check` or `sync_buffers` of those" (`client.rs:1394-1398`).

The ports add nothing here, and the interop port relies on the same mechanism:
`ServerStorage::get_resource` performs `streams.ensure_written([&binding])` with
the comment "The same claim check a read makes: a buffer a failed launch never
filled reports the failure rather than handing back a pointer to whatever was
there before" (`cubecl-runtime/src/server/base.rs:779-788`).

## Decision: delete the checks, keep the resource boundary

Independently reviewed before implementing (openai/gpt-6.1-sol, read-only). The
review verified, site by site, that the handle being published is in scope at
each check, that both the check and the following `get_resource` failure map to
the same `rejected`/`publication_rejected` outcome with unchanged owners, that no
site depends on an unrelated buffer's failure, and that upstream's claim refuses
the pointer for a buffer a failed launch never filled. It recommended the
smallest variant over a handle-threaded wrapper, and this was implemented:

- the four `runtime.check_errors(...)` calls (`transfer.rs` 279, 788, 1115, 1638)
  and both wrappers (`runtime.rs` 716, 1008) are deleted;
- the published buffer's `get_resource` claim, its error mapping and the
  unchanged-owner rejection stay exactly as they were;
- `publish_download`'s comment names the claim that rejects instead of the
  deleted call.

Option B (client-wide `check_errors` upstream) is not in scope: no caller needs a
stream-wide diagnostic, and upstream has deliberately exposed the capability per
buffer instead.

### Contract differences from the fork (recorded, not papered over)

- The fork's check was current-stream-scoped and drained the stream's error vector
  (`mem::take`), so unrelated work on the same stream could reject a publication
  and the error cleared on report. Upstream's claim is scoped to the published
  buffer and is persistent until the bytes are rewritten
  (`cubecl-server/src/stream/failures.rs:80-108`). Narrower and not
  self-clearing; no inspected caller relied on the old behaviour.
- Upstream's `check`/`sync_buffers`/`get_resource` use `unwrap_or_resume()`: a
disconnected device runner resumes a panic instead of returning the fork's typed
`ServerError::Generic`. If a fallible dispatch-error propagation is wanted, that
is a new upstream capability and would need its own proposal; it is not a reason
for a tenferro shim.
- Not literal equivalence, therefore: "the producer's failure still rejects
  publication with the owners returned unchanged" is the claim, not "identical
  error values".

## Evidence gap / next test

No execution was performed for this change. The one test that would catch a
regression here does not exist: a deterministic rejected producer writing the
published allocation, asserting the originating failure and the unchanged
rejection package before DMA/frontier admission, per admission path
(`transfer.rs` 270-348, 763-861, 1090-1187, 1626-1705). The existing tests cover
validation rejection only (`tests/pending_download.rs:200`,
`tests/pending_upload.rs:161`, `tests/source_publication.rs:224`), and
`docs/design/owning-pending-transfers-1945-u4.md:116,122-124` already records
that fault-injection coverage is absent. A compile-validation rejection should
be enough; no fault-injection framework is needed.
