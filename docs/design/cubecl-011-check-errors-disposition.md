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

## The two options

**A. Adopt the upstream shape (tenferro-side, public API only).** Replace the
client-wide `check_errors(op)` with a handle-scoped `client.check([&handle])`,
or rely on the `get_resource(handle)` claim that every one of the four call
sites already performs one statement later. No shim, no upstream addition, and
fork PR #27 is retired: upstream has the capability, better scoped and better
specified. Cost: one wrapper change plus four call-site changes on a
soundness-sensitive path (short design + independent review).

**B. File #27 upstream as it stands.** Keep `check_errors` client-wide and add
it to upstream. Cost: an upstream PR and its review, for a capability upstream
has already chosen to expose per buffer; the client-wide form also has no way to
say which buffers it answers for.

Recommendation: **A**. It follows the migration's rule (use public 0.11 APIs,
propose upstream only what 0.11 genuinely lacks) and the goal's direction
(retire the fork's soundness set rather than carry it). B should only be chosen
if a client-wide "is this device healthy at all" query is wanted as an
independent feature, which is a new upstream capability rather than a port.

## Evidence gap

No execution was performed for this note: the four call sites were read, the
upstream and fork APIs were read at the revisions above, and the outcome mapping
(`Rejected` with value+buffer returned) was compared textually. Whether
`client.check([&handle])` and the following `get_resource(handle)` report
*identical* errors on a failed launch, and whether option A's error is observed
at the same boundary in the tests, is to be established by running the transfer
tests with a fault injection once the crate compiles.
