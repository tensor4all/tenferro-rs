# NUMA-aware CPU execution

Issue #1345 adds process-affinity-aware NUMA topology, pinned fixed CPU engines,
shared arbitration, and a documented provider boundary.

The implementation treats OS node IDs as sparse and intersects node cpusets
with the process mask. `CpuBackend` clones share engines and caches. faer owns
managed placement; external BLAS accepts only `Auto` and runs exclusively
because its worker affinity is not controlled by tenferro. Graph execution
keeps a permit across supported Host/native/FFI instructions, entering Rayon
for native BLAS-backend segments and leaving it for provider calls.

Rejected alternatives were thread-count-only placement, exposing concrete BLAS
provider names as stable identities, and allowing explicit BLAS node placement
without a verifiable worker-affinity contract.

Verification covers sparse topology parsing, affinity pinning, overlapping and
disjoint permits, clone sharing, graph session count, Host work inside the
session, and the BLAS Rayon/provider transition. The online guide is
`docs/guides/cpu-execution.md`.
