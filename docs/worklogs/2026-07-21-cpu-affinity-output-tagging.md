# CPU affinity resolution and output tagging

Implemented Phase 2 Task 9 from the CPU-domain executor plan. The pure resolver
uses checked logical byte counts derived from tensor shape and dtype, treats a
scalar as one element and any zero extent as zero bytes, and reports shape or
byte multiplication overflow through typed errors.

Fresh CPU eager/session, dot, fusion, and linalg outputs now receive the
selected domain's `cpu_affinity` after successful execution. Tagging changes
only that placement field. It preserves device, memory kind, backend allocation
domain, and allocation identity. Metadata-only reshape/view results, input
storage, CPU no-op transfer clones, and caller-owned `_into` destinations are
not retagged. A borrowed non-compact view that must materialize is a fresh
allocation and is tagged.

Dot provider/domain capability preflight now runs before allocating an owned
result. Existing request validation remains before provider mutation. Linalg
outputs are tagged inside the single admitted linalg execution, including all
decomposition outputs and managed Cholesky allocations. The latter retain the
shared allocation-domain owner and managed memory kind.

Rejected alternatives were blanket-retagging every returned tensor, which
would corrupt storage-sharing metadata, and adding a dynamic map or string
dispatch to output tagging. The implementation uses closed enum matching and
in-place iteration over existing output vectors, adding no lookup or allocation
to the tagging path.
