# RunPod archive transfer

## Decisions

- On archive-cache misses, publish five balanced parts as separate Actions
  artifacts and use download-artifact v8's five concurrent downloads. Disable
  ZIP compression for these already-compressed archives. Keep the legacy single
  artifact for hosted cross-run reuse rather than changing the trusted artifact
  finder or invalidating existing caches.
- Keep transport steps inline in the trusted workflows: recovery of an older
  tested PR ref must work even when that ref lacks newly added CI helpers.
  Require all five parts and verify reconstructed original files with the hosted
  SHA-256 sums before tests. Retain bounded download retry and add bounded
  reconstruction plus elapsed-time reporting.
- The maintainer requested production adoption and merge after reviewing the
  experiment's inconclusive verdict. This explicit direction overrides the
  normal performance promotion gate for this transport change; it does not
  change that gate or classify the experiment as a success.

## Verification conclusions and constraints

- The [complete paired RunPod experiment](https://github.com/tensor4all/tenferro-benchmark/actions/runs/37781788409)
  observed a descriptive median of 42.57 s for a single artifact and 9.10 s
  for five parts. Its predeclared noise gates failed, so the result remains
  **INCONCLUSIVE** and does not establish a reproducible speedup. Full samples,
  host observations and the report are in that run's results artifact; the
  [experiment record](https://github.com/tensor4all/tenferro-benchmark/blob/3d353b6/docs/worklogs/2026-10-08-runpod-artifact-transfer.md)
  retains the protocol and limitations.
- Production uses a streamed tar container for reconstruction rather than the
  experiment's raw concatenation helper. The container is uncompressed, and the
  original archive bytes remain unchanged. Local executable regression tests
  cover unequal archive sizes, exact reconstruction, missing parts and corrupt
  content. Workflow contracts and actionlint cover the five uploads, parallel
  downloads, bounded retry and paid setup budget.
- Production transfer timing still needs observation after deployment. Keeping
  the single hosted-reuse artifact adds about one archive's worth of Actions
  storage and hosted upload work; paid-runner cache hits bypass transfer.
