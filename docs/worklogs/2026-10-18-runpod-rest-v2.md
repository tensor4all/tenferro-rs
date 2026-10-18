# RunPod REST v2 and rate-limit migration

## Decisions

- Migrate all hosted RunPod consumers together, including catalog pricing,
  lifecycle observations, watchdog, metadata, independent reaper and manual
  cleanup. Adapt the already verified benchmark HTTP/client policy while
  retaining tenferro-rs ownership, obsolete-PR cancellation and cost controls.
- Reserve attempt slots for every reviewed GPU tier, interleave static fallback
  types and deduplicate priced IDs so later tiers remain reachable without
  increasing the paid attempt budget. Cost artifacts prefer v2 placement fields
  while preserving the historical fallback.
- Keep provider cooldown across candidate changes, and reduce ordinary startup
  polling to ten seconds. Never accept an unknown observation as healthy.
- Keep historical saved cost records readable while live requests use v2 fields.
  Cost capture and reporting remain best effort; cleanup failures stay visible
  with the remaining Pod ID and the independent reaper remains a backstop.
- The integrated contract and operational limits live in
  [the provisioning design](../design/runpod-gpu-provisioning.md#rest-v2-migration-and-request-policy).

## Verification conclusions and constraints

- Offline fixtures, a local HTTP integration and mocked time cover API shapes,
  retry limits, Retry-After, startup/runtime transitions and cleanup/reporting failures without paid Pods.
- Real provider scheduling, account-wide quotas and hosted queues remain outside
  this verification. No live paid end-to-end or strict billing guarantee is claimed.
