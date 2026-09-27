# Session entry and worker dispatch documentation

## Decisions

- Keep sessions as the standard batching boundary. Explain backend reuse and
  entered-session reuse separately; neither direct faer access nor a pool
  argument automatically eliminates worker handoff.
- Route README, llms.txt, the CPU guides and bundled skill mirrors to one
  [measurement and mitigation guide](../guides/session-entry-cost.md).
- Retain the older Linux design note as historical evidence, with a link to
  the guide. Its timings and #1904's session timings are not directly
  interchangeable with the standalone Rayon measurements.

## Verification conclusions and constraints

- Dispatch probes are independent of tenferro and preserve the measured source,
  lockfile and raw results. They demonstrate cost magnitude on one machine;
  they do not isolate every scheduler component or establish portable limits.
- No public API or execution behavior changes. #1939/#1940 remain linked
  proposals rather than APIs asserted to be available.
- Both published probe recipes execute successfully. The repeat varied materially,
  so the guide explicitly reports that variation rather than treating 26 us as
  a fixed overhead. Skill mirrors, snippet sync, dependency snippets and source
  routing checks pass. The full docs-site check requires generated workspace
  rustdoc output, which is absent in this isolated checkout.
