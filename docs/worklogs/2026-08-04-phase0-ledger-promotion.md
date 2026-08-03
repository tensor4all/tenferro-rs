# Phase 0 ledger promotion

## Summary

The completion audit for #1555 found that #1596 was merged and #1556 was
closed, while `p0-control-plane` was still deferred and referenced a missing
Cargo test target. This change supplies that frozen artifact and promotes the
obligation without changing production code.

## Decision

A dedicated public-API integration target was chosen over pointing the ledger
at the broad runtime integration suite or deleting the obligation. It exercises
caller-selected identities, provider/device and event-domain separation,
endpoint-pair routing, typed duplicate registration failure, and foreign-event
rejection. It introduces no compatibility path, liveness registry, recovery
protocol, repeated validation, or cryptographic evidence.

## TDD evidence

- RED: after adding P0 to the expected active set, the 24-case ledger suite
  failed only because the production manifest still deferred it.
- GREEN: `python3 scripts/test-storage-ownership-contracts-v2.py` passed 24/24.
- The first PR normalizes the deferred artifact identity and intentionally leaves
  the artifact absent; the following state-only promotion PR owns the integration test.
  passed 3/3.
- MUTATION: temporarily replacing the transfer key with a source-only key made
  the endpoint-pair test fail at the second shared-source route; restoring the
  complete pair returned the target to 3/3.
- The ledger checker, design checker, API-parity test, element-access baseline,
  and formatting check passed before the candidate commit.

## Residual scope

P2-P13 remain deferred. This correction does not start storage implementation
or change the frozen G1-G7 design.
