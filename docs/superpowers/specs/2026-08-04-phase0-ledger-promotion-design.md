# Phase 0 ledger promotion design

Issue: #1556 under #1555
Base: `134ae1018d7c6501e9fc7116a950d4bed06391ed`

## Problem

Phase 0 is merged, but `p0-control-plane` remains deferred and names a test
target that does not exist. The umbrella checklist therefore disagrees with
the executable contract ledger.

## Decision

Add the already-declared `execution_engine_identity` integration-test target
and promote only `p0-control-plane` to `active`. The test uses public runtime
and CPU adapter APIs. It verifies caller-selected engine IDs, distinct provider
and event-domain identities, typed duplicate registration failure, endpoint-pair
routing, and foreign-event rejection.

No production API, compatibility path, registry, generation, recovery state,
cryptographic evidence, repeated validation, or hostile-runner protocol is
introduced. The target is a narrow executable record of the behavior already
merged in #1596.

## Verification

- The ledger test suite must fail when its expected active set includes P0 but
  the manifest still defers it.
- `cargo test -p tenferro-runtime --test execution_engine_identity` must pass.
- The storage ledger checker, design checker, all active obligations, and fast
  repository checks must pass from the committed candidate.
