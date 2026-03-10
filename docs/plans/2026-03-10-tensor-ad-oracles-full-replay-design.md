# Tensor AD Oracles Full Replay Design

## Goal

Update tenferro's vendored `tensor-ad-oracles` integration so the workspace
can scan the full published oracle database, validate all supported families
including scalarized HVP payloads, and explicitly report unsupported families
in checked-in documentation.

## Context

`tensor-ad-oracles` `origin/main` has moved beyond the initial v1 snapshot
already vendored in `tenferro-rs`. The current database now:

- publishes many more `(op, family)` success files
- stores first-order and second-order tolerances separately
- may attach scalarized HVP references under `pytorch_ref.hvp` and `fd_ref.hvp`

The existing tenferro replay harness only understands the earlier schema and
only targets a narrow family subset:

- `solve`
- `cholesky`
- `qr`
- `svd`
- `eigh`
- `pinv_singular`

That harness is no longer sufficient because:

- it cannot parse the new `comparison.{first_order,second_order}` shape
- it ignores published HVP references
- it silently omits most of the now-published database surface

## Scope

This design does **not** require tenferro to implement every oracle-published
operation. Instead, it requires tenferro to:

1. vendor the latest `tensor-ad-oracles` subtree
2. understand the current schema and HVP payloads
3. classify every published case as one of:
   - validated
   - expected error
   - unsupported
   - failure
4. keep unsupported coverage visible in a checked-in Markdown report linked from
   the repository README

## Architecture

### Subtree Update

Refresh `third_party/tensor-ad-oracles/` to the latest upstream snapshot on
`tensor4all/tensor-ad-oracles`.

This preserves:

- published `cases/**/*.jsonl`
- schema
- generator metadata and docs for debugging

### Replay Registry

Introduce an explicit support registry in the replay harness.

The registry key is:

- `op`
- `family`
- `observable.kind`

The registry value is one of:

- `Supported(ReplayKind)`
- `Unsupported { reason }`
- `ExpectedError(ExpectedErrorKind)`

`ReplayKind` identifies which tenferro replay implementation to use, for
example:

- `SolveIdentity`
- `CholeskyIdentity`
- `QrIdentity`
- `SvdUAbs`
- `SvdS`
- `SvdVhAbs`
- `SvdUvhProduct`
- `EighValuesVectorsAbs`
- `PinvSingularIdentity`

This registry becomes the single source of truth for both runtime
classification and documentation generation.

### Replay Outcomes

For every published oracle record, replay produces one of:

- `Validated`
- `ExpectedError`
- `Unsupported`
- `Failure`

Definitions:

- `Validated`: a supported case passed all applicable numeric checks
- `ExpectedError`: an oracle-published expected-failure case failed in the
  expected way
- `Unsupported`: the case is recognized but not implemented in tenferro replay;
  not a test failure, but recorded in coverage docs
- `Failure`: parsing error, unsupported dtype/layout contract violation,
  mismatch against oracle references, or unexpected runtime error

### First- and Second-Order Contract

Supported success cases must validate:

- first-order JVP against `fd_ref.jvp`
- first-order JVP against `pytorch_ref.jvp`
- first-order VJP against `pytorch_ref.vjp`
- adjoint consistency

When the record carries HVP data, supported cases must also validate:

- scalarized HVP against `fd_ref.hvp`
- scalarized HVP against `pytorch_ref.hvp`

Tolerance policy:

- first-order comparisons use `comparison.first_order`
- second-order HVP comparisons use `comparison.second_order`

The HVP meaning follows the oracle contract:

- `hvp = H_phi(x) v`
- `phi(x) = <bar_y, observable(x)>`

### Supported HVP Replay Strategy

For families already supported by tenferro replay, add HVP implementations
using the most stable local mechanism per operation.

Initial target is to extend the existing supported family set with HVP checks
where oracle records provide them.

If tenferro lacks a stable HVP path for a family that is otherwise replayed at
first order, that family must be marked unsupported for second-order replay
until implemented deliberately. The classification must still remain explicit.

### Hermitian Wrapper Semantics

The existing tenferro replay fix for Hermitian-wrapper families remains
required.

`eigh` and `cholesky` records are generated under upstream Hermitian wrapper
semantics. The tenferro replay must continue to mirror that structure when:

- decoding the primary input
- decoding the direction for first-order replay
- scalarizing cotangents for second-order replay
- mapping gradients/HVPs back into the serialized oracle space

This logic should stay localized rather than leaking across unrelated family
implementations.

## Data Model Changes

The local JSON parser must evolve from the old flat comparison model to the new
schema:

- old:
  - `comparison.kind`
  - `comparison.rtol`
  - `comparison.atol`
- new:
  - `comparison.first_order.kind/rtol/atol`
  - `comparison.second_order.kind/rtol/atol`

Probe payload parsing must treat HVP references as optional:

- `probe.pytorch_ref.hvp`
- `probe.fd_ref.hvp`

The parser should reject half-present HVP payloads because the upstream oracle
contract requires them to come in pairs.

## Unsupported Coverage Documentation

Add a checked-in generated report:

- `docs/generated/tensor-ad-oracles-support.md`

This report should summarize:

- supported replayed families
- expected error families
- unsupported families with reason
- record counts per family

The report should be generated deterministically from the vendored subtree plus
the local support registry.

Add a short README section linking to that report so users can discover the
current tenferro coverage against the published oracle database.

## Testing Strategy

### Integration Tests

Keep the existing oracle integration test entrypoint in
`tenferro-linalg/tests/oracle_db/main.rs`, but expand it to:

- validate the new subtree root and parser
- run full-database classification
- assert zero `failure`s
- assert the checked-in unsupported report matches freshly generated output

### Targeted Unit Tests

Add focused tests for:

- new schema decoding
- HVP payload decoding and pair validation
- support registry classification
- report rendering stability

### Workspace Behavior

The replay remains part of the normal workspace test surface. Unsupported
families do not fail the build, but unclassified or numerically failing cases
do.

## Risks

### Dataset Growth

The vendored subtree is larger now and will keep growing. The replay harness
must avoid unnecessary duplication or expensive per-record setup where possible.

### HVP Implementation Gaps

The oracle DB now includes second-order references for many families that
tenferro does not yet support at second order. The registry needs to classify
these cleanly instead of turning them into vague runtime failures.

### Contract Drift

If the support report is hand-maintained, it will rot. The report must be
generated from code and verified in tests.

## Success Criteria

The work is complete when:

1. `third_party/tensor-ad-oracles/` matches the latest upstream subtree
2. tenferro parses the current oracle schema
3. supported families validate first-order and, when implemented, second-order
   HVP references
4. every published oracle record is classified deterministically
5. zero replay failures remain
6. unsupported coverage is documented in
   `docs/generated/tensor-ad-oracles-support.md`
7. README links to that support report
