# Structured Input Semantics README Design

## Goal

Clarify the public contract for published Hermitian-family cases so downstream
replay consumers understand that some records are interpreted under upstream
gradcheck-wrapper semantics rather than naïve raw-operator semantics.

## Context

`tensor-ad-oracles` publishes machine-readable JSON cases plus a human-readable
`README.md`. Downstream consumers are expected to understand the database by
reading:

- `cases/**/*.jsonl`
- `schema/case.schema.json`
- `README.md`

Issue #3 points out that this contract is underspecified for `eigh` and
`cholesky`. Inside the repository, generation and replay already apply
PyTorch-aligned Hermitian wrapper semantics:

- primary inputs are interpreted through the upstream structure-preserving
  wrapper
- probe directions are projected through the same structure
- replay succeeds because it reuses the same runtime path

The numeric data are not wrong, but a downstream consumer cannot infer this
from the current public docs alone.

## Approaches Considered

### 1. README-only clarification

Document the structured-input contract in `README.md` and explicitly call out
the affected v1 families.

Pros:

- fixes the public-contract gap directly
- no schema migration
- no database regeneration

Cons:

- semantics remain human-readable rather than machine-readable

### 2. Schema and record extension

Add a machine-readable field such as `input_structure` or
`gradcheck_wrapper`.

Pros:

- fully self-describing records

Cons:

- schema change
- case regeneration
- downstream parser churn
- larger scope than the issue requires

### 3. Repository-internal documentation only

Document the behavior only in internal design notes or generator comments.

Pros:

- lowest effort

Cons:

- does not fix the public contract
- does not help external consumers

## Recommendation

Use **README-only clarification** for this issue.

The user explicitly chose the documentation-only fix. The issue is about a
public contract gap, not incorrect numeric data, so a focused README update is
enough to resolve it without widening scope.

## Proposed README Changes

Add a short section near the public contract and replay description that
states:

- some published families are interpreted under upstream gradcheck-wrapper
  semantics
- in v1, this applies to `eigh` and `cholesky`
- serialized `inputs` and probe `direction` payloads are raw tensor payloads,
  but oracle evaluation applies the structure-preserving Hermitian wrapper used
  upstream
- downstream replay consumers must not assume raw-operator semantics for those
  families

Also add a non-goal note:

- v1 does not yet publish a machine-readable `input_structure` field

## Non-Goals

This change does not:

- modify `schema/case.schema.json`
- regenerate `cases/**/*.jsonl`
- alter any numeric references
- add new runtime behavior
