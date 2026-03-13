# Math Notes And Oracle DB Design

## Goal

Make `tensor-ad-oracles` the source of truth for two parallel artifacts:

- mathematical AD notes for the known operation rules
- the machine-readable oracle database

The mathematical notes should cover the superset of rules known across
`tensor-ad-oracles` and `tenferro-rs`, while the oracle DB remains a separate
artifact that references the notes where needed.

## Problem

Today the repository is clearly DB-first:

- `cases/` holds the published oracle data
- `schema/` defines the public machine-readable contract
- `README.md` describes the DB and replay contract

The mathematical derivations, however, currently live in
`../tenferro-rs/docs/AD/`. That splits the conceptual contract:

- the DB and validation live here
- the derivations and operator-level AD formulas live elsewhere

This is awkward for long-term maintenance. The relationship between:

- raw operator rules such as SVD `frule` / `rrule`
- DB observables such as `u_abs` or `uvh_product`

is conceptually one system, but the documentation is split across repositories.

## Design Principles

1. Keep the mathematical notes and the oracle DB as separate first-class
   artifacts.
2. Make `tensor-ad-oracles` the source of truth for both artifacts.
3. Avoid coupling note updates to schema changes or JSONL regeneration unless
   the DB contract itself changes.
4. Keep note links stable at the `(op, family)` level.
5. Preserve the distinction between raw operator rules and DB-level observable
   families.

## Approaches Considered

### 1. Central registry with separate math notes and DB

Add a note corpus under `docs/math/`, keep `cases/` as-is, and connect them
through a central `(op, family) -> note anchor` registry.

Pros:

- low update cost when note structure changes
- no schema churn
- no JSONL churn
- keeps the DB stable and machine-readable
- supports notes that exist before DB materialization

Cons:

- the link is indirect rather than embedded in each JSON record

### 2. Embed note URLs directly in each JSON record

Publish note linkage inside every case record.

Pros:

- self-contained records

Cons:

- schema expansion
- large JSONL diffs for link-only updates
- note reorganization forces database churn
- tighter coupling between human docs and stable data artifacts

### 3. README-only cross-reference

Document the note-to-family mapping only in human-readable docs.

Pros:

- smallest immediate change

Cons:

- no structured source of truth for linkage
- difficult to validate automatically
- more fragile over time

## Recommendation

Use **Approach 1: central registry with separate math notes and DB**.

This is the cleanest maintenance boundary:

- math notes can evolve
- the DB contract stays stable
- links remain structured and verifiable
- JSON case files remain focused on numeric oracle content

## Repository Structure

The repository should treat the two artifacts explicitly:

- mathematical notes: `docs/math/<op>.md`
- oracle DB: existing `cases/<op>/<family>.jsonl`
- link registry: `docs/math/registry.json`

The structure is intentionally one-note-per-operation:

- `docs/math/svd.md`
- `docs/math/solve.md`
- `docs/math/eig.md`

DB families are represented inside the note via fixed anchors rather than
separate files per family.

## Note Model

Each note under `docs/math/<op>.md` is the source of truth for the raw operator
rule, not for a single DB observable family.

Recommended structure:

1. forward definition
2. notation and shapes
3. assumptions, singularities, and domain restrictions
4. `frule`
5. `rrule`
6. numerical stabilization and gauge handling
7. verification strategy
8. DB family mapping section

This is important for spectral operations. For example:

- the SVD note defines rules for raw `(U, S, Vh)`
- the DB families reference observables derived from that raw output
  - `u_abs = |U|`
  - `s = S`
  - `vh_abs = |Vh|`
  - `uvh_product = U @ Vh`

That preserves the conceptual distinction between:

- operator-level differentiation rules
- published oracle comparison targets

## Family Anchors

Every DB-backed family should map to a stable anchor inside the op note.

Example:

- `docs/math/svd.md#family-u-abs`
- `docs/math/svd.md#family-s`
- `docs/math/svd.md#family-vh-abs`
- `docs/math/svd.md#family-uvh-product`

The page remains one document per operation; family anchors identify the DB
projection or observable that corresponds to a published family.

## Registry Design

The registry is the source of truth for the DB-to-note linkage.

Recommended file:

- `docs/math/registry.json`

Recommended schema:

```json
{
  "version": 1,
  "entries": [
    {
      "op": "svd",
      "family": "u_abs",
      "note_path": "docs/math/svd.md",
      "anchor": "family-u-abs"
    }
  ]
}
```

The registry should store repository-internal targets, not canonical external
URLs. External URLs can be derived later from the published site layout.

This keeps the source of truth stable if:

- the site domain changes
- docs hosting changes
- generated public URLs change

## Scope Model

The mathematical note corpus should be the superset of known operation rules.

That means:

- notes may exist for operations that are not yet materialized in the DB
- the registry only needs entries for DB-backed `(op, family)` pairs

Examples of the distinction:

- `matrix_exp` and `einsum` may have notes even if there is no published DB
  family yet
- published DB families such as `svd/u_abs` must have a registry entry

## Validation And CI Contract

The linkage should be validated independently of the numeric DB validation.

Required registry checks:

1. no duplicate `(op, family)` entries
2. `note_path` exists
3. `anchor` exists in the target note
4. every materialized DB family has a registry entry

Non-requirement:

- not every note needs to appear in the registry

This keeps the note corpus free to grow ahead of DB publication while ensuring
that the published DB never points nowhere.

## Update Workflow

For note-only changes:

1. edit `docs/math/<op>.md`
2. update `docs/math/registry.json` only if note anchors changed
3. run note/registry validation

For a new DB family:

1. add or update the relevant math note
2. add the `(op, family)` registry entry
3. publish the JSONL family
4. run the normal schema/replay/regeneration validation plus note/registry
   validation

## Relationship To `tenferro-rs`

`tensor-ad-oracles` becomes the source of truth for the math notes, but this
design does not require updating `tenferro-rs` in the same change.

Immediate scope:

- add the note corpus here
- define the registry and validation here
- leave `tenferro-rs` untouched for now

This avoids mixing repository migration work with the note/DB architecture
change itself.

## Risks

### Risk: anchors become unstable

Mitigation:

- use fixed family anchors with explicit naming conventions
- validate anchors in CI

### Risk: notes drift from DB observables

Mitigation:

- require registry coverage for every published family
- keep a dedicated `DB Families` section in each note

### Risk: notes and raw rules are confused with DB observables

Mitigation:

- standardize note structure so raw formulas appear first
- keep observable mapping in a distinct section near the end of each note

## Deliverables

1. `docs/math/` note corpus scaffold
2. `docs/math/registry.json`
3. validator for registry path/anchor/coverage
4. README updates that describe the repository as math notes plus oracle DB
5. migrated or newly authored math notes for known operations

## Non-Goals

This design does not require:

- embedding note links in every JSON record
- changing `schema/case.schema.json` for note linkage
- regenerating existing JSONL files solely to add note references
- updating `tenferro-rs` in the same change
