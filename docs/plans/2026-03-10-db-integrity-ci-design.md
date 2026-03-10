# Database Integrity CI Design

## Goal

Protect the published JSON database against silent drift or arbitrary edits by
requiring that published cases remain reproducible from the pinned PyTorch
generator and internally consistent with the finite-difference oracle.

## Scope

Version 1 adds three repository-level guards:

1. replay validation must confirm both stored-oracle reproducibility and
   cross-oracle agreement inside each case tolerance
2. regeneration validation must rebuild the published case tree from the
   pinned generator and require byte-for-byte equality with the checked-in
   database
3. GitHub review policy must require human review for `cases/`, `generators/`,
   `validators/`, and workflow changes

## Recommended Approach

Use two CI lanes.

- Fast lane on pull requests and pushes:
  - schema validation
  - duplicate `case_id` detection
  - full replay of published cases
- Regeneration lane on relevant changes and on a schedule:
  - materialize the full case tree into a temporary directory
  - compare the regenerated tree with the checked-in `cases/`

This keeps the default check relatively light while still ensuring that any
database rewrite must remain reproducible from the pinned generator.

## Design Notes

- The replay validator should not stop at comparing stored values against a
  replayed run. It must also re-check:
  - `replayed_pytorch_jvp ~= replayed_fd_jvp`
  - `<bar_y, replayed_fd_jvp> ~= <replayed_pytorch_vjp, v>`
- `torch` must be version-pinned in `pyproject.toml` so the generator and
  replay validator run against a fixed upstream implementation contract.
- `CODEOWNERS` is necessary but not sufficient. GitHub branch protection still
  has to require CODEOWNERS review for it to be effective.
