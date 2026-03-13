# tensor-ad-oracles

`tensor-ad-oracles` publishes two first-class artifacts for dense tensor AD
validation:

- mathematical AD notes
- a machine-readable JSON oracle database

The oracle database covers both scalar-style `OpInfo` families and linear
algebra operations.

Version 1 targets the full PyTorch `OpInfo`-backed AD-relevant dense family set currently materialized in this repository, including:

- dense unary elementwise ops such as `abs`, `exp`, `log`, `sin`, `cos`, `tanh`, `conj`
- dense binary elementwise ops such as `add`, `mul`, `div`, `pow`, `xlogy`
- dense reduction ops such as `sum`, `prod`, `mean`, `std`, `var`
- dense functional scalarizable ops such as `nn.functional.prelu`
- the full AD-relevant linalg family set, including:

  - `cross`
  - `det`
  - `diagonal`
  - `eig`
  - `svd`
  - `eigh`
  - `eigvals`
  - `eigvalsh`
  - `solve`
  - `solve_ex`
  - `solve_triangular`
  - `cholesky`
  - `cholesky_ex`
  - `qr`
  - `lu`
  - `lu_factor`
  - `lu_factor_ex`
  - `lu_solve`
  - `inv`
  - `inv_ex`
  - `matrix_power`
  - `matrix_norm`
  - `multi_dot`
  - `norm`
  - `slogdet`
  - `svdvals`
  - `tensorinv`
  - `tensorsolve`
  - `vecdot`
  - `vector_norm`
  - `vander`
  - `pinv`
  - `pinv_hermitian`
  - `pinv_singular`

## Environment

The repository is `uv`-managed. Use the checked-in patch-pinned Python version and lockfile.

```bash
uv sync --locked --all-groups
```

Typical commands:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m generators.pytorch_v1 --list
uv run python -m generators.pytorch_v1 --materialize solve --family identity --limit 1
uv run python -m generators.pytorch_v1 --materialize-all --limit 1
uv run python -m unittest tests.test_db_replay -v
uv run python scripts/check_math_registry.py
uv run python scripts/validate_schema.py
uv run python scripts/verify_cases.py
uv run python scripts/check_replay.py
uv run python scripts/check_regeneration.py
uv run python scripts/check_tolerances.py
uv run python scripts/check_upstream_ad_tolerances.py
uv run python scripts/report_upstream_publish_coverage.py
```

Repository-managed environment files:

- `.python-version`
- `pyproject.toml`
- `uv.lock`

The repository requires an exact PyTorch dependency pin: `torch==2.10.0`.
Generated provenance stores the public version string `2.10.0`, not local
build suffixes such as `+cpu` or `+cu128`.

## Math Notes

The mathematical AD notes live under `docs/math/`.

- `docs/math/index.md` is the note corpus entrypoint
- `docs/math/registry.json` is the central mapping from published `(op, family)`
  DB families to note locations

`docs/math/*.md` is the mathematical source of truth for known operator rules in
this repository. The note corpus is maintained as a non-lossy migration target
for `tenferro-rs/docs/AD`, and operator notes are cross-checked against
PyTorch's manual autograd formulas where those exist.

The note corpus is intended to be published as a GitHub Pages docs site. The
editable source stays in `docs/math/`, and the deployable web surface is a
rendered view of that same corpus rather than a separate source tree.

The note corpus and the oracle database are maintained as separate artifacts so
that math-note updates do not require schema or JSONL churn unless the DB
contract itself changes. In steady state, the DB should reference math notes via
stable registry anchors rather than duplicating mathematical definitions inside
case files.

Use the registry check to verify that every published `(op, family)` case has a
valid note target:

```bash
uv run python scripts/check_math_registry.py
```

## What Counts As a Case

A case is defined by:

- materialized inputs
- an `observable`
- one or more paired derivative probes

The database does not require raw decomposition outputs to be the comparison target. For spectral operations, the observable may be a processed output such as `U.abs()`, `S`, `Vh.abs()`, or `U @ Vh`, following the same derivative-relevant observables used by PyTorch AD tests.

## Oracle Policy

Every `success` case must provide both:

- `pytorch_ref`
- `fd_ref`

Every published `success` case must satisfy:

- `Jv_torch ~= Jv_fd`
- `<bar_y, Jv_fd> ~= <J*bar_y_torch, v>`

First-order data is published for `float32`, `float64`, `complex64`, and `complex128`.

For upstream second-order families that remain numerically stable enough to publish, `success` probes also carry scalarized HVP data:

- `pytorch_ref.hvp`
- `fd_ref.hvp`

where `hvp` denotes `H_phi(x) v` for `phi(x) = <bar_y, observable(x)>`.

`error` cases do not require numeric references. They encode expected failure behavior with a machine-readable reason code.

## License

This repository is dual-licensed under either of:

- Apache License, Version 2.0, in [LICENSE-APACHE](LICENSE-APACHE)
- MIT license, in [LICENSE-MIT](LICENSE-MIT)

At your option, you may use this repository under either license.

## PyTorch Provenance

Version 1 uses the same AD-relevant case families as PyTorch. Each case stores upstream provenance, including the source file, source function, and source commit used to generate the record.

Publish-surface coverage against the pinned PyTorch upstream inventory is tracked in:

- `docs/generated/pytorch-upstream-publish-coverage.md`

## Repository Layout

```text
README.md
schema/
  case.schema.json
generators/
  __init__.py
  pytorch_v1.py
cases/
  abs/
  add/
  sum/
  ...
  svd/
  eigh/
  solve/
  cholesky/
  qr/
  pinv_singular/
```

Version 1 materializes these family files:

- `cases/*/*.jsonl` for every supported `(op, family)` in `generators.pytorch_v1.build_case_families()`
- `cases/svd/gauge_ill_defined.jsonl`
- `cases/eigh/gauge_ill_defined.jsonl`

## Verification Contract

For every paired probe in every `success` case:

- compare `pytorch_ref.jvp` with `fd_ref.jvp`
- check adjoint consistency with `<bar_y, Jv_fd> ~= <J*bar_y_torch, v>`
- for HVP-enabled families, compare `pytorch_ref.hvp` with `fd_ref.hvp`

All probe directions and cotangents are normalized to unit Frobenius norm after any required structure-preserving projection.

## Structured Input Semantics

Some published families are interpreted under upstream gradcheck-wrapper semantics
rather than naive raw-operator semantics.

In v1, this applies to the Hermitian-wrapper families for `eigh` and
`cholesky`. Their serialized `inputs` and probe `direction` payloads are stored
as raw tensor payloads, but oracle evaluation interprets them through the same
structure-preserving Hermitian wrapper semantics used by the upstream PyTorch
AD tests. Downstream replay consumers must not assume that these families are
published against the unconstrained raw operator on the serialized tensor
alone.

This is a documentation-level contract clarification only. v1 does not yet add
a machine-readable `input_structure` or `gradcheck_wrapper` field to the public
schema or case records.

## Database Replay Validation

The repository includes a replay validator in `validators/replay.py`. It re-executes the stored PyTorch case families from the published JSON database and checks that:

- stored `pytorch_ref.jvp` is reproducible
- stored `pytorch_ref.vjp` is reproducible
- stored `pytorch_ref.hvp` is reproducible when present
- stored `fd_ref.jvp` is reproducible
- stored `fd_ref.hvp` is reproducible when present
- replayed `pytorch_ref.jvp` still matches replayed `fd_ref.jvp` within the case tolerance
- replayed `pytorch_ref.hvp` still matches replayed `fd_ref.hvp` within the second-order case tolerance
- replayed probes still satisfy adjoint consistency within the case tolerance
- expected gauge-ill-defined spectral failures still raise

The end-to-end replay coverage is exercised by:

```bash
uv run python -m unittest tests.test_db_replay -v
uv run python scripts/check_replay.py
```

## CI Guard Rails

The repository ships two CI lanes:

- `oracle-integrity`
  - schema validation
  - duplicate `case_id` detection
  - full replay of the published database
- `oracle-regeneration`
  - full regeneration of `cases/`
  - semantic comparison against the checked-in database using each case tolerance
- `tolerance-audit`
  - verifies published family tolerances are not more than ten orders of magnitude looser than stored cross-oracle residuals
- `upstream-ad-tolerance-audit`
  - verifies direct `torch` vs finite-difference residuals stay within PyTorch's own AD test tolerances

`CODEOWNERS` covers `cases/`, `generators/`, `validators/`, `scripts/`,
`schema/`, and workflow files. To make this effective, GitHub branch protection
must require CODEOWNERS review.
