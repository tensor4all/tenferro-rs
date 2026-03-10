# AGENTS.md

This repository is a `uv`-managed Python project for generating and validating JSON oracle data.

## Environment Rules

- Use `uv` for environment creation and command execution.
- Bootstrap with:

```bash
uv sync --locked --all-groups
```

- Run Python commands through `uv run` once the environment is synced.

## Python Version Pinning

- Keep `.python-version` patch-pinned, not just major/minor.
- The current pinned interpreter is `3.12.12`.
- Reason: using only `3.12` allowed `uv` to pick an external `/sharehome/tools/opt/Python-3.12.11/...` interpreter whose `ctypes` extension was broken and caused imports like `torch` to fail before generation could start.
- After the patch pin, `.venv/bin/python` should resolve to a `uv`-managed interpreter under `~/.local/share/uv/python/...`.

## PyTorch Internal Test Dependencies

- Real case generation depends on PyTorch internal OpInfo modules, not just public `torch`.
- In addition to `torch`, `numpy`, and `jsonschema`, the environment must include `expecttest`.
- Without `expecttest`, imports such as `torch.testing._internal.opinfo.definitions.linalg` fail before `sample_inputs_*` can be used.
- Keep `torch` exactly pinned to `torch==2.10.0`.
- Treat the public version string `2.10.0` as the repository contract. Local build suffixes
  such as `+cu128` or `+cpu` should not appear in generated provenance.

## Operational Notes

- Do not run `uv lock` and `uv sync --locked --all-groups` in parallel. Update the lockfile first, then sync.
- When changing `pyproject.toml`, `.python-version`, or dependency policy, refresh `uv.lock`.
- The repository contract now includes:
  - `uv run python scripts/validate_schema.py`
  - `uv run python scripts/verify_cases.py`
  - `uv run python scripts/check_replay.py`
  - `uv run python scripts/check_regeneration.py`
- `scripts/check_regeneration.py` compares regenerated JSONL files semantically.
  Metadata must match exactly, while numeric tensors may differ only within the
  case-level `comparison.rtol` / `comparison.atol`.
- `CODEOWNERS` only helps if GitHub branch protection requires CODEOWNERS review. Keep that enabled on the public repository.
- Useful smoke checks:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m generators.pytorch_v1 --list
uv run python - <<'PY'
from torch.testing._internal.opinfo.definitions import linalg
from torch.testing._internal.opinfo.core import gradcheck_wrapper_hermitian_input
print(callable(linalg.sample_inputs_svd))
print(callable(gradcheck_wrapper_hermitian_input))
PY
```
