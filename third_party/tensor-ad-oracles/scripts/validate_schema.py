"""Schema validation helpers for tensor-ad-oracles."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schema" / "case.schema.json"
CASES_ROOT = REPO_ROOT / "cases"


def require_jsonschema():
    """Return the jsonschema module or raise a clear runtime error."""
    try:
        import jsonschema
    except ModuleNotFoundError as exc:
        raise RuntimeError("jsonschema is required to validate tensor-ad-oracles cases") from exc
    return jsonschema


def validate_case_tree(root: Path = CASES_ROOT, schema_path: Path = SCHEMA_PATH) -> int:
    """Validate all published JSONL records against the repository schema."""
    jsonschema = require_jsonschema()

    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)

    validated = 0
    for path in sorted(root.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                jsonschema.validate(instance=json.loads(line), schema=schema)
                validated += 1
    return validated


def main() -> int:
    validated = validate_case_tree()
    print(f"validated_schema_records={validated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
