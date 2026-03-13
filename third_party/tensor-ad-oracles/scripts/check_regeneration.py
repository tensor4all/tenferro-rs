"""Regenerate the published case tree and require byte-for-byte equality."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.pytorch_v1 import materialize_all_case_families


def _relative_case_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*.jsonl")}


def _record_tolerance(record: dict) -> tuple[float, float]:
    comparison = record.get("comparison", {})
    if comparison.get("kind") == "allclose":
        return float(comparison["rtol"]), float(comparison["atol"])
    if "first_order" in comparison:
        return (
            float(comparison["first_order"]["rtol"]),
            float(comparison["first_order"]["atol"]),
        )
    return 0.0, 0.0


def _record_second_order_tolerance(record: dict) -> tuple[float, float] | None:
    comparison = record.get("comparison", {})
    if "second_order" not in comparison:
        return None
    return (
        float(comparison["second_order"]["rtol"]),
        float(comparison["second_order"]["atol"]),
    )


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_comparison_tolerance_path(path: str) -> bool:
    return (
        path.endswith(".comparison.rtol")
        or path.endswith(".comparison.atol")
        or path.endswith(".comparison.first_order.rtol")
        or path.endswith(".comparison.first_order.atol")
        or path.endswith(".comparison.second_order.rtol")
        or path.endswith(".comparison.second_order.atol")
    )


def _uses_second_order_tolerance(path: str) -> bool:
    return ".hvp." in path


def _compare_values(
    expected,
    actual,
    *,
    rtol: float,
    atol: float,
    second_order_rtol: float | None,
    second_order_atol: float | None,
    path: str,
) -> None:
    active_rtol = rtol
    active_atol = atol
    if (
        second_order_rtol is not None
        and second_order_atol is not None
        and _uses_second_order_tolerance(path)
    ):
        active_rtol = second_order_rtol
        active_atol = second_order_atol
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"type mismatch at {path}: expected dict, got {type(actual).__name__}")
        if expected.keys() != actual.keys():
            raise ValueError(f"key mismatch at {path}: {sorted(expected.keys())} != {sorted(actual.keys())}")
        for key in expected:
            child = f"{path}.{key}" if path else key
            _compare_values(
                expected[key],
                actual[key],
                rtol=rtol,
                atol=atol,
                second_order_rtol=second_order_rtol,
                second_order_atol=second_order_atol,
                path=child,
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ValueError(f"type mismatch at {path}: expected list, got {type(actual).__name__}")
        if len(expected) != len(actual):
            raise ValueError(f"length mismatch at {path}: {len(expected)} != {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare_values(
                left,
                right,
                rtol=rtol,
                atol=atol,
                second_order_rtol=second_order_rtol,
                second_order_atol=second_order_atol,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(expected, float):
        if _is_comparison_tolerance_path(path):
            return
        if not isinstance(actual, (int, float)):
            raise ValueError(f"type mismatch at {path}: expected float, got {type(actual).__name__}")
        if math.isnan(expected) and math.isnan(float(actual)):
            return
        if not math.isclose(expected, float(actual), rel_tol=active_rtol, abs_tol=active_atol):
            raise ValueError(f"numeric mismatch at {path}: {expected} != {actual}")
        return

    if isinstance(expected, int) and not isinstance(expected, bool):
        if expected != actual:
            raise ValueError(f"value mismatch at {path}: {expected} != {actual}")
        return

    if expected != actual:
        raise ValueError(f"value mismatch at {path}: {expected!r} != {actual!r}")


def _compare_case_files(expected_path: Path, actual_path: Path) -> None:
    expected_records = _load_jsonl(expected_path)
    actual_records = _load_jsonl(actual_path)
    if len(expected_records) != len(actual_records):
        raise ValueError(
            f"record count mismatch for {expected_path.name}: {len(expected_records)} != {len(actual_records)}"
        )

    for index, (expected_record, actual_record) in enumerate(
        zip(expected_records, actual_records, strict=True)
    ):
        rtol, atol = _record_tolerance(expected_record)
        second_order = _record_second_order_tolerance(expected_record)
        try:
            _compare_values(
                expected_record,
                actual_record,
                rtol=rtol,
                atol=atol,
                second_order_rtol=None if second_order is None else second_order[0],
                second_order_atol=None if second_order is None else second_order[1],
                path=f"record[{index}]",
            )
        except ValueError as exc:
            raise ValueError(f"{expected_path.name}: {exc}") from exc


def compare_case_trees(expected_root: Path, actual_root: Path) -> None:
    """Raise when the two case trees differ beyond case-level tolerances."""
    expected_files = _relative_case_files(expected_root)
    actual_files = _relative_case_files(actual_root)
    if expected_files != actual_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        extra = sorted(str(path) for path in actual_files - expected_files)
        raise ValueError(f"file set mismatch: missing={missing}, extra={extra}")

    for relative in sorted(expected_files):
        expected_path = expected_root / relative
        actual_path = actual_root / relative
        try:
            _compare_case_files(expected_path, actual_path)
        except ValueError as exc:
            raise ValueError(f"content mismatch for {relative}: {exc}") from exc


def check_regeneration(cases_root: Path = CASES_ROOT) -> int:
    """Regenerate the full case tree and require equality with `cases/`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        regenerated_root = Path(tmpdir) / "cases"
        materialize_all_case_families(limit=None, cases_root=regenerated_root)
        compare_case_trees(cases_root, regenerated_root)
    return len(_relative_case_files(cases_root))


def main() -> int:
    compared = check_regeneration()
    print(f"regeneration_checked_files={compared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
