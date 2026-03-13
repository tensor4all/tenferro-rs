"""Generate a Markdown report for PyTorch-upstream publish coverage."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = REPO_ROOT / "cases"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "generated" / "pytorch-upstream-publish-coverage.md"
PREFERRED_DTYPE_NAMES = ("float64", "complex128", "float32", "complex64")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.pytorch_v1 import (  # noqa: E402
    build_case_spec_index,
    build_supported_scalar_mapping_index,
    build_supported_upstream_mapping_index,
)
from generators.upstream_inventory import collect_ad_relevant_linalg_opinfos  # noqa: E402
from generators.upstream_scalar_inventory import (  # noqa: E402
    collect_ad_relevant_scalar_opinfos,
)


@dataclass(frozen=True)
class PublishCoverageRow:
    op: str
    family: str
    expected_behavior: str
    inventory_kind: str
    upstream_name: str | None
    upstream_variant_name: str
    supported_dtype_names: tuple[str, ...]
    published_dtype_names: tuple[str, ...]
    missing_dtype_names: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=CASES_ROOT,
        help="Root of the checked-in JSONL case tree.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown output path.",
    )
    return parser.parse_args(argv)


def _format_dtypes(names: tuple[str, ...]) -> str:
    if not names:
        return "-"
    return ", ".join(names)


def _load_published_dtype_index(cases_root: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    dtype_index: dict[tuple[str, str], set[str]] = {}
    for path in sorted(cases_root.glob("*/*.jsonl")):
        op = path.parent.name
        family = path.stem
        bucket = dtype_index.setdefault((op, family), set())
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            bucket.add(json.loads(line)["dtype"])
    return {
        key: tuple(
            dtype_name
            for dtype_name in PREFERRED_DTYPE_NAMES
            if dtype_name in names
        )
        for key, names in dtype_index.items()
    }


def collect_publish_coverage_rows(
    cases_root: Path = CASES_ROOT,
) -> list[PublishCoverageRow]:
    spec_index = build_case_spec_index()
    published_dtype_index = _load_published_dtype_index(cases_root)
    rows: list[PublishCoverageRow] = []
    for key in sorted(spec_index):
        spec = spec_index[key]
        published_dtype_names = published_dtype_index.get(key, ())
        missing_dtype_names = tuple(
            dtype_name
            for dtype_name in spec.supported_dtype_names
            if dtype_name not in published_dtype_names
        )
        rows.append(
            PublishCoverageRow(
                op=spec.op,
                family=spec.family,
                expected_behavior=spec.expected_behavior,
                inventory_kind=spec.inventory_kind,
                upstream_name=spec.upstream_name,
                upstream_variant_name=spec.upstream_variant_name,
                supported_dtype_names=spec.supported_dtype_names,
                published_dtype_names=published_dtype_names,
                missing_dtype_names=missing_dtype_names,
            )
        )
    return rows


def build_report_text(cases_root: Path = CASES_ROOT) -> str:
    rows = collect_publish_coverage_rows(cases_root)
    linalg_inventory = collect_ad_relevant_linalg_opinfos()
    scalar_inventory = collect_ad_relevant_scalar_opinfos()
    mapped_success_families = sum(
        len(targets) for targets in build_supported_upstream_mapping_index().values()
    ) + sum(len(targets) for targets in build_supported_scalar_mapping_index().values())
    error_family_count = sum(1 for row in rows if row.expected_behavior == "error")
    missing_rows = [row for row in rows if row.missing_dtype_names]

    lines = [
        "# PyTorch Upstream Publish Coverage",
        "",
        "Generated from the pinned PyTorch upstream inventory, the mapped DB family surface,",
        "and the checked-in `cases/` tree.",
        "",
        "## Upstream Inventory",
        "",
        f"- AD-relevant linalg upstream variants: {len(linalg_inventory)}",
        f"- AD-relevant scalar upstream variants: {len(scalar_inventory)}",
        f"- Mapped publishable success families: {mapped_success_families}",
        f"- Explicit publishable error families: {error_family_count}",
        f"- Total tracked DB families: {len(rows)}",
        "",
        "## Publishable Family Coverage",
        "",
        "Published dtypes are read from the checked-in JSONL files. Missing dtypes indicate",
        "publishable upstream coverage that is not yet materialized in this repository.",
        "",
        "| op | family | behavior | supported dtypes | published dtypes | missing publishable dtypes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.op} | {row.family} | {row.expected_behavior} | "
            f"{_format_dtypes(row.supported_dtype_names)} | "
            f"{_format_dtypes(row.published_dtype_names)} | "
            f"{_format_dtypes(row.missing_dtype_names)} |"
        )

    lines.extend(
        [
            "",
            "## Missing Publishable Coverage",
            "",
        ]
    )
    if not missing_rows:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| op | family | behavior | published dtypes | missing publishable dtypes |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in missing_rows:
            lines.append(
                f"| {row.op} | {row.family} | {row.expected_behavior} | "
                f"{_format_dtypes(row.published_dtype_names)} | "
                f"{_format_dtypes(row.missing_dtype_names)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report_text(args.cases_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"publish_coverage_report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
