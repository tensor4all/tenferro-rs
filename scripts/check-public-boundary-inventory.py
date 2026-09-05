#!/usr/bin/env python3
"""Check and generate the public-boundary overhead inventory.

The Rust/extension sources are membership authorities.  The JSON overlay stores
only reviewed responsibility and disposition data; this script refuses drift,
ambiguous selectors, malformed authority syntax, and stale generated output.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "docs/internals/public-boundary-overhead-inventory.json"
SNAPSHOT = ROOT / "docs/internals/public-boundary-overhead-inventory.md"
BENCHMARK_EXPORT = ROOT / "docs/internals/public-boundary-benchmarks.json"
BASELINE = "0457a2ed0aeea21b14f4297f7f4731e09b3a0507"
SURFACES = ("concrete", "typed", "borrowed", "output_reuse", "eager", "traced")
DISPOSITIONS = {"measured", "alias-with-evidence", "unsupported", "follow-up"}
OWNER_ROLES = {"validation", "metadata", "planning", "admission", "execution", "wrapping"}
ROUTE_SURFACES = {"concrete", "eager", "traced"}
FAMILY_SPECS = {
    "core": ("crates/tenferro-core-ops/src/catalog.rs", "primitive_ops"),
    "einsum": ("crates/tenferro-einsum/src/cache.rs", "EINSUM_EXTENSION_FAMILY_ID"),
    "linalg": ("crates/tenferro-linalg/src/extension.rs", "LinalgOp"),
    "fft": ("crates/tenferro-fft/src/spec.rs", "FftOperation"),
    "sparse": ("ext/sparse/src/extension.rs", "FAMILY_ID"),
    "tropical": ("ext/tropical/src/extension.rs", "TROPICAL_EINSUM_FAMILY_ID"),
}
KNOWN_EXTENSION_DECLS = {
    "crates/tenferro-einsum/src/cache.rs": {"EINSUM_EXTENSION_FAMILY_ID"},
    "crates/tenferro-linalg/src/extension.rs": {"LINALG_EXTENSION_FAMILY_ID"},
    "crates/tenferro-fft/src/lib.rs": {"FFT_EXTENSION_FAMILY_ID"},
    "ext/sparse/src/extension.rs": {"FAMILY_ID", "JVP_FAMILY_ID", "VJP_FAMILY_ID"},
    "ext/tropical/src/extension.rs": {"TROPICAL_EINSUM_FAMILY_ID", "TROPICAL_EINSUM_JVP_FAMILY_ID", "TROPICAL_EINSUM_VJP_FAMILY_ID"},
}
KNOWN_EXTENSION_FILES = set(KNOWN_EXTENSION_DECLS)

@dataclasses.dataclass(frozen=True)
class Authority:
    family: str
    source: str
    symbol: str
    operations: tuple[str, ...]
    categories: dict[str, str]


def _read(path: pathlib.Path, root: pathlib.Path = ROOT) -> str:
    return (root / path).read_text(encoding="utf-8")


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _core(path: pathlib.Path, root: pathlib.Path) -> tuple[tuple[str, ...], dict[str, str]]:
    text = _read(path, root)
    match = re.search(r"macro_rules!\s+primitive_ops\s*\{(?P<body>.*?)\n\s*\};", text, re.S)
    if not match:
        raise ValueError(f"unsupported core catalog syntax in {path}")
    found: list[str] = []
    categories: dict[str, str] = {}
    for line in match.group("body").splitlines():
        line = line.strip()
        if not line or line.startswith("($") or line.startswith("$macro"):
            continue
        if line == "}":
            continue
        item = re.fullmatch(r"([A-Z][A-Za-z0-9_]*),\s*\"([a-z][a-z0-9_]*)\",\s*([A-Za-z]+),\s*[A-Za-z]+,\s*\d+,\s*(?:\d+|u8::MAX),\s*(?:true|false);", line)
        if not item:
            raise ValueError(f"unsupported core catalog entry syntax: {line}")
        found.append(item.group(2))
        categories[item.group(2)] = item.group(3).lower()
    if not found:
        raise ValueError("core catalog yielded no operations")
    return tuple(found), categories


def _enum(path: pathlib.Path, enum_name: str, root: pathlib.Path) -> tuple[str, ...]:
    text = _read(path, root)
    match = re.search(rf"enum\s+{re.escape(enum_name)}\s*\{{(?P<body>.*?)\n\}}", text, re.S)
    if not match:
        raise ValueError(f"unsupported {enum_name} syntax in {path}")
    found: list[str] = []
    in_struct_variant = False
    for line in match.group("body").splitlines():
        line = line.strip()
        if not line or line.startswith("///") or line.startswith("#"):
            continue
        if in_struct_variant:
            if line.startswith("}"):
                in_struct_variant = False
            continue
        variant = re.fullmatch(r"([A-Z][A-Za-z0-9_]*)\s*(,|\{)?", line)
        if variant:
            found.append(_snake(variant.group(1)))
            in_struct_variant = variant.group(2) == "{"
        else:
            raise ValueError(f"unsupported {enum_name} variant syntax: {line}")
    if in_struct_variant:
        raise ValueError(f"unterminated {enum_name} variant")
    if not found:
        raise ValueError(f"{enum_name} yielded no operations")
    return tuple(dict.fromkeys(found))


def _family_ids(path: pathlib.Path, prefix: str, root: pathlib.Path) -> tuple[str, ...]:
    text = _read(path, root)
    values = re.findall(rf"(?:const|pub\s+const)\s+([A-Z0-9_]*FAMILY_ID)\s*:[^=]+?=\s*\"([^\"]+)\"", text)
    if not values:
        raise ValueError(f"no family identifiers found in {path}")
    name_prefix = prefix.removesuffix("_FAMILY_ID")
    selected = [(name, value) for name, value in values if (name.endswith(prefix) if prefix == "FAMILY_ID" else name.startswith(name_prefix))]
    if not selected:
        raise ValueError(f"family identifier {prefix} not found in {path}")
    return tuple(value.rsplit(".", 2)[-2] if value.endswith(".v1") else value for _, value in selected)


def _discover_extension_files(root: pathlib.Path) -> set[str]:
    """Discover extension family declarations instead of trusting a fixed list."""
    candidates = set()
    for pattern in ("crates/*/src/**/*.rs", "ext/*/src/**/*.rs"):
        for path in root.glob(pattern):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(root).as_posix()
            if "tenferro-internal-" in relative:
                continue
            names = set(re.findall(r"(?:const|pub\s+const)\s+([A-Z0-9_]*FAMILY_ID)\s*:", text))
            if names:
                candidates.add(relative)
                expected = KNOWN_EXTENSION_DECLS.get(relative)
                if expected is not None and names != expected:
                    raise ValueError(f"extension family declarations drift in {relative}")
    unknown = candidates - KNOWN_EXTENSION_FILES
    if unknown:
        raise ValueError("unreviewed extension family sources: " + ", ".join(sorted(unknown)))
    missing = KNOWN_EXTENSION_FILES - candidates
    if missing:
        raise ValueError("missing extension family source: " + ", ".join(sorted(missing)))
    return candidates


def authorities(root: pathlib.Path = ROOT) -> dict[str, Authority]:
    _discover_extension_files(root)
    result: dict[str, Authority] = {}
    for family, (source, symbol) in FAMILY_SPECS.items():
        path = pathlib.Path(source)
        if family == "core":
            ops, categories = _core(path, root)
        elif family == "linalg":
            ops = _enum(path, "LinalgOp", root)
            categories = {op: "extension" for op in ops}
        elif family == "fft":
            ops = _enum(path, "FftOperation", root)
            categories = {op: "extension" for op in ops}
        elif family == "einsum":
            text = _read(path, root)
            if "EINSUM_EXTENSION_FAMILY_ID" not in text:
                raise ValueError("einsum family identifier not found")
            ops = ("einsum",)
            categories = {"einsum": "extension"}
        elif family == "sparse":
            ops = _family_ids(path, "FAMILY_ID", root)
            categories = {op: "extension" for op in ops}
        else:
            ops = _family_ids(path, "TROPICAL_EINSUM_FAMILY_ID", root)
            categories = {op: "extension" for op in ops}
        result[family] = Authority(family, source, symbol, tuple(sorted(set(ops))), categories)
    return result


def _source_ref(ref: dict[str, Any], root: pathlib.Path, kind: str = "owner") -> None:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("symbol"), str):
        raise ValueError(f"{kind} references must contain path and symbol")
    path = root / ref["path"]
    if not path.is_file():
        raise ValueError(f"{kind} source does not exist: {ref['path']}")
    text = path.read_text(encoding="utf-8")
    symbol = re.escape(ref["symbol"])
    visibility = r"(?:(?:pub)(?:\s*\([^)]*\))?\s+)?"
    declaration = rf"^\s*{visibility}(?:(?:async)\s+)?(?:fn|struct|trait|enum|type|const)\s+{symbol}\b"
    fn_declaration = rf"^\s*{visibility}(?:(?:async)\s+)?fn\s+{symbol}\b"
    macro = rf"^\s*macro_rules!\s+{symbol}\b"
    pattern = fn_declaration if kind == "test" else rf"(?:{declaration}|{macro})"
    if not re.search(pattern, text, re.MULTILINE):
        raise ValueError(f"{kind} symbol not found: {ref['path']}::{ref['symbol']}")


def validate_overlay(data: dict[str, Any], auth: dict[str, Authority], root: pathlib.Path = ROOT) -> list[dict[str, Any]]:
    if data.get("schema") != "tenferro.public-boundary-overhead.v1":
        raise ValueError("unsupported inventory schema")
    families = data.get("families")
    if not isinstance(families, dict) or set(families) != set(auth):
        raise ValueError("overlay families must exactly match authority families")
    rows: list[dict[str, Any]] = []
    for family, authority in auth.items():
        entry = families[family]
        if entry.get("authority", {}).get("path") != authority.source or entry.get("authority", {}).get("symbol") != authority.symbol:
            raise ValueError(f"authority reference mismatch for {family}")
        selectors = entry.get("selectors")
        if not isinstance(selectors, list) or not selectors:
            raise ValueError(f"{family} has no selectors")
        seen: dict[str, str] = {}
        for selector in selectors:
            if not isinstance(selector, dict) or not isinstance(selector.get("id"), str):
                raise ValueError(f"{family} has malformed selector")
            selected = selector.get("operations")
            if not isinstance(selected, list) or not selected:
                raise ValueError(f"{family}/{selector.get('id')} has no operations")
            if selector.get("category") is None:
                raise ValueError(f"{family}/{selector['id']} has no category")
            for op in selected:
                if op not in authority.operations:
                    raise ValueError(f"unknown operation selector {family}/{op}")
                if selector["category"] != authority.categories[op]:
                    raise ValueError(f"category mismatch for {family}/{op}")
                if op in seen:
                    raise ValueError(f"overlapping operation selector {family}/{op}")
                seen[op] = selector["id"]
            if "owners" in selector or "contracts" in selector:
                raise ValueError(f"{family}/{selector['id']} must keep owners/contracts per surface")
            surfaces = selector.get("surfaces")
            if not isinstance(surfaces, dict) or set(surfaces) != set(SURFACES):
                raise ValueError(f"{family}/{selector['id']} must resolve every surface")
            for surface, details in surfaces.items():
                if not isinstance(details, dict) or details.get("disposition") not in DISPOSITIONS:
                    raise ValueError(f"invalid disposition for {family}/{selector['id']}/{surface}")
                if not isinstance(details.get("contract"), str) or surface not in details["contract"]:
                    raise ValueError(f"{family}/{selector['id']}/{surface} needs a route-specific contract")
                if details["disposition"] in {"measured", "alias-with-evidence"} and not details.get("evidence"):
                    raise ValueError(f"{family}/{selector['id']}/{surface} needs evidence")
                if details["disposition"] in {"unsupported", "follow-up"} and not details.get("reason"):
                    raise ValueError(f"{family}/{selector['id']}/{surface} needs a reason")
                owners = details.get("owners")
                if owners is None:
                    if surface in ROUTE_SURFACES:
                        raise ValueError(f"{family}/{selector['id']}/{surface} has incomplete owners")
                elif not isinstance(owners, dict) or not OWNER_ROLES <= set(owners):
                    raise ValueError(f"{family}/{selector['id']}/{surface} has incomplete owners")
                elif owners:
                    for role, refs in owners.items():
                        if isinstance(refs, dict):
                            if refs.get("status") != "not-applicable" or not refs.get("reason"):
                                raise ValueError(f"{family}/{selector['id']}/{surface}/{role} has invalid not-applicable owner")
                            continue
                        if not isinstance(refs, list) or not refs:
                            raise ValueError(f"{family}/{selector['id']}/{surface} has empty owner role")
                        for ref in refs:
                            _source_ref(ref, root)
                contracts = details.get("contracts")
                if surface in ROUTE_SURFACES:
                    required_contracts = {"metadata", "allocation", "allocation_source", "payload_boundary", "owner_lifetime", "placement"}
                    if not isinstance(contracts, dict) or not required_contracts <= set(contracts):
                        raise ValueError(f"{family}/{selector['id']}/{surface} has incomplete contracts")
                    for key in ("allocation_source", "payload_boundary"):
                        if not isinstance(contracts[key], str) or "::" not in contracts[key]:
                            raise ValueError(f"{family}/{selector['id']}/{surface} has an ungrounded {key} contract")
            cases = selector.get("cases")
            if not isinstance(cases, list) or not cases:
                raise ValueError(f"{family}/{selector['id']} has no case contracts")
            covered: set[str] = set()
            for case in cases:
                if not isinstance(case, dict):
                    raise ValueError(f"malformed case contract for {family}/{selector['id']}")
                case_operations = case.get("operations", selected)
                if not isinstance(case_operations, list) or not case_operations:
                    raise ValueError(f"case {case.get('id')} operations must be a nonempty list")
                if any(not isinstance(op, str) for op in case_operations):
                    raise ValueError(f"case {case.get('id')} operations must contain strings")
                if len(case_operations) != len(set(case_operations)):
                    raise ValueError(f"case {case.get('id')} operations must be unique")
                external = sorted(set(case_operations) - set(selected))
                if external:
                    raise ValueError(f"case {case.get('id')} operations outside selector: {', '.join(external)}")
                if not case.get("id") or case.get("surface") not in SURFACES or not case.get("phase") or not case.get("setup_boundary"):
                    raise ValueError(f"malformed case contract for {family}/{selector['id']}")
                if not isinstance(case.get("tests"), list) or not case["tests"]:
                    raise ValueError(f"case {case.get('id')} has no regression tests")
                for test in case["tests"]:
                    _source_ref(test, root, "test")
                covered.update(case_operations)
                for op in case_operations:
                    case_id = str(case["id"]).replace("{family}", family).replace("{operation}", op)
                    row = dict(case, id=case_id)
                    row.update(
                        family=family,
                        operation=op,
                        selector=selector["id"],
                        surfaces=surfaces,
                        owners={surface: details.get("owners", {}) for surface, details in surfaces.items()},
                        contracts={surface: details.get("contracts", {}) for surface, details in surfaces.items()},
                        category=authority.categories[op],
                    )
                    rows.append(row)
            missing_cases = sorted(set(selected) - covered)
            if missing_cases:
                raise ValueError(f"uncovered operations in {family}/{selector['id']}: {', '.join(missing_cases)}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate expanded case id")
        uncovered = sorted(set(authority.operations) - set(seen))
        if uncovered:
            raise ValueError(f"uncovered operations in {family}: {', '.join(uncovered)}")
    return sorted(rows, key=lambda row: (row["family"], row["operation"], row["id"]))


def _digest(data: dict[str, Any], auth: dict[str, Authority], root: pathlib.Path) -> str:
    payload = {"overlay": data, "authorities": {name: dataclasses.asdict(value) for name, value in auth.items()}}
    for value in auth.values():
        payload["source:" + value.source] = _read(pathlib.Path(value.source), root)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_snapshot(data: dict[str, Any], auth: dict[str, Authority], rows: list[dict[str, Any]], root: pathlib.Path = ROOT) -> str:
    digest = _digest(data, auth, root)
    lines = ["<!-- GENERATED FILE: scripts/check-public-boundary-inventory.py; do not edit. -->", "# Public-boundary overhead inventory", "", f"- Baseline revision: `{data['baseline_revision']}`", f"- Generation provenance: `{data.get('generation_provenance', 'local source checkout')}`", f"- Overlay/source digest: `{digest}`", "", "| Family | Operation | Category | Surfaces (disposition) | Case contracts |", "|---|---|---|---|---|"]
    for row in rows:
        dispositions = ", ".join(f"{surface}: {row['surfaces'][surface]['disposition']}" for surface in SURFACES)
        lines.append(f"| {row['family']} | `{row['operation']}` | {row['category']} | {dispositions} | {row['id']} |")
    lines += ["", "All routes retain explicit owner, metadata, allocation/materialization, placement, and lifetime contracts in the maintained JSON overlay. No source observation is a timing or allocation count. Benchmark #95 remains pending; #96 is a follow-up.", ""]
    return "\n".join(lines)


def select_case_ids(rows: list[dict[str, Any]], changed_paths: list[str]) -> list[str]:
    """Select cases conservatively from changed owner paths."""
    normalized = {path.strip().removeprefix("./") for path in changed_paths if path.strip()}
    if not normalized:
        return []
    owner_paths = {
        ref["path"]
        for row in rows
        for surface_owners in row["owners"].values()
        for refs in surface_owners.values()
        if isinstance(refs, list)
        for ref in refs
    }
    relevant = {path for path in normalized if path in owner_paths}
    unknown_relevant = any(path.startswith(("crates/", "ext/")) and path not in owner_paths for path in normalized)
    if unknown_relevant:
        return sorted({row["id"] for row in rows})
    if not relevant:
        return []
    return sorted(
        {
            row["id"]
            for row in rows
            if any(
                ref["path"] in relevant
                for surface_owners in row["owners"].values()
                for refs in surface_owners.values()
                if isinstance(refs, list)
                for ref in refs
            )
        }
    )


def benchmark_payload(data: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [{key: row[key] for key in ("id", "family", "operation", "surface", "phase", "setup_boundary", "workflow")} | {"status": "pending"} for row in rows]
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case IDs are not unique")
    return {
        "schema": "tenferro.public-boundary-benchmarks.v1",
        "baseline_revision": data["baseline_revision"],
        "benchmarks": [
            {"id": "95", "status": "pending", "question": "measure public-boundary setup and execution overhead", "case_ids": case_ids},
            {"id": "96", "status": "follow-up", "question": "validate selected representative cases against executable benchmark contracts", "case_ids": case_ids},
        ],
        "cases": cases,
    }


def export_benchmarks(data: dict[str, Any], rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.write_text(json.dumps(benchmark_payload(data, rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="rewrite the deterministic Markdown snapshot")
    parser.add_argument("--export-benchmarks", type=pathlib.Path)
    parser.add_argument("--changed-path", action="append", default=[], help="select representative case IDs for changed paths")
    args = parser.parse_args(argv)
    try:
        auth = authorities()
        data = json.loads(OVERLAY.read_text(encoding="utf-8"))
        rows = validate_overlay(data, auth)
        rendered = render_snapshot(data, auth, rows)
        if args.generate:
            SNAPSHOT.write_text(rendered, encoding="utf-8")
        elif SNAPSHOT.read_text(encoding="utf-8") != rendered:
            raise ValueError("generated inventory snapshot is stale; run --generate")
        expected_benchmarks = json.dumps(benchmark_payload(data, rows), indent=2, sort_keys=True) + "\n"
        if args.export_benchmarks:
            export_benchmarks(data, rows, args.export_benchmarks)
        elif BENCHMARK_EXPORT.read_text(encoding="utf-8") != expected_benchmarks:
            raise ValueError("benchmark export is stale; run --export-benchmarks docs/internals/public-boundary-benchmarks.json")
        if args.changed_path:
            print(json.dumps({"selected_case_ids": select_case_ids(rows, args.changed_path)}))
        if not args.generate and not args.changed_path:
            print(f"public-boundary inventory: {len(rows)} case contracts across {len(auth)} families")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"public-boundary inventory: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
