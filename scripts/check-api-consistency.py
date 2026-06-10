#!/usr/bin/env python3
"""Generate first-pass API consistency candidates for the release freeze."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


ROOT = pathlib.Path(__file__).resolve().parents[1]

PUBLIC_RE = re.compile(
    r'^\s*pub\s+'
    r'(?!(?:\([^)]*\)|crate\b|super\b|self\b|in\b))'
    r'(?:(?:async|unsafe|extern\s+"[^"]+")\s+)*'
    r"(fn|struct|enum|trait|type|const|static|mod|use)\b"
    r"\s*([A-Za-z_][A-Za-z0-9_]*)?"
)

PER_DTYPE_CTOR_RE = re.compile(
    r"^(?:from|new|constant|scalar)_(?:f32|f64|i32|i64|bool|c32|c64)$"
)

CONCEPT_PATTERNS = {
    "reshape": re.compile(r"reshape"),
    "transpose": re.compile(r"transpose"),
    "slice": re.compile(r"slice"),
    "broadcast": re.compile(r"broadcast"),
    "matmul/dot_general": re.compile(r"matmul|dot_general"),
    "reduce": re.compile(r"reduce_(?:sum|prod|max|min)"),
    "gather/scatter": re.compile(r"gather|scatter"),
    "pad/concatenate/reverse": re.compile(r"pad|concatenate|reverse"),
    "convert": re.compile(r"convert"),
    "flat-buffer constructors": re.compile(
        r"(?:from|into|try_into)_vec_(?:col|row)_major"
    ),
}

USER_DOC_JARGON = (
    "tenferro_internal",
    "tenferro-internal",
    "computegraph",
    "ExecOp",
    "ValueRef",
    "StableHLO",
)

MATRIX_EXCLUDED_CRATES = {
    "tenferro-core-ops",
    "tenferro-internal-extension-macros",
    "tenferro-internal-ops",
}


@dataclasses.dataclass(frozen=True)
class CrateInfo:
    name: str
    path: pathlib.Path
    lib_name: str
    features: frozenset[str]
    publish: bool


@dataclasses.dataclass(frozen=True)
class PublicItem:
    crate: str
    crate_path: pathlib.Path
    file: pathlib.Path
    line: int
    kind: str
    name: str
    signature: str

    def location(self, root: pathlib.Path) -> str:
        return f"{self.file.relative_to(root)}:{self.line}"


@dataclasses.dataclass(frozen=True)
class Finding:
    category: str
    location: str
    evidence: str
    expected: str


def load_toml(path: pathlib.Path) -> dict:
    if tomllib is None:
        raise RuntimeError("Python 3.11+ is required for tomllib support")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def workspace_crates(root: pathlib.Path) -> list[CrateInfo]:
    workspace = load_toml(root / "Cargo.toml")["workspace"]
    crates: list[CrateInfo] = []
    for member in workspace["members"]:
        member_path = root / member
        manifest_path = member_path / "Cargo.toml"
        if not manifest_path.exists():
            continue
        manifest = load_toml(manifest_path)
        if "package" not in manifest:
            continue
        if member == "docs/tutorial-code":
            continue
        if "lib" not in manifest and not (member_path / "src" / "lib.rs").exists():
            continue
        package_name = manifest["package"]["name"]
        lib_name = manifest.get("lib", {}).get("name", package_name.replace("-", "_"))
        features = frozenset(manifest.get("features", {}).keys())
        publish_value = manifest["package"].get("publish", True)
        crates.append(
            CrateInfo(package_name, member_path, lib_name, features, publish_value is not False)
        )
    return crates


def rust_source_files(crate_path: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    src = crate_path / "src"
    for path in sorted(src.rglob("*.rs")):
        if "tests" in path.relative_to(src).parts or path.name == "tests.rs":
            continue
        files.append(path)
    return files


def reexport_name(line: str) -> str:
    body = line.split("pub use", 1)[1].strip().rstrip(";")
    body = body.split(" as ")[-1]
    body = body.rsplit("::", 1)[-1]
    body = body.strip("{} ")
    return body or "<reexport>"


def signature_from(lines: list[str], start: int) -> str:
    collected: list[str] = []
    parens = 0
    for line in lines[start : min(len(lines), start + 24)]:
        stripped = line.strip()
        collected.append(stripped)
        parens += stripped.count("(") - stripped.count(")")
        if parens <= 0 and (";" in stripped or "{" in stripped):
            break
    return " ".join(part for part in collected if part)


def collect_public_items(root: pathlib.Path, crates: list[CrateInfo]) -> list[PublicItem]:
    items: list[PublicItem] = []
    for crate in crates:
        for source in rust_source_files(crate.path):
            lines = source.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines):
                match = PUBLIC_RE.match(line)
                if not match:
                    continue
                kind = match.group(1)
                name = match.group(2) or ""
                if kind == "use":
                    name = reexport_name(line)
                if not name:
                    continue
                items.append(
                    PublicItem(
                        crate=crate.name,
                        crate_path=crate.path.relative_to(root),
                        file=source,
                        line=idx + 1,
                        kind=kind,
                        name=name,
                        signature=signature_from(lines, idx),
                    )
                )
    return items


def user_facing_docs(root: pathlib.Path) -> list[pathlib.Path]:
    docs = [root / "README.md"]
    guides = root / "docs" / "guides"
    if guides.exists():
        docs.extend(sorted(guides.rglob("*.md")))
    return [path for path in docs if path.exists()]


def check_public_items(root: pathlib.Path, items: list[PublicItem]) -> list[Finding]:
    findings: list[Finding] = []
    for item in items:
        location = item.location(root)
        if item.name.startswith("traced_"):
            findings.append(
                Finding(
                    "traced_prefix",
                    location,
                    f"`{item.name}` is public",
                    "Traced tensor APIs should not use a `traced_` prefix.",
                )
            )
        if item.kind == "fn" and item.name.endswith("_read"):
            if "TensorRead" not in item.signature and "Read" not in item.signature:
                findings.append(
                    Finding(
                        "read_suffix_without_read_input",
                        location,
                        item.signature,
                        "`_read` is reserved for APIs that explicitly accept read-style inputs.",
                    )
                )
        if item.kind == "fn" and PER_DTYPE_CTOR_RE.match(item.name):
            findings.append(
                Finding(
                    "per_dtype_constructor",
                    location,
                    f"`{item.name}` is public",
                    "Prefer generic TensorScalar-bounded constructors over per-dtype constructors.",
                )
            )
    return findings


def check_features(root: pathlib.Path, crates: list[CrateInfo]) -> list[Finding]:
    findings: list[Finding] = []
    for crate in crates:
        if crate.publish and "gpu" in crate.features:
            findings.append(
                Finding(
                    "public_gpu_feature",
                    f"{crate.path.relative_to(root)}/Cargo.toml",
                    f"`{crate.name}` exposes a `gpu` feature",
                    "User-facing backend features should be concrete backend names such as `cuda` or `rocm`.",
                )
            )
    return findings


def check_user_docs(root: pathlib.Path) -> list[Finding]:
    findings: list[Finding] = []
    for doc in user_facing_docs(root):
        for line_no, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            location = f"{doc.relative_to(root)}:{line_no}"
            if "tenferro::" in line:
                findings.append(
                    Finding(
                        "facade_path_in_user_docs",
                        location,
                        line.strip(),
                        "User-facing docs should import direct public crates, not a root facade path.",
                    )
                )
            for token in USER_DOC_JARGON:
                if token in line:
                    findings.append(
                        Finding(
                            "internal_jargon_in_user_docs",
                            location,
                            line.strip(),
                            "User-facing docs should avoid internal crate paths and graph/IR implementation vocabulary.",
                        )
                    )
    return findings


def surface_for(item: PublicItem) -> str:
    signature = f"{item.file.as_posix()} {item.signature}".lower()
    if item.crate == "tenferro-tensor-core":
        return "tensor-core"
    if item.crate == "tenferro-tensor":
        return "tensor"
    if item.crate == "tenferro-cpu":
        return "cpu-backend"
    if item.crate == "tenferro-gpu":
        return "gpu-backend"
    if item.crate == "tenferro-ad" and "eager" in signature:
        return "eager"
    if item.crate == "tenferro-ad":
        return "ad"
    if item.crate == "tenferro-runtime" and "traced" in signature:
        return "traced"
    if item.crate == "tenferro-runtime":
        return "runtime"
    if item.crate in {"tenferro-einsum", "tenferro-linalg", "tenferro-fft"}:
        return "extension"
    return "other"


def concept_groups(items: list[PublicItem]) -> dict[str, list[PublicItem]]:
    groups: dict[str, list[PublicItem]] = {}
    functions = [
        item
        for item in items
        if item.kind == "fn" and item.crate not in MATRIX_EXCLUDED_CRATES
    ]
    for concept, pattern in CONCEPT_PATTERNS.items():
        matches = [item for item in functions if pattern.search(item.name)]
        if matches:
            groups[concept] = matches
    return groups


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_report(
    root: pathlib.Path,
    crates: list[CrateInfo],
    items: list[PublicItem],
    findings: list[Finding],
) -> str:
    lines: list[str] = []
    lines.append("# API Consistency Report")
    lines.append("")
    lines.append(
        f"api-consistency-report: {len(crates)} crates, {len(items)} lexically public items, {len(findings)} convention findings"
    )
    lines.append("")
    lines.append("## Convention Findings")
    lines.append("")
    if findings:
        for finding in findings:
            lines.append(f"### {finding.category}: {finding.location}")
            lines.append("")
            lines.append(f"- Evidence: {markdown_escape(finding.evidence)}")
            lines.append(f"- Expected: {markdown_escape(finding.expected)}")
            lines.append("")
    else:
        lines.append("No convention findings detected by this first-pass script.")
        lines.append("")

    lines.append("## Concept-Family Matrices")
    lines.append("")
    lines.append(
        "These matrices are candidate review aids. Differences are acceptable when the owning spec, design doc, or rustdoc explains them."
    )
    lines.append("")
    for concept, members in concept_groups(items).items():
        lines.append(f"### {concept}")
        lines.append("")
        lines.append("| Surface | Crate | Item | Location |")
        lines.append("| --- | --- | --- | --- |")
        for item in members:
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_escape(surface_for(item)),
                        markdown_escape(item.crate),
                        markdown_escape(f"`{item.name}`"),
                        markdown_escape(item.location(root)),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate release-freeze public API consistency candidates."
    )
    parser.add_argument("--root-dir", default=ROOT, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit 1 when convention findings are present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root_dir.resolve()
    try:
        crates = workspace_crates(root)
        items = collect_public_items(root, crates)
        findings = [
            *check_public_items(root, items),
            *check_features(root, crates),
            *check_user_docs(root),
        ]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = render_report(root, crates, items, findings)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")

    if args.fail_on_findings and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
