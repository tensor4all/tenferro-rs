#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]

PUBLIC_FN_RE = re.compile(r"\bpub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\b")

EAGER_REQUIRED = {
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "sign",
    "conj",
    "pow",
    "maximum",
    "minimum",
    "compare",
    "select",
    "where_select",
    "clamp",
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
    "sqrt",
    "rsqrt",
    "expm1",
    "log1p",
    "reduce_sum",
    "reduce_prod",
    "reduce_max",
    "reduce_min",
    "reshape",
    "transpose",
    "broadcast_in_dim",
    "convert",
    "cast",
    "slice",
    "pad",
    "reverse",
    "gather",
    "scatter",
    "dynamic_slice",
    "concatenate",
    "stack",
    "extract_diag",
    "embed_diag",
    "tril",
    "triu",
    "dot_general",
    "matmul",
}

TRACED_REQUIRED = EAGER_REQUIRED | {
    "reshape_sym",
    "broadcast_in_dim_sym",
    "shape_of",
    "dynamic_truncate",
    "pad_to_match",
}

EAGER_SURFACE_FILES = [
    pathlib.Path("crates/tenferro-ad/src/eager_ops.rs"),
    pathlib.Path("crates/tenferro-ad/src/eager_ops_elementwise.rs"),
    pathlib.Path("crates/tenferro-ad/src/shape_packing.rs"),
]

TRACED_SURFACE_FILES = [
    pathlib.Path("crates/tenferro-runtime/src/traced.rs"),
    pathlib.Path("crates/tenferro-runtime/src/shape_packing.rs"),
]

REMOVED_MODULE_FILES = [
    pathlib.Path("crates/tenferro-ad/src/eager_tensor.rs"),
    pathlib.Path("crates/tenferro-runtime/src/traced_tensor.rs"),
    pathlib.Path("crates/tenferro-einsum/src/eager_tensor.rs"),
    pathlib.Path("crates/tenferro-einsum/src/traced_tensor.rs"),
    pathlib.Path("crates/tenferro-linalg/src/eager_tensor.rs"),
    pathlib.Path("crates/tenferro-linalg/src/traced_tensor.rs"),
]

REMOVED_MODULE_EXPORTS = [
    (pathlib.Path("crates/tenferro-runtime/src"), "traced_tensor"),
    (pathlib.Path("crates/tenferro-ad/src"), "eager_tensor"),
    (pathlib.Path("crates/tenferro-einsum/src"), "traced_tensor"),
    (pathlib.Path("crates/tenferro-einsum/src"), "eager_tensor"),
    (pathlib.Path("crates/tenferro-linalg/src"), "traced_tensor"),
    (pathlib.Path("crates/tenferro-linalg/src"), "eager_tensor"),
    (pathlib.Path("crates/tenferro-fft/src"), "traced_tensor"),
]

LIVE_DOC_ROOTS = [
    pathlib.Path("README.md"),
    pathlib.Path("AGENTS.md"),
    pathlib.Path("REPOSITORY_RULES.md"),
    pathlib.Path("docs/index.md"),
    pathlib.Path("docs/getting-started"),
    pathlib.Path("docs/tutorials"),
    pathlib.Path("docs/guides"),
    pathlib.Path("docs/performance"),
    pathlib.Path("docs/api"),
    pathlib.Path("docs/internals"),
    pathlib.Path("docs/architecture"),
    pathlib.Path("docs/spec"),
    pathlib.Path("docs/design"),
    pathlib.Path("docs/oracle"),
    pathlib.Path("docs/reference"),
]

RUSTDOC_SOURCE_ROOTS = [
    pathlib.Path("crates"),
    pathlib.Path("docs/tutorial-code/src"),
]

RENDERED_ROOTS = [
    pathlib.Path("target/doc"),
    pathlib.Path("target/docs-site"),
]

RENDERED_SUFFIXES = {".html", ".js", ".json", ".txt"}

FORBIDDEN_TENSOR_MODULE_PATTERNS = [
    "tenferro_ad::eager_tensor",
    "tenferro_runtime::traced_tensor",
    "tenferro_einsum::eager_tensor",
    "tenferro_einsum::traced_tensor",
    "tenferro_linalg::eager_tensor",
    "tenferro_linalg::traced_tensor",
    "tenferro_fft::traced_tensor",
    "use tenferro_ad::{eager_tensor",
    "use tenferro_runtime::{traced_tensor",
    "use tenferro_einsum::{eager_tensor",
    "use tenferro_einsum::{traced_tensor",
    "use tenferro_linalg::{eager_tensor",
    "use tenferro_linalg::{traced_tensor",
    "use tenferro_fft::{traced_tensor",
]


@dataclasses.dataclass(frozen=True)
class Finding:
    category: str
    location: str
    message: str


def read(path: pathlib.Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def matching_brace(text: str, open_brace: int) -> int | None:
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def inherent_public_functions_from_text(text: str, type_name: str) -> set[str]:
    names: set[str] = set()
    impl_re = re.compile(rf"\bimpl\s+{re.escape(type_name)}\s*\{{")
    for match in impl_re.finditer(text):
        open_brace = text.find("{", match.start())
        if open_brace == -1:
            continue
        close_brace = matching_brace(text, open_brace)
        if close_brace is None:
            continue
        names.update(PUBLIC_FN_RE.findall(text[open_brace + 1 : close_brace]))
    return names


def inherent_public_functions(paths: list[pathlib.Path], type_name: str) -> set[str]:
    names: set[str] = set()
    for path in paths:
        names.update(inherent_public_functions_from_text(read(path), type_name))
    return names


def markdown_files(path: pathlib.Path) -> list[pathlib.Path]:
    absolute = ROOT / path
    if absolute.is_file():
        return [path] if path.suffix == ".md" else []
    if not absolute.exists():
        return []
    return sorted(file.relative_to(ROOT) for file in absolute.rglob("*.md"))


def rust_files(path: pathlib.Path) -> list[pathlib.Path]:
    absolute = ROOT / path
    if absolute.is_file():
        return [path] if path.suffix == ".rs" else []
    if not absolute.exists():
        return []
    return sorted(file.relative_to(ROOT) for file in absolute.rglob("*.rs"))


def rendered_files(path: pathlib.Path) -> list[pathlib.Path]:
    absolute = ROOT / path
    if absolute.is_file():
        return [path] if path.suffix in RENDERED_SUFFIXES else []
    if not absolute.exists():
        return []
    return sorted(
        file.relative_to(ROOT)
        for file in absolute.rglob("*")
        if file.is_file() and file.suffix in RENDERED_SUFFIXES
    )


def forbidden_tensor_module_export_offsets(text: str, module_name: str) -> list[int]:
    export_re = re.compile(
        rf"\bpub\s+mod\s+{re.escape(module_name)}\b"
        rf"|\bpub\s+use\b(?:(?!;).)*?\bas\s+{re.escape(module_name)}\b"
        rf"|\bpub\s+use\b(?:(?!;).)*?\b{re.escape(module_name)}\b",
        re.DOTALL,
    )
    return [match.start() for match in export_re.finditer(text)]


def forbidden_tensor_module_export_findings(
    crate_src: pathlib.Path, module_name: str
) -> list[Finding]:
    findings: list[Finding] = []
    for path in rust_files(crate_src):
        text = read(path)
        for offset in forbidden_tensor_module_export_offsets(text, module_name):
            findings.append(
                Finding(
                    "removed_tensor_module",
                    f"{path}:{line_number(text, offset)}",
                    f"`{module_name}` tensor operation module is publicly exported",
                )
            )
    return findings


def check_surface(
    label: str,
    paths: list[pathlib.Path],
    required: set[str],
) -> list[Finding]:
    names = inherent_public_functions(paths, label)
    findings = []
    for missing in sorted(required - names):
        findings.append(
            Finding(
                "missing_method",
                label,
                f"{label} is missing required public method/associated function `{missing}`",
            )
        )
    return findings


def check_removed_tensor_modules() -> list[Finding]:
    findings = []
    for path in REMOVED_MODULE_FILES:
        if (ROOT / path).exists():
            findings.append(
                Finding(
                    "removed_tensor_module",
                    str(path),
                    "tensor operations must not be exposed through *_tensor module free functions",
                )
            )

    for crate_src, module_name in REMOVED_MODULE_EXPORTS:
        findings.extend(forbidden_tensor_module_export_findings(crate_src, module_name))
    return findings


def check_forbidden_live_docs(include_rendered: bool) -> list[Finding]:
    findings = []
    files: list[pathlib.Path] = []
    for root in LIVE_DOC_ROOTS:
        files.extend(markdown_files(root))
    for root in RUSTDOC_SOURCE_ROOTS:
        files.extend(rust_files(root))
    if include_rendered:
        for root in RENDERED_ROOTS:
            files.extend(rendered_files(root))

    for path in sorted(set(files)):
        text = read(path)
        for pattern in FORBIDDEN_TENSOR_MODULE_PATTERNS:
            offset = text.find(pattern)
            if offset != -1:
                findings.append(
                    Finding(
                        "forbidden_tensor_module_reference",
                        f"{path}:{line_number(text, offset)}",
                        f"live docs/source reference removed tensor operation module path `{pattern}`",
                    )
                )
    return findings


def collect_findings(*, include_rendered: bool) -> list[Finding]:
    findings = []
    findings.extend(check_removed_tensor_modules())
    findings.extend(check_surface("EagerTensor", EAGER_SURFACE_FILES, EAGER_REQUIRED))
    findings.extend(check_surface("TracedTensor", TRACED_SURFACE_FILES, TRACED_REQUIRED))
    findings.extend(check_forbidden_live_docs(include_rendered))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the operation-category public surface contract."
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Return a non-zero exit code if findings are present.",
    )
    parser.add_argument(
        "--include-rendered",
        action="store_true",
        help="Also scan rendered rustdoc and Quarto output under target/ when present.",
    )
    args = parser.parse_args()

    findings = collect_findings(include_rendered=args.include_rendered)
    if not findings:
        print("operation category checks passed")
        return 0

    for finding in findings:
        print(
            f"{finding.category}: {finding.location}: {finding.message}",
            file=sys.stderr,
        )
    return 1 if args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
