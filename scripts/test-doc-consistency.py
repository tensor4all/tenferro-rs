#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_architecture_svg_lists_cpu_crate_and_background() -> None:
    svg_path = ROOT / "docs/assets/tenferro-architecture.svg"
    text = svg_path.read_text(encoding="utf-8")

    ET.parse(svg_path)
    assert '<rect width="100%" height="100%" fill="#ffffff"/>' in text
    assert "tenferro-cpu" in text
    assert "CPU backend" in text
    assert text.index("tenferro-cpu") < text.index("faer | BLAS/LAPACK")

    tensor_section = text[text.index("tenferro-tensor") : text.index("tenferro-cpu")]
    assert "faer | BLAS/LAPACK" not in tensor_section

    assert '<line x1="460" y1="378" x2="460" y2="400" class="dep"/>' in text


def test_agents_layer_diagram_lists_cpu_crate() -> None:
    text = read("AGENTS.md")

    assert "tenferro-cpu" in text
    assert "tenferro-tensor   - Dense runtime tensors, backend traits, CPU backend" not in text
    assert "tenferro-tensor   - Dense runtime tensors, backend traits" in text


def test_docs_ci_runs_docs_script_tests() -> None:
    text = read(".github/workflows/ci.yml")

    assert "python3 scripts/test-check-docs-site.py" in text
    assert "python3 scripts/test-doc-consistency.py" in text
    assert "python3 scripts/check-guide-dependency-snippets.py" in text


def test_documentation_policy_matches_rendered_internals() -> None:
    text = read("REPOSITORY_RULES.md")
    normalized = " ".join(text.split())

    assert "**Online docs** are primarily user-facing" in normalized
    assert "**Internals section**" in normalized
    assert "architecture, specification, and active design notes" in normalized


def main() -> int:
    for test in [
        test_architecture_svg_lists_cpu_crate_and_background,
        test_agents_layer_diagram_lists_cpu_crate,
        test_docs_ci_runs_docs_script_tests,
        test_documentation_policy_matches_rendered_internals,
    ]:
        test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
