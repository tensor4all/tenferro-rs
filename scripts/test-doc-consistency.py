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
    assert "tenferro-xla" in text
    assert "StableHLO/PJRT peer executor" in text
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
    assert "python3 scripts/test-repository-rules-review.py" in text
    assert "python3 scripts/check-guide-dependency-snippets.py" in text


def test_review_bot_workflow_exists() -> None:
    text = read(".github/workflows/review_bot.yml")

    assert "name: review bot" in text
    assert "repository rules review" in text
    assert "DEEPSEEK_API_KEY" in text
    assert "rules-review:waive" in text


def test_repo_settings_requires_repository_rules_review() -> None:
    text = read("ai/repo-settings.json")

    assert '"repository rules review"' in text


def test_gpu_ci_waits_for_review_bot_llm_before_cuda_work() -> None:
    text = read(".github/workflows/CI_gpu.yml")

    assert "repository rules review (LLM)" in text
    assert text.index("pre-gpu-gate:") < text.index("cuda-archive:")
    assert text.index("pre-gpu-gate:") < text.index("runs-on: ubuntu-gpu")
    assert "needs: [pre-gpu-gate]" in text
    assert "needs: [pre-gpu-gate, cuda-archive]" in text


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
        test_review_bot_workflow_exists,
        test_repo_settings_requires_repository_rules_review,
        test_gpu_ci_waits_for_review_bot_llm_before_cuda_work,
        test_documentation_policy_matches_rendered_internals,
    ]:
        test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
