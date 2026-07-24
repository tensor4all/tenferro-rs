#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def read_toml(relative: str) -> dict:
    with (ROOT / relative).open("rb") as file:
        return tomllib.load(file)


def rust_function(source: str, name: str) -> tuple[str, str]:
    definitions = list(
        re.finditer(
            rf"(?m)^[ \t]*(?:(?:pub(?:\([^)\n]*\))?|async|const|unsafe)[ \t]+)*"
            rf"fn[ \t]+{re.escape(name)}[ \t]*\(",
            source,
        )
    )
    assert len(definitions) == 1
    start = definitions[0].start()
    parameters_start = definitions[0].end() - 1
    depth = 0
    parameters_end = None
    for index in range(parameters_start, len(source)):
        character = source[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                parameters_end = index
                break
    assert parameters_end is not None
    body_start = source.index("{", parameters_end)
    return (
        source[parameters_start + 1 : parameters_end],
        source[start:body_start],
    )


def dependency_package_name(
    alias: str,
    specification: object,
    workspace_dependencies: dict,
) -> str:
    if not isinstance(specification, dict):
        return alias
    package = specification.get("package")
    if isinstance(package, str):
        return package
    if specification.get("workspace") is True:
        workspace_specification = workspace_dependencies.get(alias)
        if isinstance(workspace_specification, dict):
            workspace_package = workspace_specification.get("package")
            if isinstance(workspace_package, str):
                return workspace_package
    return alias


def normal_and_build_dependency_tables(manifest: dict):
    for table_name in ("dependencies", "build-dependencies"):
        yield manifest.get(table_name, {})
    for target in manifest.get("target", {}).values():
        if isinstance(target, dict):
            for table_name in ("dependencies", "build-dependencies"):
                yield target.get(table_name, {})


def load_operation_categories_checker():
    path = ROOT / "scripts/check-operation-categories.py"
    spec = importlib.util.spec_from_file_location("check_operation_categories", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_api_consistency_checker():
    path = ROOT / "scripts/check-api-consistency.py"
    spec = importlib.util.spec_from_file_location("check_api_consistency", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_architecture_svg_lists_cpu_crate_xla_boundary_and_background() -> None:
    svg_path = ROOT / "docs/assets/tenferro-architecture.svg"
    text = svg_path.read_text(encoding="utf-8")

    ET.parse(svg_path)
    assert '<rect width="100%" height="100%" fill="#ffffff"/>' in text
    assert "tenferro-cpu" in text
    assert "tenferro-xla" in text
    assert "GraphProgram → StableHLO" in text
    assert "PJRT plugin" in text
    assert "tenferro-xla bridge" in text
    assert "CPU backend" in text
    assert text.index("tenferro-cpu") < text.index("faer | BLAS/LAPACK")

    tensor_section = text[text.index("tenferro-tensor") : text.index("tenferro-cpu")]
    assert "faer | BLAS/LAPACK" not in tensor_section

    assert '<line x1="460" y1="390" x2="460" y2="412" class="dep"/>' in text


def test_agents_layer_diagram_lists_cpu_crate() -> None:
    text = read("AGENTS.md")

    assert "tenferro-cpu" in text
    assert "tenferro-tensor   - Dense runtime tensors, backend traits, CPU backend" not in text
    assert "tenferro-tensor   - Dense runtime tensors, backend traits" in text


def test_docs_ci_runs_docs_script_tests() -> None:
    text = read(".github/workflows/ci.yml")
    profiles = read("scripts/ci/run_profile.py")
    build_docs_site = read("scripts/build_docs_site.sh")

    assert "python3 scripts/ci/run_profile.py docs" in text
    assert "python3 scripts/test-check-docs-site.py" in profiles
    assert "python3 scripts/test-doc-consistency.py" in profiles
    assert "python3 scripts/test-repository-rules-review.py" in profiles
    assert "python3 scripts/check-guide-dependency-snippets.py" in profiles
    assert "python3 scripts/check-operation-categories.py --fail-on-findings" in profiles
    assert (
        "python3 \"$ROOT_DIR/scripts/check-operation-categories.py\" --fail-on-findings --include-rendered"
        in build_docs_site
    )


def test_ci_enforces_public_error_docs_and_extension_clippy() -> None:
    text = read(".github/workflows/ci.yml")

    assert "scripts/check-public-error-docs.py" in text
    assert "-D clippy::missing_errors_doc" in text
    assert "-D clippy::missing_panics_doc" in text
    assert "cargo clippy --manifest-path ext/tropical/Cargo.toml" in text
    assert "cargo clippy --manifest-path ext/sparse/Cargo.toml" in text


def test_review_bot_workflow_exists() -> None:
    text = read(".github/workflows/review_bot.yml")

    assert "name: review bot" in text
    assert "repository rules review" in text
    assert "DEEPSEEK_API_KEY" in text
    assert "rules-review:waive" in text


def test_repo_settings_requires_repository_rules_review() -> None:
    text = read("ai/repo-settings.json")

    assert '"repository rules review"' in text


def test_gpu_ci_waits_for_review_bot_gate_before_cuda_work() -> None:
    text = read(".github/workflows/CI_gpu.yml")

    assert '"repository rules review"' in text
    assert "repository rules review (LLM)" not in text
    assert text.index("pre-gpu-gate:") < text.index("cuda-archive:")
    assert text.index("pre-gpu-gate:") < text.index("runs-on: ubuntu-gpu")
    # The expensive GPU runner (cuda-run) stays gated behind the review +
    # non-GPU checks: it needs both pre-gpu-gate and the archive.
    assert "needs: [pre-gpu-gate, cuda-archive]" in text
    # The cheap non-GPU archive build must NOT be gated on pre-gpu-gate: it
    # compiles in parallel with the non-GPU CI so the GPU stage can start the
    # moment the gate clears. Guard against re-adding a bare `needs:
    # [pre-gpu-gate]`, which would serialize the archive back onto the critical
    # path (only cuda-run carries the gate).
    assert "needs: [pre-gpu-gate]" not in text


def test_pre_pr_checklist_requires_local_llm_review() -> None:
    text = read("AGENTS.md")
    template = read(".github/pull_request_template.md")

    assert "python3 scripts/repository-rules-review.py" in text
    assert "--base origin/main" in text
    assert "--head HEAD" in text
    assert "--worktree" in text
    assert "local repository-rules LLM review" in template


def test_operation_surface_checker_requires_inherent_tensor_methods() -> None:
    checker = load_operation_categories_checker()
    source = textwrap.dedent(
        """
        pub fn add() {}

        impl std::ops::Add for &EagerTensor {
            pub fn add(self, rhs: &EagerTensor) {}
        }

        impl EagerTensor {
            pub fn mul(&self, rhs: &Self) {}
            pub fn concatenate(inputs: &[&Self], axis: usize) {}
        }
        """
    )

    assert checker.inherent_public_functions_from_text(source, "EagerTensor") == {
        "mul",
        "concatenate",
    }


def test_operation_surface_checker_rejects_tensor_module_exports() -> None:
    checker = load_operation_categories_checker()

    assert checker.forbidden_tensor_module_export_offsets(
        "pub mod traced_tensor;", "traced_tensor"
    )
    assert checker.forbidden_tensor_module_export_offsets(
        "pub use crate::traced as traced_tensor;", "traced_tensor"
    )
    assert checker.forbidden_tensor_module_export_offsets(
        "pub use crate::traced as\ntraced_tensor;", "traced_tensor"
    )
    assert not checker.forbidden_tensor_module_export_offsets(
        "pub mod traced;\npub mod tensor;", "traced_tensor"
    )
    assert checker.forbidden_public_module_export_offsets("pub mod tensor;", "tensor")
    assert checker.forbidden_public_module_export_offsets(
        "pub mod typed_tensor;", "typed_tensor"
    )
    assert not checker.forbidden_public_module_export_offsets(
        "mod tensor;\npub use tensor::TensorOpsExt;", "tensor"
    )


def test_operation_surface_checker_skips_rendered_search_index_metadata() -> None:
    checker = load_operation_categories_checker()

    assert checker.is_rendered_search_index(
        pathlib.Path("target/doc/search.index/path/abc123.js")
    )
    assert checker.is_rendered_search_index(
        pathlib.Path("target/docs-site/api/search.index/path/abc123.js")
    )
    assert not checker.is_rendered_search_index(
        pathlib.Path("target/docs-site/guides/tensor-operations.html")
    )


def test_api_consistency_checker_rejects_public_try_compatibility_escape() -> None:
    checker = load_api_consistency_checker()
    allowed = checker.PublicItem(
        crate="tenferro-runtime",
        crate_path=pathlib.Path("crates/tenferro-runtime"),
        file=ROOT / "crates/tenferro-runtime/src/traced.rs",
        line=777,
        kind="fn",
        name="try_concrete_shape",
        signature="pub fn try_concrete_shape(&self) -> Option<Vec<usize>> {",
    )
    disallowed = checker.PublicItem(
        crate="tenferro-ad",
        crate_path=pathlib.Path("crates/tenferro-ad"),
        file=ROOT / "crates/tenferro-ad/src/eager.rs",
        line=1,
        kind="fn",
        name="try_materialized",
        signature="pub fn try_materialized(&self) -> Result<Tensor> {",
    )

    findings = checker.check_public_items(ROOT, [allowed, disallowed])

    assert [finding.category for finding in findings] == ["public_try_prefix"]
    assert "try_materialized" in findings[0].evidence


def test_removed_tensor_module_paths_do_not_compile() -> None:
    with tempfile.TemporaryDirectory(prefix="tenferro-tensor-surface-") as tmp:
        tmp_path = pathlib.Path(tmp)
        (tmp_path / "src").mkdir()
        (tmp_path / "Cargo.toml").write_text(
            textwrap.dedent(
                f"""
                [package]
                name = "core-ad-surface-negative-check"
                version = "0.0.0"
                edition = "2021"

                [dependencies]
                tenferro-runtime = {{ path = "{ROOT / 'crates/tenferro-runtime'}" }}
                tenferro-ad = {{ path = "{ROOT / 'crates/tenferro-ad'}" }}
                tenferro-einsum = {{ path = "{ROOT / 'crates/tenferro-einsum'}", features = ["autodiff"] }}
                tenferro-linalg = {{ path = "{ROOT / 'crates/tenferro-linalg'}", features = ["autodiff"] }}
                tenferro-fft = {{ path = "{ROOT / 'crates/tenferro-fft'}", features = ["autodiff"] }}
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / "src/main.rs").write_text(
            textwrap.dedent(
                """
                use tenferro_ad::eager_tensor as ad_eager_tensor;
                use tenferro_runtime::traced_tensor as runtime_traced_tensor;
                use tenferro_runtime::tensor as runtime_tensor;
                use tenferro_runtime::typed_tensor as runtime_typed_tensor;
                use tenferro_einsum::eager_tensor as einsum_eager_tensor;
                use tenferro_einsum::traced_tensor as einsum_traced_tensor;
                use tenferro_linalg::eager_tensor as linalg_eager_tensor;
                use tenferro_linalg::traced_tensor as linalg_traced_tensor;
                use tenferro_fft::traced_tensor as fft_traced_tensor;

                fn main() {
                    let _ = (
                        ad_eager_tensor::add,
                        runtime_traced_tensor::add,
                        runtime_tensor::matmul,
                        runtime_typed_tensor::add,
                        einsum_eager_tensor::einsum,
                        einsum_traced_tensor::einsum,
                        linalg_eager_tensor::svd,
                        linalg_traced_tensor::svd,
                        fft_traced_tensor::fft,
                    );
                }
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            ["cargo", "check", "--manifest-path", str(tmp_path / "Cargo.toml"), "--quiet"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )

    assert result.returncode != 0
    assert "eager_tensor" in result.stderr
    assert "traced_tensor" in result.stderr


def test_documentation_policy_matches_rendered_internals() -> None:
    text = read("REPOSITORY_RULES.md")
    normalized = " ".join(text.split())

    assert "**Online docs** are primarily user-facing" in normalized
    assert "**Internals section**" in normalized
    assert "architecture, specification, and active design notes" in normalized


def test_traced_remainder_docs_distinguish_complex_unsupported_from_float_zero() -> None:
    text = read("crates/tenferro-runtime/src/traced.rs")
    start = text.index("    /// Elementwise remainder")
    end = text.index("    pub fn rem", start)
    docs = " ".join(
        line.removeprefix("    ///").strip() for line in text[start:end].splitlines()
    )

    assert (
        "[`Error::Unsupported`] at [`ErrorPhase::GraphBuild`] when either operand "
        "has a complex dtype"
    ) in docs
    assert "floating-point zero divisors follow their numeric semantics" in docs
    assert "floating-point and complex zero divisors" not in docs


def test_phase4_runtime_ownership_and_dependency_direction_are_canonical() -> None:
    rules = read("REPOSITORY_RULES.md")
    crates = read("docs/architecture/tenferro-crates.md")
    design = read("docs/design/execution-engine-provider-architecture.md")
    staging = read("crates/tenferro-runtime/src/compiler/semantic_staging.rs")
    workspace = read_toml("Cargo.toml")
    runtime_manifest = read_toml("crates/tenferro-runtime/Cargo.toml")
    workspace_dependencies = workspace["workspace"]["dependencies"]
    parameters, signature = rust_function(
        staging,
        "lower_semantic_to_exec_staging",
    )

    assert "GraphCompilerEinsumExt" not in rules
    assert "SemanticProgram -> SemanticProgram" in " ".join(rules.split())
    assert re.search(r"(?m)^tenferro-cpu\s+->\s+tenferro-runtime$", crates)
    normalized_crates = " ".join(crates.split())
    normalized_design = " ".join(design.split())
    assert (
        "CPU-to-runtime line records the implemented Phase 4 registration adapter"
        in normalized_crates
    )
    assert "Phase 4 runtime preparation substrate is implemented" in normalized_crates
    assert "opposite direction remains dev/test-only" in normalized_crates
    assert "lower_semantic_to_exec_staging" in design
    assert "sole private forward adapter" in design
    assert "Phase 4 now provides the immutable runtime snapshot and preparation substrate" in normalized_design
    assert re.search(r"\bprogram\s*:\s*&SemanticProgram\b", parameters)
    assert re.search(r"\boptions\s*:\s*CompilerOptions\b", parameters)
    assert not re.search(r"\b\w*[Bb]inding\w*\b", signature)

    dependency_names = {
        dependency_package_name(alias, specification, workspace_dependencies)
        for table in normal_and_build_dependency_tables(runtime_manifest)
        for alias, specification in table.items()
    }
    assert "tenferro-cpu" not in dependency_names

    dependency_lines = [
        line
        for line in crates.splitlines()
        if re.match(r"^tenferro-[a-z0-9-]+\s+->", line)
    ]
    assert len({line.index("->") for line in dependency_lines}) == 1


def main() -> int:
    for test in [
        test_architecture_svg_lists_cpu_crate_xla_boundary_and_background,
        test_agents_layer_diagram_lists_cpu_crate,
        test_docs_ci_runs_docs_script_tests,
        test_ci_enforces_public_error_docs_and_extension_clippy,
        test_review_bot_workflow_exists,
        test_repo_settings_requires_repository_rules_review,
        test_gpu_ci_waits_for_review_bot_gate_before_cuda_work,
        test_pre_pr_checklist_requires_local_llm_review,
        test_operation_surface_checker_requires_inherent_tensor_methods,
        test_operation_surface_checker_rejects_tensor_module_exports,
        test_operation_surface_checker_skips_rendered_search_index_metadata,
        test_api_consistency_checker_rejects_public_try_compatibility_escape,
        test_removed_tensor_module_paths_do_not_compile,
        test_documentation_policy_matches_rendered_internals,
        test_traced_remainder_docs_distinguish_complex_unsupported_from_float_zero,
        test_phase4_runtime_ownership_and_dependency_direction_are_canonical,
    ]:
        test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
