#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import textwrap


ROOT = pathlib.Path(__file__).resolve().parents[1]


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def make_minimal_docs_root(root: pathlib.Path) -> None:
    write(
        root / "README.md",
        """
        # demo

        Read [llms.txt](docs/llms.txt) and the
        [skill](.agents/skills/tenferro-compute/SKILL.md).
        """,
    )
    write(
        root / "Cargo.toml",
        """
        [workspace]
        members = ["crates/demo"]
        """,
    )
    write(
        root / "crates/demo/Cargo.toml",
        """
        [package]
        name = "demo-crate"
        version = "0.1.0"
        edition = "2021"

        [lib]
        name = "demo_crate"
        """,
    )
    write(root / "crates/demo/src/lib.rs", "pub fn demo() {}\n")
    write(root / "target/doc/demo_crate/index.html", "<html></html>\n")
    write(root / "target/docs-site/api/index.html", '<a href="../demo_crate/index.html">demo</a>\n')
    write(root / "target/docs-site/llms.txt", "# llms\n")
    write(
        root / "scripts/check-doc-snippets.py",
        """
        #!/usr/bin/env python3
        raise SystemExit(0)
        """,
    )
    write(
        root / "scripts/check-guide-dependency-snippets.py",
        """
        #!/usr/bin/env python3
        raise SystemExit(0)
        """,
    )
    write(
        root / "docs/_quarto.yml",
        """
        project:
          type: website
          resources:
            - llms.txt
          render:
            - index.md
            - spec/**/*.md
        """,
    )
    write(root / "docs/index.md", "# Home\n")
    write(root / ".agents/skills/tenferro-compute/SKILL.md", "# Skill\n")
    write(
        root / "docs/llms.txt",
        """
        - [README](https://github.com/tensor4all/tenferro-rs/blob/main/README.md): Router.
        - [Extension](https://tensor4all.org/tenferro-rs/spec/extension-op.html): The extension specification.
        - [Skill](https://github.com/tensor4all/tenferro-rs/blob/main/.agents/skills/tenferro-compute/SKILL.md): Downstream usage guidance.
        """,
    )
    write(root / "docs/spec/extension-op.md", "[dynamic shapes](../design/dynamic-symbolic-shapes.md)\n")
    write(
        root / "target/docs-site/spec/extension-op.html",
        '<a href="../design/dynamic-symbolic-shapes.html">dynamic shapes</a>\n',
    )


def run_checker(root: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check-docs-site.py"),
            "--root-dir",
            str(root),
            "--quiet",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_rendered_internal_link_outside_render_set_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        result = run_checker(fake_root)
        assert result.returncode != 0, "rendered links outside the Quarto render set should fail"
        assert "outside the rendered docs set" in result.stderr


def test_llms_missing_target_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        write(
            fake_root / "docs/llms.txt",
            "- [Missing](https://tensor4all.org/tenferro-rs/spec/missing.html): Not present.\n"
            "- [README](https://github.com/tensor4all/tenferro-rs/blob/main/README.md): Router.\n"
            "- [Skill](https://github.com/tensor4all/tenferro-rs/blob/main/.agents/skills/tenferro-compute/SKILL.md): Skill.\n",
        )
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "target does not exist" in result.stderr


def test_built_llms_missing_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        (fake_root / "target/docs-site/llms.txt").unlink()
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "built docs site is missing root llms.txt" in result.stderr


def test_llms_duplicate_url_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        write(
            fake_root / "docs/llms.txt",
            "- [One](https://tensor4all.org/tenferro-rs/spec/extension-op.html): One.\n"
            "- [Two](https://tensor4all.org/tenferro-rs/spec/extension-op.html): Two.\n"
            "- [README](https://github.com/tensor4all/tenferro-rs/blob/main/README.md): Router.\n"
            "- [Skill](https://github.com/tensor4all/tenferro-rs/blob/main/.agents/skills/tenferro-compute/SKILL.md): Skill.\n",
        )
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "repeats URL" in result.stderr


def test_readme_missing_llms_link_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        write(
            fake_root / "README.md",
            "# demo\n\nRead the [skill](.agents/skills/tenferro-compute/SKILL.md).\n",
        )
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "README.md must link docs/llms.txt" in result.stderr


def test_readme_plain_text_mention_is_not_a_link() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        # `docs/llms.txt` appears only in prose, not as a link target.
        write(
            fake_root / "README.md",
            "# demo\n\nSee docs/llms.txt for the index and the\n"
            "[skill](.agents/skills/tenferro-compute/SKILL.md).\n",
        )
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "README.md must link docs/llms.txt" in result.stderr


def test_readme_missing_skill_link_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        # The skill path appears only in prose, not as a link target.
        write(
            fake_root / "README.md",
            "# demo\n\nRead the [llms index](docs/llms.txt); the "
            ".agents/skills/tenferro-compute/SKILL.md is also bundled.\n",
        )
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "README.md must link .agents/skills/tenferro-compute/SKILL.md" in result.stderr


def test_llms_missing_readme_link_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        write(
            fake_root / "docs/llms.txt",
            "- [Skill](https://github.com/tensor4all/tenferro-rs/blob/main/.agents/skills/tenferro-compute/SKILL.md): Skill.\n",
        )
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "llms.txt must link back to the README" in result.stderr


def make_eager_ad_fixture(root: pathlib.Path) -> None:
    """Create the sources read by check_eager_functional_ad_docs in the checker."""
    write(root / "docs/index.md", "# Home\n\n`EagerRuntime` functional `grad`, `vjp`, and `jvp`\n")
    write(
        root / "docs/getting-started/index.md",
        "functional eager `grad`, `vjp`, and `jvp`\n",
    )
    write(
        root / "docs/getting-started/core-concepts.md",
        "functional `grad`, `vjp`, and `jvp` transforms\n",
    )
    write(
        root / "docs/getting-started/pytorch-jax-mapping.md",
        "`EagerRuntime` functional `grad`/`vjp`/`jvp`\n",
    )
    write(root / "docs/tutorials/index.md", "functional eager AD entry point\n")
    write(
        root / "docs/spec/operation-categories.md",
        "stateful `backward()` plus functional `grad`/`vjp`/`jvp`\n",
    )
    write(
        root / "docs/guides/eager-operations.md",
        "stateful reverse-mode and functional `grad`/`vjp`/`jvp`\n",
    )
    write(
        root / "docs/assets/tenferro-architecture.svg",
        "backward · grad and vjp · jvp\n",
    )
    write(
        root / "README.md",
        "# demo\n\nBoth eager and traced modes support VJP and JVP, "
        "and HVP-style higher-order composition.\n"
        "Read [llms.txt](docs/llms.txt) and the\n"
        "[skill](.agents/skills/tenferro-compute/SKILL.md).\n",
    )


def test_skill_reference_resolves_and_built_site_copy_checked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        write(
            fake_root / ".agents/skills/tenferro-compute/references/api-cheatsheet.md",
            "# API cheatsheet\n",
        )
        write(
            fake_root / "docs/llms.txt",
            "- [README](https://github.com/tensor4all/tenferro-rs/blob/main/README.md): Router.\n"
            "- [API cheatsheet](https://tensor4all.org/tenferro-rs/skill-references/api-cheatsheet.md): Recipe.\n"
            "- [Skill](https://github.com/tensor4all/tenferro-rs/blob/main/.agents/skills/tenferro-compute/SKILL.md): Skill.\n",
        )
        # The source exists but the built-site copy is missing: must fail.
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "missing republished skill reference" in result.stderr
        # Once the build copies it, the same index passes.
        write(fake_root / "target/docs-site/skill-references/api-cheatsheet.md", "# API cheatsheet\n")
        # Resolve the intentionally-broken design link from the minimal root so
        # the rendered-link check passes and the skill-reference check is the
        # deciding gate for the positive case.
        write(fake_root / "target/docs-site/design/dynamic-symbolic-shapes.html", "# design\n")
        # The checker's eager-functional-AD docs audit reads many sources after
        # the llms/reachability checks; provide them so the run can complete.
        make_eager_ad_fixture(fake_root)
        result = run_checker(fake_root)
        assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    test_rendered_internal_link_outside_render_set_fails()
    test_llms_missing_target_fails()
    test_built_llms_missing_fails()
    test_llms_duplicate_url_fails()
    test_readme_missing_llms_link_fails()
    test_readme_plain_text_mention_is_not_a_link()
    test_readme_missing_skill_link_fails()
    test_llms_missing_readme_link_fails()
    test_skill_reference_resolves_and_built_site_copy_checked()
