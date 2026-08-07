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
            "- [Skill](https://github.com/tensor4all/tenferro-rs/blob/main/.agents/skills/tenferro-compute/SKILL.md): Skill.\n",
        )
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "target does not exist" in result.stderr


def test_llms_duplicate_url_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)
        write(
            fake_root / "docs/llms.txt",
            "- [One](https://tensor4all.org/tenferro-rs/spec/extension-op.html): One.\n"
            "- [Two](https://tensor4all.org/tenferro-rs/spec/extension-op.html): Two.\n"
            "- [Skill](https://github.com/tensor4all/tenferro-rs/blob/main/.agents/skills/tenferro-compute/SKILL.md): Skill.\n",
        )
        result = run_checker(fake_root)
        assert result.returncode != 0
        assert "repeats URL" in result.stderr


if __name__ == "__main__":
    test_rendered_internal_link_outside_render_set_fails()
    test_llms_missing_target_fails()
    test_llms_duplicate_url_fails()
