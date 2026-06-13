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
          render:
            - index.md
            - spec/**/*.md
        """,
    )
    write(root / "docs/index.md", "# Home\n")
    write(root / "docs/spec/extension-op.md", "[dynamic shapes](../design/dynamic-symbolic-shapes.md)\n")
    write(
        root / "target/docs-site/spec/extension-op.html",
        '<a href="../design/dynamic-symbolic-shapes.html">dynamic shapes</a>\n',
    )


def test_rendered_internal_link_outside_render_set_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = pathlib.Path(tmp)
        make_minimal_docs_root(fake_root)

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check-docs-site.py"),
                "--root-dir",
                str(fake_root),
                "--quiet",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        assert result.returncode != 0, "rendered links outside the Quarto render set should fail"
        assert "outside the rendered docs set" in result.stderr


if __name__ == "__main__":
    test_rendered_internal_link_outside_render_set_fails()
