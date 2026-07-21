#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "gen_dep_graph.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_dep_graph", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_workspace_uses_documented_dependency_layers() -> None:
    generator = load_generator()

    dot = generator.generate_dot(ROOT)

    expected_clusters = {
        "cluster_foundation",
        "cluster_tensor_backend",
        "cluster_runtime_ad",
        "cluster_extension",
        "cluster_tutorial",
    }
    for cluster in expected_clusters:
        assert f"subgraph {cluster}" in dot

    assert "subgraph cluster_core" not in dot
    assert "style=invis" not in dot
    assert 'bgcolor="#ffffff"' in dot


def test_dependency_edges_point_from_dependency_to_consumer() -> None:
    generator = load_generator()

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["dependency", "consumer"]\n',
            encoding="utf-8",
        )
        for crate_name in ("dependency", "consumer"):
            (root / crate_name).mkdir()
        (root / "dependency" / "Cargo.toml").write_text(
            '[package]\nname = "dependency"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        (root / "consumer" / "Cargo.toml").write_text(
            '[package]\nname = "consumer"\nversion = "0.1.0"\n'
            '[dependencies]\ndependency = { path = "../dependency" }\n',
            encoding="utf-8",
        )

        dot = generator.generate_dot(root)

    assert '"dependency" -> "consumer";' in dot
    assert '"consumer" -> "dependency";' not in dot


def test_transitive_reduction_preserves_only_indispensable_edges() -> None:
    generator = load_generator()

    reduced = generator.transitive_reduction(
        ["a", "b", "c"],
        [("a", "b"), ("b", "c"), ("a", "c")],
    )

    assert reduced == [("a", "b"), ("b", "c")]


def test_render_svg_invokes_graphviz_with_dot_on_stdin() -> None:
    generator = load_generator()

    with tempfile.TemporaryDirectory() as tmp:
        fake_dot = pathlib.Path(tmp) / "dot"
        fake_dot.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "dot = sys.stdin.read()\n"
            "assert 'digraph workspace' in dot\n"
            "assert sys.argv[1:] == ['-Tsvg']\n"
            "sys.stdout.write('<svg><title>rendered</title></svg>\\n')\n",
            encoding="utf-8",
        )
        fake_dot.chmod(0o755)

        svg = generator.render_svg(
            "digraph workspace {}\n",
            dot_command=str(fake_dot),
        )

    assert "<title>rendered</title>" in svg


def test_render_svg_preserves_accessible_image_metadata() -> None:
    generator = load_generator()

    with tempfile.TemporaryDirectory() as tmp:
        fake_dot = pathlib.Path(tmp) / "dot"
        fake_dot.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "sys.stdin.read()\n"
            "sys.stdout.write('<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\\n')\n",
            encoding="utf-8",
        )
        fake_dot.chmod(0o755)

        svg = generator.render_svg("digraph workspace {}\n", dot_command=str(fake_dot))

    assert 'role="img"' in svg
    assert (
        'aria-labelledby="dependency-footprint-title dependency-footprint-desc"'
        in svg
    )
    assert '<title id="dependency-footprint-title">' in svg
    assert '<desc id="dependency-footprint-desc">' in svg


def test_checked_in_svg_matches_current_dependency_graph() -> None:
    generator = load_generator()
    svg_path = ROOT / "docs" / "assets" / "dependency-footprint.svg"

    errors = generator.validate_svg(
        generator.generate_dot(ROOT),
        svg_path.read_text(encoding="utf-8"),
    )

    assert errors == []


def test_svg_check_requires_accessible_metadata_references() -> None:
    generator = load_generator()
    svg_path = ROOT / "docs" / "assets" / "dependency-footprint.svg"
    svg = svg_path.read_text(encoding="utf-8").replace(
        ' aria-labelledby="dependency-footprint-title dependency-footprint-desc"',
        "",
        1,
    )

    errors = generator.validate_svg(generator.generate_dot(ROOT), svg)

    assert 'SVG root has invalid aria-labelledby references' in errors


def test_cli_writes_svg_through_graphviz() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        fake_dot = tmp_path / "dot"
        output = tmp_path / "dependency.svg"
        fake_dot.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "assert 'digraph workspace' in sys.stdin.read()\n"
            "assert sys.argv[1:] == ['-Tsvg']\n"
            "sys.stdout.write('<svg><title>rendered</title></svg>\\n')\n",
            encoding="utf-8",
        )
        fake_dot.chmod(0o755)

        result = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--format",
                "svg",
                "--output",
                str(output),
                "--dot-command",
                str(fake_dot),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "<title>rendered</title>" in output.read_text(encoding="utf-8")


def test_docs_build_uses_the_canonical_svg_rendering_command() -> None:
    build_script = (ROOT / "scripts" / "build_docs_site.sh").read_text(
        encoding="utf-8"
    )

    assert 'gen_dep_graph.py" --root-dir "$ROOT_DIR" --format svg' in build_script
    assert '--output "$API_DIR/dep_graph.svg"' in build_script
    assert "| dot -Tsvg" not in build_script


def test_docs_publish_the_regeneration_and_drift_check_commands() -> None:
    docs_index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert (
        "python3 scripts/gen_dep_graph.py --format svg "
        "--output docs/assets/dependency-footprint.svg"
    ) in docs_index
    assert (
        "python3 scripts/gen_dep_graph.py --check-svg "
        "docs/assets/dependency-footprint.svg"
    ) in docs_index


if __name__ == "__main__":
    test_current_workspace_uses_documented_dependency_layers()
    test_dependency_edges_point_from_dependency_to_consumer()
    test_transitive_reduction_preserves_only_indispensable_edges()
    test_render_svg_invokes_graphviz_with_dot_on_stdin()
    test_render_svg_preserves_accessible_image_metadata()
    test_svg_check_requires_accessible_metadata_references()
    test_cli_writes_svg_through_graphviz()
    test_docs_build_uses_the_canonical_svg_rendering_command()
    test_docs_publish_the_regeneration_and_drift_check_commands()
    test_checked_in_svg_matches_current_dependency_graph()
