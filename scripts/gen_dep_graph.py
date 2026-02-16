#!/usr/bin/env python3
"""Generate a Graphviz DOT dependency graph from workspace Cargo.toml files.

Reads the workspace root Cargo.toml to discover members, then parses each
member's Cargo.toml to find workspace-internal dependencies. Outputs a DOT
graph to stdout.

Members are classified into three groups based on their directory prefix:
- **extern/**    — general-purpose crates (no tenferro dependency)
- **extension/** — optional extensions (depend on core + extern)
- **(root)**     — core tenferro crates

Each group is rendered as a DOT subgraph cluster with a distinct color.

Usage:
    python3 scripts/gen_dep_graph.py [--root-dir DIR] [--api-prefix PREFIX]

Options:
    --root-dir DIR        Workspace root (default: script's parent directory)
    --api-prefix PREFIX   URL prefix for crate doc links (default: ".")
"""

import argparse
import sys
import tomllib
from pathlib import Path


# Classification of workspace members by directory prefix.
# Order matters: first match wins.
_CLASS_PREFIXES = [
    ("extern/", "extern"),
    ("extension/", "extension"),
]

_CLASS_STYLES = {
    "core": {
        "label": "core",
        "fillcolor": "#e8f5e9",
        "cluster_bg": "#f1f8e9",
        "cluster_border": "#a5d6a7",
    },
    "extension": {
        "label": "extension/",
        "fillcolor": "#e3f2fd",
        "cluster_bg": "#e8eaf6",
        "cluster_border": "#90caf9",
    },
    "extern": {
        "label": "extern/",
        "fillcolor": "#fff3e0",
        "cluster_bg": "#fff8e1",
        "cluster_border": "#ffcc80",
    },
}


def parse_workspace_members(root: Path) -> list[str]:
    """Return the list of workspace member directory paths."""
    cargo_toml = root / "Cargo.toml"
    with open(cargo_toml, "rb") as f:
        data = tomllib.load(f)
    return data["workspace"]["members"]


def classify_member(member_path: str) -> str:
    """Classify a member path into core/extension/extern."""
    for prefix, cls in _CLASS_PREFIXES:
        if member_path.startswith(prefix):
            return cls
    return "core"


def parse_crate_deps(
    crate_dir: Path, all_crate_names: set[str]
) -> tuple[str, list[str]]:
    """Return (crate_name, [internal_dependency_names])."""
    cargo_toml = crate_dir / "Cargo.toml"
    with open(cargo_toml, "rb") as f:
        data = tomllib.load(f)

    crate_name = data["package"]["name"]
    deps = data.get("dependencies", {})
    internal = []
    for dep_key, dep_val in deps.items():
        # dep_key is the dependency name (e.g. "tenferro-device")
        if dep_key in all_crate_names:
            internal.append(dep_key)
        elif isinstance(dep_val, dict) and dep_val.get("package", dep_key) in all_crate_names:
            internal.append(dep_val["package"])
    return crate_name, internal


def crate_to_doc_dir(crate_name: str) -> str:
    """Convert crate name to rustdoc directory name (hyphens -> underscores)."""
    return crate_name.replace("-", "_")


def generate_dot(
    root: Path,
    api_prefix: str = ".",
) -> str:
    """Generate DOT source for the workspace dependency graph."""
    members = parse_workspace_members(root)

    # First pass: read all crate names and classify them
    crate_info: list[tuple[str, str, str]] = []  # (member_path, crate_name, class)
    crate_name_set: set[str] = set()

    for member in members:
        crate_dir = root / member
        cargo_toml = crate_dir / "Cargo.toml"
        with open(cargo_toml, "rb") as f:
            data = tomllib.load(f)
        crate_name = data["package"]["name"]
        cls = classify_member(member)
        crate_info.append((member, crate_name, cls))
        crate_name_set.add(crate_name)

    # Second pass: collect edges
    edges: list[tuple[str, str]] = []
    for member, crate_name, _cls in crate_info:
        crate_dir = root / member
        _name, internal_deps = parse_crate_deps(crate_dir, crate_name_set)
        for dep in internal_deps:
            edges.append((dep, crate_name))  # dep -> crate (dep is depended upon)

    # Group crates by class
    groups: dict[str, list[str]] = {"extern": [], "core": [], "extension": []}
    for _member, crate_name, cls in crate_info:
        groups[cls].append(crate_name)

    # Build DOT
    lines = [
        "digraph workspace {",
        "    rankdir=BT;",
        "    compound=true;",
        '    node [shape=box, style="filled,rounded",',
        '          fontname="IBM Plex Sans", fontsize=12, margin="0.2,0.1"];',
        '    edge [color="#546e7a"];',
        "",
    ]

    # Render order: extern (bottom) -> core (middle) -> extension (top)
    render_order = ["extern", "core", "extension"]

    for cls in render_order:
        crates = groups[cls]
        if not crates:
            continue
        style = _CLASS_STYLES[cls]
        lines.append(f"    subgraph cluster_{cls} {{")
        lines.append(f'        label="{style["label"]}";')
        lines.append(f'        style="rounded,filled";')
        lines.append(f'        fillcolor="{style["cluster_bg"]}";')
        lines.append(f'        color="{style["cluster_border"]}";')
        lines.append(f'        fontname="IBM Plex Sans";')
        lines.append(f"        fontsize=14;")
        lines.append("")
        for name in crates:
            doc_dir = crate_to_doc_dir(name)
            url = f"{api_prefix}/{doc_dir}/index.html"
            fillcolor = style["fillcolor"]
            lines.append(
                f'        "{name}" [label="{name}", fillcolor="{fillcolor}", '
                f'URL="{url}", target="_parent"];'
            )
        lines.append("    }")
        lines.append("")

    # Edges
    for src, dst in edges:
        lines.append(f'    "{src}" -> "{dst}";')

    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate workspace dependency graph")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Workspace root directory",
    )
    parser.add_argument(
        "--api-prefix",
        default=".",
        help="URL prefix for crate doc links",
    )
    args = parser.parse_args()

    dot_source = generate_dot(args.root_dir, args.api_prefix)
    sys.stdout.write(dot_source)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
