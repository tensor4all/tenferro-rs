#!/usr/bin/env python3
"""Generate a Graphviz DOT dependency graph from workspace Cargo.toml files.

Reads the workspace root Cargo.toml to discover members, then parses each
member's Cargo.toml to find workspace-internal dependencies. Outputs a DOT
graph to stdout.

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


def parse_workspace_members(root: Path) -> list[str]:
    """Return the list of workspace member directory names."""
    cargo_toml = root / "Cargo.toml"
    with open(cargo_toml, "rb") as f:
        data = tomllib.load(f)
    return data["workspace"]["members"]


def parse_crate_deps(crate_dir: Path, all_members: set[str]) -> tuple[str, list[str]]:
    """Return (crate_name, [internal_dependency_names])."""
    cargo_toml = crate_dir / "Cargo.toml"
    with open(cargo_toml, "rb") as f:
        data = tomllib.load(f)

    crate_name = data["package"]["name"]
    deps = data.get("dependencies", {})
    internal = []
    for dep_key, dep_val in deps.items():
        # dep_key is the dependency name (e.g. "tenferro-device")
        if dep_key in all_members:
            internal.append(dep_key)
        elif isinstance(dep_val, dict) and dep_val.get("package", dep_key) in all_members:
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
    member_set = set(members)

    # Collect all edges
    edges: list[tuple[str, str]] = []
    crate_names: list[str] = []

    for member in members:
        crate_dir = root / member
        crate_name, internal_deps = parse_crate_deps(crate_dir, member_set)
        crate_names.append(crate_name)
        for dep in internal_deps:
            edges.append((dep, crate_name))  # dep -> crate (dep is depended upon)

    # Build DOT
    lines = [
        "digraph workspace {",
        "    rankdir=BT;",
        '    node [shape=box, style="filled,rounded", fillcolor="#e8f5e9",',
        '          fontname="IBM Plex Sans", fontsize=12, margin="0.2,0.1"];',
        '    edge [color="#546e7a"];',
        "",
    ]

    # Nodes with links
    for name in crate_names:
        doc_dir = crate_to_doc_dir(name)
        url = f"{api_prefix}/{doc_dir}/index.html"
        label = name
        lines.append(f'    "{name}" [label="{label}", URL="{url}", target="_parent"];')

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
