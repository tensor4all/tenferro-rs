#!/usr/bin/env python3
"""Generate a Graphviz DOT dependency graph from workspace Cargo.toml files.

Reads the workspace root Cargo.toml to discover members, then parses each
member's Cargo.toml to find workspace-internal dependencies. Outputs a DOT
graph to stdout.

Members are classified into the user-facing layers shown in the dependency
footprint: foundations, tensor/backends, runtime/AD, standard operation
extensions, and runnable documentation examples.

Each group is rendered as a DOT subgraph cluster with a distinct color.
Node links point to ``index.html#<crate-name>`` anchors so that clicking
a node in the embedded SVG scrolls the parent page to the crate description.

Usage:
    python3 scripts/gen_dep_graph.py [--root-dir DIR]
    python3 scripts/gen_dep_graph.py --format svg --output PATH
    python3 scripts/gen_dep_graph.py --check-svg PATH

Options:
    --root-dir DIR        Workspace root (default: script's parent directory)
    --format dot|svg      Emit DOT (default) or render SVG with Graphviz
    --output PATH         Write generated output to a file
    --check-svg PATH      Check SVG node/edge inventory against the workspace
"""

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import tomllib
from collections import defaultdict, deque
from pathlib import Path


# Keep the conceptual documentation layers explicit. Falling back to ``core``
# ensures a newly added workspace crate still appears until its intended layer
# is recorded here.
_MEMBER_CLASSES = {
    "crates/tenferro-tensor-core": "foundation",
    "crates/tenferro-core-ops": "foundation",
    "crates/tenferro-internal-extension-macros": "foundation",
    "crates/tenferro-tensor": "tensor_backend",
    "crates/tenferro-internal-cpu-kernels": "tensor_backend",
    "crates/tenferro-cpu": "tensor_backend",
    "crates/tenferro-gpu": "tensor_backend",
    "crates/tenferro-internal-ops": "tensor_backend",
    "crates/tenferro-runtime": "runtime_ad",
    "crates/tenferro-xla": "runtime_ad",
    "crates/tenferro-ad": "runtime_ad",
    "crates/tenferro-einsum": "extension",
    "crates/tenferro-linalg": "extension",
    "crates/tenferro-fft": "extension",
    "docs/tutorial-code": "tutorial",
}

_CLASS_ORDER = [
    "foundation",
    "tensor_backend",
    "runtime_ad",
    "extension",
    "tutorial",
    "core",
]

_CLASS_STYLES = {
    "foundation": {
        "label": "Foundation",
        "fillcolor": "#eef2ff",
        "cluster_bg": "#f5f7ff",
        "cluster_border": "#6366f1",
    },
    "tensor_backend": {
        "label": "Tensor and backends",
        "fillcolor": "#ecfdf5",
        "cluster_bg": "#f3fcf8",
        "cluster_border": "#059669",
    },
    "runtime_ad": {
        "label": "Runtime and AD",
        "fillcolor": "#fff7ed",
        "cluster_bg": "#fffaf5",
        "cluster_border": "#ea580c",
    },
    "core": {
        "label": "Other workspace crates",
        "fillcolor": "#f8fafc",
        "cluster_bg": "#f8fafc",
        "cluster_border": "#64748b",
    },
    "extension": {
        "label": "Standard operation extensions",
        "fillcolor": "#e0f2fe",
        "cluster_bg": "#f0f9ff",
        "cluster_border": "#0284c7",
    },
    "tutorial": {
        "label": "Runnable docs examples",
        "fillcolor": "#fefce8",
        "cluster_bg": "#fffef2",
        "cluster_border": "#ca8a04",
    },
}


def parse_workspace_members(root: Path) -> list[str]:
    """Return the list of workspace member directory paths."""
    cargo_toml = root / "Cargo.toml"
    with open(cargo_toml, "rb") as f:
        data = tomllib.load(f)
    return data["workspace"]["members"]


def classify_member(member_path: str) -> str:
    """Classify a workspace member into a dependency-footprint layer."""
    return _MEMBER_CLASSES.get(Path(member_path).as_posix().rstrip("/"), "core")


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
        # dep_key is the dependency name (e.g. "tenferro-internal-device")
        if dep_key in all_crate_names:
            internal.append(dep_key)
        elif isinstance(dep_val, dict) and dep_val.get("package", dep_key) in all_crate_names:
            internal.append(dep_val["package"])
    return crate_name, internal


def adjacency_from_edges(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Return adjacency sets for a directed graph."""
    adj: dict[str, set[str]] = defaultdict(set)
    for src, dst in edges:
        adj[src].add(dst)
        adj.setdefault(dst, set())
    return adj


def has_path_excluding_edge(
    adj: dict[str, set[str]], src: str, dst: str, excluded: tuple[str, str]
) -> bool:
    """Return whether src reaches dst without traversing excluded edge."""
    queue = deque([src])
    seen = {src}
    while queue:
        current = queue.popleft()
        for nxt in adj.get(current, ()):
            if (current, nxt) == excluded:
                continue
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def transitive_reduction(
    nodes: list[str], edges: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Drop edges implied by another path in this DAG-like workspace graph."""
    adj = adjacency_from_edges(edges)
    reduced: list[tuple[str, str]] = []
    for edge in edges:
        src, dst = edge
        if src not in nodes or dst not in nodes:
            continue
        if not has_path_excluding_edge(adj, src, dst, edge):
            reduced.append(edge)
    return reduced


def generate_dot(
    root: Path,
    *,
    reduce_transitive_edges: bool = True,
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

    if reduce_transitive_edges:
        ordered_nodes = [crate_name for _member, crate_name, _cls in crate_info]
        edges = transitive_reduction(ordered_nodes, edges)

    # Group crates by class
    groups: dict[str, list[str]] = {name: [] for name in _CLASS_ORDER}
    for _member, crate_name, cls in crate_info:
        groups[cls].append(crate_name)

    # Build DOT
    lines = [
        "digraph workspace {",
        "    rankdir=TB;",
        "    compound=true;",
        "    splines=ortho;",
        "    outputorder=edgesfirst;",
        '    bgcolor="#ffffff";',
        "    margin=0;",
        "    pad=0.08;",
        "    ranksep=0.45;",
        "    nodesep=0.20;",
        '    graph [fontname="IBM Plex Sans", fontsize=18,',
        '           label="tenferro-rs dependency footprint", labelloc=t];',
        '    node [shape=box, style="filled,rounded", fixedsize=false,',
        '          fontname="IBM Plex Sans", fontsize=12, margin="0.11,0.055"];',
        '    edge [color="#546e7a", arrowsize=0.8];',
        "",
    ]

    for cls in _CLASS_ORDER:
        crates = groups[cls]
        if not crates:
            continue
        style = _CLASS_STYLES[cls]
        title_node = f"cluster_title_{cls}"
        lines.append(f"    subgraph cluster_{cls} {{")
        lines.append('        label="";')
        lines.append(f'        style="rounded,filled";')
        lines.append(f'        fillcolor="{style["cluster_bg"]}";')
        lines.append(f'        color="{style["cluster_border"]}";')
        lines.append(f'        fontname="IBM Plex Sans";')
        lines.append(f"        fontsize=14;")
        lines.append("        margin=8;")
        lines.append("")
        lines.append(
            f'        {title_node} [label="{style["label"]}", '
            'class="cluster-title", shape=plain, style="", fixedsize=false,'
        )
        lines.append(
            '            fontname="IBM Plex Sans", fontsize=14];'
        )
        lines.append(f"        {{ rank=source; {title_node}; }}")
        lines.append("")
        for name in crates:
            url = f"index.html#{name}"
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


def render_svg(dot_source: str, *, dot_command: str = "dot") -> str:
    """Render DOT source to SVG with Graphviz."""
    try:
        result = subprocess.run(
            [dot_command, "-Tsvg"],
            input=dot_source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Graphviz executable not found: {dot_command!r}; install graphviz to render SVG"
        ) from error
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"Graphviz SVG rendering failed: {diagnostic}")
    return _add_accessible_svg_metadata(result.stdout)


def _add_accessible_svg_metadata(svg_source: str) -> str:
    match = re.search(r"<svg\b[^>]*>", svg_source)
    if match is None:
        raise RuntimeError("Graphviz SVG rendering failed: output has no <svg> root")

    opening_tag = match.group(0)
    opening_tag = (
        opening_tag[:-1]
        + ' role="img" aria-labelledby="dependency-footprint-title dependency-footprint-desc">'
    )
    metadata = (
        '\n<title id="dependency-footprint-title">'
        "tenferro-rs workspace dependency footprint"
        "</title>"
        '\n<desc id="dependency-footprint-desc">'
        "Transitive-reduced workspace dependencies. Arrows point from each "
        "dependency to the crate that uses it."
        "</desc>"
    )
    return svg_source[: match.start()] + opening_tag + metadata + svg_source[match.end() :]


def _dot_inventory(dot_source: str) -> tuple[set[str], set[tuple[str, str]]]:
    nodes = set(re.findall(r'^\s*"([^"]+)" \[label=', dot_source, flags=re.MULTILINE))
    edges = set(
        re.findall(
            r'^\s*"([^"]+)" -> "([^"]+)";$',
            dot_source,
            flags=re.MULTILINE,
        )
    )
    return nodes, edges


def _svg_inventory(svg_source: str) -> tuple[set[str], set[tuple[str, str]]]:
    root = ET.fromstring(svg_source)
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for group in root.iter():
        kind = group.attrib.get("class")
        if kind not in {"node", "edge"}:
            continue
        title = next(
            (
                child.text or ""
                for child in group
                if child.tag.rsplit("}", 1)[-1] == "title"
            ),
            "",
        )
        if kind == "node" and title:
            nodes.add(title)
        elif kind == "edge" and "->" in title:
            src, dst = title.split("->", 1)
            edges.add((src, dst))
    return nodes, edges


def validate_svg(dot_source: str, svg_source: str) -> list[str]:
    """Return semantic drift between canonical DOT and a Graphviz SVG."""
    expected_nodes, expected_edges = _dot_inventory(dot_source)
    try:
        actual_nodes, actual_edges = _svg_inventory(svg_source)
    except ET.ParseError as error:
        return [f"invalid SVG XML: {error}"]

    errors: list[str] = []
    svg_root = ET.fromstring(svg_source)
    if svg_root.attrib.get("role") != "img":
        errors.append('SVG root is missing role="img"')
    expected_label_ids = (
        "dependency-footprint-title",
        "dependency-footprint-desc",
    )
    if set(svg_root.attrib.get("aria-labelledby", "").split()) != set(
        expected_label_ids
    ):
        errors.append("SVG root has invalid aria-labelledby references")
    ids = {element.attrib.get("id") for element in svg_root.iter()}
    for required_id in expected_label_ids:
        if required_id not in ids:
            errors.append(f"SVG is missing accessible metadata: {required_id}")
    if not actual_nodes:
        errors.append("SVG has no Graphviz node groups")
    missing_nodes = sorted(expected_nodes - actual_nodes)
    unexpected_nodes = sorted(actual_nodes - expected_nodes)
    missing_edges = sorted(expected_edges - actual_edges)
    unexpected_edges = sorted(actual_edges - expected_edges)
    if missing_nodes:
        errors.append(f"missing nodes: {', '.join(missing_nodes)}")
    if unexpected_nodes:
        errors.append(f"unexpected nodes: {', '.join(unexpected_nodes)}")
    if missing_edges:
        errors.append(
            "missing edges: " + ", ".join(f"{src}->{dst}" for src, dst in missing_edges)
        )
    if unexpected_edges:
        errors.append(
            "unexpected edges: "
            + ", ".join(f"{src}->{dst}" for src, dst in unexpected_edges)
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate workspace dependency graph")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Workspace root directory",
    )
    parser.add_argument(
        "--no-transitive-reduction",
        action="store_true",
        help="Keep all direct workspace dependency edges instead of simplifying transitively implied ones",
    )
    parser.add_argument(
        "--format",
        choices=("dot", "svg"),
        default="dot",
        help="Output format (default: dot)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write output to this path instead of stdout",
    )
    parser.add_argument(
        "--dot-command",
        default="dot",
        help="Graphviz dot executable used for --format svg (default: dot)",
    )
    parser.add_argument(
        "--check-svg",
        type=Path,
        help="Check a Graphviz SVG against the current workspace graph",
    )
    args = parser.parse_args()

    dot_source = generate_dot(
        args.root_dir,
        reduce_transitive_edges=not args.no_transitive_reduction,
    )
    if args.check_svg is not None:
        errors = validate_svg(
            dot_source,
            args.check_svg.read_text(encoding="utf-8"),
        )
        if errors:
            for error in errors:
                print(f"{args.check_svg}: {error}", file=sys.stderr)
            return 1
        print(f"dependency-graph-svg-ok: {args.check_svg}")
        return 0

    try:
        output = (
            render_svg(dot_source, dot_command=args.dot_command)
            if args.format == "svg"
            else dot_source + "\n"
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
