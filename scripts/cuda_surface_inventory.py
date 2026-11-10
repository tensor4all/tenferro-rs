#!/usr/bin/env python3
"""Generate the `tenferro_gpu::cuda` surface inventory (issue #2298).

The Rust sources under `crates/tenferro-gpu/src/cuda/` are the membership
authority: every `pub`/`pub(crate)` item they declare, its visibility, whether
it carries `#[doc(hidden)]`, and whether a `pub use` chain reaches it from
`tenferro_gpu::cuda`.  Reference counts across the workspace, the crate's own
examples/tests and the guides are derived here as well.

The verdicts are reviewed judgement, not source syntax, so they live in the
`docs/internals/cuda-surface-verdicts.json` overlay.  This script refuses to
write a table while any item lacks a verdict or an overlay entry names an item
that no longer exists.

Usage:
    python3 scripts/cuda_surface_inventory.py            # regenerate the table
    python3 scripts/cuda_surface_inventory.py --check     # fail on drift
    python3 scripts/cuda_surface_inventory.py --missing   # list unverdicted items
    python3 scripts/cuda_surface_inventory.py --list-ids  # list every item id
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys
from typing import Iterator

ROOT = pathlib.Path(__file__).resolve().parents[1]
CUDA_SRC = ROOT / "crates/tenferro-gpu/src/cuda"
OVERLAY = ROOT / "docs/internals/cuda-surface-verdicts.json"
OUTPUT = ROOT / "docs/internals/cuda-surface-inventory.md"
# This tool's own output and the proposal that cites it: counting them would make the
# table depend on its own wording.


VERDICTS = {"public", "workspace-internal", "private"}

ITEM_RE = re.compile(
    r"^(?P<vis>pub(?:\s*\([^)]*\))?)?\s*"
    r"(?:default\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?"
    r"(?P<kind>struct|enum|union|trait|type|fn|const|static)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
MOD_RE = re.compile(r"^(?P<vis>pub(?:\s*\([^)]*\))?)?\s*mod\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;")
IMPL_RE = re.compile(
    r"^impl(?:<[^>]*>)?\s+"
    r"(?:(?:[A-Za-z_][\w:]*)(?:<[^>]*>)?\s+for\s+)?"
    r"(?P<ty>[A-Za-z_][\w:]*)(?:<[^>]*>)?(?:\s+where[^{]*)?\s*\{"
)
USE_RE = re.compile(r"^(?P<vis>pub(?:\s*\([^)]*\))?)?\s*use\s+(?P<body>.+?);\s*$")
USE_START_RE = re.compile(r"^(?:pub(?:\s*\([^)]*\))?\s+)?use\s+\S")
ATTRIBUTE_RE = re.compile(r"^#\[")
DOC_RE = re.compile(r"^(///|//!|/\*\*|\*/)")


@dataclasses.dataclass
class Item:
    module: str
    container: str | None
    kind: str
    name: str
    visibility: str
    doc_hidden: bool
    line: int
    cfg_test: bool

    @property
    def id(self) -> str:
        owner = f"{self.container}::" if self.container else ""
        return f"{self.module}::{owner}{self.name}"


@dataclasses.dataclass
class Export:
    module: str
    visibility: str
    line: int
    target: str
    names: list[str]
    child: bool = False


def logical_lines(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    """Yield stripped source lines, joining `use` statements broken across lines."""
    pending = ""
    start = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if pending:
            pending = f"{pending} {line}"
        elif USE_START_RE.match(line) and not line.endswith(";"):
            pending, start = line, number
            continue
        else:
            yield number, line
            continue
        if pending.endswith(";"):
            yield start, pending
            pending = ""
    if pending:
        yield start, pending


def module_of(path: pathlib.Path) -> str:
    relative = path.relative_to(CUDA_SRC).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "mod":
        parts.pop()
    return "cuda" + "".join(f"::{part}" for part in parts)


def is_test_file(path: pathlib.Path) -> bool:
    return any(part == "tests" for part in path.parts) or path.stem.endswith(
        ("tests", "test")
    )


def source_files() -> Iterator[pathlib.Path]:
    for path in sorted(CUDA_SRC.rglob("*.rs")):
        if not is_test_file(path):
            yield path


def scan_module(path: pathlib.Path) -> tuple[list[Item], list[Export]]:
    module = module_of(path)
    items: list[Item] = []
    exports: list[Export] = []
    pending: list[str] = []
    container: str | None = None
    impl_cfg_test = False
    impl_depth = 0
    depth = 0

    for number, line in logical_lines(path):
        if not line:  # blank lines keep the pending attribute block
            continue
        if ATTRIBUTE_RE.match(line) or DOC_RE.match(line) or line.startswith("//"):
            pending.append(line)
            continue

        opens = line.count("{")
        closes = line.count("}")

        use_match = USE_RE.match(line)
        if use_match:
            visibility = "pub" if use_match.group("vis") else "private"
            names, target = parse_use(use_match.group("body"))
            if names and visibility != "private":
                exports.append(Export(module, visibility, number, target, names))
            pending = []
            continue

        item_match = ITEM_RE.match(line)
        if item_match:
            attributes = "\n".join(pending)
            cfg_test = "cfg(test)" in attributes or impl_cfg_test
            public = normalize_visibility(item_match.group("vis")) != "private"
            if public and not cfg_test:
                items.append(
                    Item(
                        module=module,
                        container=container,
                        kind=item_match.group("kind"),
                        name=item_match.group("name"),
                        visibility=normalize_visibility(item_match.group("vis")),
                        doc_hidden="doc(hidden)" in attributes.replace(" ", ""),
                        line=number,
                        cfg_test=cfg_test,
                    )
                )
            pending = []
        else:
            mod_match = MOD_RE.match(line)
            if mod_match and not container:
                attributes = "\n".join(pending)
                visibility = normalize_visibility(mod_match.group("vis"))
                if visibility != "private" and "cfg(test)" not in attributes:
                    items.append(
                        Item(
                            module=module,
                            container=None,
                            kind="mod",
                            name=mod_match.group("name"),
                            visibility=visibility,
                            doc_hidden="doc(hidden)" in attributes.replace(" ", ""),
                            line=number,
                            cfg_test=False,
                        )
                    )
                if "cfg(test)" not in "\n".join(pending):
                    exports.append(
                        Export(
                            module,
                            visibility,
                            number,
                            mod_match.group("name"),
                            [mod_match.group("name")],
                            child=True,
                        )
                    )
                pending = []
            else:
                impl_match = IMPL_RE.match(line)
                if impl_match and not line.endswith(";"):
                    container = impl_match.group("ty").split("::")[-1]
                    impl_cfg_test = "cfg(test)" in "\n".join(pending)
                    impl_depth = depth
                    pending = []
                elif line and not ATTRIBUTE_RE.match(line):
                    pending = []

        depth += opens - closes
        if container is not None and depth <= impl_depth:
            container = None
            impl_cfg_test = False

    return items, exports


def normalize_visibility(raw: str | None) -> str:
    if not raw:
        return "private"
    if raw.replace(" ", "") in {"pub", "pub(crate)", "pub(super)"}:
        return raw.replace(" ", "")
    return "pub(in ...)"


def parse_use(body: str) -> tuple[list[str], str]:
    """Split a `use` body into the names it introduces and its module target."""
    body = body.strip()
    if body.startswith("::"):
        return [], ""
    if "{" not in body:
        leaf = body.split("::")[-1].strip()
        target = body[: -len(leaf)].rstrip(":")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", leaf):
            return [], ""
        return [leaf], resolve_target(target or "self")
    head, _, rest = body.partition("{")
    rest = rest.rsplit("}", 1)[0]
    names = [
        entry.strip().split(" as ")[-1].strip()
        for entry in rest.split(",")
        if entry.strip() and not entry.strip().startswith("//")
    ]
    names = [name for name in names if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)]
    return names, resolve_target(head.rstrip(":").strip() or "self")


def resolve_target(target: str) -> str:
    if target in {"self", ""}:
        return "."
    if target.startswith("super::"):
        return target
    return target


def load_overlay() -> dict[str, dict[str, str]]:
    if not OVERLAY.exists():
        return {}
    return json.loads(OVERLAY.read_text(encoding="utf-8"))["items"]


def resolve_exports(items: dict[str, Item], exports: list[Export]) -> None:
    """Attach the public path of every item a `pub` path reaches.

    Two ways to be reachable: the module chain itself is `pub` all the way from
    `tenferro_gpu::cuda` (only `raw` is), or a `pub use` re-export names the item.
    A `pub(crate)` module (`device`, `identity`, `vendor`) is not reachable, so its
    items need the re-export to count.
    """
    child_visibility = {
        f"{export.module}::{export.target}": export.visibility
        for export in exports
        if export.child
    }

    def open_module(module: str) -> bool:
        parts = module.split("::")
        return all(
            child_visibility.get("::".join(parts[: index + 1])) == "pub"
            for index in range(1, len(parts))
        )

    by_module: dict[str, dict[str, Item]] = {}
    for item in items.values():
        if item.container is None:
            by_module.setdefault(item.module, {})[item.name] = item

    for item in items.values():
        if item.container is None and item.visibility == "pub" and open_module(item.module):
            item.public_path = f"tenferro_gpu::{item.module}::{item.name}"

    for _ in range(8):  # `pub use` chains are short; iterate to a fixpoint
        changed = False
        for export in exports:
            if export.visibility != "pub":
                continue
            module = resolve_module(export.module, export.target)
            if module is None:
                continue
            for name in export.names:
                item = by_module.get(module, {}).get(name)
                if item is None or item.public_path:
                    continue
                parent = public_module_path(export.module)
                if parent is None:
                    continue
                item.public_path = f"{parent}::{name}"
                changed = True
        if not changed:
            break
    # A method is reachable wherever its owner is: the owner's re-export names the
    # type, and the method comes with it.
    owners = {
        item.public_path: item.public_path.split("::")[-1]
        for item in items.values()
        if item.container is None and item.public_path
    }
    for item in items.values():
        if item.container and item.public_path is None and item.visibility == "pub":
            for path, name in owners.items():
                if name == item.container:
                    item.public_path = f"{path}::{item.name}"
                    break


def resolve_module(current: str, target: str) -> str | None:
    if target == ".":
        return current
    if target.startswith("super::"):
        parts = current.split("::")
        return "::".join(parts[:-1] + target.split("::")[1:])
    if target.startswith("crate::"):
        return target[len("crate::") :]
    if "::" in target:
        return None
    return f"{current}::{target}" if current else target


def public_module_path(module: str) -> str | None:
    return "tenferro_gpu::" + module


def workspace_files() -> tuple[list[pathlib.Path], list[pathlib.Path], list[pathlib.Path]]:
    """Return (other-crate sources, in-crate user files, guide files)."""
    others: list[pathlib.Path] = []
    for base in (ROOT / "crates", ROOT / "ext"):
        for path in sorted(base.rglob("*.rs")):
            if path.is_relative_to(ROOT / "crates/tenferro-gpu"):
                continue
            others.append(path)
    in_crate: list[pathlib.Path] = []
    for name in ("examples", "tests", "benches"):
        in_crate.extend(sorted((ROOT / "crates/tenferro-gpu" / name).rglob("*.rs")))
    guides: list[pathlib.Path] = sorted(
        path
        for base in ("guides", "getting-started", "reference", "api", "tutorials")
        for path in (ROOT / "docs" / base).rglob("*.md")
    )
    guides.extend(sorted((ROOT / "docs/tutorial-code/src").rglob("*.rs")))
    return others, in_crate, guides


def mentions(text: str) -> bool:
    return "tenferro_gpu" in text


def reference_pattern(item: Item) -> re.Pattern[str]:
    if item.kind == "fn":
        return re.compile(rf"\b{re.escape(item.name)}\s*\(")
    return re.compile(rf"\b{re.escape(item.name)}\b")


def codeish_pattern(item: Item) -> re.Pattern[str]:
    """A prose-safe pattern: a name counts only when it is written as code.

    Prose mentions ("ATen Dispatcher", "one placement token") are not API use, so a
    documentation reference must be backticked, qualified, or called.
    """
    name = re.escape(item.name)
    if item.container:
        return re.compile(
            rf"`{name}`|\b{re.escape(item.container)}\s*::\s*{name}\b|\.{name}\s*\("
        )
    return re.compile(rf"`{name}`|::{name}(?![A-Za-z0-9_])|\b{name}\s*\(")


def count_references(
    items: dict[str, Item],
    files: list[pathlib.Path],
    limit: int = 3,
    codeish: bool = False,
    named: set[pathlib.Path] | None = None,
) -> dict[str, tuple[int, list[str]]]:
    """Count references per item, with up to `limit` `file:line` examples.

    A module-level item counts wherever its name appears as a word, but only in a file
    listed in `named`: a bare `Operand` or `Contraction` in an unrelated crate means
    nothing.  A method counts its qualified `Type::name` uses anywhere, and a receiver
    call `.name(` only in a file that also names the owner type -- otherwise every
    other type's `.stats(`, `.bytes(` and `.dtype(` would count as a use of this one.
    This is a lead, not proof: `.len(` inside a file that also names the owner type still
    counts, and a call through a variable whose type the file never spells out does not.
    The review recorded in the verdicts is what settles each item.
    """
    patterns: dict[str, list[re.Pattern[str]]] = {}
    for key, item in items.items():
        if codeish:
            patterns[key] = [codeish_pattern(item)]
            continue
        if item.container is None:
            patterns[key] = [reference_pattern(item)]
            continue
        owner = re.escape(item.container)
        name = re.escape(item.name)
        patterns[key] = [re.compile(rf"\b{owner}\s*::\s*{name}\b")]
        patterns[key].append(re.compile(rf"\.{name}\s*\("))

    by_name: dict[str, list[str]] = {}
    for key, item in items.items():
        by_name.setdefault(item.name, []).append(key)
    presence = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(by_name))) + r")\b")
    containers = sorted({item.container for item in items.values() if item.container})

    counts = {key: 0 for key in items}
    examples: dict[str, list[str]] = {key: [] for key in items}
    for path, text in (
        (path, path.read_text(encoding="utf-8", errors="replace")) for path in files
    ):
        mentioned = named is None or path in named
        owners = {name for name in containers if name in text}
        present = set(presence.findall(text))
        if not present:
            continue
        lines = text.splitlines()
        for name in present:
            for key in by_name[name]:
                item = items[key]
                if item.container is None:
                    if not mentioned:
                        continue
                    for pattern in patterns[key]:
                        for match in pattern.finditer(text):
                            counts[key] += 1
                            if len(examples[key]) < limit:
                                line = text.count("\n", 0, match.start()) + 1
                                examples[key].append(f"{path.relative_to(ROOT)}:{line}")
                    continue
                for number, line in enumerate(lines, start=1):
                    if item.name not in line:
                        continue
                    if any(p.search(line) for p in patterns[key][:1]):
                        pass  # qualified: the owner is on the line by construction
                    elif item.container in owners:
                        pass
                    else:
                        continue
                    if any(p.search(line) for p in patterns[key]):
                        counts[key] += 1
                        if len(examples[key]) < limit:
                            examples[key].append(f"{path.relative_to(ROOT)}:{number}")
    return {key: (counts[key], examples[key]) for key in items}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the table is stale")
    parser.add_argument("--missing", action="store_true", help="list unverdicted items")
    parser.add_argument("--list-ids", action="store_true", help="list every item id")
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()

    items: dict[str, Item] = {}
    exports: list[Export] = []
    for path in source_files():
        module_items, module_exports = scan_module(path)
        for item in module_items:
            if item.id in items:
                raise SystemExit(f"duplicate item id {item.id}")
            items[item.id] = item
        exports.extend(module_exports)

    for item in items.values():
        item.public_path = None  # type: ignore[attr-defined]

    resolve_exports(items, exports)

    overlay = load_overlay()
    unverdicted = sorted(key for key in items if key not in overlay)
    stale = sorted(key for key in overlay if key not in items)
    if args.list_ids or args.missing:
        for key in sorted(items) if args.list_ids else unverdicted:
            print(key)
        return 0
    if stale:
        raise SystemExit(f"overlay names items that no longer exist: {', '.join(stale)}")
    if unverdicted:
        raise SystemExit(
            f"{len(unverdicted)} items have no verdict; run --missing: {', '.join(unverdicted[:5])} ..."
        )
    for key, entry in overlay.items():
        if entry["verdict"] not in VERDICTS:
            raise SystemExit(f"{key}: unknown verdict {entry['verdict']!r}")
        if not entry.get("reason"):
            raise SystemExit(f"{key}: a verdict needs a reason")
        for site in entry.get("evidence", []):
            path, _, line = site.partition(":")
            if not (ROOT / path).is_file() or not line.isdigit():
                raise SystemExit(f"{key}: evidence {site!r} is not a file:line in this checkout")

    others, in_crate, guides = workspace_files()
    # A same-named item in another crate is not a use of this one: keep only files
    # that depend on tenferro-gpu at all.
    named = {
        path
        for path in others
        if mentions(path.read_text(encoding="utf-8", errors="replace"))
    }
    other_refs = count_references(items, others, named=named)
    crate_refs = count_references(items, in_crate)
    guide_refs = count_references(items, guides, codeish=True)

    hidden_total = sum(1 for item in items.values() if item.doc_hidden)
    cuda_hidden = sum(
        1
        for path in source_files()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "doc(hidden)" in line
    )
    verdicts = {verdict: 0 for verdict in sorted(VERDICTS)}
    for entry in overlay.values():
        verdicts[entry["verdict"]] += 1

    body = render(items, overlay, other_refs, crate_refs, guide_refs)
    header = [
        "<!-- GENERATED FILE: scripts/cuda_surface_inventory.py; do not edit. -->",
        "# `tenferro_gpu::cuda` surface inventory",
        "",
        f"- Items in the table: **{len(items)}** (`#[doc(hidden)]` lines under `cuda/`: {cuda_hidden};"
        f" table rows carrying the mark: {hidden_total})",
        "- Verdicts: "
        + ", ".join(f"{name} {count}" for name, count in verdicts.items())
        + ".",
        "- Regenerate: `python3 scripts/cuda_surface_inventory.py`;"
        " verdicts live in `docs/internals/cuda-surface-verdicts.json`.",
        "- Scope: `pub`/`pub(crate)`/`pub(super)` items declared under"
        " `crates/tenferro-gpu/src/cuda/**` -- modules, structs, enums, unions, traits, type"
        " aliases, functions, constants and statics, plus the methods of their `impl` blocks --"
        " excluding `#[cfg(test)]` items, `#[cfg(test)]` impls and test modules. Struct fields and"
        " enum variants are not rows; a `#[doc(hidden)]` on one still counts in the mark total.",
        "- Reference counts: other workspace crates (excluding `tenferro-gpu`), the crate's own"
        " `examples/`, `tests/` and `benches/`, and the user-facing guides (`docs/guides`,"
        " `docs/getting-started`, `docs/reference`, `docs/api`, `docs/tutorials`,"
        " `docs/tutorial-code`). Design notes, plans and worklogs are not counted: they say what"
        " someone once meant, not who uses the item.",
        "- Counts are name-based. A module-level item counts only in files that name"
        " `tenferro_gpu`; a method counts its qualified `Type::name` uses and its `.name(`"
        " receiver calls in files that name the owner type, so a same-named method on another"
        " type can still raise a count. Treat a count as a lead and the `evidence` column as"
        " the review.",
        "- Evidence is a citation, not a verification: the script checks only that the cited file"
        " exists and that the line field is numeric. It does not read the line, so a `(cited at ...)`"
        " site shows where a reviewer looked, not that the code there is the item on the row -- a"
        " same-named method on another receiver passes. A row whose caller evidence nobody could"
        " confirm says so in its verdict reason instead of citing a site.",
        "",
    ]
    text = "\n".join(header) + body
    destination = pathlib.Path(args.out)
    if args.check:
        current = destination.read_text(encoding="utf-8") if destination.exists() else ""
        if current != text:
            print(f"{destination.relative_to(ROOT)} is stale; regenerate it", file=sys.stderr)
            return 1
        print(f"{destination.relative_to(ROOT)} is current")
        return 0
    destination.write_text(text, encoding="utf-8")
    print(f"wrote {destination.relative_to(ROOT)}: {len(items)} items")
    return 0


def render(
    items: dict[str, Item],
    overlay: dict[str, dict[str, str]],
    other_refs: dict[str, tuple[int, list[str]]],
    crate_refs: dict[str, tuple[int, list[str]]],
    guide_refs: dict[str, tuple[int, list[str]]],
) -> str:
    lines = [
        "| item | kind | vis | hidden | reachable from `cuda` | other crates | examples/tests"
        " | guides | verdict | evidence |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(items):
        item = items[key]
        reachable = getattr(item, "public_path", None)
        other_count, other_examples = other_refs[key]
        crate_count, crate_examples = crate_refs[key]
        guide_count, guide_examples = guide_refs[key]
        where = ", ".join(other_examples) if other_examples else "—"
        in_crate = ", ".join(crate_examples) if crate_examples else "—"
        in_guides = ", ".join(guide_examples) if guide_examples else "—"
        entry = overlay[key]
        if entry.get("evidence"):
            reason = "{} (cited at {})".format(
                entry["reason"], ", ".join(f"`{site}`" for site in entry["evidence"])
            )
        else:
            reason = entry["reason"]
        lines.append(
            "| `{path}` | {kind} | `{vis}` | {hidden} | {reachable} | {others} {where} | {crate_n} {crate_where} | {guide_n} {guide_where} | **{verdict}** | {reason} |".format(
                path=reachable or f"crate::{key}",
                kind=item.kind,
                vis=item.visibility,
                hidden="yes" if item.doc_hidden else "no",
                reachable="yes" if reachable else "no",
                others=other_count,
                where=where,
                crate_n=crate_count,
                crate_where=in_crate,
                guide_n=guide_count,
                guide_where=in_guides,
                verdict=entry["verdict"],
                reason=reason,
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
