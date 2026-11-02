#!/usr/bin/env python3
"""Detect duplicated implementation paths ("twin paths") from token facts only.

The layering rule forbids duplicated logic, but nothing but diff review ever
looked for *existing* twins. This checker reports three kinds, each decided from
tokens rather than from a similarity model:

  body      Two functions whose comment- and literal-normalized token streams
            are identical, or whose 5-token shingle Jaccard is >= 0.85 with at
            least 40 tokens on both sides.
  branches  Two functions in different files that each enumerate the variants of
            the same enum. This is the table that gets half-updated when an
            operation is added.
  dead      A `pub`/`pub(crate)` function with 0-1 production references. The
            definition alone, an import and a re-export are not callers.

Reviewed exceptions live in `scripts/ci/twin-paths-allowlist.toml`. An entry needs
a reason, because an unexplained suppression is a defect. `--check` exits 1 when
a finding has no entry; `--report` never fails on findings.

Scope is `crates/<crate>/src/**/*.rs`. Test sources (`tests/`, `tests.rs`,
`*_tests.rs`, `#[cfg(test)]` items) are the users of an implementation, not an
implementation, and are out of scope. Only free functions and inherent methods
are considered: a trait impl method has callers the token stream cannot show.

Usage:
    python3 scripts/ci/check-twin-paths.py [--report|--check] [--json]

Lives under `scripts/ci/` because a new top-level `scripts/*.py` is an unknown
path in the base revision's change policy, which reserves the paid GPU lane for
an otherwise Python-only diff (scripts/ci/change_policy.py).
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path("scripts") / "ci" / "twin-paths-allowlist.toml"

# A 15-token body is a getter; calling it a duplicated path is noise.
MIN_BODY_TOKENS = 15
# Approximate matching needs enough context for a shingle to mean anything.
MIN_SHINGLE_TOKENS = 40
SHINGLE = 5
SIMILARITY = 0.85
# Fewer than this many variants is a three-arm match, not a dispatch table.
MIN_VARIANTS = 8
# Per shared variant, in the same currency as a duplicated body token: a drift in
# an operation table is a correctness hazard, a duplicated 100-token body is not.
VARIANT_WEIGHT = 8
DEFAULT_LIMIT = 30
# A declared surface with no caller is a contract defect, not a small one: its
# weight is a mid-size duplicated body, so it is not ranked by token count alone.
DEAD_ADVERTISED_WEIGHT = 250

# Trait impls whose bodies are shaped by the trait, not by a second design.
BOILERPLATE_TRAITS = frozenset(
    {
        "Add",
        "AddAssign",
        "AsRef",
        "Borrow",
        "Clone",
        "Debug",
        "Default",
        "Deref",
        "Deserialize",
        "Display",
        "Div",
        "Drop",
        "Eq",
        "Error",
        "From",
        "FromStr",
        "Hash",
        "Index",
        "Mul",
        "Neg",
        "Ord",
        "PartialEq",
        "PartialOrd",
        "Serialize",
        "Sub",
        "TryFrom",
    }
)

DIGITS = frozenset("0123456789")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER = re.compile(r"[0-9][0-9A-Za-z_]*(?:\.[0-9][0-9A-Za-z_]*)?")
RAW_STRING = re.compile(r'(?:b|c)?r(#*)"')
STRING = re.compile(r'(?:b|c)?"')
OPERATORS = (
    "<<=",
    ">>=",
    "...",
    "::",
    "->",
    "=>",
    "==",
    "!=",
    "<=",
    ">=",
    "&&",
    "||",
    "<<",
    ">>",
    "..=",
    "..",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "^=",
    "&=",
    "|=",
)
OPEN = {"(", "["}
CLOSE = {")", "]"}
PENDING_REASONS = frozenset({"pending", "todo", "tbd", "fixme"})


# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #


def lex(text: str) -> tuple[list[str], list[int]]:
    """Split Rust source into tokens, dropping comments and whitespace.

    String, character and raw-string literals become the single token
    `<literal>`: their contents never identify an implementation path, and an
    error message is not a design difference.
    """
    tokens: list[str] = []
    lines: list[int] = []
    i, line, n = 0, 1, len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        start_line = line
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    if text[i] == "\n":
                        line += 1
                    i += 1
            continue
        match = RAW_STRING.match(text, i)
        if match:
            closing = '"' + match.group(1)
            end = text.find(closing, match.end())
            end = n if end < 0 else end + len(closing)
            line += text.count("\n", i, end)
            tokens.append("<literal>")
            lines.append(start_line)
            i = end
            continue
        match = STRING.match(text, i)
        if match:
            j = match.end()
            while j < n:
                current = text[j]
                if current == "\\":
                    j += 2
                    continue
                if current == '"':
                    j += 1
                    break
                if current == "\n":
                    line += 1
                j += 1
            tokens.append("<literal>")
            lines.append(start_line)
            i = j
            continue
        if ch == "'":
            if i + 1 < n and text[i + 1] == "\\":
                j = i + 2
                while j < n and text[j] != "'":
                    if text[j] == "\n":
                        line += 1
                    j += 1
                tokens.append("<literal>")
                lines.append(start_line)
                i = j + 1
                continue
            if i + 2 < n and text[i + 2] == "'" and text[i + 1] != "\n":
                tokens.append("<literal>")
                lines.append(start_line)
                i += 3
                continue
            match = IDENT.match(text, i + 1)
            if match:
                tokens.append("'" + match.group(0))
                lines.append(start_line)
                i = match.end()
                continue
            tokens.append("'")
            lines.append(start_line)
            i += 1
            continue
        match = NUMBER.match(text, i) if ch in DIGITS else None
        if match:
            tokens.append(match.group(0))
            lines.append(start_line)
            i = match.end()
            continue
        match = IDENT.match(text, i) if ch.isalpha() or ch == "_" else None
        if match:
            tokens.append(match.group(0))
            lines.append(start_line)
            i = match.end()
            continue
        for operator in OPERATORS:
            if text.startswith(operator, i):
                tokens.append(operator)
                lines.append(start_line)
                i += len(operator)
                break
        else:
            tokens.append(ch)
            lines.append(start_line)
            i += 1
    return tokens, lines


def is_ident(token: str) -> bool:
    return IDENT.fullmatch(token) is not None


# --------------------------------------------------------------------------- #
# Source structure
# --------------------------------------------------------------------------- #


@dataclass
class Function:
    path: str
    name: str
    line: int
    visibility: str
    body: tuple[str, ...]
    variants: dict[str, frozenset[str]]
    boilerplate: bool
    exposed: bool
    container: str
    impl_trait: str | None
    crate: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.name}"

    @property
    def size(self) -> int:
        return len(self.body)


def is_test_path(relative: Path) -> bool:
    """Test sources are users of an implementation, not an implementation."""
    parts = relative.parts
    if parts[-1] == "tests.rs" or parts[-1].endswith("_tests.rs"):
        return True
    if "test_support" in parts[-1] or "test_support" in parts[3:-1]:
        return True
    return "tests" in parts[3:-1]


def header_is_test(header: list[str]) -> bool:
    """Only an attribute makes an item a test item; `test` elsewhere is a name."""
    return bool(header) and header[0] == "#" and "test" in header


def impl_trait(header: list[str]) -> str | None:
    """Trait name of an `impl ... for ...` header, or None for an inherent impl."""
    start = header.index("impl") + 1
    depth = 0
    name = None
    for token in header[start:]:
        if token == "<":
            depth += 1
        elif token == ">":
            depth -= 1
        elif depth:
            continue
        elif token == "for":
            return name
        elif is_ident(token):
            name = token
    return None


def classify(header: list[str], stack: list[tuple]) -> tuple[str, str | None, bool]:
    is_test = bool(stack and stack[-1][2]) or header_is_test(header)
    if "macro_rules" in header:
        return "macro", None, is_test
    if "impl" in header:
        return "impl", impl_trait(header), is_test
    if "trait" in header:
        return "trait", None, is_test
    if "mod" in header:
        return "mod", None, is_test
    return "block", None, is_test


def visibility(header: list[str]) -> str:
    if "pub" not in header:
        return "private"
    position = header.index("pub")
    tail = header[position + 1 : position + 5]
    if tail[:1] == ["("]:
        # `pub(crate)`, `pub(super)` and `pub(in ...)` stop at the crate boundary.
        if {"crate", "super", "in"} & set(tail):
            return "pub(crate)"
    return "pub"


def enclosing_impl(stack: list[tuple]) -> str | None:
    for kind, value, _is_test, _span in reversed(stack):
        if kind == "impl":
            return value
    return None


def enclosing_container(stack: list[tuple]) -> str:
    for kind, _value, _is_test, _span in reversed(stack):
        if kind in {"impl", "trait", "macro"}:
            return kind
    return "free"


def match_brace(tokens: list[str], opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(tokens)):
        token = tokens[index]
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def parse_function(
    tokens: list[str], lines: list[int], index: int, start: int, stack: list[tuple]
) -> dict | None:
    """Parse the definition starting at `fn` token `index`, or None if it declares."""
    cursor = index + 2
    depth = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if token in OPEN:
            depth += 1
        elif token in CLOSE:
            depth -= 1
        elif depth == 0:
            if token == ";":
                return None
            if token == "{":
                break
            if token == "}":
                return None
        cursor += 1
    if cursor >= len(tokens):
        return None
    end = match_brace(tokens, cursor)
    if end is None:
        return None
    header = tokens[start:index]
    return {
        "name": tokens[index + 1],
        "line": lines[index + 1],
        "visibility": visibility(header),
        "body": tuple(tokens[cursor + 1 : end]),
        "end": end,
        "container": enclosing_container(stack),
        "impl_trait": enclosing_impl(stack),
    }


def variant_table(body: tuple[str, ...]) -> dict[str, frozenset[str]]:
    """`Enum::Variant` occurrences grouped by enum name."""
    table: dict[str, set[str]] = defaultdict(set)
    for index in range(len(body) - 2):
        if body[index + 1] != "::":
            continue
        prefix, variant = body[index], body[index + 2]
        if not is_ident(prefix) or not is_ident(variant):
            continue
        if not variant[0].isupper() or prefix in {"Self", "self", "crate", "super"}:
            continue
        table[prefix].add(variant)
    return {name: frozenset(found) for name, found in table.items()}


@dataclass
class FileScan:
    functions: list[Function] = field(default_factory=list)
    definitions: set[int] = field(default_factory=set)
    test_spans: list[tuple[int, int]] = field(default_factory=list)
    imports: set[int] = field(default_factory=set)
    references: Counter = field(default_factory=Counter)
    reexports: Counter = field(default_factory=Counter)
    aliases: dict[str, str] = field(default_factory=dict)
    tokens: int = 0


def scan_file(relative: Path, text: str) -> FileScan:
    rel = relative.as_posix()
    tokens, lines = lex(text)
    scan = FileScan(tokens=len(tokens))
    stack: list[tuple] = []
    start = 0
    index = 0
    import_start = None
    while index < len(tokens):
        token = tokens[index]
        if token == "{":
            kind, value, is_test = classify(tokens[start:index], stack)
            span = len(scan.test_spans)
            if is_test:
                scan.test_spans.append((index, len(tokens)))
            stack.append((kind, value, is_test, span))
            start = index + 1
        elif token == "}":
            if stack:
                kind, _value, is_test, span = stack.pop()
                if is_test:
                    scan.test_spans[span] = (scan.test_spans[span][0], index)
            start = index + 1
        elif token == ";":
            if import_start is not None:
                scan.imports.update(range(import_start, index))
                import_start = None
            start = index + 1
        elif token == "use" and is_import_head(tokens, start, index):
            import_start = start
        elif token == "fn" and index + 1 < len(tokens) and is_ident(tokens[index + 1]):
            parsed = parse_function(tokens, lines, index, start, stack)
            if parsed is not None:
                scan.definitions.add(index + 1)
                in_test = bool(stack and stack[-1][2]) or header_is_test(tokens[start:index])
                if in_test:
                    index = parsed["end"] + 1
                    start = index
                    continue
                body = parsed["body"]
                boilerplate = parsed["container"] in {"trait", "macro"} or (
                    parsed["impl_trait"] in BOILERPLATE_TRAITS
                )
                scan.functions.append(
                    Function(
                        path=rel,
                        name=parsed["name"],
                        line=parsed["line"],
                        visibility=parsed["visibility"],
                        body=body,
                        variants=variant_table(body),
                        boilerplate=boilerplate,
                        exposed=parsed["visibility"] == "pub",
                        container=parsed["container"],
                        impl_trait=parsed["impl_trait"],
                        crate=relative.parts[1],
                    )
                )
                index = parsed["end"] + 1
                start = index
                continue
        index += 1

    starts = [span[0] for span in scan.test_spans]
    for index, token in enumerate(tokens):
        if not is_ident(token) or index in scan.definitions or index in scan.imports:
            continue
        if in_spans(scan.test_spans, starts, index):
            continue
        if index in scan.reexports:
            continue
        scan.references[token] += 1
    for index in sorted(scan.imports):
        if is_ident(tokens[index]):
            scan.reexports[tokens[index]] += 1
        if (
            tokens[index] == "as"
            and index + 1 < len(tokens)
            and is_ident(tokens[index + 1])
            and is_ident(tokens[index - 1])
        ):
            # A renamed import is not an exemption: `X as Y` makes every `Y` a
            # reference to X.
            scan.aliases[tokens[index + 1]] = tokens[index - 1]
    return scan


def is_import_head(tokens: list[str], start: int, index: int) -> bool:
    """True when the `use` at `index` opens a statement, after any attributes.

    `#[doc(hidden)] pub use crate::shape_infer::{...}` is a re-export, and a
    re-export is not a caller.
    """
    position = start
    while position + 1 < index and tokens[position] == "#" and tokens[position + 1] == "[":
        depth = 0
        while position < index:
            if tokens[position] == "[":
                depth += 1
            elif tokens[position] == "]":
                depth -= 1
                if depth == 0:
                    position += 1
                    break
            position += 1
    head = tokens[position:index]
    if not head:
        return True
    if head[0] != "pub":
        return False
    return all(is_ident(token) or token == "::" for token in head[1:]) or all(
        is_ident(token) or token in {"(", ")", "::"} for token in head[1:]
    )


def in_spans(spans: list[tuple[int, int]], starts: list[int], index: int) -> bool:
    position = bisect.bisect_right(starts, index) - 1
    return position >= 0 and spans[position][0] <= index <= spans[position][1]


def crate_callers(root: Path) -> dict[str, frozenset[str]]:
    """Crate directory -> every workspace crate whose references can resolve to it.

    A reference in crate C can only target a definition in C or in a crate C
    depends on, so a `pub` definition referenced nowhere in this set has no
    production caller. This is what separates "leaf-crate public helper used from
    above" from "copy nothing can reach".
    """
    names: dict[str, str] = {}
    edges: dict[str, set[str]] = defaultdict(set)
    for manifest in sorted((root / "crates").glob("*/Cargo.toml")):
        directory = manifest.parent.name
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
        package = document.get("package", {}).get("name")
        if package:
            names[package] = directory
        edges[directory] |= {directory}
        for section in ("dependencies", "build-dependencies"):
            for dependency in document.get(section, {}):
                if dependency in names or (root / "crates" / dependency).is_dir():
                    edges[directory].add(names.get(dependency, dependency))
    # `names` is complete only after every manifest has been read.
    for directory, dependencies in list(edges.items()):
        edges[directory] = {
            names.get(dependency, dependency) for dependency in dependencies
        }
    closure: dict[str, set[str]] = {}
    for directory in edges:
        reached: set[str] = set()
        pending = [directory]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(edges.get(current, ()))
        closure[directory] = reached
    callers: dict[str, set[str]] = defaultdict(set)
    for directory, reachable in closure.items():
        for target in reachable:
            callers[target].add(directory)
    return {directory: frozenset(found) for directory, found in callers.items()}


def identifiers(tokens: list[str]) -> Counter:
    return Counter(token for token in tokens if is_ident(token))


FENCE = re.compile(r"^\s*(?://[/!])\s?```")
DOC = re.compile(r"^\s*(?://[/!])\s?(.*)$")


def doctest_identifiers(text: str) -> Counter:
    """Identifiers in the fenced code blocks of `///` and `//!` comments.

    A doctest is a use of an API, so a public item exercised by one is surface
    rather than a dead path.
    """
    found: Counter = Counter()
    inside = False
    for line in text.splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        if not inside:
            continue
        match = DOC.match(line)
        if match:
            found.update(identifiers(lex(match.group(1))[0]))
    return found


def collect_external(root: Path, doctests: bool) -> tuple[Counter, Counter, int]:
    """References from examples, benches, integration tests and doctests.

    These are consumers a library item must keep working, so they are evidence
    that the item is used even though they cannot own an implementation.
    """
    external: Counter = Counter()
    from_docs: Counter = Counter()
    files = 0
    for path in sorted((root / "crates").rglob("*.rs")):
        relative = path.relative_to(root)
        consumer = relative.parts[2] in {"tests", "examples", "benches"}
        if not consumer:
            continue
        files += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        external.update(identifiers(lex(text)[0]))
        if doctests:
            from_docs.update(doctest_identifiers(text))
    if doctests:
        for path in sorted((root / "crates").rglob("*.rs")):
            relative = path.relative_to(root)
            if relative.parts[2] != "src" or is_test_path(relative):
                continue
            from_docs.update(
                doctest_identifiers(path.read_text(encoding="utf-8", errors="replace"))
            )
    return external, from_docs, files


def collect(root: Path) -> tuple[list[Function], Counter, Counter, dict]:
    functions: list[Function] = []
    references: Counter = Counter()
    crate_references: dict[str, Counter] = defaultdict(Counter)
    reexports: Counter = Counter()
    aliases: dict[str, str] = {}
    stats = {"files": 0, "tokens": 0}
    for path in sorted((root / "crates").rglob("*.rs")):
        relative = path.relative_to(root)
        if relative.parts[2] != "src" or is_test_path(relative):
            continue
        scan = scan_file(relative, path.read_text(encoding="utf-8", errors="replace"))
        stats["files"] += 1
        stats["tokens"] += scan.tokens
        functions.extend(scan.functions)
        references.update(scan.references)
        crate_references[relative.parts[1]].update(scan.references)
        reexports.update(scan.reexports)
        aliases.update(scan.aliases)
    for alias, target in aliases.items():
        count = references.get(alias, 0)
        if count:
            references[target] += count
        for crate_counter in crate_references.values():
            crate_count = crate_counter.get(alias, 0)
            if crate_count:
                crate_counter[target] += crate_count
    return functions, references, crate_references, reexports, stats


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


@dataclass
class Finding:
    kind: str
    risk: float
    reason: str
    left: Function
    right: Function | None = None
    similarity: float = 0.0
    shared_variants: tuple[str, ...] = ()
    left_only: tuple[str, ...] = ()
    right_only: tuple[str, ...] = ()
    references: int = 0
    duplicate_name: bool = False
    reexported: bool = False
    allowlisted: str | None = None
    members: tuple[Function, ...] = ()
    subject: str = ""

    @property
    def identity(self) -> tuple[str, str, str]:
        if self.kind == "branches":
            return (self.kind, f"enum:{self.subject}", "")
        if self.right is None:
            return (self.kind, self.left.key, "")
        first, second = sorted((self.left.key, self.right.key))
        return (self.kind, first, second)


def shingles(body: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    if len(body) < SHINGLE:
        return frozenset()
    return frozenset(body[i : i + SHINGLE] for i in range(len(body) - SHINGLE + 1))


def body_twins(functions: list[Function]) -> list[Finding]:
    candidates = [f for f in functions if not f.boilerplate and f.size >= MIN_BODY_TOKENS]
    findings: list[Finding] = []
    seen: set[tuple[int, int]] = set()

    exact: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for position, function in enumerate(candidates):
        exact[function.body].append(position)
    for group in exact.values():
        for i, left_position in enumerate(group):
            for right_position in group[i + 1 :]:
                left, right = candidates[left_position], candidates[right_position]
                seen.add((left_position, right_position))
                findings.append(
                    Finding(
                        kind="body",
                        risk=min(left.size, right.size)
                        * (2.0 if left.exposed and right.exposed else 1.0),
                        reason="identical normalized body"
                        + (
                            "; same function name defined twice"
                            if left.name == right.name
                            else ""
                        ),
                        left=left,
                        right=right,
                        similarity=1.0,
                    )
                )

    approximate = [
        position for position, f in enumerate(candidates) if f.size >= MIN_SHINGLE_TOKENS
    ]
    sets = {position: shingles(candidates[position].body) for position in approximate}
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for position in approximate:
        for shingle in sets[position]:
            index[shingle].append(position)
    overlap: Counter = Counter()
    for members in index.values():
        for position, left_position in enumerate(members):
            for right_position in members[position + 1 :]:
                overlap[(left_position, right_position)] += 1
    for (left_position, right_position), shared in overlap.items():
        if (left_position, right_position) in seen:
            continue
        left, right = candidates[left_position], candidates[right_position]
        union = len(sets[left_position]) + len(sets[right_position]) - shared
        similarity = shared / union if union else 0.0
        if similarity < SIMILARITY:
            continue
        findings.append(
            Finding(
                kind="body",
                risk=min(left.size, right.size)
                * similarity
                * (2.0 if left.exposed and right.exposed else 1.0),
                reason=f"bodies {similarity:.0%} identical "
                f"({shared}/{union} shared {SHINGLE}-token shingles, "
                f"{abs(left.size - right.size)} tokens differ in size)",
                left=left,
                right=right,
                similarity=similarity,
            )
        )
    return findings


def branch_twins(functions: list[Function]) -> list[Finding]:
    """One finding per enum that several functions enumerate as a table.

    The unit is the enum, not the pair: eleven functions that each match on the
    same 52 variants are one maintenance fact (52 places to update), not 55
    findings, and 55 rows would crowd out everything else in a 30-row report.
    The row keeps the two largest tables as a concrete `file:line` pair.
    """
    by_enum: dict[str, list[Function]] = defaultdict(list)
    for function in functions:
        if function.boilerplate:
            continue
        for name, variants in function.variants.items():
            if len(variants) >= MIN_VARIANTS:
                by_enum[name].append(function)

    findings: list[Finding] = []
    for enum, members in by_enum.items():
        if len({function.path for function in members}) < 2:
            continue
        ranked = sorted(members, key=lambda f: (-len(f.variants[enum]), f.key))
        left, right = ranked[0], ranked[1]
        shared = left.variants[enum] & right.variants[enum]
        if len(shared) < MIN_VARIANTS:
            continue
        common = frozenset.intersection(*(f.variants[enum] for f in members))
        both_exposed = left.exposed and right.exposed
        findings.append(
            Finding(
                kind="branches",
                risk=len(shared) * VARIANT_WEIGHT * (2.0 if both_exposed else 1.0),
                reason=f"{len(members)} functions in {len({f.path for f in members})} files "
                f"enumerate `{enum}`: the two largest tables share {len(shared)} variants "
                f"and {len(common)} appear in every member",
                left=left,
                right=right,
                similarity=len(shared) / max(len(left.variants[enum]), len(right.variants[enum])),
                shared_variants=tuple(sorted(shared)),
                left_only=tuple(sorted(left.variants[enum] - right.variants[enum])),
                right_only=tuple(sorted(right.variants[enum] - left.variants[enum])),
                members=tuple(ranked),
                subject=enum,
            )
        )
    return findings


def dead_twins(
    functions: list[Function],
    references: Counter,
    crate_references: dict[str, Counter],
    reachable: dict[str, frozenset[str]],
    consumers: Counter,
    reexports: Counter,
) -> list[Finding]:
    """Definitions no production caller can reach.

    `reachable[name]` counts references from the crates that could resolve to the
    definition: its own crate for a private or `pub(crate)` item, every crate that
    can see it for a `pub` one. Two shapes are reported, both about competing
    copies:

    * the same free-function name defined more than once where at least one copy
      has no reachable caller (`#2399`: an exported helper nothing calls while an
      internal copy is what production uses);
    * a `pub(crate)` function with no production reference at all.

    A `pub` function with no in-repo caller and no second definition is public API
    surface, which public-surface policy owns, not a twin.
    """
    def reachable_count(function: Function) -> int:
        if function.visibility != "pub":
            return crate_references[function.crate].get(function.name, 0)
        return sum(
            crate_references[crate].get(function.name, 0)
            for crate in reachable.get(function.crate, (function.crate,))
        )

    free_definitions: dict[str, list[Function]] = defaultdict(list)
    for function in functions:
        if function.container == "free" and not function.boilerplate:
            free_definitions[function.name].append(function)

    findings: list[Finding] = []
    paired: set[str] = set()
    for name, definitions in free_definitions.items():
        if len(definitions) < 2:
            continue
        if all(function.visibility == "private" for function in definitions):
            # An uncalled private function is `dead_code`, which rustc reports;
            # the twin question is about a surface someone can reach.
            continue
        unreachable = [f for f in definitions if reachable_count(f) == 0]
        if not unreachable:
            continue
        ranked = sorted(definitions, key=lambda f: (-f.size, f.key))
        left, right = ranked[0], ranked[1]
        parts = [
            f"`{name}` is a free function defined {len(definitions)}x",
            f"{unreachable[0].visibility} copy in {unreachable[0].crate} "
            f"({location(unreachable[0])}) has no reachable caller",
        ]
        if len(unreachable) > 1:
            parts.append(f"{len(unreachable)} copies are unreachable")
        if reexports.get(name, 0):
            parts.append("the name is imported or re-exported")
        exposed = any(function.visibility == "pub" for function in definitions)
        findings.append(
            Finding(
                kind="dead",
                risk=(10 + min(left.size, 60)) * 2.0 * (2.0 if exposed else 1.0),
                reason="; ".join(parts),
                left=left,
                right=right,
                references=references.get(name, 0),
                duplicate_name=True,
                reexported=reexports.get(name, 0) > 0,
                members=tuple(ranked),
                subject=name,
            )
        )
        paired.add(name)

    for function in functions:
        if function.boilerplate or function.visibility == "private":
            continue
        if function.name in paired or function.container != "free":
            continue
        if reachable_count(function):
            continue
        if consumers.get(function.name, 0):
            # Used by an example, a bench, an integration test or a doctest: it is
            # a consumer-facing surface, not a path nothing takes.
            continue
        published = "imported or re-exported" if reexports.get(function.name, 0) else ""
        findings.append(
            Finding(
                kind="dead",
                risk=DEAD_ADVERTISED_WEIGHT + min(function.size, 60),
                reason=f"`{function.visibility}` with no production reference "
                "outside the library's own unit tests"
                + (f"; the name is {published}" if published else ""),
                left=function,
                references=0,
                reexported=bool(published),
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------- #


@dataclass
class AllowEntry:
    kind: str
    left: str
    right: str
    reason: str
    issue: str


def load_allowlist(
    root: Path, path: Path
) -> tuple[dict[tuple[str, str, str], AllowEntry], list[str]]:
    errors: list[str] = []
    entries: dict[tuple[str, str, str], AllowEntry] = {}
    full = path if path.is_absolute() else root / path
    if not full.exists():
        return entries, errors
    document = tomllib.loads(full.read_text(encoding="utf-8"))
    for raw in document.get("allow", []):
        kind = str(raw.get("kind", "")).strip()
        left = str(raw.get("left", "")).strip()
        right = str(raw.get("right", "")).strip()
        reason = " ".join(str(raw.get("reason", "")).split())
        issue = str(raw.get("issue", "")).strip()
        label = f"{kind}:{left}:{right or '-'}"
        if kind not in {"body", "branches", "dead"}:
            errors.append(f"{label}: unknown kind")
            continue
        if not left or (kind == "body" and not right):
            errors.append(
                f"{label}: needs a left" + (" and right" if kind == "body" else "")
            )
            continue
        if kind == "branches" and not left.startswith("enum:"):
            errors.append(f"{label}: a branches entry names an enum, left = \"enum:Name\"")
            continue
        if not reason or reason.lower() in PENDING_REASONS or len(reason) < 20:
            errors.append(f"{label}: needs a real reason, not {reason!r}")
            continue
        pair = tuple(sorted((left, right))) if kind == "body" else (left, "")
        entries[(kind, pair[0], pair[1])] = AllowEntry(kind, pair[0], pair[1], reason, issue)
    return entries, errors


def apply_allowlist(
    findings: list[Finding], entries: dict[tuple[str, str, str], AllowEntry]
) -> list[str]:
    matched: set[tuple[str, str, str]] = set()
    for finding in findings:
        entry = entries.get(finding.identity)
        if entry is not None:
            finding.allowlisted = (
                f"{entry.reason} ({entry.issue})" if entry.issue else entry.reason
            )
            matched.add(finding.identity)
    return [
        f"{key[0]}:{key[1]}" + (f":{key[2]}" if key[2] else "")
        for key in sorted(set(entries) - matched)
    ]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_markdown(report: dict, limit: int) -> str:
    lines = [
        "# Twin path report",
        "",
        f"Scanned {report['scanned']['files']} files, "
        f"{report['scanned']['tokens']} tokens, "
        f"{report['scanned']['functions']} production functions.",
        "",
        f"Findings: {report['total']} "
        f"(body {report['counts']['body']}, "
        f"branches {report['counts']['branches']}, "
        f"dead {report['counts']['dead']}); "
        f"not allowlisted: {report['unallowlisted']}.",
    ]
    if report["total"] > limit:
        lines.append(
            f"Showing the top {limit} by risk (scaled by public reachability); "
            f"{report['total'] - limit} lower-risk findings not shown."
        )
    lines.append("")
    for position, finding in enumerate(report["findings"][:limit], start=1):
        marker = " [allowlisted]" if finding["allowlisted"] else ""
        lines.append(
            f"## {position}. {finding['kind']}{marker} "
            f"risk={finding['risk']:.0f} "
            f"similarity={finding['similarity']:.2f} "
            f"tokens={finding['left']['tokens']}"
            + (f"/{finding['right']['tokens']}" if finding["right"] else "")
        )
        lines.append(
            f"- `{finding['left']['location']}` `{finding['left']['name']}` "
            f"({finding['left']['visibility']})"
        )
        if finding["right"]:
            lines.append(
                f"- `{finding['right']['location']}` `{finding['right']['name']}` "
                f"({finding['right']['visibility']})"
            )
        if finding["members"]:
            extra = finding["members"][2:]
            lines.append(
                f"- `{finding['enum']}` table, {len(finding['members'])} functions; "
                + (
                    "also: "
                    + ", ".join(
                        f"`{member['location']}` `{member['name']}`" for member in extra[:6]
                    )
                    + (f", +{len(extra) - 6} more" if len(extra) > 6 else "")
                    if extra
                    else "no other members"
                )
            )
        if finding["shared_variants"]:
            shown = ", ".join(finding["shared_variants"][:12])
            extra = len(finding["shared_variants"]) - 12
            lines.append(
                f"- common variants ({len(finding['shared_variants'])}): "
                f"{shown}" + (f", +{extra} more" if extra > 0 else "")
            )
            if finding["left_only"] or finding["right_only"]:
                lines.append(
                    f"- only in the left table: {', '.join(finding['left_only'][:8]) or '-'}; "
                    f"only in the right: {', '.join(finding['right_only'][:8]) or '-'}"
                )
        lines.append(f"- why: {finding['reason']}")
        if finding["allowlisted"]:
            lines.append(f"- allowlisted: {finding['allowlisted']}")
        lines.append("")
    if report["stale_allowlist"]:
        lines.append("## Stale allowlist entries (delete them)")
        lines.append("")
        lines.extend(f"- `{entry}`" for entry in report["stale_allowlist"])
        lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- `body`: the two functions implement the same computation. Exact means the "
        "token streams are identical after comments, whitespace and literals are "
        f"dropped; approximate means >= {SIMILARITY:.0%} shingle Jaccard with >= "
        f"{MIN_SHINGLE_TOKENS} tokens each."
    )
    lines.append(
        f"- `branches`: one row per enum that several functions enumerate as a table "
        f"(>= {MIN_VARIANTS} variants, at least two files). The unit is the enum: "
        "eleven functions matching the same 52 variants are one maintenance fact, not "
        "55 rows."
    )
    lines.append(
        "- `dead`: a `pub`/`pub(crate)` function with 0-1 production references. "
        "A definition, an import and a re-export are not callers."
    )
    lines.append(
        "- risk = size x similarity x public reachability (branches weigh "
        f"{VARIANT_WEIGHT} per shared variant). It orders the list; it is not a score."
    )
    return "\n".join(lines).rstrip() + "\n"


def render_json(report: dict, limit: int) -> str:
    """Machine output keeps every finding; only the Markdown report is capped."""
    payload = dict(report)
    payload["shown"] = min(limit, report["total"])
    payload["md_limit"] = limit
    return json.dumps(payload, indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def location(function: Function) -> str:
    return f"{function.path}:{function.line}"


def build_report(root: Path, allowlist_path: Path) -> tuple[dict, list[str]]:
    functions, references, crate_references, reexports, stats = collect(root)
    external, doctests, consumer_files = collect_external(root, doctests=True)
    stats["consumer_files"] = consumer_files
    consumers = external + doctests
    entries, errors = load_allowlist(root, allowlist_path)

    findings = [
        *body_twins(functions),
        *branch_twins(functions),
        *dead_twins(
            functions,
            references,
            crate_references,
            crate_callers(root),
            consumers,
            reexports,
        ),
    ]
    stale = apply_allowlist(findings, entries)
    findings.sort(key=lambda f: (-f.risk, f.kind, f.left.key, f.right.key if f.right else ""))

    counts = Counter(finding.kind for finding in findings)
    report = {
        "scanned": {**stats, "functions": len(functions)},
        "total": len(findings),
        "counts": {kind: counts.get(kind, 0) for kind in ("body", "branches", "dead")},
        "unallowlisted": sum(1 for finding in findings if finding.allowlisted is None),
        "stale_allowlist": stale,
        "allowlist": allowlist_path.as_posix(),
        "findings": [
            {
                "kind": finding.kind,
                "risk": round(finding.risk, 1),
                "similarity": round(finding.similarity, 3),
                "reason": finding.reason,
                "allowlisted": finding.allowlisted,
                "left": function_json(finding.left),
                "right": function_json(finding.right) if finding.right else None,
                "enum": finding.subject or None,
                "members": [function_json(member) for member in finding.members],
                "shared_variants": list(finding.shared_variants),
                "left_only": list(finding.left_only),
                "right_only": list(finding.right_only),
                "references": finding.references,
            }
            for finding in findings
        ],
    }
    return report, errors


def function_json(function: Function) -> dict:
    return {
        "location": location(function),
        "path": function.path,
        "name": function.name,
        "line": function.line,
        "visibility": function.visibility,
        "tokens": function.size,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report duplicated implementation paths (twin paths)."
    )
    parser.add_argument("--root-dir", default=ROOT, type=Path)
    parser.add_argument("--allowlist", default=ALLOWLIST, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when a finding has no allowlist entry.",
    )
    modes.add_argument(
        "--report",
        action="store_true",
        help="Report only; findings never fail the run (default).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of Markdown."
    )
    parser.add_argument(
        "--output", type=Path, help="Also write the report to this path."
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help=f"Rows to show (default {DEFAULT_LIMIT})."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root_dir.resolve()
    if not (root / "crates").is_dir():
        print(f"error: {root} has no crates/ directory", file=sys.stderr)
        return 2
    try:
        report, errors = build_report(root, args.allowlist)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for error in errors:
        print(f"allowlist error: {error}", file=sys.stderr)
    rendered = render_json(report, args.limit) if args.json else render_markdown(report, args.limit)
    sys.stdout.write(rendered)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")

    if errors:
        return 2
    if args.check and report["unallowlisted"]:
        print(
            f"twin-path check failed: {report['unallowlisted']} finding(s) have no "
            f"allowlist entry in {args.allowlist.as_posix()}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
