#!/usr/bin/env python3
"""Audit backend-session entry mechanisms against a function-level allowlist.

Session entry is defined by the *mechanism*, not by the exported name: a
`use ... as alias` import or a fully-qualified path must be caught the same way
as the plain identifier.

Mechanisms tracked (issue #1926 / umbrella #1929):

  * `default_backend_session`            - the deleted backend-as-session factory
  * `with_session_entry_guard`           - the portable nested-entry guard
  * `install_with_pool_context`          - CPU owner operation entry
  * `install_with_pool_context_fresh`    - CPU owner operation entry (fresh output)
  * `install_with_indexed_pool_context`  - CPU indexed owner operation entry
  * `run_backend_session_cached`         - CPU session construction
  * `CpuExecSession construction`        - CPU session construction
  * `CudaExecSession construction`       - CUDA session construction
  * `WebGpuExecSession construction`     - WebGPU session construction

Every occurrence must sit inside an allowlisted function; the allowlist is keyed
by mechanism and holds `repository-relative-path::function` entries. `--bless`
rewrites it from the current tree, `--self-test` runs the alias and boundary
negative tests that prove the check is not merely name matching.

Usage:
    python3 scripts/audit-session-entry.py [--check|--bless|--self-test]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# mechanism -> (pattern, identifiers whose import can rename it)
MECHANISMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "default_backend_session": (r"\bdefault_backend_session\b", ("default_backend_session",)),
    "with_session_entry_guard": (r"\bwith_session_entry_guard\b", ("with_session_entry_guard",)),
    "install_with_pool_context": (r"\binstall_with_pool_context\b", ("install_with_pool_context",)),
    "install_with_pool_context_fresh": (r"\binstall_with_pool_context_fresh\b", ("install_with_pool_context_fresh",)),
    "install_with_indexed_pool_context": (r"\binstall_with_indexed_pool_context(?:_unmarked)?\b", ("install_with_indexed_pool_context", "install_with_indexed_pool_context_unmarked")),
    "run_backend_session_cached": (r"\brun_backend_session_cached\b", ("run_backend_session_cached",)),
    "CpuExecSession construction": (r"\bCpuExecSession\s*\{", ("CpuExecSession",)),
    "CudaExecSession construction": (r"\bCudaExecSession\s*\{", ("CudaExecSession",)),
    "WebGpuExecSession construction": (r"\bWebGpuExecSession\s*\{", ("WebGpuExecSession",)),
}

# identifier -> mechanism, for every name that can carry a mechanism
MECHANISM_NAMES: dict[str, str] = {
    name: mechanism for mechanism, (_, names) in MECHANISMS.items() for name in names
}

ALLOWLIST = Path("scripts") / "session-entry-allowlist.json"
SOURCE_ROOTS = ("crates", "ext")

USE_GROUP = re.compile(r"^\s*use\s+([A-Za-z_][\w:]*)\s*(?:::\s*\{([^}]*)\}|(?:::\s*([A-Za-z_]\w*))?\s*(?:as\s+([A-Za-z_]\w*))?)\s*;")
FN_START = re.compile(r"\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|async\s+|unsafe\s+)*fn\s+([A-Za-z_]\w*)")
IMPL_START = re.compile(r"\s*impl(?:<[^>]*>)?\s+(?:([A-Za-z_][\w:<>,\s]*?)\s+for\s+)?([A-Za-z_][\w:]*)")


def strip_comments_and_strings(text: str) -> str:
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if text[i] == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            for k in range(i, min(j, n)):
                out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


def alias_map(source: str) -> dict[str, str]:
    """Map every imported identifier to the mechanism it can name."""
    aliases: dict[str, str] = {}
    for line in source.splitlines():
        match = USE_GROUP.match(line)
        if not match:
            continue
        root, group, single, rename = match.groups()
        if group is not None:
            for item in group.split(","):
                item = item.strip()
                if not item:
                    continue
                parts = item.split(" as ")
                name = parts[0].strip().split("::")[-1]
                target = parts[1].strip() if len(parts) > 1 else name
                if name in MECHANISM_NAMES:
                    aliases[target] = name
        else:
            full = f"{root}::{single}" if single else root
            name = full.split("::")[-1]
            target = rename or name
            if name in MECHANISM_NAMES:
                aliases[target] = name
    return aliases


def item_spans(source: str) -> list[tuple[int, int, str, str]]:
    """(start, end, kind, name) for every function and impl/trait/mod block."""
    spans: list[tuple[int, int, str, str]] = []
    stack: list[tuple[int, str, str, int]] = []
    depth = 0
    position = 0
    for line in source.splitlines(keepends=True):
        at_line_start = depth
        inside_container = bool(stack) and stack[-1][1] in ("impl", "trait", "mod")
        allowed = at_line_start == 0 or (
            inside_container and at_line_start == stack[-1][3] + 1
        )
        if allowed:
            impl_match = IMPL_START.match(line)
            fn_match = None if impl_match else FN_START.match(line)
            if impl_match:
                name = (impl_match.group(1) or impl_match.group(2) or "").strip()
                stack.append((position, "impl", name, at_line_start))
            elif fn_match:
                stack.append((position, "fn", fn_match.group(1), at_line_start))
        for char in line:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                while stack and depth == stack[-1][3]:
                    open_at, kind, name, _ = stack.pop()
                    spans.append((open_at, position + len(line), kind, name))
        position += len(line)
    for open_at, kind, name, _ in stack:
        spans.append((open_at, len(source), kind, name))
    return spans


def enclosing_function(source: str, offset: int) -> str:
    """`Impl::fn` or `fn` for the innermost item containing `offset`."""
    enclosing = [span for span in item_spans(source) if span[0] <= offset < span[1]]
    if not enclosing:
        return "<module>"
    enclosing.sort(key=lambda span: span[0])
    function = None
    impl = None
    for _, _, kind, name in enclosing:
        if kind == "impl":
            impl = name
        else:
            function = name
    if function is None:
        return "<module>"
    if impl:
        return f"{impl.split('<')[0].strip()}::{function}"
    return function


def _is_entry_use(source: str, match: re.Match[str]) -> bool:
    """True when the match is a call, a path segment, a generic or a literal."""
    if match.group(0).rstrip().endswith("{"):
        return True
    rest = source[match.end() : match.end() + 4].lstrip()
    return rest.startswith(("(", "::", "<"))


def scan_source(text: str) -> list[tuple[str, str]]:
    """(mechanism, enclosing function) for every entry-mechanism occurrence."""
    source = strip_comments_and_strings(text)
    aliases = alias_map(source)
    found: set[tuple[str, str]] = set()
    for mechanism, (pattern, canonical) in MECHANISMS.items():
        needles = [pattern]
        method = mechanism.split("::")[-1] if "::" in mechanism else None
        literal = pattern.rstrip().endswith(r"\{")
        for alias, target in aliases.items():
            if target not in canonical:
                continue
            if literal:
                needles.append(rf"\b{re.escape(alias)}\s*\{{")
            elif method is not None:
                needles.append(rf"\b{re.escape(alias)}\s*::\s*{re.escape(method)}\b")
            else:
                needles.append(rf"\b{re.escape(alias)}\b")
        for needle in needles:
            for match in re.finditer(needle, source):
                if not _is_entry_use(source, match):
                    continue
                found.add((mechanism, enclosing_function(source, match.start())))
    return sorted(found)


NON_LIBRARY = ("/tests/", "/benches/", "/examples/")


def is_library_source(path: Path) -> bool:
    """Test, benchmark and example code is the boundary's user, not its owner."""
    parts = "/" + "/".join(path.parts) + "/"
    return not any(marker in parts for marker in NON_LIBRARY)


def scan_repo(repo: Path) -> dict[str, list[str]]:
    inventory: dict[str, set[str]] = {name: set() for name in MECHANISMS}
    for root in SOURCE_ROOTS:
        base = repo / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.rs")):
            if not is_library_source(path.relative_to(repo)):
                continue
            for mechanism, function in scan_source(path.read_text()):
                inventory[mechanism].add(f"{path.relative_to(repo)}::{function}")
    return {name: sorted(sites) for name, sites in inventory.items()}


def load_allowlist(repo: Path) -> dict[str, list[str]]:
    path = repo / ALLOWLIST
    if not path.is_file():
        return {name: [] for name in MECHANISMS}
    return json.loads(path.read_text())


def check(repo: Path) -> int:
    # The negative tests run with every check, so a broken matcher cannot pass
    # the gate by reporting nothing.
    if self_test() != 0:
        return 1
    inventory = scan_repo(repo)
    allowlist = load_allowlist(repo)
    failed = False
    for mechanism, sites in inventory.items():
        allowed = set(allowlist.get(mechanism, []))
        for site in sites:
            if site not in allowed:
                print(f"{mechanism}: unallowlisted session entry at {site}")
                failed = True
        for site in sorted(allowed - set(sites)):
            print(f"{mechanism}: allowlist entry is stale: {site}")
            failed = True
    return 1 if failed else 0


def bless(repo: Path) -> int:
    inventory = scan_repo(repo)
    target = repo / ALLOWLIST
    target.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(f"wrote {target.relative_to(repo)}")
    return 0


SELF_TEST_HIDDEN_ALIAS = """
use tenferro_tensor::default_backend_session as session_entry;

fn hidden(backend: &mut B) {
    session_entry(backend, |_| ());
}
"""

SELF_TEST_ALLOWLISTED = """
use tenferro_tensor::with_session_entry_guard;

fn run_backend_session_cached(&self) {
    with_session_entry_guard(|| ());
}
"""

SELF_TEST_QUALIFIED = """
fn install(&self) {
    let _ = self.install_with_pool_context(|context, buffers| ());
}
"""

SELF_TEST_UNRELATED_STRUCT = """
struct Other;

fn build() -> Other {
    Other {}
}
"""

SELF_TEST_GROUP_IMPORT = """
use tenferro_cpu::exec_session::{CpuExecSession as Session};

fn build() -> Session<'static> {
    Session {}
}
"""


def self_test() -> int:
    failures: list[str] = []

    def sites(snippet: str) -> set[tuple[str, str]]:
        return set(scan_source(snippet))

    hidden = sites(SELF_TEST_HIDDEN_ALIAS)
    if ("default_backend_session", "hidden") not in hidden:
        failures.append("a renamed import must still be traced to its mechanism")

    allowlisted = sites(SELF_TEST_ALLOWLISTED)
    if ("with_session_entry_guard", "run_backend_session_cached") not in allowlisted:
        failures.append("an allowlisted boundary must be reported with its function")

    qualified = sites(SELF_TEST_QUALIFIED)
    if ("install_with_pool_context", "install") not in qualified:
        failures.append("a method call must be reported")

    grouped = sites(SELF_TEST_GROUP_IMPORT)
    if ("CpuExecSession construction", "build") not in grouped:
        failures.append("a grouped `as` import must be traced")

    unrelated = sites(SELF_TEST_UNRELATED_STRUCT)
    if unrelated:
        failures.append("an unrelated struct literal must not be reported")

    for failure in failures:
        print(f"self-test failure: {failure}")
    if failures:
        return 1
    print("self-test passed: alias, boundary, method and grouped-import cases")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="fail on unallowlisted entries (default)")
    group.add_argument("--bless", action="store_true", help="rewrite the allowlist from the tree")
    group.add_argument("--self-test", action="store_true", help="run the alias/boundary negative tests")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.self_test:
        return self_test()
    if args.bless:
        return bless(repo)
    return check(repo)


if __name__ == "__main__":
    sys.exit(main())
