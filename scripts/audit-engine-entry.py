#!/usr/bin/env python3
"""Audit the surviving execution-engine entry surfaces against an allowlist.

Issue #2206 removed the session machinery: there is no `BackendSession`,
`BackendSessionHost`, `with_backend_session*`, `SessionEntryError`, held session,
execution scope or admission permit to audit. Execution now enters through a
caller-held engine, so the mechanisms that create or widen an execution surface
are:

  * `Engine implementation`           - `impl Engine for T`: a new operation
                                        surface over a caller-held engine
  * `ExecutionEngine implementation`  - `impl ExecutionEngine for T`: a new
                                        prepared-program target
  * `CUDA engine open`                - `CudaEngine::open`: opens a device and
                                        creates the engine's stream and memory
  * `CUDA raw scope`                  - `with_raw`: enters the raw CubeCL/CUDA
                                        context through the engine's stream

The definition of each mechanism may live in library code; every *call* in
library code must sit inside an allowlisted function or impl. The allowlist maps
`repository-relative-path::function-or-impl` entries to the reason that entry is
a legitimate boundary. A tracked mechanism must match a library definition or
site, so a rename cannot silently shrink coverage.

Test, benchmark and example sources (`tests/`, `benches/`, `examples/`,
`tests.rs`, `*_tests.rs`) are the boundary's users, not its owners, and are out
of scope. `--bless` rewrites the allowlist from the tree, keeping existing
reasons; `--self-test` runs the negative tests that prove the check is not
merely name matching.

Usage:
    python3 scripts/audit-engine-entry.py [--check|--bless|--self-test]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

IMPL_ENGINE = r"\bimpl\b[^\n{]*?(?<![A-Za-z0-9_])Engine\s+for\b[^\n{]*\{"
IMPL_EXECUTION_ENGINE = (
    r"\bimpl\b[^\n{]*?(?<![A-Za-z0-9_])ExecutionEngine\s+for\b[^\n{]*\{"
)

# mechanism -> (pattern, identifiers whose import alias can carry it)
MECHANISMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Engine implementation": (IMPL_ENGINE, ("Engine",)),
    "ExecutionEngine implementation": (IMPL_EXECUTION_ENGINE, ("ExecutionEngine",)),
    "CUDA engine open": (r"\bCudaEngine\s*::\s*open\b", ()),
    "CUDA raw scope": (r"\bwith_raw\b", ()),
}

# mechanism -> definition pattern. A mechanism with no call site in the tree is
# still tracked while its library definition exists.
DEFINITIONS: dict[str, str] = {
    "CUDA engine open": r"\bfn\s+open\s*\(",
    "CUDA raw scope": r"\bfn\s+with_raw\s*\(",
}

IMPL_MECHANISMS = frozenset({"Engine implementation", "ExecutionEngine implementation"})

# identifier -> mechanism, for every name that can carry a mechanism
MECHANISM_NAMES: dict[str, str] = {
    name: mechanism for mechanism, (_, names) in MECHANISMS.items() for name in names
}

ALLOWLIST = Path("scripts") / "engine-entry-allowlist.json"
SOURCE_ROOTS = ("crates", "ext")

USE_GROUP = re.compile(r"^\s*use\s+([A-Za-z_][\w:]*)\s*(?:::\s*\{([^}]*)\}|(?:::\s*([A-Za-z_]\w*))?\s*(?:as\s+([A-Za-z_]\w*))?)\s*;")
FN_START = re.compile(r"\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|async\s+|unsafe\s+)*fn\s+([A-Za-z_]\w*)")
IMPL_START = re.compile(r"\s*impl(?:<[^>]*>)?\s+(?:([A-Za-z_][\w:<>,\s]*?)\s+for\s+)?([A-Za-z_][\w:]*)")
# Other named containers, so entries inside them are keyed by name rather than
# `<module>`: traits (default methods), inline modules and `macro_rules!`.
CONTAINER_START = re.compile(
    r"\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:unsafe\s+)?trait|mod)\s+([A-Za-z_]\w*)|\s*macro_rules!\s*([A-Za-z_]\w*)"
)
ITEM_START = re.compile(
    r"\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|async|unsafe|extern\s+\"C\")\s+)*"
    r"(?:(?:fn|mod|impl|struct|enum|union|trait|type|use|static|const)\b|macro_rules!)"
)
# `#[cfg(test)]` / `#[cfg(all(test, ...))]`, but not `cfg(not(test))`.
CFG_TEST = re.compile(r"^\s*#\[cfg\((?:test|all\((?:[^()]|\([^()]*\))*\btest\b(?:[^()]|\([^()]*\))*\))\)\]")


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
    # (start, kind, name, depth at the item line, whether its body opened)
    stack: list[list] = []
    depth = 0
    position = 0
    for line in source.splitlines(keepends=True):
        at_line_start = depth
        inside_container = bool(stack) and stack[-1][1] == "impl"
        allowed = at_line_start == 0 or (
            inside_container and at_line_start == stack[-1][3] + 1
        )
        if allowed:
            impl_match = IMPL_START.match(line)
            container_match = None if impl_match else CONTAINER_START.match(line)
            fn_match = None if impl_match or container_match else FN_START.match(line)
            if impl_match:
                # Prefer the implemented type over the trait when both exist, so
                # `impl Engine for CudaEngine` is keyed by `CudaEngine`.
                name = (impl_match.group(2) or impl_match.group(1) or "").strip()
                stack.append([position, "impl", name, at_line_start, False])
            elif container_match:
                name = container_match.group(1) or container_match.group(2)
                stack.append([position, "impl", name, at_line_start, False])
            elif fn_match:
                stack.append([position, "fn", fn_match.group(1), at_line_start, False])
        for char in line:
            if char == "{":
                if stack and not stack[-1][4] and depth == stack[-1][3]:
                    stack[-1][4] = True
                depth += 1
            elif char == "}":
                depth -= 1
                while stack and depth == stack[-1][3] and stack[-1][4]:
                    open_at, kind, name, _, _ = stack.pop()
                    spans.append((open_at, position + len(line), kind, name))
            elif char == ";" and stack and not stack[-1][4] and depth == stack[-1][3]:
                # A bodiless item (trait method signature, `mod x;`) ends here.
                stack.pop()
        position += len(line)
    for open_at, kind, name, _, _ in stack:
        spans.append((open_at, len(source), kind, name))
    return spans


def enclosing_function(source: str, offset: int, spans=None) -> str:
    """`Impl::fn`, the impl type for an impl header, or `fn` for an item."""
    if spans is None:
        spans = item_spans(source)
    enclosing = [span for span in spans if span[0] <= offset < span[1]]
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
        # An entry in an impl header is keyed by the implemented type. An entry
        # in a bare module still has no named owner.
        return impl.split("<")[0].strip() if impl else "<module>"
    if impl:
        return f"{impl.split('<')[0].strip()}::{function}"
    return function


def _is_entry_use(source: str, match: re.Match[str]) -> bool:
    """True when the match is a call, a path segment, a generic or an impl header."""
    if match.group(0).rstrip().endswith("{"):
        return True
    # The name in `fn name(` is a definition, not a use; a definition that
    # enters is caught by the mechanism its body uses.
    if re.search(r"\bfn\s+$", source[max(0, match.start() - 8) : match.start()]):
        return False
    rest = source[match.end() : match.end() + 4].lstrip()
    return rest.startswith(("(", "::", "<"))


def strip_cfg_test_items(source: str) -> str:
    """Blank every `#[cfg(test)]` item: test code is the boundary's user."""
    out = list(source)
    lines = source.splitlines(keepends=True)
    offsets = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    index = 0
    while index < len(lines):
        if not CFG_TEST.match(lines[index]):
            index += 1
            continue
        # Only whole items are test code; a `#[cfg(test)]` match arm, field or
        # statement is left in place.
        following = index + 1
        while following < len(lines) and lines[following].lstrip().startswith("#["):
            following += 1
        if following >= len(lines) or not ITEM_START.match(lines[following]):
            index += 1
            continue
        start = offsets[index]
        # Skip further attributes, then take the item up to its `;` or the
        # brace that closes its body.
        cursor = start + len(lines[index])
        depth = 0
        end = len(source)
        opened = False
        while cursor < len(source):
            char = source[cursor]
            if char == "{":
                depth += 1
                opened = True
            elif char == "}":
                depth -= 1
                if opened and depth == 0:
                    end = cursor + 1
                    break
            elif char == ";" and not opened:
                end = cursor + 1
                break
            cursor += 1
        for k in range(start, end):
            if out[k] != "\n":
                out[k] = " "
        while index < len(lines) and offsets[index] < end:
            index += 1
    return "".join(out)


def scan_source(text: str) -> list[tuple[str, str]]:
    """(mechanism, enclosing function or impl type) for every entry occurrence."""
    source = strip_cfg_test_items(strip_comments_and_strings(text))
    spans = item_spans(source)
    aliases = alias_map(source)
    found: set[tuple[str, str]] = set()
    for mechanism, (pattern, canonical) in MECHANISMS.items():
        needles = [pattern]
        for alias, target in aliases.items():
            if target not in canonical:
                continue
            if mechanism in IMPL_MECHANISMS:
                needles.append(
                    rf"\bimpl\b[^\n{{]*?(?<![A-Za-z0-9_]){re.escape(alias)}\s+for\b[^\n{{]*\{{"
                )
            else:
                needles.append(rf"\b{re.escape(alias)}\b")
        for needle in needles:
            for match in re.finditer(needle, source):
                if not _is_entry_use(source, match):
                    continue
                found.add((mechanism, enclosing_function(source, match.start(), spans)))
    return sorted(found)


# Build output (`ext/*/target/`) is not source either.
NON_LIBRARY = ("/tests/", "/benches/", "/examples/", "/target/")


def is_library_source(path: Path) -> bool:
    """Test, benchmark and example code is the boundary's user, not its owner."""
    parts = "/" + "/".join(path.parts) + "/"
    if any(marker in parts for marker in NON_LIBRARY):
        return False
    return not (path.name == "tests.rs" or path.name.endswith("_tests.rs"))


def defined_mechanisms(text: str) -> set[str]:
    """Mechanisms whose entry is defined (`fn name`, impl header) in this source."""
    source = strip_comments_and_strings(text)
    defined: set[str] = set()
    for mechanism, pattern in DEFINITIONS.items():
        if re.search(pattern, source):
            defined.add(mechanism)
    for mechanism, (pattern, _) in MECHANISMS.items():
        if mechanism not in IMPL_MECHANISMS:
            continue
        source_without_tests = strip_cfg_test_items(source)
        if re.search(pattern, source_without_tests):
            defined.add(mechanism)
    return defined


def defined_in_repo(repo: Path) -> set[str]:
    defined: set[str] = set()
    for root in SOURCE_ROOTS:
        base = repo / root
        if base.is_dir():
            for path in sorted(base.rglob("*.rs")):
                if is_library_source(path.relative_to(repo)):
                    defined |= defined_mechanisms(path.read_text())
    return defined


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


def load_allowlist(repo: Path) -> dict[str, dict[str, str]]:
    path = repo / ALLOWLIST
    if not path.is_file():
        return {name: {} for name in MECHANISMS}
    return json.loads(path.read_text())


def check(repo: Path) -> int:
    # The negative tests run with every check, so a broken matcher cannot pass
    # the gate by reporting nothing.
    if self_test() != 0:
        return 1
    inventory = scan_repo(repo)
    defined = defined_in_repo(repo)
    # A tree without source roots (a script fixture) has nothing to match.
    has_sources = any((repo / root).is_dir() for root in SOURCE_ROOTS)
    allowlist = load_allowlist(repo)
    failed = False
    for mechanism in allowlist:
        if mechanism not in MECHANISMS:
            print(f"{mechanism}: allowlist names an untracked mechanism")
            failed = True
    for mechanism, sites in inventory.items():
        allowed = allowlist.get(mechanism, {})
        if not sites and mechanism not in defined and has_sources:
            print(f"{mechanism}: tracked mechanism matches no library definition or site; update its pattern")
            failed = True
        for site in sites:
            if site not in allowed:
                print(f"{mechanism}: unallowlisted engine entry at {site}")
                failed = True
            elif not str(allowed[site]).strip() or str(allowed[site]).startswith("TODO"):
                print(f"{mechanism}: allowlist entry has no reason: {site}")
                failed = True
            elif str(allowed[site]).startswith("PENDING"):
                # A PENDING reason marks a known-illegitimate entry awaiting
                # removal; it may not survive into a merged allowlist.
                print(f"{mechanism}: allowlist entry is pending removal: {site}")
                failed = True
        for site in sorted(set(allowed) - set(sites)):
            print(f"{mechanism}: allowlist entry is stale: {site}")
            failed = True
    return 1 if failed else 0


def bless(repo: Path) -> int:
    inventory = scan_repo(repo)
    previous = load_allowlist(repo)
    blessed: dict[str, dict[str, str]] = {}
    for mechanism, sites in inventory.items():
        reasons = previous.get(mechanism, {})
        if isinstance(reasons, list):
            reasons = {}
        blessed[mechanism] = {
            site: reasons.get(site, "TODO: state why this is a legitimate entry")
            for site in sites
        }
    target = repo / ALLOWLIST
    target.write_text(json.dumps(blessed, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {target.relative_to(repo)}")
    return 0


SELF_TEST_ENGINE_IMPL = """
use tenferro_tensor::Engine;

impl Engine for FakeEngine {
    fn name(&self) -> &'static str {
        "fake"
    }
}

impl ExtensionEngine for OtherEngine {}


impl tenferro_runtime::ExecutionEngine for FakeEngine {
    fn target(&self) -> Target {
        Target::new()
    }
}
"""

SELF_TEST_ALIASED_IMPL = """
use tenferro_tensor::Engine as Exec;

impl Exec for AliasedEngine {}
"""

SELF_TEST_CUDA_OPEN = """
fn open_engine(device: CudaDeviceId) -> Result<CudaEngine> {
    CudaEngine::open(device)
}
"""

SELF_TEST_RAW_SCOPE = """
fn launch(engine: &mut CudaEngine) -> Result<()> {
    engine.with_raw("bench.op", |raw| {
        let _ = raw.stream();
        Ok(())
    })
}

impl CudaEngine {
    fn with_raw_on_stream(&mut self) {}
}
"""

SELF_TEST_CFG_TEST = """
#[cfg(test)]
mod tests {
    fn helper(engine: &mut CudaEngine) {
        engine.with_raw("test.op", |_| Ok(()));
    }
}

#[cfg(not(test))]
fn production(engine: &mut CudaEngine) {
    engine.with_raw("production.op", |_| Ok(()));
}
"""

SELF_TEST_UNRELATED = """
struct Other;

fn build() -> Other {
    Other {}
}

fn open_stream(stream: &mut Stream) {
    stream.with_raw_on_stream();
}
"""


def self_test() -> int:
    failures: list[str] = []

    def sites(snippet: str) -> set[tuple[str, str]]:
        return set(scan_source(snippet))

    impls = sites(SELF_TEST_ENGINE_IMPL)
    if ("Engine implementation", "FakeEngine") not in impls:
        failures.append("an `impl Engine for T` header must be reported with its type")
    if ("ExecutionEngine implementation", "FakeEngine") not in impls:
        failures.append("an `impl ExecutionEngine for T` header must be reported with its type")
    if ("Engine implementation", "OtherEngine") in impls:
        failures.append("an `ExtensionEngine` impl must not be reported as the Engine trait")

    aliased = sites(SELF_TEST_ALIASED_IMPL)
    if ("Engine implementation", "AliasedEngine") not in aliased:
        failures.append("a renamed `Engine` import must still be traced")

    opened = sites(SELF_TEST_CUDA_OPEN)
    if ("CUDA engine open", "open_engine") not in opened:
        failures.append("a `CudaEngine::open` call must be reported with its function")

    raw = sites(SELF_TEST_RAW_SCOPE)
    if ("CUDA raw scope", "launch") not in raw:
        failures.append("a `with_raw` call must be reported with its function")
    if any(mechanism == "CUDA raw scope" and site != "launch" for mechanism, site in raw):
        failures.append("`with_raw_on_stream` must not be reported as the raw entry")

    cfg_test = sites(SELF_TEST_CFG_TEST)
    if ("CUDA raw scope", "helper") in cfg_test:
        failures.append("a `#[cfg(test)]` item must be out of scope")
    if ("CUDA raw scope", "production") not in cfg_test:
        failures.append("a `#[cfg(not(test))]` item must stay in scope")

    if sites(SELF_TEST_UNRELATED):
        failures.append("an unrelated struct or method must not be reported")

    for name in ("src/backend/tests.rs", "src/typed_eager_tests.rs", "tests/integration.rs"):
        if is_library_source(Path("crates/x") / name):
            failures.append(f"test source {name} must be out of scope")
    if not is_library_source(Path("crates/x/src/backend.rs")):
        failures.append("library source must stay in scope")

    for failure in failures:
        print(f"self-test failure: {failure}")
    if failures:
        return 1
    print(
        "self-test passed: engine/execution-engine impls, aliased impl, CUDA open, "
        "raw scope, cfg(test), unrelated-method cases"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="fail on unallowlisted entries (default)")
    group.add_argument("--bless", action="store_true", help="rewrite the allowlist from the tree")
    group.add_argument("--self-test", action="store_true", help="run the negative tests")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.self_test:
        return self_test()
    if args.bless:
        return bless(repo)
    return check(repo)


if __name__ == "__main__":
    sys.exit(main())
