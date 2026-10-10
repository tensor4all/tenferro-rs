#!/usr/bin/env python3
"""Audit backend-session entry mechanisms against a function-level allowlist.

Session entry is defined by the *mechanism*, not by the exported name: a
`use ... as alias` import or a fully-qualified path must be caught the same way
as the plain identifier.

Mechanisms tracked (issue #1926 / umbrella #1929, hardened by #1946):

  * `CPU execution admission`            - `execution_admission()`: the permit every
                                           CPU entry acquires
  * `run_backend_session_cached`         - CPU session construction
  * `with_execution_scope`               - CPU execution-scope entry (holds a permit)
  * `with_session_entry_guard`           - the portable nested-entry guard
  * `CpuExecSession construction`        - CPU session construction
  * `CudaExecSession construction`       - CUDA session construction
  * `WebGpuExecSession construction`     - WebGPU session construction
  * `backend session entry`              - a library call of `with_backend_session`
                                           / `with_backend_session_cached`
  * `eager session entry`                - a library call of `with_eager_session` /
                                           `with_execution_session`
  * `eager extension context entry`      - a library call that hands an extension the
                                           owner context (`with_extension_*_context`,
                                           `erased_context`)

Retired mechanisms must stay absent from library code:

  * `default_backend_session`            - the deleted backend-as-session factory
  * `with_evaluation_scope`              - execution-scope hook, not implemented (A2)
  * `CPU backend install wrappers`       - `try_install` / `install_with_pool*`, the
                                           deleted owner-side entry helpers (#1946 F6)

Every occurrence must sit inside an allowlisted function; the allowlist is keyed
by mechanism and maps `repository-relative-path::function` entries to the reason
that entry is a legitimate boundary. Every tracked (non-retired) mechanism must
match at least one library site, so a rename cannot silently shrink coverage.
Test, benchmark and example sources (`tests/`, `benches/`, `examples/`,
`tests.rs`, `*_tests.rs`) are the boundary's users, not its owners.
`--bless` rewrites the allowlist from the tree, keeping existing reasons;
`--self-test` runs the negative tests that prove the check is not merely name
matching.

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
    "CPU execution admission": (r"\bexecution_admission\b", ("execution_admission",)),
    "CPU backend install wrappers": (
        r"\b(?:try_install|install_with_pool(?:_context)?(?:_unmarked)?)\b",
        (
            "try_install",
            "install_with_pool",
            "install_with_pool_unmarked",
            "install_with_pool_context",
            "install_with_pool_context_unmarked",
        ),
    ),
    "run_backend_session_cached": (r"\brun_backend_session_cached\b", ("run_backend_session_cached",)),
    "with_execution_scope": (r"\bwith_execution_scope\b", ("with_execution_scope",)),
    "with_evaluation_scope": (r"\bwith_evaluation_scope\b", ("with_evaluation_scope",)),
    "CpuExecSession construction": (r"\bCpuExecSession\s*\{", ("CpuExecSession",)),
    "held session entry": (
        r"\b(?:open_session|adopt_held_permit)\b",
        ("open_session", "adopt_held_permit"),
    ),
    "CpuHeldSession construction": (r"\bCpuHeldSession\s*\{", ("CpuHeldSession",)),
    "CudaHeldSession construction": (r"\bCudaHeldSession\s*\{", ("CudaHeldSession",)),
    "held session marker entry": (r"\bHeldSessionMarker::enter\b", ("HeldSessionMarker",)),
    "CudaExecSession construction": (r"\bCudaExecSession\s*\{", ("CudaExecSession",)),
    "WebGpuExecSession construction": (r"\bWebGpuExecSession\s*\{", ("WebGpuExecSession",)),
    "backend session entry": (
        r"\bwith_backend_session(?:_cached)?\b",
        ("with_backend_session", "with_backend_session_cached"),
    ),
    "eager session entry": (
        r"\bwith_(?:eager|execution)_session\b",
        ("with_eager_session", "with_execution_session"),
    ),
    "eager extension context entry": (
        r"\b(?:with_extension_(?:execution|erased)_context|erased_context)\b",
        ("with_extension_execution_context", "with_extension_erased_context", "erased_context"),
    ),
}

# Retired mechanisms: any library occurrence fails, and they are exempt from
# the "must match at least one site" self-check.
RETIRED: frozenset[str] = frozenset(
    {"default_backend_session", "with_evaluation_scope", "CPU backend install wrappers"}
)

# identifier -> mechanism, for every name that can carry a mechanism
MECHANISM_NAMES: dict[str, str] = {
    name: mechanism for mechanism, (_, names) in MECHANISMS.items() for name in names
}

ALLOWLIST = Path("scripts") / "session-entry-allowlist.json"
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
                name = (impl_match.group(1) or impl_match.group(2) or "").strip()
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
    """`Impl::fn` or `fn` for the innermost item containing `offset`."""
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
        return "<module>"
    if impl:
        return f"{impl.split('<')[0].strip()}::{function}"
    return function


def _is_entry_use(source: str, match: re.Match[str]) -> bool:
    """True when the match is a call, a path segment, a generic or a literal."""
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
    """(mechanism, enclosing function) for every entry-mechanism occurrence."""
    source = strip_cfg_test_items(strip_comments_and_strings(text))
    spans = item_spans(source)
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
    """Mechanisms whose entry function is defined (`fn name`) in this source."""
    source = strip_comments_and_strings(text)
    defined: set[str] = set()
    for mechanism, (_, names) in MECHANISMS.items():
        if any(re.search(rf"\bfn\s+{re.escape(name)}\b", source) for name in names):
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
        if mechanism in RETIRED:
            for site in sites:
                print(f"{mechanism}: retired mechanism used at {site}")
                failed = True
            if allowed:
                print(f"{mechanism}: a retired mechanism cannot be allowlisted")
                failed = True
            continue
        if not sites and mechanism not in defined and has_sources:
            print(f"{mechanism}: tracked mechanism matches no library definition or site; update its pattern")
            failed = True
        for site in sites:
            if site not in allowed:
                print(f"{mechanism}: unallowlisted session entry at {site}")
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
        if mechanism in RETIRED:
            blessed[mechanism] = {}
            continue
        reasons = previous.get(mechanism, {})
        if isinstance(reasons, list):
            reasons = {}
        blessed[mechanism] = {site: reasons.get(site, "TODO: state why this is a legitimate entry") for site in sites}
    target = repo / ALLOWLIST
    target.write_text(json.dumps(blessed, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
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
    let _ = self.install_with_pool_context_unmarked(|context, buffers| ());
}
"""

SELF_TEST_CFG_TEST = """
#[cfg(test)]
mod tests {
    fn helper(&self) {
        let _ = self.runtime().with_eager_session(|session| session.run());
    }
}

#[cfg(not(test))]
fn production(&self) {
    let _ = self.runtime().with_eager_session(|session| session.run());
}
"""

SELF_TEST_CFG_TEST_ARM = """
fn dispatch(&self) {
    match self {
        #[cfg(test)]
        Self::Recording(_) => None,
        Self::Cpu(_) => Some(1),
    };
}

fn after(&self) {
    let _ = self.runtime().with_eager_session(|session| session.run());
}
"""

SELF_TEST_TRAIT_DEFAULT = """
pub trait Host {
    fn with_backend_session_cached(&mut self) {
        self.with_backend_session(|s| ());
    }
}
"""

SELF_TEST_EAGER_ENTRY = """
fn einsum(&self) {
    let _ = self.runtime().with_eager_session(|session| session.run());
}
"""

SELF_TEST_HELD_ENTRY = """
fn open_session(&self) -> Held {
    let permit = self.acquire();
    self.adopt_held_permit(permit)
}

fn adopt_held_permit(&self, permit: Permit) -> Held {
    let marker = HeldSessionMarker::enter("backend");
    Held {
        state: CpuHeldSession { permit, marker },
    }
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

SELF_TEST_SCOPE_ENTRY = """
use tenferro_cpu::with_execution_scope as scope_entry;
use tenferro_tensor::with_evaluation_scope as evaluation_scope;

fn hidden_scope(backend: &mut B) {
    scope_entry(backend, || ());
    evaluation_scope(backend, |_| ());
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
    if ("CPU backend install wrappers", "install") not in qualified:
        failures.append("an `_unmarked` install helper call must be reported")

    held = sites(SELF_TEST_HELD_ENTRY)
    if ("held session entry", "open_session") not in held:
        failures.append("a held-session entry call must be reported with its function")
    if ("CpuHeldSession construction", "adopt_held_permit") not in held:
        failures.append("held-session construction must be reported with its function")
    if ("held session marker entry", "adopt_held_permit") not in held:
        failures.append("a lifetime held-session marker must be reported with its function")

    eager = sites(SELF_TEST_EAGER_ENTRY)
    if ("eager session entry", "einsum") not in eager:
        failures.append("a library eager-session entry must be reported")

    cfg_test = sites(SELF_TEST_CFG_TEST)
    if ("eager session entry", "helper") in cfg_test:
        failures.append("a `#[cfg(test)]` item must be out of scope")
    if ("eager session entry", "production") not in cfg_test:
        failures.append("a `#[cfg(not(test))]` item must stay in scope")

    if ("eager session entry", "after") not in sites(SELF_TEST_CFG_TEST_ARM):
        failures.append("a `#[cfg(test)]` match arm must not hide the items after it")

    trait_default = sites(SELF_TEST_TRAIT_DEFAULT)
    if ("backend session entry", "Host::with_backend_session_cached") not in trait_default:
        failures.append("a trait default method must be keyed by its trait, not `<module>`")

    for name in ("src/backend/tests.rs", "src/typed_eager_tests.rs", "tests/integration.rs"):
        if is_library_source(Path("crates/x") / name):
            failures.append(f"test source {name} must be out of scope")
    if not is_library_source(Path("crates/x/src/backend.rs")):
        failures.append("library source must stay in scope")

    grouped = sites(SELF_TEST_GROUP_IMPORT)
    if ("CpuExecSession construction", "build") not in grouped:
        failures.append("a grouped `as` import must be traced")

    # Scope entry points create execution state (a permit and its resources), so
    # they are tracked by the same gate that guards session construction.
    scope_entry = sites(SELF_TEST_SCOPE_ENTRY)
    if ("with_execution_scope", "hidden_scope") not in scope_entry:
        failures.append("a renamed execution scope must be traced to its mechanism")
    if ("with_evaluation_scope", "hidden_scope") not in scope_entry:
        failures.append("a renamed evaluation-scope hook must be traced to its mechanism")

    unrelated = sites(SELF_TEST_UNRELATED_STRUCT)
    if unrelated:
        failures.append("an unrelated struct literal must not be reported")

    for failure in failures:
        print(f"self-test failure: {failure}")
    if failures:
        return 1
    print("self-test passed: alias, boundary, method, eager, held, test-scope, cfg(test), trait, grouped-import and scope cases")
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
