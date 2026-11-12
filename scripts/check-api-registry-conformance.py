#!/usr/bin/env python3
"""Registry conformance of the compiled eager surface (#2602).

The public-API snapshot (`scripts/api-snapshot.sh`) shows *that* a signature
changed; this check decides whether the new surface is *allowed*, so it still
fails after the snapshots are regenerated. It reads:

- the `api_ops!` registries (`crates/*/src/api/registry.rs`): every op row with
  a `signature <layer>: "..."` line declares a canonical compiled signature,
  and every exception row lists its permitted symbols;
- a directory of public-API snapshots (`api/*.txt`, or the freshly generated
  directory of `scripts/api-snapshot.sh --check`).

Rules, over every snapshot profile in the directory (default and CUDA):

1. required: every declared signature appears verbatim in each profile of the
   crate it belongs to (`tenferro_ad` -> `tenferro-ad*.txt`);
2. declared: a public function of an eager receiver (`EagerSession`,
   `EagerTensor`, `EagerRuntime`, `GradientValue`, `Gradients`, including trait
   methods implemented for them and re-exported paths) whose name belongs to the
   eager AD, value-extraction, leaf-construction or engine-accessor vocabulary
   is either a declared signature or an exception symbol. A second spelling of a
   declared operation (a duplicate alias) or an undeclared receiver fails here;
3. engine receiver: a public function that takes an engine (`dyn Engine`,
   `dyn ExecutionEngine`, or an `impl` of either) and names an eager type is an
   exception symbol. Eager operations take the engine once, through
   `EagerRuntime::with_eager_session`; no other eager receiver takes one;
4. no stale exception: every `tenferro_ad::` exception symbol exists in the
   default profile.

Usage:
    python3 scripts/check-api-registry-conformance.py [--surface-dir api]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]

EAGER_RECEIVERS = ("EagerSession", "EagerTensor", "EagerRuntime", "GradientValue", "Gradients")

# Names of the eager AD, value-extraction, leaf-construction and engine-accessor
# vocabulary. A function of an eager receiver with such a name must be declared.
SCOPED_NAME = re.compile(
    r"^(?:"
    r"grad|vjp|jvp|hvp|backward"
    r"|detach|duplicate|to_tensor|to_owned|into_value|copy_value|materialize"
    r"|constant_from|variable_from|from_tensor|from_vec|from_host|requires_grad|leaf"
    r"|engine|backend"
    r")\w*$|session"
)

ENGINE_PARAM = re.compile(
    r"dyn [\w:]*\b(?:Engine|ExecutionEngine)\b|impl [\w:]*\b(?:Engine|ExecutionEngine)\b"
)
EAGER_TYPE = re.compile(r"\b(?:Eager\w+|GradientValue|Gradients)\b")

OP_RE = re.compile(r"^\s{4}op (\w+) \{$")
SIGNATURE_RE = re.compile(r'^\s+signature (\w+): "(.*)";$')
EXCEPTION_RE = re.compile(r"^\s{8}(E\d+) \{$")
SYMBOL_RE = re.compile(r'^\s+"([\w:]+)",?$')


@dataclass
class Registry:
    signatures: dict[str, tuple[str, str]] = field(default_factory=dict)  # line -> (op, file)
    exceptions: dict[str, str] = field(default_factory=dict)  # symbol -> id


def parse_registries(root: pathlib.Path) -> Registry:
    reg = Registry()
    for path in sorted(root.glob("crates/*/src/api/registry.rs")):
        op = None
        exception = None
        in_symbols = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if m := OP_RE.match(line):
                op, exception = m.group(1), None
                continue
            if m := SIGNATURE_RE.match(line):
                reg.signatures[m.group(2).replace('\\"', '"')] = (op or "?", str(path.relative_to(root)))
                continue
            if m := EXCEPTION_RE.match(line):
                exception, op = m.group(1), None
                continue
            if exception and line.strip().startswith("symbols: ["):
                in_symbols = True
                inline = re.findall(r'"([\w:]+)"', line)
                for symbol in inline:
                    reg.exceptions[symbol] = exception
                if line.rstrip().endswith("];"):
                    in_symbols = False
                continue
            if in_symbols:
                if line.strip().startswith("];"):
                    in_symbols = False
                elif m := SYMBOL_RE.match(line):
                    reg.exceptions[m.group(1)] = exception
    return reg


def strip_generics(segment: str) -> str:
    return re.sub(r"<.*>$", "", segment)


def split_path(path: str) -> list[str]:
    """Split `a::B<'_>::c` on `::` outside angle brackets."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(path):
        c = path[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        if depth == 0 and path.startswith("::", i):
            parts.append("".join(cur))
            cur = []
            i += 2
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


@dataclass(frozen=True)
class Item:
    line: str
    path: list[str]
    params: str

    @property
    def name(self) -> str:
        return strip_generics(self.path[-1])

    @property
    def owner(self) -> str:
        return strip_generics(self.path[-2]) if len(self.path) >= 2 else ""


def parse_fn(line: str) -> Item | None:
    if not line.startswith("pub fn "):
        return None
    rest = line[len("pub fn ") :]
    # The path ends at the first `(` outside angle brackets.
    depth = 0
    for i, c in enumerate(rest):
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "(" and depth == 0:
            path, tail = rest[:i], rest[i:]
            break
    else:
        return None
    path_parts = split_path(path)
    # Generic parameters of the function itself (`foo<T: X>`) stay on the last
    # segment and are stripped by `name`.
    depth, end = 0, len(tail)
    for i, c in enumerate(tail):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    return Item(line=line, path=path_parts, params=tail[1:end])


def trait_impls(lines: list[str]) -> dict[str, set[str]]:
    """Trait name -> implementing eager receivers, from `impl X for Y` lines."""
    impls: dict[str, set[str]] = {}
    for line in lines:
        m = re.match(r"^impl(?:<[^>]*>)? ([\w:<>' ,]+?) for ([\w:<>' ,&]+)$", line)
        if not m:
            continue
        trait = strip_generics(split_path(m.group(1))[-1])
        target = strip_generics(split_path(m.group(2).lstrip("&"))[-1])
        if target in EAGER_RECEIVERS:
            impls.setdefault(trait, set()).add(target)
    return impls


def receivers(item: Item, impls: dict[str, set[str]]) -> set[str]:
    owner = item.owner
    if owner in EAGER_RECEIVERS:
        return {owner}
    return set(impls.get(owner, ()))


def symbol_candidates(item: Item, recv: set[str]) -> set[str]:
    full = "::".join(strip_generics(p) for p in item.path)
    out = {full}
    for r in recv:
        out.add(f"tenferro_ad::{r}::{item.name}")
    return out


def check(surface: pathlib.Path, root: pathlib.Path) -> list[str]:
    reg = parse_registries(root)
    errors: list[str] = []
    files = sorted(p for p in surface.glob("*.txt") if p.name != "allowlist.txt")
    if not files:
        return [f"{surface}: no snapshot files"]
    default_ad = surface / "tenferro-ad.txt"

    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        line_set = set(lines)
        impls = trait_impls(lines)
        crate = path.stem.split("+", 1)[0].replace("-", "_")

        # 1. required signatures
        for sig, (op, where) in reg.signatures.items():
            if sig.startswith(f"pub fn {crate}::") and sig not in line_set:
                errors.append(f"{path.name}: missing declared signature of `{op}` ({where}):\n    {sig}")

        for line in lines:
            item = parse_fn(line)
            if item is None:
                continue
            recv = receivers(item, impls)
            candidates = symbol_candidates(item, recv)
            excepted = any(c in reg.exceptions for c in candidates)
            # 2. declared receiver and spelling
            if recv and SCOPED_NAME.search(item.name):
                if line not in reg.signatures and not excepted:
                    same_name = [s for s, (op, _) in reg.signatures.items() if op == item.name]
                    hint = (
                        f"; the registry declares `{item.name}` as\n    {same_name[0]}"
                        if same_name
                        else "; declare it in an api_ops! registry (signature or exception) or remove it"
                    )
                    errors.append(
                        f"{path.name}: undeclared eager entry point {sorted(candidates)[0]}{hint}\n    found: {line}"
                    )
            # 3. engine argument outside the session boundary
            if ENGINE_PARAM.search(item.params) and (recv or EAGER_TYPE.search(line)):
                if not excepted:
                    errors.append(
                        f"{path.name}: eager entry point takes an engine outside EagerRuntime::with_eager_session; "
                        f"make it an EagerSession method:\n    {line}"
                    )

    # 4. stale exceptions
    if default_ad.exists():
        names: set[str] = set()
        lines = default_ad.read_text(encoding="utf-8").splitlines()
        impls = trait_impls(lines)
        for line in lines:
            item = parse_fn(line)
            if item:
                names |= symbol_candidates(item, receivers(item, impls))
        for symbol, ex in sorted(reg.exceptions.items()):
            if symbol.startswith("tenferro_ad::") and symbol not in names:
                errors.append(f"{default_ad.name}: exception {ex} lists `{symbol}`, which is not a public function")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--surface-dir", type=pathlib.Path, default=ROOT / "api")
    parser.add_argument("--root-dir", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    errors = check(args.surface_dir, args.root_dir)
    if errors:
        print("api-registry-conformance: the compiled eager surface does not match the registry (#2602):", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 1
    print("api-registry-conformance: ok", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
