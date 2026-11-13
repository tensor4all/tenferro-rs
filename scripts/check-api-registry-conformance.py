#!/usr/bin/env python3
"""Registry conformance of the compiled eager surface (#2602).

The public-API snapshot (`scripts/api-snapshot.sh`) shows *that* a signature
changed; this check decides whether the new surface is *allowed*, so it still
fails after the snapshots are regenerated. It reads:

- the `api_ops!` registries (`crates/*/src/api/registry.rs`): every op row with
  a `signature <layer>: "..."` line declares a canonical compiled signature,
  and every exception row lists its permitted symbols. An exception symbol
  written as a full `pub fn ...` line permits exactly that signature; a bare
  path permits the name with any signature but never an engine argument;
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
   A method of an eager receiver that takes an `&EagerTensor` and returns an
   owned `Tensor` is a value copy and must be declared whatever its name, so a
   copy alias cannot hide behind a new name;
3. engine argument: a public function whose rendered line mentions an engine
   type anywhere (a `dyn`/`impl` engine trait, a generic bound or `where`
   clause, or a concrete engine such as `CpuEngine`) and names an eager type
   must be a declared signature or an exact-signature exception. Eager
   operations take the engine once, through `EagerRuntime::with_eager_session`;
4. no stale exception: every `tenferro_ad` exception signature (or path)
   exists in every `tenferro-ad` profile.

Known limit: rustdoc JSON renders items under the type's own name, so a renamed
re-export (`pub use EagerSession as Session`) is checked under `EagerSession`;
a blanket `impl<T> Trait for T` is not attributed to the eager receivers.

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

# Any engine type: the `Engine`/`ExecutionEngine`/`*Engine` traits and the
# concrete engines (`CpuEngine`, `CudaEngine`, ...), in parameters, generic
# bounds, `where` clauses or return types.
ENGINE_TYPE = re.compile(r"\b\w*Engine\b")
COPY_SHAPE = re.compile(r"&tenferro_ad::EagerTensor\b.*\)\s*->.*\btenferro_tensor::types::Tensor\b")
EAGER_TYPE = re.compile(r"\b(?:Eager\w+|GradientValue|Gradients)\b")

OP_RE = re.compile(r"^\s{4}op (\w+) \{$")
SIGNATURE_RE = re.compile(r'^\s+signature (\w+): "(.*)";$')
EXCEPTION_RE = re.compile(r"^\s{8}(E\d+) \{$")
SYMBOL_RE = re.compile(r'^\s+"(.+)",?$')
FN_PREFIX = re.compile(r"^pub (?:(?:const|unsafe|async|extern \"[^\"]*\") )*fn ")


@dataclass
class Registry:
    signatures: dict[str, tuple[str, str]] = field(default_factory=dict)  # line -> (op, file)
    exceptions: dict[str, str] = field(default_factory=dict)  # bare path -> id
    exception_lines: dict[str, str] = field(default_factory=dict)  # exact `pub fn` line -> id

    def add_exception(self, symbol: str, ex: str) -> None:
        if symbol.startswith("pub "):
            self.exception_lines[symbol] = ex
        else:
            self.exceptions[symbol] = ex


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
                for symbol in re.findall(r'"([^"]+)"', line):
                    reg.add_exception(symbol, exception)
                if line.rstrip().endswith("];"):
                    in_symbols = False
                continue
            if in_symbols:
                if line.strip().startswith("];"):
                    in_symbols = False
                elif m := SYMBOL_RE.match(line):
                    reg.add_exception(m.group(1), exception)
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


def normalize(line: str) -> str:
    """`pub const/unsafe fn` lines compare as `pub fn` lines."""
    return FN_PREFIX.sub("pub fn ", line, count=1)


def parse_fn(line: str) -> Item | None:
    m = FN_PREFIX.match(line)
    if not m:
        return None
    rest = line[m.end() :]
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
        line = re.sub(r" where .*$", "", line)
        m = re.match(r"^(?:unsafe )?impl(?:<.*?>)? (.+?) for (.+)$", line)
        if not m:
            continue
        trait = strip_generics(split_path(m.group(1).strip())[-1])
        target = strip_generics(split_path(m.group(2).strip().lstrip("&"))[-1])
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

    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        line_set = set(lines)
        impls = trait_impls(lines)
        crate = path.stem.split("+", 1)[0].replace("-", "_")

        # 1. required signatures
        for sig, (op, where) in reg.signatures.items():
            if sig.startswith(f"pub fn {crate}::") and sig not in line_set:
                errors.append(f"{path.name}: missing declared signature of `{op}` ({where}):\n    {sig}")

        for raw in lines:
            item = parse_fn(raw)
            if item is None:
                continue
            line = normalize(raw)
            recv = receivers(item, impls)
            candidates = symbol_candidates(item, recv)
            declared = line in reg.signatures or line in reg.exception_lines
            name_excepted = any(c in reg.exceptions for c in candidates)
            names_engine = bool(ENGINE_TYPE.search(line))
            # 2. declared receiver, spelling and copy shape
            scoped = recv and (SCOPED_NAME.search(item.name) or COPY_SHAPE.search(line))
            if scoped and not declared and not name_excepted:
                same_name = [sig for sig, (op, _) in reg.signatures.items() if op == item.name]
                hint = (
                    f"; the registry declares `{item.name}` as\n    {same_name[0]}"
                    if same_name
                    else "; declare it in an api_ops! registry (signature or exception) or remove it"
                )
                errors.append(
                    f"{path.name}: undeclared eager entry point {sorted(candidates)[0]}{hint}\n    found: {line}"
                )
            # 3. engine type on an eager item: only exact declared signatures
            if names_engine and (recv or EAGER_TYPE.search(line)) and not declared:
                errors.append(
                    f"{path.name}: eager entry point names an engine outside the declared session boundary; "
                    f"make it an EagerSession method or declare its exact signature:\n    {line}"
                )

    # 4. stale exceptions, in every tenferro-ad profile
    for profile in sorted(surface.glob("tenferro-ad*.txt")):
        lines = profile.read_text(encoding="utf-8").splitlines()
        line_set = {normalize(line) for line in lines}
        impls = trait_impls(lines)
        names: set[str] = set()
        for line in lines:
            item = parse_fn(line)
            if item:
                names |= symbol_candidates(item, receivers(item, impls))
        for symbol, ex in sorted(reg.exceptions.items()):
            if symbol.startswith("tenferro_ad::") and symbol not in names:
                errors.append(f"{profile.name}: exception {ex} lists `{symbol}`, which is not a public function")
        for sig, ex in sorted(reg.exception_lines.items()):
            if sig.startswith("pub fn tenferro_ad::") and sig not in line_set:
                errors.append(f"{profile.name}: exception {ex} lists a signature that is not public:\n    {sig}")
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
