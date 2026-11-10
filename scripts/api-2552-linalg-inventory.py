#!/usr/bin/env python3
"""Public-surface inventory for the #2552 call-grammar migration of tenferro-linalg.

The script works on `cargo public-api --simplified` listings of `tenferro-linalg`
built with the docs.rs feature set (`autodiff,native,cuda`), so every public item
the crate exports under those features is covered, including trait methods.

Subcommands:

* `snapshot OUT` runs `cargo +<toolchain> public-api` and writes the listing.
* `inventory BEFORE` prints the Markdown inventory: every public item of the
  listing, grouped by symbol, with the target name under the #2552 grammar.
* `migration BEFORE AFTER` prints the removed-name table: every symbol present in
  BEFORE and absent from AFTER, with its replacement, and checks that the
  replacement symbol exists in AFTER.

The rename rules below are the single source of the target names; both tables
are generated from them, never typed by hand.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

TOOLCHAIN = "nightly-2026-07-18"
FEATURES = "autodiff,native,cuda"

CRATE = "tenferro_linalg::"

# Methods of the operation traits, by layer. Values are the canonical op names.
OPS = [
    "svd", "svdvals", "qr", "householder_qr", "rank_revealing_qr", "lu", "full_piv_lu",
    "full_piv_lu_solve", "solve", "cholesky", "eigh", "eig", "eigvals", "eigvalsh",
    "triangular_solve", "slogdet", "det", "inv", "pinv", "norm", "lstsq",
]

# Option types: setter renames `name` -> `with_name`.
OPTION_SETTERS = {
    "SvdOptions": ["gauge", "derivative_eps", "driver"],
    "EighOptions": ["gauge", "derivative_eps", "driver"],
    "QrOptions": ["gauge"],
    "RankRevealingQrOptions": ["gauge", "rtol", "atol"],
}

# Retired type aliases and traits -> replacement (symbol, note).
RETIRED_TYPES = {
    "TypedSvd": ("Svd", "`Svd<TypedTensor<T>, TypedTensor<T::Real>>`"),
    "TypedLu": ("Lu", "`Lu<TypedTensor<T>, TypedTensor<T::Real>>`"),
    "TypedFullPivLu": ("FullPivLu", "`FullPivLu<TypedTensor<T>, TypedTensor<T::Real>>`"),
    "TypedEig": ("Eig", "`Eig<TypedTensor<T::Complex>>`"),
    "TypedRankRevealingQrResult": (
        "RankRevealingQrResult",
        "`RankRevealingQrResult<TypedTensor<T>, TypedTensor<i64>>`",
    ),
    "Result": (None, "no public function returned it; use `tenferro_tensor::Result` (concrete), "
               "`tenferro_runtime::Result` (traced) or `tenferro_ad::Result` (eager)"),
    "error::Result": (None, "as `Result`"),
    "TensorReadLinalgExt": ("TensorLinalgExt", "`TensorRead<'_>` implements `LinalgInput`"),
    "TypedTensorLinalgExt": ("TensorLinalgExt", "`TypedTensor<T>` implements `LinalgInput`"),
    "TraceContextLinalgExt": ("TracedTensorLinalgExt", "issue #2552 decision 4"),
    "EagerTensorLinalgExt": ("EagerSessionLinalgExt", "issue #2552 decision 3"),
}

# Old trait-method suffix forms -> (new method, options note).
VARIANTS = {
    "svd_full": ("svd_with", "`SvdOptions::default().with_shape(SvdShape::Full)`"),
    "svd_full_read": ("svd_with", "`SvdOptions::default().with_shape(SvdShape::Full)`"),
    "pinv_with_rtol": ("pinv_with", "`PinvOptions::default().with_rtol(rtol)`"),
    "pinv_with_rtol_read": ("pinv_with", "`PinvOptions::default().with_rtol(rtol)`"),
    "solve_read_into": ("solve_into", "destination stays `TensorWrite<'_>`"),
    "svd_with_exec": ("svd_with", "receiver `TypedTensorView`; engine argument; "
                      "`full` becomes `SvdShape`"),
    "svd_values_with_exec": ("svdvals", "receiver `TypedTensorView`; engine argument"),
    "qr_with_exec": ("qr", "receiver `TypedTensorView`; engine argument"),
    "eigh_with_exec": ("eigh", "receiver `TypedTensorView`; engine argument"),
    "eigh_values_with_exec": ("eigvalsh", "receiver `TypedTensorView`; engine argument"),
    "eig_values_with_exec": ("eigvals", "receiver `TypedTensorView`; engine argument"),
}

# Operation traits per layer, with the trait their methods move to.
OP_TRAITS = {
    "TensorLinalgExt": "TensorLinalgExt",
    "TensorReadLinalgExt": "TensorLinalgExt",
    "TypedTensorLinalgExt": "TensorLinalgExt",
    "TracedTensorLinalgExt": "TracedTensorLinalgExt",
    "TraceContextLinalgExt": "TracedTensorLinalgExt",
    "EagerSessionLinalgExt": "EagerSessionLinalgExt",
    "EagerTensorLinalgExt": "EagerSessionLinalgExt",
}

# Signature changes that keep the name (bool / usize-axis arguments become options).
SIGNATURE_NOTES = {
    "triangular_solve": "four `bool` flags become `TriangularSolveOptions` "
                        "(`Side`, `Uplo`, `Transpose`, `Diagonal`) on `triangular_solve_with`",
    "norm": "`ord`, `usize` axes and `keepdim: bool` become `NormOptions` "
            "(`isize` axes, `KeepDims`) on `norm_with`",
    "rank_revealing_qr": "options move to `rank_revealing_qr_with`; the short form uses defaults",
}


def strip_generics(text: str) -> str:
    out, depth = [], 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def symbol_of(line: str) -> str | None:
    """Return the item path of one public-api line, without generics or signature."""
    m = re.match(r"^pub (?:unsafe )?(?:const )?(?:fn|trait|struct|enum|type|mod|const|static|"
                 r"use)\s+(.*)$", line)
    if m:
        rest = m.group(1)
        if line.startswith("pub fn") or line.startswith("pub const fn"):
            rest = rest.split("(", 1)[0]
        else:
            rest = re.split(r"[:=(]|\s", strip_generics(rest), maxsplit=1)[0]
        return strip_generics(rest).strip()
    m = re.match(r"^pub (.+?)::(\w+): ", line)  # struct fields
    if m:
        return f"{strip_generics(m.group(1))}::{m.group(2)}"
    m = re.match(r"^pub (.+)$", line)  # enum variants and other entries
    if m:
        return strip_generics(m.group(1)).split("(")[0].split(" ")[0]
    if line.startswith("impl"):
        return None
    return None


def load(path: Path) -> "OrderedDict[str, list[str]]":
    items: "OrderedDict[str, list[str]]" = OrderedDict()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        sym = symbol_of(line)
        if sym is None:
            continue
        items.setdefault(sym, []).append(line)
    return items


def local(sym: str) -> str:
    """Path relative to the crate; foreign-type impl listings keep their own prefix."""
    if sym.startswith(CRATE):
        sym = sym[len(CRATE):]
    if sym.startswith("prelude::"):
        sym = sym[len("prelude::"):]
    return sym


def target(sym: str) -> tuple[str, str, str]:
    """Return (status, target symbol, note) for one item path relative to the crate."""
    s = local(sym)
    parts = s.split("::")
    last = parts[-1]
    owner = parts[-2] if len(parts) >= 2 else ""

    if s in RETIRED_TYPES:
        new, note = RETIRED_TYPES[s]
        return ("removed", new or "-", note)
    if len(parts) == 1 and last in VARIANTS:  # with_exec free functions
        new, note = VARIANTS[last]
        return ("removed", f"TensorLinalgExt::{new}", note)
    if owner in OPTION_SETTERS and last in OPTION_SETTERS[owner]:
        return ("renamed", f"{owner}::with_{last}", "rule 3: `with_*` setters")

    if owner == "LinalgBackend" or (len(parts) >= 3 and parts[-2] in ("CpuEngine", "CudaEngine")):
        return ("kept", s, "provider SPI; see OPEN-QUESTIONS-2552.md Q3")

    trait = owner
    # Trait-method listings on foreign receivers (EagerSession, EagerTensor) map like the trait.
    if owner.startswith("EagerSession"):
        trait = "EagerSessionLinalgExt"
    elif owner == "EagerTensor":
        trait = "EagerTensorLinalgExt"
    if trait in OP_TRAITS:
        new_trait = OP_TRAITS[trait]
        name = last
        note = ""
        if name in VARIANTS:
            name, note = VARIANTS[name]
        elif name.endswith("_with_options_read"):
            name = name[: -len("_with_options_read")] + "_with"
        elif name.endswith("_with_options"):
            name = name[: -len("_with_options")] + "_with"
        elif name.endswith("_read"):
            name = name[: -len("_read")]
        base = name.removesuffix("_with").removesuffix("_into")
        if base in SIGNATURE_NOTES and not note:
            note = SIGNATURE_NOTES[base]
        new = f"{new_trait}::{name}"
        if trait == "TensorReadLinalgExt" and not note:
            note = "receiver `TensorRead<'_>`"
        if trait == "TypedTensorLinalgExt" and not note:
            note = "receiver `TypedTensor<T>`; typed outputs through `LinalgInput`"
        if trait == "TraceContextLinalgExt" and not note:
            note = "call on the `TracedTensor` instead of `ctx.op(value)`"
        if trait == "EagerTensorLinalgExt":
            note = "`session.solve(&a, &b)`"
        status = "kept" if (new == f"{trait}::{last}" and not note) else (
            "changed" if new == f"{trait}::{last}" else "removed")
        if status == "kept" and base in SIGNATURE_NOTES:
            status = "changed"
        return (status, new, note)
    return ("kept", s, "")


def cmd_snapshot(args: argparse.Namespace) -> int:
    out = subprocess.run(
        ["cargo", f"+{TOOLCHAIN}", "public-api", "-p", "tenferro-linalg", "--features", FEATURES,
         "-s"],
        check=True, capture_output=True, text=True,
    ).stdout
    Path(args.out).write_text(out)
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    items = load(Path(args.before))
    rows, seen = [], set()
    for sym in items:
        if local(sym) in seen:  # the prelude re-exports the same item
            continue
        seen.add(local(sym))
        status, new, note = target(sym)
        rows.append((local(sym), status, new, note))
    counts: dict[str, int] = {}
    for _, status, _, _ in rows:
        counts[status] = counts.get(status, 0) + 1
    print(f"Public symbols: {len(rows)} ({len(load_lines(Path(args.before)))} listing lines). "
          + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print()
    print("| symbol | status | target | note |")
    print("|---|---|---|---|")
    for s, status, new, note in rows:
        print(f"| `{s}` | {status} | `{new}` | {note} |")
    return 0


def load_lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]


def cmd_migration(args: argparse.Namespace) -> int:
    before = load(Path(args.before))
    after = load(Path(args.after))
    after_local = {local(s) for s in after}
    after_syms = set(after)
    removed, seen = [], set()
    for s in before:
        if s in after_syms or local(s) in seen:
            continue
        seen.add(local(s))
        removed.append(s)
    print("| removed | replacement | note | replacement present |")
    print("|---|---|---|---|")
    missing = 0
    for sym in removed:
        status, new, note = target(sym)
        present = "n/a" if new == "-" else ("yes" if new in after_local else "NO")
        if present == "NO":
            missing += 1
        print(f"| `{local(sym)}` | `{new}` | {note} | {present} |")
    print()
    print(f"Removed symbols: {len(removed)}; replacements missing from AFTER: {missing}")
    return 1 if (missing and args.strict) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("snapshot")
    p.add_argument("out")
    p.set_defaults(func=cmd_snapshot)
    p = sub.add_parser("inventory")
    p.add_argument("before")
    p.set_defaults(func=cmd_inventory)
    p = sub.add_parser("migration")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_migration)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
