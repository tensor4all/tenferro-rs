#!/usr/bin/env python3
"""Reject a tenferro release unless the CPU stack is one version set.

The release workflow must not ship a tree that mixes t4a-faer with colmat, or two
strided release trains. This is a publish preflight, not a code-style check.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    "colmat": (0, 34),
    "threadteam": (0, 6),
}
STRIDED = {
    "strided-basic": (0, 15),
    "strided-fused": (0, 15),
    "strided-kernel": (0, 15),
    "strided-nn": (0, 15),
    "strided-perm": (0, 15),
    "strided-traits": (0, 7),
    "strided-view": (0, 8),
}
FORBIDDEN_PREFIX = "t4a-faer"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default=str(ROOT / "Cargo.lock"))
    args = parser.parse_args()
    lock = Path(args.lock).read_text()
    versions = {}
    for block in lock.split("[[package]]")[1:]:
        name = re.search(r'^name = "([^"]+)"', block, re.M)
        ver = re.search(r'^version = "([^"]+)"', block, re.M)
        if name and ver:
            versions.setdefault(name.group(1), []).append(ver.group(1))
    problems = []
    for name in versions:
        if name == FORBIDDEN_PREFIX or name.startswith(FORBIDDEN_PREFIX + "-"):
            problems.append(f"forbidden dependency {name} {versions[name]}")
    wanted = {**EXPECTED, **STRIDED}
    for name, want in wanted.items():
        if name not in versions:
            problems.append(f"missing {name}")
            continue
        for v in versions[name]:
            try:
                major, minor = (int(x) for x in v.split(".")[:2])
            except ValueError:
                problems.append(f"{name} has unparsable version {v}")
                continue
            if (major, minor) != want:
                problems.append(f"{name} {v} != {want[0]}.{want[1]}.x")
        if len(set(versions[name])) > 1:
            problems.append(f"{name} has multiple versions {sorted(set(versions[name]))}")
    # Any other strided-* crate must be on the 0.15 train.
    for name in versions:
        if name.startswith("strided-") and name not in wanted:
            for v in versions[name]:
                try:
                    major, minor = (int(x) for x in v.split(".")[:2])
                except ValueError:
                    problems.append(f"{name} has unparsable version {v}")
                    continue
                if (major, minor) != (0, 15):
                    problems.append(f"unlisted {name} {v} != 0.15.x")
    for p in problems:
        print(f"FAIL stack-version: {p}")
    print(f"check-stack-versions: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
