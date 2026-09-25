#!/usr/bin/env python3
"""Capture the session-route unification baseline from Criterion run logs.

Schema: tenferro.session-route-baseline.v1

The campaign is defined by scripts/run-session-route-performance-gate.sh, which
records one run log per benchmark target. This script turns those logs into the
committed baseline artifact consumed by
scripts/compare-session-route-baseline.py.

Fail-closed: the capture refuses to write when
  * the tracked worktree is dirty (a baseline must identify exact source),
  * a benchmark target has no run log,
  * the case count for a target differs from the declared expectation.

Expectations encode the harness at the revision that captured the baseline. If
the harness case list changes, the baseline must be recaptured; that is why a
count mismatch is an error rather than a warning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "tenferro.session-route-baseline.v1"

# package|bench -> required case count at the capturing harness revision.
EXPECTED_CASES: dict[str, int] = {
    "tenferro-cpu|route_matrix": 47,
    "tenferro-runtime|session_chain": 6,
    "tenferro-runtime|elementwise_fusion": 14,
    "tenferro-ad|eager_dispatch_baseline": 28,
    "tenferro-ad|eager_backward_shape_churn": 1,
    "tenferro-linalg|linalg_vjp_gate": 4,
}

# Cases that exist only because the one-shot operation spelling still exists.
# Issue #1926 deletes that spelling, so these are before-only references and are
# not errors when they disappear from a candidate run.
def is_deleted_route_case(name: str) -> bool:
    return "/oneshot/" in name or name.endswith("/one_shot")

TIME_RE = re.compile(
    r"time:\s+\[([0-9.]+) (\S+) ([0-9.]+) (\S+) ([0-9.]+) (\S+)\]"
)
ANALYZING_RE = re.compile(r"^Benchmarking (.+): Analyzing\s*$")

# Criterion time units, normalised to nanoseconds.
UNIT_NS = {
    "ns": 1.0,
    "µs": 1e3,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
}


def parse_log(path: Path) -> dict[str, dict[str, float]]:
    """Return {case: {low_ns, mid_ns, high_ns}} with times normalised to ns."""
    cases: dict[str, dict[str, float]] = {}
    pending: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        match = ANALYZING_RE.match(line.strip())
        if match:
            pending = match.group(1)
            continue
        match = TIME_RE.search(line)
        if match and pending is not None:
            low, low_unit, mid, _, high, high_unit = match.groups()
            if low_unit != high_unit:
                raise RuntimeError(f"{path.name}: mixed units for {pending}")
            if low_unit not in UNIT_NS:
                raise RuntimeError(f"{path.name}: unknown unit {low_unit!r}")
            scale = UNIT_NS[low_unit]
            cases[pending] = {
                "low_ns": float(low) * scale,
                "mid_ns": float(mid) * scale,
                "high_ns": float(high) * scale,
            }
            pending = None
    return cases


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir", default=".", help="repository root (default: current directory)"
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="campaign output directory (default: <root>/target/session-route-performance-gate)",
    )
    parser.add_argument("--label", default="baseline", help="campaign label to read")
    parser.add_argument(
        "--output",
        default="docs/testing/session-route-baseline.json",
        help="baseline artifact to write",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="skip the clean-worktree check (diagnostic use only)",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    logs_dir = Path(args.logs_dir) if args.logs_dir else root / "target" / "session-route-performance-gate"

    if not args.allow_dirty:
        dirty = git(root, "status", "--porcelain")
        if dirty:
            raise RuntimeError(
                "baseline capture requires a clean tracked worktree; commit or "
                f"stash first. Dirty entries:\n{dirty}"
            )

    manifest_path = logs_dir / f"{args.label}-manifest.txt"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    manifest = {}
    for line in manifest_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            manifest[key] = value

    targets = {}
    for target, expected in EXPECTED_CASES.items():
        package, bench = target.split("|", 1)
        log = logs_dir / f"{args.label}-{package}-{bench}-run.log"
        if not log.is_file():
            raise RuntimeError(
                f"missing run log for {target}: {log}\n"
                "Run the campaign with --mode run first."
            )
        cases = parse_log(log)
        if len(cases) != expected:
            raise RuntimeError(
                f"{target}: expected {expected} cases, parsed {len(cases)}. "
                "The harness case list changed; recapture the baseline instead of "
                "adjusting the expectation."
            )
        targets[target] = {
            "source": f"crates/{package}/benches/{bench}.rs",
            "cases": cases,
        }

    artifact = {
        "schema": SCHEMA,
        "campaign": "session-route-unification",
        "umbrella_issue": 1929,
        "issue": 1926,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "baseline_commit": manifest.get("baseline_commit"),
        "harness_commit": manifest.get("head"),
        "label": args.label,
        "environment": {
            "cpu_affinity": manifest.get("cpu"),
            "taskset": manifest.get("taskset_available"),
            "warm_up_time_s": manifest.get("warm_up_time"),
            "measurement_time_s": manifest.get("measurement_time"),
            "sample_size": manifest.get("sample_size"),
            "thread_env": manifest.get("thread_env"),
            "nproc_as_seen_by_runner": manifest.get("nproc"),
            "host_loadavg_at_start": manifest.get("loadavg"),
        },
        "deleted_route_predicate": "/oneshot/ or trailing one_shot",
        "targets": targets,
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    total = sum(len(t["cases"]) for t in targets.values())
    deleted = sum(
        1
        for t in targets.values()
        for name in t["cases"]
        if is_deleted_route_case(name)
    )
    print(f"wrote {out}")
    print(f"  targets: {len(targets)}  cases: {total}  before-only (deleted route): {deleted}")
    print(f"  harness commit: {artifact['harness_commit']}")
    print(f"  baseline commit: {artifact['baseline_commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
