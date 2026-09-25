#!/usr/bin/env python3
"""Compare a session-route candidate run against the captured baseline.

Consumes docs/testing/session-route-baseline.json (schema
tenferro.session-route-baseline.v1) and the candidate run logs written by
scripts/run-session-route-performance-gate.sh.

Fail-closed behaviour:
  * a baseline case that is not a deleted-route case and is absent from the
    candidate is a FAILURE (missing case), not a silent skip;
  * a case whose change exceeds its predeclared threshold is a FAILURE.

Classification per case:
  PAIRED_OK     paired change within threshold
  REGRESSION    paired change above threshold
  NOISY         the low/high intervals overlap, so the change is unresolved
  DELETED       baseline-only deleted-route case (expected; before-only data)
  MISSING       baseline-only non-deleted case (failure)
  NEW           candidate-only case (informational)

Thresholds follow the existing unification gate protocol: a microsecond-scale
case (baseline mid <= 10 us) is a blocking regression only at +50% or more;
eager small-op cases use +5%; shape-churn, prepare and linalg VJP cases use
+10%.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ANALYZING_RE = re.compile(r"^Benchmarking (.+): Analyzing\s*$")
TIME_RE = re.compile(
    r"time:\s+\[([0-9.]+) (\S+) ([0-9.]+) (\S+) ([0-9.]+) (\S+)\]"
)
UNIT_NS = {"ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}

MICRO_THRESHOLD_PCT = 50.0
DEFAULT_THRESHOLD_PCT = 5.0
TEN_PERCENT_TARGETS = (
    "tenferro-ad|eager_backward_shape_churn",
    "tenferro-einsum|changing_shape_prepare",
    "tenferro-linalg|linalg_vjp_gate",
)


def is_deleted_route_case(name: str) -> bool:
    return "/oneshot/" in name or name.endswith("/one_shot")


def parse_log(path: Path) -> dict[str, dict[str, float]]:
    cases: dict[str, dict[str, float]] = {}
    pending: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        match = ANALYZING_RE.match(line.strip())
        if match:
            pending = match.group(1)
            continue
        match = TIME_RE.search(line)
        if match and pending is not None:
            low, unit, mid, _, high, high_unit = match.groups()
            if unit != high_unit or unit not in UNIT_NS:
                raise RuntimeError(f"{path.name}: bad unit for {pending}")
            scale = UNIT_NS[unit]
            cases[pending] = {
                "low_ns": float(low) * scale,
                "mid_ns": float(mid) * scale,
                "high_ns": float(high) * scale,
            }
            pending = None
    return cases


def threshold_pct(target: str, baseline_mid_ns: float) -> float:
    if baseline_mid_ns <= 10_000.0:
        return MICRO_THRESHOLD_PCT
    if target in TEN_PERCENT_TARGETS:
        return 10.0
    return DEFAULT_THRESHOLD_PCT


def fmt_ns(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.3f} ms"
    if value >= 1e3:
        return f"{value / 1e3:.3f} us"
    return f"{value:.1f} ns"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--baseline", default="docs/testing/session-route-baseline.json")
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--json-out", default=None, help="optional report path")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    logs_dir = Path(args.logs_dir) if args.logs_dir else root / "target" / "session-route-performance-gate"

    baseline = json.loads((root / args.baseline).read_text())

    rows = []
    failures = []
    for target, entry in sorted(baseline["targets"].items()):
        package, bench = target.split("|", 1)
        log = logs_dir / f"{args.label}-{package}-{bench}-run.log"
        if not log.is_file():
            failures.append(f"{target}: missing candidate run log {log}")
            continue
        candidate = parse_log(log)
        for name, base in sorted(entry["cases"].items()):
            cand = candidate.pop(name, None)
            if cand is None:
                if is_deleted_route_case(name):
                    rows.append((target, name, "DELETED", base["mid_ns"], None, None))
                else:
                    rows.append((target, name, "MISSING", base["mid_ns"], None, None))
                    failures.append(f"{target}:{name}: baseline case missing from candidate")
                continue
            delta = (cand["mid_ns"] - base["mid_ns"]) / base["mid_ns"] * 100.0
            limit = threshold_pct(target, base["mid_ns"])
            intervals_overlap = not (
                cand["high_ns"] < base["low_ns"] or base["high_ns"] < cand["low_ns"]
            )
            if delta > limit:
                status = "REGRESSION"
                failures.append(
                    f"{target}:{name}: {delta:+.1f}% exceeds +{limit:.0f}% "
                    f"({fmt_ns(base['mid_ns'])} -> {fmt_ns(cand['mid_ns'])})"
                )
            elif intervals_overlap:
                status = "NOISY"
            else:
                status = "PAIRED_OK"
            rows.append((target, name, status, base["mid_ns"], cand["mid_ns"], delta))
        for name, cand in sorted(candidate.items()):
            rows.append((target, name, "NEW", None, cand["mid_ns"], None))

    print(f"Session-route comparison: baseline vs {args.label}")
    print(f"  baseline harness commit: {baseline.get('harness_commit')}")
    print(f"  baseline library commit: {baseline.get('baseline_commit')}")
    print()
    print(f"{'status':11s} {'baseline':>11s} {'candidate':>11s} {'delta':>9s}  case")
    for target, name, status, base_mid, cand_mid, delta in rows:
        base_s = fmt_ns(base_mid) if base_mid is not None else "-"
        cand_s = fmt_ns(cand_mid) if cand_mid is not None else "-"
        delta_s = f"{delta:+.1f}%" if delta is not None else "-"
        print(f"{status:11s} {base_s:>11s} {cand_s:>11s} {delta_s:>9s}  {name}")

    paired = [r for r in rows if r[2] == "PAIRED_OK"]
    regressions = [r for r in rows if r[2] == "REGRESSION"]
    noisy = [r for r in rows if r[2] == "NOISY"]
    deleted = [r for r in rows if r[2] == "DELETED"]
    print()
    print(
        f"counts: paired_ok={len(paired)} noisy={len(noisy)} "
        f"regressions={len(regressions)} deleted_route={len(deleted)}"
    )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "baseline_harness_commit": baseline.get("harness_commit"),
                    "candidate_label": args.label,
                    "rows": [
                        {
                            "target": t,
                            "case": n,
                            "status": s,
                            "baseline_mid_ns": b,
                            "candidate_mid_ns": c,
                            "delta_pct": d,
                        }
                        for t, n, s, b, c, d in rows
                    ],
                    "failures": failures,
                },
                indent=2,
            )
            + "\n"
        )

    if failures:
        print()
        print("FAILURES:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print()
    print("no regression above the predeclared thresholds and no missing case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
