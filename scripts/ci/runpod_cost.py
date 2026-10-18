#!/usr/bin/env python3
"""Report a RunPod pod's paid time and estimated cost from its REST record.

`cleanup-runpod` reads the pod record right before deleting the pod. RunPod's
REST `lastStartedAt` is not ISO-8601: it is `2026-10-04 11:09:24.633 +0000 UTC`
(Go's `time.Time` string form). The previous inline parser only accepted ISO
with `Z`, raised inside `|| true`, and never printed a cost (#2002: 76 of 76
cleanup jobs). Both forms are parsed here, and an unparseable record prints a
`::warning::` with the raw value instead of failing silently.

Deleting the pod must never depend on this report, so the script always exits
0; the workflow additionally runs it as a non-blocking step.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from collections.abc import Mapping
from typing import Any

# Go `time.Time.String()`: "2006-01-02 15:04:05.999999999 -0700 MST", with an
# optional fractional part of up to nine digits and a trailing zone name.
_GO_TIME = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))? (?P<offset>[+-]\d{4})(?: [A-Za-z]+)?$"
)


def parse_runpod_timestamp(value: str) -> datetime.datetime:
    """Parse RunPod's Go-style or ISO-8601 timestamp into an aware datetime."""

    text = value.strip()
    match = _GO_TIME.match(text)
    if match:
        fraction = (match.group("fraction") or "").ljust(6, "0")[:6]
        return datetime.datetime.strptime(
            f"{match.group('date')} {match.group('time')}.{fraction} "
            f"{match.group('offset')}",
            "%Y-%m-%d %H:%M:%S.%f %z",
        )
    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    parsed = datetime.datetime.fromisoformat(iso)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp has no UTC offset: {value!r}")
    return parsed


def cost_report(
    pod: Mapping[str, Any], now: datetime.datetime
) -> tuple[list[str], list[str]]:
    """Return (report lines, warnings) for one pod record at time `now`."""

    cost = pod.get("cost", pod.get("adjustedCostPerHr"))
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
        cost = pod.get("costPerHr")
    started = pod.get("startedAt", pod.get("lastStartedAt"))
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return [], [f"RunPod cost data unavailable in pod record (costPerHr={cost!r})"]
    if not isinstance(started, str) or not started:
        return [], [f"RunPod start time unavailable in pod record (lastStartedAt={started!r})"]
    try:
        begin = parse_runpod_timestamp(started)
    except ValueError as error:
        return [], [f"Unparseable RunPod lastStartedAt {started!r}: {error}"]
    hours = max(0.0, (now - begin).total_seconds() / 3600.0)
    return [
        f"RunPod pod started at: {begin.isoformat()}",
        f"RunPod paid time: {hours:.2f}h at ${cost:.2f}/hr",
        f"RunPod estimated paid cost: ${hours * cost:.3f}",
    ], []


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: runpod_cost.py <pod-record.json>", file=sys.stderr)
        return 0
    try:
        with open(argv[1], encoding="utf-8") as handle:
            pod = json.load(handle)
        if not isinstance(pod, Mapping):
            raise ValueError("pod record is not a JSON object")
    except (OSError, ValueError) as error:
        print(f"::warning::RunPod pod record unreadable: {error}")
        return 0
    lines, warnings = cost_report(pod, datetime.datetime.now(datetime.timezone.utc))
    for line in lines:
        print(line)
    for warning in warnings:
        print(f"::warning::{warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
