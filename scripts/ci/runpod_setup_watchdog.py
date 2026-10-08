#!/usr/bin/env python3
"""Bound the accepted pod's paid setup window from a hosted runner."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from scripts.ci.runpod_cost import parse_runpod_timestamp


def execution_started(jobs: list[dict]) -> bool:
    """Disarm only for this workflow's GPU job, never an unrelated test."""
    for job in jobs:
        if not job.get("name", "").endswith("CUDA GPU tests on RunPod"):
            continue
        for step in job.get("steps", []):
            if step.get("name") == "Run CUDA tests from archive" and step.get("status") in {
                "in_progress", "completed"
            } and step.get("conclusion") != "skipped":
                return True
    return False


def watch(
    *, deadline: float, jobs: Callable[[], list[dict]], delete: Callable[[], None],
    now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 15,
) -> bool:
    """Return true on disarm; delete and return false when setup expires."""
    while True:
        try:
            rows = jobs()
        except (OSError, ValueError, RuntimeError) as error:
            # Unknown progress never extends the paid setup budget.
            print(f"::warning::Setup progress unavailable: {error}", flush=True)
            rows = []
        if execution_started(rows):
            print("Setup watchdog disarmed: GPU tests started.", flush=True)
            return True
        if any(job.get("name", "").endswith("CUDA GPU tests on RunPod")
               and job.get("status") == "completed" for job in rows):
            print("::error::GPU setup finished before tests; deleting pod.", flush=True)
            delete()
            return False
        remaining = deadline - now()
        if remaining <= 0:
            print("::error::Paid GPU setup deadline exceeded; deleting pod.", flush=True)
            delete()
            return False
        sleep(min(poll_seconds, remaining))


def request(url: str, token: str, method: str = "GET") -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {token}", "User-Agent": "tenferro-ci-setup-watchdog",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float, default=900)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("budget must be finite and positive")
    pod_url = f"https://rest.runpod.io/v1/pods/{os.environ['POD_ID']}"
    pod_token = os.environ["RUNPOD_API_KEY"]

    def delete() -> None:
        for attempt in range(3):
            try:
                code, _ = request(pod_url, pod_token, "DELETE")
                if 200 <= code < 300 or code == 404:
                    deleted_at = dt.datetime.now(dt.timezone.utc).isoformat()
                    print(f"Setup watchdog confirmed pod deletion at {deleted_at} (HTTP {code}).", flush=True)
                    return
                if code < 500 and code not in {408, 429}:
                    raise RuntimeError(f"Pod deletion rejected: HTTP {code}")
            except OSError:
                pass
            if attempt < 2:
                time.sleep(5)
        raise RuntimeError("Setup watchdog could not confirm pod deletion")
    try:
        for attempt in range(3):
            try:
                status, data = request(pod_url, pod_token)
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(5)
                continue
            if status not in {408, 429} and status < 500:
                break
            if attempt < 2:
                time.sleep(5)
        if status == 404:
            print("Setup watchdog: pod already deleted.")
            return 0
        if status != 200:
            raise RuntimeError(f"Cannot read paid pod start time: HTTP {status}")
        pod = json.loads(data)
        if not isinstance(pod, dict) or not isinstance(pod.get("lastStartedAt"), str):
            raise ValueError("Pod record has no readable lastStartedAt")
        started = parse_runpod_timestamp(pod["lastStartedAt"]).timestamp()
    except (OSError, ValueError, KeyError, RuntimeError):
        # Without a start time this job cannot bound billing. Fail closed and
        # confirm deletion rather than silently falling back to 45 minutes.
        delete()
        raise
    deadline = started + args.budget_seconds
    print(f"Paid setup deadline: {dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat()}", flush=True)

    def jobs() -> list[dict]:
        url = (f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
               f"{os.environ['GITHUB_RUN_ID']}/jobs?filter=latest&per_page=100")
        result: list[dict] = []
        page = 1
        while True:
            code, body = request(f"{url}&page={page}", os.environ["GH_TOKEN"])
            if code != 200:
                raise RuntimeError(f"Cannot read GPU progress: HTTP {code}")
            rows = json.loads(body)["jobs"]
            result.extend(rows)
            if len(rows) < 100:
                return result
            page += 1
    return 0 if watch(deadline=deadline, jobs=jobs, delete=delete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
