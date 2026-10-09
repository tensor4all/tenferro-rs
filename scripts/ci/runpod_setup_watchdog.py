#!/usr/bin/env python3
"""Bound setup and stop obsolete paid work throughout the accepted pod's life."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from collections.abc import Callable

from scripts.ci.runpod_cost import parse_runpod_timestamp
from scripts.ci.runpod_lifecycle import HostedClient, request
from scripts.ci.runpod_workflow_cost import report


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
    lifetime_deadline: float,
    obsolete: Callable[[], str | None] = lambda: None,
    cancel: Callable[[], None] = lambda: None,
    now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 15,
) -> bool:
    """Wait for GPU job completion; confirm deletion before cancelling lost work."""
    tests_started = False
    while True:
        try:
            reason = obsolete()
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"::warning::PR state unavailable; retaining bounded workload: {error}", flush=True)
            reason = None
        if reason:
            print(f"Stopping obsolete paid work: {reason}", flush=True)
            delete()
            cancel()
            return False
        try:
            rows = jobs()
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            # Unknown progress never extends the paid setup budget.
            print(f"::warning::Setup progress unavailable: {error}", flush=True)
            rows = []
        if execution_started(rows) and not tests_started:
            tests_started = True
            print("Setup deadline disarmed: GPU tests started; lifecycle monitoring continues.", flush=True)
        if any(job.get("name", "").endswith("CUDA GPU tests on RunPod")
               and job.get("status") == "completed" for job in rows):
            if tests_started:
                # Normal cleanup captures metadata and deletes immediately. Returning
                # here keeps the watcher out of the cleanup job's dependency chain.
                return True
            print("::error::GPU setup finished before tests; deleting pod.", flush=True)
            delete()
            cancel()
            return False
        effective_deadline = lifetime_deadline if tests_started else min(deadline, lifetime_deadline)
        remaining = effective_deadline - now()
        if remaining <= 0:
            print("::error::Paid GPU setup/lifetime deadline exceeded; deleting pod.", flush=True)
            delete()
            cancel()
            return False
        sleep(min(poll_seconds, remaining))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument("--lifetime-seconds", type=float, default=3600)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("budget must be finite and positive")
    if not math.isfinite(args.lifetime_seconds) or args.lifetime_seconds < args.budget_seconds:
        parser.error("lifetime must be finite and at least the setup budget")
    pod_url = f"https://rest.runpod.io/v1/pods/{os.environ['POD_ID']}"
    pod_token = os.environ["RUNPOD_API_KEY"]
    client = HostedClient(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"], pod_token, transport=request)
    run_id, run_attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    pod = None
    def delete() -> None:
        deleted_at = client.delete(os.environ["POD_ID"])
        # The normal cleanup can no longer GET a pod we deleted. Emit the
        # already-held metadata after DELETE and before cancellation, with no
        # extra request on the paid path. Missing metadata stays a warning.
        try:
            value = report(pod, [], deleted_at, os.environ.get("GPU_TYPE_ID", ""))
            value.update(tested_ref=os.environ.get("TESTED_REF", ""), gpu_job_conclusion="interrupted")
            print("RunPod lifecycle cost: " + json.dumps(value), flush=True)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            print(f"::warning::Lifecycle cost unavailable after deletion: {error}", flush=True)
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

    return 0 if watch(
        deadline=deadline, lifetime_deadline=started + args.lifetime_seconds,
        jobs=lambda: client.jobs(run_id, run_attempt), delete=delete,
        obsolete=lambda: client.obsolete(os.environ["PR_NUMBER"], os.environ["TARGET_HEAD_SHA"]),
        cancel=lambda: client.cancel(run_id, run_attempt),
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
