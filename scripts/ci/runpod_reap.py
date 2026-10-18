#!/usr/bin/env python3
"""Reclaim explicitly owned CI pods after cancellation or failed cleanup.

Defaults to a read-only dry run. The scheduled trusted-main workflow supplies
--execute. Foreign/untagged/debug pods and unknown GitHub states are left alone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from scripts.ci.runpod_cost import parse_runpod_timestamp
from scripts.ci.runpod_lifecycle import HostedClient, WORKFLOW_PATH, owner
from scripts.ci.runpod_workflow_cost import report


def reason_to_reap(pod: dict, run: dict, repository: str, now: dt.datetime) -> str | None:
    identity = owner(pod, repository)
    if identity is None:
        return None
    run_id, attempt = identity
    if (run.get("id") != run_id or run.get("path") != WORKFLOW_PATH
            or run.get("repository", {}).get("full_name") != repository):
        raise ValueError("CI pod does not match its owning workflow")
    current_attempt = run["run_attempt"]
    if current_attempt < attempt:
        raise ValueError("Pod belongs to a future workflow attempt")
    if current_attempt > attempt:
        return "superseded workflow attempt"
    if run["status"] == "completed":
        # Give the normal cleanup and provider listing five minutes to settle.
        elapsed = (now - parse_runpod_timestamp(run["updated_at"])).total_seconds()
        return "owning workflow completed at least five minutes ago" if elapsed >= 300 else None
    if run["status"] not in {"queued", "in_progress", "waiting", "pending", "requested"}:
        raise ValueError(f"Unknown workflow status: {run['status']}")
    # The active watcher covers 15m setup and a 60m accepted-pod lifetime.
    # This independent 2h backstop also covers failed/cancelled startup jobs.
    # A v2 Pod may be stuck in PROVISIONING and never have startedAt. Use its
    # creation time only for orphan age, never to invent a paid cost window.
    timestamp = pod.get("startedAt") or pod.get("lastStartedAt") or pod.get("createdAt")
    if not isinstance(timestamp, str):
        raise ValueError("Pod has no readable start or creation time for orphan recovery")
    started = parse_runpod_timestamp(timestamp)
    if (now - started).total_seconds() >= 7200:
        return "CI pod exceeded the two-hour lifetime backstop"
    return None


def reap(client: HostedClient, pods: list[dict], *, execute: bool, now: dt.datetime) -> dict:
    result = {"checked_at": now.isoformat(), "execute": execute, "pods": [], "errors": []}
    for pod in pods:
        identity = owner(pod, client.repository)
        if identity is None:
            continue
        run_id, attempt = identity
        record = {"pod_id": pod["id"], "run_id": run_id, "attempt": attempt}
        try:
            run = client.github(f"actions/runs/{run_id}")
            reason = reason_to_reap(pod, run, client.repository, now)
            if reason and pod["env"]["TENFERRO_CI_KEEP_FAILED"] == "true":
                if client.retained_for_debug(run_id, attempt, pod["id"]):
                    record["debug_retained"] = True
                    reason = None
            record["reason"] = reason
            if reason:
                print(f"{'Reclaiming' if execute else 'Would reclaim'} CI pod {pod['id']}: {reason}", flush=True)
                if execute:
                    record["deleted_at"] = client.delete(pod["id"])
                    try:
                        record["cost"] = report(pod, [], record["deleted_at"])
                    except (ValueError, KeyError, TypeError, AttributeError) as error:
                        record["cost_warning"] = str(error)
                    # Never cancel a newer rerun, or cancel before DELETE succeeds.
                    client.cancel(str(run_id), str(attempt))
            result["pods"].append(record)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            result["pods"].append(record)
            message = f"Pod {pod['id']}: {error}"
            result["errors"].append(message)
            print(f"::warning::{message}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = HostedClient.from_environment()
    pods = client.pods()
    result = reap(client, pods, execute=args.execute, now=dt.datetime.now(dt.timezone.utc))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
