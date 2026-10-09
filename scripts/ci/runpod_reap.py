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
from scripts.ci.runpod_lifecycle import HostedClient, PODS_URL, WORKFLOW_PATH, owner
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
    started = parse_runpod_timestamp(pod["lastStartedAt"])
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
    status, body = client.transport(PODS_URL, client.pod_token)
    if status != 200:
        raise RuntimeError(f"Cannot list RunPod pods: HTTP {status}")
    pods = json.loads(body)
    if not isinstance(pods, list):
        raise ValueError("RunPod GET /pods did not return a list")
    result = reap(client, pods, execute=args.execute, now=dt.datetime.now(dt.timezone.utc))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
