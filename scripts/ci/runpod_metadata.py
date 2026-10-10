#!/usr/bin/env python3
"""Preserve non-secret Pod cost/placement metadata before GPU execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def snapshot(pod: dict, pod_id: str) -> dict:
    """Keep only reporting fields; Pod responses also contain runner credentials."""
    if not isinstance(pod, dict) or pod.get("id") != pod_id:
        raise ValueError("Pod metadata does not match the accepted Pod")
    result = {key: pod[key] for key in (
        "id", "lastStartedAt", "costPerHr", "adjustedCostPerHr", "machineId",
    ) if key in pod}
    machine = pod.get("machine")
    if isinstance(machine, dict):
        result["machine"] = {key: machine[key] for key in ("dataCenterId",) if key in machine}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--creation-record", type=Path, required=True)
    args = parser.parse_args()
    try:
        response = subprocess.run([
            "curl", "-fsS", "--connect-timeout", "2", "--max-time", "5",
            "--url", f"https://rest.runpod.io/v1/pods/{args.pod_id}?includeMachine=true",
            "--header", f"Authorization: Bearer {os.environ['RUNPOD_API_KEY']}",
        ], check=True, capture_output=True, timeout=6)
        value = snapshot(json.loads(response.stdout), args.pod_id)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        # CalledProcessError/TimeoutExpired include argv (and the API key).
        print(f"::warning::Startup Pod metadata read unavailable ({type(error).__name__})")
        try:
            value = snapshot(json.loads(args.creation_record.read_text()), args.pod_id)
        except (OSError, ValueError, TypeError) as error:
            print(f"::warning::Creation Pod metadata unavailable: {error}")
            return 0
    # A small job output survives separate hosted runners, without an artifact
    # upload delaying the paid GPU job. Never publish the full provider response.
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("pod_metadata=" + json.dumps(value, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
