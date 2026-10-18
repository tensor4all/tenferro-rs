#!/usr/bin/env python3
"""Preserve non-secret Pod cost/placement metadata before GPU execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from scripts.ci.runpod_http import pod_request, request_with_retry

from scripts.ci.runpod_workflow_cost import report


def snapshot(pod: dict, pod_id: str) -> dict:
    """Keep only reporting fields; Pod responses also contain runner credentials."""
    if not isinstance(pod, dict) or pod.get("id") != pod_id:
        raise ValueError("Pod metadata does not match the accepted Pod")
    result = {key: pod[key] for key in (
        "id", "startedAt", "cost", "dataCenterId",
        "lastStartedAt", "costPerHr", "adjustedCostPerHr", "machineId",
    ) if key in pod}
    machine = pod.get("machine")
    if isinstance(machine, dict):
        result["machine"] = {key: machine[key] for key in ("dataCenterId",) if key in machine}
    # Validate the reporting fields with a zero-length window; no cost report
    # is emitted here. An incomplete HTTP 200 must still try the creation record.
    report(result, [], result.get("startedAt", result.get("lastStartedAt")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--creation-record", type=Path, required=True)
    args = parser.parse_args()
    try:
        response = request_with_retry(
            pod_request("https://api.runpod.io/v2/pods", os.environ["RUNPOD_API_KEY"], args.pod_id, "GET"),
            deadline=time.monotonic() + 5, timeout=5, attempts=1)
        if response.status != 200:
            raise ValueError(f"Pod metadata HTTP {response.status}")
        value = snapshot(json.loads(response.body), args.pod_id)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"::warning::Startup Pod metadata read unavailable ({type(error).__name__})")
        try:
            value = snapshot(json.loads(args.creation_record.read_text()), args.pod_id)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(f"::warning::Creation Pod metadata unavailable ({type(error).__name__})")
            return 0
    # A small job output survives separate hosted runners, without an artifact
    # upload delaying the paid GPU job. Never publish the full provider response.
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("pod_metadata=" + json.dumps(value, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
