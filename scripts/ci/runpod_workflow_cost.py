#!/usr/bin/env python3
"""Report paid GPU CI stage costs after pod deletion (never blocks cleanup)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.ci.runpod_cost import parse_runpod_timestamp


def stage(name: str) -> str:
    if name == "Run CUDA tests from archive":
        return "cuda_tests"
    if name == "Run CUDA tutorial artifact":
        return "tutorial"
    if name == "Run OpenXLA PJRT E2E tests from archive":
        return "pjrt_tests_and_setup"
    if "archive" in name.lower() and ("restore" in name.lower() or "download" in name.lower() or "transfer" in name.lower()):
        return "archive_transfer"
    if name in {"Select CUDA runtime for driver", "Restore cuTENSOR redistributable",
                "Restore CUDA minimal runtime tree", "Configure CUDA runtime libraries",
                "Verify loaded NVRTC version", "Download staged execution payload",
                "Install staged execution payload", "Transfer selected CUDA SDK",
                "Install selected CUDA SDK"}:
        return "runtime_setup"
    return "other_setup"


def report(pod: dict, jobs: list[dict], deleted_at: str, gpu_type_id: str = "") -> dict:
    begin = parse_runpod_timestamp(pod["lastStartedAt"])
    end = parse_runpod_timestamp(deleted_at)
    seconds = (end - begin).total_seconds()
    price = pod.get("adjustedCostPerHr")
    if not isinstance(price, (float, int)) or isinstance(price, bool) or price <= 0:
        price = pod.get("costPerHr")
    if not isinstance(price, (float, int)) or isinstance(price, bool) or not math.isfinite(price) or price <= 0 or seconds < 0:
        raise ValueError("Invalid paid window or hourly price")
    totals: dict[str, float] = {}
    selected = [job for job in jobs if job.get("name", "").endswith("CUDA GPU tests on RunPod") and job.get("conclusion") != "skipped"]
    if len(selected) > 1:
        raise ValueError("Multiple GPU execution jobs in this run")
    conclusion = "not_run"
    if selected:
        job = selected[0]
        conclusion = job.get("conclusion")
        started = parse_runpod_timestamp(job["started_at"])
        completed = parse_runpod_timestamp(job["completed_at"])
        totals["runner_startup_and_queue"] = max(0, (min(started, end) - begin).total_seconds())
        totals["cleanup_and_queue"] = max(0, (end - max(completed, begin)).total_seconds())
        previous = begin
        for step in job.get("steps", []):
            if step.get("conclusion") == "skipped":
                continue
            start = max(begin, parse_runpod_timestamp(step["started_at"]))
            finish = min(end, parse_runpod_timestamp(step["completed_at"]))
            if finish < start:
                continue
            if start < previous:
                raise ValueError("Overlapping GPU step timestamps")
            previous = finish
            label = stage(step["name"])
            totals[label] = totals.get(label, 0) + (finish - start).total_seconds()
    remainder = seconds - sum(totals.values())
    if remainder < -0.001:
        raise ValueError("Stage durations exceed paid window")
    totals["unassigned_overhead"] = max(0, remainder)
    return {"pod_id": pod.get("id"), "gpu_type_id": gpu_type_id, "started_at": begin.isoformat(), "deleted_at": end.isoformat(),
            "price_per_hour": price, "paid_seconds": seconds, "estimated_gpu_cost": seconds * price / 3600,
            "gpu_job_conclusion": conclusion,
            "stages": {name: {"seconds": value, "estimated_gpu_cost": value * price / 3600}
                       for name, value in totals.items()}}


def select_pod_record(path: Path, fallback: str, pod_id: str, deleted_at: str) -> tuple[dict, str]:
    """Prefer the final provider record; retain startup evidence on read failure."""
    for source in ("pre_delete", "startup"):
        try:
            pod = json.loads(path.read_text() if source == "pre_delete" else fallback)
            if not isinstance(pod, dict) or (pod_id and pod.get("id") != pod_id):
                raise ValueError("Pod metadata does not match the accepted Pod")
            report(pod, [], deleted_at)  # Check the timestamp and price without inventing either.
            return pod, source
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(f"::warning::RunPod {source} cost metadata unavailable: {error}")
    raise ValueError("Neither final nor startup Pod metadata can support a cost estimate")


def markdown(value: dict) -> str:
    lines = ["### RunPod paid GPU cost by stage", "",
             f"Estimated GPU cost: ${value['estimated_gpu_cost']:.5f} for {value['paid_seconds']:.1f}s at ${value['price_per_hour']:.2f}/hour.",
             "", "| Stage | Seconds | Estimated GPU cost |", "|---|---:|---:|"]
    for name, data in value["stages"].items():
        lines.append(f"| {name} | {data['seconds']:.1f} | ${data['estimated_gpu_cost']:.5f} |")
    lines += ["", "Estimate covers this pod through confirmed deletion; rejected provisioning attempts are reported separately. Storage and billing reconciliation are not included.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--fallback-pod-json", default="")
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--deleted-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-type-id", default="")
    parser.add_argument("--tested-ref", default="")
    parser.add_argument("--archive-cache-hit", default="")
    parser.add_argument("--cutensor-cache-hit", default="")
    parser.add_argument("--cuda-runtime-cache-hit", default="")
    args = parser.parse_args()
    try:
        payload = json.loads(args.jobs.read_text())
        # gh api --paginate --slurp produces an array of job-list pages.
        jobs = [job for page in payload for job in page["jobs"]]
        pod, source = select_pod_record(args.pod, args.fallback_pod_json, args.pod_id, args.deleted_at)
        value = report(pod, jobs, args.deleted_at, args.gpu_type_id)
        value["pod_record_source"] = source
        value["placement"] = {
            "machine_id": pod.get("machineId"),
            "data_center_id": (pod.get("machine") or {}).get("dataCenterId"),
        }
        value["tested_ref"] = args.tested_ref
        value["cache_hits"] = {
            name: {"true": True, "false": False}.get(hit)
            for name, hit in (("archives", args.archive_cache_hit),
                              ("cutensor", args.cutensor_cache_hit),
                              ("cuda_runtime", args.cuda_runtime_cache_hit))
        }
        args.output.write_text(json.dumps(value, indent=2) + "\n")
        print(markdown(value))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"::warning::RunPod stage-cost report unavailable: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
