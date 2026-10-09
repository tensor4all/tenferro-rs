#!/usr/bin/env python3
"""Collect and summarize RunPod costs without allocating a GPU.

Run as ``python3 -m scripts.ci.runpod_cost_history --help``. Raw evidence and
reports belong outside the checkout. Missing logs/costs remain unknown, never
zero. Log estimates are separate from exact-window stage artifacts and invoices.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import zipfile

from scripts.ci.runpod_cost import parse_runpod_timestamp

WORKFLOW = "runpod-gpu-test.yml"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CREATED = re.compile(r"Created pod (\w+): GPU (.+) at (?:\$([\d.]+)/hr|unknown \$/hr);")
PINNED = re.compile(r"Pinned PR #(\d+) merge ref to ([0-9a-f]{40})\.")


def gh_api(path: str, *, pages: bool = False) -> bytes:
    args = ["gh", "api", path]
    if pages:
        args += ["--paginate", "--slurp"]
    result = subprocess.run(args, capture_output=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def collect(repository: str, since: str, until: str, directory: Path) -> None:
    """Keep every run attempt, including failed/cancelled provisioning."""
    directory.mkdir(parents=True, exist_ok=True)
    pages = json.loads(gh_api(
        f"repos/{repository}/actions/workflows/{WORKFLOW}/runs"
        f"?per_page=100&created={since}..{until}", pages=True))
    if any(page["total_count"] > 1000 for page in pages):
        raise ValueError("GitHub caps filtered run searches at 1000; use a shorter date range")
    runs = [row for page in pages for row in page["workflow_runs"]]
    (directory / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")

    def one(run: dict) -> None:
        prefix = f"repos/{repository}/actions/runs/{run['id']}"
        for attempt in range(1, run["run_attempt"] + 1):
            dest = directory / str(run["id"]) / str(attempt)
            dest.mkdir(parents=True, exist_ok=True)
            errors = []
            try:
                metadata = run if attempt == run["run_attempt"] else json.loads(
                    gh_api(f"{prefix}/attempts/{attempt}"))
                (dest / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
                if metadata["status"] != "completed":
                    errors.append("Run attempt is still active; collect again after completion")
                elif metadata["conclusion"] not in {"skipped", "startup_failure"}:
                    # Successful downloads are immutable per attempt. Failed requests
                    # are retried on a later collection, never cached as empty evidence.
                    if not (dest / "logs.zip").exists():
                        (dest / "logs.zip").write_bytes(gh_api(f"{prefix}/attempts/{attempt}/logs"))
                    if not (dest / "jobs.json").exists():
                        (dest / "jobs.json").write_bytes(gh_api(
                            f"{prefix}/attempts/{attempt}/jobs?per_page=100", pages=True))
                    artifacts = json.loads(gh_api(f"{prefix}/artifacts?per_page=100", pages=True))
                    (dest / "artifacts.json").write_text(json.dumps(artifacts) + "\n")
                    name = f"runpod-stage-cost-{run['id']}-{attempt}"
                    for page in artifacts:
                        for artifact in page["artifacts"]:
                            if artifact["name"] == name and not artifact["expired"]:
                                if not (dest / "cost.zip").exists():
                                    (dest / "cost.zip").write_bytes(gh_api(
                                        f"repos/{repository}/actions/artifacts/{artifact['id']}/zip"))
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                errors.append(str(error))
            (dest / "errors.json").write_text(json.dumps(errors) + "\n")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(one, runs), start=1):
            if index % 20 == 0 or index == len(runs):
                print(f"Collected {index}/{len(runs)} workflows", flush=True)


def log_lines(archive: bytes) -> list[tuple[str, str]]:
    """Read each job once; do not double-count its per-step/system copies."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for name in zipped.namelist():
            if "/" in name or not name.endswith(".txt"):
                continue
            if not any(label in name for label in (
                "Start RunPod", "Delete RunPod", "Authorize RunPod", "Publish CI GPU gate",
                "Monitor paid GPU lifecycle")):
                continue
            for line in zipped.read(name).decode(errors="replace").splitlines():
                stamp, _, message = ANSI.sub("", line).partition(" ")
                if re.fullmatch(r"\d{4}-\d\d-\d\dT\S+Z", stamp):
                    rows.append((stamp, message))
    return sorted(rows)


def parse_attempt(run: dict, lines: list[tuple[str, str]], cost: dict | None) -> dict:
    """Prefer stage records; reconstruct older/rejected pods from actual output."""
    pods: dict[str, dict] = {}
    cost_source = "stage_artifact"
    result = {"run_id": run["id"], "attempt": run["run_attempt"],
              "url": run["html_url"], "event": run["event"],
              "workflow_sha": run["head_sha"], "created_at": run["created_at"],
              "conclusion": run["conclusion"], "pr_number": None,
              "tested_ref": None, "reused": False, "pods": []}
    cleanup_pod = None
    for stamp, message in lines:
        if match := PINNED.fullmatch(message):
            result.update(pr_number=int(match[1]), tested_ref=match[2])
        elif re.fullmatch(r"  TESTED_REF: [0-9a-f]{40}", message):
            result["tested_ref"] = message.split(": ", 1)[1]
        elif re.fullmatch(r"Tested ref [0-9a-f]{40} already passed: .+ Skipping every paid step\.", message):
            result["reused"] = True
        elif match := CREATED.match(message):
            pods[match[1]] = {"pod_id": match[1], "gpu_type_id": match[2],
                             "price_per_hour": float(match[3]) if match[3] else None,
                             "started_at": stamp, "start_source": "creation_log",
                             "accepted": False, "deleted_at": None, "stages": {}}
        elif match := re.fullmatch(r"Candidate .+ created pod (\w+) with an unverifiable GPU assignment: .+", message):
            pods[match[1]] = {"pod_id": match[1], "gpu_type_id": "unknown", "price_per_hour": None,
                             "started_at": stamp, "start_source": "creation_log",
                             "accepted": False, "deleted_at": None, "stages": {}}
        elif match := re.fullmatch(r"Runner .+ online: pod (\w+) passed the CUDA smoke proof in (\d+)s .+", message):
            if match[1] in pods:
                pods[match[1]].update(accepted=True, startup_seconds=int(match[2]))
        elif match := re.fullmatch(r"  POD_ID: (\w+)", message):
            cleanup_pod = match[1]
        elif message.startswith("RunPod pod started at: ") and cleanup_pod in pods:
            pods[cleanup_pod].update(started_at=message.split(": ", 1)[1],
                                     start_source="provider_timestamp")
        elif match := re.fullmatch(r"Deleted (?:RunPod pod: (\w+)|pod (\w+) before any test setup\.)", message):
            pod_id = match[1] or match[2]
            if pod_id in pods:
                pods[pod_id]["deleted_at"] = stamp
        elif match := re.fullmatch(r"RunPod delete HTTP status for (\w+): (\d+) \(attempt \d+/\d+\)", message):
            if match[1] in pods and (200 <= int(match[2]) < 300 or match[2] == "404"):
                pods[match[1]]["deleted_at"] = stamp
        elif message.startswith("RunPod lifecycle cost: ") and cost is None:
            cost = json.loads(message.removeprefix("RunPod lifecycle cost: "))
            cost_source = "lifecycle_log"
    for pod in pods.values():
        pod["cost_source"] = "log_window_estimate"
        pod["gpu_job_conclusion"] = run["conclusion"] if pod["accepted"] else "not_run"
        pod["paid_seconds"] = None
        pod["estimated_gpu_cost"] = None
        if pod["deleted_at"]:
            seconds = (parse_runpod_timestamp(pod["deleted_at"]) -
                       parse_runpod_timestamp(pod["started_at"])).total_seconds()
            if seconds >= 0:
                pod["paid_seconds"] = seconds
                if pod["price_per_hour"] is not None:
                    pod["estimated_gpu_cost"] = seconds * pod["price_per_hour"] / 3600
    if cost is not None:
        for name in ("paid_seconds", "estimated_gpu_cost", "price_per_hour"):
            if not isinstance(cost[name], (int, float)) or not math.isfinite(cost[name]) or cost[name] < 0:
                raise ValueError(f"Invalid stage cost {name}")
        pod_id = cost["pod_id"]
        logged_gpu = pods.get(pod_id, {}).get("gpu_type_id", "unknown")
        accepted = pods.get(pod_id, {}).get("accepted", cost["gpu_job_conclusion"] != "not_run")
        pods[pod_id] = {**pods.get(pod_id, {}), **cost, "accepted": accepted,
                        "cost_source": cost_source, "start_source": "provider_timestamp"}
        if not cost.get("gpu_type_id"):
            # A failed provision publishes its pod id before its GPU job outputs.
            pods[pod_id]["gpu_type_id"] = logged_gpu
        if cost.get("tested_ref"):
            result["tested_ref"] = cost["tested_ref"]
    result["pods"] = list(pods.values())
    return result


def summarize(directory: Path) -> dict:
    attempts, errors = [], []
    runs = json.loads((directory / "runs.json").read_text())
    for run in runs:
        for attempt in range(1, run["run_attempt"] + 1):
            dest = directory / str(run["id"]) / str(attempt)
            try:
                metadata = json.loads((dest / "run.json").read_text())
                logs = log_lines((dest / "logs.zip").read_bytes()) if (dest / "logs.zip").exists() else []
                cost = None
                if (dest / "cost.zip").exists():
                    with zipfile.ZipFile(dest / "cost.zip") as zipped:
                        cost = json.loads(zipped.read("runpod-stage-cost.json"))
                value = parse_attempt(metadata, logs, cost)
                jobs = None
                if (dest / "jobs.json").exists():
                    jobs = [j for page in json.loads((dest / "jobs.json").read_text()) for j in page["jobs"]]
                    gpu_jobs = [j for j in jobs if j["name"].endswith("CUDA GPU tests on RunPod")]
                    if len(gpu_jobs) == 1:
                        for pod in value["pods"]:
                            if pod["accepted"] and pod["cost_source"] == "log_window_estimate":
                                pod["gpu_job_conclusion"] = gpu_jobs[0]["conclusion"]
                value["evidence_complete"] = bool(logs) or jobs == [] or metadata["conclusion"] in {"skipped", "startup_failure"}
                attempts.append(value)
                if not value["evidence_complete"]:
                    errors.append(f"{run['id']}/{attempt}: logs unavailable or attempt still active")
                if (dest / "errors.json").exists():
                    errors.extend(f"{run['id']}/{attempt}: {e}" for e in json.loads((dest / "errors.json").read_text()))
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                errors.append(f"{run['id']}/{attempt}: {error}")

    devices, refs, stages = defaultdict(list), defaultdict(list), Counter()
    seen = set()
    for attempt in sorted(attempts, key=lambda a: (a["created_at"], a["attempt"])):
        for pod in attempt["pods"]:
            if pod["pod_id"] in seen:
                errors.append(f"Repeated pod record {pod['pod_id']}; counted only once")
                continue
            seen.add(pod["pod_id"])
            devices[pod["gpu_type_id"] or "unknown"].append(pod)
            if pod["accepted"] and attempt["tested_ref"]:
                # Manual confirmations are intentional and reported separately.
                refs[(attempt["event"], attempt["tested_ref"])].append({
                    "run_id": attempt["run_id"], "attempt": attempt["attempt"],
                    "pod_id": pod["pod_id"], "cost": pod["estimated_gpu_cost"]})
            for name, data in pod["stages"].items():
                stages[name] += data["seconds"]
    by_gpu = {}
    for gpu, pods in sorted(devices.items()):
        successful = [p for p in pods if p["gpu_job_conclusion"] == "success"]
        known = [p for p in pods if p["estimated_gpu_cost"] is not None]
        times = [p["paid_seconds"] for p in successful if p["paid_seconds"] is not None]
        spend = sum(p["estimated_gpu_cost"] for p in known)
        by_gpu[gpu] = {"pods": len(pods), "successful_pods": len(successful),
                      "not_accepted": sum(not p["accepted"] for p in pods),
                      "unknown_cost_pods": len(pods) - len(known),
                      "known_estimated_cost": spend,
                      "known_cost_per_success": spend / len(successful) if successful and len(known) == len(pods) else None,
                      "median_success_seconds": statistics.median(times) if times else None,
                      "failed_or_unaccepted_cost": sum(p["estimated_gpu_cost"] for p in known
                          if p["gpu_job_conclusion"] != "success")}
    return {"workflow_runs": len(runs), "attempts": attempts, "errors": errors,
            "conclusions": dict(Counter(a["conclusion"] for a in attempts)),
            "reused_attempts": sum(a["reused"] for a in attempts),
            "by_gpu": by_gpu, "stage_seconds": dict(stages),
            "same_ref_repeats": [{"event": event, "tested_ref": ref, "runs": rows}
                                 for (event, ref), rows in refs.items() if len(rows) > 1]}


def markdown(value: dict) -> str:
    rows = ["# RunPod cost history", "",
            f"Workflow runs: {value['workflow_runs']}; reused gates: {value['reused_attempts']}; "
            f"evidence warnings: {len(value['errors'])}.", "",
            "| GPU | Pods | Successes | Not accepted | Unknown cost | Known estimated $ | $ / success* | Median successful seconds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for gpu, row in value["by_gpu"].items():
        average = row["known_cost_per_success"]
        median = row["median_success_seconds"]
        rows.append(f"| {gpu} | {row['pods']} | {row['successful_pods']} | {row['not_accepted']} | "
                    f"{row['unknown_cost_pods']} | {row['known_estimated_cost']:.5f} | "
                    f"{f'{average:.5f}' if average is not None else 'n/a'} | "
                    f"{f'{median:.1f}' if median is not None else 'n/a'} |")
    rows += ["", "*Known spend including failures divided by successful GPU jobs. The ratio is unavailable when any pod "
             "cost is missing; missing costs are not zero. Mixed revisions, GPU hosts and prices make this "
             "descriptive evidence, not a selection benchmark. "
             "Stage artifacts use provider start timestamps; older/rejected pods use log windows and recorded prices. "
             "Log windows can miss provider startup/deletion latency. Storage and invoice reconciliation are excluded.",
             "", "## Same-ref repeated accepted pods", ""]
    for repeat in value["same_ref_repeats"]:
        rows.append(f"- {repeat['event']} `{repeat['tested_ref']}`: " + ", ".join(
            f"{r['run_id']}/{r['attempt']} ({r['pod_id']})" for r in repeat["runs"]))
    if not value["same_ref_repeats"]:
        rows.append("None observed in the available evidence.")
    rows += ["", "## Stage time (recorded windows only)", ""]
    total = sum(value["stage_seconds"].values())
    for name, seconds in sorted(value["stage_seconds"].items()):
        rows.append(f"- {name}: {seconds:.1f}s ({seconds / total:.1%})")
    rows += ["", "## Evidence warnings", ""] + (value["errors"] or ["None."])
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="tensor4all/tenferro-rs")
    parser.add_argument("--directory", type=Path, required=True, help="Evidence/report directory outside the checkout")
    parser.add_argument("--since", help="Inclusive UTC date YYYY-MM-DD; requires --until")
    parser.add_argument("--until", help="Inclusive UTC date YYYY-MM-DD")
    args = parser.parse_args()
    if bool(args.since) != bool(args.until):
        parser.error("--since and --until must be supplied together")
    if args.since:
        for value in (args.since, args.until):
            if not re.fullmatch(r"\d{4}-\d\d-\d\d", value):
                parser.error("dates must be YYYY-MM-DD")
        collect(args.repository, args.since, args.until, args.directory)
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = markdown(result)
    (args.directory / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
