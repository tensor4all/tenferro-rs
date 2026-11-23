#!/usr/bin/env python3
"""Find, wait for, and re-run the maintainer-approved RunPod verdict of a fork PR.

A fork PR reaches the RunPod GPU lane only through a manual dispatch of the
trusted main `runpod-gpu-test.yml`, in which a maintainer names the exact head
commit they reviewed (`recover_runpod_pr.py PR --approve-head SHA`). That run
ends in a job named `verdict_job_name(pr, sha)`, which runs only after the paid
lane passed for that head.

The verdict is read from that job rather than from a Checks API publication on
the fork's head commit, so it does not depend on whether a base-repository
check run can be attached to a fork commit. `CI_gpu.yml`'s required `CI GPU
gate` job, which GitHub attaches to the fork PR itself, waits for the verdict
(`wait`), and the verdict job re-runs that gate when it already gave up
(`rerun`).

Only a run of the main-branch `runpod-gpu-test.yml` started by
`workflow_dispatch` in this repository counts. Its `authorize` job admits a fork
PR only for an `admin`/`maintain` dispatcher and only at the approved head, so
the job name carries a verdict nobody else can produce.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

RUNPOD_WORKFLOW = "runpod-gpu-test.yml"
RUNPOD_WORKFLOW_PATH = f".github/workflows/{RUNPOD_WORKFLOW}"
GATE_WORKFLOW = "CI_gpu.yml"
RECENT_RUNS = 30
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# (method, url) -> (HTTP status, body)
Transport = Callable[[str, str], tuple[int, bytes]]


class LookupFailed(RuntimeError):
    """A GitHub API response could not be read or had an unexpected shape."""


def verdict_job_name(pr_number: int, head_sha: str) -> str:
    """Return the name of the RunPod job that carries a fork PR's verdict."""

    return f"Fork PR GPU verdict #{pr_number} @ {head_sha}"


def approval_command(pr_number: int, head_sha: str) -> str:
    """Return the command a maintainer runs to approve this head on RunPod."""

    return (
        f"python3 scripts/ci/recover_runpod_pr.py {pr_number} "
        f"--approve-head {head_sha} --wait"
    )


def _get_json(transport: Transport, url: str) -> Mapping[str, Any]:
    status, body = transport("GET", url)
    if status != 200:
        raise LookupFailed(f"GET {url} returned HTTP {status}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise LookupFailed(f"GET {url} returned invalid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise LookupFailed(f"GET {url} did not return an object")
    return payload


def _list(payload: Mapping[str, Any], key: str, url: str) -> list[Mapping[str, Any]]:
    items = payload.get(key)
    if not isinstance(items, list):
        raise LookupFailed(f"GET {url} has no '{key}' array")
    return [item for item in items if isinstance(item, Mapping)]


def is_trusted_runpod_run(run: Mapping[str, Any], repository: str) -> bool:
    """Whether `run` is a manual dispatch of the main RunPod workflow here."""

    repo = run.get("repository")
    return (
        run.get("path") == RUNPOD_WORKFLOW_PATH
        and run.get("event") == "workflow_dispatch"
        and run.get("head_branch") == "main"
        and isinstance(repo, Mapping)
        and repo.get("full_name") == repository
    )


def find_verdict(
    transport: Transport, repository: str, pr_number: int, head_sha: str
) -> str | None:
    """Return the URL of a passed verdict job for this PR head, if any."""

    query = urllib.parse.urlencode(
        {"event": "workflow_dispatch", "branch": "main", "per_page": RECENT_RUNS}
    )
    url = (
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{RUNPOD_WORKFLOW}/runs?{query}"
    )
    runs = _list(_get_json(transport, url), "workflow_runs", url)
    wanted = verdict_job_name(pr_number, head_sha)
    for run in runs:
        if not is_trusted_runpod_run(run, repository):
            continue
        run_id = run.get("id")
        if not isinstance(run_id, int):
            continue
        jobs_url = (
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"
            "/jobs?per_page=100"
        )
        for job in _list(_get_json(transport, jobs_url), "jobs", jobs_url):
            if job.get("name") == wanted and job.get("conclusion") == "success":
                return str(job.get("html_url") or run.get("html_url") or "")
    return None


def wait_for_verdict(
    transport: Transport,
    repository: str,
    pr_number: int,
    head_sha: str,
    *,
    wait_minutes: float,
    poll_seconds: float = 60.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll until the verdict appears or `wait_minutes` pass; report the result."""

    deadline = clock() + wait_minutes * 60.0
    print(
        f"Fork PR #{pr_number}: waiting for a maintainer-approved RunPod run on "
        f"{head_sha}. A maintainer who reviewed this head approves it with:\n"
        f"  {approval_command(pr_number, head_sha)}"
    )
    while True:
        try:
            url = find_verdict(transport, repository, pr_number, head_sha)
        except LookupFailed as error:
            print(f"::warning::RunPod verdict lookup failed; still waiting: {error}")
            url = None
        if url is not None:
            print(f"Approved RunPod GPU run passed for {head_sha}: {url}")
            return True
        if clock() >= deadline:
            print(
                f"::error::No approved RunPod GPU run passed for {head_sha} within "
                f"{wait_minutes:g} minutes. Approving it later re-runs this gate."
            )
            return False
        sleep(poll_seconds)


def rerun_gate(
    transport: Transport, repository: str, pr_number: int, head_sha: str
) -> str:
    """Re-run the fork PR's failed `CI_gpu.yml` gate; never raises.

    Returns a one-line account of what happened. A gate that is still waiting
    finds the verdict by itself, so only a completed, unsuccessful run is re-run.
    """

    query = urllib.parse.urlencode(
        {"event": "pull_request", "head_sha": head_sha, "per_page": 10}
    )
    url = (
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{GATE_WORKFLOW}/runs?{query}"
    )
    manual = f"re-run the failed jobs of PR #{pr_number}'s CI_gpu run by hand"
    try:
        runs = _list(_get_json(transport, url), "workflow_runs", url)
    except LookupFailed as error:
        return f"::warning::Could not list CI_gpu runs ({error}); {manual}."
    runs = [run for run in runs if run.get("head_sha") == head_sha]
    if not runs:
        return f"No CI_gpu run exists for {head_sha} yet; it will find the verdict."
    newest = max(runs, key=lambda run: (str(run.get("created_at") or ""), run.get("id") or 0))
    run_id = newest.get("id")
    if newest.get("status") != "completed":
        return f"CI_gpu run {run_id} is still {newest.get('status')}; it will find the verdict."
    if newest.get("conclusion") == "success":
        return f"CI_gpu run {run_id} already passed."
    rerun_url = f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/rerun-failed-jobs"
    status, _ = transport("POST", rerun_url)
    if status in (201, 204):
        return f"Re-ran the failed jobs of CI_gpu run {run_id}."
    return f"::warning::Re-running CI_gpu run {run_id} returned HTTP {status}; {manual}."


def _github_transport(token: str) -> Transport:
    def send(method: str, url: str) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url,
            method=method,
            data=b"" if method == "POST" else None,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tenferro-ci-fork-gpu-gate/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except (urllib.error.URLError, TimeoutError) as error:
            return 0, str(error).encode()

    return send


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("wait", "rerun"))
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--wait-minutes", type=float, default=330.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, transport: Transport | None = None) -> int:
    args = _parse_args(argv)
    if args.pr <= 0 or not SHA_PATTERN.match(args.head_sha):
        print("::error::--pr must be positive and --head-sha a full lowercase SHA.")
        return 2
    if transport is None:
        token = os.environ.get("GH_TOKEN", "")
        if not token:
            print("::error::GH_TOKEN is not set.")
            return 2
        transport = _github_transport(token)
    if args.mode == "rerun":
        # The verdict is already final; a failed re-run must not turn it red.
        print(rerun_gate(transport, args.repository, args.pr, args.head_sha))
        return 0
    passed = wait_for_verdict(
        transport,
        args.repository,
        args.pr,
        args.head_sha,
        wait_minutes=args.wait_minutes,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
