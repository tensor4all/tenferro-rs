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

Polling stays cheap on the repository's shared `GITHUB_TOKEN` budget (1,000
requests per hour on the Team plan, shared by every workflow): an approval run
is named `approval_run_name(pr, sha)` by its `run-name:`, so a poll makes one
run listing and reads jobs only for runs carrying that name; a completed run
whose jobs were already read is not read again; and HTTP 403/429 doubles the
poll interval up to `MAX_BACKOFF_SECONDS`. The run name only narrows the
search: the verdict is still the successful job inside a trusted run.
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
MAX_BACKOFF_SECONDS = 900.0
RATE_LIMIT_STATUSES = (403, 429)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# (method, url) -> (HTTP status, body)
Transport = Callable[[str, str], tuple[int, bytes]]


class LookupFailed(RuntimeError):
    """A GitHub API response could not be read or had an unexpected shape."""


class RateLimited(LookupFailed):
    """GitHub refused a request with HTTP 403 or 429; the poller backs off."""


def verdict_job_name(pr_number: int, head_sha: str) -> str:
    """Return the name of the RunPod job that carries a fork PR's verdict."""

    return f"Fork PR GPU verdict #{pr_number} @ {head_sha}"


def approval_run_name(pr_number: int, head_sha: str) -> str:
    """Return the `run-name` of the RunPod run that approves a fork PR head."""

    return f"Fork PR GPU approval #{pr_number} @ {head_sha}"


def approval_command(pr_number: int, head_sha: str) -> str:
    """Return the command a maintainer runs to approve this head on RunPod."""

    return (
        f"python3 scripts/ci/recover_runpod_pr.py {pr_number} "
        f"--approve-head {head_sha} --wait"
    )


def _get_json(transport: Transport, url: str) -> Mapping[str, Any]:
    status, body = transport("GET", url)
    if status in RATE_LIMIT_STATUSES:
        raise RateLimited(f"GET {url} returned HTTP {status}")
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


RunKey = tuple[int, int]


def _run_key(run: Mapping[str, Any]) -> RunKey | None:
    run_id = run.get("id")
    attempt = run.get("run_attempt", 1)
    if not isinstance(run_id, int) or not isinstance(attempt, int):
        return None
    return (run_id, attempt)


def candidate_runs(
    runs: list[Mapping[str, Any]],
    repository: str,
    pr_number: int,
    head_sha: str,
    checked: set[RunKey],
) -> list[Mapping[str, Any]]:
    """Return the listed runs whose jobs are worth reading for this PR head.

    Only a trusted run named for this exact approval qualifies, and a run
    attempt that had already completed when its jobs were read is skipped: its
    jobs cannot change, and a re-run is a new attempt with a new key.
    """

    wanted = approval_run_name(pr_number, head_sha)
    selected = []
    for run in runs:
        key = _run_key(run)
        if (
            key is not None
            and key not in checked
            and run.get("display_title") == wanted
            and is_trusted_runpod_run(run, repository)
        ):
            selected.append(run)
    return selected


def find_verdict(
    transport: Transport,
    repository: str,
    pr_number: int,
    head_sha: str,
    checked: set[RunKey] | None = None,
) -> str | None:
    """Return the URL of a passed verdict job for this PR head, if any.

    `checked` carries completed run attempts already read between polls; this
    call adds the ones it reads.
    """

    if checked is None:
        checked = set()
    query = urllib.parse.urlencode(
        {"event": "workflow_dispatch", "branch": "main", "per_page": RECENT_RUNS}
    )
    url = (
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{RUNPOD_WORKFLOW}/runs?{query}"
    )
    runs = _list(_get_json(transport, url), "workflow_runs", url)
    wanted = verdict_job_name(pr_number, head_sha)
    for run in candidate_runs(runs, repository, pr_number, head_sha, checked):
        key = _run_key(run)
        assert key is not None
        jobs_url = (
            f"https://api.github.com/repos/{repository}/actions/runs/{key[0]}"
            "/jobs?per_page=100"
        )
        for job in _list(_get_json(transport, jobs_url), "jobs", jobs_url):
            if job.get("name") == wanted and job.get("conclusion") == "success":
                return str(job.get("html_url") or run.get("html_url") or "")
        if run.get("status") == "completed":
            checked.add(key)
    return None


def next_poll_delay(poll_seconds: float, rate_limited_streak: int) -> float:
    """Return the wait before the next poll after `rate_limited_streak` refusals.

    Each consecutive 403/429 doubles the interval, up to `MAX_BACKOFF_SECONDS`;
    any other outcome resets the streak to zero and the interval to normal.
    """

    if rate_limited_streak <= 0:
        return poll_seconds
    return min(poll_seconds * 2.0 ** min(rate_limited_streak, 32), MAX_BACKOFF_SECONDS)


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
    checked: set[RunKey] = set()
    rate_limited_streak = 0
    while True:
        try:
            url = find_verdict(transport, repository, pr_number, head_sha, checked)
            rate_limited_streak = 0
        except RateLimited as error:
            rate_limited_streak += 1
            print(
                f"::warning::GitHub rate-limited the RunPod verdict lookup ({error}); "
                f"backing off to {next_poll_delay(poll_seconds, rate_limited_streak):g} s."
            )
            url = None
        except LookupFailed as error:
            rate_limited_streak = 0
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
        delay = next_poll_delay(poll_seconds, rate_limited_streak)
        sleep(max(min(delay, deadline - clock()), 0.0))


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
