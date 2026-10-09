"""Hosted-only HTTP and ownership rules shared by RunPod lifecycle helpers."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.request

WORKFLOW_PATH = ".github/workflows/runpod-gpu-test.yml"
PODS_URL = "https://rest.runpod.io/v1/pods"


def request(url: str, token: str, method: str = "GET") -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {token}", "User-Agent": "tenferro-ci-lifecycle",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def ownership_environment(environ: dict[str, str]) -> dict[str, str]:
    """Persist nonsecret run identity so an independent job can reclaim pods."""
    keys = {"repository": "GITHUB_REPOSITORY", "run_id": "GITHUB_RUN_ID",
            "attempt": "GITHUB_RUN_ATTEMPT"}
    if not all(environ.get(key) for key in keys.values()):
        return {}  # Local/legacy callers are not managed by the scheduled reaper.
    return {"TENFERRO_CI_OWNER": "runpod-gpu-test-v1",
            **{f"TENFERRO_CI_{name.upper()}": environ[key] for name, key in keys.items()},
            "TENFERRO_CI_KEEP_FAILED": environ.get("PROVISION_KEEP_FAILED_PODS", "false")}


def owner(pod: dict, repository: str) -> tuple[int, int] | None:
    """Ignore foreign, legacy and explicitly retained debug pods."""
    env = pod.get("env") or {}
    if (not isinstance(env, dict) or env.get("TENFERRO_CI_OWNER") != "runpod-gpu-test-v1"
            or env.get("TENFERRO_CI_REPOSITORY") != repository
            or env.get("TENFERRO_CI_KEEP_FAILED") != "false"):
        return None
    run_id, attempt = env.get("TENFERRO_CI_RUN_ID", ""), env.get("TENFERRO_CI_ATTEMPT", "")
    if not re.fullmatch(r"[1-9][0-9]*", run_id) or not re.fullmatch(r"[1-9][0-9]*", attempt):
        return None
    if pod.get("name") != f"tenferro-rs-gpu-ci-{run_id}":
        return None
    return int(run_id), int(attempt)


class HostedClient:
    """Keep provider and GitHub credentials on trusted hosted runners only."""

    def __init__(self, repository: str, github_token: str, pod_token: str = "", *, transport=request):
        self.repository = repository
        self.github_token = github_token
        self.pod_token = pod_token
        self.transport = transport

    @classmethod
    def from_environment(cls):
        return cls(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"],
                   os.environ.get("RUNPOD_API_KEY", ""))

    def github(self, path: str) -> dict:
        code, body = self.transport(
            f"https://api.github.com/repos/{self.repository}/{path}", self.github_token)
        if code != 200:
            raise RuntimeError(f"GitHub {path}: HTTP {code}")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError(f"GitHub {path}: expected an object")
        return value

    def obsolete(self, pr_number: str, expected_head: str) -> str | None:
        if pr_number == "0":
            return None  # An immutable manual validation has no moving PR owner.
        if not re.fullmatch(r"[1-9][0-9]*", pr_number) or not re.fullmatch(r"[0-9a-f]{40}", expected_head):
            raise ValueError("PR lifecycle needs a PR number and authorized head SHA")
        pr = self.github(f"pulls/{pr_number}")
        if pr.get("state") == "closed":
            return f"PR #{pr_number} closed"
        head = pr.get("head")
        if (pr.get("state") != "open" or not isinstance(head, dict)
                or not re.fullmatch(r"[0-9a-f]{40}", head.get("sha") or "")):
            raise ValueError("PR response has no readable open/closed state and head")
        if pr["head"]["sha"] != expected_head:
            return f"PR #{pr_number} head moved from {expected_head} to {pr['head']['sha']}"
        return None

    def jobs(self, run_id: str, attempt: str) -> list[dict]:
        rows, page = [], 1
        while True:
            value = self.github(f"actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100&page={page}")
            batch = value["jobs"]
            rows.extend(batch)
            if len(batch) < 100:
                return rows
            page += 1

    def delete(self, pod_id: str) -> str:
        """Bound retries and return the confirmed deletion time (404 is success)."""
        for attempt in range(3):
            try:
                code, _ = self.transport(f"{PODS_URL}/{pod_id}", self.pod_token, "DELETE")
                if 200 <= code < 300 or code == 404:
                    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
                    print(f"Confirmed pod {pod_id} deletion at {stamp} (HTTP {code}).", flush=True)
                    return stamp
                if code < 500 and code not in {408, 429}:
                    raise RuntimeError(f"Pod {pod_id} deletion rejected: HTTP {code}")
            except OSError:
                pass
            if attempt < 2:
                time.sleep(5)
        raise RuntimeError(f"Could not confirm deletion of pod {pod_id}")

    def cancel(self, run_id: str, attempt: str) -> None:
        # A run may have been rerun while an old pod was being deleted. Never
        # cancel its new attempt. Confirm DELETE before calling this method.
        run = self.github(f"actions/runs/{run_id}")
        if run["run_attempt"] != int(attempt) or run["status"] == "completed":
            return
        code, _ = self.transport(
            f"https://api.github.com/repos/{self.repository}/actions/runs/{run_id}/cancel",
            self.github_token, "POST")
        if code != 202:
            # Completion can race the cancellation request.
            if code == 409 and self.github(f"actions/runs/{run_id}")["status"] == "completed":
                return
            raise RuntimeError(f"Pod deleted but workflow cancellation failed: HTTP {code}")
        print(f"Requested cancellation of obsolete/expired workflow {run_id}/{attempt}.", flush=True)
