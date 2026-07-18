#!/usr/bin/env python3
"""Locate a reusable CUDA/PJRT archive artifact produced by a trusted run.

GPU allocation retries dispatch a fresh workflow run. Instead of rebuilding
the identical archive, the hosted archive job asks this script for an
existing, unexpired Actions artifact whose name encodes the same content
key. Only artifacts produced by trusted workflow definitions are eligible:
the artifact's producing run must belong to this repository, must have run a
trusted workflow file, and must have been triggered by an event whose
workflow definition comes from the default branch (`workflow_run`,
`workflow_dispatch`, `push`, `schedule`). Artifacts uploaded by
`pull_request`-event runs execute PR-controlled workflow definitions and are
never eligible, so an attacker cannot smuggle content into the reuse path by
uploading a name-colliding artifact from a PR.

The content key itself is computed by the trusted workflow from its own
checkout, so a name match implies the build inputs are identical.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

TRUSTED_WORKFLOW_PATHS = (
    ".github/workflows/runpod-gpu-test.yml",
    ".github/workflows/ci-cache-publish.yml",
)
TRUSTED_PRODUCER_EVENTS = ("workflow_run", "workflow_dispatch", "push", "schedule")
# For directly-triggered events the run's own branch is the workflow
# definition source, so it must be the default branch. workflow_run events
# always resolve their definition on the default branch regardless of the
# reported head branch, so they are exempt from this check.
DEFAULT_BRANCH = "main"
BRANCH_CHECKED_EVENTS = ("workflow_dispatch", "push", "schedule")

# One page of the newest artifacts is enough: reusable candidates are
# recent by construction (7-day retention) and sorted newest-first.
ARTIFACT_PAGE_SIZE = 100

Transport = Callable[[str], tuple[int, bytes]]


class FinderError(RuntimeError):
    """A trust or protocol violation while locating a reusable artifact."""


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinderError(f"{context} must be a JSON object")
    return value


def _get_json(transport: Transport, url: str) -> Mapping[str, Any]:
    status, body = transport(url)
    if status != 200:
        raise FinderError(f"GET {url} returned HTTP {status}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise FinderError(f"GET {url} returned invalid JSON: {error}") from error
    return _as_mapping(payload, f"GET {url} response")


def is_trusted_producer_run(
    run: Mapping[str, Any],
    repository: str,
    *,
    current_run_id: int,
) -> tuple[bool, str]:
    """Check that a workflow run may act as an artifact reuse source."""

    run_id = run.get("id")
    if run_id == current_run_id:
        return False, "is the current run"
    head_repository = _as_mapping(
        run.get("head_repository") or {}, "run.head_repository"
    )
    if head_repository.get("full_name") != repository:
        return False, f"head repository {head_repository.get('full_name')!r} is not {repository!r}"
    if run.get("path") not in TRUSTED_WORKFLOW_PATHS:
        return False, f"workflow path {run.get('path')!r} is not trusted"
    event = run.get("event")
    if event not in TRUSTED_PRODUCER_EVENTS:
        return False, f"producer event {event!r} is not trusted"
    if event in BRANCH_CHECKED_EVENTS and run.get("head_branch") != DEFAULT_BRANCH:
        return (
            False,
            f"{event} run on branch {run.get('head_branch')!r} is not the default branch",
        )
    return True, "trusted"


def find_reusable_artifact(
    transport: Transport,
    repository: str,
    artifact_name: str,
    *,
    current_run_id: int,
) -> Mapping[str, Any] | None:
    """Return the newest trusted, unexpired artifact with this exact name."""

    query = urllib.parse.urlencode(
        {"name": artifact_name, "per_page": ARTIFACT_PAGE_SIZE}
    )
    listing = _get_json(
        transport,
        f"https://api.github.com/repos/{repository}/actions/artifacts?{query}",
    )
    artifacts = listing.get("artifacts")
    if not isinstance(artifacts, Sequence):
        raise FinderError("artifact listing must contain an 'artifacts' array")

    for artifact_obj in artifacts:
        artifact = _as_mapping(artifact_obj, "artifact entry")
        if artifact.get("name") != artifact_name:
            continue
        if artifact.get("expired"):
            print(f"Skipping expired artifact {artifact.get('id')}.")
            continue
        run_summary = _as_mapping(
            artifact.get("workflow_run") or {}, "artifact.workflow_run"
        )
        run_id = run_summary.get("id")
        if not isinstance(run_id, int):
            print(f"Skipping artifact {artifact.get('id')} without a producing run id.")
            continue
        run = _get_json(
            transport,
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}",
        )
        trusted, reason = is_trusted_producer_run(
            run, repository, current_run_id=current_run_id
        )
        if not trusted:
            print(f"Skipping artifact {artifact.get('id')} from run {run_id}: {reason}.")
            continue
        print(
            f"Reusable artifact {artifact.get('id')} found: "
            f"run {run_id} ({run.get('event')}, {run.get('path')}), "
            f"created {artifact.get('created_at')}, "
            f"{artifact.get('size_in_bytes')} bytes."
        )
        return {
            "artifact_id": artifact["id"],
            "run_id": run_id,
            "size_in_bytes": artifact.get("size_in_bytes"),
        }
    return None


def _github_transport(token: str) -> Transport:
    def send(url: str) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tenferro-ci-archive-reuse/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    return send


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--current-run-id", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="Path receiving GitHub-output style key=value lines",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        found = find_reusable_artifact(
            _github_transport(args.token),
            args.repository,
            args.artifact_name,
            current_run_id=args.current_run_id,
        )
    except FinderError as error:
        # Reuse is an optimization: a lookup failure must never fail the
        # build path, only fall back to it.
        print(f"Artifact reuse lookup failed; falling back to build: {error}")
        found = None
    with open(args.output, "a", encoding="utf-8") as output:
        if found is None:
            output.write("found=false\n")
            print("No reusable trusted artifact found.")
        else:
            output.write("found=true\n")
            output.write(f"artifact_id={found['artifact_id']}\n")
            output.write(f"run_id={found['run_id']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
