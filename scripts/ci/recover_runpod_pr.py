#!/usr/bin/env python3
"""Recover one PR's GPU gate through the trusted main RunPod workflow."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Callable


Runner = Callable[..., subprocess.CompletedProcess[str]]


def build_dispatch_command(pr_number: int, wait: bool = False) -> list[str]:
    """Build the invariant main-ref dispatch command for a positive PR number."""

    del wait  # Waiting is a post-dispatch action and never changes trust inputs.
    if pr_number <= 0:
        raise ValueError("PR number must be positive")
    return [
        "gh",
        "workflow",
        "run",
        "runpod-gpu-test.yml",
        "--ref",
        "main",
        "-f",
        f"pr_number={pr_number}",
    ]


def recover_pr(
    pr_number: int, *, wait: bool, runner: Runner = subprocess.run
) -> str:
    """Authenticate, dispatch, return the run URL, and optionally watch it."""

    common = {"check": True, "text": True, "capture_output": True}
    runner(["gh", "auth", "status"], **common)
    result = runner(build_dispatch_command(pr_number, wait), **common)
    url = next(
        (
            token
            for token in result.stdout.split()
            if token.startswith("https://") and "/actions/runs/" in token
        ),
        "",
    )
    if not url:
        raise RuntimeError("gh workflow run did not return a run URL")
    if wait:
        run_id = url.rstrip("/").rsplit("/", 1)[-1]
        runner(
            ["gh", "run", "watch", run_id, "--exit-status"],
            **common,
        )
    return url


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_number", type=int)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        command = build_dispatch_command(args.pr_number, args.wait)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        url = recover_pr(args.pr_number, wait=args.wait)
        print(url)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"RunPod PR recovery failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
