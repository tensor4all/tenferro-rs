#!/usr/bin/env python3
"""Provision the cheapest compatible RunPod GPU with bounded retries.

Walks the price-ordered candidate plan from :mod:`runpod_pricing`. For each
candidate it creates a pod, then watches two trusted signals from the
GitHub-hosted side (no credential ever reaches the pod):

* the pod's own status — the startup script runs the CUDA smoke proof
  (:mod:`cuda_smoke_test`) BEFORE registering the runner and exits nonzero
  on an incompatible host, which stops the container and surfaces here as
  an exited pod;
* the org runner registry — the runner coming online means the smoke proof
  passed and the host is accepted.

Incompatible or stuck pods are deleted immediately and the next candidate
is tried, reusing the same immutable test archive (#1403) with no Cargo
compilation. Attempts are bounded by ``max_provision_attempts`` and failure
after exhaustion is explicit. Every attempt logs GPU type, hourly price,
outcome, rejection reason, and startup time so paid cost stays observable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.ci.runpod_client import (
    CreateRequest,
    CreateResult,
    PermanentRunPodError,
    RetryableRunPodError,
    RunPodError,
    _http_transport,
    _load_config,
    _redact,
    build_pod_payload,
    create_pod,
)
from scripts.ci.runpod_contract import configured_gpu_tiers
from scripts.ci.runpod_pricing import candidate_plan

# Pod desiredStatus values that mean the startup script stopped without
# registering the runner (smoke failure or setup failure).
_DEAD_POD_STATUSES = frozenset({"EXITED", "TERMINATED", "DEAD", "STOPPED"})


class ProvisionExhaustedError(RunPodError):
    """Every bounded candidate attempt failed; failure is explicit."""


@dataclasses.dataclass(frozen=True)
class ProvisionResult:
    """The accepted pod and its cost/startup observability data."""

    pod_id: str
    gpu_type_id: str
    gpu_tier: str
    cost_per_hr: float | None
    startup_seconds: float
    attempts: int
    body: bytes


def parse_cost_per_hr(body: bytes) -> float | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    for key in ("adjustedCostPerHr", "costPerHr"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def provision(
    config: Mapping[str, Any],
    plan: Sequence[tuple[str, Sequence[str]]],
    *,
    create: Callable[[CreateRequest], CreateResult],
    runner_online: Callable[[], bool],
    pod_status: Callable[[str], str | None],
    delete_pod: Callable[[str], None],
    publish_pod_id: Callable[[str], None] = lambda pod_id: None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ProvisionResult:
    """Try candidates in order until one passes the runtime smoke proof."""

    max_attempts = int(config.get("max_provision_attempts", 4))
    startup_timeout = float(config.get("startup_timeout_seconds", 600))
    poll_seconds = float(config.get("startup_poll_seconds", 10))
    attempts = 0
    last_reason = "no candidates attempted"

    for tier_name, gpu_type_ids in plan:
        if attempts >= max_attempts:
            break
        attempts += 1
        print(
            f"Provision attempt {attempts}/{max_attempts}: "
            f"candidate {tier_name} ({', '.join(gpu_type_ids)})"
        )
        try:
            result = create(
                CreateRequest(tier_name, b"", tuple(gpu_type_ids))
            )
        except RetryableRunPodError as error:
            last_reason = f"create failed: {error}"
            print(f"Candidate {tier_name} rejected before start: {last_reason}")
            continue
        publish_pod_id(result.pod_id)
        cost = parse_cost_per_hr(result.body)
        cost_text = f"${cost:.2f}/hr" if cost is not None else "unknown $/hr"
        print(
            f"Created pod {result.pod_id}: GPU {result.gpu_type_id or 'unknown'} "
            f"at {cost_text}; waiting for the smoke proof and runner."
        )

        started = monotonic()
        deadline = started + startup_timeout
        reason: str | None = None
        while True:
            if runner_online():
                startup_seconds = monotonic() - started
                print(
                    f"Runner online: pod {result.pod_id} passed the CUDA "
                    f"smoke proof in {startup_seconds:.0f}s "
                    f"(GPU {result.gpu_type_id or 'unknown'}, {cost_text})."
                )
                return ProvisionResult(
                    pod_id=result.pod_id,
                    gpu_type_id=result.gpu_type_id,
                    gpu_tier=result.gpu_tier,
                    cost_per_hr=cost,
                    startup_seconds=startup_seconds,
                    attempts=attempts,
                    body=result.body,
                )
            status = pod_status(result.pod_id)
            if status in _DEAD_POD_STATUSES:
                reason = (
                    f"pod exited before runner registration (status {status}); "
                    "CUDA smoke proof or startup failed"
                )
                break
            if monotonic() >= deadline:
                reason = f"startup timed out after {startup_timeout:.0f}s"
                break
            sleep(poll_seconds)

        elapsed = monotonic() - started
        estimate = (
            f"; estimated paid time {elapsed:.0f}s at {cost_text}"
            if cost is not None
            else ""
        )
        print(
            f"Rejecting candidate {tier_name} (pod {result.pod_id}, GPU "
            f"{result.gpu_type_id or 'unknown'}): {reason}{estimate}"
        )
        delete_pod(result.pod_id)
        print(f"Deleted incompatible pod {result.pod_id} before any test setup.")
        last_reason = reason or "unknown failure"

    raise ProvisionExhaustedError(
        f"all {attempts} bounded provision attempts failed; last: {last_reason}"
    )


def _runner_online_checker(
    token: str, org: str, label: str
) -> Callable[[], bool]:
    def check() -> bool:
        request = urllib.request.Request(
            f"https://api.github.com/orgs/{org}/actions/runners?per_page=100",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tenferro-ci-runpod-provision/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                payload = json.loads(response.read())
        except (OSError, json.JSONDecodeError) as error:
            print(f"Runner registry poll failed (transient): {error}")
            return False
        runners = payload.get("runners", [])
        for runner in runners:
            if runner.get("status") != "online":
                continue
            if any(
                entry.get("name") == label
                for entry in runner.get("labels", [])
            ):
                return True
        return False

    return check


def _pod_api(api_url: str, api_key: str):
    def request(pod_id: str, method: str) -> tuple[int, bytes]:
        req = urllib.request.Request(
            f"{api_url}/{pod_id}",
            method=method,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "tenferro-ci-runpod-provision/1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30.0) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def status(pod_id: str) -> str | None:
        code, body = request(pod_id, "GET")
        if code != 200:
            print(f"Pod status poll returned HTTP {code} (transient).")
            return None
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return None
        value = payload.get("desiredStatus") if isinstance(payload, Mapping) else None
        return str(value) if value else None

    def delete(pod_id: str) -> None:
        code, _body = request(pod_id, "DELETE")
        print(f"RunPod delete HTTP status for {pod_id}: {code}")

    return status, delete


def _publish(result: ProvisionResult) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        for value in (result.pod_id, result.gpu_type_id, result.gpu_tier):
            if "\n" in value or "\r" in value:
                raise PermanentRunPodError("unsafe GitHub output value")
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"pod_id={result.pod_id}\n")
            output.write(f"gpu_type_id={result.gpu_type_id}\n")
            output.write(f"gpu_tier={result.gpu_tier}\n")
            if result.cost_per_hr is not None:
                output.write(f"gpu_cost_per_hr={result.cost_per_hr:.4f}\n")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        cost = (
            f"${result.cost_per_hr:.2f}/hr"
            if result.cost_per_hr is not None
            else "unknown"
        )
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(
                f"RunPod GPU: {result.gpu_type_id} ({result.gpu_tier}) at "
                f"{cost}; startup {result.startup_seconds:.0f}s after "
                f"{result.attempts} attempt(s)\n"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--startup-script", type=Path, required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--response-file", type=Path, required=True)
    parser.add_argument(
        "--pod-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Non-secret startup-script environment (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = _load_config(args.config)
        api_key = os.environ.get("RUNPOD_API_KEY")
        jit_config = os.environ.get("RUNNER_JIT_CONFIG")
        gh_token = os.environ.get("PROVISION_GH_TOKEN")
        org = os.environ.get("PROVISION_ORG")
        label = os.environ.get("PROVISION_RUNNER_LABEL")
        if not api_key or not jit_config:
            raise PermanentRunPodError(
                "RUNPOD_API_KEY and RUNNER_JIT_CONFIG are required"
            )
        if not gh_token or not org or not label:
            raise PermanentRunPodError(
                "PROVISION_GH_TOKEN, PROVISION_ORG, and "
                "PROVISION_RUNNER_LABEL are required"
            )
        startup_script = args.startup_script.read_text(encoding="utf-8")
        extra_env: dict[str, str] = {}
        for entry in args.pod_env:
            key, sep, value = entry.partition("=")
            if not sep or not key:
                raise PermanentRunPodError(
                    f"--pod-env entries must be KEY=VALUE, got {entry!r}"
                )
            extra_env[key] = value
        plan = candidate_plan(config, list(configured_gpu_tiers(config)))
        transport = _http_transport(str(config["api_url"]), api_key)

        def create(request: CreateRequest) -> CreateResult:
            payload = build_pod_payload(
                config,
                args.image_name,
                startup_script,
                jit_config,
                request.gpu_type_ids,
                extra_env=extra_env,
            )
            print(
                f"RunPod request candidate={request.tier_name}, secrets "
                "redacted: " + json.dumps(_redact(payload), sort_keys=True)
            )
            return create_pod(
                config,
                [
                    CreateRequest(
                        request.tier_name,
                        json.dumps(payload).encode(),
                        request.gpu_type_ids,
                    )
                ],
                transport=transport,
                secrets=(api_key, jit_config),
            )

        pod_status, delete_pod = _pod_api(str(config["api_url"]), api_key)

        def publish_pod_id(pod_id: str) -> None:
            output_path = os.environ.get("GITHUB_OUTPUT")
            if output_path and "\n" not in pod_id and "\r" not in pod_id:
                with open(output_path, "a", encoding="utf-8") as output:
                    output.write(f"pod_id={pod_id}\n")

        result = provision(
            config,
            plan,
            create=create,
            runner_online=_runner_online_checker(gh_token, org, label),
            pod_status=pod_status,
            delete_pod=delete_pod,
            publish_pod_id=publish_pod_id,
        )
        args.response_file.write_bytes(result.body)
        _publish(result)
        print(f"Provisioned RunPod pod: {result.pod_id}")
    except (
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
        RunPodError,
    ) as error:
        print(f"RunPod provision failed: {str(error)!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
