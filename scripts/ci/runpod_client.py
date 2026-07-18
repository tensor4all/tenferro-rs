#!/usr/bin/env python3
"""Create RunPod pods with bounded, status-aware retries and redacted output."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import html
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.ci.runpod_contract import ContractError, configured_gpu_tiers


class RetryClass(enum.Enum):
    """Whether an HTTP failure may be retried safely."""

    RETRYABLE = enum.auto()
    PERMANENT = enum.auto()


class RunPodError(RuntimeError):
    """Base error for pod creation."""


class RetryableRunPodError(RunPodError):
    """A transient failure exhausted the configured retry budget."""


class PermanentRunPodError(RunPodError):
    """A request or protocol failure that must not be retried."""


@dataclasses.dataclass(frozen=True)
class CreateRequest:
    """One reviewed GPU tier and its encoded RunPod request."""

    tier_name: str
    payload: bytes
    gpu_type_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class CreateResult:
    """Validated result of a successful pod creation request."""

    pod_id: str
    gpu_type_id: str
    gpu_tier: str
    body: bytes


class AssignedGpuError(PermanentRunPodError):
    """A created pod reported an unsafe or unverifiable assigned GPU."""

    def __init__(self, message: str, result: CreateResult) -> None:
        super().__init__(message)
        self.result = result


Transport = Callable[
    [bytes, float], tuple[int, Mapping[str, str], bytes]
]

_CAPACITY_MESSAGES = (
    "does not have the resources to deploy your pod",
    "no available machine",
)


def classify_http_status(status: int) -> RetryClass:
    """Classify only timeout, rate-limit, and server failures as retryable."""

    if status in (408, 429) or 500 <= status < 600:
        return RetryClass.RETRYABLE
    return RetryClass.PERMANENT


def is_capacity_failure(status: int, body: bytes) -> bool:
    """Return whether RunPod reported exhausted machine capacity."""

    if not 500 <= status < 600:
        return False
    message = body.decode("utf-8", errors="replace").lower()
    return any(marker in message for marker in _CAPACITY_MESSAGES)


def backoff_seconds(
    attempt: int, *, base: float, cap: float, jitter: Callable[[], float]
) -> float:
    """Return full-jitter exponential backoff bounded before jitter."""

    if attempt < 1:
        raise ValueError("attempt must be positive")
    return min(cap, base * (2 ** (attempt - 1))) * jitter()


def build_pod_payload(
    config: Mapping[str, object],
    image_name: str,
    startup_script: str,
    jit_config: str,
    gpu_type_ids: Sequence[str],
) -> dict[str, object]:
    """Build the reviewed SECURE Cloud request from repository configuration."""

    return {
        "cloudType": config["cloud_type"],
        "computeType": config["compute_type"],
        "allowedCudaVersions": config["allowed_cuda_versions"],
        "name": f"tenferro-rs-gpu-ci-{os.environ.get('GITHUB_RUN_ID', 'local')}",
        "imageName": image_name,
        "gpuTypeIds": list(gpu_type_ids),
        "gpuTypePriority": config["gpu_type_priority"],
        "gpuCount": config["gpu_count"],
        "containerDiskInGb": config["container_disk_gb"],
        "volumeInGb": config["volume_gb"],
        "volumeMountPath": config["volume_mount_path"],
        "interruptible": False,
        "ports": [],
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [startup_script],
        "env": {
            "RUNNER_JIT_CONFIG": jit_config,
            "RUNNER_ALLOW_RUNASROOT": "1",
        },
    }


def _redact(value: object) -> object:
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if not isinstance(value, dict):
        return value
    redacted: dict[str, object] = {}
    for key, item in value.items():
        if key in {"RUNNER_JIT_CONFIG", "PUBLIC_KEY"}:
            redacted[key] = "***redacted***"
        elif key == "dockerStartCmd":
            redacted[key] = ["***redacted-startup-script***"]
        else:
            redacted[key] = _redact(item)
    return redacted


def redacted_error_message(
    body: bytes, *, secrets: Sequence[str] = ()
) -> str:
    """Render a bounded response diagnostic without request secrets."""

    try:
        parsed: object = json.loads(body)
        message = json.dumps(_redact(parsed), sort_keys=True)
    except (UnicodeDecodeError, json.JSONDecodeError):
        message = body.decode("utf-8", errors="replace")
    for secret in secrets:
        if secret:
            message = message.replace(secret, "***redacted***")
    return message[:4000]


def parse_create_response(
    status: int, body: bytes, *, gpu_tier: str = ""
) -> CreateResult:
    """Validate a successful create response and extract stable fields."""

    try:
        value: Any = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermanentRunPodError(
            f"RunPod HTTP {status} returned invalid JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise PermanentRunPodError(
            f"RunPod HTTP {status} response must be an object"
        )
    pod_id = value.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise PermanentRunPodError(
            f"RunPod HTTP {status} response is missing pod id"
        )
    gpu_type_id = ""
    machine = value.get("machine")
    if isinstance(machine, Mapping):
        direct = machine.get("gpuTypeId")
        if isinstance(direct, str):
            gpu_type_id = direct
        else:
            gpu_type = machine.get("gpuType")
            if isinstance(gpu_type, Mapping) and isinstance(
                gpu_type.get("id"), str
            ):
                gpu_type_id = gpu_type["id"]
    return CreateResult(
        pod_id=pod_id,
        gpu_type_id=gpu_type_id,
        gpu_tier=gpu_tier,
        body=body,
    )


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = next(
        (item for key, item in headers.items() if key.lower() == "retry-after"),
        None,
    )
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        return None
    return max(0.0, delay)


def create_pod(
    config: Mapping[str, object],
    requests: Sequence[CreateRequest],
    *,
    transport: Transport,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    monotonic: Callable[[], float] = time.monotonic,
    secrets: Sequence[str] = (),
) -> CreateResult:
    """Create a pod within both an attempt budget and a wall-clock deadline."""

    attempts_per_tier = int(config.get("same_tier_retries", 1)) + 1
    base = float(config["retry_base_seconds"])
    cap = float(config["retry_max_seconds"])
    deadline = monotonic() + float(config["create_deadline_seconds"])
    for tier_index, request in enumerate(requests):
        for attempt in range(1, attempts_per_tier + 1):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RetryableRunPodError(
                    "RunPod create deadline expired"
                )
            print(
                f"RunPod create tier={request.tier_name} "
                f"tier_attempt={attempt}/{attempts_per_tier}"
            )
            headers: Mapping[str, str] = {}
            try:
                status, headers, body = transport(
                    request.payload, remaining
                )
            except OSError as error:
                failure: RunPodError = RetryableRunPodError(
                    f"transport failure: {error}"
                )
            else:
                if 200 <= status < 300:
                    result = parse_create_response(
                        status, body, gpu_tier=request.tier_name
                    )
                    if request.gpu_type_ids and not result.gpu_type_id:
                        raise AssignedGpuError(
                            "RunPod response is missing assigned GPU for "
                            f"selected tier {request.tier_name}",
                            result,
                        )
                    if (
                        result.gpu_type_id
                        and request.gpu_type_ids
                        and result.gpu_type_id not in request.gpu_type_ids
                    ):
                        raise AssignedGpuError(
                            "RunPod assigned GPU outside selected tier",
                            result,
                        )
                    return result
                message = redacted_error_message(body, secrets=secrets)
                if is_capacity_failure(status, body):
                    failure = RetryableRunPodError(
                        f"RunPod capacity unavailable: {message}"
                    )
                    if monotonic() >= deadline:
                        raise failure
                    if tier_index + 1 < len(requests):
                        print(
                            "RunPod capacity unavailable in tier "
                            f"{request.tier_name}; trying "
                            f"{requests[tier_index + 1].tier_name}; "
                            f"error={str(failure)!r}"
                        )
                        break
                    raise failure
                if classify_http_status(status) is RetryClass.PERMANENT:
                    raise PermanentRunPodError(
                        f"RunPod HTTP {status}: {message}"
                    )
                failure = RetryableRunPodError(
                    f"RunPod HTTP {status}: {message}"
                )

            now = monotonic()
            if attempt == attempts_per_tier or now >= deadline:
                raise failure
            delay = _retry_after(headers)
            if delay is None:
                delay = backoff_seconds(
                    attempt, base=base, cap=cap, jitter=jitter
                )
            bounded_delay = min(delay, max(0.0, deadline - now))
            print(
                f"RunPod transient failure in tier {request.tier_name}; "
                f"retrying after {bounded_delay:.1f}s: {str(failure)!r}"
            )
            sleep(bounded_delay)
    raise AssertionError("unreachable retry loop")


def publish_github_result(
    result: CreateResult,
    *,
    output_path: Path | None,
    summary_path: Path | None,
) -> None:
    """Publish the selected pod, GPU, and tier to GitHub Actions."""

    for name, value in (
        ("pod_id", result.pod_id),
        ("gpu_type_id", result.gpu_type_id),
        ("gpu_tier", result.gpu_tier),
    ):
        if "\n" in value or "\r" in value:
            raise PermanentRunPodError(
                f"unsafe GitHub output value for {name}"
            )
    if output_path is not None:
        with output_path.open("a", encoding="utf-8") as output:
            output.write(f"pod_id={result.pod_id}\n")
            output.write(f"gpu_type_id={result.gpu_type_id}\n")
            output.write(f"gpu_tier={result.gpu_tier}\n")
    if summary_path is not None:
        gpu_tier = html.escape(result.gpu_tier).replace("`", "&#96;")
        gpu_type_id = html.escape(
            result.gpu_type_id or "unknown"
        ).replace("`", "&#96;")
        with summary_path.open("a", encoding="utf-8") as summary:
            summary.write("### RunPod GPU selection\n\n")
            summary.write(f"- Price tier: {gpu_tier}\n")
            summary.write(f"- Selected GPU: {gpu_type_id}\n")


def publish_cleanup_pod_id(
    result: CreateResult, output_path: Path | None
) -> None:
    """Publish only a validated pod ID so a rejected pod can be deleted."""

    if "\n" in result.pod_id or "\r" in result.pod_id:
        raise PermanentRunPodError("unsafe GitHub output value for pod_id")
    if output_path is not None:
        with output_path.open("a", encoding="utf-8") as output:
            output.write(f"pod_id={result.pod_id}\n")


def _http_transport(url: str, api_key: str) -> Transport:
    def send(
        payload: bytes, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "tenferro-ci-runpod/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, dict(response.headers.items()), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers.items()), error.read()

    return send


def _load_config(path: Path) -> Mapping[str, object]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermanentRunPodError("RunPod configuration root must be an object")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--startup-script", type=Path, required=True)
    create.add_argument("--image-name", required=True)
    create.add_argument("--response-file", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = _load_config(args.config)
        api_key = os.environ.get("RUNPOD_API_KEY")
        jit_config = os.environ.get("RUNNER_JIT_CONFIG")
        if not api_key or not jit_config:
            raise PermanentRunPodError(
                "RUNPOD_API_KEY and RUNNER_JIT_CONFIG are required"
            )
        startup_script = args.startup_script.read_text(encoding="utf-8")
        requests: list[CreateRequest] = []
        for tier_name, gpu_type_ids in configured_gpu_tiers(config):
            payload = build_pod_payload(
                config,
                args.image_name,
                startup_script,
                jit_config,
                gpu_type_ids,
            )
            print(
                f"RunPod request tier={tier_name}, secrets redacted: "
                + json.dumps(_redact(payload), sort_keys=True)
            )
            requests.append(
                CreateRequest(
                    tier_name,
                    json.dumps(payload).encode(),
                    gpu_type_ids,
                )
            )
        try:
            result = create_pod(
                config,
                requests,
                transport=_http_transport(str(config["api_url"]), api_key),
                secrets=(api_key, jit_config),
            )
        except AssignedGpuError as error:
            args.response_file.write_bytes(error.result.body)
            output_path = os.environ.get("GITHUB_OUTPUT")
            publish_cleanup_pod_id(
                error.result,
                Path(output_path) if output_path else None,
            )
            raise
        args.response_file.write_bytes(result.body)
        output_path = os.environ.get("GITHUB_OUTPUT")
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        publish_github_result(
            result,
            output_path=Path(output_path) if output_path else None,
            summary_path=Path(summary_path) if summary_path else None,
        )
        print(f"Created RunPod pod: {result.pod_id}")
        print(f"RunPod selected price tier: {result.gpu_tier}")
        print(f"RunPod assigned GPU type: {result.gpu_type_id or 'unknown'}")
    except (
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
        ContractError,
        RunPodError,
    ) as error:
        print(f"RunPod create failed: {str(error)!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
