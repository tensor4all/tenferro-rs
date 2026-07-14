#!/usr/bin/env python3
"""Create RunPod pods with bounded, status-aware retries and redacted output."""

from __future__ import annotations

import argparse
import dataclasses
import enum
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
class CreateResult:
    """Validated result of a successful pod creation request."""

    pod_id: str
    gpu_type_id: str
    body: bytes


Transport = Callable[[bytes], tuple[int, Mapping[str, str], bytes]]


def classify_http_status(status: int) -> RetryClass:
    """Classify only timeout, rate-limit, and server failures as retryable."""

    if status in (408, 429) or 500 <= status < 600:
        return RetryClass.RETRYABLE
    return RetryClass.PERMANENT


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
) -> dict[str, object]:
    """Build the reviewed SECURE Cloud request from repository configuration."""

    return {
        "cloudType": config["cloud_type"],
        "computeType": config["compute_type"],
        "name": f"tenferro-rs-gpu-ci-{os.environ.get('GITHUB_RUN_ID', 'local')}",
        "imageName": image_name,
        "gpuTypeIds": config["gpu_type_ids"],
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


def parse_create_response(status: int, body: bytes) -> CreateResult:
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
    return CreateResult(pod_id=pod_id, gpu_type_id=gpu_type_id, body=body)


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
    payload: bytes,
    *,
    transport: Transport,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    monotonic: Callable[[], float] = time.monotonic,
    secrets: Sequence[str] = (),
) -> CreateResult:
    """Create a pod within both an attempt budget and a wall-clock deadline."""

    max_attempts = int(config["max_create_attempts"])
    base = float(config["retry_base_seconds"])
    cap = float(config["retry_max_seconds"])
    deadline = monotonic() + float(config["create_deadline_seconds"])
    for attempt in range(1, max_attempts + 1):
        headers: Mapping[str, str] = {}
        try:
            status, headers, body = transport(payload)
        except OSError as error:
            failure: RunPodError = RetryableRunPodError(
                f"transport failure: {error}"
            )
        else:
            if 200 <= status < 300:
                return parse_create_response(status, body)
            message = redacted_error_message(body, secrets=secrets)
            if classify_http_status(status) is RetryClass.PERMANENT:
                raise PermanentRunPodError(f"RunPod HTTP {status}: {message}")
            failure = RetryableRunPodError(f"RunPod HTTP {status}: {message}")

        now = monotonic()
        if attempt == max_attempts or now >= deadline:
            raise failure
        delay = _retry_after(headers)
        if delay is None:
            delay = backoff_seconds(attempt, base=base, cap=cap, jitter=jitter)
        sleep(min(delay, max(0.0, deadline - now)))
    raise AssertionError("unreachable retry loop")


def _http_transport(url: str, api_key: str) -> Transport:
    def send(payload: bytes) -> tuple[int, Mapping[str, str], bytes]:
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
            with urllib.request.urlopen(request, timeout=60) as response:
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
        payload = build_pod_payload(
            config,
            args.image_name,
            args.startup_script.read_text(encoding="utf-8"),
            jit_config,
        )
        print(
            "RunPod request payload, secrets redacted: "
            + json.dumps(_redact(payload), sort_keys=True)
        )
        encoded = json.dumps(payload).encode()
        result = create_pod(
            config,
            encoded,
            transport=_http_transport(str(config["api_url"]), api_key),
            secrets=(api_key, jit_config),
        )
        args.response_file.write_bytes(result.body)
        if output_path := os.environ.get("GITHUB_OUTPUT"):
            with Path(output_path).open("a", encoding="utf-8") as output:
                output.write(f"pod_id={result.pod_id}\n")
                output.write(f"gpu_type_id={result.gpu_type_id}\n")
        print(f"Created RunPod pod: {result.pod_id}")
        print(f"RunPod assigned GPU type: {result.gpu_type_id or 'unknown'}")
    except (OSError, KeyError, ValueError, json.JSONDecodeError, RunPodError) as error:
        print(f"RunPod create failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
