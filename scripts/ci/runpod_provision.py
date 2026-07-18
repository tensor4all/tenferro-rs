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
    AssignedGpuError,
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
# registering the runner (smoke failure or setup failure). desiredStatus
# alone is NOT sufficient: RunPod keeps it at RUNNING after the container
# exits, so container death is detected through the GraphQL runtime field
# (present while the container runs, null once it stops).
_DEAD_POD_STATUSES = frozenset({"EXITED", "TERMINATED", "DEAD", "STOPPED"})


@dataclasses.dataclass(frozen=True)
class PodState:
    """One observation of a pod: desired status plus container liveness.

    ``has_runtime`` is None when the observation could not determine
    container state (transient poll failure); True while the container is
    running; False once GraphQL reports no runtime for the pod.
    """

    desired: str | None
    has_runtime: bool | None


class ProvisionExhaustedError(RunPodError):
    """Every bounded candidate attempt failed; failure is explicit."""


class PodLeakError(RunPodError):
    """A pod could not be confirmed deleted; creation must stop.

    The leaked pod id stays published to GITHUB_OUTPUT so the workflow's
    delete-on-failure safety net and cleanup job can still reach it.
    """


@dataclasses.dataclass(frozen=True)
class ProvisionResult:
    """The accepted pod and its cost/startup observability data."""

    pod_id: str
    gpu_type_id: str
    gpu_tier: str
    runner_label: str
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
    label_prefix: str,
    mint_runner: Callable[[str], str],
    create: Callable[[CreateRequest, str], CreateResult],
    runner_online: Callable[[str], bool],
    pod_status: Callable[[str], PodState],
    delete_pod: Callable[[str], bool],
    publish_pod_id: Callable[[str], None] = lambda pod_id: None,
    keep_failed_pods: bool = False,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ProvisionResult:
    """Try candidates in order until one passes the runtime smoke proof.

    Every attempt mints its OWN single-use JIT runner config under a fresh
    per-attempt label (``mint_runner(label)``): JIT configs cannot be
    replayed after a previous candidate registered with them, and a shared
    label would let a stale ``online`` record from an earlier pod accept a
    new pod that never passed the smoke proof.

    ``delete_pod`` must return True only when deletion is confirmed; an
    unconfirmed deletion stops the loop with :class:`PodLeakError` so a
    possibly-alive paid pod is never followed by another creation.
    """

    max_attempts = int(config.get("max_provision_attempts", 4))
    startup_timeout = float(config.get("startup_timeout_seconds", 600))
    poll_seconds = float(config.get("startup_poll_seconds", 10))
    attempts = 0
    last_reason = "no candidates attempted"

    kept_pods: list[str] = []

    def reject_and_delete(pod_id: str, description: str) -> None:
        if keep_failed_pods:
            kept_pods.append(pod_id)
            print(
                f"DEBUG MODE: keeping failed pod {pod_id} ({description}) "
                "for console-log inspection. It keeps billing until deleted "
                "manually in the RunPod dashboard."
            )
            return
        if delete_pod(pod_id):
            print(f"Deleted pod {pod_id} before any test setup.")
            return
        raise PodLeakError(
            f"could not confirm deletion of pod {pod_id} ({description}); "
            "stopping so the workflow safety net can clean it up"
        )

    for tier_name, gpu_type_ids in plan:
        if attempts >= max_attempts:
            break
        attempts += 1
        label = f"{label_prefix}-c{attempts}"
        print(
            f"Provision attempt {attempts}/{max_attempts}: "
            f"candidate {tier_name} ({', '.join(gpu_type_ids)}) "
            f"as runner {label}"
        )
        jit_config = mint_runner(label)
        try:
            result = create(
                CreateRequest(tier_name, b"", tuple(gpu_type_ids)),
                jit_config,
            )
        except AssignedGpuError as error:
            # The pod exists but its assigned GPU could not be verified:
            # publish it for the workflow safety net, then delete it here.
            publish_pod_id(error.result.pod_id)
            print(
                f"Candidate {tier_name} created pod {error.result.pod_id} "
                f"with an unverifiable GPU assignment: {error}"
            )
            reject_and_delete(error.result.pod_id, "unverifiable GPU assignment")
            last_reason = f"unverifiable GPU assignment: {error}"
            continue
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
        runtime_seen = False
        while True:
            # Check the pod BEFORE trusting the runner registry: the two
            # signals are independently eventually consistent, and a stale
            # online record (or a runner that registered and died) must not
            # accept a dead pod.
            state = pod_status(result.pod_id)
            if state.desired in _DEAD_POD_STATUSES:
                reason = (
                    f"pod exited before runner registration (status "
                    f"{state.desired}); CUDA smoke proof or startup failed"
                )
                break
            if state.has_runtime:
                runtime_seen = True
            elif runtime_seen and state.has_runtime is False:
                # The container ran and then stopped without registering
                # the runner: the startup script (usually the CUDA smoke
                # proof) failed. desiredStatus stays RUNNING in this case,
                # so this is the authoritative fast-fail signal.
                reason = (
                    "container stopped before runner registration; "
                    "CUDA smoke proof or startup failed"
                )
                break
            if runner_online(label) and state.desired is not None:
                startup_seconds = monotonic() - started
                print(
                    f"Runner {label} online: pod {result.pod_id} passed the "
                    f"CUDA smoke proof in {startup_seconds:.0f}s "
                    f"(GPU {result.gpu_type_id or 'unknown'}, {cost_text})."
                )
                return ProvisionResult(
                    pod_id=result.pod_id,
                    gpu_type_id=result.gpu_type_id,
                    gpu_tier=result.gpu_tier,
                    runner_label=label,
                    cost_per_hr=cost,
                    startup_seconds=startup_seconds,
                    attempts=attempts,
                    body=result.body,
                )
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
        reject_and_delete(result.pod_id, reason or "startup failure")
        last_reason = reason or "unknown failure"

    kept = f"; kept debug pods: {', '.join(kept_pods)}" if kept_pods else ""
    raise ProvisionExhaustedError(
        f"all {attempts} bounded provision attempts failed; "
        f"last: {last_reason}{kept}"
    )


def _runner_online_checker(token: str, org: str) -> Callable[[str], bool]:
    def check(label: str) -> bool:
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


def _runner_minter(
    token: str, org: str, runner_group_id: int
) -> Callable[[str], str]:
    """Mint one single-use JIT runner config per provision attempt."""

    def mint(label: str) -> str:
        body = json.dumps(
            {
                "name": label,
                "runner_group_id": runner_group_id,
                "labels": ["self-hosted", label],
                "work_folder": "_work",
            }
        ).encode()
        request = urllib.request.Request(
            f"https://api.github.com/orgs/{org}/actions/runners/generate-jitconfig",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "tenferro-ci-runpod-provision/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise PermanentRunPodError(
                f"generate-jitconfig returned HTTP {error.code}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise PermanentRunPodError(
                f"generate-jitconfig failed: {error}"
            ) from error
        jit_config = payload.get("encoded_jit_config")
        if not isinstance(jit_config, str) or not jit_config:
            raise PermanentRunPodError(
                "generate-jitconfig response is missing encoded_jit_config"
            )
        # Mask before the value can appear in any later log line.
        print(f"::add-mask::{jit_config}")
        print(f"Minted JIT runner config for {label}.")
        return jit_config

    return mint


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
        except OSError as error:
            # URLError, timeouts, DNS hiccups: transient. Status 0 makes
            # status() report None (keep waiting) and delete() retry, so a
            # single poll failure never aborts the bounded provision loop
            # or condemns a healthy pod.
            print(f"RunPod pod API transport failure (transient): {error}")
            return 0, b""

    def delete(pod_id: str, *, retries: int = 3) -> bool:
        """Delete a pod and only report success when it is confirmed gone."""

        for attempt in range(1, retries + 1):
            code, _body = request(pod_id, "DELETE")
            print(
                f"RunPod delete HTTP status for {pod_id}: {code} "
                f"(attempt {attempt}/{retries})"
            )
            if 200 <= code < 300 or code == 404:
                return True
            get_code, _get_body = request(pod_id, "GET")
            if get_code == 404:
                print(f"Pod {pod_id} confirmed gone after delete attempt.")
                return True
            if attempt < retries:
                time.sleep(5.0)
        return False

    return delete


def _pod_state_checker(graphql_url: str, api_key: str) -> Callable[[str], PodState]:
    """Observe desiredStatus AND container liveness through GraphQL.

    The REST desiredStatus stays RUNNING after the container exits; only
    the GraphQL ``runtime`` object (null once the container stops) tells
    whether the startup script is still alive.
    """

    query = (
        "query Pod($input: PodFilter!) { pod(input: $input) "
        "{ desiredStatus runtime { uptimeInSeconds } } }"
    )

    def check(pod_id: str) -> PodState:
        body = json.dumps(
            {"query": query, "variables": {"input": {"podId": pod_id}}}
        ).encode()
        request = urllib.request.Request(
            graphql_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "tenferro-ci-runpod-provision/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                payload = json.loads(response.read())
        except (OSError, json.JSONDecodeError) as error:
            print(f"Pod state poll failed (transient): {error}")
            return PodState(desired=None, has_runtime=None)
        data = payload.get("data") if isinstance(payload, Mapping) else None
        pod = data.get("pod") if isinstance(data, Mapping) else None
        if not isinstance(pod, Mapping):
            if isinstance(payload, Mapping) and payload.get("errors"):
                print(f"Pod state query errors (transient): {payload['errors']!r}")
            return PodState(desired=None, has_runtime=None)
        desired = pod.get("desiredStatus")
        return PodState(
            desired=str(desired) if desired else None,
            has_runtime=pod.get("runtime") is not None,
        )

    return check


def _publish(result: ProvisionResult) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        for value in (
            result.pod_id,
            result.gpu_type_id,
            result.gpu_tier,
            result.runner_label,
        ):
            if "\n" in value or "\r" in value:
                raise PermanentRunPodError("unsafe GitHub output value")
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"pod_id={result.pod_id}\n")
            output.write(f"gpu_type_id={result.gpu_type_id}\n")
            output.write(f"gpu_tier={result.gpu_tier}\n")
            output.write(f"runner_label={result.runner_label}\n")
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
        gh_token = os.environ.get("PROVISION_GH_TOKEN")
        org = os.environ.get("PROVISION_ORG")
        label_prefix = os.environ.get("PROVISION_RUNNER_LABEL")
        runner_group_id = int(os.environ.get("PROVISION_RUNNER_GROUP_ID", "1"))
        if not api_key:
            raise PermanentRunPodError("RUNPOD_API_KEY is required")
        if not gh_token or not org or not label_prefix:
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

        def create(request: CreateRequest, jit_config: str) -> CreateResult:
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

        delete_pod = _pod_api(str(config["api_url"]), api_key)
        pod_status = _pod_state_checker(
            str(config.get("graphql_url", "https://api.runpod.io/graphql")),
            api_key,
        )
        keep_failed_pods = (
            os.environ.get("PROVISION_KEEP_FAILED_PODS", "").lower() == "true"
        )

        def publish_pod_id(pod_id: str) -> None:
            output_path = os.environ.get("GITHUB_OUTPUT")
            if output_path and "\n" not in pod_id and "\r" not in pod_id:
                with open(output_path, "a", encoding="utf-8") as output:
                    output.write(f"pod_id={pod_id}\n")

        result = provision(
            config,
            plan,
            label_prefix=label_prefix,
            mint_runner=_runner_minter(gh_token, org, runner_group_id),
            create=create,
            runner_online=_runner_online_checker(gh_token, org),
            pod_status=pod_status,
            delete_pod=delete_pod,
            publish_pod_id=publish_pod_id,
            keep_failed_pods=keep_failed_pods,
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
