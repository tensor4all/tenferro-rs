#!/usr/bin/env python3
"""Standard-library-only bounded HTTP policy for hosted RunPod callers."""
# Adapted from tensor4all/tenferro-benchmark PRs #214 and #215 (main 9012f4b).
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


@dataclass
class Cooldown:
    """A process-local endpoint cooldown shared by pricing and pod observations."""

    until: float = 0.0


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


def retry_after(headers, *, wall_time=time.time):
    value = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
    if value is None:
        return None
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            delay = date.timestamp() - wall_time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


def retryable(status):
    return status in (0, 408, 429) or 500 <= status < 600


def retry_delay(headers, attempt, *, base=5.0, cap=60.0, wall_time=time.time):
    advertised = retry_after(headers, wall_time=wall_time)
    backoff = min(cap, base * 2 ** min(attempt - 1, 10))
    backoff = min(cap, backoff * random.uniform(1.0, 1.2))
    # Never shorten the provider's minimum, even when it exceeds our backoff cap.
    return max(advertised or 0.0, backoff)


def request_once(request, *, timeout):
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Response(response.status, dict(response.headers), response.read())
    except urllib.error.HTTPError as error:
        with error:
            return Response(error.code, dict(error.headers or {}), error.read())
    except (OSError, http.client.HTTPException):
        # Do not print URLs, response bodies, or exception strings containing credentials.
        return Response(0, {}, b'')


def request_with_retry(
    request,
    *,
    deadline,
    attempts=5,
    timeout=20.0,
    monotonic=time.monotonic,
    sleep=time.sleep,
    transport=request_once,
    cooldown=None,
):
    cooldown = cooldown if cooldown is not None else Cooldown()
    response = Response(0, {}, b'')
    for attempt in range(1, attempts + 1):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        wait = max(0.0, cooldown.until - monotonic())
        if wait >= remaining:
            break
        if wait:
            sleep(wait)
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        response = transport(request, timeout=min(timeout, remaining))
        if not retryable(response.status):
            break
        delay = retry_delay(response.headers, attempt)
        cooldown.until = max(cooldown.until, monotonic() + delay)
        if attempt == attempts:
            break
        if delay >= deadline - monotonic():
            break
        print(
            f'RunPod HTTP {response.status}; retry {attempt}/{attempts} after {delay:.1f}s'
        )
        sleep(delay)
    return response


def pod_request(api_url, api_key, pod_id, method):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', pod_id):
        raise ValueError('invalid pod ID')
    return urllib.request.Request(
        f'{api_url.rstrip("/")}/{pod_id}',
        method=method,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
            'User-Agent': 'tenferro-ci-runpod/2',
        },
    )


def report_leak(pod_id):
    message = (
        f'Could not confirm deletion of RunPod pod {pod_id}. '
        'It may still be billing. Check and delete it in the RunPod dashboard.'
    )
    print(f'::error::{message}', file=sys.stderr)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as stream:
            stream.write(f'\n### RunPod cleanup requires attention\n\n{message}\n')


def delete_pod(
    api_url,
    api_key,
    pod_id,
    *,
    deadline_seconds=180.0,
    attempts=5,
    monotonic=time.monotonic,
    sleep=time.sleep,
    transport=request_once,
    cooldown=None,
    record_path=None,
):
    cooldown = cooldown if cooldown is not None else Cooldown()
    deadline = monotonic() + deadline_seconds
    if record_path is not None and monotonic() >= cooldown.until:
        # Best-effort reporting never blocks DELETE, but a provider cooldown
        # from that GET also applies to the following DELETE.
        observation = transport(pod_request(api_url, api_key, pod_id, 'GET'), timeout=5.0)
        if observation.status == 200:
            try:
                Path(record_path).write_bytes(observation.body)
            except OSError:
                print('::warning::Could not save pre-delete cost metadata')
        if retryable(observation.status):
            cooldown.until = max(cooldown.until, monotonic() + retry_delay(observation.headers, 1))
    response = request_with_retry(
        pod_request(api_url, api_key, pod_id, 'DELETE'),
        deadline=deadline,
        attempts=attempts,
        monotonic=monotonic,
        sleep=sleep,
        transport=transport,
        cooldown=cooldown,
    )
    print(f'RunPod delete pod={pod_id} HTTP {response.status}')
    if 200 <= response.status < 300 or response.status == 404:
        return True
    # A lost DELETE response can mean the deletion succeeded. Do not issue a
    # verification GET after 429/503 etc.: Retry-After applies to that too.
    if response.status == 0 and monotonic() < deadline:
        check = request_with_retry(
            pod_request(api_url, api_key, pod_id, 'GET'),
            deadline=deadline,
            attempts=attempts,
            monotonic=monotonic,
            sleep=sleep,
            transport=transport,
            cooldown=cooldown,
        )
        if check.status == 404:
            return True
    report_leak(pod_id)
    return False


def list_pods(
    api_url,
    api_key,
    *,
    deadline_seconds=120.0,
    max_pages=100,
    include_private=False,
    cooldown=None,
    monotonic=time.monotonic,
    sleep=time.sleep,
    transport=request_once,
):
    """Read wrapped, cursor-paginated v2 pod lists without exposing env/commands."""
    deadline = monotonic() + deadline_seconds
    cooldown = cooldown if cooldown is not None else Cooldown()
    pods = []
    cursor = None
    seen = set()
    for _ in range(max_pages):
        url = (
            api_url
            if cursor is None
            else api_url + "?" + urllib.parse.urlencode({"cursor": cursor})
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )
        response = request_with_retry(
            request,
            deadline=deadline,
            cooldown=cooldown,
            monotonic=monotonic,
            sleep=sleep,
            transport=transport,
        )
        if response.status != 200:
            raise ValueError(f"Pod listing failed (HTTP {response.status})")
        payload = json.loads(response.body)
        if not isinstance(payload, dict) or not isinstance(payload.get("pods"), list):
            raise ValueError("v2 pod list requires a pods array")
        for pod in payload["pods"]:
            if not isinstance(pod, dict) or not isinstance(pod.get("id"), str):
                raise ValueError("v2 pod list contains an invalid pod")
            pods.append(pod if include_private else {field: pod.get(field) for field in ("id", "name", "status", "cost")})
        pagination = payload.get("pagination")
        if not isinstance(pagination, dict) or not isinstance(
            pagination.get("hasNextPage"), bool
        ):
            raise ValueError("v2 pod list requires pagination metadata")
        if not pagination["hasNextPage"]:
            return pods
        cursor = pagination.get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise ValueError("v2 pod list has an invalid or repeated cursor")
        seen.add(cursor)
    raise ValueError("v2 pod list exceeded the page limit")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('delete', 'get', 'list'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--api-url')
    parser.add_argument('--delay', type=float, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--record-output', type=Path)
    parser.add_argument('--pod-name')
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text()) if args.config else {}
        api_url = args.api_url or config.get('api_url')
        key = os.environ.get('RUNPOD_API_KEY')
        pod_id = os.environ.get('POD_ID') or os.environ.get('RUNPOD_POD_ID')
        if not api_url or not key:
            raise ValueError('API URL and RUNPOD_API_KEY are required')
        if args.action == 'list':
            print(json.dumps({'pods': list_pods(api_url, key)}))
            return 0
        if not pod_id and args.pod_name:
            pods = list_pods(api_url, key)
            matches = [pod['id'] for pod in pods if pod.get('name') == args.pod_name]
            if not matches:
                print('No pod found with the requested name.')
                return 0
            if len(matches) != 1:
                raise ValueError('multiple pods match the requested name; use a Pod ID')
            pod_id = matches[0]
        if not pod_id:
            raise ValueError('POD_ID/RUNPOD_POD_ID is required')
        pod_request(api_url, key, pod_id, 'GET')  # validate before sleeping
        if args.delay:
            time.sleep(args.delay)
        if args.action == 'delete':
            return 0 if delete_pod(api_url, key, pod_id, record_path=args.record_output) else 1
        response = request_with_retry(
            pod_request(api_url, key, pod_id, 'GET'), deadline=time.monotonic() + 60
        )
        if 200 <= response.status < 300 and args.output:
            args.output.write_bytes(response.body)
        return 0 if 200 <= response.status < 300 else 1
    except (OSError, ValueError):
        print(
            'RunPod HTTP command failed; check configuration and pod ID.',
            file=sys.stderr,
        )
        return 1


if __name__ == '__main__':
    sys.exit(main())
