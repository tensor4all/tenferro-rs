#!/usr/bin/env python3
"""Validate repository RunPod configuration against the live OpenAPI schema."""

# Adapted from tensor4all/tenferro-benchmark PRs #214 and #215 (main 9012f4b).
# CUDA minimum-version handling: tensor4all/tenferro-benchmark PR #260.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.ci.runpod_http import request_with_retry


DEFAULT_CONFIG = Path(__file__).with_name("runpod_config.json")


class ContractError(RuntimeError):
    """The RunPod schema or repository configuration violates the contract."""


def configured_gpu_tiers(
    config: Mapping[str, object],
) -> list[tuple[str, tuple[str, ...]]]:
    """Validate and return ordered, disjoint GPU tiers."""

    value = config.get("gpu_tiers")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ContractError("runpod_config.json gpu_tiers must be a nonempty array")
    tiers: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ContractError("each GPU tier must be an object")
        name = item.get("name")
        ids = item.get("gpu_type_ids")
        if not isinstance(name, str) or not name:
            raise ContractError("each GPU tier requires a nonempty name")
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
            raise ContractError(f"GPU tier {name} requires nonempty gpu_type_ids")
        if any(not isinstance(gpu_id, str) or not gpu_id for gpu_id in ids):
            raise ContractError(f"GPU tier {name} IDs must be nonempty strings")
        duplicate = seen.intersection(ids)
        if duplicate:
            raise ContractError(f"duplicate GPU ID across tiers: {sorted(duplicate)}")
        seen.update(ids)
        tiers.append((name, tuple(ids)))
    return tiers


def configured_min_cuda_version(config: Mapping[str, object]) -> str:
    """Return the open-ended driver floor shared by catalog and pod creation."""
    value = config.get("min_cuda_version")
    if "allowed_cuda_versions" in config:
        raise ContractError("use min_cuda_version instead of an exact CUDA allowlist")
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+", value):
        raise ContractError("runpod_config.json min_cuda_version must be major.minor")
    return value


def resolve_local_ref(document: Mapping[str, object], ref: str) -> Mapping[str, object]:
    """Resolve one RFC 6901 local JSON pointer to an object."""

    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise ContractError(f"unsupported non-local OpenAPI reference: {ref}")
    value: object = document
    for segment in ref[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or segment not in value:
            raise ContractError(f"unresolved OpenAPI reference: {ref}")
        value = value[segment]
    if not isinstance(value, Mapping):
        raise ContractError(f"OpenAPI reference is not an object schema: {ref}")
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"OpenAPI {context} must be an object")
    return value


def schema_properties(document, schema):
    """Collect object properties through local refs and v2 allOf composition."""
    schema = _mapping(schema, "object schema")
    properties = {}
    if ref := schema.get("$ref"):
        properties.update(schema_properties(document, resolve_local_ref(document, ref)))
    for part in schema.get("allOf", []):
        properties.update(schema_properties(document, part))
    properties.update(schema.get("properties", {}))
    return properties


def _pod_create_properties(document):
    try:
        schema = document["paths"]["/v2/pods"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        return schema_properties(document, schema)
    except (KeyError, TypeError) as error:
        raise ContractError("invalid POST /v2/pods OpenAPI schema") from error


def _gpu_properties(document):
    properties = _pod_create_properties(document)
    for field in ("image", "entrypoint", "cmd", "disk", "mounts", "gpu", "env"):
        if field not in properties:
            raise ContractError(f"POST /v2/pods schema is missing {field}")
    return schema_properties(document, properties["gpu"])


def extract_gpu_type_ids(catalog):
    """v2 GPU IDs are catalog data, not an enum in the create schema."""
    gpus = catalog.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ContractError("v2 catalog must contain a nonempty gpus array")
    ids = set()
    for gpu in gpus:
        if (
            not isinstance(gpu, Mapping)
            or not isinstance(gpu.get("id"), str)
            or not gpu["id"]
        ):
            raise ContractError("v2 catalog GPU IDs must be nonempty strings")
        ids.add(gpu["id"])
    return frozenset(ids)


def validate_gpu_type_ids(document, configured_ids, catalog):
    properties = _gpu_properties(document)
    if properties.get("id", {}).get("type") != "string":
        raise ContractError("POST /v2/pods gpu.id must be a string")
    invalid = sorted(set(configured_ids) - extract_gpu_type_ids(catalog))
    if invalid:
        raise ContractError(
            "configured RunPod GPU IDs absent from v2 catalog: " + ", ".join(invalid)
        )


def validate_min_cuda_version(document, version):
    try:
        field = _gpu_properties(document)["minCudaVersion"]
        pattern = field["pattern"]
        if field["type"] != "string":
            raise KeyError("type")
    except (KeyError, TypeError) as error:
        raise ContractError(
            "POST /v2/pods gpu.minCudaVersion must declare a string pattern"
        ) from error
    if not re.fullmatch(pattern, version):
        raise ContractError("CUDA version does not match v2 format: " + version)
    # v2 permits any major.minor string. This validates syntax, not live stock;
    # availability is checked by the catalog query and the CUDA startup proof.


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"failed to read JSON from {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ContractError(f"JSON root in {path} must be an object")
    return value


def fetch_openapi(url: str, api_key: str) -> Mapping[str, object]:
    """Fetch OpenAPI or catalog JSON with bounded retries and no secret output."""
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "tenferro-ci-contract/2",
        },
    )
    response = request_with_retry(request, deadline=time.monotonic() + 60)
    if response.status != 200:
        raise ContractError(f"RunPod contract request failed (HTTP {response.status})")
    try:
        value = json.loads(response.body)
    except (ValueError, UnicodeError) as error:
        raise ContractError("RunPod contract response is not JSON") from error
    if not isinstance(value, Mapping):
        raise ContractError("RunPod contract response root must be an object")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema-file", type=Path)
    parser.add_argument("--catalog-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = _load_json(args.config)
        tiers = configured_gpu_tiers(config)
        min_cuda_version = configured_min_cuda_version(config)
        configured_ids = [gpu_id for _name, gpu_ids in tiers for gpu_id in gpu_ids]
        if args.schema_file:
            schema = _load_json(args.schema_file)
            if not args.catalog_file:
                raise ContractError(
                    "--schema-file requires --catalog-file for offline validation"
                )
            catalog = _load_json(args.catalog_file)
        else:
            api_key = os.environ.get("RUNPOD_API_KEY")
            if not api_key:
                raise ContractError(
                    "RUNPOD_API_KEY is required for live OpenAPI validation"
                )
            openapi_url = config.get("openapi_url")
            if not isinstance(openapi_url, str) or not openapi_url:
                raise ContractError("runpod_config.json openapi_url must be a string")
            schema = fetch_openapi(openapi_url, api_key)
            catalog = fetch_openapi(str(config["catalog_url"]), api_key)
        validate_gpu_type_ids(schema, configured_ids, catalog)
        validate_min_cuda_version(schema, min_cuda_version)
        print(
            f"RunPod v2 catalog recognizes all {len(configured_ids)} configured GPU "
            f"IDs; CUDA version syntax is valid: >= {min_cuda_version}. "
            "Live capacity is checked separately."
        )
    except (ContractError, KeyError) as error:
        print(f"RunPod contract error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
