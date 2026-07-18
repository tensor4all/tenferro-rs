#!/usr/bin/env python3
"""Validate repository RunPod configuration against the live OpenAPI schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("runpod_config.json")


class ContractError(RuntimeError):
    """The RunPod schema or repository configuration violates the contract."""


def configured_gpu_tiers(
    config: Mapping[str, object],
) -> list[tuple[str, tuple[str, ...]]]:
    """Validate and return ordered, disjoint GPU tiers."""

    value = config.get("gpu_tiers")
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ContractError(
            "runpod_config.json gpu_tiers must be a nonempty array"
        )
    tiers: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ContractError("each GPU tier must be an object")
        name = item.get("name")
        ids = item.get("gpu_type_ids")
        if not isinstance(name, str) or not name:
            raise ContractError("each GPU tier requires a nonempty name")
        if (
            not isinstance(ids, Sequence)
            or isinstance(ids, (str, bytes))
            or not ids
        ):
            raise ContractError(
                f"GPU tier {name} requires nonempty gpu_type_ids"
            )
        if any(not isinstance(gpu_id, str) or not gpu_id for gpu_id in ids):
            raise ContractError(
                f"GPU tier {name} IDs must be nonempty strings"
            )
        duplicate = seen.intersection(ids)
        if duplicate:
            raise ContractError(
                f"duplicate GPU ID across tiers: {sorted(duplicate)}"
            )
        seen.update(ids)
        tiers.append((name, tuple(ids)))
    return tiers


def configured_cuda_versions(config: Mapping[str, object]) -> tuple[str, ...]:
    """Validate and return the CUDA driver versions accepted for a pod."""

    value = config.get("allowed_cuda_versions")
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(version, str) or not version for version in value)
    ):
        raise ContractError(
            "runpod_config.json allowed_cuda_versions must be a nonempty "
            "array of strings"
        )
    if len(set(value)) != len(value):
        raise ContractError("allowed_cuda_versions must not contain duplicates")
    return tuple(value)


def resolve_local_ref(
    document: Mapping[str, object], ref: str
) -> Mapping[str, object]:
    """Resolve one RFC 6901 local JSON pointer to an object."""

    if not ref.startswith("#/"):
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


def _property(value: Mapping[str, object], name: str, context: str) -> object:
    if name not in value:
        raise ContractError(f"OpenAPI {context} is missing {name!r}")
    return value[name]


def _pod_create_properties(
    document: Mapping[str, object],
) -> Mapping[str, object]:
    try:
        paths = _mapping(_property(document, "paths", "document"), "paths")
        pods = _mapping(_property(paths, "/pods", "paths"), "POST /pods")
        post = _mapping(_property(pods, "post", "POST /pods"), "POST /pods")
        request_body = _mapping(
            _property(post, "requestBody", "POST /pods"),
            "POST /pods requestBody",
        )
        content = _mapping(
            _property(request_body, "content", "POST /pods requestBody"),
            "POST /pods requestBody content",
        )
        json_content = _mapping(
            _property(content, "application/json", "POST /pods content"),
            "POST /pods application/json",
        )
        schema = _mapping(
            _property(json_content, "schema", "POST /pods application/json"),
            "POST /pods request schema",
        )
    except ContractError as error:
        if "POST /pods" in str(error):
            raise
        raise ContractError(f"invalid POST /pods OpenAPI schema: {error}") from error

    if ref := schema.get("$ref"):
        if not isinstance(ref, str):
            raise ContractError("POST /pods schema $ref must be a string")
        schema = resolve_local_ref(document, ref)

    properties = _mapping(
        _property(schema, "properties", "POST /pods schema"),
        "POST /pods properties",
    )
    return properties


def _extract_string_enum_property(
    document: Mapping[str, object], property_name: str
) -> frozenset[str]:
    properties = _pod_create_properties(document)
    property_schema = _mapping(
        _property(properties, property_name, "POST /pods properties"),
        f"POST /pods {property_name}",
    )
    items = _mapping(
        _property(property_schema, "items", f"POST /pods {property_name}"),
        f"POST /pods {property_name} items",
    )
    enum_values = _property(
        items, "enum", f"POST /pods {property_name} items"
    )
    if (
        not isinstance(enum_values, Sequence)
        or isinstance(enum_values, (str, bytes))
        or not enum_values
        or any(not isinstance(value, str) for value in enum_values)
    ):
        raise ContractError(
            f"POST /pods {property_name} enum must contain strings"
        )
    return frozenset(enum_values)


def extract_gpu_type_ids(document: Mapping[str, object]) -> frozenset[str]:
    """Extract the accepted ``gpuTypeIds`` enum from ``POST /pods``."""

    return _extract_string_enum_property(document, "gpuTypeIds")


def extract_allowed_cuda_versions(
    document: Mapping[str, object],
) -> frozenset[str]:
    """Extract accepted ``allowedCudaVersions`` values from ``POST /pods``."""

    return _extract_string_enum_property(document, "allowedCudaVersions")


def validate_gpu_type_ids(
    document: Mapping[str, object], configured_ids: Sequence[str]
) -> None:
    """Require every configured GPU ID to occur in the current schema."""

    accepted = extract_gpu_type_ids(document)
    invalid = sorted(set(configured_ids) - accepted)
    if invalid:
        raise ContractError(
            "configured RunPod GPU IDs absent from POST /pods schema: "
            + ", ".join(invalid)
        )


def validate_cuda_versions(
    document: Mapping[str, object], configured_versions: Sequence[str]
) -> None:
    """Require every configured CUDA version to occur in the current schema."""

    accepted = extract_allowed_cuda_versions(document)
    invalid = sorted(set(configured_versions) - accepted)
    if invalid:
        raise ContractError(
            "configured RunPod CUDA versions absent from POST /pods schema: "
            + ", ".join(invalid)
        )


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"failed to read JSON from {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ContractError(f"JSON root in {path} must be an object")
    return value


def fetch_openapi(url: str, api_key: str) -> Mapping[str, object]:
    """Fetch and decode the authenticated OpenAPI document without logging secrets."""

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "tenferro-ci-contract/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value: Any = json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise ContractError(f"failed to fetch RunPod OpenAPI document: {error}") from error
    if not isinstance(value, Mapping):
        raise ContractError("RunPod OpenAPI document root must be an object")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = _load_json(args.config)
        tiers = configured_gpu_tiers(config)
        cuda_versions = configured_cuda_versions(config)
        configured_ids = [
            gpu_id for _name, gpu_ids in tiers for gpu_id in gpu_ids
        ]
        if args.schema_file:
            schema = _load_json(args.schema_file)
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
        validate_gpu_type_ids(schema, configured_ids)
        validate_cuda_versions(schema, cuda_versions)
        print(
            f"RunPod OpenAPI accepts all {len(configured_ids)} configured GPU "
            f"IDs and CUDA versions {', '.join(cuda_versions)}."
        )
    except ContractError as error:
        print(f"RunPod contract error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
