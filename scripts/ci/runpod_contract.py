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


def extract_gpu_type_ids(document: Mapping[str, object]) -> frozenset[str]:
    """Extract the accepted ``gpuTypeIds`` enum from ``POST /pods``."""

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
    gpu_types = _mapping(
        _property(properties, "gpuTypeIds", "POST /pods properties"),
        "POST /pods gpuTypeIds",
    )
    items = _mapping(
        _property(gpu_types, "items", "POST /pods gpuTypeIds"),
        "POST /pods gpuTypeIds items",
    )
    enum_values = _property(items, "enum", "POST /pods gpuTypeIds items")
    if (
        not isinstance(enum_values, Sequence)
        or isinstance(enum_values, (str, bytes))
        or not enum_values
        or any(not isinstance(value, str) for value in enum_values)
    ):
        raise ContractError("POST /pods gpuTypeIds enum must contain string GPU IDs")
    return frozenset(enum_values)


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
        configured_ids = config.get("gpu_type_ids")
        if (
            not isinstance(configured_ids, Sequence)
            or isinstance(configured_ids, (str, bytes))
            or not configured_ids
            or any(not isinstance(value, str) for value in configured_ids)
        ):
            raise ContractError("runpod_config.json gpu_type_ids must be nonempty strings")
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
        print(
            f"RunPod OpenAPI accepts all {len(configured_ids)} configured GPU IDs."
        )
    except ContractError as error:
        print(f"RunPod contract error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
