#!/usr/bin/env python3
"""Order reviewed RunPod GPU candidates by live Secure Cloud price.

The reviewed tier allowlist in ``runpod_config.json`` stays the eligibility
boundary: live data never adds a GPU type that maintainers did not review.
Live pricing only decides the order in which eligible types are attempted
and filters out types that are out of stock or below the VRAM requirement.

When the pricing query fails or returns nothing usable, callers fall back
to the static reviewed tier order so CI availability never depends on the
pricing endpoint.
"""

# Adapted from tensor4all/tenferro-benchmark PRs #214 and #215 (main 9012f4b).
from __future__ import annotations

import dataclasses
import json
import math
import time
import urllib.request
import urllib.parse

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from scripts.ci.runpod_contract import configured_min_cuda_version
from scripts.ci.runpod_http import Cooldown, request_once, retry_delay, retryable

CATALOG_URL = "https://api.runpod.io/v2/catalog/gpus"

Transport = Callable[[], tuple[int, bytes]]


class PricingError(RuntimeError):
    """The live pricing query failed or returned an unusable payload."""


@dataclasses.dataclass(frozen=True)
class GpuOffer:
    """One eligible GPU type with its live Secure Cloud offer."""

    gpu_type_id: str
    price_per_hr: float
    stock_status: str
    memory_gb: float


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    return None


def parse_gpu_offers(
    body: bytes,
    eligible_gpu_type_ids: Sequence[str],
    *,
    min_vram_gb: float,
) -> list[GpuOffer]:
    """Extract in-stock Secure Cloud offers for reviewed GPU types.

    Returns offers sorted by ascending hourly price; ties keep the reviewed
    allowlist order so behavior stays deterministic.
    """

    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError) as error:
        raise PricingError(f"pricing response is not JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise PricingError("pricing response root must be an object")
    gpu_types = payload.get("gpus")
    if not isinstance(gpu_types, list):
        raise PricingError("catalog response is missing 'gpus'")

    eligible_order = {gpu_id: idx for idx, gpu_id in enumerate(eligible_gpu_type_ids)}
    offers: list[GpuOffer] = []
    for entry in gpu_types:
        if not isinstance(entry, Mapping):
            continue
        gpu_id = entry.get("id")
        if not isinstance(gpu_id, str) or gpu_id not in eligible_order:
            continue
        if not entry.get("secure"):
            print(f"Pricing: {gpu_id} has no Secure Cloud offer; skipping.")
            continue
        stock = entry.get("availability")
        if stock not in ("HIGH", "MEDIUM", "LOW"):
            print(
                f"Pricing: {gpu_id} is out of stock or availability is unknown; skipping."
            )
            continue
        prices = entry.get("price")
        prices = prices if isinstance(prices, Mapping) else {}
        price = _as_float(prices.get("secure"))
        if price is None or price <= 0:
            print(f"Pricing: {gpu_id} has no usable price; skipping.")
            continue
        memory = _as_float(entry.get("memory")) or 0.0
        if memory < min_vram_gb:
            print(
                f"Pricing: {gpu_id} has {memory:g} GB VRAM, below the "
                f"{min_vram_gb:g} GB requirement; skipping."
            )
            continue
        offers.append(
            GpuOffer(
                gpu_type_id=str(gpu_id),
                price_per_hr=price,
                stock_status=stock,
                memory_gb=memory,
            )
        )

    offers.sort(key=lambda o: (o.price_per_hr, eligible_order[o.gpu_type_id]))
    return offers


def fetch_gpu_offers(
    eligible_gpu_type_ids: Sequence[str],
    *,
    min_vram_gb: float,
    transport: Transport,
) -> list[GpuOffer]:
    """Query live pricing and return eligible offers, cheapest first."""

    try:
        status, body = transport()
    except OSError as error:
        raise PricingError(f"pricing transport failure: {error}") from error
    if status != 200:
        raise PricingError(f"pricing query returned HTTP {status}")
    return parse_gpu_offers(body, eligible_gpu_type_ids, min_vram_gb=min_vram_gb)


def http_transport(
    url: str = CATALOG_URL,
    *,
    api_key: str = "",
    config: Mapping[str, Any] | None = None,
    cooldown: Cooldown | None = None,
    monotonic=time.monotonic,
    transport=request_once,
) -> Transport:
    """One authenticated v2 catalog read; no pricing retry loop."""
    config = config or {}
    cooldown = cooldown if cooldown is not None else Cooldown()
    params = {
        "include": "AVAILABILITY",
        "product": "POD",
        "cloud": "SECURE",
        "count": config.get("gpu_count", 1),
    }
    if config.get("min_cuda_version"):
        params["minCudaVersion"] = configured_min_cuda_version(config)
    catalog_url = url + "?" + urllib.parse.urlencode(params)

    def send() -> tuple[int, bytes]:
        if not api_key:
            raise PricingError("API key required for the v2 catalog")
        if monotonic() < cooldown.until:
            return 429, b""
        request = urllib.request.Request(
            catalog_url,
            method="GET",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "tenferro-ci-runpod-pricing/2",
            },
        )
        response = transport(request, timeout=20.0)
        if retryable(response.status):
            cooldown.until = monotonic() + retry_delay(response.headers, 1, base=10.0)
        return response.status, response.body

    return send


def candidate_plan(
    config: Mapping[str, Any],
    tiers: Sequence[tuple[str, Sequence[str]]],
    *,
    transport: Transport | None = None,
    api_key: str = "",
) -> list[tuple[str, list[str]]]:
    """Build the ordered create-attempt plan.

    Live-priced candidates come first, one GPU type per attempt, cheapest
    first and capped by ``max_price_candidates``. Reserve room for one
    candidate from each static tier within the provision attempt budget.
    Interleave the remaining static GPUs by tier and never retry the same
    GPU type as both a priced and static candidate.
    """

    eligible: list[str] = []
    for _tier_name, gpu_type_ids in tiers:
        for gpu_id in gpu_type_ids:
            if gpu_id not in eligible:
                eligible.append(gpu_id)

    budget = int(config.get("max_provision_attempts", 4))
    active_tiers = [(name, ids) for name, ids in tiers if ids]
    if budget < len(active_tiers):
        raise ValueError("max_provision_attempts must allow at least one candidate per GPU tier")
    plan: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    try:
        offers = fetch_gpu_offers(
            eligible,
            min_vram_gb=float(config.get("min_vram_gb", 0)),
            transport=transport
            or http_transport(
                str(config.get("catalog_url", CATALOG_URL)),
                api_key=api_key,
                config=config,
            ),
        )
    except PricingError as error:
        print(f"Live pricing unavailable; using static tier order: {error}")
        offers = []

    limit = min(int(config.get("max_price_candidates", 4)), budget - len(active_tiers))
    for offer in offers[:limit]:
        print(
            f"Price-ordered candidate: {offer.gpu_type_id} at "
            f"${offer.price_per_hr:.2f}/hr (stock {offer.stock_status}, "
            f"{offer.memory_gb:g} GB VRAM)"
        )
        if offer.gpu_type_id not in seen:
            plan.append((
                f"price-{offer.price_per_hr:.2f}-{offer.gpu_type_id}",
                [offer.gpu_type_id],
            ))
            seen.add(offer.gpu_type_id)
    if not offers:
        print("No live-priced candidates; static reviewed tiers only.")

    for index in range(max((len(ids) for _, ids in active_tiers), default=0)):
        for tier_name, ids in active_tiers:
            if index < len(ids) and ids[index] not in seen:
                plan.append((tier_name, [ids[index]]))
                seen.add(ids[index])
    return plan
