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

from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

GRAPHQL_URL = "https://api.runpod.io/graphql"

# The public pricing query needs no authentication; never send the RunPod
# API key here.
GPU_TYPES_QUERY = """
query CandidatePricing {
  gpuTypes {
    id
    memoryInGb
    secureCloud
    securePrice
    lowestPrice(input: {gpuCount: 1, secureCloud: true}) {
      stockStatus
      uninterruptablePrice
    }
  }
}
""".strip()

Transport = Callable[[bytes], tuple[int, bytes]]


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
        return float(value)
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
    except json.JSONDecodeError as error:
        raise PricingError(f"pricing response is not JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise PricingError("pricing response root must be an object")
    if payload.get("errors"):
        raise PricingError(f"pricing query returned errors: {payload['errors']!r}")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise PricingError("pricing response is missing 'data'")
    gpu_types = data.get("gpuTypes")
    if not isinstance(gpu_types, Sequence):
        raise PricingError("pricing response is missing 'gpuTypes'")

    eligible_order = {gpu_id: idx for idx, gpu_id in enumerate(eligible_gpu_type_ids)}
    offers: list[GpuOffer] = []
    for entry in gpu_types:
        if not isinstance(entry, Mapping):
            continue
        gpu_id = entry.get("id")
        if gpu_id not in eligible_order:
            continue
        if not entry.get("secureCloud"):
            print(f"Pricing: {gpu_id} has no Secure Cloud offer; skipping.")
            continue
        lowest = entry.get("lowestPrice")
        lowest = lowest if isinstance(lowest, Mapping) else {}
        stock = lowest.get("stockStatus")
        if not isinstance(stock, str) or not stock:
            print(f"Pricing: {gpu_id} is out of stock; skipping.")
            continue
        price = _as_float(lowest.get("uninterruptablePrice"))
        if price is None:
            price = _as_float(entry.get("securePrice"))
        if price is None or price <= 0:
            print(f"Pricing: {gpu_id} has no usable price; skipping.")
            continue
        memory = _as_float(entry.get("memoryInGb")) or 0.0
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

    request_body = json.dumps({"query": GPU_TYPES_QUERY}).encode()
    try:
        status, body = transport(request_body)
    except OSError as error:
        raise PricingError(f"pricing transport failure: {error}") from error
    if status != 200:
        raise PricingError(f"pricing query returned HTTP {status}")
    return parse_gpu_offers(
        body, eligible_gpu_type_ids, min_vram_gb=min_vram_gb
    )


def http_transport(url: str = GRAPHQL_URL) -> Transport:
    """Unauthenticated transport for the public pricing query."""

    def send(payload: bytes) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "tenferro-ci-runpod-pricing/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20.0) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    return send


def candidate_plan(
    config: Mapping[str, Any],
    tiers: Sequence[tuple[str, Sequence[str]]],
    *,
    transport: Transport | None = None,
) -> list[tuple[str, list[str]]]:
    """Build the ordered create-attempt plan.

    Live-priced candidates come first, one GPU type per attempt, cheapest
    first and capped by ``max_price_candidates``. The static reviewed tiers
    always follow as the documented fallback so a stale or failed pricing
    answer can never make CI lose GPU coverage entirely.
    """

    eligible: list[str] = []
    for _tier_name, gpu_type_ids in tiers:
        for gpu_id in gpu_type_ids:
            if gpu_id not in eligible:
                eligible.append(gpu_id)

    plan: list[tuple[str, list[str]]] = []
    try:
        offers = fetch_gpu_offers(
            eligible,
            min_vram_gb=float(config.get("min_vram_gb", 0)),
            transport=transport or http_transport(str(config.get("graphql_url", GRAPHQL_URL))),
        )
    except PricingError as error:
        print(f"Live pricing unavailable; using static tier order: {error}")
        offers = []

    limit = int(config.get("max_price_candidates", 4))
    for offer in offers[:limit]:
        print(
            f"Price-ordered candidate: {offer.gpu_type_id} at "
            f"${offer.price_per_hr:.2f}/hr (stock {offer.stock_status}, "
            f"{offer.memory_gb:g} GB VRAM)"
        )
        plan.append(
            (
                f"price-{offer.price_per_hr:.2f}-{offer.gpu_type_id}",
                [offer.gpu_type_id],
            )
        )
    if not offers:
        print("No live-priced candidates; static reviewed tiers only.")

    for tier_name, gpu_type_ids in tiers:
        plan.append((tier_name, list(gpu_type_ids)))
    return plan
