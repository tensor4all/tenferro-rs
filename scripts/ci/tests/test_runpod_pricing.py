import json
import unittest

from scripts.ci.runpod_pricing import (
    GpuOffer,
    PricingError,
    candidate_plan,
    fetch_gpu_offers,
    parse_gpu_offers,
)

ELIGIBLE = (
    "NVIDIA RTX A4000",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L40S",
)


def gpu_entry(
    gpu_id: str,
    *,
    price: float | None,
    stock: str | None = "High",
    memory: float = 24,
    secure: bool = True,
) -> dict:
    return {
        "id": gpu_id,
        "memoryInGb": memory,
        "secureCloud": secure,
        "securePrice": price,
        "lowestPrice": {"stockStatus": stock, "uninterruptablePrice": price},
    }


def body(*entries: dict) -> bytes:
    return json.dumps({"data": {"gpuTypes": list(entries)}}).encode()


class ParseGpuOffersTests(unittest.TestCase):
    def test_orders_eligible_offers_by_ascending_price(self) -> None:
        offers = parse_gpu_offers(
            body(
                gpu_entry("NVIDIA GeForce RTX 4090", price=0.69),
                gpu_entry("NVIDIA A40", price=0.44),
                gpu_entry("NVIDIA L40S", price=0.79),
                gpu_entry("NVIDIA H100 PCIe", price=1.99),
            ),
            ELIGIBLE,
            min_vram_gb=0,
        )
        self.assertEqual(
            [o.gpu_type_id for o in offers],
            ["NVIDIA A40", "NVIDIA GeForce RTX 4090", "NVIDIA L40S"],
        )

    def test_price_tie_keeps_reviewed_allowlist_order(self) -> None:
        offers = parse_gpu_offers(
            body(
                gpu_entry("NVIDIA GeForce RTX 4090", price=0.5),
                gpu_entry("NVIDIA RTX A4000", price=0.5),
            ),
            ELIGIBLE,
            min_vram_gb=0,
        )
        self.assertEqual(
            [o.gpu_type_id for o in offers],
            ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 4090"],
        )

    def test_filters_stock_vram_secure_and_priceless_offers(self) -> None:
        offers = parse_gpu_offers(
            body(
                gpu_entry("NVIDIA RTX A4000", price=0.3, stock=None),
                gpu_entry("NVIDIA A40", price=0.44, memory=8),
                gpu_entry("NVIDIA GeForce RTX 4090", price=None),
                gpu_entry("NVIDIA L40S", price=0.79, secure=False),
            ),
            ELIGIBLE,
            min_vram_gb=16,
        )
        self.assertEqual(offers, [])

    def test_unusable_payloads_raise(self) -> None:
        for payload in (b"not json", b"{}", b'{"data": {}}'):
            with self.subTest(payload=payload):
                with self.assertRaises(PricingError):
                    parse_gpu_offers(payload, ELIGIBLE, min_vram_gb=0)
        with self.assertRaises(PricingError):
            parse_gpu_offers(
                b'{"errors": [{"message": "nope"}]}', ELIGIBLE, min_vram_gb=0
            )

    def test_fetch_raises_on_http_error(self) -> None:
        with self.assertRaises(PricingError):
            fetch_gpu_offers(
                ELIGIBLE,
                min_vram_gb=0,
                transport=lambda payload: (500, b"{}"),
            )

    def test_missing_uninterruptable_price_falls_back_to_secure_price(self) -> None:
        entry = gpu_entry("NVIDIA A40", price=0.44)
        entry["lowestPrice"]["uninterruptablePrice"] = None
        offers = parse_gpu_offers(body(entry), ELIGIBLE, min_vram_gb=0)
        self.assertEqual(offers, [GpuOffer("NVIDIA A40", 0.44, "High", 24)])


class CandidatePlanTests(unittest.TestCase):
    TIERS = [
        ("cost-preferred", ["NVIDIA RTX A4000", "NVIDIA A40"]),
        ("premium", ["NVIDIA L40S"]),
    ]

    def test_priced_candidates_precede_static_tiers(self) -> None:
        plan = candidate_plan(
            {"min_vram_gb": 0, "max_price_candidates": 4},
            self.TIERS,
            transport=lambda payload: (
                200,
                body(
                    gpu_entry("NVIDIA A40", price=0.44),
                    gpu_entry("NVIDIA L40S", price=0.79),
                ),
            ),
        )
        self.assertEqual(
            plan,
            [
                ("price-0.44-NVIDIA A40", ["NVIDIA A40"]),
                ("price-0.79-NVIDIA L40S", ["NVIDIA L40S"]),
                ("cost-preferred", ["NVIDIA RTX A4000", "NVIDIA A40"]),
                ("premium", ["NVIDIA L40S"]),
            ],
        )

    def test_candidate_cap_bounds_priced_attempts(self) -> None:
        plan = candidate_plan(
            {"min_vram_gb": 0, "max_price_candidates": 1},
            self.TIERS,
            transport=lambda payload: (
                200,
                body(
                    gpu_entry("NVIDIA A40", price=0.44),
                    gpu_entry("NVIDIA L40S", price=0.79),
                ),
            ),
        )
        self.assertEqual(plan[0][0], "price-0.44-NVIDIA A40")
        self.assertEqual([name for name, _ in plan[1:]], ["cost-preferred", "premium"])

    def test_pricing_failure_falls_back_to_static_tiers(self) -> None:
        def failing_transport(payload: bytes) -> tuple[int, bytes]:
            raise OSError("network down")

        plan = candidate_plan(
            {"min_vram_gb": 0},
            self.TIERS,
            transport=failing_transport,
        )
        self.assertEqual(
            [name for name, _ in plan], ["cost-preferred", "premium"]
        )

    def test_pricing_never_adds_unreviewed_gpu_types(self) -> None:
        plan = candidate_plan(
            {"min_vram_gb": 0},
            self.TIERS,
            transport=lambda payload: (
                200,
                body(gpu_entry("NVIDIA H100 PCIe", price=0.01)),
            ),
        )
        for _name, gpu_ids in plan:
            for gpu_id in gpu_ids:
                self.assertIn(
                    gpu_id,
                    {"NVIDIA RTX A4000", "NVIDIA A40", "NVIDIA L40S"},
                )


if __name__ == "__main__":
    unittest.main()
