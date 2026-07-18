import json
import unittest

from scripts.ci.runpod_client import CreateResult, RetryableRunPodError
from scripts.ci.runpod_provision import (
    ProvisionExhaustedError,
    parse_cost_per_hr,
    provision,
)

CONFIG = {
    "max_provision_attempts": 3,
    "startup_timeout_seconds": 100,
    "startup_poll_seconds": 1,
}

PLAN = [
    ("price-0.44-NVIDIA A40", ["NVIDIA A40"]),
    ("price-0.69-NVIDIA GeForce RTX 4090", ["NVIDIA GeForce RTX 4090"]),
    ("cost-preferred", ["NVIDIA RTX A4000", "NVIDIA A40"]),
    ("premium", ["NVIDIA L40S"]),
]


def created(pod_id: str, gpu: str, tier: str, cost: float = 0.44) -> CreateResult:
    return CreateResult(
        pod_id=pod_id,
        gpu_type_id=gpu,
        gpu_tier=tier,
        body=json.dumps({"id": pod_id, "costPerHr": cost}).encode(),
    )


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class ParseCostTests(unittest.TestCase):
    def test_prefers_adjusted_cost(self) -> None:
        body = json.dumps({"adjustedCostPerHr": 0.4, "costPerHr": 0.5}).encode()
        self.assertEqual(parse_cost_per_hr(body), 0.4)

    def test_missing_or_invalid_cost_is_none(self) -> None:
        self.assertIsNone(parse_cost_per_hr(b"{}"))
        self.assertIsNone(parse_cost_per_hr(b"not json"))
        self.assertIsNone(parse_cost_per_hr(json.dumps({"costPerHr": True}).encode()))


class ProvisionTests(unittest.TestCase):
    def test_first_candidate_accepted_when_runner_comes_online(self) -> None:
        clock = Clock()
        online_after = {"count": 3}

        def runner_online() -> bool:
            online_after["count"] -= 1
            return online_after["count"] <= 0

        deleted: list[str] = []
        result = provision(
            CONFIG,
            PLAN,
            create=lambda req: created("pod-1", "NVIDIA A40", req.tier_name),
            runner_online=runner_online,
            pod_status=lambda pod_id: "RUNNING",
            delete_pod=deleted.append,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(result.pod_id, "pod-1")
        self.assertEqual(result.gpu_tier, "price-0.44-NVIDIA A40")
        self.assertEqual(result.cost_per_hr, 0.44)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(deleted, [])

    def test_smoke_failure_deletes_pod_and_tries_next_candidate(self) -> None:
        clock = Clock()
        pods = iter(
            [
                created("pod-bad", "NVIDIA A40", "price-0.44-NVIDIA A40"),
                created("pod-good", "NVIDIA GeForce RTX 4090", "x", cost=0.69),
            ]
        )
        statuses = {"pod-bad": "EXITED", "pod-good": "RUNNING"}
        accepted = {"pod-good"}
        deleted: list[str] = []
        published: list[str] = []
        current: dict[str, str] = {}

        def create(req):
            pod = next(pods)
            current["id"] = pod.pod_id
            return pod

        result = provision(
            CONFIG,
            PLAN,
            create=create,
            runner_online=lambda: current["id"] in accepted,
            pod_status=lambda pod_id: statuses[pod_id],
            delete_pod=deleted.append,
            publish_pod_id=published.append,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(result.pod_id, "pod-good")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(deleted, ["pod-bad"])
        self.assertEqual(published, ["pod-bad", "pod-good"])

    def test_startup_timeout_deletes_pod(self) -> None:
        clock = Clock()
        deleted: list[str] = []
        with self.assertRaises(ProvisionExhaustedError):
            provision(
                {**CONFIG, "max_provision_attempts": 1},
                PLAN,
                create=lambda req: created("pod-slow", "NVIDIA A40", req.tier_name),
                runner_online=lambda: False,
                pod_status=lambda pod_id: "RUNNING",
                delete_pod=deleted.append,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        self.assertEqual(deleted, ["pod-slow"])

    def test_capacity_failures_move_to_next_candidate_without_pods(self) -> None:
        clock = Clock()
        calls: list[str] = []

        def create(req):
            calls.append(req.tier_name)
            if len(calls) < 3:
                raise RetryableRunPodError("RunPod capacity unavailable")
            return created("pod-3", "NVIDIA RTX A4000", req.tier_name)

        result = provision(
            CONFIG,
            PLAN,
            create=create,
            runner_online=lambda: True,
            pod_status=lambda pod_id: "RUNNING",
            delete_pod=lambda pod_id: self.fail("no pod to delete"),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(result.attempts, 3)
        self.assertEqual(calls, [name for name, _ in PLAN[:3]])

    def test_exhaustion_is_explicit_and_bounded(self) -> None:
        clock = Clock()
        calls: list[str] = []

        def create(req):
            calls.append(req.tier_name)
            raise RetryableRunPodError("RunPod capacity unavailable")

        with self.assertRaises(ProvisionExhaustedError) as caught:
            provision(
                CONFIG,
                PLAN,
                create=create,
                runner_online=lambda: True,
                pod_status=lambda pod_id: "RUNNING",
                delete_pod=lambda pod_id: None,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        self.assertEqual(len(calls), CONFIG["max_provision_attempts"])
        self.assertIn("capacity unavailable", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
