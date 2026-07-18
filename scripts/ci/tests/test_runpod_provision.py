import json
import unittest
from unittest import mock

import scripts.ci.runpod_provision as provision_module
from scripts.ci.runpod_client import (
    AssignedGpuError,
    CreateResult,
    RetryableRunPodError,
)
from scripts.ci.runpod_provision import (
    PodLeakError,
    PodState,
    _pod_state_checker,
    ProvisionExhaustedError,
    _pod_api,
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


def live(desired: str = "RUNNING") -> PodState:
    return PodState(desired=desired, has_runtime=True)


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

        def runner_online(label: str) -> bool:
            online_after["count"] -= 1
            return online_after["count"] <= 0

        deleted: list[str] = []
        result = provision(
            CONFIG,
            PLAN,
            label_prefix="runpod-1-1",
            mint_runner=lambda label: f"jit-{label}",
            create=lambda req, jit: created("pod-1", "NVIDIA A40", req.tier_name),
            runner_online=runner_online,
            pod_status=lambda pod_id: live(),
            delete_pod=lambda pod_id: (deleted.append(pod_id), True)[1],
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

        def create(req, jit):
            pod = next(pods)
            current["id"] = pod.pod_id
            return pod

        result = provision(
            CONFIG,
            PLAN,
            label_prefix="runpod-1-1",
            mint_runner=lambda label: f"jit-{label}",
            create=create,
            runner_online=lambda label: current["id"] in accepted,
            pod_status=lambda pod_id: live(statuses[pod_id]),
            delete_pod=lambda pod_id: (deleted.append(pod_id), True)[1],
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
                label_prefix="runpod-1-1",
                mint_runner=lambda label: f"jit-{label}",
                create=lambda req, jit: created("pod-slow", "NVIDIA A40", req.tier_name),
                runner_online=lambda label: False,
                pod_status=lambda pod_id: live(),
                delete_pod=lambda pod_id: (deleted.append(pod_id), True)[1],
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        self.assertEqual(deleted, ["pod-slow"])

    def test_capacity_failures_move_to_next_candidate_without_pods(self) -> None:
        clock = Clock()
        calls: list[str] = []

        def create(req, jit):
            calls.append(req.tier_name)
            if len(calls) < 3:
                raise RetryableRunPodError("RunPod capacity unavailable")
            return created("pod-3", "NVIDIA RTX A4000", req.tier_name)

        result = provision(
            CONFIG,
            PLAN,
            label_prefix="runpod-1-1",
            mint_runner=lambda label: f"jit-{label}",
            create=create,
            runner_online=lambda label: True,
            pod_status=lambda pod_id: live(),
            delete_pod=lambda pod_id: self.fail("no pod to delete"),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(result.attempts, 3)
        self.assertEqual(calls, [name for name, _ in PLAN[:3]])

    def test_each_attempt_mints_a_fresh_label_and_jit_config(self) -> None:
        clock = Clock()
        pods = iter(
            [
                created("pod-bad", "NVIDIA A40", "a"),
                created("pod-good", "NVIDIA GeForce RTX 4090", "b"),
            ]
        )
        statuses = {"pod-bad": "EXITED", "pod-good": "RUNNING"}
        minted: list[str] = []
        jits: list[str] = []
        waited: list[str] = []

        def mint_runner(label: str) -> str:
            minted.append(label)
            return f"jit-{label}"

        def runner_online(label: str) -> bool:
            waited.append(label)
            return label == "runpod-1-1-c2"

        result = provision(
            CONFIG,
            PLAN,
            label_prefix="runpod-1-1",
            mint_runner=mint_runner,
            create=lambda req, jit: (jits.append(jit), next(pods))[1],
            runner_online=runner_online,
            pod_status=lambda pod_id: live(statuses[pod_id]),
            delete_pod=lambda pod_id: True,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(result.runner_label, "runpod-1-1-c2")
        self.assertEqual(minted, ["runpod-1-1-c1", "runpod-1-1-c2"])
        self.assertEqual(jits, ["jit-runpod-1-1-c1", "jit-runpod-1-1-c2"])
        # The wait for attempt 2 must never look at attempt 1's label.
        self.assertNotIn("runpod-1-1-c1", waited)

    def test_unverifiable_gpu_pod_is_published_deleted_and_skipped(self) -> None:
        clock = Clock()
        deleted: list[str] = []
        published: list[str] = []
        calls = {"n": 0}

        def create(req, jit):
            calls["n"] += 1
            if calls["n"] == 1:
                raise AssignedGpuError(
                    "RunPod assigned GPU outside selected tier",
                    created("pod-rogue", "NVIDIA H100 PCIe", req.tier_name),
                )
            return created("pod-ok", "NVIDIA GeForce RTX 4090", req.tier_name)

        result = provision(
            CONFIG,
            PLAN,
            label_prefix="runpod-1-1",
            mint_runner=lambda label: f"jit-{label}",
            create=create,
            runner_online=lambda label: True,
            pod_status=lambda pod_id: live(),
            delete_pod=lambda pod_id: (deleted.append(pod_id), True)[1],
            publish_pod_id=published.append,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(result.pod_id, "pod-ok")
        self.assertEqual(deleted, ["pod-rogue"])
        self.assertEqual(published, ["pod-rogue", "pod-ok"])

    def test_unconfirmed_deletion_stops_creating_more_pods(self) -> None:
        clock = Clock()
        calls = {"n": 0}

        def create(req, jit):
            calls["n"] += 1
            return created("pod-stuck", "NVIDIA A40", req.tier_name)

        with self.assertRaises(PodLeakError):
            provision(
                CONFIG,
                PLAN,
                label_prefix="runpod-1-1",
                mint_runner=lambda label: f"jit-{label}",
                create=create,
                runner_online=lambda label: False,
                pod_status=lambda pod_id: live("EXITED"),
                delete_pod=lambda pod_id: False,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        self.assertEqual(calls["n"], 1)

    def test_stale_online_runner_does_not_accept_a_dead_pod(self) -> None:
        clock = Clock()
        pods = iter(
            [
                created("pod-dead", "NVIDIA A40", "price-0.44-NVIDIA A40"),
                created("pod-live", "NVIDIA GeForce RTX 4090", "x", cost=0.69),
            ]
        )
        statuses = {"pod-dead": "EXITED", "pod-live": "RUNNING"}
        deleted: list[str] = []

        result = provision(
            CONFIG,
            PLAN,
            label_prefix="runpod-1-1",
            mint_runner=lambda label: f"jit-{label}",
            create=lambda req, jit: next(pods),
            # The runner registry reports online the whole time (stale or
            # died-after-registering record); the dead pod must still be
            # rejected.
            runner_online=lambda label: True,
            pod_status=lambda pod_id: live(statuses[pod_id]),
            delete_pod=lambda pod_id: (deleted.append(pod_id), True)[1],
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(result.pod_id, "pod-live")
        self.assertEqual(deleted, ["pod-dead"])

    def test_container_stop_is_detected_without_desired_status_change(self) -> None:
        """desiredStatus stays RUNNING after container exit; runtime is the signal."""

        clock = Clock()
        states = iter(
            [
                PodState(desired="RUNNING", has_runtime=True),
                PodState(desired="RUNNING", has_runtime=False),
            ]
        )
        deleted: list[str] = []
        with self.assertRaises(ProvisionExhaustedError) as caught:
            provision(
                {**CONFIG, "max_provision_attempts": 1},
                PLAN,
                label_prefix="runpod-1-1",
                mint_runner=lambda label: f"jit-{label}",
                create=lambda req, jit: created("pod-dead", "NVIDIA A40", req.tier_name),
                runner_online=lambda label: False,
                pod_status=lambda pod_id: next(states),
                delete_pod=lambda pod_id: (deleted.append(pod_id), True)[1],
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        self.assertEqual(deleted, ["pod-dead"])
        self.assertIn("container stopped", str(caught.exception))
        # Detected on the second poll, long before the startup timeout.
        self.assertLess(clock.now, CONFIG["startup_timeout_seconds"])

    def test_boot_phase_null_runtime_is_not_treated_as_death(self) -> None:
        clock = Clock()
        boot = {"polls": 0}

        def pod_status(pod_id: str) -> PodState:
            boot["polls"] += 1
            # Image pull phase: runtime not yet present.
            return PodState(desired="RUNNING", has_runtime=boot["polls"] > 3)

        result = provision(
            CONFIG,
            PLAN,
            label_prefix="runpod-1-1",
            mint_runner=lambda label: f"jit-{label}",
            create=lambda req, jit: created("pod-boot", "NVIDIA A40", req.tier_name),
            runner_online=lambda label: boot["polls"] > 5,
            pod_status=pod_status,
            delete_pod=lambda pod_id: self.fail("healthy pod must not be deleted"),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertEqual(result.pod_id, "pod-boot")

    def test_keep_failed_pods_skips_deletion_and_reports_ids(self) -> None:
        clock = Clock()
        with self.assertRaises(ProvisionExhaustedError) as caught:
            provision(
                {**CONFIG, "max_provision_attempts": 1},
                PLAN,
                label_prefix="runpod-1-1",
                mint_runner=lambda label: f"jit-{label}",
                create=lambda req, jit: created("pod-debug", "NVIDIA A40", req.tier_name),
                runner_online=lambda label: False,
                pod_status=lambda pod_id: live("EXITED"),
                delete_pod=lambda pod_id: self.fail("debug mode must not delete"),
                keep_failed_pods=True,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        self.assertIn("kept debug pods: pod-debug", str(caught.exception))

    def test_exhaustion_is_explicit_and_bounded(self) -> None:
        clock = Clock()
        calls: list[str] = []

        def create(req, jit):
            calls.append(req.tier_name)
            raise RetryableRunPodError("RunPod capacity unavailable")

        with self.assertRaises(ProvisionExhaustedError) as caught:
            provision(
                CONFIG,
                PLAN,
                label_prefix="runpod-1-1",
                mint_runner=lambda label: f"jit-{label}",
                create=create,
                runner_online=lambda label: True,
                pod_status=lambda pod_id: live(),
                delete_pod=lambda pod_id: True,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        self.assertEqual(len(calls), CONFIG["max_provision_attempts"])
        self.assertIn("capacity unavailable", str(caught.exception))


class PodApiTransportTests(unittest.TestCase):
    def test_transport_errors_are_transient_not_fatal(self) -> None:
        """URLError/timeouts must not escape and abort the provision loop."""

        import urllib.error

        def failing_urlopen(*args, **kwargs):
            raise urllib.error.URLError("dns hiccup")

        delete_pod = _pod_api("https://rest.example/v1/pods", "key")
        pod_status = _pod_state_checker("https://gql.example/graphql", "key")
        with mock.patch.object(
            provision_module.urllib.request, "urlopen", failing_urlopen
        ), mock.patch.object(provision_module.time, "sleep"):
            # status: transient -> unknown state (keep waiting), no exception
            state = pod_status("pod-1")
            self.assertIsNone(state.desired)
            self.assertIsNone(state.has_runtime)
            # delete: retries then reports unconfirmed -> False, no exception
            self.assertFalse(delete_pod("pod-1"))

    def test_pod_state_reads_desired_status_and_runtime(self) -> None:
        import io

        payloads = iter(
            [
                b'{"data": {"pod": {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 12}}}}',
                b'{"data": {"pod": {"desiredStatus": "RUNNING", "runtime": null}}}',
            ]
        )

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        pod_status = _pod_state_checker("https://gql.example/graphql", "key")
        with mock.patch.object(
            provision_module.urllib.request,
            "urlopen",
            lambda *a, **k: FakeResponse(next(payloads)),
        ):
            first = pod_status("pod-1")
            self.assertEqual(first, PodState(desired="RUNNING", has_runtime=True))
            second = pod_status("pod-1")
            self.assertEqual(second, PodState(desired="RUNNING", has_runtime=False))


if __name__ == "__main__":
    unittest.main()
