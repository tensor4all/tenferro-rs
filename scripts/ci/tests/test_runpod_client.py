import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.ci.runpod_client import (
    AssignedGpuError,
    CreateRequest,
    CreateResult,
    PermanentRunPodError,
    RetryClass,
    RetryableRunPodError,
    backoff_seconds,
    build_pod_payload,
    classify_http_status,
    create_pod,
    is_capacity_failure,
    parse_create_response,
    publish_cleanup_pod_id,
    publish_github_result,
    redacted_error_message,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads(
    (ROOT / "scripts" / "ci" / "runpod_config.json").read_text()
)


class RunPodClientTests(unittest.TestCase):
    def test_publish_github_result_records_tier_and_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            summary = Path(directory) / "summary"
            result = CreateResult(
                pod_id="pod-1",
                gpu_type_id="NVIDIA L40S",
                gpu_tier="premium",
                body=b"{}",
            )

            publish_github_result(
                result,
                output_path=output,
                summary_path=summary,
            )

            self.assertEqual(
                output.read_text(),
                "pod_id=pod-1\n"
                "gpu_type_id=NVIDIA L40S\n"
                "gpu_tier=premium\n",
            )
            self.assertIn(
                "Selected GPU: NVIDIA L40S", summary.read_text()
            )
            self.assertIn("Price tier: premium", summary.read_text())

    def test_publish_github_result_rejects_multiline_provider_values(
        self,
    ) -> None:
        result = CreateResult(
            pod_id="pod-1",
            gpu_type_id="NVIDIA L40S\ngpu_tier=forged",
            gpu_tier="premium",
            body=b"{}",
        )
        with self.assertRaisesRegex(
            PermanentRunPodError, "unsafe GitHub output"
        ):
            publish_github_result(
                result,
                output_path=Path("unused-output"),
                summary_path=None,
            )

    def test_create_rejects_gpu_outside_selected_tier(self) -> None:
        with self.assertRaisesRegex(
            AssignedGpuError, "outside selected tier"
        ) as caught:
            create_pod(
                CONFIG,
                [
                    CreateRequest(
                        "premium",
                        b"premium",
                        ("NVIDIA L40S",),
                    )
                ],
                transport=lambda _payload, _timeout: (
                    201,
                    {},
                    b'{"id":"pod-1","machine":'
                    b'{"gpuTypeId":"NVIDIA H100 80GB HBM3"}}',
                ),
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            publish_cleanup_pod_id(caught.exception.result, output)
            self.assertEqual(output.read_text(), "pod_id=pod-1\n")

    def test_create_requires_assigned_gpu_for_selected_tier(self) -> None:
        with self.assertRaisesRegex(
            PermanentRunPodError, "missing assigned GPU"
        ):
            create_pod(
                CONFIG,
                [
                    CreateRequest(
                        "premium",
                        b"premium",
                        ("NVIDIA L40S",),
                    )
                ],
                transport=lambda _payload, _timeout: (
                    201,
                    {},
                    b'{"id":"pod-1"}',
                ),
            )

    def test_summary_escapes_markdown_from_provider_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary"
            publish_github_result(
                CreateResult(
                    pod_id="pod-1",
                    gpu_type_id="GPU `spoof`",
                    gpu_tier="premium",
                    body=b"{}",
                ),
                output_path=None,
                summary_path=summary,
            )
            self.assertIn("GPU &#96;spoof&#96;", summary.read_text())

    def test_capacity_failure_requires_server_error_and_known_message(
        self,
    ) -> None:
        body = (
            b'{"error":"This machine does not have the resources to deploy '
            b'your pod"}'
        )
        self.assertTrue(is_capacity_failure(500, body))
        self.assertFalse(is_capacity_failure(400, body))
        self.assertFalse(
            is_capacity_failure(500, b'{"error":"internal failure"}')
        )

    def test_capacity_failure_moves_tier_without_sleep(self) -> None:
        requests = [
            CreateRequest("cost-preferred", b"cheap"),
            CreateRequest("premium", b"premium"),
        ]
        responses = iter(
            [
                (
                    500,
                    {},
                    b'{"error":"This machine does not have the resources '
                    b'to deploy your pod"}',
                ),
                (
                    201,
                    {},
                    b'{"id":"pod-1","machine":'
                    b'{"gpuTypeId":"NVIDIA L40S"}}',
                ),
            ]
        )
        seen: list[bytes] = []

        def transport(
            payload: bytes, _timeout: float
        ) -> tuple[int, dict[str, str], bytes]:
            seen.append(payload)
            return next(responses)

        result = create_pod(
            CONFIG,
            requests,
            transport=transport,
            sleep=lambda _delay: self.fail(
                "capacity failover must not sleep"
            ),
        )

        self.assertEqual(seen, [b"cheap", b"premium"])
        self.assertEqual(result.gpu_tier, "premium")
        self.assertEqual(result.gpu_type_id, "NVIDIA L40S")

    def test_retry_classification(self) -> None:
        for status in (408, 429, 500, 502, 503):
            with self.subTest(status=status):
                self.assertIs(
                    classify_http_status(status), RetryClass.RETRYABLE
                )
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertIs(
                    classify_http_status(status), RetryClass.PERMANENT
                )

    def test_backoff_is_bounded_and_jittered(self) -> None:
        delays = [
            backoff_seconds(index, base=5, cap=60, jitter=lambda: 0.5)
            for index in range(1, 8)
        ]
        self.assertEqual(delays, [2.5, 5.0, 10.0, 20.0, 30.0, 30.0, 30.0])

    def test_payload_preserves_secure_trust_boundary(self) -> None:
        gpu_type_ids = CONFIG["gpu_tiers"][0]["gpu_type_ids"]
        payload = build_pod_payload(
            CONFIG,
            "image",
            "startup",
            "jit",
            gpu_type_ids,
        )
        self.assertEqual(payload["cloudType"], "SECURE")
        self.assertIs(payload["interruptible"], False)
        self.assertEqual(payload["gpuTypeIds"], gpu_type_ids)
        self.assertEqual(payload["env"]["RUNNER_JIT_CONFIG"], "jit")
        self.assertEqual(payload["dockerStartCmd"], ["startup"])

    def test_success_without_pod_id_is_protocol_error(self) -> None:
        with self.assertRaisesRegex(PermanentRunPodError, "missing pod id"):
            parse_create_response(201, b"{}")

    def test_permanent_error_is_not_retried(self) -> None:
        calls = 0

        def transport(
            _payload: bytes, _timeout: float
        ) -> tuple[int, dict[str, str], bytes]:
            nonlocal calls
            calls += 1
            return 400, {}, b'{"error":"bad gpu id"}'

        with self.assertRaisesRegex(PermanentRunPodError, "HTTP 400"):
            create_pod(
                CONFIG,
                [CreateRequest("cost-preferred", b"{}")],
                transport=transport,
                sleep=lambda _delay: self.fail("must not sleep"),
            )
        self.assertEqual(calls, 1)

    def test_retryable_responses_sleep_then_succeed(self) -> None:
        responses = iter(
            [
                (500, {}, b'{"error":"busy"}'),
                (201, {}, b'{"id":"pod-1","machine":{"gpuTypeId":"NVIDIA A40"}}'),
            ]
        )
        sleeps: list[float] = []

        result = create_pod(
            CONFIG,
            [CreateRequest("cost-preferred", b"{}")],
            transport=lambda _payload, _timeout: next(responses),
            sleep=sleeps.append,
            jitter=lambda: 0.5,
        )

        self.assertEqual(result.pod_id, "pod-1")
        self.assertEqual(result.gpu_type_id, "NVIDIA A40")
        self.assertEqual(sleeps, [5.0])

    def test_generic_service_failure_retries_same_tier_once(self) -> None:
        responses = iter(
            [
                (503, {"Retry-After": "2"}, b'{"error":"busy"}'),
                (
                    201,
                    {},
                    b'{"id":"pod-1","machine":'
                    b'{"gpuTypeId":"NVIDIA L40S"}}',
                ),
            ]
        )
        seen: list[bytes] = []
        sleeps: list[float] = []

        def transport(
            payload: bytes, _timeout: float
        ) -> tuple[int, dict[str, str], bytes]:
            seen.append(payload)
            return next(responses)

        result = create_pod(
            CONFIG | {"same_tier_retries": 1},
            [CreateRequest("premium", b"premium")],
            transport=transport,
            sleep=sleeps.append,
        )

        self.assertEqual(seen, [b"premium", b"premium"])
        self.assertEqual(sleeps, [2.0])
        self.assertEqual(result.gpu_tier, "premium")

    def test_generic_service_failure_stops_after_one_retry(self) -> None:
        calls = 0

        def transport(
            _payload: bytes, _timeout: float
        ) -> tuple[int, dict[str, str], bytes]:
            nonlocal calls
            calls += 1
            if calls == 3:
                return 201, {}, b'{"id":"too-late"}'
            return 503, {}, b'{"error":"busy"}'

        with self.assertRaises(RetryableRunPodError):
            create_pod(
                CONFIG | {"same_tier_retries": 1},
                [CreateRequest("premium", b"premium")],
                transport=transport,
                sleep=lambda _delay: None,
                monotonic=lambda: 0.0,
            )
        self.assertEqual(calls, 2)

    def test_creation_deadline_caps_retry_sleep(self) -> None:
        ticks = iter([0.0, 0.0, 59.0, 60.0])
        sleeps: list[float] = []
        with self.assertRaises(RetryableRunPodError):
            create_pod(
                CONFIG
                | {
                    "same_tier_retries": 1,
                    "create_deadline_seconds": 60,
                },
                [CreateRequest("cost-preferred", b"cheap")],
                transport=lambda _payload, _timeout: (
                    503,
                    {"Retry-After": "30"},
                    b"{}",
                ),
                sleep=sleeps.append,
                monotonic=lambda: next(ticks),
            )
        self.assertEqual(sleeps, [1.0])

    def test_creation_deadline_caps_each_transport_timeout(self) -> None:
        ticks = iter([0.0, 0.0, 59.0, 59.0])
        timeouts: list[float] = []
        responses = iter(
            [
                (503, {}, b'{"error":"busy"}'),
                (201, {}, b'{"id":"pod-1"}'),
            ]
        )

        def transport(
            _payload: bytes, timeout: float
        ) -> tuple[int, dict[str, str], bytes]:
            timeouts.append(timeout)
            return next(responses)

        result = create_pod(
            CONFIG,
            [CreateRequest("cost-preferred", b"cheap")],
            transport=transport,
            sleep=lambda _delay: None,
            monotonic=lambda: next(ticks),
        )

        self.assertEqual(result.pod_id, "pod-1")
        self.assertEqual(timeouts, [60.0, 1.0])

    def test_transport_failure_is_retryable_but_bounded(self) -> None:
        calls = 0

        def transport(
            _payload: bytes, _timeout: float
        ) -> tuple[int, dict[str, str], bytes]:
            nonlocal calls
            calls += 1
            raise OSError("connection reset")

        config: dict[str, Any] = CONFIG | {"same_tier_retries": 1}
        with self.assertRaisesRegex(RetryableRunPodError, "transport failure"):
            create_pod(
                config,
                [CreateRequest("cost-preferred", b"{}")],
                transport=transport,
                sleep=lambda _delay: None,
                jitter=lambda: 0.5,
            )
        self.assertEqual(calls, 2)

    def test_error_redaction_removes_secret_values_and_fields(self) -> None:
        body = json.dumps(
            {
                "env": {"RUNNER_JIT_CONFIG": "jit-secret"},
                "dockerStartCmd": ["startup-secret"],
                "message": "request echoed jit-secret",
            }
        ).encode()
        message = redacted_error_message(body, secrets=("jit-secret",))
        self.assertNotIn("jit-secret", message)
        self.assertNotIn("startup-secret", message)
        self.assertIn("***redacted***", message)


if __name__ == "__main__":
    unittest.main()
