import json
import unittest
from pathlib import Path
from typing import Any

from scripts.ci.runpod_client import (
    PermanentRunPodError,
    RetryClass,
    RetryableRunPodError,
    backoff_seconds,
    build_pod_payload,
    classify_http_status,
    create_pod,
    parse_create_response,
    redacted_error_message,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads(
    (ROOT / "scripts" / "ci" / "runpod_config.json").read_text()
)


class RunPodClientTests(unittest.TestCase):
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
        payload = build_pod_payload(CONFIG, "image", "startup", "jit")
        self.assertEqual(payload["cloudType"], "SECURE")
        self.assertIs(payload["interruptible"], False)
        self.assertEqual(payload["gpuTypeIds"], CONFIG["gpu_type_ids"])
        self.assertEqual(payload["env"]["RUNNER_JIT_CONFIG"], "jit")
        self.assertEqual(payload["dockerStartCmd"], ["startup"])

    def test_success_without_pod_id_is_protocol_error(self) -> None:
        with self.assertRaisesRegex(PermanentRunPodError, "missing pod id"):
            parse_create_response(201, b"{}")

    def test_permanent_error_is_not_retried(self) -> None:
        calls = 0

        def transport(_payload: bytes) -> tuple[int, dict[str, str], bytes]:
            nonlocal calls
            calls += 1
            return 400, {}, b'{"error":"bad gpu id"}'

        with self.assertRaisesRegex(PermanentRunPodError, "HTTP 400"):
            create_pod(
                CONFIG,
                b"{}",
                transport=transport,
                sleep=lambda _delay: self.fail("must not sleep"),
            )
        self.assertEqual(calls, 1)

    def test_retryable_responses_sleep_then_succeed(self) -> None:
        responses = iter(
            [
                (500, {}, b'{"error":"busy"}'),
                (503, {"Retry-After": "7"}, b'{"error":"busy"}'),
                (201, {}, b'{"id":"pod-1","machine":{"gpuTypeId":"NVIDIA A40"}}'),
            ]
        )
        sleeps: list[float] = []

        result = create_pod(
            CONFIG,
            b"{}",
            transport=lambda _payload: next(responses),
            sleep=sleeps.append,
            jitter=lambda: 0.5,
        )

        self.assertEqual(result.pod_id, "pod-1")
        self.assertEqual(result.gpu_type_id, "NVIDIA A40")
        self.assertEqual(sleeps, [5.0, 7.0])

    def test_transport_failure_is_retryable_but_bounded(self) -> None:
        calls = 0

        def transport(_payload: bytes) -> tuple[int, dict[str, str], bytes]:
            nonlocal calls
            calls += 1
            raise OSError("connection reset")

        config: dict[str, Any] = CONFIG | {"max_create_attempts": 2}
        with self.assertRaisesRegex(RetryableRunPodError, "transport failure"):
            create_pod(
                config,
                b"{}",
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
