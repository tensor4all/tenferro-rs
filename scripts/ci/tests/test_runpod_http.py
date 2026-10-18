import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from scripts.ci import runpod_http as http


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class HttpTests(unittest.TestCase):
    def test_retry_after_formats_and_invalid_values(self):
        self.assertEqual(http.retry_after({'retry-after': '12'}), 12)
        self.assertEqual(
            http.retry_after(
                {'Retry-After': 'Thu, 01 Jan 1970 00:01:00 GMT'}, wall_time=lambda: 10
            ),
            50,
        )
        self.assertEqual(http.retry_after({'Retry-After': '-2'}), 0)
        for value in ('bad', 'nan', 'inf'):
            self.assertIsNone(http.retry_after({'Retry-After': value}))

    def delete(self, responses, **kwargs):
        clock = Clock()
        calls = []
        replies = iter(responses)

        def transport(request, *, timeout):
            calls.append((request.method, clock.now, timeout))
            return next(replies)

        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = http.delete_pod(
                'https://example.test/pods',
                'secret',
                'pod-1',
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                transport=transport,
                **kwargs,
            )
        return result, calls, clock

    def test_delete_success_and_already_gone(self):
        for status in (200, 204, 404):
            result, calls, _ = self.delete([http.Response(status, {}, b'')])
            self.assertTrue(result)
            self.assertEqual(len(calls), 1)

    def test_delete_retry_after_then_transient_then_success(self):
        result, calls, clock = self.delete(
            [
                http.Response(429, {'Retry-After': '30'}, b''),
                http.Response(503, {}, b''),
                http.Response(204, {}, b''),
            ]
        )
        self.assertTrue(result)
        self.assertGreaterEqual(calls[1][1], 30)
        self.assertGreaterEqual(clock.sleeps[1], 10)
        self.assertEqual([c[0] for c in calls], ['DELETE'] * 3)

    def test_http_date_retry_after_controls_actual_resend(self):
        from email.utils import formatdate
        import time

        result, calls, _ = self.delete(
            [
                http.Response(
                    429,
                    {'Retry-After': formatdate(time.time() + 120, usegmt=True)},
                    b'',
                ),
                http.Response(204, {}, b''),
            ]
        )
        self.assertTrue(result)
        self.assertGreater(calls[1][1], 118)

    def test_no_request_or_verification_before_long_retry_after(self):
        result, calls, _ = self.delete(
            [http.Response(429, {'Retry-After': '1000'}, b'')], deadline_seconds=10
        )
        self.assertFalse(result)
        self.assertEqual(len(calls), 1)

    def test_permanent_error_is_not_retried(self):
        for status in (400, 401, 403):
            result, calls, _ = self.delete([http.Response(status, {}, b'secret')])
            self.assertFalse(result)
            self.assertEqual(len(calls), 1)

    def test_attempt_limit(self):
        result, calls, _ = self.delete([http.Response(503, {}, b'')] * 3, attempts=3)
        self.assertFalse(result)
        self.assertEqual(len(calls), 3)

    def test_lost_response_can_be_confirmed_gone(self):
        result, calls, _ = self.delete(
            [http.Response(0, {}, b''), http.Response(404, {}, b'')], attempts=1
        )
        self.assertTrue(result)
        self.assertEqual([c[0] for c in calls], ['DELETE', 'GET'])

    def test_request_deadline_caps_timeout_and_stops_new_attempts(self):
        clock = Clock()
        calls = []

        def timeout_transport(request, *, timeout):
            calls.append(timeout)
            clock.now += timeout
            return http.Response(0, {}, b'')

        response = http.request_with_retry(
            urllib.request.Request('https://example.test'),
            deadline=3,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            transport=timeout_transport,
        )
        self.assertEqual(response.status, 0)
        self.assertEqual(calls, [3])
        self.assertEqual(clock.sleeps, [])

    def test_transport_http_error_preserves_headers_and_timeout_is_transient(self):
        request = urllib.request.Request('https://example.test')
        error = urllib.error.HTTPError(
            request.full_url, 429, 'limited', {'Retry-After': '40'}, io.BytesIO(b'body')
        )
        with mock.patch.object(http.urllib.request, 'urlopen', side_effect=error):
            response = http.request_once(request, timeout=3)
        self.assertEqual(response.status, 429)
        self.assertEqual(response.headers['Retry-After'], '40')
        from http.client import IncompleteRead

        for error in (TimeoutError('secret'), IncompleteRead(b'partial')):
            with mock.patch.object(http.urllib.request, 'urlopen', side_effect=error):
                self.assertEqual(http.request_once(request, timeout=3).status, 0)

    def test_failure_summary_contains_pod_id_but_no_credentials_or_body(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / 'summary.md'
            with (
                mock.patch.dict(os.environ, {'GITHUB_STEP_SUMMARY': str(summary)}),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.delete([http.Response(401, {}, b'secret')])
            self.assertIn('pod-1', summary.read_text())
            self.assertIn('dashboard', summary.read_text())
            self.assertNotIn('secret', summary.read_text())

    def test_cli_exit_codes(self):
        with mock.patch.dict(os.environ, {'RUNPOD_API_KEY': 'key', 'POD_ID': 'pod-1'}):
            for success, expected in ((True, 0), (False, 1)):
                with mock.patch.object(http, 'delete_pod', return_value=success):
                    self.assertEqual(
                        http.main(['delete', '--api-url', 'https://example.test/pods']),
                        expected,
                    )
