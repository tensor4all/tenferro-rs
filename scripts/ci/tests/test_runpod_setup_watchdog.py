"""A stalled setup must delete its pod without waiting for the GPU job."""
import unittest
import datetime as dt
import io
import json
import urllib.error
from unittest.mock import patch

from scripts.ci.runpod_setup_watchdog import execution_started, main, request, watch


class SetupWatchdogTests(unittest.TestCase):
    def test_stalled_setup_deletes_at_aggregate_deadline(self):
        clock = [100.0]
        deleted = []
        def sleep(seconds):
            clock[0] += seconds
        self.assertFalse(watch(deadline=131, jobs=lambda: [],
                               delete=lambda: deleted.append(clock[0]),
                               now=lambda: clock[0], sleep=sleep, poll_seconds=15))
        self.assertEqual(deleted, [131])

    def test_real_cuda_start_disarms_without_deletion(self):
        jobs = [{"name": "Paid GPU lifecycle / CUDA GPU tests on RunPod",
                 "status": "in_progress", "steps": [{"name": "Run CUDA tests from archive",
                                                       "status": "in_progress"}]}]
        self.assertTrue(watch(deadline=0, jobs=lambda: jobs,
                              delete=lambda: self.fail("started tests were deleted"), now=lambda: 1))

    def test_failed_setup_is_deleted_without_waiting_for_cleanup_queue(self):
        rows = [{"name": "CUDA GPU tests on RunPod", "status": "completed", "conclusion": "failure"}]
        deleted = []
        self.assertFalse(execution_started(rows))
        self.assertFalse(watch(deadline=1000, jobs=lambda: rows, now=lambda: 1,
                               delete=lambda: deleted.append(True)))
        self.assertEqual(deleted, [True])

    def test_queued_skipped_and_unrelated_tests_do_not_disarm(self):
        for rows in ([{"name": "CPU tests", "status": "completed"}],
                     [{"name": "CUDA GPU tests on RunPod", "status": "queued", "steps": []}],
                     [{"name": "CUDA GPU tests on RunPod", "status": "in_progress", "steps": [
                         {"name": "Run CUDA tests from archive", "status": "completed", "conclusion": "skipped"}]}]):
            with self.subTest(jobs=rows):
                self.assertFalse(execution_started(rows))

    def test_progress_errors_do_not_extend_deadline(self):
        deleted = []
        def unavailable():
            raise OSError("temporary API outage")
        self.assertFalse(watch(deadline=1, jobs=unavailable, now=lambda: 2,
                               delete=lambda: deleted.append(True)))
        self.assertEqual(deleted, [True])

    def test_deletion_failure_is_visible(self):
        def rejected():
            raise RuntimeError("delete rejected")
        with self.assertRaisesRegex(RuntimeError, "delete rejected"):
            watch(deadline=0, jobs=lambda: [], delete=rejected, now=lambda: 1)

    def test_unreadable_pod_metadata_deletes_before_failing(self):
        environment = {"POD_ID": "test-pod", "RUNPOD_API_KEY": "fixture-key"}
        with patch.dict("os.environ", environment), patch("sys.argv", ["watchdog"]), \
             patch("scripts.ci.runpod_setup_watchdog.time.sleep"), \
             patch("scripts.ci.runpod_setup_watchdog.request", side_effect=[(503, b"")] * 3 + [(204, b"")]) as request:
            with self.assertRaisesRegex(RuntimeError, "start time"):
                main()
            self.assertEqual(request.call_args_list[-1].args[-1], "DELETE")

    def test_pod_already_deleted_is_idempotent(self):
        with patch.dict("os.environ", {"POD_ID": "test-pod", "RUNPOD_API_KEY": "fixture-key"}), \
             patch("sys.argv", ["watchdog"]), \
             patch("scripts.ci.runpod_setup_watchdog.request", return_value=(404, b"")) as request:
            self.assertEqual(main(), 0)
            self.assertEqual(request.call_count, 1)

    def test_workflow_keeps_watchdog_hosted_and_read_only(self):
        from pathlib import Path
        from scripts.ci.tests.test_runpod_cost_contracts import job
        child = (Path(__file__).resolve().parents[3] / ".github/workflows/runpod-gpu-execute.yml").read_text()
        guard = job(child, "setup-watchdog")
        self.assertIn("runs-on: ubuntu-latest", guard)
        self.assertIn("actions: read", guard)
        self.assertIn("--budget-seconds 900", guard)
        self.assertIn("ref: ${{ github.workflow_sha }}", guard)
        self.assertNotIn("actions: write", guard)
        self.assertNotIn("secrets.", job(child, "run-gpu-tests"))

    def test_failed_setup_delete_failure_is_not_a_progress_warning(self):
        rows = [{"name": "CUDA GPU tests on RunPod", "status": "completed", "conclusion": "failure"}]
        def rejected():
            raise RuntimeError("delete rejected")
        with self.assertRaisesRegex(RuntimeError, "delete rejected"):
            watch(deadline=1000, jobs=lambda: rows, now=lambda: 1, delete=rejected)

    def test_cli_deadline_retries_transient_deletion_and_returns_failure(self):
        environment = {"POD_ID": "test-pod", "RUNPOD_API_KEY": "fixture-key",
                       "GITHUB_REPOSITORY": "tensor4all/fixture", "GITHUB_RUN_ID": "123",
                       "GH_TOKEN": "fixture-github-key"}
        replies = [(200, b'{"lastStartedAt":"2000-01-01T00:00:00Z"}'),
                   (200, b'{"jobs":[]}'), (503, b''), (204, b'')]
        with patch.dict("os.environ", environment), patch("sys.argv", ["watchdog"]), \
             patch("scripts.ci.runpod_setup_watchdog.time.sleep"), \
             patch("scripts.ci.runpod_setup_watchdog.request", side_effect=replies) as call:
            self.assertEqual(main(), 1)
            self.assertEqual([c.args[-1] for c in call.call_args_list[-2:]], ["DELETE", "DELETE"])

    def test_cli_follows_job_pages_and_disarms_on_real_test_start(self):
        environment = {"POD_ID": "test-pod", "RUNPOD_API_KEY": "fixture-key",
                       "GITHUB_REPOSITORY": "tensor4all/fixture", "GITHUB_RUN_ID": "123",
                       "GH_TOKEN": "fixture-github-key"}
        pod = json.dumps({"lastStartedAt": dt.datetime.now(dt.timezone.utc).isoformat()}).encode()
        first = json.dumps({"jobs": [{"name": "Other job", "status": "completed"}] * 100}).encode()
        second = json.dumps({"jobs": [{"name": "CUDA GPU tests on RunPod", "status": "in_progress",
                          "steps": [{"name": "Run CUDA tests from archive", "status": "in_progress"}]}]}).encode()
        with patch.dict("os.environ", environment), patch("sys.argv", ["watchdog"]), \
             patch("scripts.ci.runpod_setup_watchdog.request", side_effect=[(200, pod), (200, first), (200, second)]) as call:
            self.assertEqual(main(), 0)
            self.assertIn("filter=latest", call.call_args_list[1].args[0])
            self.assertIn("page=2", call.call_args_list[2].args[0])
            self.assertEqual(call.call_count, 3)

    def test_http_error_status_is_preserved_for_retry_decisions(self):
        error = urllib.error.HTTPError("https://fixture.invalid", 429, "rate limited", {}, io.BytesIO(b"retry"))
        with patch("urllib.request.urlopen", side_effect=error):
            self.assertEqual(request("https://fixture.invalid", "fixture-key"), (429, b"retry"))

    def test_missing_or_null_start_time_confirms_deletion(self):
        for body in (b'null', b'{"lastStartedAt":null}', b'{}'):
            with self.subTest(body=body), \
                 patch.dict("os.environ", {"POD_ID": "test-pod", "RUNPOD_API_KEY": "fixture-key"}), \
                 patch("sys.argv", ["watchdog"]), \
                 patch("scripts.ci.runpod_setup_watchdog.request", side_effect=[(200, body), (204, b"")]) as call:
                with self.assertRaisesRegex(ValueError, "lastStartedAt"):
                    main()
                self.assertEqual(call.call_args_list[-1].args[-1], "DELETE")
