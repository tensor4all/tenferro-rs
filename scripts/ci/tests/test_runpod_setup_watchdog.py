"""Exercise the real lifecycle loop without renting a pod or waiting in real time."""
import datetime as dt
import io
import json
import os
from pathlib import Path
import subprocess
import unittest
import urllib.error
from unittest.mock import patch

from scripts.ci.runpod_setup_watchdog import execution_started, main, request, watch

GPU = 'Paid GPU lifecycle / CUDA GPU tests on RunPod'
STARTED = {'name': GPU, 'status': 'in_progress', 'steps': [
    {'name': 'Run CUDA tests from archive', 'status': 'in_progress'}]}
FINISHED = {**STARTED, 'status': 'completed'}
ENV = {'POD_ID': 'test-pod', 'RUNPOD_API_KEY': 'fixture-key',
       'GITHUB_REPOSITORY': 'tensor4all/fixture', 'GITHUB_RUN_ID': '123',
       'GITHUB_RUN_ATTEMPT': '2', 'GH_TOKEN': 'fixture-github-key',
       'PR_NUMBER': '0', 'TARGET_HEAD_SHA': ''}


class SetupWatchdogTests(unittest.TestCase):
    def test_stalled_setup_deletes_at_aggregate_deadline_then_cancels(self):
        clock, actions = [100.0], []
        def sleep(seconds):
            clock[0] += seconds
        self.assertFalse(watch(deadline=131, lifetime_deadline=1000, jobs=lambda: [],
            delete=lambda: actions.append(('delete', clock[0])),
            cancel=lambda: actions.append(('cancel', clock[0])),
            now=lambda: clock[0], sleep=sleep))
        self.assertEqual(actions, [('delete', 131), ('cancel', 131)])

    def test_real_cuda_start_disarms_setup_but_keeps_monitoring_until_completion(self):
        rows = iter([[STARTED], [FINISHED]])
        sleeps = []
        self.assertTrue(watch(deadline=0, lifetime_deadline=1000, jobs=lambda: next(rows),
            delete=lambda: self.fail('healthy tests deleted'), now=lambda: 1, sleep=sleeps.append))
        self.assertEqual(sleeps, [15])

    def test_head_move_after_cuda_start_deletes_before_cancelling(self):
        reasons = iter([None, 'PR head moved'])
        actions = []
        self.assertFalse(watch(deadline=0, lifetime_deadline=1000, jobs=lambda: [STARTED],
            obsolete=lambda: next(reasons), delete=lambda: actions.append('delete'),
            cancel=lambda: actions.append('cancel'), now=lambda: 1, sleep=lambda _: None))
        self.assertEqual(actions, ['delete', 'cancel'])

    def test_api_outage_after_test_start_does_not_rearm_setup_deadline(self):
        count, deletes = [0], []
        def jobs():
            count[0] += 1
            if count[0] == 1:
                return [STARTED]
            raise OSError('progress temporarily unavailable')
        clock = [1000]
        def sleep(seconds):
            clock[0] += seconds
        self.assertFalse(watch(deadline=900, lifetime_deadline=1030, jobs=jobs,
            delete=lambda: deletes.append(clock[0]), now=lambda: clock[0], sleep=sleep))
        self.assertEqual(deletes, [1030])

    def test_pr_api_error_never_means_obsolete(self):
        def unavailable():
            raise RuntimeError('HTTP 503')
        self.assertTrue(watch(deadline=100, lifetime_deadline=1000, jobs=lambda: [FINISHED],
            obsolete=unavailable, delete=lambda: self.fail('unknown PR deleted'), now=lambda: 1))

    def test_failed_setup_is_deleted_without_waiting_for_cleanup_queue(self):
        actions = []
        self.assertFalse(watch(deadline=1000, lifetime_deadline=2000,
            jobs=lambda: [{'name': GPU, 'status': 'completed', 'conclusion': 'failure'}],
            delete=lambda: actions.append('delete'), now=lambda: 1))
        self.assertEqual(actions, ['delete'])

    def test_queued_skipped_and_unrelated_tests_do_not_disarm(self):
        for rows in ([{'name': 'CPU tests', 'status': 'completed'}],
                     [{'name': GPU, 'status': 'queued', 'steps': []}],
                     [{'name': GPU, 'status': 'in_progress', 'steps': [
                         {'name': 'Run CUDA tests from archive', 'status': 'completed', 'conclusion': 'skipped'}]}]):
            with self.subTest(jobs=rows):
                self.assertFalse(execution_started(rows))

    def test_delete_failure_is_visible_and_never_cancels(self):
        def rejected():
            raise RuntimeError('delete rejected')
        with self.assertRaisesRegex(RuntimeError, 'delete rejected'):
            watch(deadline=0, lifetime_deadline=10, jobs=lambda: [], delete=rejected,
                  cancel=lambda: self.fail('cancelled before deletion'), now=lambda: 1)

    def test_unreadable_pod_metadata_deletes_before_failing(self):
        with patch.dict(os.environ, ENV), patch('sys.argv', ['watchdog']), \
             patch('scripts.ci.runpod_setup_watchdog.time.sleep'), \
             patch('scripts.ci.runpod_setup_watchdog.request', side_effect=[(503, b'')] * 3 + [(204, b'')]) as call:
            with self.assertRaisesRegex(RuntimeError, 'start time'):
                main()
            self.assertEqual(call.call_args_list[-1].args[-1], 'DELETE')

    def test_pod_already_deleted_is_idempotent(self):
        with patch.dict(os.environ, ENV), patch('sys.argv', ['watchdog']), \
             patch('scripts.ci.runpod_setup_watchdog.request', return_value=(404, b'')) as call:
            self.assertEqual(main(), 0)
            self.assertEqual(call.call_count, 1)

    def test_cli_deadline_retries_transient_deletion_then_cancels(self):
        replies = [(200, b'{"lastStartedAt":"2000-01-01T00:00:00Z"}'),
                   (200, b'{"jobs":[]}'), (503, b''), (204, b''),
                   (200, b'{"run_attempt":2,"status":"in_progress"}'), (202, b'')]
        with patch.dict(os.environ, ENV), patch('sys.argv', ['watchdog']), \
             patch('scripts.ci.runpod_lifecycle.time.sleep'), \
             patch('scripts.ci.runpod_setup_watchdog.request', side_effect=replies) as call:
            self.assertEqual(main(), 1)
            self.assertEqual([c.args[-1] for c in call.call_args_list[2:4]], ['DELETE', 'DELETE'])
            self.assertEqual(call.call_args_list[-1].args[-1], 'POST')

    def test_cli_follows_attempt_specific_job_pages(self):
        pod = json.dumps({'lastStartedAt': dt.datetime.now(dt.timezone.utc).isoformat()}).encode()
        first = json.dumps({'jobs': [{'name': 'Other job', 'status': 'completed'}] * 100}).encode()
        second = json.dumps({'jobs': [FINISHED]}).encode()
        with patch.dict(os.environ, ENV), patch('sys.argv', ['watchdog']), \
             patch('scripts.ci.runpod_setup_watchdog.request', side_effect=[(200, pod), (200, first), (200, second)]) as call:
            self.assertEqual(main(), 0)
            self.assertIn('/attempts/2/jobs?', call.call_args_list[1].args[0])
            self.assertIn('page=2', call.call_args_list[2].args[0])
            self.assertEqual(call.call_count, 3)

    def test_http_error_status_is_preserved_for_retry_decisions(self):
        error = urllib.error.HTTPError('https://fixture.invalid', 429, 'rate limited', {}, io.BytesIO(b'retry'))
        with patch('urllib.request.urlopen', side_effect=error):
            self.assertEqual(request('https://fixture.invalid', 'fixture-key'), (429, b'retry'))

    def test_missing_or_null_start_time_confirms_deletion(self):
        for body in (b'null', b'{"lastStartedAt":null}', b'{}'):
            with self.subTest(body=body), patch.dict(os.environ, ENV), \
                 patch('sys.argv', ['watchdog']), \
                 patch('scripts.ci.runpod_setup_watchdog.request', side_effect=[(200, body), (204, b'')]) as call:
                with self.assertRaisesRegex(ValueError, 'lastStartedAt'):
                    main()
                self.assertEqual(call.call_args_list[-1].args[-1], 'DELETE')

    def test_workflow_keeps_privileged_watcher_hosted_on_trusted_source(self):
        from scripts.ci.tests.test_runpod_cost_contracts import job
        child = (Path(__file__).resolve().parents[3] / '.github/workflows/runpod-gpu-execute.yml').read_text()
        guard = job(child, 'setup-watchdog')
        self.assertIn('runs-on: ubuntu-latest', guard)
        self.assertIn('actions: write', guard)
        self.assertIn('--budget-seconds 900 --lifetime-seconds 3600', guard)
        self.assertIn('ref: ${{ github.workflow_sha }}', guard)
        self.assertIn('persist-credentials: false', guard)
        self.assertIn('PR_NUMBER: ${{ inputs.pr_number }}', guard)
        self.assertIn('TARGET_HEAD_SHA: ${{ inputs.target_head_sha }}', guard)
        gpu = job(child, 'run-gpu-tests')
        self.assertNotIn('actions: write', gpu)
        self.assertNotIn('secrets.', gpu)

    def test_checkout_or_helper_failure_has_bounded_idempotent_deletion(self):
        from scripts.ci.tests.test_runpod_cost_contracts import job, step_script
        child = (Path(__file__).resolve().parents[3] / '.github/workflows/runpod-gpu-execute.yml').read_text()
        step = child.split('- name: Delete pod if setup watchdog could not finish', 1)[1].split('\n  run-gpu-tests:', 1)[0]
        self.assertIn('if: failure()', step)
        self.assertIn('timeout-minutes: 1', step)
        script = step_script(job(child, 'setup-watchdog'), 'Delete pod if setup watchdog could not finish')
        stub = ('curl() { [[ " $* " == *" -X DELETE https://rest.runpod.io/v1/pods/fixture "* ]] '
                '|| return 2; printf "%s" "$FIXTURE_STATUS"; }\n')
        for status, success in (('204', True), ('404', True), ('403', False), ('503', False)):
            with self.subTest(status=status):
                run = subprocess.run(['bash', '-c', stub + script], capture_output=True, text=True,
                    env={**os.environ, 'FIXTURE_STATUS': status, 'POD_ID': 'fixture',
                         'RUNPOD_API_KEY': 'fixture-key'})
                self.assertEqual(run.returncode == 0, success, run.stdout + run.stderr)
