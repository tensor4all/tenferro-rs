import subprocess
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci.runpod_metadata import main, snapshot
from scripts.ci.runpod_workflow_cost import report, select_pod_record

POD = {'id': 'accepted', 'lastStartedAt': '2026-10-10T15:44:00Z',
       'costPerHr': 0.59, 'machineId': 'machine', 'machine': {'dataCenterId': 'CA-MTL-1'}}
FINISH = '2026-10-10T15:49:42Z'


class PodMetadataTests(unittest.TestCase):
    def test_snapshot_retains_reporting_fields_without_credentials(self):
        value = snapshot(dict(POD, env={'JIT_CONFIG': 'secret'}, dockerStartCmd=['secret'],
                              machine={'dataCenterId': 'CA-MTL-1', 'other': 'private'}), 'accepted')
        self.assertEqual(value, POD)
        with self.assertRaises(ValueError):
            snapshot(POD, 'different-pod')

    def test_final_dns_failure_keeps_exact_cost_and_placement_from_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pod.json'
            for content in (None, '', '<html>HTTP 500</html>', '{}', json.dumps(dict(POD, id='wrong'))):
                with self.subTest(content=content):
                    if content is not None:
                        path.write_text(content)
                    value, source = select_pod_record(path, json.dumps(POD), 'accepted', FINISH)
                    self.assertEqual(source, 'startup')
                    self.assertEqual(value, POD)
                    self.assertEqual(report(value, [], FINISH)['paid_seconds'], 342)
                    self.assertAlmostEqual(report(value, [], FINISH)['estimated_gpu_cost'], 342 * 0.59 / 3600)

    def test_final_record_wins_and_no_price_or_start_is_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pod.json'
            latest = dict(POD, adjustedCostPerHr=0.4)
            path.write_text(json.dumps(latest))
            value, source = select_pod_record(path, json.dumps(POD), 'accepted', FINISH)
            self.assertEqual((value, source), (latest, 'pre_delete'))
            path.write_text('{}')
            for fallback in ({'id': 'accepted'}, dict(POD, id='wrong'), dict(POD, costPerHr=0)):
                with self.assertRaises(ValueError):
                    select_pod_record(path, json.dumps(fallback), 'accepted', FINISH)

    def test_startup_capture_and_creation_fallback_publish_only_snapshot(self):
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                full = dict(POD, env={'JIT_CONFIG': 'secret'})
                (root / 'creation').write_text(json.dumps(full))
                with patch('sys.argv', ['metadata', '--pod-id', 'accepted', '--creation-record', str(root/'creation')]), \
                     patch.dict(os.environ, RUNPOD_API_KEY='key', GITHUB_OUTPUT=str(root/'output')), \
                     patch('scripts.ci.runpod_metadata.subprocess.run') as request:
                    if unavailable:
                        request.side_effect = TimeoutError('DNS timeout')
                    else:
                        request.return_value = subprocess.CompletedProcess([], 0, json.dumps(full).encode())
                    self.assertEqual(main(), 0)
                    value = (root/'output').read_text()
                    self.assertEqual(json.loads(value.removeprefix('pod_metadata=')), POD)
                    self.assertNotIn('secret', value)
                    self.assertEqual(request.call_args.kwargs['timeout'], 6)
                    self.assertIn('--max-time', request.call_args.args[0])

    def test_missing_startup_and_creation_data_does_not_fail_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('sys.argv', ['metadata', '--pod-id', 'accepted', '--creation-record', directory+'/missing']), \
             patch.dict(os.environ, RUNPOD_API_KEY='key', GITHUB_OUTPUT=directory+'/output'), \
             patch('scripts.ci.runpod_metadata.subprocess.run', side_effect=TimeoutError()):
            self.assertEqual(main(), 0)
            self.assertFalse(Path(directory, 'output').exists())
