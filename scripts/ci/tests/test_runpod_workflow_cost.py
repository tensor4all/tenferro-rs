from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from scripts.ci.runpod_workflow_cost import report


class StageCostTests(unittest.TestCase):
    def test_stage_costs_reconcile_with_exact_paid_window(self):
        pod = {'id': 'pod', 'lastStartedAt': '2026-10-08 13:00:00.000 +0000 UTC',
               'costPerHr': 0.5, 'adjustedCostPerHr': 0.4}
        job = {'name': 'Paid GPU lifecycle / CUDA GPU tests on RunPod',
               'started_at': '2026-10-08T13:01:00Z', 'completed_at': '2026-10-08T13:09:00Z',
               'conclusion': 'success', 'steps': [
                   {'name': 'Download CUDA/PJRT test archives', 'conclusion': 'success',
                    'started_at': '2026-10-08T13:01:00Z', 'completed_at': '2026-10-08T13:02:00Z'},
                   {'name': 'Restore CUDA minimal runtime tree', 'conclusion': 'success',
                    'started_at': '2026-10-08T13:02:00Z', 'completed_at': '2026-10-08T13:03:00Z'},
                   {'name': 'Retry CUDA/PJRT test archive download', 'conclusion': 'skipped'},
                   {'name': 'Run CUDA tests from archive', 'conclusion': 'success',
                    'started_at': '2026-10-08T13:03:00Z', 'completed_at': '2026-10-08T13:09:00Z'}]}
        result = report(pod, [job], '2026-10-08T13:10:00Z')
        self.assertEqual(result['paid_seconds'], 600)
        self.assertAlmostEqual(result['estimated_gpu_cost'], 1 / 15)
        self.assertEqual(result['stages']['cuda_tests']['seconds'], 360)
        self.assertEqual(result['stages']['archive_transfer']['seconds'], 60)
        self.assertEqual(sum(s['seconds'] for s in result['stages'].values()), 600)
        self.assertAlmostEqual(sum(s['estimated_gpu_cost'] for s in result['stages'].values()), result['estimated_gpu_cost'])

    def test_no_gpu_job_remains_paid_overhead_not_success(self):
        result = report({'costPerHr': 0.5, 'lastStartedAt': '2026-10-08T13:00:00Z'}, [], '2026-10-08T13:01:00Z')
        self.assertEqual(result['gpu_job_conclusion'], 'not_run')
        self.assertEqual(result['stages']['unassigned_overhead']['seconds'], 60)

    def test_invalid_metadata_warns_without_failing_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pod').write_text('{}')
            (root / 'jobs').write_text('[]')
            proc = subprocess.run([sys.executable, '-m', 'scripts.ci.runpod_workflow_cost',
                '--pod', str(root / 'pod'), '--jobs', str(root / 'jobs'),
                '--deleted-at', '2026-10-08T13:01:00Z', '--output', str(root / 'output')],
                cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            self.assertIn('::warning::', proc.stdout)
            self.assertFalse((root / 'output').exists())

    def test_bad_price_or_reversed_window_is_rejected(self):
        for price, finish in ((float('nan'), '2026-10-08T13:01:00Z'),
                              (-0.5, '2026-10-08T13:01:00Z'),
                              (0.5, '2026-10-08T12:59:00Z')):
            with self.subTest(price=price, finish=finish):
                with self.assertRaises(ValueError):
                    report({'costPerHr': price, 'lastStartedAt': '2026-10-08T13:00:00Z'}, [], finish)

    def test_cli_preserves_failed_workload_and_paginated_cache_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pod').write_text(json.dumps({'id': 'actual-pod', 'costPerHr': 0.5,
                'lastStartedAt': '2026-10-08T13:00:00Z'}))
            (root / 'jobs').write_text(json.dumps([{'jobs': [{'name': 'Prepare', 'conclusion': 'success'}]},
                {'jobs': [{'name': 'CUDA GPU tests on RunPod', 'conclusion': 'failure',
                'started_at': '2026-10-08T13:00:10Z', 'completed_at': '2026-10-08T13:00:50Z',
                'steps': [{'name': 'Run CUDA tests from archive', 'conclusion': 'failure',
                'started_at': '2026-10-08T13:00:10Z', 'completed_at': '2026-10-08T13:00:50Z'}]}]}]))
            proc = subprocess.run([sys.executable, '-m', 'scripts.ci.runpod_workflow_cost',
                '--pod', str(root / 'pod'), '--jobs', str(root / 'jobs'),
                '--deleted-at', '2026-10-08T13:01:00Z', '--output', str(root / 'output'),
                '--tested-ref', 'measured-ref', '--archive-cache-hit', 'true',
                '--cutensor-cache-hit', 'false'], cwd=Path(__file__).resolve().parents[3],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            value = json.loads((root / 'output').read_text())
            self.assertEqual(value['gpu_job_conclusion'], 'failure')
            self.assertEqual(value['tested_ref'], 'measured-ref')
            self.assertEqual(value['cache_hits'], {'archives': True, 'cutensor': False, 'cuda_runtime': None})
            self.assertEqual(value['stages']['cuda_tests']['seconds'], 40)
            self.assertEqual(value['paid_seconds'], 60)
