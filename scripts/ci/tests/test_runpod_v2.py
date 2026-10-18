import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse


from scripts.ci.runpod_client import (
    CreateRequest,
    PermanentRunPodError,
    RetryableRunPodError,
    build_pod_payload,
    create_pod,
    parse_create_response,
)
from scripts.ci.runpod_contract import validate_gpu_type_ids, validate_cuda_versions
from scripts.ci.runpod_http import Cooldown, Response, delete_pod, list_pods
from scripts.ci.runpod_pricing import candidate_plan, http_transport, parse_gpu_offers
from scripts.ci.runpod_provision import PodState, _pod_state_checker, parse_cost_per_hr
from scripts.ci.tests.test_runpod_http import Clock

FIXTURES = Path(__file__).with_name('fixtures')
ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads((ROOT / 'scripts/ci/runpod_config.json').read_text())
SCHEMA = json.loads((FIXTURES / 'runpod_v2_openapi.json').read_text())


def fixture(name):
    return (FIXTURES / f'runpod_v2_{name}.json').read_bytes()


class V2FixtureTests(unittest.TestCase):
    def test_published_response_examples_expose_v2_fields(self):
        for name in ('create', 'get'):
            pod = json.loads(fixture(name))
            self.assertEqual(pod['gpu']['id'], 'NVIDIA GeForce RTX 4090')
            self.assertEqual(pod['cost'], 0.44)
            self.assertIn('startedAt', pod)
        listing = json.loads(fixture('list'))
        self.assertEqual(listing['pods'][0], json.loads(fixture('get')))
        self.assertFalse(listing['pagination']['hasNextPage'])
        self.assertIn('gpus', json.loads(fixture('catalog')))

    def test_created_gpu_and_hourly_cost_use_v2_fields(self):
        result = parse_create_response(201, fixture('create'))
        self.assertEqual(result.gpu_type_id, 'NVIDIA GeForce RTX 4090')
        self.assertEqual(parse_cost_per_hr(result.body), 0.44)

    def test_catalog_replaces_pricing_query_and_enforces_allowlist(self):
        offers = parse_gpu_offers(
            fixture('catalog'), ['NVIDIA GeForce RTX 4090'], min_vram_gb=8
        )
        self.assertEqual(offers[0].price_per_hr, 0.44)
        self.assertEqual(offers[0].stock_status, 'HIGH')
        self.assertEqual(
            parse_gpu_offers(fixture('catalog'), ['NVIDIA A40'], min_vram_gb=8), []
        )

    def test_catalog_get_is_authenticated_and_requests_pod_cuda_availability(self):
        calls = []

        def transport(request, *, timeout):
            calls.append(request)
            return Response(200, {}, fixture('catalog'))

        send = http_transport(
            CONFIG['catalog_url'],
            api_key='key',
            config=CONFIG,
            transport=transport,
        )
        self.assertEqual(send()[0], 200)
        request = calls[0]
        self.assertEqual(request.method, 'GET')
        self.assertIsNone(request.data)
        self.assertEqual(request.get_header('Authorization'), 'Bearer key')
        query = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(
            query,
            {
                'include': ['AVAILABILITY'],
                'product': ['POD'],
                'cloud': ['SECURE'],
                'count': ['1'],
                'cudaVersions': [','.join(CONFIG['allowed_cuda_versions'])],
            },
        )

    def test_static_fallback_is_expanded_into_single_gpu_requests(self):
        plan = candidate_plan(
            {}, [('static', ['GPU-A', 'GPU-B'])], transport=lambda: (429, b'')
        )
        self.assertEqual(plan, [('static', ['GPU-A']), ('static', ['GPU-B'])])
        for _, ids in plan:
            payload = build_pod_payload(CONFIG, 'image', 'startup', 'jit', ids)
            self.assertEqual(payload['gpu']['id'], ids[0])
            self.assertEqual(payload['mounts']['persistent']['path'], '/workspace')

    def test_multi_gpu_type_payload_is_rejected(self):
        with self.assertRaisesRegex(PermanentRunPodError, 'exactly one GPU type'):
            build_pod_payload(CONFIG, 'image', 'startup', 'jit', ['GPU-A', 'GPU-B'])

    def test_runtime_and_terminal_status_replace_state_query(self):
        replies = iter(
            [
                json.loads(fixture('get')),
                {**json.loads(fixture('get')), 'status': 'ERROR', 'runtime': None},
            ]
        )
        requests = []

        def transport(request, *, timeout):
            requests.append(request)
            return Response(200, {}, json.dumps(next(replies)).encode())

        checker = _pod_state_checker(CONFIG['api_url'], 'key', transport=transport)
        self.assertEqual(checker('pod-1'), PodState('RUNNING', True))
        self.assertEqual(checker('pod-1'), PodState('ERROR', False))
        self.assertEqual(requests[0].full_url, CONFIG['api_url'] + '/pod-1')
        self.assertEqual(requests[0].method, 'GET')
        self.assertIsNone(requests[0].data)

    def test_wrapped_pod_listing_follows_cursor_and_omits_sensitive_fields(self):
        page = json.loads(fixture('list'))
        first = {**page, 'pagination': {'hasNextPage': True, 'nextCursor': 'opaque+/='}}
        replies = iter([first, page])
        requests = []

        def transport(request, *, timeout):
            requests.append(request)
            return Response(200, {}, json.dumps(next(replies)).encode())

        pods = list_pods(CONFIG['api_url'], 'key', transport=transport)
        self.assertEqual(len(pods), 2)
        self.assertEqual(set(pods[0]), {'id', 'name', 'status', 'cost'})
        self.assertEqual(
            parse_qs(urlparse(requests[1].full_url).query), {'cursor': ['opaque+/=']}
        )
        self.assertNotIn('env', json.dumps(pods))

    def test_listing_rejects_bare_arrays_bad_pagination_and_cycles(self):
        for payload in (
            [],
            {'pods': []},
            {'pods': [], 'pagination': {'hasNextPage': True, 'nextCursor': None}},
        ):
            with self.assertRaises(ValueError):
                list_pods(
                    CONFIG['api_url'],
                    'key',
                    transport=lambda request, **kw: Response(
                        200, {}, json.dumps(payload).encode()
                    ),
                )
        cyclic = {'pods': [], 'pagination': {'hasNextPage': True, 'nextCursor': 'same'}}
        with self.assertRaisesRegex(ValueError, 'repeated cursor'):
            list_pods(
                CONFIG['api_url'],
                'key',
                transport=lambda request, **kw: Response(
                    200, {}, json.dumps(cyclic).encode()
                ),
            )

    def test_listing_deadline_respects_long_retry_after_without_resend(self):
        calls = []
        clock = Clock()

        def transport(request, *, timeout):
            calls.append(request)
            return Response(429, {'Retry-After': '300'}, b'')

        with self.assertRaises(ValueError):
            list_pods(
                CONFIG['api_url'],
                'key',
                deadline_seconds=30,
                transport=transport,
                sleep=clock.sleep,
                monotonic=clock.monotonic,
            )
        self.assertEqual(len(calls), 1)

    def test_v2_problem_capacity_moves_to_next_candidate_but_validation_does_not(self):
        requests = [
            CreateRequest('first', b'first'),
            CreateRequest('second', b'second'),
        ]
        for failure in (
            (
                400,
                {},
                b'{"title":"Bad Request","status":400,"detail":"No available machine"}',
            ),
            (
                403,
                {},
                b'{"title":"Forbidden","status":403,"detail":"pool inaccessible"}',
            ),
        ):
            responses = iter([failure, (201, {}, fixture('create'))])
            result = create_pod(
                CONFIG,
                requests,
                transport=lambda payload, timeout: next(responses),
                sleep=lambda seconds: self.fail(
                    'candidate rejection must not retry in place'
                ),
            )
            self.assertEqual(result.gpu_tier, 'second')
        from scripts.ci.runpod_client import PermanentRunPodError

        with self.assertRaises(PermanentRunPodError):
            create_pod(
                CONFIG,
                requests,
                transport=lambda payload, timeout: (
                    422,
                    {},
                    b'{"title":"Unprocessable Entity","status":422,"detail":"invalid body","errors":["gpu required"]}',
                ),
            )

    def test_create_rate_limit_cannot_be_bypassed_by_next_candidate_or_poll(self):
        clock = Clock()
        cooldown = Cooldown()
        calls = []

        def transport(payload, timeout):
            calls.append(payload)
            return (
                429,
                {'Retry-After': '120'},
                b'{"title":"Too Many Requests","status":429,"detail":"limited"}',
            )

        for name in ('first', 'second'):
            with self.assertRaises(RetryableRunPodError):
                create_pod(
                    CONFIG,
                    [CreateRequest(name, name.encode())],
                    cooldown=cooldown,
                    transport=transport,
                    sleep=clock.sleep,
                    monotonic=clock.monotonic,
                )
        self.assertEqual(calls, [b'first'])
        checker = _pod_state_checker(
            CONFIG['api_url'],
            'key',
            cooldown=cooldown,
            monotonic=clock.monotonic,
            transport=lambda *a, **kw: self.fail('cooldown must prevent GET'),
        )
        self.assertEqual(checker('pod-1', deadline=420).retry_seconds, 120)

    def test_delete_waits_for_shared_cooldown_before_request(self):
        clock = Clock()
        times = []

        def transport(request, *, timeout):
            times.append(clock.now)
            return Response(
                404, {}, b'{"title":"Not Found","status":404,"detail":"gone"}'
            )

        self.assertTrue(
            delete_pod(
                CONFIG['api_url'],
                'key',
                'pod-1',
                cooldown=Cooldown(30),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                transport=transport,
            )
        )
        self.assertEqual(times, [30])

    def test_contract_cli_validates_offline_schema_and_catalog_fixtures(self):
        config = {
            **CONFIG,
            'gpu_tiers': [
                {'name': 'fixture', 'gpu_type_ids': ['NVIDIA GeForce RTX 4090']}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps(config))
            result = subprocess.run(
                [
                    'python3',
                    '-m',
                    'scripts.ci.runpod_contract',
                    '--config',
                    str(config_path),
                    '--schema-file',
                    str(FIXTURES / 'runpod_v2_openapi.json'),
                    '--catalog-file',
                    str(FIXTURES / 'runpod_v2_catalog.json'),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CUDA version syntax is valid', result.stdout)
        self.assertIn('Live capacity is checked separately', result.stdout)

    def test_v2_cost_snapshot_and_reaper_preserve_ownership(self):
        from scripts.ci.runpod_metadata import snapshot
        from scripts.ci.runpod_workflow_cost import report
        from scripts.ci.runpod_reap import reason_to_reap
        from scripts.ci.tests.test_runpod_lifecycle import POD, RUN, NOW, REPO
        pod = json.loads(fixture('get'))
        saved = snapshot(pod, pod['id'])
        self.assertEqual(set(saved), {'id', 'cost', 'startedAt', 'dataCenterId'})
        self.assertEqual(report(saved, [], '2026-06-01T13:02:00Z')['estimated_gpu_cost'], 0.44)
        self.assertNotIn('env', saved)
        owned = {**POD, 'startedAt': '2026-10-09T10:00:00Z', 'cost': 0.6}
        owned.pop('lastStartedAt')
        self.assertIn('two-hour', reason_to_reap(owned, RUN, REPO, NOW))
        owned.update(startedAt=None, createdAt='2026-10-09T10:00:00Z')
        self.assertIn('two-hour', reason_to_reap(owned, RUN, REPO, NOW))
        # Creation time must never be used to estimate paid time.
        with self.assertRaises((ValueError, AttributeError)):
            snapshot(owned, owned['id'])

    def test_pre_delete_read_cooldown_is_respected_by_delete(self):
        clock = Clock()
        calls = []
        def transport(request, *, timeout):
            calls.append((request.method, clock.now))
            if request.method == 'GET':
                return Response(429, {'Retry-After': '35'}, b'private')
            return Response(404, {}, b'')
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory)/'record.json'
            self.assertTrue(delete_pod(CONFIG['api_url'], 'key', 'pod',
                transport=transport, monotonic=clock.monotonic, sleep=clock.sleep,
                record_path=record))
            self.assertFalse(record.exists())
        self.assertEqual(calls[0], ('GET', 0))
        self.assertEqual(calls[1], ('DELETE', 35))

    def test_manual_cleanup_name_lookup_rejects_ambiguous_names(self):
        import os
        from unittest.mock import patch
        from scripts.ci.runpod_http import main
        with patch.dict(os.environ, {'RUNPOD_API_KEY':'key', 'POD_ID':''}), \
             patch('scripts.ci.runpod_http.list_pods', return_value=[
                 {'id':'first', 'name':'duplicate'}, {'id':'second', 'name':'duplicate'}]), \
             patch('scripts.ci.runpod_http.delete_pod') as delete:
            self.assertEqual(main(['delete', '--api-url', CONFIG['api_url'], '--pod-name', 'duplicate']), 1)
            delete.assert_not_called()

    def test_workflow_shell_blocks_parse_without_running_provider_calls(self):
        import re
        for path in ('.github/workflows/runpod-gpu-execute.yml', '.github/workflows/runpod-cleanup.yml'):
            content = (ROOT/path).read_text()
            self.assertNotIn('rest.runpod.io/v1', content)
            self.assertNotIn('api.runpod.io/graphql', content)
            # Parse each multiline run block without introducing a YAML dependency.
            for match in re.finditer(r'^        run: \|\n((?:^          .*\n|^\n)+)', content, re.M):
                script = '\n'.join(line[10:] for line in match.group(1).splitlines())
                script = re.sub(r'\$\{\{.*?\}\}', 'fixture', script)
                run = subprocess.run(['bash', '-n'], input=script, text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stderr)

    def test_operational_paths_have_no_retired_api_dependency(self):
        config = json.dumps(CONFIG)
        self.assertNotIn('graphql', config.lower())
        self.assertNotIn('/v1', config)
        for name in (
            'runpod_client.py',
            'runpod_provision.py',
            'runpod_pricing.py',
            'runpod_http.py',
            'runpod_lifecycle.py',
            'runpod_metadata.py',
            'runpod_reap.py',
            'runpod_setup_watchdog.py',
        ):
            source = (ROOT / 'scripts/ci' / name).read_text()
            self.assertNotIn('api.runpod.io/graphql', source)
            self.assertNotIn('rest.runpod.io/v1', source)


class LocalHttpIntegrationTests(unittest.TestCase):
    def test_real_http_create_catalog_liveness_list_metadata_and_retry_delete(self):
        """Exercise serialization and urllib without any provider credentials."""
        import http.server
        import threading
        from scripts.ci.runpod_client import _http_transport
        from scripts.ci.runpod_http import request_once
        from scripts.ci.runpod_lifecycle import HostedClient, owner
        from scripts.ci.runpod_metadata import snapshot
        from unittest.mock import patch

        pod = json.loads(fixture('get'))
        pod.update(name='tenferro-rs-gpu-ci-123', env={
            'TENFERRO_CI_OWNER': 'runpod-gpu-test-v1',
            'TENFERRO_CI_REPOSITORY': 'tensor4all/tenferro-rs',
            'TENFERRO_CI_RUN_ID': '123', 'TENFERRO_CI_ATTEMPT': '2',
            'TENFERRO_CI_KEEP_FAILED': 'false', 'RUNNER_JIT_CONFIG': 'jit'})
        observed = []
        deletes = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def send(self, status, value):
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                if status == 429:
                    self.send_header('Retry-After', '12')
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                observed.append((self.command, self.path, self.headers['Authorization'], body))
                self.send(201, json.loads(fixture('create')))

            def do_GET(self):
                observed.append((self.command, self.path, self.headers['Authorization'], None))
                if self.path.startswith('/v2/catalog/gpus'):
                    self.send(200, json.loads(fixture('catalog')))
                elif self.path == '/v2/pods':
                    self.send(200, {'pods': [pod], 'pagination': {'hasNextPage': False, 'nextCursor': None}})
                else:
                    self.send(200, pod)

            def do_DELETE(self):
                observed.append((self.command, self.path, self.headers['Authorization'], None))
                deletes.append(self.path)
                self.send(429 if len(deletes) == 1 else 204, {})

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}/v2'
            payload = build_pod_payload(CONFIG, 'image', 'startup', 'jit', ['NVIDIA GeForce RTX 4090'])
            created = create_pod(CONFIG,
                [CreateRequest('fixture', json.dumps(payload).encode(), ('NVIDIA GeForce RTX 4090',))],
                transport=_http_transport(base+'/pods', 'key'))
            self.assertEqual(created.pod_id, pod['id'])
            offers = http_transport(base+'/catalog/gpus', api_key='key', config=CONFIG)()
            self.assertEqual(parse_gpu_offers(offers[1], [created.gpu_type_id], min_vram_gb=8)[0].price_per_hr, 0.44)
            self.assertEqual(_pod_state_checker(base+'/pods', 'key')(pod['id']), PodState('RUNNING', True))
            with patch('scripts.ci.runpod_lifecycle.PODS_URL', base+'/pods'):
                client = HostedClient('tensor4all/tenferro-rs', 'github-unused', 'key')
                records = client.pods()
                self.assertEqual(owner(records[0], client.repository), (123, 2))
            self.assertNotIn('env', snapshot(records[0], pod['id']))
            clock = Clock()
            with tempfile.TemporaryDirectory() as directory:
                record = Path(directory)/'pod.json'
                self.assertTrue(delete_pod(base+'/pods', 'key', pod['id'],
                    record_path=record, monotonic=clock.monotonic, sleep=clock.sleep,
                    transport=request_once))
                self.assertEqual(json.loads(record.read_text())['cost'], 0.44)
            self.assertGreaterEqual(clock.now, 12)
            self.assertEqual(len(deletes), 2)
            self.assertTrue(all(call[2] == 'Bearer key' for call in observed))
            self.assertEqual(observed[0][3]['gpu']['count'], 1)
            self.assertTrue(all('/graphql' not in call[1] and '/v1/' not in call[1] for call in observed))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
