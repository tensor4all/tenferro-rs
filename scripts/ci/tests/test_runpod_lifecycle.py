import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.ci.runpod_lifecycle import HostedClient, WORKFLOW_PATH, owner, ownership_environment
from scripts.ci.runpod_reap import main, reap, reason_to_reap

REPO = "tensor4all/tenferro-rs"
HEAD = "a" * 40
ENV = {"GITHUB_REPOSITORY": REPO, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"}
NOW = dt.datetime(2026, 10, 9, 12, tzinfo=dt.timezone.utc)
POD = {"id": "pod", "name": "tenferro-rs-gpu-ci-123", "env": ownership_environment(ENV),
       "lastStartedAt": "2026-10-09T11:50:00Z"}
RUN = {"id": 123, "run_attempt": 2, "status": "in_progress", "path": WORKFLOW_PATH,
       "repository": {"full_name": REPO}, "updated_at": "2026-10-09T11:50:00Z"}


class HostedLifecycleTests(unittest.TestCase):
    def test_metadata_has_no_tokens_and_legacy_local_calls_are_not_reaped(self):
        env = ownership_environment({**ENV, "RUNPOD_API_KEY": "secret", "GH_TOKEN": "secret"})
        self.assertNotIn("secret", json.dumps(env))
        self.assertEqual(owner({**POD, "env": env}, REPO), (123, 2))
        self.assertEqual(ownership_environment({}), {})
        self.assertIsNone(owner({**POD, "env": {}}, REPO))

    def test_foreign_debug_and_bad_identity_pods_are_never_owned(self):
        for changes in ({"TENFERRO_CI_REPOSITORY": "other/repo"},
                        {"TENFERRO_CI_KEEP_FAILED": "unknown"},
                        {"TENFERRO_CI_RUN_ID": "0"}, {"TENFERRO_CI_ATTEMPT": "x"},
                        {"TENFERRO_CI_OWNER": "some-other-service"}):
            self.assertIsNone(owner({**POD, "env": {**POD["env"], **changes}}, REPO))
        self.assertIsNone(owner({**POD, "name": "my-personal-training"}, REPO))

    def test_debug_dispatch_is_owned_until_actual_retention_is_recorded(self):
        debug = {**POD, "env": {**POD["env"], "TENFERRO_CI_KEEP_FAILED": "true"}}
        self.assertEqual(owner(debug, REPO), (123, 2))
        transport = Mock(return_value=(200, json.dumps({"artifacts": [
            {"name": "runpod-debug-retained-123-1-old"},
            {"name": "runpod-debug-retained-123-2-rejected,another"}]}).encode()))
        client = HostedClient(REPO, "token", transport=transport)
        self.assertTrue(client.retained_for_debug(123, 2, "rejected"))
        self.assertFalse(client.retained_for_debug(123, 2, "accepted"))
        self.assertFalse(client.retained_for_debug(123, 2, "old"))

    def test_manual_validation_does_not_query_pr_state(self):
        client = HostedClient(REPO, "token", transport=Mock(side_effect=AssertionError("unexpected API")))
        self.assertIsNone(client.obsolete("0", ""))

    def test_only_closed_or_moved_heads_are_obsolete(self):
        transport = Mock(return_value=(200, json.dumps({"state": "open", "head": {"sha": HEAD}}).encode()))
        client = HostedClient(REPO, "token", transport=transport)
        self.assertIsNone(client.obsolete("7", HEAD))
        transport.return_value = (200, b'{"state":"closed"}')
        self.assertIn("closed", client.obsolete("7", HEAD))
        transport.return_value = (200, json.dumps({"state": "open", "head": {"sha": "b" * 40}}).encode())
        self.assertIn("head moved", client.obsolete("7", HEAD))
        for response in ((503, b""), (200, b"{}"), (200, b'{"state":"open","head":null}')):
            transport.return_value = response
            with self.assertRaises((RuntimeError, ValueError)):
                client.obsolete("7", HEAD)

    def test_cancel_never_targets_a_newer_attempt_or_completed_run(self):
        for run in ({**RUN, "run_attempt": 3}, {**RUN, "status": "completed"}):
            transport = Mock(return_value=(200, json.dumps(run).encode()))
            HostedClient(REPO, "token", transport=transport).cancel("123", "2")
            self.assertEqual(transport.call_count, 1)

    def test_cancel_conflict_requires_confirmed_completion(self):
        for status, success in (("completed", True), ("in_progress", False)):
            transport = Mock(side_effect=[(200, json.dumps(RUN).encode()), (409, b""),
                (200, json.dumps({**RUN, "status": status}).encode())])
            client = HostedClient(REPO, "token", transport=transport)
            if success:
                client.cancel("123", "2")
            else:
                with self.assertRaisesRegex(RuntimeError, "cancellation failed"):
                    client.cancel("123", "2")

    def test_delete_retries_transient_errors_but_rejects_permanent_failure(self):
        transport = Mock(side_effect=[OSError("network"), (429, b""), (404, b"")])
        with patch("scripts.ci.runpod_lifecycle.time.sleep"):
            HostedClient(REPO, "token", "pod-key", transport=transport).delete("pod")
        self.assertEqual(transport.call_count, 3)
        transport = Mock(return_value=(403, b""))
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            HostedClient(REPO, "token", "pod-key", transport=transport).delete("pod")
        self.assertEqual(transport.call_count, 1)


class ReaperTests(unittest.TestCase):
    def test_completed_grace_and_active_deadline(self):
        self.assertIsNone(reason_to_reap(POD, RUN, REPO, NOW))
        self.assertIn("completed", reason_to_reap(POD, {**RUN, "status": "completed"}, REPO, NOW))
        self.assertIsNone(reason_to_reap(POD, {**RUN, "status": "completed",
            "updated_at": "2026-10-09T11:59:00Z"}, REPO, NOW))
        expired = {**POD, "lastStartedAt": "2026-10-09T10:00:00Z"}
        self.assertIn("two-hour", reason_to_reap(expired, RUN, REPO, NOW))

    def test_superseded_attempt_is_reclaimed_but_mismatched_owner_is_error(self):
        self.assertIn("superseded", reason_to_reap(POD, {**RUN, "run_attempt": 3}, REPO, NOW))
        for changes in ({"run_attempt": 1}, {"path": "another-workflow.yml"},
                        {"repository": {"full_name": "other/repo"}}, {"id": 999}):
            with self.assertRaises(ValueError):
                reason_to_reap(POD, {**RUN, **changes}, REPO, NOW)

    def test_dry_run_and_foreign_pods_never_delete_or_cancel(self):
        client = Mock(repository=REPO)
        client.github.return_value = {**RUN, "status": "completed"}
        result = reap(client, [POD, {**POD, "env": {}}], execute=False, now=NOW)
        self.assertEqual(len(result["pods"]), 1)
        self.assertEqual(result["errors"], [])
        client.delete.assert_not_called()
        client.cancel.assert_not_called()

    def test_delete_happens_before_cancel_and_other_pods_survive_error(self):
        client = Mock(repository=REPO)
        client.github.return_value = {**RUN, "status": "completed"}
        client.delete.side_effect = [RuntimeError("deletion failed"), "2026-10-09T12:00:00Z"]
        result = reap(client, [POD, {**POD, "id": "second"}], execute=True, now=NOW)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(client.cancel.call_count, 1)
        calls = [c[0] for c in client.mock_calls]
        self.assertEqual(calls, ["github", "delete", "github", "delete", "cancel"])

    def test_reaper_cost_uses_provider_start_and_confirmed_deletion(self):
        client = Mock(repository=REPO)
        client.github.return_value = {**RUN, "status": "completed"}
        client.delete.return_value = "2026-10-09T12:00:00Z"
        result = reap(client, [{**POD, "costPerHr": 0.6}], execute=True, now=NOW)
        self.assertEqual(result["pods"][0]["cost"]["paid_seconds"], 600)
        self.assertAlmostEqual(result["pods"][0]["cost"]["estimated_gpu_cost"], 0.1)

    def test_unknown_github_state_does_not_delete_even_old_pod(self):
        client = Mock(repository=REPO)
        client.github.side_effect = RuntimeError("HTTP 503")
        result = reap(client, [{**POD, "lastStartedAt": "2000-01-01T00:00:00Z"}], execute=True, now=NOW)
        self.assertEqual(len(result["errors"]), 1)
        client.delete.assert_not_called()

    def test_debug_flag_alone_never_exempts_accepted_or_obsolete_pods(self):
        debug = {**POD, "env": {**POD["env"], "TENFERRO_CI_KEEP_FAILED": "true"}}
        for retained in (True, False):
            client = Mock(repository=REPO)
            client.github.return_value = {**RUN, "status": "completed"}
            client.retained_for_debug.return_value = retained
            client.delete.return_value = NOW.isoformat()
            result = reap(client, [debug], execute=True, now=NOW)
            self.assertEqual(result["errors"], [])
            self.assertEqual(client.delete.called, not retained)
            self.assertEqual(client.cancel.called, not retained)
        client = Mock(repository=REPO)
        client.github.return_value = {**RUN, "status": "completed"}
        client.retained_for_debug.side_effect = RuntimeError("HTTP 503")
        result = reap(client, [debug], execute=True, now=NOW)
        self.assertEqual(len(result["errors"]), 1)
        client.delete.assert_not_called()

    def test_cli_reads_provider_array_and_writes_only_decisions(self):
        client = Mock(repository=REPO)
        client.transport.return_value = (200, json.dumps([{**POD, "env": {}}]).encode())
        with tempfile.TemporaryDirectory() as folder, patch("sys.argv", ["reaper", "--output", folder + "/result.json"]), \
                patch("scripts.ci.runpod_reap.HostedClient.from_environment", return_value=client):
            self.assertEqual(main(), 0)
            result = json.loads((Path(folder) / "result.json").read_text())
            self.assertFalse(result["execute"])
            self.assertEqual(result["pods"], [])
            self.assertNotIn("env", result)

    def test_scheduled_workflow_is_independent_hosted_and_trusted(self):
        path = Path(__file__).resolve().parents[3] / ".github/workflows/runpod-reap.yml"
        workflow = path.read_text()
        self.assertIn("schedule:", workflow)
        self.assertIn("if: github.ref == 'refs/heads/main'", workflow)
        self.assertIn("ref: ${{ github.workflow_sha }}", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("runs-on: ubuntu-24.04", workflow)
        self.assertNotIn("self-hosted", workflow)
        self.assertNotIn("pull_request", workflow)
