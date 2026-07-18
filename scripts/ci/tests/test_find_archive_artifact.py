import json
import unittest

from scripts.ci.find_archive_artifact import (
    FinderError,
    find_reusable_artifact,
    is_trusted_producer_run,
)

REPO = "tensor4all/tenferro-rs"
NAME = "cuda-tests-abc123"


def run_payload(
    run_id: int,
    *,
    path: str = ".github/workflows/runpod-gpu-test.yml",
    event: str = "workflow_run",
    head_repo: str = REPO,
    head_branch: str = "main",
) -> dict:
    return {
        "id": run_id,
        "path": path,
        "event": event,
        "head_repository": {"full_name": head_repo},
        "head_branch": head_branch,
    }


def artifact_payload(
    artifact_id: int,
    run_id: int,
    *,
    name: str = NAME,
    expired: bool = False,
) -> dict:
    return {
        "id": artifact_id,
        "name": name,
        "expired": expired,
        "size_in_bytes": 123456,
        "created_at": "2026-07-18T00:00:00Z",
        "workflow_run": {"id": run_id},
    }


class FakeTransport:
    def __init__(self, listing: dict, runs: dict[int, dict]) -> None:
        self.listing = listing
        self.runs = runs
        self.urls: list[str] = []

    def __call__(self, url: str) -> tuple[int, bytes]:
        self.urls.append(url)
        if "/actions/artifacts?" in url:
            return 200, json.dumps(self.listing).encode()
        run_id = int(url.rsplit("/", 1)[-1])
        run = self.runs.get(run_id)
        if run is None:
            return 404, b"{}"
        return 200, json.dumps(run).encode()


class TrustedProducerTests(unittest.TestCase):
    def test_trusted_workflow_run_from_same_repo_is_accepted(self) -> None:
        trusted, reason = is_trusted_producer_run(
            run_payload(7), REPO, current_run_id=99
        )
        self.assertTrue(trusted, reason)

    def test_current_run_is_rejected(self) -> None:
        trusted, reason = is_trusted_producer_run(
            run_payload(99), REPO, current_run_id=99
        )
        self.assertFalse(trusted)
        self.assertIn("current run", reason)

    def test_pull_request_event_producer_is_rejected(self) -> None:
        trusted, reason = is_trusted_producer_run(
            run_payload(7, event="pull_request"), REPO, current_run_id=99
        )
        self.assertFalse(trusted)
        self.assertIn("pull_request", reason)

    def test_untrusted_workflow_path_is_rejected(self) -> None:
        trusted, reason = is_trusted_producer_run(
            run_payload(7, path=".github/workflows/ci.yml"),
            REPO,
            current_run_id=99,
        )
        self.assertFalse(trusted)
        self.assertIn("not trusted", reason)

    def test_fork_head_repository_is_rejected(self) -> None:
        trusted, reason = is_trusted_producer_run(
            run_payload(7, head_repo="attacker/tenferro-rs"),
            REPO,
            current_run_id=99,
        )
        self.assertFalse(trusted)
        self.assertIn("attacker/tenferro-rs", reason)

    def test_dispatch_on_non_default_branch_is_rejected(self) -> None:
        trusted, reason = is_trusted_producer_run(
            run_payload(
                7,
                path=".github/workflows/ci-cache-publish.yml",
                event="workflow_dispatch",
                head_branch="attacker-branch",
            ),
            REPO,
            current_run_id=99,
        )
        self.assertFalse(trusted)
        self.assertIn("attacker-branch", reason)

    def test_push_on_non_default_branch_is_rejected(self) -> None:
        trusted, reason = is_trusted_producer_run(
            run_payload(
                7,
                path=".github/workflows/ci-cache-publish.yml",
                event="push",
                head_branch="feature",
            ),
            REPO,
            current_run_id=99,
        )
        self.assertFalse(trusted)
        self.assertIn("default branch", reason)

    def test_workflow_run_event_skips_branch_check(self) -> None:
        # workflow_run definitions always resolve on the default branch, so
        # a PR head branch on the producing run is still trusted.
        trusted, reason = is_trusted_producer_run(
            run_payload(7, head_branch="pr-branch"),
            REPO,
            current_run_id=99,
        )
        self.assertTrue(trusted, reason)

    def test_cache_publish_workflow_push_is_accepted(self) -> None:
        trusted, reason = is_trusted_producer_run(
            run_payload(
                7,
                path=".github/workflows/ci-cache-publish.yml",
                event="push",
            ),
            REPO,
            current_run_id=99,
        )
        self.assertTrue(trusted, reason)


class FindReusableArtifactTests(unittest.TestCase):
    def test_newest_trusted_artifact_wins(self) -> None:
        transport = FakeTransport(
            {
                "artifacts": [
                    artifact_payload(31, 7),
                    artifact_payload(30, 6),
                ]
            },
            {7: run_payload(7), 6: run_payload(6)},
        )
        found = find_reusable_artifact(transport, REPO, NAME, current_run_id=99)
        self.assertIsNotNone(found)
        self.assertEqual(found["artifact_id"], 31)
        self.assertEqual(found["run_id"], 7)

    def test_expired_and_untrusted_artifacts_are_skipped(self) -> None:
        transport = FakeTransport(
            {
                "artifacts": [
                    artifact_payload(33, 9, expired=True),
                    artifact_payload(32, 8),
                    artifact_payload(31, 7),
                ]
            },
            {
                8: run_payload(8, event="pull_request"),
                7: run_payload(7),
            },
        )
        found = find_reusable_artifact(transport, REPO, NAME, current_run_id=99)
        self.assertIsNotNone(found)
        self.assertEqual(found["artifact_id"], 31)

    def test_name_mismatch_and_missing_run_are_skipped(self) -> None:
        other = artifact_payload(40, 9, name="cuda-tests-other")
        missing_run = artifact_payload(41, 12345)
        no_run = artifact_payload(42, 7)
        no_run["workflow_run"] = {}
        transport = FakeTransport(
            {"artifacts": [other, missing_run, no_run]},
            {},
        )
        with self.assertRaises(FinderError):
            # The missing producing run is a protocol error surfaced to the
            # caller, which downgrades it to a build fallback in main().
            find_reusable_artifact(transport, REPO, NAME, current_run_id=99)

    def test_no_candidates_returns_none(self) -> None:
        transport = FakeTransport({"artifacts": []}, {})
        self.assertIsNone(
            find_reusable_artifact(transport, REPO, NAME, current_run_id=99)
        )

    def test_current_run_artifact_is_not_reused(self) -> None:
        transport = FakeTransport(
            {"artifacts": [artifact_payload(31, 99)]},
            {99: run_payload(99)},
        )
        self.assertIsNone(
            find_reusable_artifact(transport, REPO, NAME, current_run_id=99)
        )

    def test_http_error_raises_finder_error(self) -> None:
        def transport(url: str) -> tuple[int, bytes]:
            return 500, b"{}"

        with self.assertRaises(FinderError):
            find_reusable_artifact(transport, REPO, NAME, current_run_id=99)


if __name__ == "__main__":
    unittest.main()
