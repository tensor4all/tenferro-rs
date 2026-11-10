"""Focused tests for the oracle nightly failure reporter.

The reporter is ``scripts/ci/report_oracle_nightly.sh``; the nightly workflow
only calls it.  Each test shells out to the real script with a stub ``gh`` first
on ``PATH``, so the issue body, the deduplication and the recovery decision are
exercised the way the nightly runs them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "report_oracle_nightly.sh"
FIXTURES = Path(__file__).with_name("fixtures") / "oracle-nightly"
RUNS = json.loads((FIXTURES / "runs.json").read_text())
LABEL = "nightly-oracle-failure"

# `gh` stands in for the API. `LABELS_STATE` is the repository's label set, so a
# lookup that needs a missing label fails the way the real CLI does. The
# `--search` emulation matches the phrase as a substring only, so the exact-title
# filter in the reporter is exercised rather than assumed.
STUB_GH = """#!{python}
import json, os, sys

args = sys.argv[1:]
calls_file = os.environ["GH_CALLS"]
labels_file = os.environ["LABELS_STATE"]


def labels():
    if not os.path.exists(labels_file):
        return set()
    return set(open(labels_file, encoding="utf-8").read().split())


def record():
    with open(calls_file, "a", encoding="utf-8") as calls:
        calls.write(json.dumps(args) + "\\n")


def issues():
    return json.loads(open(os.environ["ISSUES_FIXTURE"], encoding="utf-8").read())


record()
command = args[:2]
if command == ["label", "create"]:
    with open(labels_file, "a", encoding="utf-8") as state:
        state.write(args[2] + "\\n")
    sys.exit(0)
if command == ["run", "view"]:
    sys.stdout.write("- Root-facade oracle replay\\n")
    sys.exit(0)
if command == ["issue", "list"]:
    if os.environ.get("GH_FAIL_LOOKUP") == "1" and "--label" in args:
        sys.stderr.write("HTTP 502: Bad Gateway\\n")
        sys.exit(1)
    found = issues()
    if "--label" in args:
        wanted = args[args.index("--label") + 1]
        if wanted not in labels():
            # `gh issue list --label` fails when the label does not exist yet.
            sys.stderr.write("the label %s was not found\\n" % wanted)
            sys.exit(1)
        found = [i for i in found if wanted in [l["name"] for l in i["labels"]]]
    if "--search" in args:
        found = [i for i in found if "oracle-replay-nightly is failing" in i["title"]]
    json.dump(found, sys.stdout)
    sys.exit(0)
if command == ["issue", "create"]:
    if "--label" in args and args[args.index("--label") + 1] not in labels():
        sys.stderr.write("could not add label: not found\\n")
        sys.exit(1)
    sys.stdout.write("https://example.invalid/tensor4all/tenferro-rs/issues/99\\n")
    sys.exit(0)
if command in (["issue", "comment"], ["issue", "close"]):
    sys.exit(0)
if command == ["issue", "edit"]:
    if not set(args[args.index("--add-label") + 1:]).issubset(labels()):
        sys.stderr.write("could not add label: not found\\n")
        sys.exit(1)
    sys.exit(0)
sys.stderr.write("stub gh: unsupported command: " + " ".join(args) + "\\n")
sys.exit(1)
""".format(python=sys.executable)

RUN_URL = "https://example.invalid/tensor4all/tenferro-rs/actions/runs/1234"
RUN_SHA = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
).strip()
DB_TREE = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD:third_party/tensor-ad-oracles"], text=True
).strip()


class OracleNightlyReporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="oracle-nightly-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        stub = self.tmp / "gh"
        stub.write_text(STUB_GH, encoding="utf-8")
        stub.chmod(0o755)
        self.calls_path = self.tmp / "gh-calls.txt"
        self.labels_state = self.tmp / "labels.txt"

    def calls(self) -> list[list[str]]:
        if not self.calls_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.calls_path.read_text(encoding="utf-8").splitlines()
        ]

    def calls_starting_with(self, *prefix: str) -> list[list[str]]:
        return [call for call in self.calls() if call[: len(prefix)] == list(prefix)]

    def run_nightly(
        self,
        run: str,
        issues: str,
        log: str | None = None,
        run_id: str = "1234",
        existing_labels: str = LABEL,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=self.env(run, issues, log, run_id, existing_labels),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"reporter exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result

    def env(
        self,
        run: str,
        issues: str,
        log: str | None = None,
        run_id: str = "1234",
        existing_labels: str = LABEL,
        gh_fail_lookup: bool = False,
    ) -> dict[str, str]:
        fixture = RUNS[run]
        log_name = fixture["log"] if log is None else log
        self.labels_state.write_text(
            "".join(f"{name}\n" for name in existing_labels.split(",") if name), encoding="utf-8"
        )
        return {
            "PATH": f"{self.tmp}{os.pathsep}{os.environ['PATH']}",
            "HOME": os.environ.get("HOME", str(self.tmp)),
            "GH_REPO": "tensor4all/tenferro-rs",
            "GH_CALLS": str(self.calls_path),
            "ISSUES_FIXTURE": str(FIXTURES / issues),
            "LABELS_STATE": str(self.labels_state),
            "GH_FAIL_LOOKUP": "1" if gh_fail_lookup else "0",
            "GITHUB_RUN_ID": run_id,
            "RUN_URL": RUN_URL,
            "HEAD_SHA": RUN_SHA,
            "RECENT_RESULT": fixture["RECENT_RESULT"],
            "REPLAY_RESULT": fixture["REPLAY_RESULT"],
            "RUN_ORACLE": fixture["RUN_ORACLE"],
            "ORACLE_LOG": "" if log_name is None else str(FIXTURES / log_name),
        }

    def body_of(self, *prefix: str) -> str:
        matches = self.calls_starting_with(*prefix)
        self.assertEqual(len(matches), 1, f"expected one {prefix} call, got {self.calls()}")
        call = matches[0]
        self.assertIn("--body", call, f"{prefix} was called without a body: {call}")
        return call[call.index("--body") + 1]

    def test_a_failure_issue_lists_the_failing_case_id(self) -> None:
        self.run_nightly("failed_replay", "issues-none.json")

        body = self.body_of("issue", "create")
        self.assertIn("solve_f64_identity_004", body)
        self.assertIn(RUN_URL, body)
        self.assertIn(RUN_SHA, body)
        self.assertIn("Root-facade oracle replay", body)

    def test_a_failure_issue_lists_a_complex128_case_id(self) -> None:
        # The database publishes 2137 c128 records; a pattern that stops at c64
        # would silently drop them from the report.
        self.run_nightly("failed_replay_complex", "issues-none.json")

        self.assertIn("incremental_householder_qr_c128_append_qr_001", self.body_of("issue", "create"))

    def test_a_failure_issue_reports_the_oracle_db_pin(self) -> None:
        self.run_nightly("failed_replay", "issues-none.json")

        body = self.body_of("issue", "create")
        # The vendored database has no upstream revision pin recorded in the
        # repository; the body must say so instead of inventing one.
        self.assertIn("no upstream revision pin is recorded in this repository", body)
        self.assertIn("third_party/tensor-ad-oracles", body)
        self.assertIn("torch==2.10.0", body)
        self.assertIn("third_party/tensor-ad-oracles/pyproject.toml:", body)
        self.assertIn(DB_TREE, body)

    def test_a_failure_without_a_case_id_still_opens_an_issue(self) -> None:
        self.run_nightly("failed_build_without_case_ids", "issues-none.json")

        body = self.body_of("issue", "create")
        self.assertIn("No case ID could be extracted", body)
        self.assertIn("Root-facade oracle replay", body)

    def test_a_failure_without_a_run_id_still_opens_an_issue(self) -> None:
        # A missing API input must not fail the notify job, which would leave
        # nobody notified at all.
        self.run_nightly("failed_replay", "issues-none.json", run_id="")

        self.assertIn("run id unavailable", self.body_of("issue", "create"))

    def test_a_missing_log_still_opens_an_issue(self) -> None:
        self.run_nightly("failed_replay", "issues-none.json", log="does-not-exist.txt")

        self.assertIn("No case ID could be extracted", self.body_of("issue", "create"))

    def test_a_new_issue_is_labelled(self) -> None:
        self.run_nightly("failed_replay", "issues-none.json", existing_labels="")

        created = self.calls_starting_with("issue", "create")
        self.assertEqual(len(created), 1, self.calls())
        self.assertIn("--label", created[0])
        self.assertEqual(created[0][created[0].index("--label") + 1], LABEL)

    def test_a_repeat_failure_comments_on_the_labelled_issue(self) -> None:
        self.run_nightly("failed_replay", "issues-labeled.json")

        self.assertEqual(len(self.calls_starting_with("issue", "comment", "17")), 1, self.calls())
        self.assertFalse(self.calls_starting_with("issue", "create"), self.calls())

    def test_a_repeat_failure_finds_the_issue_that_was_renamed_under_the_label(self) -> None:
        # Label-first: the title was edited, so only the label can find it. Without
        # the labelled lookup this would open a second issue.
        self.run_nightly("failed_replay", "issues-labeled-renamed.json")

        self.assertEqual(len(self.calls_starting_with("issue", "comment", "31")), 1, self.calls())
        self.assertFalse(self.calls_starting_with("issue", "create"), self.calls())

    def test_a_repeat_failure_prefers_the_exact_title_over_the_first_labelled_issue(self) -> None:
        self.run_nightly("failed_replay", "issues-two-labeled.json")

        calls = self.calls()
        self.assertEqual(len(self.calls_starting_with("issue", "comment", "41")), 1, calls)
        self.assertFalse(self.calls_starting_with("issue", "comment", "40"), calls)

    def test_a_repeat_failure_finds_an_issue_that_has_no_label_yet(self) -> None:
        # The first failure ever: no label in the repository, so the labelled
        # lookup fails and the server-side title search is the only path. The
        # near-miss title comes first and survives that search, so only an
        # exact-title filter picks the right issue.
        self.run_nightly("failed_replay", "issues-title-only.json", existing_labels="")

        calls = self.calls()
        searches = [
            call for call in calls if call[:2] == ["issue", "list"] and "--search" in call
        ]
        self.assertTrue(searches, calls)
        self.assertIn("in:title", searches[0][searches[0].index("--search") + 1])
        self.assertEqual(len(self.calls_starting_with("issue", "comment", "23")), 1, calls)
        self.assertFalse(self.calls_starting_with("issue", "comment", "24"), calls)
        self.assertEqual(len(self.calls_starting_with("issue", "edit", "23")), 1, calls)
        self.assertEqual(self.calls_starting_with("issue", "edit", "23")[0][-1], LABEL)

    def test_a_failed_lookup_does_not_open_a_second_issue(self) -> None:
        # A transient API failure must be visible rather than silently becoming
        # "no tracker" and filing a duplicate under a renamed title.
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=self.env("failed_replay", "issues-labeled-renamed.json", gh_fail_lookup=True),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("labelled issue lookup failed", result.stderr)
        self.assertFalse(self.calls_starting_with("issue", "create"), self.calls())

    def test_an_ambiguous_labelled_set_is_not_closed_on_recovery(self) -> None:
        # Two labelled issues with edited titles: neither is provably the failure
        # issue, so recovery must touch neither.
        self.run_nightly("successful_replay", "issues-two-renamed.json")

        calls = self.calls()
        self.assertFalse(self.calls_starting_with("issue", "comment"), calls)
        self.assertFalse(self.calls_starting_with("issue", "close"), calls)

    def test_the_label_is_created_before_it_is_added_to_an_existing_issue(self) -> None:
        self.run_nightly("failed_replay", "issues-title-only.json", existing_labels="")

        calls = self.calls()
        labels = [index for index, call in enumerate(calls) if call[:2] == ["label", "create"]]
        edits = [index for index, call in enumerate(calls) if call[:2] == ["issue", "edit"]]
        self.assertTrue(labels and edits, calls)
        self.assertLess(labels[0], edits[0], calls)

    def test_a_skipped_run_closes_nothing(self) -> None:
        result = self.run_nightly("skipped_by_gate", "issues-labeled.json")

        self.assertFalse(
            self.calls_starting_with("issue", "close"),
            f"a run_oracle=false run must not close the issue, calls were {self.calls()}",
        )
        self.assertFalse(self.calls_starting_with("issue", "comment"), self.calls())
        self.assertIn("run_oracle=false", result.stdout)

    def test_a_cancelled_run_closes_nothing(self) -> None:
        self.run_nightly("cancelled", "issues-labeled.json")

        self.assertFalse(
            self.calls_starting_with("issue", "close"),
            f"a cancelled run must not close the issue, calls were {self.calls()}",
        )

    def test_a_failure_does_not_close_the_issue(self) -> None:
        self.run_nightly("failed_replay", "issues-labeled.json")

        self.assertFalse(
            self.calls_starting_with("issue", "close"),
            f"a failing run must not close the issue, calls were {self.calls()}",
        )

    def test_a_successful_replay_closes_the_issue_with_provenance(self) -> None:
        self.run_nightly("successful_replay", "issues-labeled.json")

        calls = self.calls()
        comment = calls.index(self.calls_starting_with("issue", "comment", "17")[0])
        close = calls.index(self.calls_starting_with("issue", "close", "17")[0])
        self.assertLess(comment, close, "the provenance comment must precede the close")
        body = self.body_of("issue", "comment", "17")
        self.assertIn(RUN_URL, body)
        self.assertIn(RUN_SHA, body)
        self.assertIn("no upstream revision pin is recorded in this repository", body)
        self.assertIn(DB_TREE, body)
        self.assertIn("torch==2.10.0", body)

    def test_a_successful_replay_without_an_open_issue_is_a_no_op(self) -> None:
        self.run_nightly("successful_replay", "issues-none.json")

        self.assertFalse(
            self.calls_starting_with("issue", "close") + self.calls_starting_with("issue", "create"),
            self.calls(),
        )


if __name__ == "__main__":
    unittest.main()
