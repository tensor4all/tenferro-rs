"""Focused tests for the oracle nightly failure reporter.

The reporter is ``scripts/ci/report_oracle_nightly.sh``; the nightly workflow
only calls it. Each test shells out to the real script with a stub ``gh`` first
on ``PATH``, so the issue body, the tracker selection and the recovery decision
are exercised the way the nightly runs them. The stub models the parts of the
API the safety properties depend on: open-only listings, ``--limit`` truncation,
per-issue bodies and comments, label existence, and a selectable API failure.
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
PAGE_LIMIT = 1000
TITLE = "oracle-replay-nightly is failing"

STUB_GH = """#!{python}
import json, os, re, sys

args = sys.argv[1:]
calls_file = os.environ["GH_CALLS"]
labels_file = os.environ["LABELS_STATE"]
label = "{label}"


def labels():
    if not os.path.exists(labels_file):
        return set()
    return set(open(labels_file, encoding="utf-8").read().split())


def sub(text):
    return (
        text.replace("__TRACKER_SHA__", os.environ.get("TRACKER_SHA", ""))
        .replace("__TRACKER_FIRST_SHA__", os.environ.get("TRACKER_FIRST_SHA", ""))
    )


def issues():
    if os.environ.get("SEARCH_OVERFLOW") == "1":
        # Every page entry is a near miss, so only the page-size handling can
        # stop the reporter from creating a duplicate.
        return [
            {{"number": 900 + i, "title": "oracle-replay-nightly is failing on shard %d" % i,
              "labels": [], "body": "", "comments": []}}
            for i in range({page_limit})
        ]
    found = json.loads(open(os.environ["ISSUES_FIXTURE"], encoding="utf-8").read())
    for issue in found:
        issue["body"] = sub(issue.get("body", ""))
        issue["comments"] = [
            {{"body": sub(comment["body"])}} for comment in issue.get("comments", [])
        ]
    return found


def find_issue(number):
    for issue in issues():
        if issue["number"] == number:
            return issue
    sys.stderr.write("no issue %s\\n" % number)
    sys.exit(1)


def limit_of(parsed):
    if "--limit" in parsed:
        return int(parsed[parsed.index("--limit") + 1])
    return 30


def record():
    with open(calls_file, "a", encoding="utf-8") as calls:
        calls.write(json.dumps(args) + "\\n")


record()

# `gh api` carries the per-issue body and the paginated comment stream.
if args[:1] == ["api"]:
    path = next(arg for arg in args if arg.startswith("repos/"))
    number = re.search(r"/issues/(\\d+)(/comments)?$", path)
    if number.group(2) and os.environ.get("GH_FAIL_COMMENTS") == "1":
        # A 502 on the comment stream must not leave the body looking like the whole
        # failure history.
        sys.stderr.write("HTTP 502: Bad Gateway\\n")
        sys.exit(1)
    issue = find_issue(int(number.group(1)))
    if number.group(2):
        for comment in issue["comments"]:
            sys.stdout.write(comment["body"] + "\\n")
    else:
        sys.stdout.write(issue["body"])
    sys.exit(0)

command = args[:2]
if command == ["label", "create"]:
    with open(labels_file, "a", encoding="utf-8") as state:
        state.write(args[2] + "\\n")
    sys.exit(0)
if command == ["run", "view"]:
    sys.stdout.write("- Root-facade oracle replay\\n")
    sys.exit(0)
if command == ["issue", "list"]:
    found = issues()
    if "--label" in args:
        if os.environ.get("GH_FAIL_LOOKUP") == "1":
            sys.stderr.write("HTTP 502: Bad Gateway\\n")
            sys.exit(1)
        wanted = args[args.index("--label") + 1]
        if wanted not in labels():
            # `gh issue list --label` fails when the label does not exist yet.
            sys.stderr.write("the label %s was not found\\n" % wanted)
            sys.exit(1)
        found = [i for i in found if wanted in [l["name"] for l in i["labels"]]]
    if "--search" in args:
        phrase = args[args.index("--search") + 1].replace(" in:title", "")
        found = [i for i in found if phrase in i["title"]]
    json.dump(found[: limit_of(args)], sys.stdout)
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
    if "--add-label" in args and not set(args[args.index("--add-label") + 1:]).issubset(labels()):
        sys.stderr.write("could not add label: not found\\n")
        sys.exit(1)
    sys.exit(0)
sys.stderr.write("stub gh: unsupported command: " + " ".join(args) + "\\n")
sys.exit(1)
""".format(python=sys.executable, label=LABEL, page_limit=PAGE_LIMIT)

# BSD grep (macOS) has no `-P`, so the reporter must not depend on it. This shim
# takes the place of `grep` on PATH and rejects the GNU-only options exactly the
# way Apple's grep does. It delegates with `command -p`, i.e. through the
# platform's default PATH, because the real grep is not at a fixed path (/bin/grep
# on Linux, /usr/bin/grep on macOS).
BSD_GREP_SHIM = """#!/usr/bin/env bash
for arg in "$@"; do
  case "${arg}" in
    -P|--perl-regexp|-oP|-Po) echo "grep: invalid option -- P" >&2; exit 2 ;;
  esac
done
real_grep="$(command -p -v grep)" || { echo "shim grep: no grep on the default PATH" >&2; exit 127; }
exec "${real_grep}" "$@"
"""

RUN_URL = "https://example.invalid/tensor4all/tenferro-rs/actions/runs/1234"


def rev(ref: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", ref], text=True).strip()


class OracleNightlyReporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.head = rev("HEAD")
        cls.db_tree = rev(f"HEAD:third_party/tensor-ad-oracles")
        # The ancestry tests need real history; a shallow checkout cannot run them.
        cls.older = rev("HEAD~2")

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="oracle-nightly-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        stub = self.tmp / "gh"
        stub.write_text(STUB_GH, encoding="utf-8")
        stub.chmod(0o755)
        bsd_grep = self.tmp / "grep"
        bsd_grep.write_text(BSD_GREP_SHIM, encoding="utf-8")
        bsd_grep.chmod(0o755)
        # Fail with the reason rather than with a confusing empty issue body when the
        # shim cannot reach a real grep on this platform.
        probe = subprocess.run(
            [str(bsd_grep), "-qxF", "probe"], input="probe\n", text=True, capture_output=True
        )
        self.assertEqual(
            probe.returncode, 0, f"the BSD-grep shim cannot delegate to a real grep: {probe.stderr}"
        )
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

    def env(
        self,
        run: str,
        issues: str,
        log: str | None = None,
        head_sha: str | None = None,
        run_id: str = "1234",
        existing_labels: str = LABEL,
        tracker_sha: str | None = None,
        tracker_first_sha: str | None = None,
        gh_fail_lookup: bool = False,
        gh_fail_comments: bool = False,
        search_overflow: bool = False,
    ) -> dict[str, str]:
        fixture = RUNS[run]
        log_name = fixture["log"] if log is None else log
        self.labels_state.write_text(
            "".join(f"{name}\n" for name in existing_labels.split(",") if name), encoding="utf-8"
        )
        head = self.head if head_sha is None else head_sha
        return {
            "PATH": f"{self.tmp}{os.pathsep}{os.environ['PATH']}",
            "HOME": os.environ.get("HOME", str(self.tmp)),
            "GH_REPO": "tensor4all/tenferro-rs",
            "GH_CALLS": str(self.calls_path),
            "ISSUES_FIXTURE": str(FIXTURES / issues),
            "LABELS_STATE": str(self.labels_state),
            "GH_FAIL_LOOKUP": "1" if gh_fail_lookup else "0",
            "GH_FAIL_COMMENTS": "1" if gh_fail_comments else "0",
            "SEARCH_OVERFLOW": "1" if search_overflow else "0",
            "TRACKER_SHA": head if tracker_sha is None else tracker_sha,
            "TRACKER_FIRST_SHA": self.older if tracker_first_sha is None else tracker_first_sha,
            "GITHUB_RUN_ID": run_id,
            "RUN_URL": RUN_URL,
            "HEAD_SHA": head,
            "RECENT_RESULT": fixture["RECENT_RESULT"],
            "REPLAY_RESULT": fixture["REPLAY_RESULT"],
            "RUN_ORACLE": fixture["RUN_ORACLE"],
            "ORACLE_LOG": "" if log_name is None else str(FIXTURES / log_name),
        }

    def invoke(self, **kwargs: str | bool | None) -> subprocess.CompletedProcess[str]:
        run = str(kwargs.pop("run"))
        issues = str(kwargs.pop("issues"))
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=self.env(run, issues, **kwargs),
            text=True,
            capture_output=True,
            check=False,
        )

    def run_nightly(self, run: str, issues: str, **kwargs: str | bool | None) -> subprocess.CompletedProcess[str]:
        result = self.invoke(run=run, issues=issues, **kwargs)
        self.assertEqual(
            result.returncode,
            0,
            f"reporter exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result

    def body_of(self, *prefix: str) -> str:
        matches = self.calls_starting_with(*prefix)
        self.assertEqual(len(matches), 1, f"expected one {prefix} call, got {self.calls()}")
        call = matches[0]
        self.assertIn("--body", call, f"{prefix} was called without a body: {call}")
        return call[call.index("--body") + 1]

    # --- the failure report -------------------------------------------------

    def test_a_failure_issue_lists_the_failing_case_id(self) -> None:
        self.run_nightly("failed_replay", "issues-none.json")

        body = self.body_of("issue", "create")
        self.assertIn("solve_f64_identity_004", body)
        self.assertIn(RUN_URL, body)
        self.assertIn(self.head, body)
        self.assertIn("Root-facade oracle replay", body)

    def test_a_failure_issue_lists_a_complex128_case_id(self) -> None:
        # The database publishes 2137 c128 records; a pattern that stops at c64
        # would silently drop them from the report.
        self.run_nightly("failed_replay_complex", "issues-none.json")

        self.assertIn(
            "incremental_householder_qr_c128_append_qr_001", self.body_of("issue", "create")
        )

    def test_case_ids_are_whole_tokens(self) -> None:
        # A longer token (`…_0012`), a `__`-prefixed op name and a path component
        # must not be reported as a different, shorter case ID. The `grep` on PATH
        # here rejects GNU-only options, so a `grep -oP` implementation cannot pass
        # on this test on any platform, macOS included.
        self.run_nightly("failed_replay_tokens", "issues-none.json")

        body = self.body_of("issue", "create")
        self.assertIn("`solve_f64_identity_005`", body)
        self.assertNotIn("solve_f64_identity_001", body)
        self.assertNotIn("radd___f64_identity_001", body)
        self.assertNotIn("cache_f64_worker_001", body)

    def test_a_failure_issue_reports_the_oracle_db_pin(self) -> None:
        self.run_nightly("failed_replay", "issues-none.json")

        body = self.body_of("issue", "create")
        # The vendored database has no upstream revision pin recorded in the
        # repository; the body must say so instead of inventing one.
        self.assertIn("no upstream revision pin is recorded in this repository", body)
        self.assertIn("third_party/tensor-ad-oracles", body)
        self.assertIn("torch==2.10.0", body)
        self.assertIn("third_party/tensor-ad-oracles/pyproject.toml:", body)
        self.assertIn(self.db_tree, body)

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

    def test_the_label_is_created_before_it_is_added_to_an_existing_issue(self) -> None:
        self.run_nightly("failed_replay", "issues-title-only.json", existing_labels="")

        calls = self.calls()
        labels = [index for index, call in enumerate(calls) if call[:2] == ["label", "create"]]
        edits = [index for index, call in enumerate(calls) if call[:2] == ["issue", "edit"]]
        self.assertTrue(labels and edits, calls)
        self.assertLess(labels[0], edits[0], calls)

    def test_an_already_labelled_tracker_is_not_relabelled(self) -> None:
        self.run_nightly("failed_replay", "issues-labeled.json")

        calls = self.calls()
        self.assertFalse(self.calls_starting_with("label", "create"), calls)
        self.assertFalse(self.calls_starting_with("issue", "edit"), calls)

    def test_an_ambiguous_labelled_set_does_not_create_a_third_issue(self) -> None:
        # Two labelled trackers with edited titles: neither is provably the
        # tracker, and the state must not turn into a third issue.
        result = self.invoke(run="failed_replay", issues="issues-two-renamed.json")

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("#51 #52", result.stderr)
        self.assertIn("refusing", result.stderr)
        self.assertFalse(self.calls_starting_with("issue", "create"), self.calls())
        self.assertFalse(self.calls_starting_with("issue", "comment"), self.calls())

    def test_a_full_title_search_page_is_refused(self) -> None:
        # Every returned issue is a near miss and the page is full, so a tracker
        # could be outside it: creating one would be a guess.
        result = self.invoke(
            run="failed_replay", issues="issues-none.json", existing_labels="", search_overflow=True
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"full page of {PAGE_LIMIT}", result.stderr)
        self.assertFalse(self.calls_starting_with("issue", "create"), self.calls())

    def test_a_failed_lookup_does_not_open_a_second_issue(self) -> None:
        # A transient API failure must be visible rather than silently becoming
        # "no tracker" and filing a duplicate under a renamed title.
        result = self.invoke(
            run="failed_replay", issues="issues-labeled-renamed.json", gh_fail_lookup=True
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("labelled issue lookup failed", result.stderr)
        self.assertFalse(self.calls_starting_with("issue", "create"), self.calls())

    # --- nothing to do ------------------------------------------------------

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

    # --- recovery -----------------------------------------------------------

    def test_a_successful_replay_closes_the_issue_with_provenance(self) -> None:
        self.run_nightly("successful_replay", "issues-labeled.json")

        calls = self.calls()
        comment = calls.index(self.calls_starting_with("issue", "comment", "17")[0])
        close = calls.index(self.calls_starting_with("issue", "close", "17")[0])
        self.assertLess(comment, close, "the provenance comment must precede the close")
        body = self.body_of("issue", "comment", "17")
        self.assertIn(RUN_URL, body)
        self.assertIn(self.head, body)
        self.assertIn("no upstream revision pin is recorded in this repository", body)
        self.assertIn(self.db_tree, body)
        self.assertIn("torch==2.10.0", body)

    def test_a_successful_replay_of_a_descendant_commit_closes_the_issue(self) -> None:
        # The tracker recorded a failure at an older commit; the recovering run
        # is further ahead, so it covers that failure.
        self.run_nightly(
            "successful_replay", "issues-labeled.json", tracker_sha=self.older
        )

        calls = self.calls()
        self.assertEqual(len(self.calls_starting_with("issue", "close", "17")), 1, calls)

    def test_a_replay_of_an_older_commit_does_not_close_a_newer_failure(self) -> None:
        # A re-run of an older main commit is not evidence about the newer tracked
        # failure, even though its ref is still refs/heads/main.
        self.run_nightly(
            "successful_replay", "issues-labeled.json", head_sha=self.older,
            tracker_sha=self.head,
        )

        calls = self.calls()
        self.assertFalse(self.calls_starting_with("issue", "close"), calls)
        body = self.body_of("issue", "comment", "17")
        self.assertIn("Recovery evidence is older than the tracked failure", body)
        self.assertIn(self.older, body)
        self.assertIn(self.head, body)

    def test_the_newest_recorded_failure_decides_recovery(self) -> None:
        # The body holds the first failure and the newest is a comment; recovery
        # must relate to the newest, not the first.
        self.run_nightly(
            "successful_replay",
            "issues-labeled-history.json",
            head_sha=self.older,
            tracker_sha=self.head,
            tracker_first_sha=self.older,
        )

        self.assertFalse(self.calls_starting_with("issue", "close"), self.calls())
        self.assertIn(
            "Recovery evidence is older than the tracked failure", self.body_of("issue", "comment", "17")
        )

    def test_a_later_failure_of_an_older_commit_does_not_reopen_recovery(self) -> None:
        # The body records the newer failure B, the newest comment re-files an older
        # commit A. Arrival order would pick A, and a successful replay of A would then
        # close the still-unresolved B.
        self.run_nightly(
            "successful_replay",
            "issues-labeled-out-of-order.json",
            head_sha=self.older,
            tracker_sha=self.head,
            tracker_first_sha=self.older,
        )

        self.assertFalse(self.calls_starting_with("issue", "close"), self.calls())
        body = self.body_of("issue", "comment", "17")
        self.assertIn("Recovery evidence is older than the tracked failure", body)
        self.assertIn(f"- Tracked failure commit: `{self.head}`", body)

    def test_a_failed_comment_read_does_not_close_the_issue(self) -> None:
        # The newest failure lives in a comment; a 502 on the comment stream must not
        # let the body's older marker stand in for the whole history, which would let
        # this successful replay close a newer, unresolved failure.
        self.run_nightly(
            "successful_replay",
            "issues-labeled-history.json",
            head_sha=self.head,
            tracker_sha=self.head,
            tracker_first_sha=self.older,
            gh_fail_comments=True,
        )

        self.assertFalse(self.calls_starting_with("issue", "close"), self.calls())
        self.assertIn(
            "cannot be related to the tracked failure", self.body_of("issue", "comment", "17")
        )

    def test_markers_that_cannot_be_ordered_do_not_close_the_issue(self) -> None:
        # An unknown commit makes "the newest failure" undecidable; guessing would
        # risk closing an unresolved failure.
        self.run_nightly("successful_replay", "issues-labeled-unknown-commit.json")

        self.assertFalse(self.calls_starting_with("issue", "close"), self.calls())
        self.assertIn(
            "cannot be related to the tracked failure", self.body_of("issue", "comment", "17")
        )

    def test_an_ambiguous_labelled_set_is_not_closed_on_recovery(self) -> None:
        result = self.invoke(run="successful_replay", issues="issues-two-renamed.json")

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("#51 #52", result.stderr)
        self.assertFalse(self.calls_starting_with("issue", "close"), self.calls())
        self.assertFalse(self.calls_starting_with("issue", "comment"), self.calls())

    def test_a_successful_replay_without_an_open_issue_is_a_no_op(self) -> None:
        self.run_nightly("successful_replay", "issues-none.json")

        self.assertFalse(
            self.calls_starting_with("issue", "close") + self.calls_starting_with("issue", "create"),
            self.calls(),
        )


if __name__ == "__main__":
    unittest.main()
