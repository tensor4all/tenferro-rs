import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.ci.runpod_cost_history import collect, log_lines, parse_attempt, summarize

REF = "a" * 40
RUN = {"id": 123, "run_attempt": 1, "html_url": "https://example.invalid/run/123",
       "event": "workflow_run", "head_sha": "b" * 40, "conclusion": "success",
       "status": "completed", "created_at": "2026-10-09T00:00:00Z"}


def archive(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        for name, content in files.items():
            zipped.writestr(name, content)
    return output.getvalue()


def evidence(pod="first", accepted=True):
    return [("2026-10-09T00:00:00Z", f"Pinned PR #7 merge ref to {REF}."),
            ("2026-10-09T00:00:01Z", f"Created pod {pod}: GPU A40 at $0.60/hr; waiting for the smoke proof and runner."),
            ("2026-10-09T00:00:11Z", f"Runner runpod-123-1-c1 online: pod {pod} passed the CUDA smoke proof in 10s (GPU A40, $0.60/hr)." if accepted else "rejected"),
            ("2026-10-09T00:01:01Z", f"Deleted RunPod pod: {pod}")]


class CostHistoryTests(unittest.TestCase):
    def test_echoed_source_and_per_step_copies_do_not_count(self):
        actual = "\n".join(f"{time} {line}" for time, line in evidence())
        echoed = '2026-10-09T00:00:00Z \x1b[36;1mecho "Created pod fake: GPU A40 at $0.60/hr;"\x1b[0m'
        rows = log_lines(archive({"0_Start RunPod org runner.txt": actual + "\n" + echoed,
                                  "Start RunPod org runner/system.txt": actual}))
        parsed = parse_attempt(RUN, rows, None)
        self.assertEqual(len(parsed["pods"]), 1)
        self.assertEqual(parsed["pods"][0]["paid_seconds"], 60)

    def test_rejected_pods_are_costed_once_and_not_counted_as_successes(self):
        lines = evidence("rejected", accepted=False) + evidence("accepted")
        result = parse_attempt(RUN, sorted(lines), None)
        self.assertEqual(len(result["pods"]), 2)
        self.assertEqual(sum(p["estimated_gpu_cost"] for p in result["pods"]), 0.02)
        self.assertEqual(sum(p["accepted"] for p in result["pods"]), 1)
        self.assertEqual(result["pods"][1]["gpu_job_conclusion"], "not_run")

    def test_stage_artifact_replaces_log_estimate_and_keeps_gpu_on_failed_startup(self):
        cost = {"pod_id": "first", "gpu_type_id": "", "paid_seconds": 63,
                "estimated_gpu_cost": 0.0105, "price_per_hour": 0.6, "stages": {},
                "gpu_job_conclusion": "not_run", "tested_ref": REF}
        result = parse_attempt({**RUN, "conclusion": "cancelled"}, evidence(accepted=False), cost)
        self.assertEqual(len(result["pods"]), 1)
        self.assertEqual(result["pods"][0]["gpu_type_id"], "A40")
        self.assertEqual(result["pods"][0]["paid_seconds"], 63)
        self.assertFalse(result["pods"][0]["accepted"])

    def test_missing_deletion_keeps_cost_unknown(self):
        result = parse_attempt(RUN, evidence()[:-1], None)
        self.assertIsNone(result["pods"][0]["estimated_gpu_cost"])

    def test_unverifiable_assignment_still_counts_as_a_paid_pod(self):
        lines = [("2026-10-09T00:00:00Z", "Candidate fallback created pod unknown with an unverifiable GPU assignment: no GPU id"),
                 ("2026-10-09T00:01:00Z", "Deleted pod unknown before any test setup.")]
        result = parse_attempt(RUN, lines, None)
        self.assertEqual(result["pods"][0]["paid_seconds"], 60)
        self.assertIsNone(result["pods"][0]["estimated_gpu_cost"])
        self.assertFalse(result["pods"][0]["accepted"])

    def test_watchdog_deletion_record_preserves_interrupted_cost(self):
        cost = {"pod_id": "first", "gpu_type_id": "A40", "paid_seconds": 63,
                "estimated_gpu_cost": 0.0105, "price_per_hour": 0.6, "stages": {},
                "gpu_job_conclusion": "interrupted", "tested_ref": REF}
        lines = evidence()[:-1] + [("2026-10-09T00:01:04Z", "RunPod lifecycle cost: " + json.dumps(cost))]
        parsed = parse_attempt({**RUN, "conclusion": "cancelled"}, lines, None)
        self.assertEqual(parsed["pods"][0]["estimated_gpu_cost"], 0.0105)
        self.assertEqual(parsed["pods"][0]["gpu_job_conclusion"], "interrupted")

    def test_summary_separates_manual_repeats_and_unknown_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = [{**RUN, "id": i, "event": "workflow_run" if i < 3 else "workflow_dispatch"}
                    for i in range(1, 6)]
            (root / "runs.json").write_text(json.dumps(runs))
            for run in runs:
                dest = root / str(run["id"]) / "1"
                dest.mkdir(parents=True)
                (dest / "run.json").write_text(json.dumps(run))
                if run["id"] != 5:
                    lines = "\n".join(f"{time} {line}" for time, line in evidence(f"pod{run['id']}"))
                    (dest / "logs.zip").write_bytes(archive({"0_Start RunPod.txt": lines}))
            result = summarize(root)
            self.assertEqual(len(result["same_ref_repeats"]), 2)
            self.assertEqual({r["event"] for r in result["same_ref_repeats"]}, {"workflow_run", "workflow_dispatch"})
            self.assertEqual(result["by_gpu"]["A40"]["successful_pods"], 4)
            self.assertEqual(len(result["errors"]), 1)

    def test_collect_fetches_each_attempt_and_only_its_small_cost_artifact(self):
        calls = []
        def api(path, pages=False):
            calls.append(path)
            if "/workflows/" in path:
                return json.dumps([{"total_count": 1, "workflow_runs": [{**RUN, "run_attempt": 2}]}]).encode()
            if path.endswith("/attempts/1"):
                return json.dumps({**RUN, "conclusion": "failure"}).encode()
            if "/jobs?" in path:
                return b'[{"jobs":[]}]'
            if "/logs" in path:
                return archive({})
            if "/artifacts?" in path:
                return json.dumps([{"artifacts": [
                    {"id": i, "name": f"runpod-stage-cost-123-{i}", "expired": False} for i in (1, 2)] +
                    [{"id": 99, "name": "giant-runtime", "expired": False}]}]).encode()
            if path.endswith("/zip"):
                return archive({"runpod-stage-cost.json": "{}"})
            self.fail(path)
        with tempfile.TemporaryDirectory() as folder, patch("scripts.ci.runpod_cost_history.gh_api", side_effect=api):
            collect("owner/repo", "2026-10-08", "2026-10-09", Path(folder))
            self.assertTrue((Path(folder) / "123/1/cost.zip").exists())
            self.assertTrue((Path(folder) / "123/2/cost.zip").exists())
            self.assertFalse(any("/99/zip" in call for call in calls))

    def test_collection_rejects_truncated_github_window(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.ci.runpod_cost_history.gh_api",
                return_value=b'[{"total_count":1001,"workflow_runs":[]}]'):
            with self.assertRaisesRegex(ValueError, "shorter date"):
                collect("owner/repo", "2026-10-08", "2026-10-09", Path(folder))
