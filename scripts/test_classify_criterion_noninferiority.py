#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("classify_criterion_noninferiority.py")
SPEC = importlib.util.spec_from_file_location("criterion_classifier", SCRIPT)
classifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(classifier)


THREAD_ENVIRONMENT = {
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def write_estimate(path, lower=-0.01, upper=0.02, point=0.005):
    path.write_text(
        json.dumps(
            {
                "mean": {
                    "confidence_interval": {
                        "lower_bound": lower,
                        "upper_bound": upper,
                    },
                    "point_estimate": point,
                }
            }
        ),
        encoding="utf-8",
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(role, binary="candidate", selected_cpu=3, load=0.10):
    return {
        "role": role,
        "binary": binary,
        "binary_sha256": ("b" if binary == "baseline" else "c") * 64,
        "completed": True,
        "exit_status": 0,
        "monitor_violations": [],
        "observed_affinity": str(selected_cpu),
        "normalized_load_start": load,
        "normalized_load_end": load,
    }


def make_campaign(root):
    orders = ["A/B", "B/A", "A/B"]
    cases = {}
    for case, benchmark in classifier.CANONICAL_CASES.items():
        pair_entries = {}
        for pair, order in enumerate(orders, start=1):
            identities = (
                ["candidate", "baseline", "candidate", "candidate"]
                if order == "A/B"
                else ["candidate", "candidate", "baseline", "candidate"]
            )
            pair_dir = root / case / f"pair{pair}"
            pair_dir.mkdir(parents=True)
            change = pair_dir / "change-estimates.json"
            sentinel = pair_dir / "sentinel-change-estimates.json"
            write_estimate(change)
            write_estimate(sentinel)
            validity = {
                "protocol_version": 1,
                "case": case,
                "pair": pair,
                "order": order,
                "selected_cpu": 3,
                "allowed_cpu_count": 8,
                "valid": True,
                "runs": [
                    make_run(role, binary)
                    for role, binary in zip(classifier.RUN_ROLES, identities)
                ],
                "artifacts": {
                    "change-estimates.json": {"sha256": sha256(change)},
                    "sentinel-change-estimates.json": {"sha256": sha256(sentinel)},
                },
            }
            validity_path = pair_dir / "validity.json"
            validity_path.write_text(json.dumps(validity), encoding="utf-8")
            pair_entries[str(pair)] = {
                "order": order,
                "validity": f"{case}/pair{pair}/validity.json",
                "validity_sha256": sha256(validity_path),
            }
        cases[case] = {"benchmark": benchmark, "pairs": pair_entries}
    campaign = {
        "protocol_version": 1,
        "lock_sha256": "a" * 64,
        "binaries": {
            "baseline": {"sha256": "b" * 64},
            "candidate": {"sha256": "c" * 64},
        },
        "selected_cpu": 3,
        "allowed_cpus": "0-7",
        "allowed_cpu_count": 8,
        "normalized_load_limit": 0.25,
        "thread_environment": THREAD_ENVIRONMENT,
        "orders": orders,
        "criterion": {
            "warm_up_seconds": 2,
            "measurement_seconds": 5,
            "sample_size": 100,
            "confidence_level": 0.95,
        },
        "completed_at": "2026-07-20T00:00:00+00:00",
        "cases": cases,
    }
    (root / "campaign.json").write_text(json.dumps(campaign), encoding="utf-8")
    return campaign


def rewrite_pair(root, campaign, pair, validity):
    path = root / "lazy_neg_1" / f"pair{pair}" / "validity.json"
    path.write_text(json.dumps(validity), encoding="utf-8")
    campaign["cases"]["lazy_neg_1"]["pairs"][str(pair)]["validity_sha256"] = sha256(
        path
    )
    (root / "campaign.json").write_text(json.dumps(campaign), encoding="utf-8")


class ClassificationBoundaryTests(unittest.TestCase):
    def test_pass_requires_every_upper_endpoint_at_or_below_threshold(self):
        intervals = [(-0.01, 0.05), (0.00, 0.049), (0.01, 0.05)]
        self.assertEqual(classifier.classify(intervals), "PASS")

    def test_one_upper_endpoint_above_threshold_is_not_pass(self):
        intervals = [(-0.01, 0.05), (0.00, 0.050001), (0.01, 0.05)]
        self.assertEqual(classifier.classify(intervals), "INCONCLUSIVE")

    def test_fail_requires_at_least_two_lower_endpoints_strictly_above_threshold(self):
        intervals = [(0.050001, 0.06), (0.050001, 0.07), (-0.01, 0.01)]
        self.assertEqual(classifier.classify(intervals), "FAIL")

    def test_lower_endpoint_equal_to_threshold_does_not_count_as_fail(self):
        intervals = [(0.05, 0.06), (0.050001, 0.07), (-0.01, 0.01)]
        self.assertEqual(classifier.classify(intervals), "INCONCLUSIVE")

    def test_b_over_a_interval_is_inverted_to_candidate_over_baseline(self):
        lower, upper, point = classifier.invert_interval(-0.10, -0.06, -0.08)
        self.assertAlmostEqual(lower, 1.0 / 0.94 - 1.0)
        self.assertAlmostEqual(upper, 1.0 / 0.90 - 1.0)
        self.assertAlmostEqual(point, 1.0 / 0.92 - 1.0)

    def test_sentinel_is_valid_when_interval_touches_band_boundaries(self):
        self.assertFalse(classifier.sentinel_breached(0.05, 0.08))
        self.assertFalse(classifier.sentinel_breached(-0.08, -0.05))

    def test_sentinel_is_invalid_for_wholly_outside_interval_in_either_direction(self):
        self.assertTrue(classifier.sentinel_breached(0.050001, 0.08))
        self.assertTrue(classifier.sentinel_breached(-0.08, -0.050001))


class CampaignValidityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tempdir.name)
        self.campaign = make_campaign(self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def validity(self, pair=1):
        path = self.root / "lazy_neg_1" / f"pair{pair}" / "validity.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_complete_manifest_is_classified(self):
        cases = classifier.load_validated_campaign(self.root)
        self.assertEqual(len(cases), 28)
        self.assertTrue(all(case[2] == "PASS" for case in cases))

    def test_missing_campaign_manifest_is_rejected(self):
        (self.root / "campaign.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "campaign.json"):
            classifier.load_validated_campaign(self.root)

    def test_unfinished_campaign_is_rejected(self):
        del self.campaign["completed_at"]
        (self.root / "campaign.json").write_text(
            json.dumps(self.campaign), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "not marked complete"):
            classifier.load_validated_campaign(self.root)

    def test_inconsistent_allowed_cpu_inventory_is_rejected(self):
        self.campaign["allowed_cpus"] = "0-6"
        (self.root / "campaign.json").write_text(
            json.dumps(self.campaign), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "allowed CPU inventory"):
            classifier.load_validated_campaign(self.root)

    def test_missing_sentinel_artifact_is_rejected(self):
        sentinel = self.root / "lazy_neg_1/pair1/sentinel-change-estimates.json"
        sentinel.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "sentinel-change-estimates"):
            classifier.load_validated_campaign(self.root)

    def test_breached_sentinel_is_rejected_even_when_hashes_are_consistent(self):
        pair_dir = self.root / "lazy_neg_1/pair1"
        sentinel = pair_dir / "sentinel-change-estimates.json"
        write_estimate(sentinel, lower=0.051, upper=0.08, point=0.06)
        validity = self.validity()
        validity["artifacts"]["sentinel-change-estimates.json"]["sha256"] = sha256(
            sentinel
        )
        rewrite_pair(self.root, self.campaign, 1, validity)
        with self.assertRaisesRegex(ValueError, "sentinel.*breach"):
            classifier.load_validated_campaign(self.root)

    def test_incomplete_four_run_record_is_rejected(self):
        validity = self.validity()
        validity["runs"].pop()
        rewrite_pair(self.root, self.campaign, 1, validity)
        with self.assertRaisesRegex(ValueError, "four runs"):
            classifier.load_validated_campaign(self.root)

    def test_monitor_violation_is_rejected(self):
        validity = self.validity()
        validity["runs"][1]["monitor_violations"] = ["cargo overlap"]
        rewrite_pair(self.root, self.campaign, 1, validity)
        with self.assertRaisesRegex(ValueError, "monitor violation"):
            classifier.load_validated_campaign(self.root)

    def test_affinity_mismatch_is_rejected(self):
        validity = self.validity()
        validity["runs"][2]["observed_affinity"] = "3-4"
        rewrite_pair(self.root, self.campaign, 1, validity)
        with self.assertRaisesRegex(ValueError, "affinity"):
            classifier.load_validated_campaign(self.root)

    def test_binary_identity_mismatch_is_rejected(self):
        validity = self.validity()
        validity["runs"][1]["binary"] = "candidate"
        validity["runs"][1]["binary_sha256"] = "c" * 64
        rewrite_pair(self.root, self.campaign, 1, validity)
        with self.assertRaisesRegex(ValueError, "binary identity"):
            classifier.load_validated_campaign(self.root)

    def test_endpoint_normalized_load_above_limit_is_rejected(self):
        validity = self.validity()
        validity["runs"][0]["normalized_load_end"] = 0.250001
        rewrite_pair(self.root, self.campaign, 1, validity)
        with self.assertRaisesRegex(ValueError, "normalized load"):
            classifier.load_validated_campaign(self.root)


if __name__ == "__main__":
    unittest.main()
