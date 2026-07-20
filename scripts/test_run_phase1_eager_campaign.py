#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("run_phase1_eager_campaign.py")
SPEC = importlib.util.spec_from_file_location("phase1_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class RunnerProtocolTests(unittest.TestCase):
    def test_canonical_matrix_contains_all_28_cases(self):
        self.assertEqual(len(runner.CANONICAL_CASES), 28)
        self.assertEqual(
            runner.CANONICAL_CASES["lazy_neg_1"],
            "eager_dispatch_baseline/lazy/neg_f64/1",
        )
        self.assertEqual(
            runner.CANONICAL_CASES["materialized_dot_2"],
            "eager_dispatch_baseline/materialized/dot_general_f64/2",
        )

    def test_criterion_directory_combines_group_and_mode(self):
        path = runner.criterion_directory(
            pathlib.Path("target/criterion"),
            "eager_dispatch_baseline/lazy/reduce_sum_f64/8",
        )
        self.assertEqual(
            path,
            pathlib.Path("target/criterion/eager_dispatch_baseline_lazy/reduce_sum_f64/8"),
        )

    def test_cpu_list_parser_and_formatter_round_trip(self):
        cpus = runner.parse_cpu_list("0-3,8,10-11")
        self.assertEqual(cpus, {0, 1, 2, 3, 8, 10, 11})
        self.assertEqual(runner.format_cpu_list(cpus), "0-3,8,10-11")

    def test_pair_two_target_order_is_candidate_then_baseline(self):
        identities = runner.run_identities("B/A")
        self.assertEqual(
            identities,
            ["candidate", "candidate", "baseline", "candidate"],
        )

    def test_benchmark_command_uses_one_exact_filter_and_named_baseline(self):
        command = runner.benchmark_command(
            pathlib.Path("/tmp/candidate"),
            "eager_dispatch_baseline/lazy/neg_f64/1",
            "--save-baseline",
            "phase1-sentinel-c1-p1-a1",
        )
        self.assertEqual(
            command,
            [
                "/tmp/candidate",
                "--bench",
                "eager_dispatch_baseline/lazy/neg_f64/1",
                "--save-baseline",
                "phase1-sentinel-c1-p1-a1",
                "--noplot",
            ],
        )


if __name__ == "__main__":
    unittest.main()
