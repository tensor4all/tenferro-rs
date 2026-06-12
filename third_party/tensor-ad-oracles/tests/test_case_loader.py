import unittest
from pathlib import Path

from validators import case_loader


CASES_ROOT = Path(__file__).resolve().parents[1] / "cases"


class CaseLoaderTests(unittest.TestCase):
    def test_load_case_file_reads_jsonl_records(self) -> None:
        records = case_loader.load_case_file(CASES_ROOT / "solve" / "identity.jsonl")

        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0]["op"], "solve")

    def test_iter_case_files_lists_jsonl_files(self) -> None:
        paths = case_loader.iter_case_files(CASES_ROOT)

        self.assertIn(CASES_ROOT / "solve" / "identity.jsonl", paths)
