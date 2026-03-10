import json
import tempfile
import unittest
from pathlib import Path

from generators import pytorch_v1
from tests.test_encoding import FakeTensor


class MaterializeTests(unittest.TestCase):
    def test_case_output_path_uses_op_and_family_jsonl(self) -> None:
        spec = pytorch_v1.build_case_spec_index()[("svd", "s")]

        path = pytorch_v1.case_output_path(spec, cases_root=Path("/tmp/cases"))

        self.assertEqual(path, Path("/tmp/cases/svd/s.jsonl"))

    def test_write_case_records_writes_jsonl(self) -> None:
        spec = pytorch_v1.build_case_spec_index()[("solve", "identity")]
        record = {
            "schema_version": 1,
            "case_id": "solve_f64_identity_001",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = pytorch_v1.write_case_records(
                spec,
                [record],
                cases_root=Path(tmpdir),
            )

            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), record)

    def test_make_success_case_uses_spec_and_payload(self) -> None:
        spec = pytorch_v1.build_case_spec_index()[("solve", "identity")]
        provenance = pytorch_v1.build_provenance(
            spec,
            source_commit="deadbeef",
            seed=17,
            torch_version="2.8.0",
        )
        case = pytorch_v1.make_success_case(
            spec,
            case_id="solve_f64_identity_001",
            dtype="float64",
            inputs={"a": {"dtype": "float64", "shape": [1], "order": "row_major", "data": [1.0]}},
            comparison={
                "first_order": {"kind": "allclose", "rtol": 1e-11, "atol": 1e-12}
            },
            probes=[
                {
                    "probe_id": "p0",
                    "direction": {},
                    "cotangent": {},
                    "pytorch_ref": {"jvp": {}, "vjp": {}},
                    "fd_ref": {"method": "central_difference", "stencil_order": 2, "step": 1e-6, "jvp": {}},
                }
            ],
            provenance=provenance,
        )

        self.assertEqual(case["observable"], {"kind": "identity"})
        self.assertEqual(case["expected_behavior"], "success")
        self.assertEqual(case["family"], "identity")

    def test_make_error_case_sets_empty_probes(self) -> None:
        spec = pytorch_v1.build_case_spec_index()[("svd", "gauge_ill_defined")]
        provenance = pytorch_v1.build_provenance(
            spec,
            source_commit="deadbeef",
            seed=23,
            torch_version="2.8.0",
        )
        case = pytorch_v1.make_error_case(
            spec,
            case_id="svd_c128_gauge_ill_defined_001",
            dtype="complex128",
            inputs={"a": {"dtype": "complex128", "shape": [1], "order": "row_major", "data": [[1.0, 0.0]]}},
            reason_code="gauge_ill_defined",
            provenance=provenance,
        )

        self.assertEqual(case["expected_behavior"], "error")
        self.assertEqual(case["comparison"]["kind"], "expect_error")
        self.assertEqual(case["probes"], [])

    def test_materialize_success_case_encodes_raw_probe_payloads(self) -> None:
        spec = pytorch_v1.build_case_spec_index()[("solve", "identity")]
        provenance = pytorch_v1.build_provenance(
            spec,
            source_commit="deadbeef",
            seed=17,
            torch_version="2.8.0",
        )

        case = pytorch_v1.materialize_success_case(
            spec,
            case_id="solve_f64_identity_001",
            dtype="float64",
            raw_inputs={"a": FakeTensor([[4.0, 1.0], [1.0, 3.0]], dtype="float64")},
            comparison={
                "first_order": {"kind": "allclose", "rtol": 1e-11, "atol": 1e-12},
                "second_order": {"kind": "allclose", "rtol": 1e-8, "atol": 1e-9},
            },
            probe_id="p0",
            raw_direction={"a": FakeTensor([[1.0, 0.0], [0.0, 1.0]], dtype="float64")},
            raw_cotangent={"value": FakeTensor([[0.5], [0.25]], dtype="float64")},
            raw_pytorch_jvp={"value": FakeTensor([[0.1], [0.2]], dtype="float64")},
            raw_pytorch_vjp={"a": FakeTensor([[0.3, 0.4], [0.5, 0.6]], dtype="float64")},
            raw_pytorch_hvp={"a": FakeTensor([[0.7, 0.8], [0.9, 1.0]], dtype="float64")},
            fd_step=1e-6,
            raw_fd_jvp={"value": FakeTensor([[0.1], [0.2]], dtype="float64")},
            raw_fd_hvp={"a": FakeTensor([[0.7, 0.8], [0.9, 1.0]], dtype="float64")},
            provenance=provenance,
        )

        self.assertEqual(case["inputs"]["a"]["data"], [4.0, 1.0, 1.0, 3.0])
        self.assertEqual(case["probes"][0]["direction"]["a"]["shape"], [2, 2])
        self.assertEqual(case["probes"][0]["fd_ref"]["jvp"]["value"]["data"], [0.1, 0.2])
        self.assertEqual(case["probes"][0]["pytorch_ref"]["hvp"]["a"]["data"], [0.7, 0.8, 0.9, 1.0])
        self.assertEqual(case["probes"][0]["fd_ref"]["hvp"]["a"]["data"], [0.7, 0.8, 0.9, 1.0])
        self.assertEqual(case["observable"], {"kind": "identity"})


if __name__ == "__main__":
    unittest.main()
