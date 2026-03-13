import unittest

from generators import pytorch_v1
from validators.encoding import decode_tensor_map


class ScalarGenerationTests(unittest.TestCase):
    def _require_runtime(self) -> None:
        try:
            pytorch_v1.ensure_runtime_dependencies()
        except Exception as exc:
            self.skipTest(f"uv generation dependencies unavailable: {exc}")

    def test_generate_abs_identity_records_materializes_first_order_oracles(self) -> None:
        self._require_runtime()
        spec = pytorch_v1.build_scalar_case_spec_index()[("abs", "identity")]

        records = pytorch_v1._generate_success_records(spec, limit=1, seed=17)

        self.assertTrue(records)
        dtypes = {record["dtype"] for record in records}
        self.assertIn("float32", dtypes)
        self.assertIn("float64", dtypes)
        probe = records[0]["probes"][0]
        self.assertIn("jvp", probe["pytorch_ref"])
        self.assertIn("vjp", probe["pytorch_ref"])
        self.assertIn("jvp", probe["fd_ref"])

    def test_generate_add_identity_records_include_single_precision_complex(self) -> None:
        self._require_runtime()
        spec = pytorch_v1.build_scalar_case_spec_index()[("add", "identity")]

        records = pytorch_v1._generate_success_records(spec, limit=1, seed=17)

        dtypes = {record["dtype"] for record in records}
        self.assertTrue({"float32", "float64", "complex64", "complex128"} <= dtypes)

    def test_generate_sum_identity_records_persist_generic_call_kwargs(self) -> None:
        self._require_runtime()
        spec = pytorch_v1.build_scalar_case_spec_index()[("sum", "identity")]

        records = pytorch_v1._generate_success_records(spec, limit=4, seed=17)

        kwargs_records = [record for record in records if record.get("op_kwargs")]
        self.assertTrue(kwargs_records)
        self.assertIn("dim", kwargs_records[0]["op_kwargs"])
        self.assertIn("keepdim", kwargs_records[0]["op_kwargs"])

    def test_generate_conj_identity_records_publish_first_order_only_when_hvp_is_unstable(
        self,
    ) -> None:
        self._require_runtime()
        spec = pytorch_v1.build_scalar_case_spec_index()[("conj", "identity")]

        records = pytorch_v1._generate_success_records(spec, limit=1, seed=17)

        self.assertTrue(records)
        self.assertNotIn("second_order", records[0]["comparison"])
        self.assertNotIn("hvp", records[0]["probes"][0]["pytorch_ref"])
        self.assertNotIn("hvp", records[0]["probes"][0]["fd_ref"])

    def test_generate_rpow_identity_records_skip_nonfinite_samples(self) -> None:
        self._require_runtime()
        spec = pytorch_v1.build_scalar_case_spec_index()[("__rpow__", "identity")]

        records = pytorch_v1._generate_success_records(spec, limit=1, seed=17)

        self.assertTrue(records)
        probe = records[0]["probes"][0]
        jvp = decode_tensor_map(probe["pytorch_ref"]["jvp"])
        tensor = next(iter(jvp.values()))
        self.assertTrue(tensor.isfinite().all().item())


if __name__ == "__main__":
    unittest.main()
