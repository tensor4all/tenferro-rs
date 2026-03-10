import unittest

from generators import probes


class ProbeTests(unittest.TestCase):
    def test_tensor_norm_handles_complex_pair_encoding(self) -> None:
        tensor = {
            "dtype": "complex128",
            "shape": [2],
            "order": "row_major",
            "data": [[3.0, 4.0], [0.0, 12.0]],
        }

        self.assertAlmostEqual(probes.tensor_norm(tensor), 13.0)

    def test_normalize_tensor_map_returns_unit_norm_tensors(self) -> None:
        mapping = {
            "a": {
                "dtype": "float64",
                "shape": [2],
                "order": "row_major",
                "data": [3.0, 4.0],
            }
        }

        normalized = probes.normalize_tensor_map(mapping)

        self.assertAlmostEqual(probes.tensor_norm(normalized["a"]), 1.0)
        self.assertEqual(normalized["a"]["shape"], [2])

    def test_normalize_tensor_allows_zero_sized_tensor(self) -> None:
        tensor = {"dtype": "float64", "shape": [0], "order": "row_major", "data": []}

        normalized = probes.normalize_tensor(tensor)

        self.assertEqual(normalized, tensor)

    def test_make_probe_record_assembles_expected_shape(self) -> None:
        direction = {
            "a": {
                "dtype": "float64",
                "shape": [1],
                "order": "row_major",
                "data": [1.0],
            }
        }
        cotangent = {
            "value": {
                "dtype": "float64",
                "shape": [1],
                "order": "row_major",
                "data": [1.0],
            }
        }

        probe = probes.make_probe_record(
            probe_id="p0",
            direction=direction,
            cotangent=cotangent,
            pytorch_jvp=cotangent,
            pytorch_vjp=direction,
            fd_step=1e-6,
            fd_jvp=cotangent,
        )

        self.assertEqual(probe["probe_id"], "p0")
        self.assertEqual(probe["fd_ref"]["method"], "central_difference")
        self.assertEqual(probe["pytorch_ref"]["vjp"], direction)

    def test_make_probe_record_includes_optional_hvp_payloads(self) -> None:
        direction = {
            "a": {
                "dtype": "float64",
                "shape": [1],
                "order": "row_major",
                "data": [1.0],
            }
        }
        cotangent = {
            "value": {
                "dtype": "float64",
                "shape": [1],
                "order": "row_major",
                "data": [1.0],
            }
        }

        probe = probes.make_probe_record(
            probe_id="p0",
            direction=direction,
            cotangent=cotangent,
            pytorch_jvp=cotangent,
            pytorch_vjp=direction,
            pytorch_hvp=direction,
            fd_step=1e-6,
            fd_jvp=cotangent,
            fd_hvp=direction,
        )

        self.assertEqual(probe["pytorch_ref"]["hvp"], direction)
        self.assertEqual(probe["fd_ref"]["hvp"], direction)


if __name__ == "__main__":
    unittest.main()
