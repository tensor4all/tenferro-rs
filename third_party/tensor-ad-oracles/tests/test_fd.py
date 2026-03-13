import unittest

from generators import fd


class ComputeStepTests(unittest.TestCase):
    def test_compute_step_scales_with_input_norm(self) -> None:
        self.assertAlmostEqual(fd.compute_step("float32", input_norm=0.5), 1e-3)
        self.assertAlmostEqual(fd.compute_step("float64", input_norm=0.5), 1e-6)
        self.assertAlmostEqual(fd.compute_step("complex64", input_norm=2.0), 2e-4)
        self.assertAlmostEqual(fd.compute_step("float64", input_norm=3.0), 3e-6)
        self.assertAlmostEqual(fd.compute_step("complex128", input_norm=4.0), 4e-7)


if __name__ == "__main__":
    unittest.main()
