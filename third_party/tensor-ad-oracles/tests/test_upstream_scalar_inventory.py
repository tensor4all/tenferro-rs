import unittest

from generators import upstream_scalar_inventory


class UpstreamScalarInventoryTests(unittest.TestCase):
    def test_collect_ad_relevant_scalar_opinfos_lists_expected_variants(self) -> None:
        rows = upstream_scalar_inventory.collect_ad_relevant_scalar_opinfos()
        keys = {(row.name, row.variant_name) for row in rows}

        self.assertGreater(len(rows), 100)
        self.assertIn(("abs", ""), keys)
        self.assertIn(("add", ""), keys)
        self.assertIn(("sum", ""), keys)
        self.assertIn(("special.i0e", ""), keys)
        self.assertNotIn(("linalg.svd", ""), keys)
        self.assertNotIn(("masked.sum", ""), keys)
        self.assertNotIn(("__rand__", ""), keys)
        self.assertNotIn(("__ror__", ""), keys)
        self.assertNotIn(("__rxor__", ""), keys)

    def test_inventory_preserves_class_and_sample_metadata(self) -> None:
        rows = upstream_scalar_inventory.collect_ad_relevant_scalar_opinfos()
        index = {(row.name, row.variant_name): row for row in rows}

        abs_row = index[("abs", "")]
        self.assertEqual(abs_row.opinfo_class_name, "UnaryUfuncInfo")
        self.assertEqual(abs_row.family_class, "unary")
        self.assertEqual(abs_row.sample_inputs_func_name, "sample_inputs_elementwise_unary")
        self.assertEqual(abs_row.gradcheck_wrapper_name, None)
        self.assertEqual(abs_row.sample_output_process_fn_names, ("<lambda>",))
        self.assertTrue(abs_row.supports_forward_ad)
        self.assertTrue(abs_row.supports_fwgrad_bwgrad)

        add_row = index[("add", "")]
        self.assertEqual(add_row.opinfo_class_name, "BinaryUfuncInfo")
        self.assertEqual(add_row.family_class, "binary")
        self.assertEqual(add_row.sample_inputs_func_name, "sample_inputs_add_sub")

        sum_row = index[("sum", "")]
        self.assertEqual(sum_row.opinfo_class_name, "ReductionOpInfo")
        self.assertEqual(sum_row.family_class, "reduction")
        self.assertEqual(sum_row.sample_inputs_func_name, "sample_inputs_func")

    def test_resolve_upstream_scalar_ad_tolerance_uses_gradcheck_defaults_for_float32(self) -> None:
        first_order = upstream_scalar_inventory.resolve_upstream_scalar_ad_tolerance(
            "add",
            "",
            order="first_order",
            dtype_name="float32",
        )
        second_order = upstream_scalar_inventory.resolve_upstream_scalar_ad_tolerance(
            "add",
            "",
            order="second_order",
            dtype_name="float32",
        )

        self.assertEqual(first_order, {"rtol": 1e-3, "atol": 1e-5})
        self.assertEqual(second_order, {"rtol": 1e-3, "atol": 1e-5})


if __name__ == "__main__":
    unittest.main()
