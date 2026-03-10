import unittest

from generators import upstream_inventory


class UpstreamInventoryTests(unittest.TestCase):
    def test_collect_ad_relevant_linalg_opinfos_lists_expected_variants(self) -> None:
        rows = upstream_inventory.collect_ad_relevant_linalg_opinfos()
        keys = {(row.name, row.variant_name) for row in rows}

        self.assertEqual(len(rows), 38)
        self.assertIn(("linalg.svd", ""), keys)
        self.assertIn(("linalg.eigh", ""), keys)
        self.assertIn(("linalg.pinv", "singular"), keys)
        self.assertIn(("linalg.lstsq", "grad_oriented"), keys)
        self.assertNotIn(("linalg.matrix_rank", ""), keys)

    def test_inventory_preserves_sample_wrapper_and_process_metadata(self) -> None:
        rows = upstream_inventory.collect_ad_relevant_linalg_opinfos()
        index = {(row.name, row.variant_name): row for row in rows}

        svd = index[("linalg.svd", "")]
        self.assertEqual(svd.sample_inputs_func_name, "sample_inputs_svd")
        self.assertEqual(svd.gradcheck_wrapper_name, None)
        self.assertEqual(svd.sample_output_process_fn_names, ("fn_S", "fn_U", "fn_UVh", "fn_Vh"))
        self.assertTrue(svd.gradcheck_fast_mode)

        eigh = index[("linalg.eigh", "")]
        self.assertEqual(eigh.sample_inputs_func_name, "sample_inputs_linalg_eigh")
        self.assertEqual(eigh.gradcheck_wrapper_name, "gradcheck_wrapper_hermitian_input")
        self.assertEqual(eigh.sample_output_process_fn_names, ("out_fn",))

        solve = index[("linalg.solve", "")]
        self.assertEqual(solve.sample_inputs_func_name, "sample_inputs_linalg_solve")
        self.assertEqual(solve.sample_output_process_fn_names, ("<lambda>",))
        self.assertTrue(solve.supports_forward_ad)
        self.assertTrue(solve.supports_fwgrad_bwgrad)

    def test_inventory_exposes_second_order_support_flags(self) -> None:
        rows = upstream_inventory.collect_ad_relevant_linalg_opinfos()
        index = {(row.name, row.variant_name): row for row in rows}

        svd = index[("linalg.svd", "")]
        self.assertTrue(svd.supports_forward_ad)
        self.assertTrue(svd.supports_fwgrad_bwgrad)

        norm_subgrad = index[("linalg.norm", "subgradients_at_zero")]
        self.assertTrue(norm_subgrad.supports_forward_ad)
        self.assertTrue(norm_subgrad.supports_fwgrad_bwgrad)

    def test_resolve_upstream_ad_tolerance_uses_gradcheck_defaults_for_float64(self) -> None:
        first_order = upstream_inventory.resolve_upstream_ad_tolerance(
            "linalg.solve",
            "",
            order="first_order",
            dtype_name="float64",
        )
        second_order = upstream_inventory.resolve_upstream_ad_tolerance(
            "linalg.solve",
            "",
            order="second_order",
            dtype_name="float64",
        )

        self.assertEqual(first_order, {"rtol": 1e-3, "atol": 1e-5})
        self.assertEqual(second_order, {"rtol": 1e-3, "atol": 1e-5})


if __name__ == "__main__":
    unittest.main()
