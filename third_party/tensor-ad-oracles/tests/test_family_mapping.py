import unittest

from generators import pytorch_v1, upstream_inventory, upstream_scalar_inventory


class FamilyMappingTests(unittest.TestCase):
    def test_every_upstream_inventory_entry_is_mapped_or_explicitly_unsupported(self) -> None:
        inventory_keys = {
            (row.name, row.variant_name)
            for row in upstream_inventory.collect_ad_relevant_linalg_opinfos()
        }
        supported = set(pytorch_v1.build_supported_upstream_mapping_index())
        unsupported = set(pytorch_v1.build_unsupported_upstream_mapping_index())

        self.assertFalse(inventory_keys - supported - unsupported)

    def test_supported_mapping_preserves_spectral_family_splits(self) -> None:
        supported = pytorch_v1.build_supported_upstream_mapping_index()

        svd = supported[("linalg.svd", "")]
        self.assertEqual(
            {(spec.op, spec.family) for spec in svd},
            {
                ("svd", "u_abs"),
                ("svd", "s"),
                ("svd", "vh_abs"),
                ("svd", "uvh_product"),
            },
        )

        eigh = supported[("linalg.eigh", "")]
        self.assertEqual(
            {(spec.op, spec.family) for spec in eigh},
            {("eigh", "values_vectors_abs")},
        )

        pinv_singular = supported[("linalg.pinv", "singular")]
        self.assertEqual(
            {(spec.op, spec.family) for spec in pinv_singular},
            {("pinv_singular", "identity")},
        )
        self.assertTrue(all(spec.hvp_enabled for spec in pinv_singular))

        solve = supported[("linalg.solve", "")]
        self.assertTrue(all(spec.hvp_enabled for spec in solve))

    def test_supported_mapping_tracks_publishable_dtypes(self) -> None:
        supported = pytorch_v1.build_supported_upstream_mapping_index()

        svd = supported[("linalg.svd", "")]
        self.assertTrue(
            all("float64" in spec.supported_dtype_names for spec in svd)
        )
        self.assertTrue(
            all("complex128" in spec.supported_dtype_names for spec in svd)
        )

        eig = supported[("linalg.eig", "")]
        self.assertTrue(
            all("complex128" in spec.supported_dtype_names for spec in eig)
        )

    def test_known_upstream_xfail_family_is_explicitly_classified(self) -> None:
        unsupported = pytorch_v1.build_unsupported_upstream_mapping_index()

        self.assertIn(("linalg.norm", "subgradients_at_zero"), unsupported)

    def test_every_scalar_inventory_entry_is_mapped_or_explicitly_unsupported(self) -> None:
        inventory_keys = {
            (row.name, row.variant_name)
            for row in upstream_scalar_inventory.collect_ad_relevant_scalar_opinfos()
        }
        supported = set(pytorch_v1.build_supported_scalar_mapping_index())
        unsupported = set(pytorch_v1.build_unsupported_scalar_mapping_index())

        self.assertFalse(inventory_keys - supported - unsupported)

    def test_fd_hostile_scalar_families_are_explicitly_classified(self) -> None:
        unsupported = pytorch_v1.build_unsupported_scalar_mapping_index()

        self.assertIn(("div", "floor_rounding"), unsupported)
        self.assertIn(("div", "trunc_rounding"), unsupported)
        self.assertIn(("fmod", ""), unsupported)
        self.assertIn(("remainder", ""), unsupported)


if __name__ == "__main__":
    unittest.main()
