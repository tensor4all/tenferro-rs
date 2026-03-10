import unittest

from generators import pytorch_v1, upstream_inventory


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

    def test_known_upstream_xfail_family_is_explicitly_classified(self) -> None:
        unsupported = pytorch_v1.build_unsupported_upstream_mapping_index()

        self.assertIn(("linalg.norm", "subgradients_at_zero"), unsupported)


if __name__ == "__main__":
    unittest.main()
