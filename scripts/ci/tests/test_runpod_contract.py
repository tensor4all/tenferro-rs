import unittest

from scripts.ci.runpod_contract import (
    ContractError,
    configured_gpu_tiers,
    extract_gpu_type_ids,
    resolve_local_ref,
    validate_gpu_type_ids,
)


SCHEMA = {
    "paths": {
        "/pods": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/CreatePod"
                            }
                        }
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "CreatePod": {
                "type": "object",
                "properties": {
                    "gpuTypeIds": {
                        "type": "array",
                        "items": {
                            "enum": [
                                "NVIDIA A40",
                                "NVIDIA GeForce RTX 4090",
                            ]
                        },
                    }
                },
            }
        }
    },
}


class RunPodContractTests(unittest.TestCase):
    def test_configured_gpu_tiers_preserve_order(self) -> None:
        tiers = configured_gpu_tiers(
            {
                "gpu_tiers": [
                    {
                        "name": "cheap",
                        "gpu_type_ids": ["NVIDIA A40"],
                    },
                    {
                        "name": "premium",
                        "gpu_type_ids": ["NVIDIA L40S"],
                    },
                ]
            }
        )
        self.assertEqual(
            tiers,
            [
                ("cheap", ("NVIDIA A40",)),
                ("premium", ("NVIDIA L40S",)),
            ],
        )

    def test_configured_gpu_tiers_reject_duplicates(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate GPU ID"):
            configured_gpu_tiers(
                {
                    "gpu_tiers": [
                        {
                            "name": "cheap",
                            "gpu_type_ids": ["NVIDIA A40"],
                        },
                        {
                            "name": "premium",
                            "gpu_type_ids": ["NVIDIA A40"],
                        },
                    ]
                }
            )

    def test_extract_gpu_enum_follows_local_ref(self) -> None:
        self.assertEqual(
            extract_gpu_type_ids(SCHEMA),
            frozenset({"NVIDIA A40", "NVIDIA GeForce RTX 4090"}),
        )

    def test_validate_reports_every_invalid_configured_id(self) -> None:
        with self.assertRaisesRegex(
            ContractError, "Tesla T4.*Unknown GPU"
        ):
            validate_gpu_type_ids(SCHEMA, ["Tesla T4", "Unknown GPU"])

    def test_missing_post_schema_is_a_hard_error(self) -> None:
        with self.assertRaisesRegex(ContractError, "POST /pods"):
            extract_gpu_type_ids({"paths": {}})

    def test_non_local_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "non-local"):
            resolve_local_ref(SCHEMA, "https://example.test/CreatePod")

    def test_non_string_enum_member_is_rejected(self) -> None:
        schema = {
            **SCHEMA,
            "components": {
                "schemas": {
                    "CreatePod": {
                        "properties": {
                            "gpuTypeIds": {
                                "items": {"enum": ["NVIDIA A40", 7]}
                            }
                        }
                    }
                }
            },
        }
        with self.assertRaisesRegex(ContractError, "string GPU IDs"):
            extract_gpu_type_ids(schema)


if __name__ == "__main__":
    unittest.main()
