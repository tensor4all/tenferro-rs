import unittest

from scripts.ci.runpod_contract import (
    ContractError,
    configured_cuda_versions,
    configured_gpu_tiers,
    extract_allowed_cuda_versions,
    extract_gpu_type_ids,
    resolve_local_ref,
    validate_cuda_versions,
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
                    },
                    "allowedCudaVersions": {
                        "type": "array",
                        "items": {"enum": ["13.0", "12.9", "12.8", "12.4"]},
                    },
                },
            }
        }
    },
}


class RunPodContractTests(unittest.TestCase):
    def test_configured_cuda_versions_are_required_and_unique(self) -> None:
        self.assertEqual(
            configured_cuda_versions(
                {
                    "allowed_cuda_versions": [
                        "13.0",
                        "12.9",
                        "12.8",
                        "12.4",
                    ]
                }
            ),
            ("13.0", "12.9", "12.8", "12.4"),
        )
        for value in ([], ["12.8", "12.8"], "12.8"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                configured_cuda_versions({"allowed_cuda_versions": value})

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
        self.assertEqual(
            extract_allowed_cuda_versions(SCHEMA),
            frozenset({"13.0", "12.9", "12.8", "12.4"}),
        )

    def test_validate_reports_every_invalid_configured_id(self) -> None:
        with self.assertRaisesRegex(
            ContractError, "Tesla T4.*Unknown GPU"
        ):
            validate_gpu_type_ids(SCHEMA, ["Tesla T4", "Unknown GPU"])

    def test_validate_reports_unsupported_cuda_version(self) -> None:
        with self.assertRaisesRegex(ContractError, "12.7"):
            validate_cuda_versions(SCHEMA, ["12.8", "12.7"])

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
        with self.assertRaisesRegex(ContractError, "must contain strings"):
            extract_gpu_type_ids(schema)


if __name__ == "__main__":
    unittest.main()
