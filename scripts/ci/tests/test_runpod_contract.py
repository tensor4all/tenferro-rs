import unittest
import json
from pathlib import Path

from scripts.ci.runpod_contract import (
    ContractError,
    configured_cuda_versions,
    configured_gpu_tiers,
    extract_gpu_type_ids,
    resolve_local_ref,
    schema_properties,
    validate_cuda_versions,
    validate_gpu_type_ids,
)


FIXTURES = Path(__file__).with_name("fixtures")
SCHEMA = json.loads((FIXTURES / "runpod_v2_openapi.json").read_text())
CATALOG = {"gpus": [{"id": "NVIDIA A40"}, {"id": "NVIDIA GeForce RTX 4090"}]}


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

    def test_extract_gpu_ids_uses_catalog_not_schema_enum(self):
        self.assertEqual(
            extract_gpu_type_ids(CATALOG),
            frozenset({"NVIDIA A40", "NVIDIA GeForce RTX 4090"}),
        )
        validate_gpu_type_ids(SCHEMA, ["NVIDIA A40"], CATALOG)

    def test_invalid_catalog_ids_are_reported(self):
        with self.assertRaisesRegex(ContractError, "Tesla T4.*Unknown GPU"):
            validate_gpu_type_ids(SCHEMA, ["Tesla T4", "Unknown GPU"], CATALOG)

    def test_cuda_validation_checks_v2_pattern_not_old_enum(self):
        validate_cuda_versions(SCHEMA, ["12.7", "13.0"])
        with self.assertRaisesRegex(ContractError, "13"):
            validate_cuda_versions(SCHEMA, ["13"])

    def test_missing_v2_post_schema_is_a_hard_error(self):
        with self.assertRaisesRegex(ContractError, "POST /v2/pods"):
            validate_gpu_type_ids({"paths": {}}, ["NVIDIA A40"], CATALOG)

    def test_non_local_reference_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "non-local"):
            resolve_local_ref(SCHEMA, "https://example.test/CreatePod")

    def test_malformed_catalog_is_rejected(self):
        for catalog in ({}, {"gpus": []}, {"gpus": [{"id": 7}]}):
            with self.assertRaises(ContractError):
                extract_gpu_type_ids(catalog)

    def test_generated_payload_matches_published_v2_schema(self):
        from scripts.ci.runpod_client import build_pod_payload

        config = json.loads(
            (Path(__file__).resolve().parents[1] / "runpod_config.json").read_text()
        )
        payload = build_pod_payload(config, "image", "startup", "jit", ["NVIDIA A40"])
        properties = schema_properties(SCHEMA, {"$ref": "#/components/schemas/CreatePodRequest"})
        self.assertLessEqual(set(payload), set(properties))
        self.assertNotIn("gpuTypeIds", properties)
        self.assertEqual(payload["image"], "image")
        self.assertEqual(payload["entrypoint"], ["bash", "-lc"])
        self.assertEqual(payload["cmd"], ["startup"])
        for field in ("gpu", "mounts"):
            allowed = schema_properties(SCHEMA, properties[field])
            self.assertLessEqual(set(payload[field]), set(allowed))
        self.assertEqual(payload["gpu"]["id"], "NVIDIA A40")
        self.assertEqual(payload["gpu"]["count"], config["gpu_count"])
        self.assertEqual(payload["mounts"]["persistent"], {"size": config["volume_gb"], "path": config["volume_mount_path"]})


if __name__ == "__main__":
    unittest.main()
