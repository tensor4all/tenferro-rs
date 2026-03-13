import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DocsSiteContractTests(unittest.TestCase):
    def test_docs_site_contract_files_exist(self) -> None:
        self.assertTrue((REPO_ROOT / "docs" / "_quarto.yml").exists())
        self.assertTrue((REPO_ROOT / "docs" / "index.md").exists())
        self.assertTrue((REPO_ROOT / "docs" / "math-registry.md").exists())

    def test_quarto_config_declares_math_notes_site(self) -> None:
        config = (REPO_ROOT / "docs" / "_quarto.yml").read_text(encoding="utf-8")

        self.assertIn("type: website", config)
        self.assertIn("output-dir: ../target/docs-site", config)
        self.assertIn("html-math-method: katex", config)
        self.assertIn("- math-registry.md", config)
        self.assertIn("- math/registry.json", config)

    def test_math_index_links_to_registry_reference(self) -> None:
        index_text = (REPO_ROOT / "docs" / "math" / "index.md").read_text(encoding="utf-8")

        self.assertIn("../math-registry.md", index_text)

    def test_build_script_renders_quarto_site(self) -> None:
        script = (REPO_ROOT / "scripts" / "build_docs_site.sh").read_text(encoding="utf-8")

        self.assertIn("quarto render", script)
        self.assertIn("target/docs-site", script)
        self.assertIn("check_docs_site.py", script)
        self.assertIn("python3", script)
        self.assertNotIn("uv run python", script)


if __name__ == "__main__":
    unittest.main()
