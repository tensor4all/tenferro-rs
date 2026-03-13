# Math Notes Pages Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish `docs/math/*.md` as a GitHub Pages site with KaTeX-rendered formulas, stable note URLs, and a small human-facing registry reference, without changing the oracle DB contract.

**Architecture:** Use Quarto as a thin website layer over the existing docs corpus. Keep `docs/math/*.md` and `docs/math/registry.json` as the source material, add a small `docs/index.md` and `docs/math-registry.md`, render to `target/docs-site` through a shared `scripts/build_docs_site.sh`, and deploy that artifact with a dedicated GitHub Pages workflow.

**Tech Stack:** Markdown, Quarto, KaTeX, GitHub Pages Actions, Python 3.12, `uv`, `unittest`, shell scripts.

---

### Task 1: Add the docs-site repository contract

**Files:**
- Create: `tests/test_docs_site.py`
- Modify: `tests/test_repo_config.py`
- Modify: `README.md`

**Step 1: Write the failing test**

Create `tests/test_docs_site.py` with checks that require the docs-site inputs to
exist once the feature lands. Add assertions for:

- `docs/_quarto.yml`
- `docs/index.md`
- `docs/math-registry.md`
- `.github/workflows/docs.yml`
- `scripts/build_docs_site.sh`

Extend `tests/test_repo_config.py` so the README must mention:

- GitHub Pages docs deployment
- `docs/math/`
- `docs/math/registry.json`

Example assertions:

```python
def test_docs_site_contract_files_exist(self) -> None:
    self.assertTrue((REPO_ROOT / "docs" / "_quarto.yml").exists())
    self.assertTrue((REPO_ROOT / "docs" / "index.md").exists())
    self.assertTrue((REPO_ROOT / "docs" / "math-registry.md").exists())
    self.assertTrue((REPO_ROOT / ".github" / "workflows" / "docs.yml").exists())
    self.assertTrue((REPO_ROOT / "scripts" / "build_docs_site.sh").exists())
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_docs_site tests.test_repo_config -v`

Expected: FAIL because none of the docs-site files exist yet and the README does
not document the deployment contract.

**Step 3: Write minimal implementation**

Update `README.md` with a short section that states:

- math notes are published via GitHub Pages
- the source corpus is `docs/math/`
- the registry source is `docs/math/registry.json`

Leave the actual docs-site files for the next tasks.

**Step 4: Run test to verify it still fails for the missing files**

Run: `uv run python -m unittest tests.test_docs_site tests.test_repo_config -v`

Expected: FAIL, but now only because the docs-site files are not present.

**Step 5: Commit**

```bash
git add README.md tests/test_docs_site.py tests/test_repo_config.py
git commit -m "test: define docs site contract"
```

### Task 2: Add the Quarto site scaffold

**Files:**
- Create: `docs/_quarto.yml`
- Create: `docs/index.md`
- Create: `docs/math-registry.md`
- Modify: `docs/math/index.md`
- Modify: `tests/test_docs_site.py`

**Step 1: Write the failing test**

Extend `tests/test_docs_site.py` so it requires the Quarto config to include:

- website project mode
- output directory `../target/docs-site`
- `html-math-method: katex`
- sidebar entries for `index.md`, `math/index.md`, and `math-registry.md`
- `math/registry.json` listed as a site resource

Example assertions:

```python
config = (REPO_ROOT / "docs" / "_quarto.yml").read_text(encoding="utf-8")
self.assertIn("type: website", config)
self.assertIn("output-dir: ../target/docs-site", config)
self.assertIn("html-math-method: katex", config)
self.assertIn("- math-registry.md", config)
self.assertIn("- math/registry.json", config)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_docs_site -v`

Expected: FAIL because the Quarto config and public landing pages do not exist.

**Step 3: Write minimal implementation**

Create:

- `docs/_quarto.yml` with:
  - `project.type: website`
  - `project.output-dir: ../target/docs-site`
  - `project.render` limited to `index.md`, `math/**/*.md`, and `math-registry.md`
  - `project.resources` including `math/registry.json`
  - `format.html.html-math-method: katex`
  - a small sidebar with Home, Math Notes, and Reference
- `docs/index.md` as the site home page
- `docs/math-registry.md` as the human explanation page for the registry

Update `docs/math/index.md` to link to the registry reference page.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_docs_site -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/_quarto.yml docs/index.md docs/math-registry.md docs/math/index.md tests/test_docs_site.py
git commit -m "docs: add quarto scaffold for math notes site"
```

### Task 3: Add the shared docs build entry point and artifact checker

**Files:**
- Create: `scripts/build_docs_site.sh`
- Create: `scripts/check_docs_site.py`
- Modify: `scripts/__init__.py`
- Modify: `tests/test_docs_site.py`
- Modify: `tests/test_scripts.py`

**Step 1: Write the failing test**

Add tests that require:

- `scripts.check_docs_site.main()` to accept a built site root and reject missing
  expected files
- `scripts/build_docs_site.sh` to invoke Quarto on `docs/` and to write into
  `target/docs-site`

Example Python-side check:

```python
with self.assertRaisesRegex(SystemExit, "math/svd.html"):
    check_docs_site.main(["--site-root", str(site_root)])
```

Example build-script text check:

```python
script = (REPO_ROOT / "scripts" / "build_docs_site.sh").read_text(encoding="utf-8")
self.assertIn("quarto render", script)
self.assertIn("target/docs-site", script)
self.assertIn("check_docs_site.py", script)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_docs_site tests.test_scripts -v`

Expected: FAIL because the build script and checker do not exist.

**Step 3: Write minimal implementation**

Create `scripts/check_docs_site.py` that verifies the built artifact contains:

- `index.html`
- `math/index.html`
- `math/svd.html`
- `math/registry.json`

Create `scripts/build_docs_site.sh` that:

1. removes any old `target/docs-site`
2. runs `quarto render docs`
3. runs the checker against `target/docs-site`

Export the checker from `scripts/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_docs_site tests.test_scripts -v`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/build_docs_site.sh scripts/check_docs_site.py scripts/__init__.py tests/test_docs_site.py tests/test_scripts.py
git commit -m "feat: add docs site build and verification"
```

### Task 4: Add the GitHub Pages workflow

**Files:**
- Create: `.github/workflows/docs.yml`
- Modify: `tests/test_repo_config.py`

**Step 1: Write the failing test**

Extend `tests/test_repo_config.py` so the docs workflow must include:

- trigger on `push` to `main`
- `workflow_dispatch`
- permissions for `pages: write` and `id-token: write`
- `actions/configure-pages`
- `actions/upload-pages-artifact`
- `actions/deploy-pages`
- `./scripts/build_docs_site.sh`

Example assertion:

```python
workflow = (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
self.assertIn("pages: write", workflow)
self.assertIn("actions/deploy-pages@v4", workflow)
self.assertIn("./scripts/build_docs_site.sh", workflow)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: FAIL because the workflow file does not exist.

**Step 3: Write minimal implementation**

Create `.github/workflows/docs.yml` following the `tenferro-rs` pattern but
trimmed for this repository:

- checkout
- install Quarto
- configure pages
- run `./scripts/build_docs_site.sh`
- upload `target/docs-site`
- deploy with `actions/deploy-pages@v4`

Do not add rustdoc or graphviz steps.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: PASS

**Step 5: Commit**

```bash
git add .github/workflows/docs.yml tests/test_repo_config.py
git commit -m "ci: deploy math notes site to github pages"
```

### Task 5: Run end-to-end verification and polish links

**Files:**
- Verify: `docs/index.md`
- Verify: `docs/math/index.md`
- Verify: `docs/math-registry.md`
- Verify: `docs/_quarto.yml`
- Verify: `scripts/build_docs_site.sh`

**Step 1: Run the targeted unit tests**

Run:

```bash
uv run python -m unittest tests.test_docs_site tests.test_scripts tests.test_repo_config -v
```

Expected: PASS

**Step 2: Run a local docs build smoke test**

Run:

```bash
./scripts/build_docs_site.sh
```

Expected:

- Quarto render succeeds
- `target/docs-site/index.html` exists
- `target/docs-site/math/svd.html` exists
- `target/docs-site/math/registry.json` exists

**Step 3: Review the generated output**

Open the built pages locally and verify:

- KaTeX renders formulas in `svd.html`
- sidebar links work
- `math-registry.md` links to the raw registry JSON

**Step 4: Commit**

```bash
git add README.md docs .github/workflows/docs.yml scripts tests
git commit -m "docs: publish math notes site"
```
