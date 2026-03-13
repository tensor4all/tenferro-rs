# Math Notes Pages Design

## Goal

Publish the mathematical AD notes in `tensor-ad-oracles` as a deployable web
site, with correct formula rendering and stable per-note URLs, while keeping
`docs/math/*.md` as the source of truth.

## Current State

The repository now contains the math-note corpus under `docs/math/` and a
structured linkage file at `docs/math/registry.json`. GitHub Pages is already
enabled for the repository, but the published site still returns 404 because
there is no docs build/deploy workflow yet.

`tenferro-rs` already deploys its design and AD notes through Quarto and uses
KaTeX for math rendering. That is a useful reference because the note corpus in
this repository originates from the same documentation style and formula
notation.

## Requirements

1. Render the existing math notes on the web with correct equation layout.
2. Keep `docs/math/*.md` as the editable source, not generated HTML.
3. Publish stable URLs per note so note links are durable over time.
4. Expose the DB-to-note linkage through a human-readable registry page and the
   raw `registry.json`.
5. Keep the scope local to `tensor-ad-oracles`; do not change `tenferro-rs`.
6. Avoid introducing a large docs stack that is heavier than the repository
   needs.

## Approaches Considered

### 1. Quarto website with KaTeX and GitHub Pages

Add a small `docs/_quarto.yml`, render only the public docs pages we care
about, and deploy the generated site to GitHub Pages.

Pros:

- matches the `tenferro-rs` documentation toolchain
- KaTeX formula rendering is already proven for this note style
- gives sidebar navigation and clean per-page URLs with little custom code
- keeps the source material as Markdown

Cons:

- adds Quarto as a docs build dependency

### 2. Custom Pandoc-based static site

Write a bespoke build script that converts Markdown to HTML and injects MathJax
or KaTeX by hand.

Pros:

- smaller dependency surface than Quarto

Cons:

- more custom code to maintain
- navigation and page structure become our problem
- easier to regress as the note corpus grows

### 3. MkDocs-based docs site

Adopt a Python docs framework and theme.

Pros:

- familiar docs-site pattern with navigation and search

Cons:

- extra Python dependencies and configuration
- less aligned with `tenferro-rs`
- more infrastructure than this repository currently needs

## Recommendation

Use **Approach 1: Quarto website with KaTeX and GitHub Pages**.

This is the best balance between correctness, maintainability, and consistency
with `tenferro-rs`. It is not the absolute smallest tool choice, but it is the
smallest choice that already solves math rendering, navigation, and stable page
generation without a pile of local glue code.

## Site Structure

The deployable docs surface should be intentionally small.

Recommended source layout:

- `docs/_quarto.yml`
- `docs/index.md`
- `docs/math/index.md`
- `docs/math/*.md`
- `docs/math-registry.md`
- `docs/math/registry.json`

Recommended published shape:

- `/tensor-ad-oracles/`
- `/tensor-ad-oracles/math/index.html`
- `/tensor-ad-oracles/math/svd.html`
- `/tensor-ad-oracles/math/solve.html`
- `/tensor-ad-oracles/math/registry.json`

The site home page should stay thin: one short explanation plus links to the
math notes index and the registry reference page.

## Navigation Model

The sidebar should stay simple:

- Home
- Math Notes
  - index page
  - each operation note
- Reference
  - registry explanation page

`docs/math/registry.json` should remain the machine-readable source of truth.
The human-facing site should add a short `docs/math-registry.md` page that
explains what the registry means and links to the raw JSON artifact.

## Math Rendering

Use KaTeX through Quarto's `html-math-method: katex`, matching the current
`tenferro-rs` docs configuration.

This gives us:

- predictable rendering for `$...$` and `$$...$$`
- output that does not depend on client-side Markdown conversion
- consistency across the two repositories

No formula migration should be required. The existing Markdown notes should
render as-is, aside from any incidental cleanup discovered during smoke tests.

## Build And Deploy

Add a dedicated docs workflow that runs on pushes to `main` and on manual
dispatch.

Recommended flow:

1. checkout repository
2. install Quarto
3. configure GitHub Pages
4. run `scripts/build_docs_site.sh`
5. upload `target/docs-site`
6. deploy with `actions/deploy-pages`

The local and CI entry point should be the same shell script so the repository
has one build contract.

## Verification

Validation should happen at two levels.

### 1. Repository contract tests

Add targeted tests that require:

- `docs/_quarto.yml` to exist
- `docs/index.md` to exist
- `docs/math-registry.md` to exist
- the Quarto config to include KaTeX and the expected sidebar entries
- the docs workflow to exist

### 2. Build smoke verification

Add a small checker for the rendered site and run it from
`scripts/build_docs_site.sh`.

It should verify that the generated artifact includes at least:

- `index.html`
- `math/index.html`
- one representative note page such as `math/svd.html`
- `math/registry.json`

CI should run the docs workflow build, which is the real end-to-end proof that
the render still works.

## Non-Goals

- changing the oracle DB schema
- moving docs back into `tenferro-rs`
- adding full-text search or a large documentation theme system
- embedding note URLs into JSON case records

## Rollout

Implement the docs site in this repository only. Once merged, the canonical
published location for math notes becomes the GitHub Pages site for
`tensor-ad-oracles`, while `tenferro-rs` can continue to link outward as needed.
