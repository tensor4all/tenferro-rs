# Docs Site

This directory contains source assets for the unified documentation top page.

- `index.html`: entry page linking to:
  - formal architecture/design docs (`docs/design`)
  - Rust API reference (`cargo doc`)

Build the combined site with:

```bash
./scripts/build_docs_site.sh
```

Default output location:

```text
target/docs-site/
```

For local preview:

```bash
./scripts/serve_docs_site.sh
```
