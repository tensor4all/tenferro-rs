# Math Registry Reference

The oracle database and the math notes are linked through
`docs/math/registry.json`.

Each entry maps a published `(op, family)` pair to:

- a note file under `docs/math/`
- a stable anchor inside that note

This keeps the JSON oracle database machine-readable while letting the math
notes evolve as normal Markdown documents.

The intended contract is:

- the math notes are the mathematical source of truth
- the oracle DB is the machine-readable test artifact
- the registry is the stable bridge from DB families to note anchors

As long as a note keeps its published anchor, prose and derivation detail can be
expanded without changing the DB schema.

Published raw registry artifact:

- [math/registry.json](./math/registry.json)

For the note corpus entrypoint, see [Math Notes](./math/index.md).
