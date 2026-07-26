# Architecture

Design rationale for each subsystem in the tenferro stack. Describes *what*
each layer does and *why*. For normative specifications (trait signatures,
op semantics), see [Specification](../spec/).

| Document | Covers |
|----------|--------|
| [tenferro-crates.md](./tenferro-crates.md) | Current crate structure, dependency boundaries, extension boundary, AD boundary |
| [computegraph.md](./computegraph.md) | GraphOperation, Graph, resolve/materialize/compile/eval pipeline |
| [primitive-ad.md](./primitive-ad.md) | Primitive AD contract and rule structure |
| [semantic-ad.md](./semantic-ad.md) | Current semantic AD ownership, rule dispatch, cache ownership |
| [ad-pipeline.md](./ad-pipeline.md) | End-to-end AD pipeline, scalar/vector examples, golden tests |
