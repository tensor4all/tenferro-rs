# Architecture

Design rationale for each subsystem in the tenferro stack. Describes *what*
each layer does and *why*. For normative specifications (trait signatures,
op semantics), see [Specification](../spec/).

| Document | Covers |
|----------|--------|
| [tenferro-crates.md](./tenferro-crates.md) | Current crate structure, dependency boundaries, extension boundary, AD boundary |
| [computegraph.md](./computegraph.md) | GraphOp, Operand, Fragment, resolve/materialize/compile/eval pipeline |
| [chainrules.md](./chainrules.md) | PrimitiveOp trait, AD rule structure |
| [tidu.md](./tidu.md) | differentiate, transpose, LinearFragment, higher-order AD |
| [ad-pipeline.md](./ad-pipeline.md) | End-to-end AD pipeline, scalar/vector examples, golden tests |
