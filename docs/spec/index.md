# Specification

Normative specifications — source of truth for trait signatures, op semantics,
and backend contracts. Each fact has exactly one owner document. Other documents
link here rather than re-stating.

| Document | Owns |
|----------|------|
| [primitive-catalog.md](./primitive-catalog.md) | Tenferro IR op vocabulary, per-op semantics, and core primitive catalog boundary |
| [backend-contract.md](./backend-contract.md) | Backend pipeline, Execution IR dispatch, backend trait signatures |
| [ad-contract.md](./ad-contract.md) | Primitive trait signature, linearize/transpose_rule requirements |
| [optimizer-passes.md](./optimizer-passes.md) | Optimization pass algorithms and ordering |
| [tensor-semantics.md](./tensor-semantics.md) | Tensor type semantics, stride model, contiguity rules |
| [extension-op.md](./extension-op.md) | ExtensionOp trait contract (identity, AD, dispatch, registry) |
| [api-conventions.md](./api-conventions.md) | Public API naming, module shape, feature naming, documentation-surface checks, and checker mapping |
| [operation-categories.md](./operation-categories.md) | User-facing operation categories and the Eager/Traced surface-parity contract |
