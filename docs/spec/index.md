# Specification

Normative specifications — source of truth for trait signatures, op semantics,
and backend contracts. Each fact has exactly one owner document. Other documents
link here rather than re-stating.

| Document | Owns |
|----------|------|
| [primitive-catalog.md](./primitive-catalog.md) | Tenferro IR op vocabulary, per-op semantics, StableHLO lowering rules |
| [backend-contract.md](./backend-contract.md) | Backend pipeline, Execution IR dispatch, backend trait signatures |
| [ad-contract.md](./ad-contract.md) | PrimitiveOp trait signature, linearize/transpose_rule requirements |
| [optimizer-passes.md](./optimizer-passes.md) | Optimization pass algorithms and ordering |
| [tensor-semantics.md](./tensor-semantics.md) | Tensor type semantics, stride model, contiguity rules |
