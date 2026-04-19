# Design V3 Proposal Set

This directory collects the `v3` design proposal set for the tenferro traced
graph and AD stack.

Status: proposal only. These documents are not the current canonical design.
They exist to review a coherent direction before any implementation work
begins.

This proposal set is intentionally isolated from the existing design docs:

- the current `docs/design/*.md` files continue to describe the present system
  or earlier accepted subsystem designs
- `docs/design/design_v3/` captures a possible next architecture spanning
  traced tensors, AD, shape metadata, algebra boundaries, and tropical support

Supersedes (closed in favour of this proposal set):

- `#738` tropical extension revival
- `#740` AD pipeline redesign proposal
- `#741` move Category C shape fields to value-side metadata

## Reading Order

1. [00-overview.md](./00-overview.md)
2. [10-ad-model.md](./10-ad-model.md)
3. [20-shape-metadata.md](./20-shape-metadata.md)
4. [30-algebra-and-tropical.md](./30-algebra-and-tropical.md)
5. [40-extension-boundary.md](./40-extension-boundary.md)
6. [90-migration-plan.md](./90-migration-plan.md)

## Scope

The proposal set answers these questions (with primary chapter reference):

- What is the core graph op vocabulary for traced execution and AD?
  → [`10-ad-model.md`](./10-ad-model.md)
- Which information belongs on the op, and which belongs on the value?
  → [`20-shape-metadata.md`](./20-shape-metadata.md)
- What role, if any, should `SemiringOp`, `SemiringBackend`, and
  `tenferro-algebra` continue to play?
  → [`30-algebra-and-tropical.md`](./30-algebra-and-tropical.md)
- How should tropical support fit into the system?
  → [`30-algebra-and-tropical.md`](./30-algebra-and-tropical.md)
- What extension boundary is safe for out-of-tree primitives?
  → [`40-extension-boundary.md`](./40-extension-boundary.md)
- What migration order keeps the repository stable?
  → [`90-migration-plan.md`](./90-migration-plan.md)

## Non-Goals

These documents do not:

- approve implementation work
- replace the current public API by themselves
- require immediate deletion of existing design documents
- define a final serialized graph compatibility policy

## Current Recommendation

The recommended interpretation of `v3` is:

- keep the full v2 tensor API surface (`EagerTensor<B>`, `TracedTensor`,
  `Tensor`, `TypedTensor<T>`) unchanged
- simplify the traced graph around one core op vocabulary
- move shape snapshots off most op variants and onto value-side metadata
- ship tropical as an external crate: composition first (Stage 4), fused
  `ExtensionOp` second (Stage 7)
- commit to a staged generic extension mechanism — contract (Stage 5),
  implementation (Stage 6), tropical self-test (Stage 7) — rather than
  leaving it as an open-ended deferred question

## Roadmap

```text
         USER-FACING API  (unchanged)
         EagerTensor<B>  TracedTensor  Tensor  TypedTensor<T>
                             │
═════════════════════════════╪═══════════════════════════════
                             │
  Stage 0  ┌──────────────────▼──────────────────┐
  approve  │ close #738/#740/#741                │
           │ design_v3 = source of truth         │
           └──────────────────┬──────────────────┘
                              │
  Stage 1  ┌──────────────────▼──────────────────┐
  shape    │ value-side shape/dtype metadata     │
  cleanup  │ Cat C fields off ops                │
           │ ★ prereq: serialized-graph policy   │
           └──────────────────┬──────────────────┘
                              │
  Stage 2  ┌──────────────────▼──────────────────┐
  story    │ SemiringOp demoted (docs + tests)   │
           └──────────────────┬──────────────────┘
                              │
  Stage 3  ┌──────────────────▼──────────────────┐
  AD       │ AD rules read value metadata        │
  consol.  │ symbolic-shape AD correctness       │
           │ ★ exit gate: algebra A(delete) / B  │
           └──────────────────┬──────────────────┘
                              │
  Stage 4a ┌──────────────────▼──────────────────┐
  tropical │ external tenferro-ext-tropical      │
  concrete │ concrete-shape composition          │
           │ traced facade only                  │
           │ in-tree tropical tests migrated     │
           └──────────────────┬──────────────────┘
                              │
  Stage 4b ┌──────────────────▼──────────────────┐
  tropical │ same composition on symbolic shapes │
  symbolic │ contract test for Stage 3           │
           │ symbolic-AD correctness self-test   │
           └──────────────────┬──────────────────┘
                              │
  Stage 5  ┌──────────────────▼──────────────────┐
  ext.     │ spec doc: ExtensionOp contract      │
  spec     │   identity · hash · Eq · Clone      │
           │   AD closure · serialization ver.   │
           └──────────────────┬──────────────────┘
                              │
  Stage 6  ┌──────────────────▼──────────────────┐
  ext.     │ TensorOp::Extension(Arc<dyn Ext>)   │
  impl     │ engine integration + registry API   │
           └──────────────────┬──────────────────┘
                              │
  Stage 7  ┌──────────────────▼──────────────────┐
  tropical │ FusedTropicalDotGeneral             │
  Phase 2  │ = first canonical ExtensionOp       │
           │ = contract self-test                │
           │ AD via argmax + Gather/Scatter      │
           └──────────────────┬──────────────────┘
                              │
  Stage 8  ┌──────────────────▼──────────────────┐
  (opt)    │ core-owned fused ops                │
           │ only if composition too slow        │
           └─────────────────────────────────────┘
```

### Invariants Held Across All Stages

- `EagerTensor<B>`, `TracedTensor`, `Tensor`, and `TypedTensor<T>` are not
  renamed, merged, or removed
- for Standard-scalar users of the `tenferro` facade, the public API stays
  source-compatible
- oracle-replay baselines stay green
- cotangent space is always Standard — never algebra-generic
- AD rules emit only core op vocabulary (via `ExtensionOp` decomposition
  from Stage 6 onwards)
