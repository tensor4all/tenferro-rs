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

- keep the existing `TypedTensor<T>` plus runtime `Tensor` split
- simplify the traced graph around one core op vocabulary
- move shape snapshots off most op variants and onto value-side metadata
- treat tropical primarily as a composition and extension story, not as a
  second graph substrate
- defer any generic extension mechanism until op identity and hashing are fully
  specified
