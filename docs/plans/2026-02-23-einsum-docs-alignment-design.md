# Einsum Docs/Contracts Alignment Design

**Issue**: #147 — Define guaranteed vs planned einsum behavior and align docs/contracts
**Date**: 2026-02-23
**Status**: Approved
**Scope**: Docs-only. No behavioral or code changes.

## Problem

Design docs (`docs/design/einsum.md`) and crate-level docs (`tenferro-einsum/src/lib.rs`)
mix implemented behavior with planned/aspirational behavior, creating ambiguity.

## Approach

Add inline **status annotations** to each unimplemented or partially-implemented feature
in both files. Format:

```markdown
> **Status: Not yet implemented.** [Brief explanation + issue ref]
```

## Annotations (7 total)

### In `docs/design/einsum.md`:

1. **Parenthesized contraction order** — Parsed but silently discarded;
   optimizer picks order. See #144.
2. **Contract fallback Path B** — Not implemented; error returned when
   Contract extension unavailable. See #141.
3. **`_owned` buffer reuse** — Not implemented; `_owned` delegates to
   borrowed API without buffer reuse.
4. **`tracked_einsum` tape recording** — Rule built but discarded; no tape
   access in current API. See #136.
5. **GPU async chaining** — Not implemented; CPU-only execution. See #141.

### In `tenferro-einsum/src/lib.rs` (crate-level docs):

6. **GPU async section** — Mark examples as aspirational, not yet functional.
7. **`_owned` / `tracked_einsum` function-level docs** — Add current-limitation
   notes to doc comments.

## Deliverables

- A short inline note at each mismatch site marking status.
- No new sections or restructuring of existing docs.
- Follow-up issues can reference these annotations as the contract baseline.
