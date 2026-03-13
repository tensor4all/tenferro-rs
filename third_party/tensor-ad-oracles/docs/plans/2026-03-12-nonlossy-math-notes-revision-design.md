# Non-Lossy Math Notes Revision Design

## Goal

Revise `docs/math/*.md` so that `tensor-ad-oracles` becomes the detailed source
of truth for AD math notes without losing information currently present in
`../tenferro-rs/docs/AD/*.md`.

The revised notes should remain aligned with the actual derivative formulas used
by PyTorch, but the migration must be non-lossy with respect to the existing
`tenferro-rs` note corpus.

## Problem

The current `docs/math/*.md` corpus successfully established this repository as
the documentation source of truth, but several notes were compressed too
aggressively during migration.

This is especially visible in:

- `docs/math/svd.md`
- `docs/math/qr.md`

Compared with:

- `../tenferro-rs/docs/AD/svd.md`
- `../tenferro-rs/docs/AD/qr.md`

the current notes often omit:

- helper definitions
- intermediate matrices
- case splits
- implementation correspondence details
- explicit formula structure needed to audit the AD rule against code

That makes the current notes readable, but not yet strong enough as the
canonical mathematical reference.

## Non-Lossy Migration Constraint

This revision is governed by one hard rule:

> Moving the math notes from `tenferro-rs` into `tensor-ad-oracles` must not
> lose mathematical or implementation-relevant information.

In practice, this means:

- formulas present in `../tenferro-rs/docs/AD/*.md` must remain represented
- helper operators and projections must remain represented
- case splits and domain restrictions must remain represented
- implementation-facing explanations must remain represented

The text may be reorganized, expanded, or clarified, but not simplified by
deleting meaningful content.

## Source Hierarchy

The revision should use three sources with different roles.

### 1. Primary migration source: `tenferro-rs`

`../tenferro-rs/docs/AD/*.md` is the non-lossy baseline. If information is
present there, the revised note in this repository should preserve it.

### 2. Implementation alignment source: PyTorch

PyTorch manual autograd formulas are the implementation-alignment reference:

- `../pytorch/tools/autograd/derivatives.yaml`
- `../pytorch/torch/csrc/autograd/FunctionsManual.cpp`

These sources are used to:

- validate that the note matches the actual backward / JVP structure
- restore implementation-facing detail when the current note drifted too far
- document PyTorch-specific case handling where useful

### 3. Published source of truth: `tensor-ad-oracles`

The final canonical notes live in:

- `docs/math/*.md`

This repository remains the publication target and DB-link target through
`docs/math/registry.json`.

## Approaches Considered

### 1. Non-lossy revision with selective expansion

Expand each note until it preserves `tenferro-rs` content, using PyTorch
implementation references where needed, but do not force all notes to the same
length.

Pros:

- preserves important content without bloating simple notes
- focuses detail where the formulas are subtle
- keeps maintenance cost reasonable
- matches the user's requirement directly

Cons:

- requires careful per-note judgment

### 2. Uniformly expand every note to the same template depth

Use a rigid format and push every note to the same granularity.

Pros:

- highly regular structure

Cons:

- verbose for simple rules
- wastes effort on notes that do not need full derivation density
- makes future updates heavier

### 3. Mostly copy `tenferro-rs` verbatim and append DB sections

Treat the old notes as near-verbatim source and only add local anchors and DB
mapping sections.

Pros:

- minimizes migration risk

Cons:

- may carry over repository-local assumptions from `tenferro-rs`
- misses the chance to align the notes more directly with PyTorch formulas
- does not cleanly standardize the `tensor-ad-oracles` corpus

## Recommendation

Use **Approach 1: non-lossy revision with selective expansion**.

This keeps the hard non-lossy constraint while still letting the published notes
be reorganized into a cleaner, repository-native format. The key is to treat
`tenferro-rs` as a content floor, not as a formatting ceiling.

## Note Structure

Each note should use the following high-level structure where applicable:

1. forward definition
2. notation and shapes
3. assumptions, gauge conditions, and singularities
4. helper definitions
5. forward-mode rule
6. reverse-mode rule
7. case splits
8. implementation correspondence
9. verification
10. DB family mapping

Not every note needs every section at the same depth, but the structure should
be rich enough to preserve all mathematical and implementation-relevant detail.

## Detail Tiers

The corpus should be revised in three detail tiers.

### Tier A: Implementation-auditable notes

These notes should be detailed enough that an engineer can map the prose and
equations back to the implementation without guessing.

Target notes:

- `docs/math/svd.md`
- `docs/math/qr.md`
- `docs/math/lu.md`
- `docs/math/eig.md`
- `docs/math/eigen.md`
- `docs/math/solve.md`
- `docs/math/lstsq.md`
- `docs/math/pinv.md`
- `docs/math/norm.md`

These notes should preserve:

- helper matrices
- projection operators
- full case splits
- implementation correspondence to PyTorch and/or `tenferro-rs`

### Tier B: Moderately expanded derivation notes

These notes should retain the full conceptual derivation and important caveats,
without necessarily matching every implementation helper line-for-line.

Target notes:

- `docs/math/cholesky.md`
- `docs/math/inv.md`
- `docs/math/det.md`
- `docs/math/matrix_exp.md`
- `docs/math/dyadtensor_reverse.md`

### Tier C: Shared-pattern notes

These notes should preserve all relevant information, but organize it by shared
pattern rather than by one formula per operator.

Target notes:

- `docs/math/scalar_ops.md`

For these, the revision should make shared derivative patterns more explicit,
including broadcasting, reductions, nondifferentiable branches, and family-level
observable expectations.

## Implementation Correspondence

Detailed notes should explicitly state which implementation source they match.

For example:

- PyTorch helper or entry point from `FunctionsManual.cpp`
- generated binding from `derivatives.yaml`
- prior derivation source from `../tenferro-rs/docs/AD/*.md`

This correspondence is not just a bibliography. It is part of the note's job:
the reader should be able to see how the documented rule maps to code.

## Validation Strategy

Validation must cover both repository integrity and content integrity.

### 1. Existing repository integrity

Keep the current guarantees:

- note files exist
- anchors remain stable
- `docs/math/registry.json` stays valid

### 2. New note completeness checks

Add targeted tests that assert representative notes contain the expected
non-lossy content.

Examples:

- `svd.md` contains the `F` matrix, inverse singular-value helper, non-square
  correction terms, and gauge discussion
- `qr.md` contains `copyltu`, full-rank and wide-case structure, and triangular
  solve discussion

The tests should not attempt a byte-for-byte mirror of `tenferro-rs`, but they
should protect against regression back into over-compression.

### 3. Manual comparison pass

The implementation should include a deliberate comparison pass against:

- `../tenferro-rs/docs/AD/*.md`
- PyTorch manual formulas

This comparison is necessary because "non-lossy" is a semantic requirement, not
just a formatting one.

## Non-Goals

- changing the JSON oracle schema
- changing DB family anchors
- rewriting the notes into a theorem-proof textbook style
- making every note equally long regardless of mathematical complexity
- modifying `../tenferro-rs`

## Rollout

The revision should happen entirely inside `tensor-ad-oracles`.

After completion:

- `docs/math/*.md` becomes the detailed canonical note corpus
- `tenferro-rs` may remain as historical context, but no longer needs to carry
  the richer source of truth
- the published Pages site will expose the fuller notes without changing the DB
  linkage model
