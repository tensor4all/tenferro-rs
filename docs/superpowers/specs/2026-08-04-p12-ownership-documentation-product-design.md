# P12 Ownership-Model Documentation Product Design

Date: 2026-08-04

Status: design complete; content is prepared before P13-A and audited after freeze

Authority: #1555, #1569, `docs/design/storage-ownership-contracts.md`
(G4/G6), and `scripts/storage-ownership-contracts.toml`

## Scope and freeze sequencing

P12 delivers documentation from which a user can select and run the final
owner/view/group/runtime/provider model without reading implementation source.
The final product pages, examples, rustdoc, navigation, and checker scripts are
part of the P13-A candidate. After freeze, P12 only renders, executes, and
source-blind-audits that candidate and adds evidence reports.

This sequencing avoids changing the candidate while preserving the dependency
that P12 acceptance targets an already frozen API. A documentation correction
after freeze is a product change: it creates a new P13-A candidate and
invalidates affected P11/P12/P13 evidence.

## Audience and learning outcome

A reader familiar with Rust arrays or accelerator tensors must be able to:

1. choose `TypedTensor`, `Tensor`, immutable view, or mutable view;
2. understand one move-only owner and borrowed aliasing;
3. choose checked random access, contiguous slice iteration, or prepared
   strided traversal from their costs;
4. split mutable work only through the checked N-way API;
5. distinguish descriptor reinterpretation from numeric conversion;
6. distinguish duplicate/upload/download from mapping/synchronization;
7. use CUDA, WebGPU, and Apple/Metal without assuming a fixed device or hidden
   fallback;
8. choose detached consuming versus synchronous scoped read-only execution and
   interpret every outcome;
9. recover a standalone owner only through structural extraction or an
   explicit duplicate.

## Documentation set

### New canonical pages

- `docs/storage-ownership.md` — concise product-level ownership and runtime
  contract, linked from the landing/core-concepts pages.
- `docs/guides/views-and-slicing.md` — complete user guide and the canonical
  element-access/performance explanation.
- `docs/tutorial-code/src/bin/storage_element_access.rs` — runnable known-value
  owner/view/view-mut tutorial.
- `docs/testing/storage-documentation-audit.md` — post-freeze source-blind audit
  evidence.

`docs/_quarto.yml` adds the views guide under Guides. `docs/index.md` and
`docs/getting-started/core-concepts.md` link the ownership overview before
advanced operation guides.

### Existing pages updated

- `README.md` and getting-started pages;
- `docs/spec/tensor-semantics.md` storage section;
- device/backend, eager, autodiff, execution-model, memory-order, FFT, linalg,
  and relevant tutorial pages;
- public rustdoc for owners, views, prepared host access, splitting,
  reinterpretation, group extraction, transfer methods, runtime inputs,
  outcomes, and unsafe CUDA interop.

No rendered user/spec/design page describes removed `Buffer<T>`,
`BackendBuffer<T>`, shallow tensor cloning, `TensorOwnedView`, legacy map APIs,
implicit canonicalization copies, fixed engine IDs, or compatibility paths.
The normative design history may name a removed type only in an explicitly
marked removal/history section; checkers distinguish that context from current
API instructions.

## Required content

### Ownership and views

The guide states:

- owners are move-only and represent one physical span owner;
- immutable views may alias and borrow the owner;
- mutable views require exclusive borrowing;
- `as_view()`/`as_view_mut()` are O(1), allocation/refcount/provider/layout-
  clone free, and preserve static rank;
- N-way mutable views require retained injectivity and conservative disjoint
  byte envelopes;
- structural extraction either moves the owner or returns the unchanged group;
- `duplicate()` is explicit and creates a fresh allocation identity.

Examples assert known values, alias visibility, unchanged allocation identity
for views/reinterpretation, and changed identity for explicit duplication.

### Element access and performance

A dedicated heading named **Element access and performance** compares:

| Path | Setup | Inner work | Intended use |
|---|---|---|---|
| checked `get`/`get_mut` | none | O(rank) bounds/offset per call | sparse/random access |
| contiguous prepared guard/slice | one preparation/map | typed slice iteration | compact hot loops |
| prepared strided iterator | one preparation/map and cursor init | typed access + stride/carry | noncompact traversal |
| device launch | one prepare/bind | provider kernel work | device-resident tensors |

The guide explicitly warns against repeated multidimensional `get` inside a
full-tensor hot loop when a prepared bulk path is available. It explains that
validation is complete before prepared access exists and that the strided inner
loop does not resolve storage/provider state or decode flat indices.

Host-visible and device-only storage are distinct. Device-only data requires an
explicit download for host bytes; no host accessor silently transfers or
materializes.

### Reinterpretation and conversion

The guide covers only the sealed C32↔F32 and C64↔F64 representation pairs,
rank policy, alignment/divisibility errors, immutable aliases, and exclusive
mutable constraints. It contrasts this with numeric cast, which computes a new
allocation. It never suggests raw-parts vector retagging.

### Runtime ownership

The ownership overview and execution guide document:

- `ExecutionInputs` consuming detached submission;
- exact pre-admission rejection recovery;
- completed and retired-failed ownership only after retirement;
- completion-unproven diagnostics with no owner;
- handle drop as observation detach, not cancel;
- CPU-only synchronous scoped read-only execution and accelerator rejection;
- alias-safe bundles and consuming output extraction.

No cancellation behavior is documented because no cancellation state machine
exists.

### Providers and Apple shared access

The device guide uses canonical P10 provider namespaces and caller-selected
identities. It explains allocation domain versus access endpoint. Apple upload
creates one shared allocation; CPU/Metal transitions synchronize/map the same
allocation and transfer zero bytes; download is an explicit transfer to a new
host owner.

CUDA raw interop is explicitly unsafe and binding scoped. WebGPU/Metal has no
safe raw-handle escape or invented pointer parity.

### Public rustdoc cost contract

Every affected public method states, as applicable:

- whether it allocates or copies;
- whether it resolves/dispatches a provider;
- whether it synchronizes/maps;
- per-element bounds/stride cost;
- whether it preserves static rank;
- whether it can transfer or materialize;
- concrete `# Errors` for every `Result` API.

Runnable examples assert meaningful values or identities rather than only
shape/nonempty conditions.

## Executable tutorial

`storage_element_access.rs` runs without accelerator hardware and checks:

1. construct a fixed-rank column-major owner with known values;
2. borrow an immutable view and verify zero-copy metadata/values;
3. mutate a checked disjoint pair/N-way split and verify final values;
4. traverse compact data through a contiguous typed iterator;
5. traverse a transpose/reverse descriptor through prepared strided iteration;
6. create and read a complex/real representation view while preserving
   allocation identity;
7. explicitly duplicate and verify equal values plus a new identity;
8. exercise one unchanged structural-extraction failure.

The exact ledger command remains:

```text
cargo test -p tenferro-tutorial-code --release \
  tutorial_binaries_run_successfully -- --exact
```

## Documentation checkers

### `check-storage-docs.py`

```text
python3 scripts/check-storage-docs.py --include-rendered
```

The checker builds the Quarto/rustdoc documentation into a temporary directory
when `--include-rendered` is supplied, then validates:

- required files, navigation entries, and reciprocal links;
- canonical API names and provider namespaces;
- absence of stale current-tense legacy/implicit-copy language;
- required runtime outcome and Apple transfer distinctions;
- meaningful runnable examples and links to generated rustdoc;
- candidate identity shared by the freeze and audit reports.

The stale-language inventory is documentation-specific. It is not treated as a
Rust ownership proof.

### `check-storage-element-access-docs.py`

```text
python3 scripts/check-storage-element-access-docs.py \
  docs/guides/views-and-slicing.md
```

The checker requires the exact cost-table concepts, owner/view/view-mut,
static/dynamic rank, O(rank) random access, allocation-free view construction,
contiguous slice/iterator, prepared strided cursor, one provider resolution per
launch/traversal, host-visible versus device-only behavior, explicit download,
no hidden materialization, and links to P10 benchmark/codegen reports. It also
checks that named rustdoc entries carry the required cost/error sections.

These checkers use structured headings, links, code-block metadata, and a
bounded stale-term inventory. They do not attempt natural-language theorem
proving.

## Source-blind audit

An independent reviewer receives only rendered Quarto, generated rustdoc, and
the package needed to compile a downstream example. The reviewer receives no
repository source links or design/issues.

The audit asks the reviewer to:

1. explain owner/view/view-mut and aliasing rules;
2. choose the lowest-overhead path for random, contiguous, and strided access;
3. write and run a CPU example covering owner, immutable view, disjoint mutable
   views, duplicate, and reinterpretation;
4. identify explicit CUDA/WebGPU upload/download flow;
5. explain why Apple CPU/Metal switching is not a transfer;
6. distinguish detached and scoped outcomes, including ownerless
   completion-unproven;
7. list remaining ambiguities.

The generated downstream example is compile-checked and run against the frozen
candidate. An explain-only result is insufficient if the reviewer cannot build
the example.

`docs/testing/storage-documentation-audit.md` contains exactly one fenced JSON
record with schema `tenferro.storage-documentation-audit.v1`, candidate commit,
rendered artifact paths, reviewer boundary, tasks, generated example path,
compile/run commands, outcomes, and findings. Any Critical or Important
usability finding blocks P12. Fixing docs creates a new candidate.

## Validation set

P12 runs:

```text
python3 scripts/check-storage-docs.py --include-rendered
python3 scripts/check-storage-element-access-docs.py docs/guides/views-and-slicing.md
cargo test -p tenferro-tutorial-code --release tutorial_binaries_run_successfully -- --exact
python3 scripts/ci/run_profile.py docs
cargo test --doc --workspace --profile ci
python3 scripts/check-public-error-docs.py
python3 scripts/check-operation-categories.py --fail-on-findings --include-rendered
python3 scripts/check-docs-site.py
```

All commands target the same P13-A candidate. P12 evidence-only descendants may
add the audit report and promote ledger states, but product documentation and
checker scripts cannot change.

## Proportional-safety boundary and exit

Documentation cites exact Git commits and tracked paths. It does not document
digest receipts, nonce/attestation, malicious-runner behavior, quarantine,
poison recovery, cancellation, or removed compatibility APIs.

P12 is complete when all three ledger obligations pass, the rendered site is
navigable and free of stale current API claims, examples run with known-value
assertions, and the source-blind build audit has no Critical/Important finding
on the same P13-A candidate as P11.
